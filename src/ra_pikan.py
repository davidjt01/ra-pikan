"""
RA-PIKAN: Residual-Adaptive Physics-Informed Kolmogorov–Arnold Networks.

Implements the four-phase training algorithm:
  Phase 1 — Base training (Adam only; n_lbfgs=0 in default RAPiKANConfig)
  Phase 2 — Residual diagnosis on a dense auxiliary set
  Phase 3 — Spatially targeted grid densification + RAD collocation augmentation
  Phase 4 — Continued training with reduced LR; repeat phases 2-4 for N cycles

Key function:
    ra_pikan_train(model, benchmark, config) -> result_dict
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .metrics import gini_coefficient, relative_l2_error
from .models import BSplinePIKAN
from .trainers import TrainConfig, rad_resample, train_model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RAPiKANConfig:
    """Full configuration for an RA-PIKAN run."""
    # Phase 1 and refinement phases tuned for practical CPU runtime.
    # Each BSpline Adam step takes ~400ms on CPU with 2000 collocation pts.
    # These settings target ~10-15 min per run; increase for final experiments.
    phase1: TrainConfig = field(default_factory=lambda: TrainConfig(
        n_adam=500, n_lbfgs=0, lr_adam=1e-3, lr_lbfgs=1.0,
        lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=100,
    ))
    # Refinement phases (phases 2-4 repeated)
    phase_refine: TrainConfig = field(default_factory=lambda: TrainConfig(
        n_adam=500, n_lbfgs=0, lr_adam=3e-4, lr_lbfgs=1.0,
        lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=100,
    ))
    n_lbfgs_final: int = 100       # L-BFGS steps after all cycles (Wu et al. 2023 equal-budget convention)
    lr_lbfgs_final: float = 1.0
    n_cycles: int = 3
    n_colloc_init: int = 2000
    n_aux_diag: int = 5000       # dense aux set for residual diagnosis
    n_rad_add: int = 200         # RAD points added per cycle
    residual_percentile: float = 90.0
    n_new_knots_per_subdomain: int = 5
    max_grid_size: int = 20      # cap on grid size after extension
    device: str = "cpu"
    probe_steps_ratio: float = 0.3   # fraction of phase budget used as a probe
    min_improvement: float = 0.10    # min relative L2 drop to continue full training
    n_targeted_cycles: int = 2       # cycles 0..n_targeted_cycles-1 use targeted extension;
                                     # remaining cycles use uniform extension (avoids
                                     # ill-conditioned lstsq from concentrated input-layer knots)


# ---------------------------------------------------------------------------
# Phase 2: Residual Diagnosis
# ---------------------------------------------------------------------------

def diagnose_residuals(
    model: nn.Module,
    pde_fn: Callable,
    n_aux: int,
    rng: np.random.Generator,
    domain: Dict[str, Tuple[float, float]],
    is_2d: bool = False,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a dense auxiliary set and evaluate the PDE residual magnitude.

    Args:
        domain: {'t': (t_lo, t_hi), 'x': (x_lo, x_hi)} or
                {'x': (...), 'y': (...)} for 2D
    Returns:
        (aux_pts, dim1_vals, dim2_vals) where dim1/2 are the coordinate arrays,
        along with a residual_magnitudes tensor (n_aux,).
    """
    if is_2d:
        x_lo, x_hi = domain["x"]
        y_lo, y_hi = domain["y"]
        xa = rng.uniform(x_lo, x_hi, (n_aux, 1)).astype(np.float32)
        ya = rng.uniform(y_lo, y_hi, (n_aux, 1)).astype(np.float32)
        x_t = torch.tensor(xa, requires_grad=True, device=device)
        y_t = torch.tensor(ya, requires_grad=True, device=device)
        res = pde_fn(model, x_t, y_t)
        return x_t.detach(), y_t.detach(), res.detach().abs()
    else:
        t_lo, t_hi = domain["t"]
        x_lo, x_hi = domain["x"]
        ta = rng.uniform(t_lo, t_hi, (n_aux, 1)).astype(np.float32)
        xa = rng.uniform(x_lo, x_hi, (n_aux, 1)).astype(np.float32)
        t_t = torch.tensor(ta, requires_grad=True, device=device)
        x_t = torch.tensor(xa, requires_grad=True, device=device)
        res = pde_fn(model, t_t, x_t)
        return t_t.detach(), x_t.detach(), res.detach().abs()


# ---------------------------------------------------------------------------
# Phase 3: Identify high-residual sub-domains
# ---------------------------------------------------------------------------

def identify_high_residual_regions(
    dim1_vals: torch.Tensor,
    dim2_vals: torch.Tensor,
    residuals: torch.Tensor,
    percentile: float = 90.0,
    n_bins: int = 10,
) -> List[Dict]:
    """
    Identify spatial sub-domains where the mean residual exceeds the threshold.

    Bins the domain in both dimensions and returns the bounding boxes of bins
    whose mean residual is above the *percentile*-th percentile.

    Returns:
        list of dicts: [{'dim': 0 or 1, 'lo': float, 'hi': float}, ...]
    """
    with torch.no_grad():
        d1 = dim1_vals.cpu().numpy().ravel()
        d2 = dim2_vals.cpu().numpy().ravel()
        r = residuals.cpu().numpy().ravel()

        regions = []

        for dim_idx, d in enumerate([d1, d2]):
            d_min, d_max = d.min(), d.max()
            bins = np.linspace(d_min, d_max, n_bins + 1)
            bin_means = np.zeros(n_bins)
            for b in range(n_bins):
                mask = (d >= bins[b]) & (d < bins[b + 1])
                if mask.sum() > 0:
                    bin_means[b] = r[mask].mean()

            # Compare bin means to the percentile of BIN MEANS (not individual values)
            non_zero = bin_means[bin_means > 0]
            if len(non_zero) == 0:
                continue
            threshold = np.percentile(non_zero, percentile)

            for b in range(n_bins):
                if bin_means[b] >= threshold:
                    regions.append({
                        "dim": dim_idx,
                        "lo": float(bins[b]),
                        "hi": float(bins[b + 1]),
                    })

    return regions


# ---------------------------------------------------------------------------
# Main RA-PIKAN training loop
# ---------------------------------------------------------------------------

def ra_pikan_train(
    model: BSplinePIKAN,
    pde_fn: Callable,
    ic_sampler: Optional[Callable],
    bc_sampler: Optional[Callable],
    domain: Dict[str, Tuple[float, float]],
    test_grid_fn: Callable,
    cfg: RAPiKANConfig,
    seed: int = 42,
    is_2d: bool = False,
) -> Dict:
    """
    Full RA-PIKAN training loop.

    Args:
        model:         BSplinePIKAN instance
        pde_fn:        pde residual function (model, coord1, coord2) -> residual
        ic_sampler:    callable -> (pts, u_vals) or None
        bc_sampler:    callable -> (pts, u_vals) or None
        domain:        dict of coordinate ranges
        test_grid_fn:  callable -> (pts, u_ref) for error evaluation
        cfg:           RAPiKANConfig
        seed:          random seed

    Returns:
        dict with: loss_history, l2_errors, gini_history, elapsed, n_params_history
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model.to(cfg.device)
    model.train()

    loss_history: List[float] = []
    l2_errors: List[float] = []
    gini_history: List[float] = []
    n_params_history: List[int] = []
    cycle_times: List[float] = []
    residual_snapshots: List[Dict] = []
    total_start = time.perf_counter()

    # Initial collocation points
    if is_2d:
        x_lo, x_hi = domain["x"]
        y_lo, y_hi = domain["y"]
        xa = rng.uniform(x_lo, x_hi, (cfg.n_colloc_init, 1)).astype(np.float32)
        ya = rng.uniform(y_lo, y_hi, (cfg.n_colloc_init, 1)).astype(np.float32)
        collocator_pts = (
            torch.tensor(xa, requires_grad=True, device=cfg.device),
            torch.tensor(ya, requires_grad=True, device=cfg.device),
        )
    else:
        t_lo, t_hi = domain["t"]
        x_lo, x_hi = domain["x"]
        ta = rng.uniform(t_lo, t_hi, (cfg.n_colloc_init, 1)).astype(np.float32)
        xa = rng.uniform(x_lo, x_hi, (cfg.n_colloc_init, 1)).astype(np.float32)
        collocator_pts = (
            torch.tensor(ta, requires_grad=True, device=cfg.device),
            torch.tensor(xa, requires_grad=True, device=cfg.device),
        )

    def collocator():
        return collocator_pts

    def _ic():
        return ic_sampler() if ic_sampler else None

    def _bc():
        return bc_sampler() if bc_sampler else None

    # ---- Phase 1: Base training ----------------------------------------
    print("=== Phase 1: Base training ===")
    n_params_history.append(sum(p.numel() for p in model.parameters()))
    result = train_model(
        model, collocator,
        ic_sampler if ic_sampler else None,
        bc_sampler if bc_sampler else None,
        pde_fn, cfg.phase1, is_2d=is_2d,
        loss_history=loss_history,
    )

    # Evaluate after phase 1
    pts_test, u_ref = test_grid_fn()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
        model.train()
    l2_errors.append(relative_l2_error(pred, u_ref))
    print(f"  After Phase 1: L2={l2_errors[-1]:.4e}")

    # ---- Refinement cycles (Phases 2-4) --------------------------------
    carry_adam = 0  # steps saved from early-stopped cycles, reused after all cycles
    for cycle in range(cfg.n_cycles):
        print(f"\n=== Cycle {cycle+1}/{cfg.n_cycles} ===")
        cycle_start = time.perf_counter()

        # Phase 2: Residual diagnosis
        d1, d2, residuals = diagnose_residuals(
            model, pde_fn, cfg.n_aux_diag, rng, domain, is_2d, cfg.device
        )
        gini = gini_coefficient(residuals)
        gini_history.append(gini)
        residual_snapshots.append({
            "aux_pts": torch.stack([d1.squeeze(), d2.squeeze()], dim=1).cpu().numpy(),
            "residuals": residuals.cpu().numpy(),
        })
        print(f"  Gini coefficient of residuals: {gini:.4f}")
        print(f"  Mean|residual|={residuals.mean():.4e}  Max={residuals.max():.4e}")

        # Phase 3a: Identify high-residual sub-domains
        regions = identify_high_residual_regions(
            d1, d2, residuals, percentile=cfg.residual_percentile
        )
        print(f"  High-residual regions found: {len(regions)}")

        # Phase 3b: Grid extension (targeted for early cycles, uniform for later ones)
        fit_pts = torch.stack([d1.squeeze(), d2.squeeze()], dim=1)

        current_grid_sizes = [
            layer.n_basis_per_dim()[0] - model.layers[0].spline_order + 1
            for layer in model.layers
        ]
        avg_grid_size = int(np.mean(current_grid_sizes))
        if avg_grid_size < cfg.max_grid_size:
            if cycle < cfg.n_targeted_cycles:
                # Targeted: concentrate knots in high-residual sub-domains
                for region in regions:
                    model.extend_targeted(
                        dim=region["dim"],
                        sub_low=region["lo"],
                        sub_high=region["hi"],
                        n_new_points=cfg.n_new_knots_per_subdomain,
                        fit_pts=fit_pts,
                    )
                # Extend deeper layers once per cycle (not once per region)
                model.extend_deeper_uniform(
                    n_new_points=cfg.n_new_knots_per_subdomain,
                    fit_pts=fit_pts,
                )
                new_n = sum(p.numel() for p in model.parameters())
                print(f"  After targeted extension: {new_n} params")
            else:
                # Uniform: spread knots across full domain — avoids ill-conditioned
                # lstsq that arises when input-layer knots are already very concentrated
                # from prior targeted extensions.
                current_n = model.layers[0].n_basis_per_dim()[0]
                new_grid_size = min(current_n + cfg.n_new_knots_per_subdomain, cfg.max_grid_size)
                if new_grid_size > current_n:
                    model.extend_uniform(new_grid_size, fit_pts=fit_pts)
                new_n = sum(p.numel() for p in model.parameters())
                print(f"  After uniform extension: {new_n} params")
        else:
            print(f"  Skipping extension (grid size {avg_grid_size} >= max {cfg.max_grid_size})")

        # Phase 3c: RAR-D style collocation augmentation (Wu et al., 2023, CMAME)
        # This follows RAR-D (Residual-based Adaptive Refinement - Distribution),
        # NOT pure RAD (which fully replaces the collocation set each cycle).
        # Differences from pure RAD:
        #   - Points drawn from the DIAGNOSIS set, not fresh domain samples
        #   - Existing collocation set is AUGMENTED (concatenated), not replaced
        #   - Sampling: p_i = |r_i| / sum(|r_j|)  [k=1, c=0 in Wu et al. formula]
        new_pts = rad_resample(
            (d1, d2), residuals, cfg.n_rad_add, rng, cfg.device
        )
        # Concatenate with existing collocation set
        collocator_pts_list = []
        for old, new in zip(collocator_pts, new_pts):
            combined = torch.cat([old.detach(), new.detach()], dim=0)
            collocator_pts_list.append(
                torch.tensor(combined.cpu().numpy(), dtype=torch.float32,
                             requires_grad=True, device=cfg.device)
            )
        collocator_pts = tuple(collocator_pts_list)
        print(f"  Collocation size: {collocator_pts[0].shape[0]}")

        n_params_history.append(sum(p.numel() for p in model.parameters()))

        # Phase 4: Probe-gated training
        print(f"  Phase 4: Continued training (cycle {cycle+1})")
        prev_l2 = l2_errors[-1]
        phase_steps = cfg.phase_refine.n_adam
        probe_n = max(50, int(phase_steps * cfg.probe_steps_ratio))
        remain_n = phase_steps - probe_n

        probe_cfg = TrainConfig(
            n_adam=probe_n, n_lbfgs=0,
            lr_adam=cfg.phase_refine.lr_adam,
            lr_lbfgs=cfg.phase_refine.lr_lbfgs,
            lambda_r=cfg.phase_refine.lambda_r,
            lambda_bc=cfg.phase_refine.lambda_bc,
            lambda_ic=cfg.phase_refine.lambda_ic,
            log_every=cfg.phase_refine.log_every,
        )
        train_model(
            model, lambda: collocator_pts,
            ic_sampler, bc_sampler,
            pde_fn, probe_cfg, is_2d=is_2d,
            loss_history=loss_history,
        )
        with torch.no_grad():
            model.eval()
            pred = model(pts_test).squeeze()
            model.train()
        probe_l2 = relative_l2_error(pred, u_ref)
        rel_improvement = (prev_l2 - probe_l2) / (prev_l2 + 1e-12)

        if rel_improvement >= cfg.min_improvement and remain_n > 0:
            remain_cfg = TrainConfig(
                n_adam=remain_n, n_lbfgs=0,
                lr_adam=cfg.phase_refine.lr_adam,
                lr_lbfgs=cfg.phase_refine.lr_lbfgs,
                lambda_r=cfg.phase_refine.lambda_r,
                lambda_bc=cfg.phase_refine.lambda_bc,
                lambda_ic=cfg.phase_refine.lambda_ic,
                log_every=cfg.phase_refine.log_every,
            )
            train_model(
                model, lambda: collocator_pts,
                ic_sampler, bc_sampler,
                pde_fn, remain_cfg, is_2d=is_2d,
                loss_history=loss_history,
            )
            with torch.no_grad():
                model.eval()
                pred = model(pts_test).squeeze()
                model.train()
            cycle_l2 = relative_l2_error(pred, u_ref)
        else:
            carry_adam += remain_n
            print(f"  Probe: {rel_improvement:+.1%} improvement; "
                  f"carrying {remain_n} steps forward")
            cycle_l2 = probe_l2

        l2_errors.append(cycle_l2)
        cycle_times.append(time.perf_counter() - cycle_start)
        print(f"  After cycle {cycle+1}: L2={l2_errors[-1]:.4e}  "
              f"({cycle_times[-1]:.1f}s)")

    # Bonus Adam from early-stopped cycles
    if carry_adam > 0:
        print(f"=== Bonus Adam phase ({carry_adam} steps) ===")
        bonus_cfg = TrainConfig(
            n_adam=carry_adam, n_lbfgs=0,
            lr_adam=cfg.phase_refine.lr_adam,
            lr_lbfgs=cfg.phase_refine.lr_lbfgs,
            lambda_r=cfg.phase_refine.lambda_r,
            lambda_bc=cfg.phase_refine.lambda_bc,
            lambda_ic=cfg.phase_refine.lambda_ic,
            log_every=cfg.phase_refine.log_every,
        )
        train_model(
            model, lambda: collocator_pts,
            ic_sampler, bc_sampler,
            pde_fn, bonus_cfg, is_2d=is_2d,
            loss_history=loss_history,
        )
        with torch.no_grad():
            model.eval()
            pred = model(pts_test).squeeze()
            model.train()
        bonus_l2 = relative_l2_error(pred, u_ref)
        l2_errors.append(bonus_l2)
        print(f"  After bonus Adam: L2={bonus_l2:.4e}")

    # Final L-BFGS polish after all cycles — equal budget per Wu et al. 2023 convention
    if cfg.n_lbfgs_final > 0:
        print("=== Final L-BFGS polish ===")
        lbfgs_cfg = TrainConfig(
            n_adam=0, n_lbfgs=cfg.n_lbfgs_final, lr_adam=3e-4, lr_lbfgs=cfg.lr_lbfgs_final,
            lambda_r=cfg.phase_refine.lambda_r,
            lambda_bc=cfg.phase_refine.lambda_bc,
            lambda_ic=cfg.phase_refine.lambda_ic,
        )
        train_model(
            model, lambda: collocator_pts,
            ic_sampler, bc_sampler,
            pde_fn, lbfgs_cfg, is_2d=is_2d,
            loss_history=loss_history,
        )

    total_elapsed = time.perf_counter() - total_start

    return {
        "loss_history": loss_history,
        "l2_errors": l2_errors,
        "gini_history": gini_history,
        "n_params_history": n_params_history,
        "total_elapsed": total_elapsed,
        "cycle_times": cycle_times,
        "final_l2": l2_errors[-1],
        "final_n_params": n_params_history[-1],
        "residual_snapshots": residual_snapshots,
    }


# ---------------------------------------------------------------------------
# Uniform-extension PIKAN training (Rigas et al. baseline)
# ---------------------------------------------------------------------------

def uniform_extension_train(
    model: BSplinePIKAN,
    pde_fn: Callable,
    ic_sampler: Optional[Callable],
    bc_sampler: Optional[Callable],
    domain: Dict[str, Tuple[float, float]],
    test_grid_fn: Callable,
    cfg: RAPiKANConfig,
    seed: int = 42,
    is_2d: bool = False,
) -> Dict:
    """
    Uniform-extension PIKAN: same as RA-PIKAN but grid extension is global
    rather than spatially targeted. Isolates the spatial-targeting contribution.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(cfg.device)
    model.train()

    loss_history: List[float] = []
    l2_errors: List[float] = []
    gini_history: List[float] = []
    n_params_history: List[int] = []
    residual_snapshots: List[Dict] = []
    total_start = time.perf_counter()

    if is_2d:
        x_lo, x_hi = domain["x"]
        y_lo, y_hi = domain["y"]
        xa = rng.uniform(x_lo, x_hi, (cfg.n_colloc_init, 1)).astype(np.float32)
        ya = rng.uniform(y_lo, y_hi, (cfg.n_colloc_init, 1)).astype(np.float32)
        collocator_pts = (
            torch.tensor(xa, requires_grad=True, device=cfg.device),
            torch.tensor(ya, requires_grad=True, device=cfg.device),
        )
    else:
        t_lo, t_hi = domain["t"]
        x_lo, x_hi = domain["x"]
        ta = rng.uniform(t_lo, t_hi, (cfg.n_colloc_init, 1)).astype(np.float32)
        xa = rng.uniform(x_lo, x_hi, (cfg.n_colloc_init, 1)).astype(np.float32)
        collocator_pts = (
            torch.tensor(ta, requires_grad=True, device=cfg.device),
            torch.tensor(xa, requires_grad=True, device=cfg.device),
        )

    n_params_history.append(sum(p.numel() for p in model.parameters()))

    print("=== Phase 1: Base training (Uniform-extension PIKAN) ===")
    train_model(
        model, lambda: collocator_pts,
        ic_sampler, bc_sampler,
        pde_fn, cfg.phase1, is_2d=is_2d, loss_history=loss_history,
    )

    pts_test, u_ref = test_grid_fn()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
        model.train()
    l2_errors.append(relative_l2_error(pred, u_ref))

    for cycle in range(cfg.n_cycles):
        print(f"\n=== Cycle {cycle+1}/{cfg.n_cycles} (Uniform extension) ===")
        # Residual for Gini tracking
        d1, d2, residuals = diagnose_residuals(
            model, pde_fn, cfg.n_aux_diag, rng, domain, is_2d, cfg.device
        )
        gini_history.append(gini_coefficient(residuals))
        residual_snapshots.append({
            "aux_pts": torch.stack([d1.squeeze(), d2.squeeze()], dim=1).cpu().numpy(),
            "residuals": residuals.cpu().numpy(),
        })

        # Uniform extension
        fit_pts = torch.stack([d1.squeeze(), d2.squeeze()], dim=1)
        current_n = model.layers[0].n_basis_per_dim()[0]
        new_grid_size = min(current_n + 5, cfg.max_grid_size)
        if new_grid_size > current_n:
            model.extend_uniform(new_grid_size, fit_pts=fit_pts)

        n_params_history.append(sum(p.numel() for p in model.parameters()))

        train_model(
            model, lambda: collocator_pts,
            ic_sampler, bc_sampler,
            pde_fn, cfg.phase_refine, is_2d=is_2d, loss_history=loss_history,
        )
        with torch.no_grad():
            model.eval()
            pred = model(pts_test).squeeze()
            model.train()
        l2_errors.append(relative_l2_error(pred, u_ref))
        print(f"  After cycle {cycle+1}: L2={l2_errors[-1]:.4e}")

    # Final L-BFGS polish after all cycles — equal budget per Wu et al. 2023 convention
    if cfg.n_lbfgs_final > 0:
        print("=== Final L-BFGS polish ===")
        lbfgs_cfg = TrainConfig(
            n_adam=0, n_lbfgs=cfg.n_lbfgs_final, lr_adam=3e-4, lr_lbfgs=cfg.lr_lbfgs_final,
            lambda_r=cfg.phase_refine.lambda_r,
            lambda_bc=cfg.phase_refine.lambda_bc,
            lambda_ic=cfg.phase_refine.lambda_ic,
        )
        train_model(
            model, lambda: collocator_pts,
            ic_sampler, bc_sampler,
            pde_fn, lbfgs_cfg, is_2d=is_2d,
            loss_history=loss_history,
        )

    return {
        "loss_history": loss_history,
        "l2_errors": l2_errors,
        "gini_history": gini_history,
        "n_params_history": n_params_history,
        "total_elapsed": time.perf_counter() - total_start,
        "final_l2": l2_errors[-1],
        "final_n_params": n_params_history[-1],
        "residual_snapshots": residual_snapshots,
    }
