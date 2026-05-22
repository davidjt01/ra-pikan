"""
Run all methods on the 2D steady advection–diffusion benchmark.

Usage:
    python experiments/run_advection_diffusion.py [--Pe 100 500] [--seeds 42 123 456]
                                                    [--methods all] [--device cpu]

Outputs (in results/):
    adv_diff_{method}_Pe{Pe}_seed{seed}.csv
    adv_diff_convergence_{method}_Pe{Pe}_seed{seed}.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks import AdvectionDiffusionBenchmark
from src.metrics import count_parameters, memory_tracker, relative_l2_error
from src.models import BSplinePIKAN, ChebyPIKAN, MLPPINN
from src.ra_pikan import RAPiKANConfig, ra_pikan_train, uniform_extension_train
from src.trainers import TrainConfig, train_model, rad_resample

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_WIDTHS_KAN = [2, 10, 10, 1]
DEFAULT_WIDTHS_MLP = [2, 64, 64, 64, 1]
GRID_SIZE_INIT = 5
SPLINE_ORDER = 3
CHEBY_DEGREE = 5
N_COLLOC = 5000
N_BC = 300

BASE_CFG = TrainConfig(
    n_adam=3000, n_lbfgs=200, lr_adam=1e-3,
    lambda_r=1.0, lambda_bc=100.0, lambda_ic=0.0, log_every=500,
)


def run_mlp_pinn(Pe: float, seed: int, device: str) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bench = AdvectionDiffusionBenchmark(Pe=Pe, device=device, seed=seed)
    model = MLPPINN(DEFAULT_WIDTHS_MLP).to(device)
    with memory_tracker() as mem:
        result = train_model(
            model, lambda: bench.sample_collocation(N_COLLOC),
            None, lambda: bench.sample_bc(N_BC),
            bench.pde_residual, BASE_CFG, is_2d=True,
        )
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    return {"method": "mlp_pinn", "Pe": Pe, "seed": seed,
            "final_l2": relative_l2_error(pred, u_ref),
            "n_params": count_parameters(model),
            "elapsed": result["elapsed_seconds"],
            "peak_bytes": mem["peak_bytes"],
            "loss_history": result["loss_history"],
            "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}


def run_rad_pinn(Pe: float, seed: int, device: str) -> Dict:
    """RAD-PINN: MLP with residual-adaptive collocation resampling.

    Step budget matches MLP-PINN: 3000 Adam + 200 L-BFGS total.
    Structure: 750 Adam (Phase 1) -> 3x(resample + 750 Adam) -> 200 L-BFGS.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    bench = AdvectionDiffusionBenchmark(Pe=Pe, device=device, seed=seed)
    model = MLPPINN(DEFAULT_WIDTHS_MLP).to(device)
    N_CYCLES, n_add = 3, 500
    loss_history = []
    t0 = time.perf_counter()

    x_c, y_c = bench.sample_collocation(N_COLLOC)
    collocator_pts = (x_c, y_c)

    cfg_phase1 = TrainConfig(
        n_adam=750, n_lbfgs=0, lr_adam=1e-3,
        lambda_r=1.0, lambda_bc=100.0, lambda_ic=0.0, log_every=500,
    )
    cfg_cycle = TrainConfig(
        n_adam=750, n_lbfgs=0, lr_adam=3e-4,
        lambda_r=1.0, lambda_bc=100.0, lambda_ic=0.0, log_every=500,
    )
    cfg_lbfgs = TrainConfig(
        n_adam=0, n_lbfgs=200, lr_adam=3e-4, lr_lbfgs=1.0,
        lambda_r=1.0, lambda_bc=100.0, lambda_ic=0.0, log_every=500,
    )

    with memory_tracker() as mem:
        train_model(model, lambda: collocator_pts, None, lambda: bench.sample_bc(N_BC),
                    bench.pde_residual, cfg_phase1, is_2d=True, loss_history=loss_history)
        for _ in range(N_CYCLES):
            x_cp, y_cp = collocator_pts
            x_r = x_cp.detach().requires_grad_(True)
            y_r = y_cp.detach().requires_grad_(True)
            res = bench.pde_residual(model, x_r, y_r).abs().detach()
            new_x, new_y = rad_resample((x_cp, y_cp), res, n_add, rng, device)
            collocator_pts = (
                torch.cat([x_cp.detach(), new_x], dim=0).requires_grad_(True),
                torch.cat([y_cp.detach(), new_y], dim=0).requires_grad_(True),
            )
            train_model(model, lambda: collocator_pts, None, lambda: bench.sample_bc(N_BC),
                        bench.pde_residual, cfg_cycle, is_2d=True, loss_history=loss_history)
        train_model(model, lambda: collocator_pts, None, lambda: bench.sample_bc(N_BC),
                    bench.pde_residual, cfg_lbfgs, is_2d=True, loss_history=loss_history)

    elapsed = time.perf_counter() - t0
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    return {"method": "rad_pinn", "Pe": Pe, "seed": seed,
            "final_l2": relative_l2_error(pred, u_ref),
            "n_params": count_parameters(model),
            "elapsed": elapsed,
            "peak_bytes": mem["peak_bytes"],
            "loss_history": loss_history,
            "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}


def run_fixed_pikan(Pe: float, seed: int, device: str) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bench = AdvectionDiffusionBenchmark(Pe=Pe, device=device, seed=seed)
    model = ChebyPIKAN(DEFAULT_WIDTHS_KAN, degree=CHEBY_DEGREE).to(device)
    with memory_tracker() as mem:
        result = train_model(
            model, lambda: bench.sample_collocation(N_COLLOC),
            None, lambda: bench.sample_bc(N_BC),
            bench.pde_residual, BASE_CFG, is_2d=True,
        )
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    return {"method": "fixed_pikan", "Pe": Pe, "seed": seed,
            "final_l2": relative_l2_error(pred, u_ref),
            "n_params": count_parameters(model),
            "elapsed": result["elapsed_seconds"],
            "peak_bytes": mem["peak_bytes"],
            "loss_history": result["loss_history"],
            "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}


def run_fixed_bspline_pikan(Pe: float, seed: int, device: str) -> Dict:
    """Fixed B-spline PIKAN: same BSplinePIKAN as RA-PIKAN, no refinement.

    Isolates the contribution of targeted refinement from basis-type choice.
    Budget: 3000 Adam + 200 L-BFGS (matches all other methods on Adv-Diff).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    bench = AdvectionDiffusionBenchmark(Pe=Pe, device=device, seed=seed)
    model = BSplinePIKAN(DEFAULT_WIDTHS_KAN, GRID_SIZE_INIT, SPLINE_ORDER, (-0.1, 1.1)).to(device)
    cfg = TrainConfig(
        n_adam=3000, n_lbfgs=200, lr_adam=1e-3, lr_lbfgs=1.0,
        lambda_r=1.0, lambda_bc=100.0, lambda_ic=0.0, log_every=500,
    )
    with memory_tracker() as mem:
        result = train_model(
            model, lambda: bench.sample_collocation(N_COLLOC),
            None, lambda: bench.sample_bc(N_BC),
            bench.pde_residual, cfg, is_2d=True,
        )
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
    return {"method": "fixed_bspline_pikan", "Pe": Pe, "seed": seed,
            "final_l2": relative_l2_error(pred, u_ref),
            "n_params": count_parameters(model),
            "elapsed": result["elapsed_seconds"],
            "steps_per_second": result["steps_per_second"],
            "peak_bytes": mem["peak_bytes"],
            "loss_history": result["loss_history"],
            "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}


def run_uniform_pikan(Pe: float, seed: int, device: str) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bench = AdvectionDiffusionBenchmark(Pe=Pe, device=device, seed=seed)
    model = BSplinePIKAN(DEFAULT_WIDTHS_KAN, GRID_SIZE_INIT, SPLINE_ORDER, (-0.1, 1.1)).to(device)
    # Budget: 750 + 3×750 + 200 L-BFGS = 3000 + 200 (matches all other Adv-Diff methods)
    cfg = RAPiKANConfig(
        n_cycles=3, n_colloc_init=N_COLLOC, device=device,
        phase1=TrainConfig(n_adam=750, n_lbfgs=0, lr_adam=1e-3,
                           lambda_r=1.0, lambda_bc=100.0, lambda_ic=0.0, log_every=100),
        phase_refine=TrainConfig(n_adam=750, n_lbfgs=0, lr_adam=3e-4,
                                 lambda_r=1.0, lambda_bc=100.0, lambda_ic=0.0, log_every=100),
        n_lbfgs_final=200, lr_lbfgs_final=1.0,
    )
    with memory_tracker() as mem:
        result = uniform_extension_train(
            model, bench.pde_residual, None, lambda: bench.sample_bc(N_BC),
            domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            test_grid_fn=bench.test_grid, cfg=cfg, seed=seed, is_2d=True,
        )
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
    result.update({"method": "uniform_pikan", "Pe": Pe, "seed": seed,
                   "peak_bytes": mem["peak_bytes"],
                   "n_params": result.get("final_n_params"),
                   "elapsed": result.get("total_elapsed"),
                   "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()})
    return result


def run_ra_pikan(Pe: float, seed: int, device: str) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bench = AdvectionDiffusionBenchmark(Pe=Pe, device=device, seed=seed)
    model = BSplinePIKAN(DEFAULT_WIDTHS_KAN, GRID_SIZE_INIT, SPLINE_ORDER, (-0.1, 1.1)).to(device)
    # Budget: 750 + 3×750 + 200 L-BFGS = 3000 + 200 (matches all other Adv-Diff methods)
    cfg = RAPiKANConfig(
        n_cycles=3, n_colloc_init=N_COLLOC, n_rad_add=500, device=device,
        residual_percentile=_RA_PERCENTILE,
        phase1=TrainConfig(n_adam=750, n_lbfgs=0, lr_adam=1e-3,
                           lambda_r=1.0, lambda_bc=100.0, lambda_ic=0.0, log_every=100),
        phase_refine=TrainConfig(n_adam=750, n_lbfgs=0, lr_adam=3e-4,
                                 lambda_r=1.0, lambda_bc=100.0, lambda_ic=0.0, log_every=100),
        n_lbfgs_final=200, lr_lbfgs_final=1.0,
    )
    with memory_tracker() as mem:
        result = ra_pikan_train(
            model, bench.pde_residual, None, lambda: bench.sample_bc(N_BC),
            domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            test_grid_fn=bench.test_grid, cfg=cfg, seed=seed, is_2d=True,
        )
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
    result.update({"method": "ra_pikan", "Pe": Pe, "seed": seed,
                   "peak_bytes": mem["peak_bytes"],
                   "n_params": result.get("final_n_params"),
                   "elapsed": result.get("total_elapsed"),
                   "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()})
    return result


METHOD_RUNNERS = {
    "mlp_pinn": run_mlp_pinn,
    "rad_pinn": run_rad_pinn,
    "fixed_pikan": run_fixed_pikan,
    "fixed_bspline_pikan": run_fixed_bspline_pikan,
    "uniform_pikan": run_uniform_pikan,
    "ra_pikan": run_ra_pikan,
}


_SKIP_CSV = {"loss_history", "l2_errors", "gini_history", "_pred", "_pts", "_ref", "residual_snapshots"}

# Module-level percentile; overridden by --percentile CLI argument
_RA_PERCENTILE: float = 90.0


def save_result(result: Dict) -> None:
    method = result["method"]
    Pe = result["Pe"]
    seed = result["seed"]

    # Solution field snapshot (seed 42 only)
    if seed == 42 and "_pred" in result:
        np.savez_compressed(
            RESULTS_DIR / f"adv_diff_{method}_Pe{Pe}_seed42_pred.npz",
            pts=result["_pts"], pred=result["_pred"], exact=result["_ref"],
        )
    if seed == 42 and result.get("residual_snapshots"):
        snaps = result["residual_snapshots"]
        np.savez_compressed(
            RESULTS_DIR / f"adv_diff_{method}_Pe{Pe}_seed42_residuals.npz",
            aux_pts=np.stack([s["aux_pts"] for s in snaps]),
            residuals=np.stack([s["residuals"] for s in snaps]),
        )

    row = {k: v for k, v in result.items() if k not in _SKIP_CSV}
    path = RESULTS_DIR / f"adv_diff_{method}_Pe{Pe}_seed{seed}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    if "loss_history" in result:
        conv = RESULTS_DIR / f"adv_diff_convergence_{method}_Pe{Pe}_seed{seed}.csv"
        with open(conv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
            for i, l in enumerate(result["loss_history"]):
                writer.writerow([i, l])
    if "l2_errors" in result:
        l2p = RESULTS_DIR / f"adv_diff_l2history_{method}_Pe{Pe}_seed{seed}.csv"
        with open(l2p, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cycle", "l2_error"])
            for i, l2 in enumerate(result["l2_errors"]):
                writer.writerow([i, l2])
    if "gini_history" in result and result["gini_history"]:
        gp = RESULTS_DIR / f"adv_diff_ginihistory_{method}_Pe{Pe}_seed{seed}.csv"
        with open(gp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cycle", "gini"])
            for i, g in enumerate(result["gini_history"]):
                writer.writerow([i + 1, g])
    print(f"  Saved {path.name}")


def main():
    global _RA_PERCENTILE
    parser = argparse.ArgumentParser()
    parser.add_argument("--Pe", type=float, nargs="+", default=[100.0, 500.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--methods", type=str, nargs="+", default=list(METHOD_RUNNERS.keys()))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--percentile", type=float, default=90.0,
                        help="Residual percentile threshold for RA-PIKAN spatial targeting")
    args = parser.parse_args()
    _RA_PERCENTILE = args.percentile

    for Pe in args.Pe:
        for seed in args.seeds:
            for method in args.methods:
                summary_path = RESULTS_DIR / f"adv_diff_{method}_Pe{Pe}_seed{seed}.csv"
                if summary_path.exists():
                    print(f"  SKIP (exists): {summary_path.name}")
                    continue
                print(f"\n{'='*60}\n  Adv-Diff Pe={Pe} seed={seed} method={method}\n{'='*60}")
                try:
                    result = METHOD_RUNNERS[method](Pe, seed, args.device)
                    save_result(result)
                    print(f"  => L2={result.get('final_l2', 'N/A'):.4e}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
