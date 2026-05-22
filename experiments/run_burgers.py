"""
Run all methods on the 1D Burgers benchmark.

Usage:
    python experiments/run_burgers.py [--nu 0.01] [--seeds 42 123 456]
                                      [--methods all] [--device cpu]

Outputs:
    results/burgers_{method}_nu{nu}_seed{seed}.csv   (one row per method-seed)
    results/burgers_convergence_nu{nu}_seed{seed}.csv (loss/L2 history)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks import BurgersBenchmark
from src.metrics import count_parameters, memory_tracker, relative_l2_error
from src.models import BSplinePIKAN, ChebyPIKAN, MLPPINN
from src.ra_pikan import RAPiKANConfig, ra_pikan_train, uniform_extension_train
from src.trainers import TrainConfig, train_model

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---- Default hyper-parameters (same for all methods) ----------------------
DEFAULT_WIDTHS_KAN = [2, 10, 10, 1]
DEFAULT_WIDTHS_MLP = [2, 64, 64, 64, 1]
GRID_SIZE_INIT = 5
SPLINE_ORDER = 3
CHEBY_DEGREE = 5
N_COLLOC = 2000
N_IC = 200
N_BC = 200

BASE_TRAIN_CFG = TrainConfig(
    n_adam=2000, n_lbfgs=100, lr_adam=1e-3,
    lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=500,
)

# Faster config for BSpline-based methods (RA-PIKAN, uniform-extension)
# which have ~400ms/step vs ~50ms/step for MLP/ChebyKAN
FAST_TRAIN_CFG = TrainConfig(
    n_adam=500, n_lbfgs=0, lr_adam=1e-3,
    lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=100,
)


def run_mlp_pinn(nu: float, seed: int, device: str) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    bench = BurgersBenchmark(nu=nu, device=device, seed=seed)
    model = MLPPINN(DEFAULT_WIDTHS_MLP).to(device)

    def collocator():
        t, x = bench.sample_collocation(N_COLLOC)
        return t, x

    with memory_tracker() as mem:
        result = train_model(
            model, collocator,
            lambda: bench.sample_ic(N_IC),
            lambda: bench.sample_bc(N_BC),
            bench.pde_residual, BASE_TRAIN_CFG, is_2d=False,
        )

    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    l2 = relative_l2_error(pred, u_ref)
    return {
        "method": "mlp_pinn",
        "nu": nu, "seed": seed,
        "final_l2": l2,
        "n_params": count_parameters(model),
        "elapsed": result["elapsed_seconds"],
        "peak_bytes": mem["peak_bytes"],
        "loss_history": result["loss_history"],
        "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy(),
    }


def run_rad_pinn(nu: float, seed: int, device: str) -> Dict:
    """RAD-PINN: MLP with residual-adaptive collocation resampling.

    Step budget matches MLP-PINN: 2000 Adam + 100 L-BFGS total.
    Structure: 500 Adam (Phase 1) -> 3x(resample + 500 Adam) -> 100 L-BFGS.
    """
    from src.trainers import rad_resample

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    bench = BurgersBenchmark(nu=nu, device=device, seed=seed)
    model = MLPPINN(DEFAULT_WIDTHS_MLP).to(device)

    N_CYCLES = 3
    n_add = 200
    loss_history = []
    t0 = time.perf_counter()

    t_c, x_c = bench.sample_collocation(N_COLLOC)
    collocator_pts = (t_c, x_c)

    cfg_phase1 = TrainConfig(
        n_adam=500, n_lbfgs=0, lr_adam=1e-3,
        lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=500,
    )
    cfg_cycle = TrainConfig(
        n_adam=500, n_lbfgs=0, lr_adam=3e-4,
        lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=500,
    )
    cfg_lbfgs = TrainConfig(
        n_adam=0, n_lbfgs=100, lr_adam=3e-4, lr_lbfgs=1.0,
        lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=500,
    )

    with memory_tracker() as mem:
        train_model(
            model, lambda: collocator_pts,
            lambda: bench.sample_ic(N_IC),
            lambda: bench.sample_bc(N_BC),
            bench.pde_residual, cfg_phase1, is_2d=False,
            loss_history=loss_history,
        )
        for _ in range(N_CYCLES):
            t_cp, x_cp = collocator_pts
            t_r = t_cp.detach().requires_grad_(True)
            x_r = x_cp.detach().requires_grad_(True)
            res = bench.pde_residual(model, t_r, x_r).abs().detach()
            new_t, new_x = rad_resample((t_cp, x_cp), res, n_add, rng, device)
            collocator_pts = (
                torch.cat([t_cp.detach(), new_t], dim=0).requires_grad_(True),
                torch.cat([x_cp.detach(), new_x], dim=0).requires_grad_(True),
            )
            train_model(
                model, lambda: collocator_pts,
                lambda: bench.sample_ic(N_IC),
                lambda: bench.sample_bc(N_BC),
                bench.pde_residual, cfg_cycle, is_2d=False,
                loss_history=loss_history,
            )
        train_model(
            model, lambda: collocator_pts,
            lambda: bench.sample_ic(N_IC),
            lambda: bench.sample_bc(N_BC),
            bench.pde_residual, cfg_lbfgs, is_2d=False,
            loss_history=loss_history,
        )

    elapsed = time.perf_counter() - t0
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    l2 = relative_l2_error(pred, u_ref)
    return {
        "method": "rad_pinn",
        "nu": nu, "seed": seed,
        "final_l2": l2,
        "n_params": count_parameters(model),
        "elapsed": elapsed,
        "peak_bytes": mem["peak_bytes"],
        "loss_history": loss_history,
        "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy(),
    }


def run_fixed_pikan(nu: float, seed: int, device: str) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    bench = BurgersBenchmark(nu=nu, device=device, seed=seed)
    model = ChebyPIKAN(DEFAULT_WIDTHS_KAN, degree=CHEBY_DEGREE).to(device)

    with memory_tracker() as mem:
        result = train_model(
            model,
            lambda: bench.sample_collocation(N_COLLOC),
            lambda: bench.sample_ic(N_IC),
            lambda: bench.sample_bc(N_BC),
            bench.pde_residual, BASE_TRAIN_CFG, is_2d=False,
        )

    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    l2 = relative_l2_error(pred, u_ref)
    return {
        "method": "fixed_pikan",
        "nu": nu, "seed": seed,
        "final_l2": l2,
        "n_params": count_parameters(model),
        "elapsed": result["elapsed_seconds"],
        "peak_bytes": mem["peak_bytes"],
        "loss_history": result["loss_history"],
        "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy(),
    }


def run_fixed_bspline_pikan(nu: float, seed: int, device: str) -> Dict:
    """Fixed B-spline PIKAN: same BSplinePIKAN as RA-PIKAN, no refinement.

    Isolates the contribution of targeted refinement from basis-type choice.
    Budget: 2000 Adam + 100 L-BFGS (matches all other methods on Burgers/Allen-Cahn).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    bench = BurgersBenchmark(nu=nu, device=device, seed=seed)
    model = BSplinePIKAN(
        DEFAULT_WIDTHS_KAN, grid_size=GRID_SIZE_INIT,
        spline_order=SPLINE_ORDER, grid_range=(-1.5, 1.5),
    ).to(device)
    cfg = TrainConfig(
        n_adam=2000, n_lbfgs=100, lr_adam=1e-3, lr_lbfgs=1.0,
        lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=500,
    )
    with memory_tracker() as mem:
        result = train_model(
            model,
            lambda: bench.sample_collocation(N_COLLOC),
            lambda: bench.sample_ic(N_IC),
            lambda: bench.sample_bc(N_BC),
            bench.pde_residual, cfg, is_2d=False,
        )
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
    return {
        "method": "fixed_bspline_pikan", "nu": nu, "seed": seed,
        "final_l2": relative_l2_error(pred, u_ref),
        "n_params": count_parameters(model),
        "elapsed": result["elapsed_seconds"],
        "steps_per_second": result["steps_per_second"],
        "peak_bytes": mem["peak_bytes"],
        "loss_history": result["loss_history"],
        "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy(),
    }


def run_uniform_pikan(nu: float, seed: int, device: str) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    bench = BurgersBenchmark(nu=nu, device=device, seed=seed)
    model = BSplinePIKAN(
        DEFAULT_WIDTHS_KAN, grid_size=GRID_SIZE_INIT,
        spline_order=SPLINE_ORDER, grid_range=(-1.5, 1.5),
    ).to(device)
    cfg = RAPiKANConfig(n_cycles=3, n_colloc_init=N_COLLOC, device=device,
                        phase1=FAST_TRAIN_CFG)

    with memory_tracker() as mem:
        result = uniform_extension_train(
            model,
            bench.pde_residual,
            lambda: bench.sample_ic(N_IC),
            lambda: bench.sample_bc(N_BC),
            domain={"t": (0.0, 1.0), "x": (-1.0, 1.0)},
            test_grid_fn=bench.test_grid,
            cfg=cfg, seed=seed, is_2d=False,
        )

    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
    result.update({"method": "uniform_pikan", "nu": nu, "seed": seed,
                   "peak_bytes": mem["peak_bytes"],
                   "n_params": result.get("final_n_params"),
                   "elapsed": result.get("total_elapsed"),
                   "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()})
    return result


def run_ra_pikan(nu: float, seed: int, device: str) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    bench = BurgersBenchmark(nu=nu, device=device, seed=seed)
    model = BSplinePIKAN(
        DEFAULT_WIDTHS_KAN, grid_size=GRID_SIZE_INIT,
        spline_order=SPLINE_ORDER, grid_range=(-1.5, 1.5),
    ).to(device)
    cfg = RAPiKANConfig(n_cycles=3, n_colloc_init=N_COLLOC, device=device)

    with memory_tracker() as mem:
        result = ra_pikan_train(
            model,
            bench.pde_residual,
            lambda: bench.sample_ic(N_IC),
            lambda: bench.sample_bc(N_BC),
            domain={"t": (0.0, 1.0), "x": (-1.0, 1.0)},
            test_grid_fn=bench.test_grid,
            cfg=cfg, seed=seed, is_2d=False,
        )

    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
    result.update({"method": "ra_pikan", "nu": nu, "seed": seed,
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


_SKIP_CSV = {"loss_history", "_pred", "_pts", "_ref", "residual_snapshots"}


def save_result(result: Dict) -> None:
    method = result["method"]
    nu = result["nu"]
    seed = result["seed"]

    # Solution field snapshot (seed 42 only)
    if seed == 42 and "_pred" in result:
        np.savez_compressed(
            RESULTS_DIR / f"burgers_{method}_nu{nu}_seed42_pred.npz",
            pts=result["_pts"], pred=result["_pred"], exact=result["_ref"],
        )
    if seed == 42 and result.get("residual_snapshots"):
        snaps = result["residual_snapshots"]
        np.savez_compressed(
            RESULTS_DIR / f"burgers_{method}_nu{nu}_seed42_residuals.npz",
            aux_pts=np.stack([s["aux_pts"] for s in snaps]),
            residuals=np.stack([s["residuals"] for s in snaps]),
        )

    # Summary row
    summary_path = RESULTS_DIR / f"burgers_{method}_nu{nu}_seed{seed}.csv"
    row = {k: v for k, v in result.items() if k not in _SKIP_CSV}
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    # Convergence history
    if "loss_history" in result:
        conv_path = RESULTS_DIR / f"burgers_convergence_{method}_nu{nu}_seed{seed}.csv"
        with open(conv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
            for i, l in enumerate(result["loss_history"]):
                writer.writerow([i, l])

    # L2 history (for multi-cycle methods)
    if "l2_errors" in result:
        l2_path = RESULTS_DIR / f"burgers_l2history_{method}_nu{nu}_seed{seed}.csv"
        with open(l2_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cycle", "l2_error"])
            for i, l2 in enumerate(result["l2_errors"]):
                writer.writerow([i, l2])

    # Gini history (for multi-cycle KAN methods)
    if "gini_history" in result and result["gini_history"]:
        gini_path = RESULTS_DIR / f"burgers_ginihistory_{method}_nu{nu}_seed{seed}.csv"
        with open(gini_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cycle", "gini"])
            for i, g in enumerate(result["gini_history"]):
                writer.writerow([i + 1, g])

    print(f"  Saved {summary_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Run Burgers benchmark")
    parser.add_argument("--nu", type=float, nargs="+", default=[0.01, 0.005, 0.001])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--methods", type=str, nargs="+", default=list(METHOD_RUNNERS.keys()))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Running Burgers benchmark: nu={args.nu}, seeds={args.seeds}, "
          f"methods={args.methods}, device={args.device}")

    for nu in args.nu:
        for seed in args.seeds:
            for method in args.methods:
                summary_path = RESULTS_DIR / f"burgers_{method}_nu{nu}_seed{seed}.csv"
                if summary_path.exists():
                    print(f"  SKIP (exists): {summary_path.name}")
                    continue
                print(f"\n{'='*60}")
                print(f"  nu={nu}  seed={seed}  method={method}")
                print(f"{'='*60}")
                try:
                    result = METHOD_RUNNERS[method](nu, seed, args.device)
                    save_result(result)
                    print(f"  => L2={result.get('final_l2', 'N/A'):.4e}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
