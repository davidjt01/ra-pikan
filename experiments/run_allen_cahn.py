"""
Run all methods on the 1D Allen–Cahn benchmark.

Usage:
    python experiments/run_allen_cahn.py [--eps 0.1 0.05] [--seeds 42 123 456]
                                          [--methods all] [--device cpu]
                                          [--n_adam N] [--n_cycles C]

Options:
    --n_adam N    Total Adam steps for all methods (default: 2000).
                  Multi-phase methods split evenly across (n_cycles + 1) phases.
    --n_cycles C  Refinement cycles for RA-PIKAN / Uniform-ext PIKAN (default: 3).
                  Non-default values of n_adam or n_cycles append a tag to filenames
                  (e.g. _n10000, _c2, or _n10000_c2) to avoid conflicts.

Outputs (in results/):
    allen_cahn_{method}_eps{eps}_seed{seed}.csv             (default budget + cycles)
    allen_cahn_{method}_eps{eps}_n{N}_seed{seed}.csv        (custom budget)
    allen_cahn_{method}_eps{eps}_c{C}_seed{seed}.csv        (custom cycles)
    allen_cahn_{method}_eps{eps}_n{N}_c{C}_seed{seed}.csv   (both custom)
    allen_cahn_convergence_{method}_eps{eps}[_tag]_seed{seed}.csv
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

from src.benchmarks import AllenCahnBenchmark
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
N_COLLOC = 2000
N_IC = 200
N_BC = 200
N_ADAM_DEFAULT = 2000
N_LBFGS = 100
N_CYCLES_DEFAULT = 3


def _make_configs(n_adam: int, n_cycles: int = N_CYCLES_DEFAULT):
    """Build TrainConfig and RAPiKANConfig for a given total Adam budget."""
    steps_per_phase = n_adam // (n_cycles + 1)
    base_cfg = TrainConfig(
        n_adam=n_adam, n_lbfgs=N_LBFGS, lr_adam=1e-3,
        lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=500,
    )
    fixed_bspline_cfg = TrainConfig(
        n_adam=n_adam, n_lbfgs=N_LBFGS, lr_adam=1e-3, lr_lbfgs=1.0,
        lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=500,
    )
    ra_cfg = RAPiKANConfig(
        phase1=TrainConfig(
            n_adam=steps_per_phase, n_lbfgs=0, lr_adam=1e-3, lr_lbfgs=1.0,
            lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=100,
        ),
        phase_refine=TrainConfig(
            n_adam=steps_per_phase, n_lbfgs=0, lr_adam=3e-4, lr_lbfgs=1.0,
            lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=100,
        ),
        n_lbfgs_final=N_LBFGS, n_cycles=n_cycles, n_colloc_init=N_COLLOC,
    )
    return base_cfg, fixed_bspline_cfg, ra_cfg, steps_per_phase


def run_mlp_pinn(eps: float, seed: int, device: str, n_adam: int = N_ADAM_DEFAULT, n_cycles: int = N_CYCLES_DEFAULT) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bench = AllenCahnBenchmark(eps=eps, device=device, seed=seed)
    model = MLPPINN(DEFAULT_WIDTHS_MLP).to(device)
    base_cfg, _, _, _ = _make_configs(n_adam, n_cycles)
    with memory_tracker() as mem:
        result = train_model(
            model, lambda: bench.sample_collocation(N_COLLOC),
            lambda: bench.sample_ic(N_IC), lambda: bench.sample_bc(N_BC),
            bench.pde_residual, base_cfg, is_2d=False,
        )
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    return {"method": "mlp_pinn", "eps": eps, "seed": seed,
            "final_l2": relative_l2_error(pred, u_ref),
            "n_params": count_parameters(model),
            "elapsed": result["elapsed_seconds"],
            "peak_bytes": mem["peak_bytes"],
            "loss_history": result["loss_history"],
            "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}


def run_rad_pinn(eps: float, seed: int, device: str, n_adam: int = N_ADAM_DEFAULT, n_cycles: int = N_CYCLES_DEFAULT) -> Dict:
    """RAD-PINN: MLP with residual-adaptive collocation resampling.

    Step budget matches MLP-PINN: n_adam Adam + N_LBFGS L-BFGS total.
    Structure: phase1 Adam -> n_cycles x (resample + phase_refine Adam) -> L-BFGS.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    bench = AllenCahnBenchmark(eps=eps, device=device, seed=seed)
    model = MLPPINN(DEFAULT_WIDTHS_MLP).to(device)
    n_add = 200
    loss_history = []
    t0 = time.perf_counter()
    _, _, _, steps_per_phase = _make_configs(n_adam, n_cycles)

    t_c, x_c = bench.sample_collocation(N_COLLOC)
    collocator_pts = (t_c, x_c)

    cfg_phase1 = TrainConfig(
        n_adam=steps_per_phase, n_lbfgs=0, lr_adam=1e-3,
        lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=500,
    )
    cfg_cycle = TrainConfig(
        n_adam=steps_per_phase, n_lbfgs=0, lr_adam=3e-4,
        lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=500,
    )
    cfg_lbfgs = TrainConfig(
        n_adam=0, n_lbfgs=N_LBFGS, lr_adam=3e-4, lr_lbfgs=1.0,
        lambda_r=1.0, lambda_bc=10.0, lambda_ic=10.0, log_every=500,
    )

    with memory_tracker() as mem:
        train_model(model, lambda: collocator_pts,
                    lambda: bench.sample_ic(N_IC), lambda: bench.sample_bc(N_BC),
                    bench.pde_residual, cfg_phase1, is_2d=False, loss_history=loss_history)
        for _ in range(n_cycles):
            t_cp, x_cp = collocator_pts
            t_r = t_cp.detach().requires_grad_(True)
            x_r = x_cp.detach().requires_grad_(True)
            res = bench.pde_residual(model, t_r, x_r).abs().detach()
            new_t, new_x = rad_resample((t_cp, x_cp), res, n_add, rng, device)
            collocator_pts = (
                torch.cat([t_cp.detach(), new_t], dim=0).requires_grad_(True),
                torch.cat([x_cp.detach(), new_x], dim=0).requires_grad_(True),
            )
            train_model(model, lambda: collocator_pts,
                        lambda: bench.sample_ic(N_IC), lambda: bench.sample_bc(N_BC),
                        bench.pde_residual, cfg_cycle, is_2d=False, loss_history=loss_history)
        train_model(model, lambda: collocator_pts,
                    lambda: bench.sample_ic(N_IC), lambda: bench.sample_bc(N_BC),
                    bench.pde_residual, cfg_lbfgs, is_2d=False, loss_history=loss_history)

    elapsed = time.perf_counter() - t0
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    return {"method": "rad_pinn", "eps": eps, "seed": seed,
            "final_l2": relative_l2_error(pred, u_ref),
            "n_params": count_parameters(model),
            "elapsed": elapsed,
            "peak_bytes": mem["peak_bytes"],
            "loss_history": loss_history,
            "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}


def run_fixed_pikan(eps: float, seed: int, device: str, n_adam: int = N_ADAM_DEFAULT, n_cycles: int = N_CYCLES_DEFAULT) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bench = AllenCahnBenchmark(eps=eps, device=device, seed=seed)
    model = ChebyPIKAN(DEFAULT_WIDTHS_KAN, degree=CHEBY_DEGREE).to(device)
    base_cfg, _, _, _ = _make_configs(n_adam, n_cycles)
    with memory_tracker() as mem:
        result = train_model(
            model, lambda: bench.sample_collocation(N_COLLOC),
            lambda: bench.sample_ic(N_IC), lambda: bench.sample_bc(N_BC),
            bench.pde_residual, base_cfg, is_2d=False,
        )
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    return {"method": "fixed_pikan", "eps": eps, "seed": seed,
            "final_l2": relative_l2_error(pred, u_ref),
            "n_params": count_parameters(model),
            "elapsed": result["elapsed_seconds"],
            "peak_bytes": mem["peak_bytes"],
            "loss_history": result["loss_history"],
            "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}


def run_fixed_bspline_pikan(eps: float, seed: int, device: str, n_adam: int = N_ADAM_DEFAULT, n_cycles: int = N_CYCLES_DEFAULT) -> Dict:
    """Fixed B-spline PIKAN: same BSplinePIKAN as RA-PIKAN, no refinement.

    Isolates the contribution of targeted refinement from basis-type choice.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    bench = AllenCahnBenchmark(eps=eps, device=device, seed=seed)
    model = BSplinePIKAN(DEFAULT_WIDTHS_KAN, GRID_SIZE_INIT, SPLINE_ORDER, (-1.5, 1.5)).to(device)
    _, fixed_bspline_cfg, _, _ = _make_configs(n_adam, n_cycles)
    with memory_tracker() as mem:
        result = train_model(
            model, lambda: bench.sample_collocation(N_COLLOC),
            lambda: bench.sample_ic(N_IC), lambda: bench.sample_bc(N_BC),
            bench.pde_residual, fixed_bspline_cfg, is_2d=False,
        )
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
    return {"method": "fixed_bspline_pikan", "eps": eps, "seed": seed,
            "final_l2": relative_l2_error(pred, u_ref),
            "n_params": count_parameters(model),
            "elapsed": result["elapsed_seconds"],
            "steps_per_second": result["steps_per_second"],
            "peak_bytes": mem["peak_bytes"],
            "loss_history": result["loss_history"],
            "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}


def run_uniform_pikan(eps: float, seed: int, device: str, n_adam: int = N_ADAM_DEFAULT, n_cycles: int = N_CYCLES_DEFAULT) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bench = AllenCahnBenchmark(eps=eps, device=device, seed=seed)
    model = BSplinePIKAN(DEFAULT_WIDTHS_KAN, GRID_SIZE_INIT, SPLINE_ORDER, (-1.5, 1.5)).to(device)
    _, _, cfg, _ = _make_configs(n_adam, n_cycles)
    cfg.device = device
    with memory_tracker() as mem:
        result = uniform_extension_train(
            model, bench.pde_residual,
            lambda: bench.sample_ic(N_IC), lambda: bench.sample_bc(N_BC),
            domain={"t": (0.0, 1.0), "x": (-1.0, 1.0)},
            test_grid_fn=bench.test_grid, cfg=cfg, seed=seed, is_2d=False,
        )
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
    result.update({"method": "uniform_pikan", "eps": eps, "seed": seed,
                   "peak_bytes": mem["peak_bytes"],
                   "n_params": result.get("final_n_params"),
                   "elapsed": result.get("total_elapsed"),
                   "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()})
    return result


def run_ra_pikan(eps: float, seed: int, device: str, n_adam: int = N_ADAM_DEFAULT, n_cycles: int = N_CYCLES_DEFAULT) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bench = AllenCahnBenchmark(eps=eps, device=device, seed=seed)
    model = BSplinePIKAN(DEFAULT_WIDTHS_KAN, GRID_SIZE_INIT, SPLINE_ORDER, (-1.5, 1.5)).to(device)
    _, _, cfg, _ = _make_configs(n_adam, n_cycles)
    cfg.device = device
    with memory_tracker() as mem:
        result = ra_pikan_train(
            model, bench.pde_residual,
            lambda: bench.sample_ic(N_IC), lambda: bench.sample_bc(N_BC),
            domain={"t": (0.0, 1.0), "x": (-1.0, 1.0)},
            test_grid_fn=bench.test_grid, cfg=cfg, seed=seed, is_2d=False,
        )
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
    result.update({"method": "ra_pikan", "eps": eps, "seed": seed,
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


def _run_tag(n_adam: int, n_cycles: int) -> str:
    """Return a filename tag encoding non-default budget and/or cycle count."""
    tag = f"_n{n_adam}" if n_adam != N_ADAM_DEFAULT else ""
    tag += f"_c{n_cycles}" if n_cycles != N_CYCLES_DEFAULT else ""
    return tag


def save_result(result: Dict, n_adam: int = N_ADAM_DEFAULT, n_cycles: int = N_CYCLES_DEFAULT) -> None:
    method = result["method"]
    eps = result["eps"]
    seed = result["seed"]
    tag = _run_tag(n_adam, n_cycles)

    # Solution field snapshot (seed 42 only, default budget and cycles only)
    if seed == 42 and not tag and "_pred" in result:
        np.savez_compressed(
            RESULTS_DIR / f"allen_cahn_{method}_eps{eps}_seed42_pred.npz",
            pts=result["_pts"], pred=result["_pred"], exact=result["_ref"],
        )
    if seed == 42 and not tag and result.get("residual_snapshots"):
        snaps = result["residual_snapshots"]
        np.savez_compressed(
            RESULTS_DIR / f"allen_cahn_{method}_eps{eps}_seed42_residuals.npz",
            aux_pts=np.stack([s["aux_pts"] for s in snaps]),
            residuals=np.stack([s["residuals"] for s in snaps]),
        )

    row = {k: v for k, v in result.items() if k not in _SKIP_CSV}
    path = RESULTS_DIR / f"allen_cahn_{method}_eps{eps}{tag}_seed{seed}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    if "loss_history" in result:
        conv = RESULTS_DIR / f"allen_cahn_convergence_{method}_eps{eps}{tag}_seed{seed}.csv"
        with open(conv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
            for i, l in enumerate(result["loss_history"]):
                writer.writerow([i, l])
    if "l2_errors" in result:
        l2p = RESULTS_DIR / f"allen_cahn_l2history_{method}_eps{eps}{tag}_seed{seed}.csv"
        with open(l2p, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cycle", "l2_error"])
            for i, l2 in enumerate(result["l2_errors"]):
                writer.writerow([i, l2])
    if "gini_history" in result and result["gini_history"]:
        gp = RESULTS_DIR / f"allen_cahn_ginihistory_{method}_eps{eps}{tag}_seed{seed}.csv"
        with open(gp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cycle", "gini"])
            for i, g in enumerate(result["gini_history"]):
                writer.writerow([i + 1, g])
    print(f"  Saved {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.05])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--methods", type=str, nargs="+", default=list(METHOD_RUNNERS.keys()))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_adam", type=int, default=N_ADAM_DEFAULT,
                        help="Total Adam steps for all methods (default: %(default)s). "
                             "Multi-phase methods split evenly across n_cycles+1 phases.")
    parser.add_argument("--n_cycles", type=int, default=N_CYCLES_DEFAULT,
                        help="Refinement cycles for RA-PIKAN and Uniform-ext PIKAN "
                             "(default: %(default)s). Non-default values append _c{n} to filenames.")
    args = parser.parse_args()
    tag = _run_tag(args.n_adam, args.n_cycles)

    for eps in args.eps:
        for seed in args.seeds:
            for method in args.methods:
                summary_path = RESULTS_DIR / f"allen_cahn_{method}_eps{eps}{tag}_seed{seed}.csv"
                if summary_path.exists():
                    print(f"  SKIP (exists): {summary_path.name}")
                    continue
                print(f"\n{'='*60}\n  Allen-Cahn eps={eps} seed={seed} method={method}\n{'='*60}")
                try:
                    result = METHOD_RUNNERS[method](eps, seed, args.device, args.n_adam, args.n_cycles)
                    save_result(result, args.n_adam, args.n_cycles)
                    print(f"  => L2={result.get('final_l2', 'N/A'):.4e}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
