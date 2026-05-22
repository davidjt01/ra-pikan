"""
Run all methods on the 1D Duffing oscillator benchmark (symbolic extraction sub-study).

Usage:
    python experiments/run_duffing.py [--seeds 42 123 456] [--methods all] [--device cpu]

Outputs (in results/):
    duffing_{method}_seed{seed}.csv
    duffing_convergence_{method}_seed{seed}.csv
    duffing_symbolic_{method}_seed{seed}.csv   (for KAN methods)
    duffing_{method}_seed42_pred.npz           (solution field, seed 42 only)
    duffing_{method}_seed42_activations.npz    (KAN activations, seed 42 only)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks import DuffingBenchmark
from src.kan_layers import ChebyKANLayer
from src.metrics import count_parameters, memory_tracker, relative_l2_error
from src.models import BSplinePIKAN, ChebyPIKAN, MLPPINN
from src.ra_pikan import RAPiKANConfig, uniform_extension_train
from src.symbolic_extraction import symbolic_extraction_report
from src.trainers import TrainConfig, train_model

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_WIDTHS_KAN = [1, 10, 10, 1]
DEFAULT_WIDTHS_MLP = [1, 64, 64, 1]
GRID_SIZE_INIT = 5
SPLINE_ORDER = 3
CHEBY_DEGREE = 5
N_COLLOC = 200
T_MAX = 10.0

BASE_CFG = TrainConfig(
    n_adam=3000, n_lbfgs=200, lr_adam=1e-3,
    lambda_r=1.0, lambda_bc=50.0, lambda_ic=50.0, log_every=500,
)

_SKIP_CSV = {"loss_history", "symbolic", "l2_errors", "_pred", "_pts", "_ref", "_activations"}


def _ic_loss_fn(model, device, lambda_ic):
    """Compute initial condition loss: x(0)=1, dx/dt(0)=0."""
    t_ic = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True, device=device)
    x0 = model(t_ic)
    dx_dt = torch.autograd.grad(x0, t_ic, torch.ones_like(x0), create_graph=True)[0]
    return lambda_ic * ((x0 - 1.0) ** 2 + (dx_dt - 0.0) ** 2).mean()


def _save_kan_activations(model, RESULTS_DIR, prefix, device, t_lo=0.0, t_hi=T_MAX):
    """Save per-edge activation function samples for ChebyPIKAN."""
    n_pts = 300
    activation_data = {}
    for li, layer in enumerate(model.layers):
        if not isinstance(layer, ChebyKANLayer):
            continue
        # Use the natural input range for this layer
        if li == 0:
            x_grid = torch.linspace(t_lo, t_hi, n_pts, device=device).unsqueeze(1)
        else:
            x_grid = torch.linspace(-3.0, 3.0, n_pts, device=device).unsqueeze(1)
        x_in = x_grid.expand(-1, layer.in_features)  # (n_pts, in_features)
        with torch.no_grad():
            T = layer._chebyshev(x_in)  # (n_pts, in_features, degree+1)
        for i in range(layer.in_features):
            for j in range(layer.out_features):
                phi = (T[:, i, :] @ layer.coef[j, i, :]).detach().cpu().numpy()
                activation_data[f"L{li}_j{j}_i{i}"] = phi
        activation_data[f"L{li}_xgrid"] = x_grid.squeeze().cpu().numpy()
    np.savez_compressed(RESULTS_DIR / f"{prefix}_activations.npz", **activation_data)


class DuffingTrainer:
    """Thin wrapper to handle the 1D Duffing case (single input dimension)."""

    def __init__(self, bench: DuffingBenchmark, model, device: str) -> None:
        self.bench = bench
        self.model = model
        self.device = device

    def run(self, cfg: TrainConfig, loss_history: List[float]) -> Dict:
        bench = self.bench
        model = self.model
        device = self.device

        t0 = time.perf_counter()

        adam = torch.optim.Adam(model.parameters(), lr=cfg.lr_adam)
        for step in range(cfg.n_adam):
            adam.zero_grad()
            t_r = bench.sample_collocation(N_COLLOC)
            res = bench.pde_residual(model, t_r)
            loss_ode = cfg.lambda_r * (res ** 2).mean()
            loss_ic = _ic_loss_fn(model, device, cfg.lambda_ic)
            loss = loss_ode + loss_ic
            loss.backward()
            adam.step()
            loss_history.append(loss.item())
            if (step + 1) % cfg.log_every == 0:
                print(f"  Adam {step+1}/{cfg.n_adam}  loss={loss.item():.4e}")

        if cfg.n_lbfgs > 0:
            # Fix collocation once — L-BFGS requires a consistent (non-stochastic)
            # objective; the closure must return the same loss for the same parameters.
            t_fixed = bench.sample_collocation(N_COLLOC)
            lbfgs = torch.optim.LBFGS(model.parameters(), lr=cfg.lr_lbfgs,
                                       max_iter=cfg.n_lbfgs, history_size=50,
                                       line_search_fn="strong_wolfe")
            def closure():
                lbfgs.zero_grad()
                res = bench.pde_residual(model, t_fixed)
                loss = cfg.lambda_r * (res ** 2).mean() + _ic_loss_fn(model, device, cfg.lambda_ic)
                loss.backward()
                return loss
            lbfgs.step(closure)
            print("  L-BFGS done")

        return {"elapsed_seconds": time.perf_counter() - t0}


def run_mlp_pinn(seed: int, device: str) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bench = DuffingBenchmark(device=device, seed=seed)
    model = MLPPINN(DEFAULT_WIDTHS_MLP).to(device)
    loss_history = []
    with memory_tracker() as mem:
        trainer = DuffingTrainer(bench, model, device)
        result = trainer.run(BASE_CFG, loss_history)
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    return {"method": "mlp_pinn", "seed": seed,
            "final_l2": relative_l2_error(pred, u_ref),
            "n_params": count_parameters(model),
            "elapsed": result["elapsed_seconds"],
            "peak_bytes": mem["peak_bytes"],
            "loss_history": loss_history,
            "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}


def run_rad_pinn(seed: int, device: str) -> Dict:
    """RAD-PINN for Duffing: residual-adaptive resampling between cycles.

    Step budget matches MLP-PINN: 3000 Adam + 200 L-BFGS total.
    Structure: 750 Adam (Phase 1) -> 3x(resample + 750 Adam) -> 200 L-BFGS.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    bench = DuffingBenchmark(device=device, seed=seed)
    model = MLPPINN(DEFAULT_WIDTHS_MLP).to(device)
    N_CYCLES, n_add = 3, 50
    loss_history = []

    t_c = bench.sample_collocation(N_COLLOC)  # (N, 1)
    t0_time = time.perf_counter()

    def _train_fixed(t_fixed, n_steps, lr):
        adam = torch.optim.Adam(model.parameters(), lr=lr)
        for step in range(n_steps):
            adam.zero_grad()
            res = bench.pde_residual(model, t_fixed)
            loss = (res ** 2).mean() + _ic_loss_fn(model, device, BASE_CFG.lambda_ic)
            loss.backward()
            adam.step()
            loss_history.append(loss.item())

    with memory_tracker() as mem:
        _train_fixed(t_c, 750, 1e-3)
        for _ in range(N_CYCLES):
            t_r = t_c.detach().requires_grad_(True)
            res = bench.pde_residual(model, t_r).abs().detach().squeeze()
            weights = res / (res.sum() + 1e-12)
            idx = torch.multinomial(weights, num_samples=n_add, replacement=True)
            new_t = t_c[idx].detach()
            t_c = torch.cat([t_c.detach(), new_t], dim=0).requires_grad_(True)
            _train_fixed(t_c, 750, 3e-4)
        # L-BFGS final polish
        t_fixed = t_c
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0,
                                   max_iter=200, history_size=50,
                                   line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            res = bench.pde_residual(model, t_fixed)
            loss = (res ** 2).mean() + _ic_loss_fn(model, device, BASE_CFG.lambda_ic)
            loss.backward()
            return loss
        lbfgs.step(closure)
        print("  L-BFGS done")

    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    return {"method": "rad_pinn", "seed": seed,
            "final_l2": relative_l2_error(pred, u_ref),
            "n_params": count_parameters(model),
            "elapsed": time.perf_counter() - t0_time,
            "peak_bytes": mem["peak_bytes"],
            "loss_history": loss_history,
            "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}


def run_fixed_pikan(seed: int, device: str) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bench = DuffingBenchmark(device=device, seed=seed)
    model = ChebyPIKAN(DEFAULT_WIDTHS_KAN, degree=CHEBY_DEGREE).to(device)
    loss_history = []
    with memory_tracker() as mem:
        trainer = DuffingTrainer(bench, model, device)
        result = trainer.run(BASE_CFG, loss_history)
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    sym = symbolic_extraction_report(model, r2_threshold=0.90)
    out = {"method": "fixed_pikan", "seed": seed,
           "final_l2": relative_l2_error(pred, u_ref),
           "n_params": count_parameters(model),
           "elapsed": result["elapsed_seconds"],
           "peak_bytes": mem["peak_bytes"],
           "loss_history": loss_history,
           "symbolic": sym,
           "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}
    if seed == 42:
        _save_kan_activations(model, RESULTS_DIR, f"duffing_fixed_pikan_seed42", device)
    return out


def run_fixed_bspline_pikan(seed: int, device: str) -> Dict:
    """Fixed B-spline PIKAN: same BSplinePIKAN as uniform/RA PIKAN, no grid extension.

    Isolates the contribution of grid extension from basis-type choice.
    Budget: 3000 Adam + 200 L-BFGS (matches all other Duffing methods).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    bench = DuffingBenchmark(device=device, seed=seed)
    model = BSplinePIKAN(DEFAULT_WIDTHS_KAN, GRID_SIZE_INIT, SPLINE_ORDER, (0.0, T_MAX)).to(device)
    loss_history = []
    t0_time = time.perf_counter()

    def _train(n_steps, lr):
        adam = torch.optim.Adam(model.parameters(), lr=lr)
        for step in range(n_steps):
            adam.zero_grad()
            t_r = bench.sample_collocation(N_COLLOC)
            res = bench.pde_residual(model, t_r)
            loss = (res ** 2).mean() + _ic_loss_fn(model, device, BASE_CFG.lambda_ic)
            loss.backward()
            adam.step()
            loss_history.append(loss.item())

    with memory_tracker() as mem:
        _train(3000, 1e-3)
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=200,
                                   history_size=50, line_search_fn="strong_wolfe")
        t_fixed = bench.sample_collocation(N_COLLOC)
        def closure():
            lbfgs.zero_grad()
            res = bench.pde_residual(model, t_fixed)
            loss = (res ** 2).mean() + _ic_loss_fn(model, device, BASE_CFG.lambda_ic)
            loss.backward()
            return loss
        lbfgs.step(closure)
        print("  L-BFGS done")

    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
    return {"method": "fixed_bspline_pikan", "seed": seed,
            "final_l2": relative_l2_error(pred, u_ref),
            "n_params": count_parameters(model),
            "elapsed": time.perf_counter() - t0_time,
            "peak_bytes": mem["peak_bytes"],
            "loss_history": loss_history,
            "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}


def run_uniform_pikan(seed: int, device: str) -> Dict:
    """Uniform-extension BSplinePIKAN for Duffing (1D time domain).

    Budget: 750 + 3×750 + 200 L-BFGS = 3000 + 200 (matches all other Duffing methods).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    bench = DuffingBenchmark(device=device, seed=seed)
    model = BSplinePIKAN(DEFAULT_WIDTHS_KAN, GRID_SIZE_INIT, SPLINE_ORDER, (0.0, T_MAX)).to(device)
    loss_history = []
    t0_time = time.perf_counter()
    N_CYCLES = 3

    def _train(n_steps, lr):
        adam = torch.optim.Adam(model.parameters(), lr=lr)
        for step in range(n_steps):
            adam.zero_grad()
            t_r = bench.sample_collocation(N_COLLOC)
            res = bench.pde_residual(model, t_r)
            loss = (res ** 2).mean() + _ic_loss_fn(model, device, BASE_CFG.lambda_ic)
            loss.backward()
            adam.step()
            loss_history.append(loss.item())

    with memory_tracker() as mem:
        _train(750, 1e-3)
        for _ in range(N_CYCLES):
            fit_pts = bench.sample_collocation(N_COLLOC)  # (N, 1)
            current_n_basis = model.layers[0].n_basis_per_dim()[0]
            current_grid_size = current_n_basis - SPLINE_ORDER + 1
            new_grid_size = min(current_grid_size + 5, 20)
            if new_grid_size > current_grid_size:
                model.extend_uniform(new_grid_size, fit_pts=fit_pts)
            _train(750, 3e-4)
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=200,
                                   history_size=50, line_search_fn="strong_wolfe")
        t_fixed = bench.sample_collocation(N_COLLOC)
        def closure():
            lbfgs.zero_grad()
            res = bench.pde_residual(model, t_fixed)
            loss = (res ** 2).mean() + _ic_loss_fn(model, device, BASE_CFG.lambda_ic)
            loss.backward()
            return loss
        lbfgs.step(closure)
        print("  L-BFGS done")

    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        model.eval()
        pred = model(pts_test).squeeze()
    return {"method": "uniform_pikan", "seed": seed,
            "final_l2": relative_l2_error(pred, u_ref),
            "n_params": count_parameters(model),
            "elapsed": time.perf_counter() - t0_time,
            "peak_bytes": mem["peak_bytes"],
            "loss_history": loss_history,
            "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}


def run_ra_pikan(seed: int, device: str) -> Dict:
    """Simplified RA-PIKAN for 1D Duffing (ChebyPIKAN, 2 refinement cycles).

    Budget: 1000 + 2×1000 + 200 L-BFGS = 3000 + 200 (matches all other Duffing methods).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    bench = DuffingBenchmark(device=device, seed=seed)
    model = ChebyPIKAN(DEFAULT_WIDTHS_KAN, degree=CHEBY_DEGREE).to(device)
    loss_history = []
    t0_time = time.perf_counter()
    with memory_tracker() as mem:
        trainer = DuffingTrainer(bench, model, device)
        # Phase 1
        trainer.run(
            TrainConfig(n_adam=1000, n_lbfgs=0, lr_adam=1e-3,
                        lambda_r=1.0, lambda_ic=50.0, lambda_bc=0.0, log_every=500),
            loss_history
        )
        # Two refinement cycles
        for cycle in range(2):
            print(f"  RA-PIKAN cycle {cycle+1}/2")
            trainer.run(
                TrainConfig(n_adam=1000, n_lbfgs=0, lr_adam=3e-4,
                            lambda_r=1.0, lambda_ic=50.0, lambda_bc=0.0, log_every=500),
                loss_history
            )
        # Final L-BFGS polish
        trainer.run(
            TrainConfig(n_adam=0, n_lbfgs=200, lr_adam=3e-4, lr_lbfgs=1.0,
                        lambda_r=1.0, lambda_ic=50.0, lambda_bc=0.0, log_every=500),
            loss_history
        )
    pts_test, u_ref = bench.test_grid()
    with torch.no_grad():
        pred = model(pts_test).squeeze()
    sym = symbolic_extraction_report(model, r2_threshold=0.90)
    out = {"method": "ra_pikan", "seed": seed,
           "final_l2": relative_l2_error(pred, u_ref),
           "n_params": count_parameters(model),
           "elapsed": time.perf_counter() - t0_time,
           "peak_bytes": mem["peak_bytes"],
           "loss_history": loss_history,
           "symbolic": sym,
           "_pred": pred.cpu().numpy(), "_pts": pts_test.cpu().numpy(), "_ref": u_ref.cpu().numpy()}
    if seed == 42:
        _save_kan_activations(model, RESULTS_DIR, f"duffing_ra_pikan_seed42", device)
    return out


METHOD_RUNNERS = {
    "mlp_pinn": run_mlp_pinn,
    "rad_pinn": run_rad_pinn,
    "fixed_pikan": run_fixed_pikan,
    "fixed_bspline_pikan": run_fixed_bspline_pikan,
    "uniform_pikan": run_uniform_pikan,
    "ra_pikan": run_ra_pikan,
}


def save_result(result: Dict) -> None:
    method = result["method"]
    seed = result["seed"]

    # Solution field snapshot (seed 42 only)
    if seed == 42 and "_pred" in result:
        np.savez_compressed(
            RESULTS_DIR / f"duffing_{method}_seed42_pred.npz",
            pts=result["_pts"], pred=result["_pred"], exact=result["_ref"],
        )

    row = {k: v for k, v in result.items() if k not in _SKIP_CSV}
    path = RESULTS_DIR / f"duffing_{method}_seed{seed}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    if "loss_history" in result:
        conv = RESULTS_DIR / f"duffing_convergence_{method}_seed{seed}.csv"
        with open(conv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
            for i, l in enumerate(result["loss_history"]):
                writer.writerow([i, l])
    if "symbolic" in result and result["symbolic"]:
        sym_path = RESULTS_DIR / f"duffing_symbolic_{method}_seed{seed}.csv"
        with open(sym_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["layer_idx", "in_idx", "out_idx", "formula", "r2"])
            writer.writeheader()
            for row in result["symbolic"]:
                writer.writerow(row)
    print(f"  Saved {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--methods", type=str, nargs="+", default=list(METHOD_RUNNERS.keys()))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    for seed in args.seeds:
        for method in args.methods:
            summary_path = RESULTS_DIR / f"duffing_{method}_seed{seed}.csv"
            if summary_path.exists():
                print(f"  SKIP (exists): {summary_path.name}")
                continue
            print(f"\n{'='*60}\n  Duffing seed={seed} method={method}\n{'='*60}")
            try:
                result = METHOD_RUNNERS[method](seed, args.device)
                save_result(result)
                print(f"  => L2={result.get('final_l2', 'N/A'):.4e}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
