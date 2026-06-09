"""
Sensitivity analysis: RA-PIKAN residual percentile threshold on Adv-Diff.

Runs RA-PIKAN on Pe=100 and Pe=500 with percentile thresholds in {70, 80, 90, 95}
across seeds 42, 123, 456.  Results saved to results/sensitivity/.

Usage:
    python experiments/run_sensitivity.py [--device cpu]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks import AdvectionDiffusionBenchmark
from src.metrics import count_parameters, memory_tracker, relative_l2_error
from src.models import BSplinePIKAN
from src.ra_pikan import RAPiKANConfig, ra_pikan_train
from src.trainers import TrainConfig

RESULTS_DIR = Path(__file__).parent.parent / "results" / "sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PERCENTILES = [70.0, 80.0, 90.0, 95.0]
PES = [100.0, 500.0]
SEEDS = [42, 123, 456]

GRID_SIZE_INIT = 5
SPLINE_ORDER = 3
N_COLLOC = 5000
N_BC = 300

_SKIP_CSV = {"loss_history", "l2_errors", "gini_history", "_pred", "_pts", "_ref", "residual_snapshots"}


def run_ra_pikan(Pe: float, seed: int, percentile: float, device: str) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bench = AdvectionDiffusionBenchmark(Pe=Pe, device=device, seed=seed)
    model = BSplinePIKAN([2, 10, 10, 1], GRID_SIZE_INIT, SPLINE_ORDER, (-0.1, 1.1)).to(device)
    cfg = RAPiKANConfig(
        n_cycles=3, n_colloc_init=N_COLLOC, n_rad_add=500, device=device,
        residual_percentile=percentile,
        phase1=TrainConfig(n_adam=500, n_lbfgs=0, lr_adam=1e-3,
                           lambda_r=1.0, lambda_bc=100.0, lambda_ic=0.0, log_every=100),
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
    result.update({
        "method": "ra_pikan",
        "Pe": Pe,
        "seed": seed,
        "percentile": percentile,
        "peak_bytes": mem["peak_bytes"],
        "n_params": result.get("final_n_params"),
        "elapsed": result.get("total_elapsed"),
    })
    return result


def save_result(result: Dict) -> None:
    Pe = result["Pe"]
    seed = result["seed"]
    pct = result["percentile"]
    row = {k: v for k, v in result.items() if k not in _SKIP_CSV}
    path = RESULTS_DIR / f"sensitivity_ra_pikan_Pe{Pe}_seed{seed}_p{int(pct)}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"  Saved {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--percentiles", type=float, nargs="+", default=PERCENTILES)
    parser.add_argument("--Pe", type=float, nargs="+", default=PES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    total = len(args.percentiles) * len(args.Pe) * len(args.seeds)
    done = 0
    for pct in args.percentiles:
        for Pe in args.Pe:
            for seed in args.seeds:
                path = RESULTS_DIR / f"sensitivity_ra_pikan_Pe{Pe}_seed{seed}_p{int(pct)}.csv"
                if path.exists():
                    print(f"  SKIP (exists): {path.name}")
                    done += 1
                    continue
                print(f"\n{'='*60}")
                print(f"  Sensitivity: Pe={Pe}  seed={seed}  percentile={pct}  [{done+1}/{total}]")
                print(f"{'='*60}")
                try:
                    result = run_ra_pikan(Pe, seed, pct, args.device)
                    save_result(result)
                    print(f"  => L2={result.get('final_l2', 'N/A'):.4e}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback; traceback.print_exc()
                done += 1


if __name__ == "__main__":
    main()
