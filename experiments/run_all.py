"""
Master script: run the full experiment suite.

Usage:
    python experiments/run_all.py [--device cpu] [--seeds 42 123 456 789 1234] [--quick]

--quick reduces problem sizes for a fast smoke-test.

Full suite (4 benchmarks × 6 methods × 5 seeds + Adv-Diff 10 seeds) takes ~250-400
hours on CPU.  Use --quick for a 15-30 minute test run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).parent.parent
EXPERIMENTS = ROOT / "experiments"


def run_script(script: str, extra_args: list) -> int:
    cmd = [PYTHON, str(EXPERIMENTS / script)] + extra_args
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full PIKAN experiment suite")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1234])
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke-test (1 seed, smallest nu/eps/Pe, fewer steps)")
    args = parser.parse_args()

    seeds_str = [str(s) for s in args.seeds]
    device = args.device

    if args.quick:
        print("=== QUICK MODE: 1 seed, smallest difficulty levels ===")
        seeds_str = ["42"]
        burgers_nu = ["0.01"]
        ac_eps = ["0.1"]
        ad_pe = ["100"]
        methods = ["mlp_pinn", "fixed_pikan", "fixed_bspline_pikan", "ra_pikan"]
    else:
        burgers_nu = ["0.01", "0.005", "0.001"]
        ac_eps = ["0.1", "0.05"]
        ad_pe = ["100.0", "500.0"]
        methods = ["mlp_pinn", "rad_pinn", "fixed_pikan", "fixed_bspline_pikan",
                   "uniform_pikan", "ra_pikan"]

    errors = []

    # 1D Burgers
    rc = run_script("run_burgers.py", [
        "--nu"] + burgers_nu + [
        "--seeds"] + seeds_str + [
        "--methods"] + methods + [
        "--device", device,
    ])
    if rc != 0:
        errors.append("run_burgers.py failed")

    # 1D Allen-Cahn
    rc = run_script("run_allen_cahn.py", [
        "--eps"] + ac_eps + [
        "--seeds"] + seeds_str + [
        "--methods"] + methods + [
        "--device", device,
    ])
    if rc != 0:
        errors.append("run_allen_cahn.py failed")

    # 2D Advection-Diffusion: 10 seeds for Wilcoxon signed-rank tests
    adv_diff_seeds = seeds_str if args.quick else [
        str(s) for s in [42, 123, 456, 789, 1234, 2024, 3141, 4096, 5678, 6789]
    ]
    rc = run_script("run_advection_diffusion.py", [
        "--Pe"] + ad_pe + [
        "--seeds"] + adv_diff_seeds + [
        "--methods"] + methods + [
        "--device", device,
    ])
    if rc != 0:
        errors.append("run_advection_diffusion.py failed")

    # 1D Duffing (symbolic sub-study)
    duffing_methods = methods if not args.quick else ["mlp_pinn", "fixed_pikan", "fixed_bspline_pikan", "ra_pikan"]
    rc = run_script("run_duffing.py", [
        "--seeds"] + seeds_str + [
        "--methods"] + duffing_methods + [
        "--device", device,
    ])
    if rc != 0:
        errors.append("run_duffing.py failed")

    # Sensitivity analysis (skipped in quick mode)
    if not args.quick:
        rc = run_script("run_sensitivity.py", [
            "--device", device,
            "--seeds"] + seeds_str,
        )
        if rc != 0:
            errors.append("run_sensitivity.py failed")

    print("\n=== Suite complete ===")
    if errors:
        print(f"  Errors: {errors}")
        sys.exit(1)
    else:
        print("  All benchmarks completed successfully.")
        print("  Run: python experiments/make_figures_and_tables.py")


if __name__ == "__main__":
    main()
