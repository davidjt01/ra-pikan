"""
Generate all paper figures and tables from results/ CSV files.

Usage:
    python experiments/make_figures_and_tables.py [--results_dir results/]
                                                   [--output_dir results/]

Outputs:
    results/table_accuracy.csv          — main accuracy table (all benchmarks)
    results/table_compute.csv           — compute cost table
    results/fig_convergence_burgers.pdf — convergence plots (Burgers)
    results/fig_convergence_ac.pdf      — convergence plots (Allen-Cahn)
    results/fig_convergence_ad.pdf      — convergence plots (Adv-Diff)
    results/fig_l2_vs_cycle.pdf         — L2 error vs refinement cycle
    results/fig_gini.pdf                — Gini coefficient evolution
    results/fig_duffing_symbolic.pdf    — Duffing symbolic extraction summary
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    # PLOS ONE font requirements: Arial, Times, or Symbol; 8-12 pt.
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
    matplotlib.rcParams['font.size'] = 9
    matplotlib.rcParams['axes.titlesize'] = 9
    matplotlib.rcParams['axes.labelsize'] = 9
    matplotlib.rcParams['xtick.labelsize'] = 8
    matplotlib.rcParams['ytick.labelsize'] = 8
    matplotlib.rcParams['legend.fontsize'] = 8
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available; skipping figures.")

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

METHOD_LABELS = {
    "mlp_pinn": "MLP-PINN",
    "rad_pinn": "RAR-D PINN",
    "fixed_pikan": "Fixed ChebyPIKAN",
    "fixed_bspline_pikan": "Fixed B-spline PIKAN",
    "uniform_pikan": "Uniform-ext. PIKAN",
    "ra_pikan": "RA-PIKAN (ours)",
}

METHOD_COLORS = {
    "mlp_pinn": "#1f77b4",
    "rad_pinn": "#ff7f0e",
    "fixed_pikan": "#2ca02c",
    "fixed_bspline_pikan": "#17becf",
    "uniform_pikan": "#9467bd",
    "ra_pikan": "#d62728",
}

METHOD_LINESTYLES = {
    "mlp_pinn": "-",
    "rad_pinn": "--",
    "fixed_pikan": "-.",
    "fixed_bspline_pikan": (0, (3, 1, 1, 1)),
    "uniform_pikan": ":",
    "ra_pikan": "-",
}


def load_summary_csv(pattern: str) -> pd.DataFrame:
    """Load all CSVs matching a glob pattern, concatenate."""
    files = list(RESULTS_DIR.glob(pattern))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_history_csv(pattern: str) -> Dict[str, pd.DataFrame]:
    """Load per-run history CSVs, keyed by filename stem."""
    files = list(RESULTS_DIR.glob(pattern))
    result = {}
    for f in files:
        try:
            result[f.stem] = pd.read_csv(f)
        except Exception:
            pass
    return result


def parse_method_from_stem(stem: str) -> Optional[str]:
    # Sort by descending length so "fixed_bspline_pikan" is checked before "fixed_pikan".
    for m in sorted(METHOD_LABELS.keys(), key=len, reverse=True):
        if m in stem:
            return m
    return None


def build_accuracy_table() -> pd.DataFrame:
    rows = []

    df = load_summary_csv("burgers_*_seed*.csv")
    df = df[~df["method"].isin(["convergence", "l2history"]) if "method" in df else df]
    for _, row in df.iterrows():
        if "final_l2" in row and "nu" in row:
            rows.append({
                "benchmark": "Burgers",
                "param": f"nu={row['nu']}",
                "method": row.get("method", ""),
                "seed": row.get("seed", ""),
                "l2_error": row["final_l2"],
                "n_params": row.get("n_params", ""),
                "elapsed_s": row.get("elapsed", ""),
            })

    df = load_summary_csv("allen_cahn_*_seed*.csv")
    for _, row in df.iterrows():
        if "final_l2" in row and "eps" in row:
            rows.append({
                "benchmark": "Allen-Cahn",
                "param": f"eps={row['eps']}",
                "method": row.get("method", ""),
                "seed": row.get("seed", ""),
                "l2_error": row["final_l2"],
                "n_params": row.get("n_params", ""),
                "elapsed_s": row.get("elapsed", ""),
            })

    df = load_summary_csv("adv_diff_*_seed*.csv")
    for _, row in df.iterrows():
        if "final_l2" in row and "Pe" in row:
            rows.append({
                "benchmark": "Adv-Diff",
                "param": f"Pe={row['Pe']}",
                "method": row.get("method", ""),
                "seed": row.get("seed", ""),
                "l2_error": row["final_l2"],
                "n_params": row.get("n_params", ""),
                "elapsed_s": row.get("elapsed", ""),
            })

    df = load_summary_csv("duffing_*_seed*.csv")
    for _, row in df.iterrows():
        if "final_l2" in row:
            rows.append({
                "benchmark": "Duffing",
                "param": "symbolic",
                "method": row.get("method", ""),
                "seed": row.get("seed", ""),
                "l2_error": row["final_l2"],
                "n_params": row.get("n_params", ""),
                "elapsed_s": row.get("elapsed", ""),
            })

    if not rows:
        print("  No summary data found; accuracy table empty.")
        return pd.DataFrame()

    df_all = pd.DataFrame(rows)
    agg = df_all.groupby(["benchmark", "param", "method"]).agg(
        l2_mean=("l2_error", "mean"),
        l2_std=("l2_error", "std"),
        n_params=("n_params", "first"),
        elapsed_mean=("elapsed_s", "mean"),
    ).reset_index()
    agg["l2_mean_std"] = agg.apply(
        lambda r: f"{r['l2_mean']:.3e} ± {r['l2_std']:.1e}" if not np.isnan(r['l2_std'])
        else f"{r['l2_mean']:.3e}", axis=1
    )
    GROUP_MAP = {
        "mlp_pinn": "A_MLP", "rad_pinn": "A_MLP",
        "fixed_pikan": "B_Chebyshev",
        "fixed_bspline_pikan": "C_BSpline",
        "uniform_pikan": "C_BSpline",
        "ra_pikan": "C_BSpline",
    }
    agg["method_group"] = agg["method"].map(GROUP_MAP).fillna("D_Other")
    return agg


def save_accuracy_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("  No data to save in accuracy table.")
        return
    path = RESULTS_DIR / "table_accuracy.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {path.name}")

    print("\n=== Accuracy Table ===")
    method_order = ["mlp_pinn", "rad_pinn", "fixed_pikan",
                    "fixed_bspline_pikan", "uniform_pikan", "ra_pikan"]
    pivot = df.pivot_table(
        index=["benchmark", "param"],
        columns="method",
        values="l2_mean_std",
        aggfunc="first",
    )
    ordered_cols = [m for m in method_order if m in pivot.columns]
    pivot = pivot[ordered_cols]
    print(pivot.to_string())


def build_compute_table() -> pd.DataFrame:
    rows = []
    for pat, bench in [
        ("burgers_*_seed*.csv", "Burgers"),
        ("allen_cahn_*_seed*.csv", "Allen-Cahn"),
        ("adv_diff_*_seed*.csv", "Adv-Diff"),
        ("duffing_*_seed*.csv", "Duffing"),
    ]:
        df = load_summary_csv(pat)
        for _, row in df.iterrows():
            if "elapsed" in row:
                rows.append({
                    "benchmark": bench,
                    "method": row.get("method", ""),
                    "n_params": row.get("n_params", ""),
                    "elapsed_s": row.get("elapsed", ""),
                    "peak_bytes": row.get("peak_bytes", ""),
                    "steps_per_second": row.get("steps_per_second", float("nan")),
                    "seed": row.get("seed", ""),
                })
    if not rows:
        return pd.DataFrame()
    df_all = pd.DataFrame(rows)
    df_all["steps_per_second"] = pd.to_numeric(df_all["steps_per_second"], errors="coerce")
    agg = df_all.groupby(["benchmark", "method"]).agg(
        n_params=("n_params", "first"),
        elapsed_mean=("elapsed_s", "mean"),
        peak_mb=("peak_bytes", lambda x: x.mean() / 1e6 if x.notna().any() else np.nan),
        steps_per_second=("steps_per_second", "mean"),
    ).reset_index()
    return agg


def save_compute_table(df: pd.DataFrame) -> None:
    if df.empty:
        return
    path = RESULTS_DIR / "table_compute.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {path.name}")


def plot_convergence(bench_prefix: str, param_str: str, output_name: str) -> None:
    if not HAS_MPL:
        return
    pattern = f"{bench_prefix}_convergence_*_{param_str}*.csv"
    histories = load_history_csv(pattern)
    if not histories:
        print(f"  No convergence data for {bench_prefix}/{param_str}")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = set()
    for stem, df in histories.items():
        method = parse_method_from_stem(stem)
        if method is None or method in plotted:
            continue
        if "loss" not in df.columns:
            continue
        ax.semilogy(
            df["step"], df["loss"],
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
            linestyle=METHOD_LINESTYLES.get(method, "-"),
            alpha=0.8,
        )
        plotted.add(method)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"Convergence: {bench_prefix} ({param_str})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / output_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {output_name}")


BENCH_DISPLAY = {"burgers": "Burgers", "allen_cahn": "Allen-Cahn", "adv_diff": "Adv-Diff"}


def plot_l2_vs_cycle(bench_prefix: str, param_str: str, output_name: str) -> None:
    """Per-seed L2 trajectories (thin) plus the per-seed mean (bold).

    ``param_str`` must not include a seed (e.g. "Pe100.0"); all available seeds
    are aggregated. A symmetric mean +/- std band is intentionally avoided: on a
    log axis it is misleading when one seed diverges mid-training, so individual
    seed trajectories are shown directly.
    """
    if not HAS_MPL:
        return
    from collections import defaultdict
    pattern = f"{bench_prefix}_l2history_*_{param_str}*.csv"
    histories = load_history_csv(pattern)
    if not histories:
        return

    by_method = defaultdict(list)
    for stem, df in histories.items():
        method = parse_method_from_stem(stem)
        if method is None or "l2_error" not in df.columns or "cycle" not in df.columns:
            continue
        by_method[method].append(df[["cycle", "l2_error"]])
    if not by_method:
        return

    fig, ax = plt.subplots(figsize=(5.8, 4))
    n_seeds = 0
    for method, dfs in by_method.items():
        n_seeds = max(n_seeds, len(dfs))
        color = METHOD_COLORS.get(method, None)
        for d in dfs:
            ax.plot(d["cycle"], d["l2_error"], color=color, alpha=0.22, lw=1.0, zorder=1)
        mean = pd.concat(dfs).groupby("cycle")["l2_error"].mean()
        ax.plot(mean.index, mean.values, color=color, lw=2.3, marker="o", ms=5,
                zorder=3, label=METHOD_LABELS.get(method, method) + " (mean)")

    ax.set_yscale("log")
    ax.set_xlabel("Training phase")
    ax.set_ylabel("Relative L² Error")
    disp = BENCH_DISPLAY.get(bench_prefix, bench_prefix)
    ax.set_title(f"L² error per phase: {disp} ({param_str}), {n_seeds} seeds")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / output_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {output_name}")


def plot_final_l2_bar(df: pd.DataFrame, benchmark: str, params: List[str], output_name: str) -> None:
    if not HAS_MPL or df.empty:
        return
    sub = df[df["benchmark"] == benchmark].copy()
    if sub.empty:
        return

    methods = sorted(sub["method"].unique(), key=lambda m: list(METHOD_LABELS.keys()).index(m)
                     if m in METHOD_LABELS else 99)
    n_params = len(params)
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_params, figsize=(min(4 * n_params, 7.5), 4), sharey=True)
    if n_params == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        sub_p = sub[sub["param"] == param]
        xs = np.arange(n_methods)
        means = [sub_p[sub_p["method"] == m]["l2_mean"].values[0]
                 if not sub_p[sub_p["method"] == m].empty else np.nan
                 for m in methods]
        stds = [sub_p[sub_p["method"] == m]["l2_std"].values[0]
                if not sub_p[sub_p["method"] == m].empty else 0.0
                for m in methods]
        bars = ax.bar(xs, means, yerr=stds,
                      color=[METHOD_COLORS.get(m, "#aaa") for m in methods],
                      capsize=4, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(param)
        ax.set_ylabel("Rel. L² Error")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{benchmark}: Final L² Error by Method")
    fig.tight_layout()
    out = RESULTS_DIR / output_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {output_name}")


def plot_symbolic_summary() -> None:
    if not HAS_MPL:
        return
    sym_files = list(RESULTS_DIR.glob("duffing_symbolic_*.csv"))
    if not sym_files:
        print("  No symbolic extraction data found.")
        return
    rows = []
    for f in sym_files:
        df = pd.read_csv(f)
        method = parse_method_from_stem(f.stem)
        seed_match = re.search(r"seed(\d+)", f.stem)
        seed = int(seed_match.group(1)) if seed_match else 0
        for _, row in df.iterrows():
            rows.append({
                "method": method, "seed": seed,
                "layer": row.get("layer_idx", 0),
                "formula": row.get("formula", ""),
                "r2": float(row.get("r2", 0.0)),
            })
    if not rows:
        return
    df_sym = pd.DataFrame(rows)
    print("\n=== Symbolic Extraction (R² >= 0.90) ===")
    high_r2 = df_sym[df_sym["r2"] >= 0.90]
    print(high_r2[["method", "seed", "layer", "formula", "r2"]].to_string(index=False))

    path = RESULTS_DIR / "table_symbolic.csv"
    df_sym.to_csv(path, index=False)
    print(f"  Saved {path.name}")

    fig, ax = plt.subplots(figsize=(6, 4))
    for m in df_sym["method"].unique():
        vals = df_sym[df_sym["method"] == m]["r2"].values
        ax.hist(vals, bins=20, alpha=0.6, label=METHOD_LABELS.get(m, m),
                color=METHOD_COLORS.get(m, None))
    ax.set_xlabel("R² of symbolic fit")
    ax.set_ylabel("Count")
    ax.set_title("Symbolic Extraction: R² Distribution (Duffing)")
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / "fig_duffing_symbolic.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved fig_duffing_symbolic.pdf")


def plot_solution_fields_1d(benchmark: str, param_key: str, param_val: str,
                             xlabel: str, output_name: str) -> None:
    """Plot predicted vs exact solution profiles for all methods (1D benchmarks)."""
    if not HAS_MPL:
        return
    methods = list(METHOD_LABELS.keys())
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(7.5, 3.5), sharey=True)

    exact_plotted = False
    for ax, method in zip(axes[:-1], methods):
        pattern = f"{benchmark}_{method}_{param_key}{param_val}_seed42_pred.npz"
        files = list(RESULTS_DIR.glob(pattern))
        if not files:
            ax.set_title(METHOD_LABELS.get(method, method), fontsize=8)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            continue
        data = np.load(files[0])
        pts = data["pts"]
        pred = data["pred"]
        exact = data["exact"]

        # Plot at the final time slice (t closest to 1.0) or spatial profile.
        if pts.shape[1] == 2:
            t_vals = pts[:, 0]
            x_vals = pts[:, 1]
            t_max = t_vals.max()
            mask = np.abs(t_vals - t_max) < 1e-6
            if mask.sum() == 0:
                mask = t_vals == t_vals.max()
            x_slice = x_vals[mask]
            sort_idx = np.argsort(x_slice)
            ax.plot(x_slice[sort_idx], pred[mask][sort_idx],
                    color=METHOD_COLORS.get(method, "C0"), lw=1.5)
            if not exact_plotted:
                axes[-1].plot(x_slice[sort_idx], exact[mask][sort_idx], "k-", lw=1.5, label="Exact")
                exact_plotted = True
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=7)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("u", fontsize=9)
    axes[-1].set_title("Exact", fontsize=8)
    axes[-1].set_xlabel(xlabel, fontsize=8)
    axes[-1].grid(True, alpha=0.3)
    fig.suptitle(f"Solution profiles: {benchmark} ({param_key}={param_val}, t=final)", fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / output_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {output_name}")


def plot_solution_field_2d(Pe: str, output_name: str) -> None:
    """Plot 2D Adv-Diff solution as heatmap for selected methods."""
    if not HAS_MPL:
        return
    show_methods = ["mlp_pinn", "ra_pikan"]
    titles = ["MLP-PINN", "RA-PIKAN (ours)", "Exact"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2))

    for ax, method in zip(axes[:2], show_methods):
        pattern = f"adv_diff_{method}_Pe{Pe}_seed42_pred.npz"
        files = list(RESULTS_DIR.glob(pattern))
        if not files:
            ax.set_title(METHOD_LABELS.get(method, method), fontsize=8)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        data = np.load(files[0])
        pts = data["pts"]
        pred = data["pred"]
        x, y = pts[:, 0], pts[:, 1]
        sc = ax.scatter(x, y, c=pred, cmap="viridis", s=3, vmin=0.0, vmax=1.0)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=8)
        plt.colorbar(sc, ax=ax, fraction=0.046)

    # Exact solution from the last file loaded.
    if files:
        data = np.load(files[0])
        pts = data["pts"]
        exact = data["exact"]
        x, y = pts[:, 0], pts[:, 1]
        sc = axes[2].scatter(x, y, c=exact, cmap="viridis", s=3, vmin=0.0, vmax=1.0)
        axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
        axes[2].set_title("Exact")
        plt.colorbar(sc, ax=axes[2], fraction=0.046)

    fig.suptitle(f"Adv-Diff solution (Pe={Pe}, seed 42)", fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / output_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {output_name}")


def plot_residual_maps(benchmark: str, param_key: str, param_val: str,
                       method: str, output_name: str) -> None:
    """Plot residual magnitude at each RA-PIKAN cycle."""
    if not HAS_MPL:
        return
    pattern = f"{benchmark}_{method}_{param_key}{param_val}_seed42_residuals.npz"
    files = list(RESULTS_DIR.glob(pattern))
    if not files:
        print(f"  No residual data for {benchmark}/{param_key}{param_val}/{method}")
        return
    data = np.load(files[0])
    aux_pts = data["aux_pts"]
    residuals = data["residuals"]
    n_cycles = aux_pts.shape[0]

    fig, axes = plt.subplots(1, n_cycles, figsize=(min(4 * n_cycles, 7.5), 3.5))
    if n_cycles == 1:
        axes = [axes]
    vmax = residuals.max()

    for ci, ax in enumerate(axes):
        pts = aux_pts[ci]
        res = residuals[ci]
        sc = ax.scatter(pts[:, 1], pts[:, 0], c=res, cmap="hot_r", s=4,
                        vmin=0.0, vmax=vmax, alpha=0.7)
        ax.set_title(f"Cycle {ci}", fontsize=9)
        ax.set_xlabel("x" if not benchmark.startswith("adv") else "x")
        ax.set_ylabel("t" if not benchmark.startswith("adv") else "y")
        plt.colorbar(sc, ax=ax, fraction=0.046, label="|residual|")

    fig.suptitle(f"Residual maps: {benchmark} {param_key}={param_val} ({METHOD_LABELS.get(method, method)})",
                 fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / output_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {output_name}")


def plot_gini_evolution(output_name: str = "fig_gini_evolution.pdf") -> None:
    """Plot Gini coefficient vs refinement cycle for RA-PIKAN and Uniform-PIKAN."""
    if not HAS_MPL:
        return

    # Glob patterns (all seeds aggregated). Adv-diff is shown at Pe=500 to match
    # the Gini values quoted in the Results text.
    configs = [
        ("burgers_ginihistory_ra_pikan_nu0.01_seed*.csv", "Burgers ν=0.01 RA", "#d62728", "-"),
        ("burgers_ginihistory_uniform_pikan_nu0.01_seed*.csv", "Burgers ν=0.01 Uni", "#9467bd", "--"),
        ("allen_cahn_ginihistory_ra_pikan_eps0.1_seed*.csv", "Allen-Cahn ε=0.1 RA", "#2ca02c", "-"),
        ("allen_cahn_ginihistory_uniform_pikan_eps0.1_seed*.csv", "Allen-Cahn ε=0.1 Uni", "#bcbd22", "--"),
        ("adv_diff_ginihistory_ra_pikan_Pe500.0_seed*.csv", "Adv-Diff Pe=500 RA", "#1f77b4", "-"),
        ("adv_diff_ginihistory_uniform_pikan_Pe500.0_seed*.csv", "Adv-Diff Pe=500 Uni", "#aec7e8", "--"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    any_data = False
    for pattern, label, color, ls in configs:
        files = list(RESULTS_DIR.glob(pattern))
        if not files:
            continue
        dfs = [pd.read_csv(f)[["cycle", "gini"]] for f in files]
        g = pd.concat(dfs).groupby("cycle")["gini"]
        mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
        x = mean.index.values - mean.index.min()   # 0-index cycles to match the text.
        ax.fill_between(x, (mean - std).values, (mean + std).values, color=color, alpha=0.15)
        ax.plot(x, mean.values, marker="o", markersize=5, label=label, color=color,
                linestyle=ls, lw=1.6)
        any_data = True

    if not any_data:
        print("  No Gini history data found; skipping Gini evolution plot.")
        plt.close(fig)
        return

    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Refinement Cycle")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Residual Concentration (Gini) vs Refinement Cycle")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    out = RESULTS_DIR / output_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {output_name}")


def plot_kan_activations(output_name: str = "fig_kan_activations_duffing.pdf") -> None:
    """Plot learned activation functions for ChebyPIKAN layers 0 and 2 (Duffing)."""
    if not HAS_MPL:
        return

    methods_to_plot = [
        ("duffing_fixed_pikan_seed42_activations.npz", "Fixed PIKAN"),
        ("duffing_ra_pikan_seed42_activations.npz", "RA-PIKAN"),
    ]

    found = [(p, lbl) for p, lbl in methods_to_plot if (RESULTS_DIR / p).exists()]
    if not found:
        print("  No KAN activation data found; skipping activation plot.")
        return

    n_cols = len(found)
    fig, axes = plt.subplots(2, n_cols, figsize=(min(5 * n_cols, 7.5), 7))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, (fname, label) in enumerate(found):
        data = np.load(RESULTS_DIR / fname)
        keys = list(data.keys())

        for row, layer_idx in enumerate([0, 2]):
            ax = axes[row, col]
            layer_keys = [k for k in keys if k.startswith(f"L{layer_idx}_j") and "_i" in k]
            xgrid_key = f"L{layer_idx}_xgrid"
            if xgrid_key not in data or not layer_keys:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            x = data[xgrid_key]
            for k in layer_keys:
                ax.plot(x, data[k], alpha=0.6, lw=1.0)
            ax.set_xlabel("t" if layer_idx == 0 else "hidden unit value", fontsize=8)
            ax.set_ylabel(r"$\varphi_{j,i}(x)$", fontsize=9)
            ax.set_title(f"{label} — Layer {layer_idx}", fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("KAN Activation Functions (Duffing, seed 42)", fontsize=10)
    fig.tight_layout()
    out = RESULTS_DIR / output_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {output_name}")


def plot_sensitivity(output_name: str = "fig_sensitivity_percentile.pdf") -> None:
    """Plot RA-PIKAN L2 error vs residual percentile threshold (Adv-Diff)."""
    if not HAS_MPL:
        return
    sens_dir = RESULTS_DIR.parent / "results" / "sensitivity"
    if not sens_dir.exists():
        sens_dir = RESULTS_DIR / "sensitivity"
    if not sens_dir.exists():
        print("  No sensitivity results directory found; skipping sensitivity plot.")
        return

    import glob as _glob
    files = list(sens_dir.glob("sensitivity_ra_pikan_*.csv"))
    if not files:
        print("  No sensitivity data found; skipping sensitivity plot.")
        return

    rows = []
    for f in files:
        try:
            import csv as _csv
            with open(f) as fh:
                reader = _csv.DictReader(fh)
                for row in reader:
                    rows.append({
                        "Pe": float(row.get("Pe", 0)),
                        "seed": int(row.get("seed", 0)),
                        "percentile": float(row.get("percentile", 90)),
                        "final_l2": float(row.get("final_l2", float("nan"))),
                    })
        except Exception:
            pass
    if not rows:
        return

    df = pd.DataFrame(rows)
    pes = sorted(df["Pe"].unique())
    fig, axes = plt.subplots(1, len(pes), figsize=(min(5 * len(pes), 7.5), 4), sharey=False)
    if len(pes) == 1:
        axes = [axes]

    DIVERGE_THRESH = 50.0  # L2 above this is treated as diverged.

    for ax, pe in zip(axes, pes):
        sub = df[df["Pe"] == pe].groupby("percentile").agg(
            mean=("final_l2", "mean"), std=("final_l2", "std")
        ).reset_index()

        stable = sub[sub["mean"] <= DIVERGE_THRESH]
        unstable = sub[sub["mean"] > DIVERGE_THRESH]

        ax.set_yscale("log")
        ax.errorbar(
            stable["percentile"], stable["mean"],
            yerr=stable["std"].clip(lower=0),
            marker="o", capsize=4, color="#d62728", lw=1.5, label="stable",
        )
        for _, row in unstable.iterrows():
            ax.annotate(
                f"diverged\n(mean={row['mean']:.0f})",
                xy=(row["percentile"], stable["mean"].max() * 3),
                ha="center", va="bottom", fontsize=7, color="#d62728",
                arrowprops=None,
            )
            ax.plot(row["percentile"], stable["mean"].max() * 3,
                    marker="x", ms=10, color="#d62728", lw=2, zorder=5)

        ax.set_xlabel("Percentile threshold")
        ax.set_ylabel("Mean Relative L² Error (log scale)")
        ax.set_title(f"Adv-Diff Pe={int(pe)}")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("RA-PIKAN Sensitivity to Residual Percentile Threshold", fontsize=10)
    fig.tight_layout()
    out = RESULTS_DIR / output_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {output_name}")


def main():
    global RESULTS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    RESULTS_DIR = Path(args.results_dir)

    print("=== Building accuracy table ===")
    acc_table = build_accuracy_table()
    save_accuracy_table(acc_table)

    print("\n=== Building compute table ===")
    comp_table = build_compute_table()
    save_compute_table(comp_table)

    print("\n=== Convergence plots ===")
    for nu in ["0.01", "0.005", "0.001"]:
        plot_convergence("burgers", f"nu{nu}_seed42",
                         f"fig_convergence_burgers_nu{nu}.pdf")
        plot_l2_vs_cycle("burgers", f"nu{nu}",
                         f"fig_l2cycle_burgers_nu{nu}.pdf")

    for eps in ["0.1", "0.05"]:
        plot_convergence("allen_cahn", f"eps{eps}_seed42",
                         f"fig_convergence_ac_eps{eps}.pdf")
        plot_l2_vs_cycle("allen_cahn", f"eps{eps}",
                         f"fig_l2cycle_ac_eps{eps}.pdf")

    for Pe in ["100.0", "500.0"]:
        plot_convergence("adv_diff", f"Pe{Pe}_seed42",
                         f"fig_convergence_ad_Pe{Pe}.pdf")
        plot_l2_vs_cycle("adv_diff", f"Pe{Pe}",
                         f"fig_l2cycle_ad_Pe{Pe}.pdf")

    print("\n=== Bar charts ===")
    if not acc_table.empty:
        plot_final_l2_bar(acc_table, "Burgers",
                          [f"nu={v}" for v in ["0.01", "0.005", "0.001"]],
                          "fig_bar_burgers.pdf")
        plot_final_l2_bar(acc_table, "Allen-Cahn",
                          [f"eps={v}" for v in ["0.1", "0.05"]],
                          "fig_bar_allen_cahn.pdf")
        plot_final_l2_bar(acc_table, "Adv-Diff",
                          [f"Pe={v}" for v in ["100.0", "500.0"]],
                          "fig_bar_adv_diff.pdf")

    print("\n=== Symbolic extraction ===")
    plot_symbolic_summary()

    print("\n=== Solution field plots ===")
    for nu in ["0.01", "0.005", "0.001"]:
        plot_solution_fields_1d("burgers", "nu", nu, "x", f"fig_solution_burgers_nu{nu}.pdf")
    for eps in ["0.1", "0.05"]:
        plot_solution_fields_1d("allen_cahn", "eps", eps, "x", f"fig_solution_ac_eps{eps}.pdf")
    plot_solution_field_2d("100.0", "fig_solution_adv_diff_Pe100.pdf")
    plot_solution_field_2d("500.0", "fig_solution_adv_diff_Pe500.pdf")

    print("\n=== Residual maps ===")
    for nu in ["0.01", "0.001"]:
        for method in ["ra_pikan", "uniform_pikan"]:
            plot_residual_maps("burgers", "nu", nu, method,
                               f"fig_residuals_burgers_nu{nu}_{method}.pdf")
    for eps in ["0.1", "0.05"]:
        plot_residual_maps("allen_cahn", "eps", eps, "ra_pikan",
                           f"fig_residuals_ac_eps{eps}_ra_pikan.pdf")
    for Pe in ["100.0", "500.0"]:
        for method in ["ra_pikan", "uniform_pikan"]:
            plot_residual_maps("adv_diff", "Pe", Pe, method,
                               f"fig_residuals_adv_diff_Pe{Pe}_{method}.pdf")

    print("\n=== Gini evolution ===")
    plot_gini_evolution()

    print("\n=== KAN activation functions ===")
    plot_kan_activations()

    print("\n=== Sensitivity analysis ===")
    plot_sensitivity()

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
