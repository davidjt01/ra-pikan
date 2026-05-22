# Experiments

Scripts to run the RA-PIKAN and baseline experiments and generate figures/tables.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_burgers.py` | 1D Burgers for all 6 methods; writes metrics to `results/` |
| `run_allen_cahn.py` | 1D Allen-Cahn for all 6 methods |
| `run_advection_diffusion.py` | 2D steady advection-diffusion for all 6 methods |
| `run_duffing.py` | Duffing benchmark and symbolic extraction sub-study |
| `run_sensitivity.py` | RA-PIKAN sensitivity to residual percentile threshold {70,80,90,95}; writes to `results/sensitivity/` |
| `run_all.py` | Entry point: full suite (all benchmarks × 6 methods × seeds); calls `run_sensitivity.py` in full mode |
| `make_figures_and_tables.py` | Reads `results/*.csv` and produces all paper figures and tables |

## Quick start

```bash
# Smoke test (one seed, one difficulty level)
python experiments/run_all.py --quick --device cpu

# Generate figures and tables (after running experiments)
python experiments/make_figures_and_tables.py
```

## Full reproduction

```bash
python experiments/run_all.py --device cpu
python experiments/make_figures_and_tables.py
```

Already-present result files are skipped, so interrupted runs can be safely resumed.

## Individual benchmarks

```bash
python experiments/run_burgers.py --nu 0.01 --seeds 42 --device cpu
python experiments/run_allen_cahn.py --eps 0.1 --seeds 42 --device cpu
python experiments/run_advection_diffusion.py --Pe 100 --seeds 42 --device cpu
python experiments/run_duffing.py --seeds 42 --device cpu
```

## Reproducibility

All scripts fix `torch.manual_seed` and `numpy.random.default_rng`; hyperparameters
are defined at the top of each script. Results use the naming convention
`{benchmark}_{method}_{param}_seed{seed}.csv`.
