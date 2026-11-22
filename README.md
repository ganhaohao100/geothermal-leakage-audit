# Code and derived results

Reproduction package for the manuscript on target-type-dependent label leakage in
machine-learning geothermal favorability mapping (mainland China, 0.05 deg grid).

Release 2.4 (2026-11-22) renumbers the supplement so that Tables S1-S6 and Figs. S1-S2 follow the order in which the main text cites them, and adds a synthetic quick test under examples/. It also adds the split-sensitivity and robustness scripts/results and synchronizes the archived results with the final manuscript (10/22/50 components at P99/P98/P95; 40,603 km2 at P99).

Archived release: v2.4, DOI: [10.5281/zenodo.24441638](https://doi.org/10.5281/zenodo.24441638).

This package contains the analysis code and the aggregated results behind the
reported tables and figures. It does not contain the spring and well inventories
used as labels, nor any raster derived from them.
No national boundary, coastline, point-level record or per-cell raster is
included; the candidate-area table is stored at 0.05 deg grid precision.

## Contents

```
code/            22 analysis scripts (one per experiment)
results/         archived JSON/CSV outputs, as reported in the manuscript
data/            input schema and the baseline CV report; restricted inputs are not included
nine_region.py   nine-region tectonic-regime index (Ts/Td input)
common.py        grid, kernel smoothing, fault-parameter utilities
config.py        path configuration
docs/            data sources and release notes
examples/        synthetic quick test (no restricted data)
```

## Quick test

A synthetic smoke test verifies the environment without the restricted bundle:

```bash
python examples/quick_test.py
```

It trains a small random forest on generated data across five stratified folds,
prints the mean held-out AUC, and ends with `quick test passed`.

## Inputs
All scripts read a single raster bundle, `features.npz`. Most write JSON to
`output/`; `stepT` and `stepU` also write their tables to `results/`. Point `GEOTHERMAL_DATA` at the directory holding that bundle:

```bash
export GEOTHERMAL_DATA=/path/to/data
python code/stepC_leak_decomposition.py
```

On Windows PowerShell:

```powershell
$env:GEOTHERMAL_DATA = "D:\path\to\data"
python code\stepC_leak_decomposition.py
```

`features.npz` is **not** redistributed because it contains the coordinates of the
spring and well records. Its schema is documented in `data/README.md`, so a reader
holding equivalent label data can rebuild it.

Two further inputs are expected in `GEOTHERMAL_DATA`:

- `cv_report.json` — baseline cross-validation report (included here, `data/cv_report.json`)
- `fig1_cache.pkl` — cached boundary and fault segments used by
  `load_cache()`; obtainable from the corresponding author

Scripts that call `load_cache()` (`stepG`, `stepG2`, `stepH`, `stepQ`, `stepQ4`,
`stepQ5`, `stepQ6`, `stepR`) need `fig1_cache.pkl`; the remaining scripts run
from `features.npz` alone.

## Scripts and reported results

| Script | Produces | Reported in |
|---|---|---|
| `stepC_leak_decomposition.py` | `leak_decomposition.json` | Table 2, Fig. 3 |
| `stepC2_lfo_buffer_sweep.py` | `lfo_buffer_sweep.json` | Sec. 3.3, buffer sweep |
| `stepC2b_lfo_field_diag.py` | `lfo_field_diag.json` | Table 2, per-field LFO diagnostic |
| `stepF_fault_increment.py` | `fault_increment.json` | Table 3, fault-family ablation |
| `stepF2_hf_grad_ablation.py` | `hf_grad_ablation.json` | Table 3, without `hf_grad` |
| `stepF3_stress_ablation.py` | `stress_ablation.json` | Table 3, without stress family |
| `stepG_hf_well_colocation.py` | `hf_well_colocation.json` | Sec. 4.2, heat-flow/well co-location |
| `stepG2_hf_well_exclusion.py` | `hf_well_exclusion.json` | Sec. 4.2, 10 km well exclusion |
| `stepH_hf_analogue.py` | `hf_analogue.json` | Sec. 4.4, Table S6, Fig. S2 |
| `stepP_perm_metric.py` | `perm_metric_check.json` | Sec. 3.2, permutation-metric check |
| `stepQ_p2_diagnostics.py` | `p2_diagnostics.json` | Sec. 2.2, 2.4, 3.2, 3.3 |
| `stepQ2_fields.py` | merges into `p2_diagnostics.json` | Sec. 2.2, derived-field correlations |
| `stepQ3_buffer_sweep.py` | merges into `p2_diagnostics.json` | Sec. 3.3, buffer sweep |
| `stepQ4_groups_multiseed.py` | `hf_analogue_groups.json`, `multiseed.json` | Sec. 3.2, family permutation and multi-seed check; Sec. 4.4, analogue family importance |
| `stepQ5_geo_ci.py` | merges into `p2_diagnostics.json` | Table 2, bootstrap confidence intervals |
| `stepQ6_perm_geo10.py` | merges into `p2_diagnostics.json` | Table 3, Fig. 6c |
| `stepR_region.py` | `region_sea.json` | Sec. 3.3, Fig. 5, South China replication |
| `stepS_sigma_sweep.py` | `sigma_sweep.json` | Sec. 3.3, Fig. 4, smoothing-scale sweep |
| `stepT_group_transfer.py` | `group_transfer.json` | Table 1, Sec. 3.1, play-type transfer |
| `stepT2_split_sensitivity.py` | `split_sensitivity.json` | Sec. 3.1, 38 and 75 km split sensitivity |
| `stepU_well_surface_targets.py` | `targets_well_geo.csv`, `well_surface_summary.json` | Fig. 6, Table S5, Sec. 3.4 |
| `stepV_robustness.py` | `robustness.json` | Tables S3 and S4, pseudo-negative resampling, algorithm and leaf-capacity checks |
| `stepW_hf_coverage.py` (archived) | `stepW_hf_coverage.json` | Table 3, heat-flow support flag and distance to the nearest site; requires the heat-flow site coordinates |

`results/targets_well_geo.csv` summarises the candidate areas of Sec. 3.4 and Fig. 6
by rank, tier, area and favourability. `lonc` and `latc` locate the grid cell of
maximum favourability in each component, not its centroid; Table S5 reports the
area-weighted centroid. Polygon vertices are not included. `stepU` reproduces the 10/22/50 counts and the 40,603 km2 total of the
99th, 98th and 95th percentiles.
Within each percentile tier, `rank` orders components by decreasing `fav_max`.
Table S5 lists the same ten P99 components in geographical order under its own
IDs, so the two are matched by area and cell count, not by rank.

`stepT` implements the Table 1 design. The script reports the five-fold mean, the
sample standard deviation and the 2.5th-97.5th percentile interval of 1,000
spatial-block bootstrap resamples; `manuscript_table1` in `group_transfer.json`
lists the same values in the row order of Table 1.

## Reproducibility notes

- Random seed 20260927. LightGBM runs with `n_jobs=-1`; last digits may differ
  between machines.
- Configurations: 400 trees, learning rate 0.03, 31 leaves, 30 samples per leaf,
  subsample 0.8, L2 regularisation 1.0.
- Spatial blocks are 50 km x 50 km with a 10 km validation buffer, five folds,
  pseudo-negatives at 1:3 without replacement.
- Reported software versions: scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0.

## License

MIT. See `LICENSE`.
