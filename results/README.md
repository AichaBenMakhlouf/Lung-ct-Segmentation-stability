# Results

| Folder | Content | Written by |
|---|---|---|
| `runs/` | one CSV row per training run, training histories, per-image Dice | training scripts |
| `tables/` | `analysis_report.txt` (full statistical report) and `table_*.csv` | `notebooks/analysis.ipynb` |
| `figures/` | article figures `fig4_*` to `fig13_*` (PNG and PDF); `supplementary/` additional figures | `notebooks/analysis.ipynb` |

## Files in `runs/`

| File | Experiment |
|---|---|
| `baseline_accumulation.csv`, `baseline_accumulation_histories.json` | baseline with gradient accumulation (20 seeds) |
| `literal_batch10.csv`, `literal_batch10_history_seed<k>.csv` | baseline with a literal batch of 10 (8 seeds) |
| `progressive_protocol.csv` | arms A–D (12 seeds each) |
| `architecture_benchmark.csv`, `per_image_dice/<arch>_seed<k>.csv` | architecture benchmark (5 seeds each) |
| `split_sensitivity.csv` | random, Z-blocked and leave-one-patient-out splits |
| `qualitative/unet_seed0.npz` | predictions for Figure 12 (contains CT data, not tracked by git) |

Each row of a run CSV holds the full run configuration, the seed, the split sizes, the threshold used
and the held-out (test) metrics: Dice, IoU, accuracy, sensitivity, specificity, soft Dice, expected
calibration error and Brier score.
