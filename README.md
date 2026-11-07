# Training stability and architecture comparison in deep learning-based lung CT segmentation

Code for the article *"Training stability and architecture comparison in deep learning-based lung CT
segmentation"* (submitted to *PeerJ
Computer Science*).

The code trains 2D lung segmentation networks on chest CT slices from the public COVID-19-CT-Seg
dataset and measures how stable and reproducible the results are across repeated runs with different
random seeds.

| Experiment | Article | Runs |
|---|---|---|
| Baseline: ResUNet++ with a literal batch of 10 and with gradient accumulation | 4.2–4.3, 5 | 8 and 20 seeds |
| Progressive protocol: best epoch + threshold tuning, lung window, augmentation + cosine LR | 4.4, 5.1, Table 4 | 4 arms × 12 seeds |
| Architecture benchmark: U-Net, Attention U-Net, ResU-Net, ResUNet++ | 4.5, 5.2, Table 5 | 4 × 5 seeds |
| Residual-error regression on lung area, attenuation and slice position | 5.3, Table 6 | — |
| Split sensitivity: random vs Z-blocked vs leave-one-patient-out | 4.6, 5.4 | 6 seeds, 6 × 4 folds, 10 folds |

## Repository structure

```
lung-ct-segmentation-stability/
├── src/lungct/                  Python package
│   ├── data.py                  slice loading, intensity mapping, augmentation, data splits
│   ├── models.py                U-Net, Attention U-Net, ResU-Net, ResUNet++
│   ├── metrics.py               Dice, IoU, accuracy, sensitivity, specificity, calibration
│   ├── training.py              training and evaluation of one run (Algorithm 1)
│   ├── experiments.py           the experiments of the article
│   └── paths.py                 repository folders
├── scripts/                     command-line entry points (run in this order)
│   ├── prepare_data.py          1  CT volumes -> 267 lung slices and per-slice covariates
│   ├── check_near_duplicates.py 2  near-duplicate check (NCC > 0.98)
│   ├── run_experiments.py       3  baseline, progressive protocol, benchmark, split sensitivity
│   ├── train_literal_batch10.py 4  literal-batch-10 baseline (.ps1 launcher for Windows)
│   └── make_qualitative.py      5  predictions for Figure 12
├── notebooks/analysis.ipynb     6  statistics, Tables 4–6 and Figures 4–13
├── tests/                       unit tests (pytest)
├── data/                        dataset (not included, see data/README.md)
├── results/                     runs, tables and figures (see results/README.md)
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

## Installation

Python 3.10 or later. Training needs an NVIDIA GPU (the article used a GeForce GTX 1650 with 4 GB);
the analysis notebook runs on a CPU.

```bash
git clone https://github.com/<user>/lung-ct-segmentation-stability.git
cd lung-ct-segmentation-stability
python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pytest                                        # optional: run the unit tests
```

## Data

The study uses the **COVID-19 CT Lung and Infection Segmentation Dataset**, version 1.0
([Zenodo 10.5281/zenodo.3757476](https://doi.org/10.5281/zenodo.3757476)). The data are not included
in this repository. Download `COVID-19-CT-Seg_20cases.zip` and `Lung_Mask.zip` and unzip them into
`data/raw/` (details in [data/README.md](data/README.md)).

Only the ten `coronacases` volumes are used, because they are stored in Hounsfield units. From them,
267 lung-containing axial slices are extracted (26–27 per volume, evenly spaced over the lung) and the
left and right lung masks are merged.

## Usage

Run all commands from the repository root.

```bash
# 1. prepare the slices
python scripts/prepare_data.py

# 2. near-duplicate check
python scripts/check_near_duplicates.py

# 3. gradient-accumulation experiments (about two days on a GTX 1650; resumable)
python scripts/run_experiments.py
python scripts/run_experiments.py --only benchmark          # or a single experiment

# 4. literal-batch-10 baseline, 8 seeds (about 3 hours)
python scripts/train_literal_batch10.py                     # Windows: scripts\train_literal_batch10.ps1

# 5. predictions for the qualitative figure
python scripts/make_qualitative.py

# 6. statistics, tables and figures
cd notebooks
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

A quick check that training works (one seed, two epochs):
`python scripts/train_literal_batch10.py --seeds 1 --epochs 2 --out quick_test.csv`.

On a GPU with 4 GB of memory, the literal batch of 10 needs
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.8`; the Windows
launcher sets it.

## Method

| Step | Setting |
|---|---|
| Intensity mapping | raw HU / 255 (baseline, arms A–B) or lung window: clip to [−1350, +150] HU and rescale to [0, 1] |
| Resizing | 512 × 512 → 256 × 256, bilinear, anti-aliased; masks binarised at 0.5 |
| Splits | per seed: 80/20 train/test, then 20% of the training part as validation |
| Validation split | learning-rate schedule, best-epoch selection, early stopping, threshold tuning (0.05–0.95) |
| Test split | used once, for the reported held-out metrics |
| Optimisation | Adam, learning rate 10⁻³; ReduceLROnPlateau on the validation loss (baseline, arms A–C) or cosine schedule |
| Batch | literal batch of 10, or gradient accumulation with micro-batches of at most 4 (10 = 4 + 4 + 2; 8 = 4 + 4) |
| Loss | binary cross-entropy; BCE + soft Dice for the benchmark and the split analysis |
| Epochs | 20 (baseline, arms A–C), up to 40 (arm D), up to 60 (benchmark, split analysis); early stopping patience 15 |
| Augmentation | horizontal and vertical flips, 90° rotations, gamma 0.8–1.25 (arm D, benchmark, split analysis) |
| Split sensitivity | Z-blocked: four blocks of within-volume slice position, each held out once per seed; leave-one-patient-out: ten folds |
| Statistics | paired bootstrap CIs (10,000 resamples), Holm correction, Levene test, Friedman with Nemenyi post-hoc, TOST (±0.02 on the 90% CI), practical margin 0.03 for split sensitivity, Wilson CIs, fractional-response logit GLM with cluster-robust standard errors |

Seeds: 0–7 (literal batch 10), 0–19 (gradient-accumulation baseline), 0–11 (arms A–D), 0–4
(benchmark), 0–5 (random and Z-blocked splits). Training is deterministic: the same seed on the same
GPU and library versions gives the same result.

## Results

The training scripts write one CSV row per run to `results/runs/`. The notebook writes the full report
(`results/tables/analysis_report.txt`), the tables (`results/tables/table_*.csv`) and the article
figures (`results/figures/fig4_*` to `fig13_*`, PNG and PDF). See
[results/README.md](results/README.md).

## Citation

Please cite the article and the dataset if you use this code (see [CITATION.cff](CITATION.cff)):

> Training stability and architecture comparison in deep learning-based lung CT segmentation. Submitted to *PeerJ Computer Science*.

> Ma J. et al. (2020). COVID-19 CT Lung and Infection Segmentation Dataset (Version 1.0). Zenodo.
> https://doi.org/10.5281/zenodo.3757476

> Ma J. et al. (2021). Toward data-efficient learning: A benchmark for COVID-19 CT lung and infection
> segmentation. *Medical Physics* 48(3):1197–1210. https://doi.org/10.1002/mp.14676

Architectures: Ronneberger et al. (2015), U-Net; Oktay et al. (2018), Attention U-Net; Zhang et al.
(2018), ResU-Net; Jha et al. (2019), ResUNet++.

## License

The code is released under the [MIT License](LICENSE). The dataset is not covered by this licence:
the Zenodo record is CC BY 4.0, and the source scans and annotations are CC BY-NC-SA (non-commercial
research use, with attribution).

## Contact

Aicha Ben Makhlouf — a.benmakhlouf@qu.edu.sa. Questions and contributions are welcome through GitHub
issues and pull requests.
