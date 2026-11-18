# Training Stability and Architecture Comparison in Deep Learning-Based Lung CT Segmentation

## Description

This repository contains the code used in the article *"Training stability and architecture
comparison in deep learning-based lung CT segmentation"* (Ben Makhlouf, Elloumi and Louhichi,
submitted to PeerJ Computer Science).

Repository: https://github.com/AichaBenMakhlouf/Lung-ct-Segmentation-stability

The code trains and evaluates 2-D lung segmentation networks on chest CT slices and measures how
stable and reproducible the results are across repeated runs with different random seeds. It
reproduces all experiments in the paper:

1. **Baseline instability:** ResUNet++ with a literal batch size of 10 (8 seeds) versus gradient
   accumulation with micro-batches of 4, three per optimizer step (12 slices per step; 20 seeds).
2. **Progressive protocol (arms A–D, 12 matched seeds):** best-epoch selection and threshold tuning,
   lung-window HU scaling, then augmentation, cosine LR, longer training and early stopping.
3. **Architecture benchmark (5 seeds each):** U-Net, Attention U-Net, ResU-Net and ResUNet++ under
   the stabilized protocol, with Friedman/Nemenyi tests and pairwise TOST equivalence (margin ±0.02).
4. **Residual-error analysis:** fractional-response GLM of per-image Dice on lung area fraction,
   mean lung attenuation and slice position.
5. **Split sensitivity:** random-slice vs Z-blocked (K = 4) vs leave-one-patient-out splits for
   U-Net and ResUNet++.
6. **Figures and tables** of the paper (Figures 4–13 and Tables 4–6). Figures 1–3 are concept
   diagrams and are not produced by this code.

## Dataset information

- **Name:** COVID-19 CT Lung and Infection Segmentation Dataset (COVID-19-CT-Seg), version 1.0
- **DOI:** https://doi.org/10.5281/zenodo.3757476
- **URL:** https://zenodo.org/records/3757476
- **Reference publication:** Ma J. et al. (2021). Toward data-efficient learning: A benchmark for
  COVID-19 CT lung and infection segmentation. *Medical Physics* 48(3):1197–1210.
  https://doi.org/10.1002/mp.14676
- **Content:** 20 chest CT volumes (NIfTI) of COVID-19 patients with left-lung, right-lung and
  infection masks.
- **Subset used:** only the ten `coronacases_*` volumes, which are stored in Hounsfield units
  (512 × 512 slices). From them, 267 axial slices containing lung were extracted (26–27 per volume,
  evenly spaced over the lung extent; the selection is deterministic). Left and right lung labels are
  merged into one binary mask; infection masks are not used.
- **Licence:** the Zenodo record is CC BY 4.0; the dataset authors state that the source scans and
  annotations are CC BY-NC-SA. Use for non-commercial research only, with attribution.
- **The dataset is not redistributed in this repository.** Download it from Zenodo (see Usage).

## Code information

| Path | Purpose |
|---|---|
| `05_code/01_data_loading/convert_covid_ct_seg.py` | Reads the NIfTI volumes, selects the lung slices, merges lung labels, computes per-slice covariates, saves the slice set |
| `05_code/02_data_cleaning/check_near_duplicates.py` | Near-duplicate check (NCC > 0.98) |
| `05_code/colab/lug_method.py` | The four architectures (7.77 M, 7.85 M, 8.12 M and 4.09 M parameters); preprocessing and augmentation; training with gradient accumulation, best-epoch selection, early stopping, LR schedules and threshold tuning; Dice, IoU, accuracy and ECE; data splits; experiment functions |
| `05_code/06_main_analysis/run_local_chain_covid.py` | Runs the accumulation baseline, arms A–D, the architecture benchmark and the split-sensitivity experiments |
| `05_code/06_main_analysis/train_faithful.py`, `model_resunetpp.py`, `data_faithful.py` | Literal-batch-10 baseline |
| `05_code/06_main_analysis/run_truebatch10_covid.ps1` | Windows launcher for the literal-batch-10 runs |
| `05_code/11_final_results/make_qualitative_covid.py` | Trains one U-Net and saves predictions for Figure 12 |
| `05_code/11_final_results/phase7_analysis_covid.ipynb` | Bootstrap CIs, Wilcoxon, Levene, Friedman/Nemenyi, TOST, fractional-response GLM; all tables and figures |
| `05_code/common.py`, `05_code/config_covid.yaml` | Shared helpers and configuration |
| `00_inputs/covid19_ct_seg/results/`, `07_analysis_outputs/covid19_ct_seg/` | Per-run results of the study, saved predictions, and the generated tables and figures |
| `requirements.txt` | Python dependencies with versions |

## Requirements

- Python 3.12.5
- PyTorch 2.7.1 with CUDA 11.8
- NumPy, SciPy, pandas, scikit-learn, scikit-image, statsmodels, NiBabel, tifffile, PyYAML, Matplotlib,
  Jupyter — exact versions in `requirements.txt`
- Hardware used in the paper: NVIDIA GeForce GTX 1650 (4 GB), Intel Core i5-10300H CPU, 8 GB RAM,
  Windows 10 Pro 64-bit. Training requires an NVIDIA GPU, because the training scripts run on CUDA.
  Reproducing the statistics and figures from the included results (step 4 below) needs no GPU.

Install, from the folder that contains this README:

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Usage instructions

Run the commands from the folder that contains this README.

1. **Download the data** from https://doi.org/10.5281/zenodo.3757476
   (`COVID-19-CT-Seg_20cases.zip` and `Lung_Mask.zip`) and unzip them, keeping the original file
   names, into:
   ```
   00_inputs/covid19_ct_seg/COVID-19-CT-Seg_20cases/
   00_inputs/covid19_ct_seg/Lung_Mask/
   ```
2. **Prepare the slices:**
   `python 05_code/01_data_loading/convert_covid_ct_seg.py --root 00_inputs/covid19_ct_seg`
3. **Near-duplicate check:**
   `python 05_code/02_data_cleaning/check_near_duplicates.py`
4. **Statistics, tables and figures** from the included per-run results (no training needed):
   ```bash
   cd 05_code/11_final_results
   jupyter nbconvert --to notebook --execute --inplace phase7_analysis_covid.ipynb
   ```
   Tables go to `07_analysis_outputs/covid19_ct_seg/tables/` (`phase7_covid_report.txt` is the full
   report) and figures to `07_analysis_outputs/covid19_ct_seg/figures_paper/`, named by figure number.
   Steps 2 and 3 are needed first (the notebook reads the slice index and the CT slices). The saved
   U-Net predictions for Figure 12 (`07_analysis_outputs/covid19_ct_seg/results/qualitative/corrected_unet.npz`)
   contain CT data and are not included; the notebook then skips the Figure 12 cell and keeps the
   included Figure 12. To redraw it, run `05_code/11_final_results/make_qualitative_covid.py` first
   (one U-Net training run, a few minutes on a GPU).
5. **Optional: repeat the training.** First delete `00_inputs/covid19_ct_seg/results/`, because
   finished runs are skipped; running the same command again resumes an interrupted run.
   ```bash
   python 05_code/06_main_analysis/run_local_chain_covid.py   # accumulation, arms A–D, benchmark, splits
   python 05_code/11_final_results/make_qualitative_covid.py  # predictions for Figure 12
   ```
   Literal batch 10 (8 seeds): on Windows run `05_code/06_main_analysis/run_truebatch10_covid.ps1`.
   On Linux, from `05_code/06_main_analysis`:
   ```bash
   LUG_CONFIG=config_covid.yaml PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.8 \
   python train_faithful.py --runs 8 --out faithful_truebatch10_all.csv
   ```
   The allocator setting was needed to fit a batch of 10 in 4 GB of GPU memory.

Approximate run time on a GTX 1650: 417 s (U-Net) to 813 s (ResUNet++) per benchmark run; about one
day for `run_local_chain_covid.py` and about 3 hours for the literal-batch-10 runs. Step 4 takes a
few minutes.

Random seeds used: 0–7 (literal batch 10), 0–19 (accumulation baseline), 0–11 (arms A–D), 0–4
(architecture benchmark), 0–5 (random and Z-blocked splits). Leave-one-patient-out uses one fold per
volume (10 folds). Deterministic mode was not enabled (`torch.backends.cudnn.benchmark = True` in
`lug_method.py`), so retrained results can vary slightly across runs, GPUs and library versions.

## Methodology

1. Keep the 10 HU volumes; extract 267 lung-containing axial slices; merge left/right lung masks.
2. Quality control: readability, size and mask checks; near-duplicate check (NCC > 0.98 on
   lung-windowed slices).
3. Resize to 256 × 256 (bilinear, anti-aliased for images).
4. Intensity mapping: HU / 255 (baseline, arms A–B) or lung window (centre −600 HU, width
   1500 HU), clipped and rescaled to [0, 1].
5. Split per seed: random 80/20 train/test at slice level. From arm B onwards, 20% of the training
   set is kept as an inner validation split. The baseline and arm A have no inner split, so the
   held-out slices also drive the `ReduceLROnPlateau` schedule in those runs.
6. Train: Adam, initial LR 1e-3, BCE loss (BCE + soft Dice for the benchmark and split analysis),
   ReduceLROnPlateau (baseline, arms A–C) or cosine schedule (arm D, benchmark, split analysis).
   Epochs: 20 (baseline, arms A–C), up to 40 (arm D, split analysis), up to 60 (benchmark), with early
   stopping (patience 15) where used. Augmentation: flips, 90° rotations and mild gamma (arm D,
   benchmark, split analysis).
   Batch handling: the literal baseline uses a batch of 10. In the accumulation engine, micro-batches
   of 4 are used: a configured batch of 10 (accumulation baseline, arms A–D) accumulates 3 micro-batches,
   i.e. 12 samples per optimizer step, and a configured batch of 8 (benchmark, split analysis)
   accumulates 2 micro-batches, i.e. 8 samples per step.
7. From arm B onwards, select the best epoch on validation soft Dice and tune the decision threshold
   in 0.05–0.95 on the validation split; evaluate once on the test split. The baseline and arm A use
   the final epoch and a fixed threshold of 0.5.
8. Statistics: paired bootstrap CIs (10,000 resamples), Holm correction, Levene test, Friedman with
   Nemenyi post-hoc, TOST (margin 0.02 on the 90% CI; practical margin 0.03 for split sensitivity),
   Wilson CIs for collapse rates, fractional-response logit GLM with cluster-robust standard errors.

## Citations

If you use this code, please cite the article (Ben Makhlouf A., Elloumi N., Louhichi B. Training
stability and architecture comparison in deep learning-based lung CT segmentation, submitted to
PeerJ Computer Science) and the dataset:

- Ma J. et al. (2020). COVID-19 CT Lung and Infection Segmentation Dataset (Version 1.0) [Data set].
  Zenodo. https://doi.org/10.5281/zenodo.3757476
- Ma J. et al. (2021). Toward data-efficient learning: A benchmark for COVID-19 CT lung and
  infection segmentation. *Medical Physics* 48(3):1197–1210. https://doi.org/10.1002/mp.14676

Architectures: Ronneberger et al. 2015 (U-Net); Oktay et al. 2018 (Attention U-Net); Zhang et al.
2018 (ResU-Net); Jha et al. 2019 (ResUNet++).

## License and contribution guidelines

- Code licence: MIT (see `LICENSE`).
- The dataset is not covered by this licence; follow its own terms (CC BY 4.0 record; CC BY-NC-SA
  source data).
- Questions and contributions: contact Aicha Ben Makhlouf, a.benmakhlouf@qu.edu.sa.
