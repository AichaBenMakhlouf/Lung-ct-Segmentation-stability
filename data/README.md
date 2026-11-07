# Data

The data are **not** included in this repository. They are the COVID-19 CT Lung and Infection
Segmentation Dataset, version 1.0 (Ma et al., 2020): https://doi.org/10.5281/zenodo.3757476
(Zenodo record CC BY 4.0; source scans and annotations CC BY-NC-SA — non-commercial research only).

1. Download `COVID-19-CT-Seg_20cases.zip` and `Lung_Mask.zip` from the Zenodo record.
2. Unzip them, keeping the original file names, so that you have

   ```
   data/raw/COVID-19-CT-Seg_20cases/coronacases_001.nii.gz ...
   data/raw/Lung_Mask/coronacases_001.nii.gz ...
   ```
3. From the repository root, run `python scripts/prepare_data.py`. It writes

   ```
   data/processed/2d_images/<case>_Z_<k>.tif    CT slice in HU (float32)
   data/processed/2d_masks/<case>_Z_<k>.tif     lung mask, 0/255
   data/processed/dataset_index.csv             one row per slice (patient, position, covariates)
   data/processed/lung_stats.csv                per-slice lung area and attenuation
   ```

Only the ten `coronacases_*` volumes are used (267 slices); the `radiopaedia_*` volumes are 8-bit
display images, not Hounsfield units. Both `raw/` and `processed/` are ignored by git.
