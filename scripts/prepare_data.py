"""Extract the 2D lung slices used in the study from the COVID-19-CT-Seg volumes.

Input  (Zenodo record 3757476):
    data/raw/COVID-19-CT-Seg_20cases/*.nii.gz   CT volumes
    data/raw/Lung_Mask/*.nii.gz                 left/right lung masks
Output:
    data/processed/2d_images/<case>_Z_<k>.tif   CT slice in Hounsfield units (float32)
    data/processed/2d_masks/<case>_Z_<k>.tif    lung mask, 0/255 (left and right merged)
    data/processed/lung_stats.csv               per-slice lung area and attenuation
    data/processed/dataset_index.csv            one row per slice (patient, position, covariates)

Only the ten `coronacases` volumes are used by default: they are stored in Hounsfield
units, whereas the `radiopaedia` volumes are 8-bit display images. 267 lung-containing
slices are kept, divided evenly over the patients and evenly spaced over the lung extent
of each volume (deterministic). Slices are rotated so that anterior is up; non-square
slices are padded to square with air.

Usage (from the repository root):
    python scripts/prepare_data.py
"""
from __future__ import annotations
import argparse
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
import tifffile


def _case_name(path: str) -> str:
    return os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")


def _pad_square(a: np.ndarray, fill) -> np.ndarray:
    h, w = a.shape
    s = max(h, w)
    if h == w:
        return a
    out = np.full((s, s), fill, a.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = a
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--raw", default=os.path.join(repo, "data", "raw"),
                    help="folder with COVID-19-CT-Seg_20cases/ and Lung_Mask/")
    ap.add_argument("--out", default=os.path.join(repo, "data", "processed"))
    ap.add_argument("--n", type=int, default=267,
                    help="total number of slices to keep (default 267); 0 = every lung slice")
    ap.add_argument("--source", default="coronacases", choices=["coronacases", "radiopaedia", "all"],
                    help="coronacases: the 10 volumes in Hounsfield units (default)")
    args = ap.parse_args()

    raw, out = os.path.abspath(args.raw), os.path.abspath(args.out)
    ct_dir, mk_dir = os.path.join(raw, "COVID-19-CT-Seg_20cases"), os.path.join(raw, "Lung_Mask")
    out_img, out_msk = os.path.join(out, "2d_images"), os.path.join(out, "2d_masks")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)

    cts = {_case_name(p): p for p in glob.glob(os.path.join(ct_dir, "**", "*.nii*"), recursive=True)}
    mks = {_case_name(p): p for p in glob.glob(os.path.join(mk_dir, "**", "*.nii*"), recursive=True)}
    cases = sorted(set(cts) & set(mks))
    if args.source != "all":
        cases = [c for c in cases if c.startswith(args.source)]
    missing = sorted(set(cts) ^ set(mks))
    if missing:
        print("WARNING: CT/mask mismatch:", missing)
    print(f"{len(cases)} cases")

    # Equal allocation per patient (balanced leave-one-patient-out folds); the
    # remainder goes to the first cases. Within a patient, slices are evenly
    # spaced over the lung extent -- deterministic, no seed.
    per_case = {c: 0 for c in cases}
    if args.n:
        base, extra = divmod(args.n, len(cases))
        per_case = {c: base + (i < extra) for i, c in enumerate(cases)}

    rows = []
    for pid, case in enumerate(cases):
        ci, mi = nib.load(cts[case]), nib.load(mks[case])
        ct = np.asarray(ci.dataobj, dtype=np.float32)
        mk = np.asarray(mi.dataobj) > 0
        assert ct.shape == mk.shape, f"{case}: CT {ct.shape} vs mask {mk.shape}"
        sx, sy, sz = (float(v) for v in ci.header.get_zooms()[:3])
        lung_k = np.where(mk.reshape(-1, mk.shape[2]).any(0))[0]
        k0, k1 = lung_k.min(), lung_k.max()
        air = float(ct.min())
        n_lung = len(lung_k)
        if per_case[case]:
            lung_k = lung_k[np.unique(np.round(np.linspace(0, n_lung - 1, per_case[case])).astype(int))]
        for k in lung_k:
            img = _pad_square(np.rot90(ct[:, :, k]), air)
            msk = _pad_square(np.rot90(mk[:, :, k]), False)
            name = f"{case}_Z_{k:04d}.tif"
            tifffile.imwrite(os.path.join(out_img, name), img.astype(np.float32))
            tifffile.imwrite(os.path.join(out_msk, name), (msk * 255).astype(np.uint8))
            lung = img[msk]
            area = int(msk.sum())
            rows.append(dict(
                img_id=name,
                image_path=os.path.join(out_img, name),
                mask_path=os.path.join(out_msk, name),
                z=int(k),
                z_norm=float((k - k0) / max(1, k1 - k0)),
                patient_id=case,
                patient_idx=pid,
                source="coronacases" if case.startswith("coronacases") else "radiopaedia",
                pixel_mm=sx, slice_mm=sz,
                lung_area_px=area,
                lung_area_mm2=area * sx * sy,
                lung_volume_fraction=area / msk.size,
                lung_mean_hu=float(lung.mean()),
                lung_pd95_hu=float(np.percentile(lung, 95)),
                lung_pd05_hu=float(np.percentile(lung, 5)),
                img_min=float(ct[:, :, k].min()), img_max=float(ct[:, :, k].max()),
            ))
        print(f"  {case:32s} kept {len(lung_k):3d} of {n_lung:4d} lung slices  "
              f"range [{ct.min():.0f}, {ct.max():.0f}]")

    df = pd.DataFrame(rows)
    stat_cols = ["img_id", "lung_area_px", "lung_area_mm2", "lung_volume_fraction",
                 "lung_mean_hu", "lung_pd95_hu", "lung_pd05_hu"]
    df[stat_cols].to_csv(os.path.join(out, "lung_stats.csv"), index=False)
    df.to_csv(os.path.join(out, "dataset_index.csv"), index=False)
    print(f"\nwrote {len(df)} slice/mask pairs from {df.patient_id.nunique()} patients")
    print(df.groupby("source").agg(n=("img_id", "size"), min_val=("img_min", "min"),
                                   max_val=("img_max", "max")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
