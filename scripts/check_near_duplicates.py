"""Near-duplicate check (article Section 4.1).

Normalised cross-correlation (NCC) between all pairs of lung-windowed slices resized to
128 x 128. Pairs with NCC > 0.98 are reported and written to
results/tables/near_duplicates.csv.

Usage (from the repository root):
    python scripts/check_near_duplicates.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.transform import resize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lungct import paths  # noqa: E402
from lungct.data import LUNG_WINDOW, load_index  # noqa: E402

THRESHOLD = 0.98


def main() -> None:
    idx = load_index()
    lo, hi = LUNG_WINDOW
    x = np.stack([resize(np.clip(tifffile.imread(p).astype(np.float32), lo, hi), (128, 128),
                         preserve_range=True, anti_aliasing=True).ravel() for p in idx.image_path])
    x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    ncc = x @ x.T / x.shape[1]
    iu = np.triu_indices(len(x), 1)
    v = ncc[iu]
    same = idx.patient_id.to_numpy()[iu[0]] == idx.patient_id.to_numpy()[iu[1]]
    print(f"pairs with NCC > {THRESHOLD}: {(v > THRESHOLD).sum()}; "
          f"max within a volume {v[same].max():.4f}; max across volumes {v[~same].max():.4f}")

    rows = []
    for i, j in zip(*[a[v > THRESHOLD] for a in iu]):
        a, b = idx.loc[i], idx.loc[j]
        order = idx[idx.patient_id == a.patient_id].sort_values("z").img_id.tolist()
        gap = abs(order.index(a.img_id) - order.index(b.img_id))
        print(f"  {a.patient_id} z={a.z} / z={b.z}  NCC {ncc[i, j]:.4f}  slice gap {gap}")
        rows.append(dict(img_a=a.img_id, img_b=b.img_id, patient_a=a.patient_id, patient_b=b.patient_id,
                         ncc=float(ncc[i, j]), same_volume=a.patient_id == b.patient_id, slice_gap=gap))
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    out = paths.TABLES / "near_duplicates.csv"
    pd.DataFrame(rows, columns=["img_a", "img_b", "patient_a", "patient_b", "ncc", "same_volume", "slice_gap"]
                 ).to_csv(out, index=False)
    print("wrote", out.relative_to(paths.REPO))


if __name__ == "__main__":
    main()
