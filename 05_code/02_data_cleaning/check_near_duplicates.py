"""
Near-duplicate check of paper Section 4: normalized cross-correlation (NCC) between all pairs of
lung-windowed slices, resized to 128x128. Run from 05_code/02_data_cleaning.
"""
import os
import numpy as np
import pandas as pd
import tifffile
from skimage.transform import resize

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
idx = pd.read_csv(os.path.join(BASE, "00_inputs", "covid19_ct_seg", "data", "dataset_index.csv")).sort_values("img_id").reset_index(drop=True)
X = np.stack([resize(np.clip(tifffile.imread(os.path.join(BASE, "00_inputs", "covid19_ct_seg", "data", "2d_images", n)).astype(np.float32), -1350, 150),
                     (128, 128), preserve_range=True, anti_aliasing=True).ravel() for n in idx.img_id])
X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
ncc = X @ X.T / X.shape[1]
iu = np.triu_indices(len(X), 1)
v = ncc[iu]
same = idx.patient_id.values[iu[0]] == idx.patient_id.values[iu[1]]
print(f"pairs with NCC > 0.98: {(v > 0.98).sum()}; max within a volume {v[same].max():.4f}; max across volumes {v[~same].max():.4f}")
for i, j in zip(*[a[v > 0.98] for a in iu]):
    a, b = idx.loc[i], idx.loc[j]
    pos = idx[idx.patient_id == a.patient_id].sort_values("z").img_id.tolist()
    gap = abs(pos.index(a.img_id) - pos.index(b.img_id))
    print(f"  {a.patient_id} z={a.z} / z={b.z}  NCC {ncc[i, j]:.4f}  selected-slice gap {gap}")
