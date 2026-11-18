"""
06_main_analysis / data_faithful.py
-------------------------------------------------------------------------------
Data pipeline that reproduces the released notebook EXACTLY (cells [5]-[10]):

  * read int16 HU TIFF and uint8 mask TIFF
  * resize 512 -> 256 with skimage.transform.resize(order=1, preserve_range=True)
    (bilinear; anti_aliasing default True for downsampling) -- same call as cell [8]
  * x_data /= 255 ; y_data /= 255            (cell [9]) -- the defect on int16 HU
  * sklearn.train_test_split(test_size=0.2)  (cell [10]) -- NO random_state

Returns float32 tensors shaped (N, 1, 256, 256).  Nothing is windowed or
augmented here -- that is the point of the "faithful" arm.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
from skimage.transform import resize as sk_resize
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_config, read_tif  # noqa: E402


def _load_stack(cfg) -> tuple[np.ndarray, np.ndarray, list[str]]:
    idx = pd.read_csv(cfg["paths"]["dataset_index"]).sort_values("img_id").reset_index(drop=True)
    size = cfg["faithful_pipeline"]["input_size"]
    n = len(idx)
    x = np.empty((n, size, size, 1), np.float32)
    y = np.empty((n, size, size, 1), np.float32)
    for i, r in enumerate(idx.itertuples()):
        im = read_tif(r.image_path).astype(np.float32)
        mk = read_tif(r.mask_path).astype(np.float32)
        x[i] = sk_resize(im, (size, size, 1), order=1, preserve_range=True)
        y[i] = sk_resize(mk, (size, size, 1), order=1, preserve_range=True)
    x /= 255.0          # released bug: int16 HU / 255  -> approx [-4, 4]
    y /= 255.0          # mask 0/255 -> 0/1
    return x, y, idx["img_id"].tolist()


_CACHE: dict = {}


def get_faithful_split(cfg, seed=None):
    """seed=None reproduces the released code (unseeded split). Pass an int to fix it."""
    key = ("stack",)
    if key not in _CACHE:
        _CACHE[key] = _load_stack(cfg)
    x, y, ids = _CACHE[key]
    vf = cfg["faithful_pipeline"]["val_fraction"]
    xtr, xva, ytr, yva, itr, iva = train_test_split(
        x, y, ids, test_size=vf, random_state=seed)
    # to (N,1,H,W)
    def chw(a): return np.transpose(a, (0, 3, 1, 2)).copy()
    return dict(x_train=chw(xtr), y_train=chw(ytr), id_train=itr,
                x_val=chw(xva), y_val=chw(yva), id_val=iva)


if __name__ == "__main__":
    cfg = load_config()
    d = get_faithful_split(cfg, seed=0)
    for k, v in d.items():
        if hasattr(v, "shape"):
            print(f"{k:9s} {v.shape}  dtype {v.dtype}  range [{v.min():.3f}, {v.max():.3f}]")
        else:
            print(f"{k:9s} n={len(v)}  e.g. {v[:2]}")
