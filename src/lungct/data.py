"""Slice loading, intensity mapping, augmentation and data splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from . import paths

IMAGE_SIZE = 256
LUNG_WINDOW = (-1350.0, 150.0)       # centre -600 HU, width 1500 HU
TEST_FRACTION = 0.2                  # held-out test split
VALIDATION_FRACTION = 0.2            # validation split, fraction of the training part


# ----------------------------------------------------------------------------- loading
def load_index(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Slice index written by scripts/prepare_data.py, sorted by slice id."""
    data_dir = Path(data_dir or paths.DATA_PROCESSED)
    df = pd.read_csv(data_dir / "dataset_index.csv").sort_values("img_id").reset_index(drop=True)
    df["image_path"] = [str(data_dir / "2d_images" / i) for i in df.img_id]
    df["mask_path"] = [str(data_dir / "2d_masks" / i) for i in df.img_id]
    return df


def map_intensity(hu: np.ndarray, mode: str) -> np.ndarray:
    """Map Hounsfield units to network inputs.

    hu255        raw HU divided by 255 (unbounded, sign-changing)
    lung_window  clip to the lung window [-1350, +150] HU and rescale to [0, 1]
    """
    if mode == "hu255":
        return hu / 255.0
    if mode == "lung_window":
        lo, hi = LUNG_WINDOW
        return (np.clip(hu, lo, hi) - lo) / (hi - lo)
    raise ValueError(f"unknown intensity mapping: {mode}")


def _resize(a: np.ndarray, size: int) -> np.ndarray:
    return resize(a, (size, size), order=1, preserve_range=True, anti_aliasing=True)


_CACHE: dict = {}


def load_dataset(intensity: str, size: int = IMAGE_SIZE, data_dir=None):
    """Return (index, inputs, masks).

    Each slice is mapped to network inputs at native resolution and then resized to
    size x size (bilinear, anti-aliased); masks are resized and binarised at 0.5.
    """
    key = (str(data_dir), intensity, size)
    if key not in _CACHE:
        df = load_index(data_dir)
        x = np.empty((len(df), size, size), np.float32)
        y = np.empty((len(df), size, size), np.float32)
        for i, (ip, mp) in enumerate(zip(df.image_path, df.mask_path)):
            hu = tifffile.imread(ip).astype(np.float32)
            x[i] = _resize(map_intensity(hu, intensity), size)
            y[i] = _resize(tifffile.imread(mp).astype(np.float32), size) > 127.5
        _CACHE[key] = (df, x, y)
    return _CACHE[key]


def load_hu(size: int = IMAGE_SIZE, data_dir=None) -> np.ndarray:
    """Resized slices in Hounsfield units (for display)."""
    df = load_index(data_dir)
    return np.stack([_resize(tifffile.imread(p).astype(np.float32), size) for p in df.image_path])


# ----------------------------------------------------------------------------- splits
def split_indices(df: pd.DataFrame, scheme: str, seed: int, z_blocks: int = 4, z_fold: int = 0):
    """Return (train, test) slice indices.

    random_slice  random 80/20 split of the slices (seeded)
    z_blocked     slices sorted by within-volume position (z_norm) and cut into z_blocks
                  consecutive blocks; block z_fold is the test set
    patient_lopo  leave-one-patient-out; patient (seed mod number of patients) is the test set
    """
    n = len(df)
    if scheme == "random_slice":
        return train_test_split(np.arange(n), test_size=TEST_FRACTION, random_state=seed)
    if scheme == "z_blocked":
        order = df.sort_values("z_norm", kind="stable").index.to_numpy()
        test = np.sort(np.array_split(order, z_blocks)[z_fold])
        return np.setdiff1d(np.arange(n), test), test
    if scheme == "patient_lopo":
        patients = sorted(df.patient_id.unique())
        test = np.flatnonzero(df.patient_id.to_numpy() == patients[seed % len(patients)])
        return np.setdiff1d(np.arange(n), test), test
    raise ValueError(f"unknown split scheme: {scheme}")


def validation_split(train: np.ndarray, seed: int):
    """Return (train, validation): 20% of the training part, seeded."""
    return train_test_split(train, test_size=VALIDATION_FRACTION, random_state=seed)


# ----------------------------------------------------------------------------- augmentation
def augment(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Horizontal and vertical flips, 90-degree rotations, mild gamma (image only)."""
    if rng.random() < 0.5:
        x, y = x[:, ::-1], y[:, ::-1]
    if rng.random() < 0.5:
        x, y = x[::-1, :], y[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        x, y = np.rot90(x, k), np.rot90(y, k)
    if rng.random() < 0.5:
        x = np.clip(x, 0, 1) ** float(rng.uniform(0.8, 1.25))
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augmentation: bool = False, seed: int = 0):
        self.x, self.y, self.augmentation = x, y, augmentation
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        xi, yi = self.x[i], self.y[i]
        if self.augmentation:
            xi, yi = augment(xi, yi, self.rng)
        return torch.from_numpy(xi[None].copy()), torch.from_numpy(yi[None].copy())
