"""
lug_method.py  --  self-contained Phase-5 engine for Google Colab (GPU).
===============================================================================
Study: exact reproduction + error decomposition of a preprinted ResUNet++
lung-CT segmentation pipeline (Kaggle "Finding Lungs in CT Data", 267 slices).

Design & pre-registration: see 04_new_study_protocol/protocol.md (v2).
Nothing here fabricates numbers; every result is written to a CSV with the seed,
split and config that produced it.

Usage on Colab
--------------
    !pip -q install medpy tifffile scikit-image
    from lug_method import *
    prepare_data("/content/drive/MyDrive/LugMethod")          # one-time
    run_faithful(out_dir, n_seeds=20)                         # O1
    run_gap_decomposition(out_dir, n_seeds=20)                # O2
    run_arch_benchmark(out_dir, n_seeds=5)                    # O3/O4
    run_leakage_probe(out_dir, n_seeds=20, Ks=(4, 8))         # O6

Each runner appends to a CSV under <out_dir>/results/ and is safe to re-run
(rows carry a unique (experiment, seed, arm, ...) key; duplicates are dropped on
load in the analysis stage, not here).

v2 (2026-09-03): _load_raw no longer relies on DataFrame.itertuples attribute
access or on the image_path/mask_path columns -- file paths are re-resolved from
img_id by scanning the data dir, so a malformed dataset_index.csv (e.g. from a
zip-wrapped lung_stats.csv on Colab) no longer breaks training. prepare_data is
also hardened (robust lung_stats reader, duplicate-column drop).
===============================================================================
"""
from __future__ import annotations

import glob
import io
import json
import os
import time
import urllib.request
import zipfile
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize as sk_resize
from sklearn.model_selection import train_test_split

torch.backends.cudnn.benchmark = True
EPS = 1e-7
KAGGLE_SLUG = "kmader/finding-lungs-in-ct-data"
PRIOR_TABLE1 = dict(reported_dsc=0.9738, reported_acc=0.9599, reported_loss=0.0241)


# =============================================================================
# 1. DATA
# =============================================================================
def _kaggle_file(slug: str, fname: str, dest: str):
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}/{fname}"
    print(f"  downloading {fname} ...", flush=True)
    urllib.request.urlretrieve(url, dest)


def _double_unzip(zpath: str, out_dir: str):
    """Kaggle per-file endpoint wraps the file in an outer zip; unwrap twice."""
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        z.extractall(out_dir)
    for n in names:
        inner = os.path.join(out_dir, n)
        if inner.endswith(".zip"):
            with zipfile.ZipFile(inner) as z2:
                z2.extractall(os.path.join(out_dir, n[:-4]))


def prepare_data(root: str, kaggle_json: str | None = None):
    """
    Create <root>/data/{2d_images,2d_masks}/*.tif, <root>/data/lung_stats.csv and
    <root>/data/dataset_index.csv. Idempotent. If the Kaggle anonymous endpoint
    is blocked, upload the three files manually to <root>/data/ and re-run.
    """
    data = os.path.join(root, "data")
    os.makedirs(data, exist_ok=True)
    if kaggle_json and os.path.exists(kaggle_json):
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        os.replace(kaggle_json, os.path.expanduser("~/.kaggle/kaggle.json"))
        os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

    if not glob.glob(os.path.join(data, "2d_images", "**", "*.tif"), recursive=True):
        for fn in ("2d_images.zip", "2d_masks.zip"):
            zp = os.path.join(data, fn)
            if not os.path.exists(zp):
                _kaggle_file(KAGGLE_SLUG, fn, zp)
            _double_unzip(zp, data)
    ls = os.path.join(data, "lung_stats.csv")
    if not os.path.exists(ls):
        _kaggle_file(KAGGLE_SLUG, "lung_stats.csv", ls)

    imgs, msks = _scan_tif_dirs(data)
    common = sorted(set(imgs) & set(msks))
    rows = [dict(img_id=n, image_path=imgs[n], mask_path=msks[n],
                 z=int(n.split("_Z_")[1].split(".")[0])) for n in common]
    df = pd.DataFrame(rows)

    stats = _read_lung_stats(ls)
    if stats is not None:
        df = df.merge(stats, on="img_id", how="left")
    else:
        print("  WARNING: lung_stats.csv unreadable/missing -- covariate columns omitted "
              "(O5 failure-mode regression will need it later).")

    assert len(df) == 267, f"expected 267 pairs, got {len(df)}"
    # de-duplicate any accidental repeated columns from a bad merge
    df = df.loc[:, ~df.columns.duplicated()]
    idx_path = os.path.join(data, "dataset_index.csv")
    df.to_csv(idx_path, index=False)
    print(f"prepared {len(df)} pairs -> {idx_path}   columns={list(df.columns)}")
    return idx_path


def _scan_tif_dirs(data: str):
    """Return {basename: path} for images and masks, whatever the nesting depth."""
    all_tif = glob.glob(os.path.join(data, "**", "*.tif"), recursive=True)
    norm = lambda p: p.replace("\\", "/")
    imgs = {os.path.basename(p): p for p in all_tif if "/2d_images" in norm(p)}
    msks = {os.path.basename(p): p for p in all_tif if "/2d_masks" in norm(p)}
    return imgs, msks


def _read_lung_stats(path: str):
    """Robustly read lung_stats.csv even if the download wrapped it in a zip."""
    if not os.path.exists(path):
        return None
    expected = {"img_id", "lung_area_px", "lung_mean_hu"}
    for attempt in ("plain", "zip"):
        try:
            if attempt == "plain":
                df = pd.read_csv(path)
            else:
                with zipfile.ZipFile(path) as z:
                    inner = [n for n in z.namelist() if n.endswith(".csv")][0]
                    with z.open(inner) as fh:
                        df = pd.read_csv(fh)
            if expected.issubset(df.columns):
                return df.loc[:, ~df.columns.duplicated()]
        except Exception:
            continue
    return None


_RAW_CACHE: dict = {}


def _load_raw(root: str, size: int):
    key = (root, size)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]
    data = os.path.join(root, "data")
    df = pd.read_csv(os.path.join(data, "dataset_index.csv"))
    df = df.loc[:, ~df.columns.duplicated()].sort_values("img_id").reset_index(drop=True)

    # Resolve file paths by img_id (robust to a bad image_path/mask_path column).
    imgs, msks = _scan_tif_dirs(data)
    def resolve(row, table, col):
        p = str(row[col]) if col in df.columns else ""
        if p and os.path.exists(p):
            return p
        return table[row["img_id"]]
    img_paths = [resolve(r, imgs, "image_path") for _, r in df.iterrows()]
    msk_paths = [resolve(r, msks, "mask_path") for _, r in df.iterrows()]

    n = len(df)
    hu = np.empty((n, size, size), np.float32)     # raw HU, resized
    mk = np.empty((n, size, size), np.float32)     # {0,1}, resized
    for i, (ip, mp) in enumerate(zip(img_paths, msk_paths)):
        a = tifffile.imread(ip).astype(np.float32)
        b = tifffile.imread(mp).astype(np.float32)
        hu[i] = sk_resize(a, (size, size), order=1, preserve_range=True, anti_aliasing=True)
        mk[i] = (sk_resize(b, (size, size), order=1, preserve_range=True,
                           anti_aliasing=True) > 127.5).astype(np.float32)
    df["image_path"], df["mask_path"] = img_paths, msk_paths
    _RAW_CACHE[key] = (df, hu, mk)
    return _RAW_CACHE[key]


def _preprocess(hu: np.ndarray, mode: str, wc=-600, ww=1500) -> np.ndarray:
    if mode == "divide_255":                        # released defect (on int16 HU)
        return hu / 255.0
    if mode == "hu_window_minmax":                  # corrected
        lo, hi = wc - ww / 2, wc + ww / 2
        x = np.clip(hu, lo, hi)
        return (x - lo) / (hi - lo)
    raise ValueError(mode)


# ---- light augmentation (numpy, no extra deps) ------------------------------
def _augment(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, level: str):
    if level == "none":
        return x, y
    if rng.random() < 0.5:
        x, y = x[:, ::-1], y[:, ::-1]
    if level in ("light", "full") and rng.random() < 0.5:
        x, y = x[::-1, :], y[::-1, :]
    if level == "full":
        k = int(rng.integers(0, 4))
        if k:
            x, y = np.rot90(x, k), np.rot90(y, k)
        if rng.random() < 0.5:                       # gamma on the image only
            g = float(rng.uniform(0.8, 1.25))
            x = np.clip(x, 0, 1) ** g
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


class SliceDS(torch.utils.data.Dataset):
    def __init__(self, x, y, aug="none", seed=0):
        self.x, self.y, self.aug = x, y, aug
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        xi, yi = self.x[i], self.y[i]
        if self.aug != "none":
            xi, yi = _augment(xi, yi, self.rng, self.aug)
        return (torch.from_numpy(xi[None].copy()).float(),
                torch.from_numpy(yi[None].copy()).float())


# =============================================================================
# 2. MODELS
# =============================================================================
BN_EPS, BN_MOM = 1e-3, 0.01


def _bn(c):
    return nn.BatchNorm2d(c, eps=BN_EPS, momentum=BN_MOM)


# ---- 2a. faithful ResUNet++ (exact port of the released Keras model) --------
class _SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.f1 = nn.Linear(c, c // r, bias=False)
        self.f2 = nn.Linear(c // r, c, bias=False)

    def forward(self, x):
        s = F.relu(self.f1(x.mean((2, 3))))
        return x * torch.sigmoid(self.f2(s))[:, :, None, None]


class _Stem(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(ci, co, 3, padding=1), _bn(co)
        self.c2, self.sc = nn.Conv2d(co, co, 3, padding=1), nn.Conv2d(ci, co, 1)

    def forward(self, x):
        y = self.c2(F.relu(self.b1(self.c1(x))))
        return y + self.sc(x)


class _Res(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.se = _SE(ci)
        self.b1, self.c1 = _bn(ci), nn.Conv2d(ci, co, 3, stride=stride, padding=1)
        self.b2, self.c2 = _bn(co), nn.Conv2d(co, co, 3, padding=1)
        self.sc = nn.Conv2d(ci, co, 1, stride=stride)

    def forward(self, x):
        x = self.se(x)
        y = self.c1(F.relu(self.b1(x)))
        y = self.c2(F.relu(self.b2(y)))
        return y + self.sc(x)


class _ASPP(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.b = nn.ModuleList(
            nn.Sequential(nn.Conv2d(ci, co, 3, padding=d, dilation=d), _bn(co))
            for d in (6, 12, 18))
        self.b4 = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), _bn(co))
        self.proj = nn.Conv2d(co, co, 1)

    def forward(self, x):
        return self.proj(sum(b(x) for b in self.b) + self.b4(x))


class _Attn(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        f = c2
        self.b1, self.k1 = _bn(c1), nn.Conv2d(c1, f, 3, padding=1)
        self.b2, self.k2 = _bn(c2), nn.Conv2d(c2, f, 3, padding=1)
        self.b3, self.k3 = _bn(f), nn.Conv2d(f, f, 3, padding=1)

    def forward(self, x1, x2):
        a = F.max_pool2d(self.k1(F.relu(self.b1(x1))), 2)
        b = self.k2(F.relu(self.b2(x2)))
        return self.k3(F.relu(self.b3(a + b))) * x2


class ResUNetPP(nn.Module):
    """4,085,873 trainable params -- matches the released Keras summary."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.c1 = _Stem(in_ch, 16)
        self.c2, self.c3, self.c4 = _Res(16, 32, 2), _Res(32, 64, 2), _Res(64, 128, 2)
        self.b1 = _ASPP(128, 256)
        self.a1, self.r1 = _Attn(64, 256), _Res(320, 128)
        self.a2, self.r2 = _Attn(32, 128), _Res(160, 64)
        self.a3, self.r3 = _Attn(16, 64), _Res(80, 32)
        self.ha, self.hc = _ASPP(32, 16), nn.Conv2d(16, 1, 1)

    def forward(self, x):
        c1 = self.c1(x); c2 = self.c2(c1); c3 = self.c3(c2); c4 = self.c4(c3)
        b1 = self.b1(c4)
        d = self.r1(torch.cat([F.interpolate(self.a1(c3, b1), scale_factor=2), c3], 1))
        d = self.r2(torch.cat([F.interpolate(self.a2(c2, d), scale_factor=2), c2], 1))
        d = self.r3(torch.cat([F.interpolate(self.a3(c1, d), scale_factor=2), c1], 1))
        return torch.sigmoid(self.hc(self.ha(d)))


# ---- 2b. baselines (comparable scale, standard designs) --------------------
def _cbr(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), _bn(co), nn.ReLU(inplace=True),
                         nn.Conv2d(co, co, 3, padding=1), _bn(co), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3, self.e4 = _cbr(in_ch, b), _cbr(b, 2*b), _cbr(2*b, 4*b), _cbr(4*b, 8*b)
        self.pool = nn.MaxPool2d(2)
        self.bott = _cbr(8*b, 16*b)
        self.u4 = nn.ConvTranspose2d(16*b, 8*b, 2, 2); self.d4 = _cbr(16*b, 8*b)
        self.u3 = nn.ConvTranspose2d(8*b, 4*b, 2, 2);  self.d3 = _cbr(8*b, 4*b)
        self.u2 = nn.ConvTranspose2d(4*b, 2*b, 2, 2);  self.d2 = _cbr(4*b, 2*b)
        self.u1 = nn.ConvTranspose2d(2*b, b, 2, 2);    self.d1 = _cbr(2*b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3)); bn = self.bott(self.pool(e4))
        d = self.d4(torch.cat([self.u4(bn), e4], 1))
        d = self.d3(torch.cat([self.u3(d), e3], 1))
        d = self.d2(torch.cat([self.u2(d), e2], 1))
        d = self.d1(torch.cat([self.u1(d), e1], 1))
        return torch.sigmoid(self.out(d))


class _AttnGate(nn.Module):
    def __init__(self, cg, cx, ci):
        super().__init__()
        self.wg = nn.Conv2d(cg, ci, 1); self.wx = nn.Conv2d(cx, ci, 1)
        self.psi = nn.Conv2d(ci, 1, 1)

    def forward(self, g, x):
        a = torch.sigmoid(self.psi(F.relu(self.wg(g) + self.wx(x))))
        return x * a


class AttUNet(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3, self.e4 = _cbr(in_ch, b), _cbr(b, 2*b), _cbr(2*b, 4*b), _cbr(4*b, 8*b)
        self.pool = nn.MaxPool2d(2); self.bott = _cbr(8*b, 16*b)
        self.u4 = nn.ConvTranspose2d(16*b, 8*b, 2, 2); self.g4 = _AttnGate(8*b, 8*b, 4*b); self.d4 = _cbr(16*b, 8*b)
        self.u3 = nn.ConvTranspose2d(8*b, 4*b, 2, 2);  self.g3 = _AttnGate(4*b, 4*b, 2*b); self.d3 = _cbr(8*b, 4*b)
        self.u2 = nn.ConvTranspose2d(4*b, 2*b, 2, 2);  self.g2 = _AttnGate(2*b, 2*b, b);   self.d2 = _cbr(4*b, 2*b)
        self.u1 = nn.ConvTranspose2d(2*b, b, 2, 2);    self.g1 = _AttnGate(b, b, b // 2);  self.d1 = _cbr(2*b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3)); bn = self.bott(self.pool(e4))
        u = self.u4(bn); d = self.d4(torch.cat([u, self.g4(u, e4)], 1))
        u = self.u3(d);  d = self.d3(torch.cat([u, self.g3(u, e3)], 1))
        u = self.u2(d);  d = self.d2(torch.cat([u, self.g2(u, e2)], 1))
        u = self.u1(d);  d = self.d1(torch.cat([u, self.g1(u, e1)], 1))
        return torch.sigmoid(self.out(d))


class _ResCBR(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.body = _cbr(ci, co)
        self.sc = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x):
        return F.relu(self.body(x) + self.sc(x))


class ResUNet(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3, self.e4 = _ResCBR(in_ch, b), _ResCBR(b, 2*b), _ResCBR(2*b, 4*b), _ResCBR(4*b, 8*b)
        self.pool = nn.MaxPool2d(2); self.bott = _ResCBR(8*b, 16*b)
        self.u4 = nn.ConvTranspose2d(16*b, 8*b, 2, 2); self.d4 = _ResCBR(16*b, 8*b)
        self.u3 = nn.ConvTranspose2d(8*b, 4*b, 2, 2);  self.d3 = _ResCBR(8*b, 4*b)
        self.u2 = nn.ConvTranspose2d(4*b, 2*b, 2, 2);  self.d2 = _ResCBR(4*b, 2*b)
        self.u1 = nn.ConvTranspose2d(2*b, b, 2, 2);    self.d1 = _ResCBR(2*b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3)); bn = self.bott(self.pool(e4))
        d = self.d4(torch.cat([self.u4(bn), e4], 1))
        d = self.d3(torch.cat([self.u3(d), e3], 1))
        d = self.d2(torch.cat([self.u2(d), e2], 1))
        d = self.d1(torch.cat([self.u1(d), e1], 1))
        return torch.sigmoid(self.out(d))


ARCHS = {"unet": UNet, "attn_unet": AttUNet, "resunet": ResUNet, "resunetpp": ResUNetPP}


def build(name: str) -> nn.Module:
    return ARCHS[name]()


# =============================================================================
# 3. METRICS
# =============================================================================
def soft_dice(p, t):
    p, t = p.reshape(-1), t.reshape(-1)
    return ((2 * (p * t).sum() + EPS) / (p.sum() + t.sum() + EPS)).item()


def hard_stats(p, t, thr=0.5):
    p = (p > thr).float().reshape(-1); t = (t > 0.5).float().reshape(-1)
    i = (p * t).sum()
    return dict(dice=((2 * i + EPS) / (p.sum() + t.sum() + EPS)).item(),
                iou=((i + EPS) / ((p + t - p * t).sum() + EPS)).item(),
                acc=(p == t).float().mean().item(),
                sens=((i + EPS) / (t.sum() + EPS)).item(),
                spec=(((1 - p) * (1 - t)).sum() + EPS) / ((1 - t).sum() + EPS))


def per_image_dice(pred, tgt, thr=0.5):
    p = (pred > thr).float(); t = (tgt > 0.5).float()
    d = []
    for k in range(len(p)):
        pk, tk = p[k].reshape(-1), t[k].reshape(-1)
        i = (pk * tk).sum()
        d.append(((2 * i + EPS) / (pk.sum() + tk.sum() + EPS)).item())
    return np.array(d)


def ece_brier(pred, tgt, bins=15):
    p = pred.reshape(-1).cpu().numpy(); t = (tgt.reshape(-1) > 0.5).cpu().numpy().astype(float)
    brier = float(np.mean((p - t) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            ece += m.mean() * abs(t[m].mean() - p[m].mean())
    return float(ece), brier


# =============================================================================
# 4. TRAIN / EVAL CORE
# =============================================================================
@dataclass
class Cfg:
    arm: str = "faithful"          # faithful | corrected
    arch: str = "resunetpp"
    size: int = 256
    norm: str = "divide_255"       # divide_255 | hu_window_minmax
    aug: str = "none"              # none | light | full
    loss: str = "bce"             # bce | bce_dice
    epochs: int = 20
    batch: int = 10
    lr: float = 1e-3
    sched: str = "plateau"        # plateau | cosine
    early_stop: int = 0           # 0 = off (faithful)
    select: str = "last"          # last | best_val
    thr: str = "fixed"           # fixed(0.5) | tuned
    split: str = "random_slice"   # random_slice | z_blocked | patient_lopo
    z_K: int = 4


def _split_indices(df, cfg: Cfg, seed):
    n = len(df)
    if cfg.split == "random_slice":
        tr, va = train_test_split(np.arange(n), test_size=0.2, random_state=seed)
        return tr, va
    if cfg.split == "patient_lopo":
        # leave-one-patient-out: seed s holds out the s-th patient (protocol §Sensitivity:
        # replaces the Z-blocked approximation when true patient labels exist).
        pats = sorted(df["patient_id"].unique())
        held = pats[seed % len(pats)]
        va = np.flatnonzero(df["patient_id"].to_numpy() == held)
        tr = np.setdiff1d(np.arange(n), va)
        return tr, va
    # multi-volume data: raw slice index is not comparable across patients, so band
    # on the within-patient lung position when available.
    zcol = "z_norm" if "z_norm" in df.columns else "z"
    order = df.sort_values(zcol).index.to_numpy()
    bands = np.array_split(order, cfg.z_K)
    rng = np.random.default_rng(seed)
    held = bands[rng.integers(0, cfg.z_K)]
    va = np.sort(held)
    tr = np.setdiff1d(np.arange(n), va)
    return tr, va


def _loss_fn(kind):
    bce = nn.BCELoss()
    if kind == "bce":
        return lambda o, y: bce(o, y)
    def _bd(o, y):
        d = 1 - (2 * (o * y).sum() + 1) / (o.sum() + y.sum() + 1)
        return bce(o, y) + d
    return _bd


def _safe_micro_batch(size: int) -> int:
    """VRAM-safe micro-batch on a 2GB card (profiled 2026-09-04: bs=10@256px thrashes
    VRAM and drops throughput ~100x; bs<=4@256px and bs<=8@128px are fast and stay well
    under the cliff). True batch semantics for larger cfg.batch are restored via
    gradient accumulation in train_one, so this only trades a bit of BatchNorm-statistics
    fidelity (computed per micro-batch) for the ability to run at all on this GPU."""
    if size <= 128:
        return 8
    if size <= 256:
        return 4
    return 2


@torch.no_grad()
def _infer(model, x, device, bs=None):
    model.eval()
    bs = bs or _safe_micro_batch(x.shape[-1]) * 2
    out = [model(torch.from_numpy(x[i:i+bs][:, None]).float().to(device)).cpu()
           for i in range(0, len(x), bs)]
    return torch.cat(out)


def train_one(root, cfg: Cfg, seed, device="cuda", verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    df, hu, mk = _load_raw(root, cfg.size)
    X = _preprocess(hu, cfg.norm)
    tr, va = _split_indices(df, cfg, seed)

    inner_tr, inner_va = (tr, None)
    if cfg.select == "best_val" or cfg.thr == "tuned" or cfg.early_stop:
        inner_tr, inner_va = train_test_split(tr, test_size=0.2, random_state=seed)

    micro = min(cfg.batch, _safe_micro_batch(cfg.size)) if device == "cuda" else cfg.batch
    accum = max(1, -(-cfg.batch // micro))     # ceil(cfg.batch / micro); emulates true batch size
    ds = SliceDS(X[inner_tr], mk[inner_tr], aug=cfg.aug, seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=micro, shuffle=True, drop_last=False)

    model = build(cfg.arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.sched == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.2, patience=10, min_lr=1e-5)
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    lossf = _loss_fn(cfg.loss)

    best_state, best_metric, bad = None, -1, 0
    monX = X[inner_va] if inner_va is not None else X[va]
    monY = mk[inner_va] if inner_va is not None else mk[va]
    hist = []
    for ep in range(cfg.epochs):
        model.train()
        tl = n = 0.0
        opt.zero_grad()
        for i, (xb, yb) in enumerate(dl):
            xb, yb = xb.to(device), yb.to(device)
            o = model(xb); l = lossf(o, yb)
            (l / accum).backward()
            if (i + 1) % accum == 0:
                opt.step(); opt.zero_grad()
            tl += l.item(); n += 1
        if n % accum != 0:
            opt.step(); opt.zero_grad()
        pv = _infer(model, monX, device)
        vloss = nn.functional.binary_cross_entropy(pv, torch.from_numpy(monY[:, None]).float()).item()
        vdice = soft_dice(pv, torch.from_numpy(monY[:, None]).float())
        if cfg.sched == "plateau":
            sch.step(vloss)
        else:
            sch.step()
        hist.append(dict(epoch=ep+1, train_loss=tl/n, mon_loss=vloss, mon_soft_dice=vdice,
                         lr=opt.param_groups[0]["lr"]))
        if verbose:
            print(f"    ep{ep+1:02d} tl {tl/n:.4f}  mon_loss {vloss:.4f}  mon_dice {vdice:.4f}")
        score = vdice
        if score > best_metric:
            best_metric, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if cfg.early_stop and bad >= cfg.early_stop:
                break

    if cfg.select == "best_val" and best_state is not None:
        model.load_state_dict(best_state)

    # threshold
    thr = 0.5
    if cfg.thr == "tuned" and inner_va is not None:
        pv = _infer(model, X[inner_va], device)
        tv = torch.from_numpy(mk[inner_va][:, None]).float()
        thr = max(np.arange(0.05, 0.96, 0.05),
                  key=lambda t: soft_dice((pv > t).float(), tv))

    # held-out evaluation
    pt = _infer(model, X[va], device)
    tt = torch.from_numpy(mk[va][:, None]).float()
    hs = hard_stats(pt, tt, thr=thr)
    pid = per_image_dice(pt, tt, thr=thr)
    ece, brier = ece_brier(pt, tt)
    last = hist[-1]

    row = dict(
        **{k: v for k, v in asdict(cfg).items()},
        seed=seed, n_train=len(tr), n_val=len(va), epochs_run=len(hist),
        thr_used=float(thr),
        # what the prior Table 1 actually shows == final-epoch TRAIN row:
        reportedlike_train_loss=last["train_loss"],
        # true held-out:
        heldout_soft_dice=soft_dice(pt, tt),
        heldout_dice=hs["dice"], heldout_iou=hs["iou"], heldout_acc=hs["acc"],
        heldout_sens=hs["sens"], heldout_spec=float(hs["spec"]),
        heldout_dice_median=float(np.median(pid)),
        heldout_dice_iqr=float(np.percentile(pid, 75) - np.percentile(pid, 25)),
        heldout_frac_empty_pred=float(np.mean(pid == 0)),
        heldout_ece=ece, heldout_brier=brier,
        best_mon_soft_dice=best_metric,
    )
    val_ids = df.iloc[np.asarray(va)]["img_id"].tolist()
    pid_df = pd.DataFrame(dict(img_id=val_ids, dice=pid))
    return row, hist, pid_df


# =============================================================================
# 5. EXPERIMENT RUNNERS
# =============================================================================
def _append_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    return path


def _already_done(path, key_cols: dict) -> bool:
    """True if a row matching every key_cols item already exists in <path> (resume support)."""
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    m = pd.Series(True, index=df.index)
    for k, v in key_cols.items():
        if k not in df.columns:
            return False
        m &= (df[k].astype(str) == str(v))
    return bool(m.any())


def run_faithful(out_dir, n_seeds=20, device="cuda", resume=True):
    """O1 -- exact released recipe, n_seeds repetitions (unseeded split emulated by seed sweep)."""
    root = out_dir
    csv = os.path.join(out_dir, "results", "faithful_runs.csv")
    rows, hists = [], {}
    for s in range(n_seeds):
        if resume and _already_done(csv, dict(experiment="faithful", seed=s)):
            print(f"[faithful seed {s}] already done -- skip"); continue
        t0 = time.time()
        cfg = Cfg(arm="faithful", arch="resunetpp", size=256, norm="divide_255",
                  aug="none", loss="bce", epochs=20, batch=10, lr=1e-3,
                  sched="plateau", early_stop=0, select="last", thr="fixed",
                  split="random_slice")
        row, hist, _ = train_one(root, cfg, seed=s, device=device, verbose=(s == 0))
        row["experiment"] = "faithful"
        row["secs"] = round(time.time() - t0)
        _append_csv(csv, [row])
        rows.append(row); hists[s] = hist
        print(f"[faithful seed {s}] heldout Dice@thr {row['heldout_dice']:.4f} "
              f"soft {row['heldout_soft_dice']:.4f} acc {row['heldout_acc']:.4f} "
              f"empty-pred {row['heldout_frac_empty_pred']:.2f}  ({row['secs']}s)")
    hp = os.path.join(out_dir, "results", "faithful_histories.json")
    if hists:
        old = json.load(open(hp)) if os.path.exists(hp) else {}
        old.update({str(k): v for k, v in hists.items()})
        json.dump(old, open(hp, "w"))
    p = csv
    d = pd.read_csv(csv)
    d = d[d.experiment == "faithful"]
    print("\n=== O1 SUMMARY (n_seeds={}) ===".format(n_seeds))
    print(f"prior Table 1 (labelled 'validation'): DSC {PRIOR_TABLE1['reported_dsc']}  "
          f"acc {PRIOR_TABLE1['reported_acc']}")
    print(f"reproduced true held-out Dice@0.5 : {d.heldout_dice.mean():.4f} +/- {d.heldout_dice.std():.4f}"
          f"  [{d.heldout_dice.min():.4f}, {d.heldout_dice.max():.4f}]")
    print(f"reproduced held-out soft-Dice     : {d.heldout_soft_dice.mean():.4f} +/- {d.heldout_soft_dice.std():.4f}")
    print(f"reproduced held-out accuracy      : {d.heldout_acc.mean():.4f} +/- {d.heldout_acc.std():.4f}")
    print(f"runs predicting ~empty mask       : {(d.heldout_frac_empty_pred>0.5).mean()*100:.0f}%")
    print("wrote", p)
    return p


def run_gap_decomposition(out_dir, n_seeds=20, device="cuda"):
    """O2 -- switch on one correction at a time; attribute the Dice change to each."""
    arms = [
        ("A_faithful",        Cfg(arm="faithful", norm="divide_255", aug="none",
                                  select="last", thr="fixed", epochs=20, sched="plateau")),
        ("B_heldout_thresh",  Cfg(arm="faithful", norm="divide_255", aug="none",
                                  select="best_val", thr="tuned", epochs=20, sched="plateau")),
        ("C_hu_window",       Cfg(arm="corrected", norm="hu_window_minmax", aug="none",
                                  select="best_val", thr="tuned", epochs=20, sched="plateau")),
        ("D_augmentation",    Cfg(arm="corrected", norm="hu_window_minmax", aug="full",
                                  select="best_val", thr="tuned", epochs=40, sched="cosine",
                                  early_stop=15)),
    ]
    csv = os.path.join(out_dir, "results", "gap_decomposition.csv")
    for name, cfg in arms:
        for s in range(n_seeds):
            if _already_done(csv, dict(experiment="gap_decomposition", arm_name=name, seed=s)):
                print(f"[{name} seed {s}] skip"); continue
            t0 = time.time()
            row, _, _ = train_one(out_dir, cfg, seed=s, device=device)
            row.update(experiment="gap_decomposition", arm_name=name, secs=round(time.time()-t0))
            _append_csv(csv, [row])
            print(f"[{name} seed {s}] heldout Dice {row['heldout_dice']:.4f}  ({row['secs']}s)")
    p = csv
    d = pd.read_csv(csv)
    d = d[d.experiment == "gap_decomposition"].groupby("arm_name").heldout_dice.agg(["mean", "std"])
    print("\n=== O2 gap decomposition (held-out Dice@thr) ===")
    print(d)
    print("wrote", p)
    return p


def run_arch_benchmark(out_dir, n_seeds=5, device="cuda", archs=tuple(ARCHS)):
    """O3/O4 -- corrected protocol, all architectures, matched budget."""
    csv = os.path.join(out_dir, "results", "arch_benchmark.csv")
    for arch in archs:
        for s in range(n_seeds):
            if _already_done(csv, dict(experiment="arch_benchmark", arch=arch, seed=s)):
                print(f"[{arch} seed {s}] skip"); continue
            t0 = time.time()
            cfg = Cfg(arm="corrected", arch=arch, norm="hu_window_minmax", aug="full",
                      loss="bce_dice", epochs=60, batch=8, sched="cosine", early_stop=15,
                      select="best_val", thr="tuned", split="random_slice")
            row, _, pid_df = train_one(out_dir, cfg, seed=s, device=device)
            row.update(experiment="arch_benchmark", secs=round(time.time()-t0))
            _append_csv(csv, [row])
            pid_df.assign(arch=arch, seed=s).to_csv(
                os.path.join(out_dir, "results", f"pid_{arch}_seed{s}.csv"), index=False)
            print(f"[{arch} seed {s}] heldout Dice {row['heldout_dice']:.4f} "
                  f"IoU {row['heldout_iou']:.4f} ECE {row['heldout_ece']:.3f}  ({row['secs']}s)")
    p = csv
    print(pd.read_csv(csv).query("experiment=='arch_benchmark'").groupby("arch").heldout_dice.agg(["mean", "std"]))
    print("wrote", p)
    return p


def run_leakage_probe(out_dir, n_seeds=20, Ks=(4, 8), device="cuda",
                      archs=tuple(ARCHS), schemes=("random_slice", "z_blocked"),
                      n_lopo=None):
    """O6 -- random-slice vs Z-blocked split, same models. Approximate leakage bound.
    schemes may include "patient_lopo" (needs a patient_id column); it runs one fold per
    patient (n_lopo folds, default = number of patients) instead of n_seeds."""
    csv = os.path.join(out_dir, "results", "leakage_probe.csv")
    for arch in archs:
        for scheme in schemes:
            K_list = list(Ks) if scheme == "z_blocked" else [0]
            if scheme == "patient_lopo":
                df_idx = pd.read_csv(os.path.join(out_dir, "data", "dataset_index.csv"))
                seeds = range(n_lopo or df_idx["patient_id"].nunique())
            else:
                seeds = range(n_seeds)
            for K in K_list:
                for s in seeds:
                    if _already_done(csv, dict(experiment="leakage_probe", arch=arch,
                                               scheme=scheme, z_K=K, seed=s)):
                        continue
                    t0 = time.time()
                    cfg = Cfg(arm="corrected", arch=arch, norm="hu_window_minmax",
                              aug="full", loss="bce_dice", epochs=40, batch=8,
                              sched="cosine", early_stop=15, select="best_val",
                              thr="tuned", split=scheme, z_K=(K or 4))
                    row, _, _ = train_one(out_dir, cfg, seed=s, device=device)
                    row.update(experiment="leakage_probe", scheme=scheme, z_K=K,
                               secs=round(time.time()-t0))
                    _append_csv(csv, [row])
            print(f"[{arch} {scheme}] done")
    p = csv
    d = pd.read_csv(csv).query("experiment=='leakage_probe'").groupby(
        ["arch", "scheme"]).heldout_dice.mean().unstack()
    print("\n=== O6 leakage probe (mean held-out Dice) ===")
    print(d)
    print("wrote", p)
    return p


def param_counts():
    return {k: sum(p.numel() for p in build(k).parameters()) for k in ARCHS}


if __name__ == "__main__":
    print("param counts:", param_counts())
