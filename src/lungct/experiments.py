"""Experiments of the article. Each runner appends one CSV row per run to results/runs/
and skips runs that are already there, so an interrupted experiment can be resumed."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import paths
from .data import load_index
from .training import RunConfig, stabilized_config, train_run

ARCHS = ("unet", "attn_unet", "resunet", "resunetpp")

# Progressive protocol (article Section 4.4, Table 3): cumulative arms
PROGRESSIVE_ARMS = {
    "A_baseline": RunConfig(),
    "B_best_epoch_threshold": RunConfig(model_selection="best", threshold="tuned"),
    "C_lung_window": RunConfig(model_selection="best", threshold="tuned", intensity="lung_window"),
    "D_augmentation": RunConfig(model_selection="best", threshold="tuned", intensity="lung_window",
                                augmentation=True, scheduler="cosine", epochs=40, early_stopping=15),
}


def _done(csv: Path, **key) -> bool:
    if not csv.exists():
        return False
    df = pd.read_csv(csv)
    if not set(key) <= set(df.columns):
        return False
    m = pd.Series(True, index=df.index)
    for k, v in key.items():
        m &= df[k].astype(str) == str(v)
    return bool(m.any())


def _append(csv: Path, row: dict) -> None:
    csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if csv.exists():
        df = pd.concat([pd.read_csv(csv), df], ignore_index=True)
    df.to_csv(csv, index=False)


def run_baseline_accumulation(n_seeds: int = 20, device: str = "cuda", out_dir=None):
    """Baseline configuration with gradient accumulation (Section 4.3): ResUNet++, HU/255,
    BCE, batch 10 as micro-batches 4 + 4 + 2, 20 epochs, final epoch, threshold 0.5."""
    out = Path(out_dir or paths.RUNS)
    csv, hist_file = out / "baseline_accumulation.csv", out / "baseline_accumulation_histories.json"
    histories = json.loads(hist_file.read_text()) if hist_file.exists() else {}
    for seed in range(n_seeds):
        if _done(csv, seed=seed):
            continue
        row, hist, _, _ = train_run(RunConfig(), seed, device)
        _append(csv, row)
        histories[str(seed)] = hist
        hist_file.write_text(json.dumps(histories))
        print(f"[baseline seed {seed}] held-out Dice {row['heldout_dice']:.4f} ({row['secs']} s)")
    return csv


def run_progressive_protocol(n_seeds: int = 12, device: str = "cuda", out_dir=None):
    """Progressive protocol, arms A-D on the same seeds (Section 4.4, Table 3)."""
    csv = Path(out_dir or paths.RUNS) / "progressive_protocol.csv"
    for arm, cfg in PROGRESSIVE_ARMS.items():
        for seed in range(n_seeds):
            if _done(csv, arm=arm, seed=seed):
                continue
            row, _, _, _ = train_run(cfg, seed, device)
            _append(csv, dict(arm=arm, **row))
            print(f"[{arm} seed {seed}] held-out Dice {row['heldout_dice']:.4f} ({row['secs']} s)")
    return csv


def run_architecture_benchmark(n_seeds: int = 5, archs=ARCHS, device: str = "cuda", out_dir=None):
    """Architecture benchmark under the stabilized protocol (Section 4.5, Table 5).
    Also writes the per-image test Dice of every run to per_image_dice/."""
    out = Path(out_dir or paths.RUNS)
    csv = out / "architecture_benchmark.csv"
    for arch in archs:
        for seed in range(n_seeds):
            if _done(csv, arch=arch, seed=seed):
                continue
            row, _, per_image, _ = train_run(stabilized_config(arch), seed, device)
            _append(csv, row)
            (out / "per_image_dice").mkdir(parents=True, exist_ok=True)
            per_image.assign(arch=arch, seed=seed).to_csv(out / "per_image_dice" / f"{arch}_seed{seed}.csv", index=False)
            print(f"[{arch} seed {seed}] held-out Dice {row['heldout_dice']:.4f} ({row['secs']} s)")
    return csv


def run_split_sensitivity(n_seeds: int = 6, archs=("unet", "resunetpp"), z_blocks: int = 4,
                          device: str = "cuda", out_dir=None):
    """Split sensitivity (Section 4.6) under the stabilized protocol:
    random_slice  one 80/20 split per seed
    z_blocked     each of the z_blocks position blocks held out once per seed
    patient_lopo  each patient held out once (one fold per patient)"""
    csv = Path(out_dir or paths.RUNS) / "split_sensitivity.csv"
    n_patients = load_index().patient_id.nunique()
    jobs = []
    for arch in archs:
        jobs += [(arch, "random_slice", s, 0) for s in range(n_seeds)]
        jobs += [(arch, "z_blocked", s, f) for s in range(n_seeds) for f in range(z_blocks)]
        jobs += [(arch, "patient_lopo", p, 0) for p in range(n_patients)]
    for arch, scheme, seed, fold in jobs:
        if _done(csv, arch=arch, split=scheme, seed=seed, z_fold=fold):
            continue
        cfg = stabilized_config(arch, split=scheme, z_blocks=z_blocks, z_fold=fold)
        row, _, _, _ = train_run(cfg, seed, device)
        _append(csv, row)
        print(f"[{arch} {scheme} seed {seed} fold {fold}] held-out Dice {row['heldout_dice']:.4f}")
    return csv
