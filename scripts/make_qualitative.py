"""Predictions for the qualitative figure (article Figure 12).

Trains one U-Net under the stabilized protocol (seed 0) and saves its test-split
probability maps, the reference masks and the CT slices to
results/runs/qualitative/unet_seed0.npz (not tracked by git: it contains CT data).

Usage (from the repository root):
    python scripts/make_qualitative.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lungct import data, paths  # noqa: E402
from lungct.training import stabilized_config, train_run  # noqa: E402

SEED = 0


def main() -> None:
    cfg = stabilized_config("unet")
    row, _, per_image, prob = train_run(cfg, SEED, device="cuda")
    df, _, masks = data.load_dataset(cfg.intensity, cfg.size)
    _, test = data.split_indices(df, cfg.split, SEED)
    out = paths.RUNS / "qualitative"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "unet_seed0.npz", ids=df.img_id.iloc[test].to_numpy(),
                        img=data.load_hu(cfg.size)[test], gt=masks[test], pred=prob,
                        thr=row["threshold_used"], dice=per_image.dice.to_numpy())
    print(f"held-out Dice {row['heldout_dice']:.4f}, threshold {row['threshold_used']:.2f}, "
          f"{len(test)} slices -> {(out / 'unet_seed0.npz').relative_to(paths.REPO)}")


if __name__ == "__main__":
    main()
