"""Literal-batch-10 baseline (article Sections 4.2-4.3, Figures 4-5).

ResUNet++, raw HU / 255, BCE, Adam (lr 1e-3), a real batch of 10 slices per optimizer
step (no accumulation), 20 epochs, ReduceLROnPlateau (factor 0.2, patience 10,
min_lr 1e-5) on the validation loss, no early stopping. For a given seed the
train/validation/test slices are the same as in scripts/run_experiments.py.

Reported per run (test split, threshold 0.5):
    heldout_dice              final-epoch weights
    heldout_dice_best_epoch   weights of the epoch with the best validation soft Dice

Outputs:
    results/runs/literal_batch10.csv                  one row per seed
    results/runs/literal_batch10_history_seed<k>.csv  per-epoch validation and test curves

Usage (from the repository root):
    python scripts/train_literal_batch10.py                         # 8 seeds
    python scripts/train_literal_batch10.py --seeds 1 --epochs 2    # quick test
On a 4 GB GPU set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.8
(scripts/train_literal_batch10.ps1 does this on Windows).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lungct import data, metrics, paths  # noqa: E402
from lungct.models import ResUNetPP, count_parameters  # noqa: E402
from lungct.training import predict  # noqa: E402  (also enables deterministic training)

BATCH, EPOCHS, LR = 10, 20, 1e-3


def evaluate(model, x, y, device):
    prob = predict(model, x, device)
    target = torch.from_numpy(y[:, None])
    m = metrics.thresholded_metrics(prob, target, 0.5)
    return dict(loss=nn.functional.binary_cross_entropy(prob, target).item(),
                soft_dice=metrics.soft_dice(prob, target), dice=m["dice"], iou=m["iou"], acc=m["acc"])


def run_seed(seed: int, epochs: int, device: str) -> dict:
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    df, x, y = data.load_dataset("hu255")
    train, test = data.split_indices(df, "random_slice", seed)
    train, val = data.validation_split(train, seed)

    model = ResUNetPP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.2, patience=10, min_lr=1e-5)
    loss_fn = nn.BCELoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train)
        losses = []
        for s in range(0, len(order), BATCH):
            j = np.sort(order[s:s + BATCH])
            xb = torch.from_numpy(x[j][:, None]).to(device)
            yb = torch.from_numpy(y[j][:, None]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        v = evaluate(model, x[val], y[val], device)
        sched.step(v["loss"])
        t = evaluate(model, x[test], y[test], device)
        history.append(dict(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=float(np.mean(losses)),
                            **{f"val_{k}": val_ for k, val_ in v.items()},
                            **{f"test_{k}": val_ for k, val_ in t.items()}))
        print(f"  seed {seed} epoch {epoch:2d}/{epochs}  train loss {np.mean(losses):.4f} | "
              f"val loss {v['loss']:.4f}  val Dice {v['dice']:.4f} | test Dice {t['dice']:.4f}", flush=True)

    h = pd.DataFrame(history)
    paths.RUNS.mkdir(parents=True, exist_ok=True)
    h.to_csv(paths.RUNS / f"literal_batch10_history_seed{seed}.csv", index=False)
    last, best = h.iloc[-1], h.loc[h.val_soft_dice.idxmax()]
    return dict(seed=seed, n_train=len(train), n_val=len(val), n_test=len(test), epochs=epochs,
                final_train_loss=last.train_loss,
                heldout_soft_dice=last.test_soft_dice, heldout_dice=last.test_dice,
                heldout_iou=last.test_iou, heldout_acc=last.test_acc, heldout_loss=last.test_loss,
                best_epoch=int(best.epoch), heldout_dice_best_epoch=best.test_dice,
                secs=round(time.time() - t0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8, help="number of seeds (0..seeds-1)")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default="literal_batch10.csv", help="output file name in results/runs/")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ResUNet++ parameters: {count_parameters(ResUNetPP()):,}  device: {device}")

    out = paths.RUNS / args.out
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        row = run_seed(seed, args.epochs, device)
        prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
        pd.concat([prev, pd.DataFrame([row])], ignore_index=True).drop_duplicates("seed", keep="last") \
          .sort_values("seed").to_csv(out, index=False)
        print(f"[seed {seed}] held-out Dice: final epoch {row['heldout_dice']:.4f}, "
              f"best epoch {row['heldout_dice_best_epoch']:.4f} ({row['secs']} s)")
    df = pd.read_csv(out)
    print(f"\nheld-out Dice, final epoch: {df.heldout_dice.mean():.4f} +/- {df.heldout_dice.std():.4f} "
          f"(n={len(df)}); written to {out.relative_to(paths.REPO)}")


if __name__ == "__main__":
    main()
