"""
06_main_analysis / train_faithful.py
-------------------------------------------------------------------------------
O1 -- reproduce the released training run exactly and report, side by side:
  * the TRAINING-row metrics (what the prior Table 1 actually reports), and
  * the true HELD-OUT metrics (thresholded Dice/IoU/accuracy on the val split).

Released recipe (cells [13]-[16]):
  loss = BCE ; optimiser = Adam(lr 1e-3) ; batch 10 ; 20 epochs ;
  callbacks = ReduceLROnPlateau(monitor val_loss, factor 0.2, patience 10, min_lr 1e-5) ;
  no EarlyStopping ; no checkpoint -> weights = last epoch ;
  monitored metric = SOFT dice_coef (no threshold), cell [11].

Because the released split is unseeded, we repeat the whole run --runs times
(default 20, matching pre-registered H1d) with seeds 0..runs-1 for the split and
init, plus one extra truly-unseeded run for reference.

Outputs
  07_analysis_outputs/tables/faithful_runs.csv        one row per run
  07_analysis_outputs/tables/faithful_history_seed<k>.csv   per-epoch curves
  07_analysis_outputs/logs/faithful_train.log
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_config, ensure_dirs, set_seed  # noqa: E402
from model_resunetpp import ResUNetPP  # noqa: E402
from data_faithful import get_faithful_split  # noqa: E402

EPS = 1e-7


def soft_dice(pred, tgt):
    p = pred.reshape(-1); t = tgt.reshape(-1)
    return ((2 * (p * t).sum() + EPS) / (p.sum() + t.sum() + EPS)).item()


def hard_metrics(pred, tgt, thr=0.5):
    p = (pred > thr).float().reshape(-1); t = (tgt > 0.5).float().reshape(-1)
    inter = (p * t).sum()
    dice = (2 * inter + EPS) / (p.sum() + t.sum() + EPS)
    iou = (inter + EPS) / ((p + t - p * t).sum() + EPS)
    acc = (p == t).float().mean()
    return dict(dice=dice.item(), iou=iou.item(), acc=acc.item())


@torch.no_grad()
def evaluate(model, x, y, device, bs=16):
    model.eval()
    preds = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(device)
        preds.append(model(xb).cpu())
    pred = torch.cat(preds); tgt = torch.from_numpy(y)
    bce = nn.functional.binary_cross_entropy(pred, tgt).item()
    return pred, tgt, bce


def run_once(cfg, seed, device, log):
    fp = cfg["faithful_pipeline"]
    set_seed(seed)
    d = get_faithful_split(cfg, seed=seed)
    xtr, ytr = d["x_train"], d["y_train"]
    xva, yva = d["x_val"], d["y_val"]

    model = ResUNetPP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=fp["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.2, patience=10, min_lr=1e-5)
    lossf = nn.BCELoss()
    bs = fp["batch_size"]

    hist = []
    idx = np.arange(len(xtr))
    for ep in range(fp["epochs"]):
        model.train()
        np.random.shuffle(idx)
        tr_loss = tr_dice = nb = 0.0
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            xb = torch.from_numpy(xtr[j]).to(device)
            yb = torch.from_numpy(ytr[j]).to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item(); tr_dice += soft_dice(out.detach(), yb); nb += 1
        tr_loss /= nb; tr_dice /= nb

        pred, tgt, va_loss = evaluate(model, xva, yva, device)
        va_soft = soft_dice(pred, tgt)
        va_hard = hard_metrics(pred, tgt)
        sched.step(va_loss)
        lr_now = opt.param_groups[0]["lr"]
        hist.append(dict(epoch=ep + 1, lr=lr_now,
                         train_loss=tr_loss, train_soft_dice=tr_dice,
                         val_loss=va_loss, val_soft_dice=va_soft,
                         val_dice_thr=va_hard["dice"], val_iou_thr=va_hard["iou"],
                         val_acc_thr=va_hard["acc"]))
        log(f"  seed {seed} ep {ep+1:2d}/20  "
            f"train_loss {tr_loss:.4f} train_softDice {tr_dice:.4f} | "
            f"val_loss {va_loss:.4f} val_softDice {va_soft:.4f} "
            f"val_Dice@0.5 {va_hard['dice']:.4f} val_acc {va_hard['acc']:.4f}  lr {lr_now:.0e}")

    h = pd.DataFrame(hist)
    tdir = cfg["paths"]["tables"]
    h.to_csv(os.path.join(tdir, f"faithful_history_seed{seed}.csv"), index=False)

    last = hist[-1]
    return dict(
        seed=seed, n_train=len(xtr), n_val=len(xva),
        # what the prior Table 1 reports (final-epoch TRAINING row):
        reported_like_train_softDice=last["train_soft_dice"],
        reported_like_train_loss=last["train_loss"],
        # true held-out (final epoch = released model-selection):
        heldout_softDice=last["val_soft_dice"],
        heldout_Dice_thr0p5=last["val_dice_thr"],
        heldout_IoU_thr0p5=last["val_iou_thr"],
        heldout_acc_thr0p5=last["val_acc_thr"],
        heldout_loss=last["val_loss"],
        best_heldout_softDice=max(x["val_soft_dice"] for x in hist),
        best_heldout_Dice_thr=max(x["val_dice_thr"] for x in hist),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--start-seed", type=int, default=0, help="first seed (for resuming/extending)")
    ap.add_argument("--epochs", type=int, default=None, help="override for a smoke test")
    ap.add_argument("--out", type=str, default=None, help="override output CSV name")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    if args.epochs:
        cfg["faithful_pipeline"]["epochs"] = args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logf = open(os.path.join(cfg["paths"]["logs"], "faithful_train.log"), "a")

    def log(m):
        print(m); logf.write(m + "\n"); logf.flush()

    log(f"\n=== faithful reproduction | {time.strftime('%Y-%m-%d %H:%M')} | "
        f"device={device} | runs={args.runs} | epochs={cfg['faithful_pipeline']['epochs']} ===")
    n = sum(p.numel() for p in ResUNetPP().parameters())
    log(f"ResUNetPP trainable params: {n:,}  (Keras released: 4,085,873)")

    t0 = time.time()
    seeds = list(range(args.start_seed, args.start_seed + args.runs))
    out = os.path.join(cfg["paths"]["tables"], args.out or "faithful_runs.csv")
    for i, k in enumerate(seeds):
        log(f"[run {i+1}/{len(seeds)}] seed={k}")
        r = run_once(cfg, k, device, log)
        prev = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
        pd.concat([prev, pd.DataFrame([r])], ignore_index=True) \
          .drop_duplicates(subset="seed", keep="last").to_csv(out, index=False)
    df = pd.read_csv(out)

    log("\n--- SUMMARY over runs (mean +/- SD) ---")
    for col in ["reported_like_train_softDice", "heldout_softDice",
                "heldout_Dice_thr0p5", "heldout_IoU_thr0p5", "heldout_acc_thr0p5",
                "best_heldout_Dice_thr"]:
        log(f"  {col:32s} {df[col].mean():.4f} +/- {df[col].std():.4f}  "
            f"[min {df[col].min():.4f}, max {df[col].max():.4f}]")
    log(f"\nPrior Table 1 claim: ResUNet++ DSC 0.9738 / acc 0.9599 (labelled 'validation').")
    log(f"wrote {out}   ({time.time()-t0:.0f}s total)")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
