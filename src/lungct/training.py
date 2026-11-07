"""Training and evaluation of one run (Algorithm 1 of the article).

For every run the slices are split into train / validation / test (seeded). The
validation split drives the learning-rate schedule, the monitored soft Dice, best-epoch
selection, early stopping and threshold tuning; the test split is used once, for the
reported held-out metrics. Gradient accumulation makes every optimizer step use exactly
`batch` slices, processed as micro-batches of at most `micro_batch` slices (BatchNorm
statistics are computed per micro-batch). Training is deterministic.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import data, metrics
from .models import build_model

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


@dataclass
class RunConfig:
    arch: str = "resunetpp"            # unet | attn_unet | resunet | resunetpp
    intensity: str = "hu255"           # hu255 | lung_window
    augmentation: bool = False
    loss: str = "bce"                  # bce | bce_dice
    epochs: int = 20
    batch: int = 10                    # slices per optimizer step
    micro_batch: int = 4               # slices per forward pass (4 GB GPU)
    lr: float = 1e-3
    scheduler: str = "plateau"         # plateau (on validation loss) | cosine
    early_stopping: int = 0            # patience in epochs; 0 = off
    model_selection: str = "last"      # last | best (validation soft Dice)
    threshold: str = "fixed"           # fixed (0.5) | tuned (on validation)
    split: str = "random_slice"        # random_slice | z_blocked | patient_lopo
    z_blocks: int = 4
    z_fold: int = 0
    size: int = data.IMAGE_SIZE


def stabilized_config(arch: str, **overrides) -> RunConfig:
    """Stabilized protocol of the architecture benchmark (article Section 4.5)."""
    cfg = RunConfig(arch=arch, intensity="lung_window", augmentation=True, loss="bce_dice",
                    epochs=60, batch=8, scheduler="cosine", early_stopping=15,
                    model_selection="best", threshold="tuned")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _loss_fn(kind: str):
    bce = nn.BCELoss()
    if kind == "bce":
        return bce

    def bce_dice(prob, target):
        dice = (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
        return bce(prob, target) + 1 - dice
    return bce_dice


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: str, batch: int = 8) -> torch.Tensor:
    model.eval()
    out = [model(torch.from_numpy(x[i:i + batch][:, None]).to(device)).cpu() for i in range(0, len(x), batch)]
    return torch.cat(out)


def _optimizer_steps(dataset, batch: int, micro_batch: int):
    """Shuffle once per epoch and yield optimizer steps of `batch` slices, each a list of
    (x, y, weight) micro-batches; weights make the accumulated gradient the mean loss."""
    order = torch.randperm(len(dataset)).tolist()
    for s in range(0, len(order), batch):
        step = order[s:s + batch]
        out = []
        for m in range(0, len(step), micro_batch):
            items = [dataset[j] for j in step[m:m + micro_batch]]
            out.append((torch.stack([i[0] for i in items]), torch.stack([i[1] for i in items]),
                        len(items) / len(step)))
        yield out


def train_run(cfg: RunConfig, seed: int, device: str = "cuda", data_dir=None, verbose: bool = False):
    """Train and evaluate one run.

    Returns (row, history, per_image, test_prob): a dict of configuration and held-out
    metrics, the per-epoch history, a DataFrame of per-image test Dice, and the test
    probability maps.
    """
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    df, x, y = data.load_dataset(cfg.intensity, cfg.size, data_dir)
    train, test = data.split_indices(df, cfg.split, seed, cfg.z_blocks, cfg.z_fold)
    train, val = data.validation_split(train, seed)

    dataset = data.SliceDataset(x[train], y[train], cfg.augmentation, seed)
    micro = min(cfg.batch, cfg.micro_batch)
    model = build_model(cfg.arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.2, patience=10, min_lr=1e-5)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = _loss_fn(cfg.loss)
    y_val = torch.from_numpy(y[val][:, None])

    history, best_state, best_val, stale = [], None, -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        for step in _optimizer_steps(dataset, cfg.batch, micro):
            opt.zero_grad()
            for xb, yb, w in step:
                loss = loss_fn(model(xb.to(device)), yb.to(device))
                (loss * w).backward()
                train_loss += loss.item(); n += 1
            opt.step()
        p_val = predict(model, x[val], device)
        val_loss = nn.functional.binary_cross_entropy(p_val, y_val).item()
        val_soft_dice = metrics.soft_dice(p_val, y_val)
        if cfg.scheduler == "plateau":
            sched.step(val_loss)
        else:
            sched.step()
        history.append(dict(epoch=epoch, train_loss=train_loss / n, val_loss=val_loss,
                            val_soft_dice=val_soft_dice, lr=opt.param_groups[0]["lr"]))
        if verbose:
            print(f"  epoch {epoch:3d}  train loss {train_loss / n:.4f}  "
                  f"val loss {val_loss:.4f}  val soft Dice {val_soft_dice:.4f}")
        if val_soft_dice > best_val:
            best_val, stale = val_soft_dice, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if cfg.early_stopping and stale >= cfg.early_stopping:
                break

    if cfg.model_selection == "best":
        model.load_state_dict(best_state)

    threshold = 0.5
    if cfg.threshold == "tuned":
        p_val = predict(model, x[val], device)
        threshold = float(max(THRESHOLD_GRID, key=lambda t: metrics.thresholded_metrics(p_val, y_val, t)["dice"]))

    p_test = predict(model, x[test], device)
    y_test = torch.from_numpy(y[test][:, None])
    m = metrics.thresholded_metrics(p_test, y_test, threshold)
    pid = metrics.per_image_dice(p_test, y_test, threshold)
    ece, brier = metrics.calibration(p_test, y_test)

    row = dict(**asdict(cfg), seed=seed, n_train=len(train), n_val=len(val), n_test=len(test),
               epochs_run=len(history), best_epoch=int(np.argmax([h["val_soft_dice"] for h in history]) + 1),
               threshold_used=threshold, final_train_loss=history[-1]["train_loss"],
               best_val_soft_dice=best_val,
               heldout_soft_dice=metrics.soft_dice(p_test, y_test),
               heldout_dice=m["dice"], heldout_iou=m["iou"], heldout_acc=m["acc"],
               heldout_sens=m["sens"], heldout_spec=m["spec"],
               heldout_dice_median=float(np.median(pid)),
               heldout_frac_empty_pred=float(np.mean(pid == 0)),
               heldout_ece=ece, heldout_brier=brier,
               test_patients=";".join(sorted(df.patient_id.iloc[test].unique())),
               secs=round(time.time() - t0))
    per_image = pd.DataFrame(dict(img_id=df.img_id.iloc[test].to_numpy(), dice=pid))
    return row, history, per_image, p_test[:, 0].numpy()
