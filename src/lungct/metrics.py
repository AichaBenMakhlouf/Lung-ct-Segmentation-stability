"""Segmentation metrics (pooled over pixels unless stated otherwise)."""
from __future__ import annotations

import numpy as np
import torch

EPS = 1e-7


def soft_dice(prob: torch.Tensor, target: torch.Tensor) -> float:
    """Dice on probabilities (no threshold), pooled over all pixels."""
    p, t = prob.reshape(-1), target.reshape(-1)
    return ((2 * (p * t).sum() + EPS) / (p.sum() + t.sum() + EPS)).item()


def thresholded_metrics(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Dice, IoU, accuracy, sensitivity and specificity after thresholding, pooled over pixels."""
    p = (prob > threshold).float().reshape(-1)
    t = (target > 0.5).float().reshape(-1)
    inter = (p * t).sum()
    return dict(
        dice=((2 * inter + EPS) / (p.sum() + t.sum() + EPS)).item(),
        iou=((inter + EPS) / ((p + t - p * t).sum() + EPS)).item(),
        acc=(p == t).float().mean().item(),
        sens=((inter + EPS) / (t.sum() + EPS)).item(),
        spec=((((1 - p) * (1 - t)).sum() + EPS) / ((1 - t).sum() + EPS)).item(),
    )


def per_image_dice(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Thresholded Dice of every image."""
    p = (prob > threshold).float().flatten(1)
    t = (target > 0.5).float().flatten(1)
    inter = (p * t).sum(1)
    return ((2 * inter + EPS) / (p.sum(1) + t.sum(1) + EPS)).cpu().numpy()


def calibration(prob: torch.Tensor, target: torch.Tensor, bins: int = 15) -> tuple[float, float]:
    """Expected calibration error (equal-width bins) and Brier score."""
    p = prob.reshape(-1).cpu().numpy()
    t = (target.reshape(-1) > 0.5).cpu().numpy().astype(float)
    brier = float(np.mean((p - t) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            ece += m.mean() * abs(t[m].mean() - p[m].mean())
    return float(ece), brier
