"""Shared helpers: config loading, paths, seeding, IO."""
from __future__ import annotations
import os
import random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str | None = None) -> dict:
    import yaml
    # LUG_CONFIG selects an alternative dataset config (e.g. config_covid.yaml)
    path = path or os.path.join(HERE, os.environ.get("LUG_CONFIG", "config.yaml"))
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    # resolve every paths.* entry relative to 05_code/
    cfg["paths"] = {k: os.path.normpath(os.path.join(HERE, v)) for k, v in cfg["paths"].items()}
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for k in ("outputs", "figures", "tables", "logs", "checkpoints"):
        os.makedirs(cfg["paths"][k], exist_ok=True)


def set_seed(seed: int | None) -> None:
    """Faithful arm uses seed=None (matches the released code, which sets no split seed)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_tif(path: str) -> np.ndarray:
    import tifffile
    return tifffile.imread(path)
