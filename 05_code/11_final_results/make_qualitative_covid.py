"""
Qualitative-figure inputs for the COVID-19-CT-Seg re-run: train one corrected-protocol
U-Net (same Cfg as the O3 benchmark, seed 0) and keep its held-out predictions.
train_one() does not return the model, so the final held-out inference call is captured
by wrapping lug_method._infer (the last call inside train_one is the held-out set).

Writes 07_analysis_outputs/covid19_ct_seg/results/qualitative/corrected_unet.npz
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "colab"))
import lug_method as L  # noqa: E402

BASE = os.path.normpath(os.path.join(HERE, "..", ".."))
ROOT = os.path.join(BASE, "00_inputs", "covid19_ct_seg")
OUT = os.path.join(BASE, "07_analysis_outputs", "covid19_ct_seg", "results", "qualitative")
os.makedirs(OUT, exist_ok=True)

calls = []
_orig = L._infer
def _capture(model, x, device, bs=None):
    out = _orig(model, x, device, bs)
    calls.append(out.numpy()[:, 0])
    return out
L._infer = _capture

seed = 0
cfg = L.Cfg(arm="corrected", arch="unet", norm="hu_window_minmax", aug="full", loss="bce_dice",
            epochs=60, batch=8, sched="cosine", early_stop=15, select="best_val", thr="tuned",
            split="random_slice")
row, _, pid = L.train_one(ROOT, cfg, seed=seed, device="cuda")
df, hu, mk = L._load_raw(ROOT, cfg.size)
_, va = L._split_indices(df, cfg, seed)
pred = calls[-1]
assert len(pred) == len(va)
np.savez_compressed(os.path.join(OUT, "corrected_unet.npz"),
                    ids=df.iloc[va]["img_id"].to_numpy(), img=hu[va], gt=mk[va], pred=pred,
                    thr=row["thr_used"], dice=pid["dice"].to_numpy())
print(f"held-out Dice {row['heldout_dice']:.4f} (O3 seed 0 for comparison), thr {row['thr_used']:.2f}, "
      f"{len(va)} slices -> {OUT}")
