"""Redraw plots/training_curves.png from the saved pilot-run history (no retraining).

Same panels as train.py wrote originally; only the colour encoding changed, to satisfy the
CVD rule (green-free palette, every series also carries a linestyle/marker).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvd_style import CVD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hist = json.load(open(os.path.join(ROOT, "results", "train_hist.json")))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(hist["step"], hist["train_loss"], color=CVD[0], ls="-", lw=1.6, alpha=0.8,
           label="train (solid)")
ax[0].plot(hist["step"], hist["val_loss"], color=CVD[1], ls="--", marker="s", ms=4, lw=2,
           label="validation (dashed, squares)")
ax[0].set_xlabel("training step"); ax[0].set_ylabel("cross-entropy loss (nats)")
ax[0].set_title("Training curves — 12L/12H char GPT"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(hist["step"], hist["val_acc"], color=CVD[0], ls="-", marker="o", ms=4, lw=2)
ax[1].set_xlabel("training step"); ax[1].set_ylabel("validation next-char accuracy")
ax[1].set_title(f"Final val acc = {hist['val_acc'][-1]:.3f}"); ax[1].grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(ROOT, "plots", "training_curves.png")
fig.savefig(out, dpi=120)
plt.close(fig)
print("wrote", out)
