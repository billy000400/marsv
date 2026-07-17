"""Plot fresh char + BPE training dynamics (train/val loss + next-token acc vs step)
from the running training logs. Read-only on logs; headless Agg PNG.

Motivation: the Figure-9 grokking gate needs models trained well past ordinary
convergence. This figure shows what the two fresh runs are actually doing during
training (memorisation vs generalisation), independent of the (still-running) LC/PGD
gate evaluation. Grokking would show val loss DROPPING long after train loss saturates;
overfitting shows val loss RISING. Parsed line format:
  [grok_char] step  16750 lr 4.70e-04 train 0.711 val 1.979 acc 0.556 [50.0m]
"""
import re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(
    r"step\s+(\d+)\s+lr\s+[\d.eE+-]+\s+train\s+([\d.]+)\s+val\s+([\d.]+)\s+acc\s+([\d.]+)"
)

def parse(path):
    steps, tr, va, ac = [], [], [], []
    try:
        for ln in open(path):
            m = LINE.search(ln)
            if m:
                steps.append(int(m.group(1)))
                tr.append(float(m.group(2)))
                va.append(float(m.group(3)))
                ac.append(float(m.group(4)))
    except FileNotFoundError:
        pass
    return steps, tr, va, ac

runs = [
    ("char", "results/train_grok_char.log", "12-layer char GPT (vocab 65)"),
    ("bpe", "results/train_grok_bpe.log", "12-layer BPE GPT (vocab 50257)"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (tag, path, title) in zip(axes, runs):
    steps, tr, va, ac = parse(path)
    if not steps:
        ax.set_title(f"{title}\n(no log yet)")
        continue
    ax.plot(steps, tr, color="C0", label="train loss")
    ax.plot(steps, va, color="C3", label="val loss")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title(f"{title}\nlast step {steps[-1]}: train {tr[-1]:.2f} / val {va[-1]:.2f} / acc {ac[-1]:.2f}")
    ax2 = ax.twinx()
    ax2.plot(steps, ac, color="C2", ls="--", label="val next-token acc")
    ax2.set_ylabel("val next-token accuracy", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    # merge legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("Fresh Grokking-horizon runs — training dynamics (IN PROGRESS, budget-capped 30k schedule)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("plots/fresh_training_dynamics.png", dpi=110)
plt.close(fig)
print("saved plots/fresh_training_dynamics.png")
for tag, path, _ in runs:
    steps, tr, va, ac = parse(path)
    if steps:
        print(f"{tag}: {len(steps)} pts, last step {steps[-1]}, "
              f"val {va[0]:.2f}->{va[-1]:.2f}, minval {min(va):.2f}@{steps[va.index(min(va))]}, "
              f"acc {ac[0]:.2f}->{ac[-1]:.2f}")
