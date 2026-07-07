"""Render the Qwen AE-study figures into ../plots/ from ae_study/results/*.json.
Headless Agg, savefig+close. No recompute."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
PLOTS = os.path.join(os.path.dirname(HERE), "plots")
os.makedirs(PLOTS, exist_ok=True)


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), dpi=130, bbox_inches="tight")
    plt.close(fig); print("wrote", name)


# ---- Fig A: Qwen layer-2 & layer-10 reconstruction vs k (the elbow figure) ----
L2 = load("qwen_sweep_L2.json")
L10 = load("qwen_sweep_L10.json")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for d, lab, col in [(L2, "Qwen L2 (last-tok)", "C0"), (L10, "Qwen L10 (last-tok)", "C2")]:
    if d is None:
        continue
    ks = [r["k"] for r in d["rows"]]
    axes[0].plot(ks, [r["val_fvu"] for r in d["rows"]], "o-", color=col, label=lab)
    axes[1].plot(ks, [r["val_rel_l2"] for r in d["rows"]], "o-", color=col, label=lab)
    axes[2].plot(ks, [r["val_cos"] for r in d["rows"]], "o-", color=col, label=lab)
axes[0].set_ylabel("held-out FVU (lower better)"); axes[0].set_title("FVU vs k")
axes[1].set_ylabel("held-out rel-L2 error (lower better)"); axes[1].set_title("Reconstruction error vs k")
axes[2].set_ylabel("held-out cosine (higher better)"); axes[2].set_title("Cosine similarity vs k")
for ax in axes:
    ax.set_xlabel("bottleneck k"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Qwen3-1.7B last-token AE bottleneck sweep (deep AE 2048-4096-4096-2048-k, 4000 steps)")
fig.tight_layout()
save(fig, "qwen_ae_sweep.png")

# ---- Fig B: controlled experiment — last-token vs pooled (Qwen L2) ----
pooled = load("qwen_sweep_L2_pooled.json")
inj = load("qwen_sweep_L2_inject.json")
if pooled or inj:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if L2:
        ax.plot([r["k"] for r in L2["rows"]], [r["val_fvu"] for r in L2["rows"]],
                "o-", color="C0", label=f"last-token (top1_var={L2['top1_var_frac']:.3f})")
    if pooled:
        ax.plot([r["k"] for r in pooled["rows"]], [r["val_fvu"] for r in pooled["rows"]],
                "s-", color="C3", label=f"all-token pooled (top1_var={pooled['top1_var_frac']:.3f})")
    if inj:
        ax.plot([r["k"] for r in inj["rows"]], [r["val_fvu"] for r in inj["rows"]],
                "^--", color="C1", label=f"last-token + injected massive dim (top1_var={inj['top1_var_frac']:.3f})")
    ax.set_xlabel("bottleneck k"); ax.set_ylabel("held-out FVU")
    ax.set_title("Controlled experiment (Qwen L2): what a massive-activation dim does to the AE curve")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "qwen_ae_controlled.png")


if __name__ == "__main__":
    pass
