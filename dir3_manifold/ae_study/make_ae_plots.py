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


def kneedle_log2(ks, ys):
    x = np.log2(np.asarray(ks, float)); y = np.asarray(ys, float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    if yn[-1] < yn[0]:
        yn = 1 - yn
    d = np.abs(yn - xn)
    return int(ks[int(np.argmax(d))]), float(d.max())


# ---- Fig A: faithful reproduction — Qwen L2 & L10 last-token, colleague k range ----
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
fig.suptitle("Faithful reproduction of colleague's setup: Qwen3-1.7B last-token AE sweep "
             "(deep AE 2048-4096-4096-2048-k, 4000 steps) — smooth decline, no plateau")
fig.tight_layout()
save(fig, "qwen_ae_sweep.png")


# ---- Fig B: DECISIVE controlled experiment — wide k, isotropic vs injected-massive ----
base = load("qwen_sweep_L2_wide.json")
inj = load("qwen_sweep_L2_wide_inject.json")
pool = load("qwen_sweep_L2_pooled_wide.json")
if base and inj:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for d, lab, col, mk in [
        (base, "Qwen L2 baseline (isotropic, top1={:.3f})", "C0", "o-"),
        (pool, "Qwen L2 all-token pooled (top1={:.3f})", "C2", "s-"),
        (inj, "Qwen L2 + injected massive dim (top1={:.2f})", "C3", "^--"),
    ]:
        if d is None:
            continue
        ks = [r["k"] for r in d["rows"]]
        fv = [r["val_fvu"] for r in d["rows"]]
        kk, contrast = kneedle_log2(ks, fv)
        ax.plot(ks, fv, mk, color=col, label=lab.format(d["top1_var_frac"]) +
                f"  [Kneedle k*={kk}, contrast={contrast:.2f}]")
        j = ks.index(kk)
        ax.scatter([kk], [fv[j]], s=170, facecolors="none", edgecolors=col, linewidths=2, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("bottleneck k (log2 scale)")
    ax.set_ylabel("held-out FVU (lower = better reconstruction)")
    ax.set_title("When does an AE reconstruction elbow appear?\n"
                 "Only concentrated variance yields a sharp low-k knee (○ = Kneedle knee)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="upper right")
    save(fig, "qwen_ae_wide_controlled.png")


# ---- Fig C: anisotropy diagnostic — why (GPT-2 vs Qwen) ----
labels = ["GPT-2 L6\n(all-token)", "Qwen L2\n(last-tok)", "Qwen L10\n(last-tok)"]
top1 = [0.904, 0.034, 0.145]      # top PCA eigenvalue fraction
pr = [1.22, 245.4, 41.9]          # participation ratio
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
bars = a1.bar(labels, top1, color=["C3", "C0", "C2"])
a1.set_ylabel("top-1 PCA eigenvalue fraction")
a1.set_title("Variance concentration (higher = more anisotropic)")
a1.axhline(0.5, color="k", ls=":", lw=1)
for b, v in zip(bars, top1):
    a1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
bars = a2.bar(labels, pr, color=["C3", "C0", "C2"])
a2.set_ylabel("participation ratio (effective # directions)")
a2.set_title("Effective dimensionality (log scale)")
a2.set_yscale("log")
for b, v in zip(bars, pr):
    a2.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:.0f}", ha="center", fontsize=9)
fig.suptitle("Why GPT-2 shows a low-k bend but Qwen last-token does not: "
             "GPT-2 L6 puts 90% of variance in one direction; Qwen spreads it over 42-245")
fig.tight_layout()
save(fig, "qwen_anisotropy.png")


# ---- Fig D: reproduce the colleague's plot (lasse.png) — wide k to 500, linear x ----
# Two panels: cosine similarity (higher=better) and relative L2 error (lower=better)
# vs bottleneck dimension, matching the colleague's axes/range, so the reconstruction
# minimum / cosine peak at intermediate k is directly comparable.
lasse = load("qwen_sweep_L2_lasse.json")
if lasse:
    rows = lasse["rows"]
    ks = [r["k"] for r in rows]
    cos = [r["val_cos"] for r in rows]
    rel = [r["val_rel_l2"] for r in rows]
    k_cos = ks[int(np.argmax(cos))]           # best (peak) cosine
    k_rel = ks[int(np.argmin(rel))]           # best (min) rel-L2 error
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.plot(ks, cos, "o-", color="C0", label="Autoencoder (67M)")
    a1.scatter([k_cos], [max(cos)], s=170, facecolors="none", edgecolors="C3",
               linewidths=2, zorder=5, label=f"peak at k={k_cos}")
    a1.set_ylabel("Cosine similarity"); a1.set_xlabel("Bottleneck dimension")
    a1.set_title("Reconstruction quality vs bottleneck")
    a2.plot(ks, rel, "o-", color="C0", label="held-out (val)")
    if all("train_rel_l2" in r for r in rows):
        a2.plot(ks, [r["train_rel_l2"] for r in rows], "s--", color="C7",
                label="train", alpha=0.8)
    a2.scatter([k_rel], [min(rel)], s=170, facecolors="none", edgecolors="C3",
               linewidths=2, zorder=5, label=f"val min at k={k_rel}")
    a2.set_ylabel("Relative L2 error"); a2.set_xlabel("Bottleneck dimension")
    a2.set_title("Reconstruction error vs bottleneck (val vs train)")
    for ax in (a1, a2):
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("Reproduction of the colleague's Qwen3-1.7B AE sweep (last-token L2, k up to 500, "
                 f"{lasse['n_steps']} steps): quality peaks / error bottoms at intermediate k, then degrades")
    fig.tight_layout()
    save(fig, "qwen_ae_lasse_repro.png")


if __name__ == "__main__":
    pass
