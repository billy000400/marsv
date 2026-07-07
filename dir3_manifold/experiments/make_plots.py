"""Render every reported quantitative result from saved JSON artifacts into plots/.
Pure matplotlib (Agg, headless); reads results/*.json, no GPU / no recompute.
savefig + close only (never show), per CLAUDE.md."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---- Fig 1: nonlinear ID per layer (n=50k) vs linear d95 vs d_model ----
idn = load("id_nonlinear.json")
pca = {r["layer"]: r for r in load("pca_pr.json")}
layers = [0, 3, 6, 9, 11]


def get(layer, prep, key):
    for r in idn:
        if r["layer"] == layer and r["n"] == 50000 and r["prep"] == prep:
            return r[key]
    return None


twonn_c = [get(L, "centered", "twonn") for L in layers]
mle_c = [get(L, "centered", "mle_invavg") for L in layers]
twonn_s = [get(L, "standardized", "twonn") for L in layers]
mle_s = [get(L, "standardized", "mle_invavg") for L in layers]
d95 = [pca[L]["pca_d95"] for L in layers]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(layers, twonn_c, "o-", color="C0", label="TwoNN (centered)")
ax.plot(layers, mle_c, "s-", color="C1", label="MLE (centered)")
ax.plot(layers, twonn_s, "o--", color="C0", alpha=0.5, label="TwoNN (standardized)")
ax.plot(layers, mle_s, "s--", color="C1", alpha=0.5, label="MLE (standardized)")
ax.plot(layers, d95, "^-", color="C3", label="linear PCA d95")
ax.axhline(768, color="k", ls=":", lw=1, label="d_model = 768")
ax.set_yscale("log")
ax.set_xlabel("layer (block output / resid_post)")
ax.set_ylabel("intrinsic dimension (log scale)")
ax.set_title("Nonlinear local ID vs linear d95 vs ambient, GPT-2 small (n=50k)\n"
             "L11 is post-final-layernorm (caveat)")
ax.set_xticks(layers)
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
save(fig, "id_per_layer.png")

# ---- Fig 1b: TwoNN vs MLE ONLY (operator request 2026-07-07) ----
# Same nonlinear-ID data as Fig 1 but WITHOUT linear PCA d95 and the d_model=768
# reference line, and on a linear y-axis, so the reader can judge directly how
# closely the two nonlinear estimators (TwoNN and MLE) agree with each other.
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(layers, twonn_c, "o-", color="C0", label="TwoNN (centered)")
ax.plot(layers, mle_c, "s-", color="C1", label="MLE (centered)")
ax.plot(layers, twonn_s, "o--", color="C0", alpha=0.5, label="TwoNN (standardized)")
ax.plot(layers, mle_s, "s--", color="C1", alpha=0.5, label="MLE (standardized)")
ax.set_xlabel("layer (block output / resid_post)")
ax.set_ylabel("intrinsic dimension (linear scale)")
ax.set_title("Do TwoNN and MLE agree? Nonlinear local ID only, GPT-2 small (n=50k)\n"
             "L11 is post-final-layernorm (caveat)")
ax.set_xticks(layers)
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
save(fig, "id_twonn_vs_mle.png")

# ---- Fig 2: AE FVU bottleneck sweep (the headline AE figure) ----
ks = [2, 4, 8, 16, 24, 32, 48, 64, 128, 256]
cpu = {r["k"]: r["val_fvu"] for r in load("ae_results.json")}
gpu = {r["k"]: r["val_fvu"] for r in load("ae_results_gpu.json")}
v2 = load("ae_results_gpu_v2.json")
matched = {r["k"]: r["val_fvu"] for r in load("ae_results_matched.json")}

# raw seed-mean +/- std across seeds {0,1,2}
raw_by_k = {k: [] for k in ks}
std_by_k = {k: [] for k in ks}
for r in v2:
    if r["prep"] == "raw":
        raw_by_k[r["k"]].append(r["val_fvu"])
    else:
        std_by_k[r["k"]].append(r["val_fvu"])
raw_mean = np.array([np.mean(raw_by_k[k]) for k in ks])
raw_std = np.array([np.std(raw_by_k[k]) for k in ks])
std_mean = np.array([np.mean(std_by_k[k]) for k in ks])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ks, [cpu[k] for k in ks], "o-", color="C7", alpha=0.7,
        label="CPU budget (1200 steps)")
ax.errorbar(ks, raw_mean, yerr=raw_std, fmt="o-", color="C0", capsize=3,
            label="GPU raw, seed-mean (10000 steps)")
ax.plot(ks, [matched[k] for k in ks], "D--", color="C2",
        label="GPU raw, param-matched")
ax.plot(ks, std_mean, "s-", color="C3",
        label="GPU standardized (knee gone)")
ax.axvline(8, color="C0", ls=":", lw=1)
ax.text(8.3, 0.6, "raw bend k≈8", color="C0", fontsize=8)
ax.set_xscale("log", base=2)
ax.set_xticks(ks)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("bottleneck width k (log2)")
ax.set_ylabel("held-out FVU = mean||x-x_hat||^2 / mean||x-mu||^2")
ax.set_title("AE bottleneck sweep, layer 6: raw bend vanishes under standardization")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
save(fig, "ae_fvu_sweep.png")

# ---- Fig 2b: per-doubling marginal gain (no-plateau evidence) ----
def deltas(vals):
    return [vals[i - 1] - vals[i] for i in range(1, len(vals))]
raw_d = deltas(list(raw_mean))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ks[1:], raw_d, "o-", color="C0")
ax.axhline(0, color="k", lw=0.6)
ax.set_xscale("log", base=2)
ax.set_xticks(ks[1:])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("k (log2)")
ax.set_ylabel("ΔFVU per doubling (raw, seed-mean)")
ax.set_title("No plateau: marginal gain stays ~0.006-0.015 out to k=256")
ax.grid(alpha=0.3, which="both")
save(fig, "ae_marginal_gain.png")

# ---- Fig 3: synthetic validation of estimators ----
val = load("id_validation.json")
td = [r["true_d"] for r in val]
fig, ax = plt.subplots(figsize=(5.5, 5))
lim = max(td) + 5
ax.plot([0, lim], [0, lim], "k:", label="ideal (y=x)")
ax.plot(td, [r["twonn"] for r in val], "o-", color="C0", label="TwoNN")
ax.plot(td, [r["mle_invavg"] for r in val], "s-", color="C1", label="MLE")
ax.set_xlabel("true dimension (Gaussian linearly embedded in 768-d)")
ax.set_ylabel("estimated ID")
ax.set_title("Estimator validation on synthetic data (n=20k)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
save(fig, "id_validation.png")

# ---- Fig 4: ID by token-position bucket (layer 6) ----
pos = [r for r in load("id_by_position.json") if r["twonn"] is not None]
labels = [r["bucket"] for r in pos]
x = np.arange(len(pos))
fig, ax = plt.subplots(figsize=(7, 4))
w = 0.38
ax.bar(x - w / 2, [r["twonn"] for r in pos], w, color="C0", label="TwoNN")
ax.bar(x + w / 2, [r["mle_invavg"] for r in pos], w, color="C1", label="MLE")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("intrinsic dimension")
ax.set_title("Layer-6 ID by token-position bucket: low in every bucket")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
save(fig, "id_by_position.png")

# ---- Fig 5: bootstrap CIs + duplicate/self-masking diagnostic ----
diag = load("id_diagnostics.json")
bs = diag["bootstrap"]
dup = diag["duplicate_diagnostic"]
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
# left: bootstrap means with CI
ax = axes[0]
names = ["TwoNN", "MLE"]
means = [bs["twonn_mean"], bs["mle_mean"]]
los = [bs["twonn_ci95"][0], bs["mle_ci95"][0]]
his = [bs["twonn_ci95"][1], bs["mle_ci95"][1]]
err = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
ax.errorbar([0, 1], means, yerr=err, fmt="o", color="C0", capsize=6, ms=8)
ax.set_xlim(-0.5, 1.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(names)
ax.set_ylabel("intrinsic dimension")
ax.set_title(f"Bootstrap 95% CI (B={bs['B']}, n={bs['n_per_sample']})\nsampling CI ~±0.1")
ax.grid(alpha=0.3, axis="y")
# right: naive vs robust (self-masking) for the layer-6 estimate
ax = axes[1]
xx = np.arange(2)
ax.bar(xx - 0.2, [dup["twonn_naive"], dup["mle_naive"]], 0.4, color="C0", label="naive")
ax.bar(xx + 0.2, [dup["twonn_robust"], dup["mle_robust"]], 0.4, color="C2",
       label="robust (self-index masking)")
ax.set_xticks(xx)
ax.set_xticklabels(["TwoNN", "MLE"])
ax.set_ylabel("intrinsic dimension")
ax.set_title(f"Duplicate/self-masking: shift ≤0.17\n({dup['exact_duplicate_rows']} dupes / {dup['n_subsample']})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
save(fig, "id_diagnostics.png")

# ---- Fig 6: cumulative PCA variance per layer with 95%/99% crossings ----
cv = load("pca_cumvar.json")
colors = {0: "C0", 3: "C1", 6: "C2", 9: "C4", 11: "C3"}
fig, ax = plt.subplots(figsize=(7.5, 4.8))
dims = np.arange(1, 769)
for r in cv:
    L = r["layer"]
    cum = np.array(r["cumvar"])
    c = colors[L]
    ax.plot(dims, cum, "-", color=c, lw=1.6,
            label=f"L{L}  (d95={r['d95']}, d99={r['d99']})")
    # mark the 95% and 99% crossing points on each curve
    ax.plot(r["d95"], cum[r["d95"] - 1], "o", color=c, ms=5)
    ax.plot(r["d99"], cum[r["d99"] - 1], "s", color=c, ms=5, mfc="none")
ax.axhline(0.95, color="k", ls="--", lw=0.9)
ax.axhline(0.99, color="k", ls=":", lw=0.9)
ax.text(770, 0.95, "95%", va="center", ha="left", fontsize=8)
ax.text(770, 0.99, "99%", va="center", ha="left", fontsize=8)
ax.set_xscale("log")
ax.set_xlim(1, 900)
ax.set_ylim(0, 1.02)
ax.set_xlabel("number of principal components (log scale)")
ax.set_ylabel("cumulative variance explained")
ax.set_title("Cumulative PCA variance, GPT-2 small residual stream\n"
             "● = 95% crossing, □ = 99% crossing  (L11 is post-final-layernorm)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3, which="both")
save(fig, "pca_cumvar.png")

# ---- Fig 6b: same curves, LINEAR x-axis (operator request 2026-07-02) ----
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for r in cv:
    L = r["layer"]
    cum = np.array(r["cumvar"])
    c = colors[L]
    ax.plot(dims, cum, "-", color=c, lw=1.6,
            label=f"L{L}  (d95={r['d95']}, d99={r['d99']})")
    ax.plot(r["d95"], cum[r["d95"] - 1], "o", color=c, ms=5)
    ax.plot(r["d99"], cum[r["d99"] - 1], "s", color=c, ms=5, mfc="none")
ax.axhline(0.95, color="k", ls="--", lw=0.9)
ax.axhline(0.99, color="k", ls=":", lw=0.9)
ax.text(772, 0.95, "95%", va="center", ha="left", fontsize=8)
ax.text(772, 0.99, "99%", va="center", ha="left", fontsize=8)
ax.set_xlim(0, 768)
ax.set_ylim(0, 1.02)
ax.set_xlabel("number of principal components (linear scale)")
ax.set_ylabel("cumulative variance explained")
ax.set_title("Cumulative PCA variance, GPT-2 small residual stream (linear x)\n"
             "● = 95% crossing, □ = 99% crossing  (L11 is post-final-layernorm)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
save(fig, "pca_cumvar_linear.png")

print("done")
