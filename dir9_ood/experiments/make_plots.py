"""ROC + score-distribution plots from results/raw_scores.npz (ID) — recomputes OOD on the fly
is not possible, so we plot from the saved id/ood arrays. We instead read auroc_table.csv for the
ROC curves using the stored raw scores. Minimal, matplotlib only."""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
PLOTS = os.path.join(RES, "plots")
os.makedirs(PLOTS, exist_ok=True)
data = np.load(os.path.join(RES, "scores_full.npz"), allow_pickle=True)
# data keys: f"{setname}|{method}|{point}" -> array
keys = list(data.keys())
sets = sorted({k.split("|")[0] for k in keys})
methods_points = sorted({(k.split("|")[1], k.split("|")[2]) for k in keys})
ood_sets = [s for s in sets if s != "id"]


def roc(id_s, ood_s):
    s = np.concatenate([id_s, ood_s]); y = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
    order = np.argsort(-s); y = y[order]
    tpr = np.cumsum(y) / max(y.sum(), 1); fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return np.concatenate([[0], fpr]), np.concatenate([[0], tpr])


for oset in ood_sets:
    plt.figure(figsize=(6, 6))
    for (m, p) in methods_points:
        kid = f"id|{m}|{p}"; kood = f"{oset}|{m}|{p}"
        if kid in data and kood in data:
            fpr, tpr = roc(data[kid], data[kood])
            plt.plot(fpr, tpr, label=f"{m}@{p}", lw=1)
    plt.plot([0, 1], [0, 1], "k--", lw=0.5)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC: fineweb vs {oset}")
    plt.legend(fontsize=6, loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, f"roc_{oset}.png"), dpi=90); plt.close()

# score distributions for the headline method (perturbation & jacobian at resid6) per ood set
for m in ["plateau-jacFrob", "selfNLL-grad", "plateau-perturbation", "baseline-L2norm"]:
    p = "resid6"
    plt.figure(figsize=(7, 4))
    kid = f"id|{m}|{p}"
    if kid in data:
        plt.hist(data[kid], bins=20, alpha=0.5, label="id (fineweb)", density=True)
        for oset in ood_sets:
            kood = f"{oset}|{m}|{p}"
            if kood in data:
                plt.hist(data[kood], bins=20, alpha=0.5, label=oset, density=True)
        plt.title(f"{m}@{p} score distribution"); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS, f"dist_{m}.png"), dpi=90); plt.close()
print("plots written to", PLOTS, os.listdir(PLOTS))
