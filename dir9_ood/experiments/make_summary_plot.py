"""Headline figure: best plateau variant vs best baseline per OOD set, with each bar
annotated by the ACTUAL method@point it represents (per operator request). Reads the
current-best AUROCs from results/auroc_table.csv. CPU-only matplotlib, no GPU."""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
PLOTS = os.path.join(RES, "plots")

# plateau variants = the methods under test; selfNLL-grad is a confidence control (excluded).
PLATEAU = {"plateau-jacFrob", "plateau-perturbation"}
BASELINE = {"baseline-MSP", "baseline-L2norm", "baseline-mahalanobis", "cup-RMD", "cup-QUE"}

rows = list(csv.DictReader(open(os.path.join(RES, "auroc_table.csv"))))
sets = ["random", "shuffled", "code"]


def best(oset, group):
    cand = [r for r in rows if r["ood_set"] == oset and r["method"] in group]
    r = max(cand, key=lambda r: float(r["auroc"]))
    pt = "" if r["measurement_point"] == "n/a" else "@" + r["measurement_point"]
    return r["method"].replace("baseline-", "") + pt, float(r["auroc"])

plat = [best(s, PLATEAU) for s in sets]
base = [best(s, BASELINE) for s in sets]

x = range(len(sets)); w = 0.38
fig, ax = plt.subplots(figsize=(8.4, 5.2))
b1 = ax.bar([i - w / 2 for i in x], [v for _, v in plat], w, color="#b5495b", label="best plateau variant")
b2 = ax.bar([i + w / 2 for i in x], [v for _, v in base], w, color="#3b6ea5", label="best baseline")
ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance")

for bars, labels in [(b1, plat), (b2, base)]:
    for rect, (name, val) in zip(bars, labels):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.012, f"{val:.2f}", ha="center", fontsize=10)
        ax.text(rect.get_x() + rect.get_width() / 2, 0.04, name, ha="center", va="bottom",
                rotation=90, fontsize=9, color="white", fontweight="bold")

ax.set_xticks(list(x)); ax.set_xticklabels(sets); ax.set_ylim(0, 1.18); ax.set_ylabel("AUROC")
ax.set_title("Best plateau variant vs best baseline per OOD set\n"
             "(GPT-2 small, canonical split, N=200; baselines win every set)")
ax.legend(loc="upper center", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "summary_best_per_set.png"), dpi=90); plt.close(fig)
print("plateau:", plat)
print("baseline:", base)
print("wrote", os.path.join(PLOTS, "summary_best_per_set.png"))
