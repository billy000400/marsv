"""Render fig11 — in-context vs out-of-context plateau-KL discrimination AUROC."""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); PLOTS = os.path.join(ROOT, "plots")
rows = list(csv.DictReader(open(os.path.join(RES, "incontext_discrimination.csv"))))

fams = ["interp", "tangent_pert", "cov_gauss"]

def get(fam, score, ctx):
    for r in rows:
        if r["family"] == fam and r["score"] == score and r["context"] == ctx:
            return float(r["auroc"])
    return np.nan

ooc = [get(f, "plateau_kl", "out_of_context") for f in fams]
ic = [get(f, "plateau_kl", "in_context") for f in fams]
maha = [get(f, "maha_twosided", "n/a") for f in fams]
knn = [get(f, "knn_distance", "n/a") for f in fams]

x = np.arange(len(fams)); w = 0.2
fig, ax = plt.subplots(figsize=(8.4, 4.6))
ax.bar(x - 1.5 * w, maha, w, label="maha_twosided (stat)", color="#bbbbbb".replace(" ", ""))
ax.bar(x - 0.5 * w, knn, w, label="knn_distance (stat)", color="#888888")
ax.bar(x + 0.5 * w, ooc, w, label="plateau_kl (out-of-context)", color="#f0a35e")
ax.bar(x + 1.5 * w, ic, w, label="plateau_kl (IN-CONTEXT)", color="#2b7bba")
for xi, v in zip(x - 1.5 * w, maha): ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=7)
for xi, v in zip(x - 0.5 * w, knn): ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=7)
for xi, v in zip(x + 0.5 * w, ooc): ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=7)
for xi, v in zip(x + 1.5 * w, ic): ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
ax.set_xticks(x); ax.set_xticklabels(fams)
ax.set_ylabel("AUROC (real vs family)"); ax.set_ylim(0, 1.05)
ax.set_title("In-context evaluation sharpens the functional plateau-KL realness signal\n"
             "(catches the too-central `interp` family that defeats all statistics)")
ax.legend(fontsize=8, ncol=2, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "fig11_incontext_discrimination.png"), dpi=130)
plt.close()
print("wrote fig11")
