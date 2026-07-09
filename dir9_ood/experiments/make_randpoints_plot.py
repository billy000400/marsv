"""Plot the randomly-sampled-points detector vs baselines, per OOD set, for each model.
Reads results/auroc_randpoints.csv; writes results/plots/randpoints_<model>.png (best measurement
point per method). Headless Agg; savefig + close, never show."""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "..", "results", "auroc_randpoints.csv")
OUTDIR = os.path.join(HERE, "..", "results", "plots")
os.makedirs(OUTDIR, exist_ok=True)

rows = list(csv.DictReader(open(CSV)))
models = sorted({r["model"] for r in rows})
OOD = ["random", "shuffled", "code"]
# best (max) AUROC over measurement points, per (model, ood, method)
METHODS = ["rand-points-disp", "rand-points-ent", "baseline-MSP", "baseline-mahalanobis"]
COLORS = {"rand-points-disp": "#c0392b", "rand-points-ent": "#e67e22",
          "baseline-MSP": "#2980b9", "baseline-mahalanobis": "#27ae60"}


def best(model, ood, method):
    vals = [(float(r["auroc"]), r["measurement_point"]) for r in rows
            if r["model"] == model and r["ood_set"] == ood and r["method"] == method]
    return max(vals) if vals else (float("nan"), "")


for model in models:
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(OOD)); w = 0.2
    for i, m in enumerate(METHODS):
        vals = [best(model, o, m)[0] for o in OOD]
        pts = [best(model, o, m)[1] for o in OOD]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=COLORS[m])
        for b, v, p in zip(bars, vals, pts):
            lab = f"{v:.2f}" + (f"\n@{p}" if p not in ("n/a", "") else "")
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, lab,
                    ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(OOD)
    ax.set_ylim(0, 1.08); ax.set_ylabel("AUROC (higher = better OOD detection)")
    ax.set_title(f"Randomly-sampled-points detector vs baselines — {model}\n"
                 "(best measurement point per method; dashed = chance)")
    ax.legend(fontsize=7, ncol=2, loc="lower center")
    fig.tight_layout()
    out = os.path.join(OUTDIR, f"randpoints_{model}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)
