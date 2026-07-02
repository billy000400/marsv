"""fig12 — Phase 6b manifold-projection repair: ext KL vs L2 move, kNN direction vs matched random."""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
rows = {r["method"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results", "manifold_repair_metrics.csv")))}
def pt(name): return float(rows[name]["move_from_corrupt"]), float(rows[name]["ext_KL_clean"])

start = float(rows["corrupted(start)"]["ext_KL_clean"])
knn = [(0.0, start), pt("knn_project(t=0.10)"), pt("knn_project(t=0.25)"),
       pt("knn_project(t=0.50)"), pt("knn_project(t=1.00)")]
rnd = [(0.0, start), pt("random_move(t=0.25-matched)"), pt("random_move(t=0.50-matched)"),
       pt("random_move(knn1-matched)")]
orc = [pt("shrink_clean(oracle,t=0.25-matched)"), pt("shrink_clean(oracle,knn1-matched)")]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.axhline(start, ls="--", c="0.5", lw=1, label=f"corrupted start (KL={start:.2f})")
ax.plot(*zip(*knn), "-o", c="#1f77b4", label="kNN manifold projection")
ax.plot(*zip(*rnd), "-s", c="#d62728", label="random move (matched size)")
ax.plot(*zip(*orc), "^", c="#2ca02c", ms=9, label="oracle (toward true clean)")
ax.annotate("t=0.25\n(best partial step)", xy=knn[2], xytext=(28, 0.35),
            arrowprops=dict(arrowstyle="->", color="#1f77b4"), fontsize=9, color="#1f77b4")
ax.set_xlabel("L2 move from corrupted activation")
ax.set_ylabel("in-context KL(clean || repaired)  (lower = better)")
ax.set_title("Phase 6b: a fractional kNN-manifold step is the only objective-free repair\n"
             "that beats the corrupted start; the full step overshoots (shell-distance trap)")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "plots", "fig12_manifold_repair.png"), dpi=110)
plt.close(fig)
print("wrote plots/fig12_manifold_repair.png")
