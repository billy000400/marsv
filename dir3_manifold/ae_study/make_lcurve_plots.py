"""Render the learning-curve diagnostic (operator ask 2026-07-08 #3) from
results/qwen_lcurve_L2.json into ../plots/. Two panels:
  (left)  held-out val rel-L2 vs training step, one line per k -> shows whether
          large-k models are still improving at the 3000-step lasse checkpoint.
  (right) the k-sweep val rel-L2 at the early (3000-step) checkpoint vs the final
          (converged-budget) checkpoint -> shows whether the U-shape flattens once
          the big bottlenecks are given time to train.
Agg headless; savefig + close only."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(os.path.dirname(HERE), "plots")
d = json.load(open(os.path.join(HERE, "results", "qwen_lcurve_L2.json")))
curves = d["curves"]
ks = sorted(int(k) for k in curves)
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(ks)))

fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5))

# left: learning curves
CKPT = 3000  # lasse-repro budget
for c, k in zip(cmap, ks):
    cur = curves[str(k)]
    axl.plot(cur["step"], cur["val_rel_l2"], "-", color=c, lw=1.7, label=f"k={k}")
axl.axvline(CKPT, color="k", ls="--", lw=1.0)
axl.text(CKPT, axl.get_ylim()[1], " 3000-step\n lasse budget", va="top", fontsize=8)
axl.set_xlabel("training step")
axl.set_ylabel("held-out rel-L2 error")
axl.set_title("Learning curves per bottleneck k (Qwen L2, 67M DeepAE)\n"
              "large-k curves still descend past the 3000-step budget = undertrained")
axl.legend(fontsize=8, ncol=2)
axl.grid(alpha=0.3)

# right: k-sweep early vs final
def at_step(cur, s):
    arr = np.array(cur["step"])
    i = int(np.argmin(np.abs(arr - s)))
    return cur["val_rel_l2"][i]

early = [at_step(curves[str(k)], CKPT) for k in ks]
final = [curves[str(k)]["val_rel_l2"][-1] for k in ks]
nsteps = d["n_steps"]
axr.plot(ks, early, "o--", color="#888", lw=1.6, label=f"@ {CKPT} steps (lasse budget)")
axr.plot(ks, final, "s-", color="#1f77b4", lw=1.9, label=f"@ {nsteps} steps (longer budget)")
axr.set_xscale("log", base=2)
axr.set_xticks(ks); axr.set_xticklabels([str(k) for k in ks])
axr.set_xlabel("bottleneck width k")
axr.set_ylabel("held-out rel-L2 error")
axr.set_title("k-sweep: the large-k rise shrinks with more training\n"
              "(is the lasse U a real ID minimum or an undertraining artifact?)")
axr.legend(fontsize=9)
axr.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "qwen_ae_lcurve.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote qwen_ae_lcurve.png")
print("early:", [round(x, 4) for x in early])
print("final:", [round(x, 4) for x in final])
