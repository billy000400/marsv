"""S7 joint timeline: put the Grokking side (Figure-9 metrics) on one checkpoint axis for
the pilot + both fresh runs, annotated with each run's preregistered gate verdict, and mark the
plateau-assay reference from the reconstruction (Matthew's relative distance transition width).

This figure answers PLAN Experiment 4's "one checkpoint-aligned figure": the top row shows the
Grokking evolution (test local complexity + eps=0.03 PGD adversarial accuracy vs training step)
for every model we have Figure-9 curves for; the bottom text panel states the bounded relationship
verdict. The plateau side is a per-model summary (transition width of Matthew's d(t)); a per-
checkpoint plateau sweep on the fresh models is separate (S6) and its status is annotated honestly.

Usage: plot_joint_timeline.py
Headless (Agg); writes plots/joint_timeline.png.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from cvd_style import CVD

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def R(p): return os.path.join(HERE, p)

# (name, curves, verdict, colour, linestyle, marker) — colour is never the only channel
RUNS = [
    ("Pilot char (3.5k)", "results/fig9_pilot_char.json", "results/fig9_pilot_char_verdict.json", "0.45",  ":",  "^"),
    ("Fresh char (30k)",  "results/fig9_grok_char.json",  "results/fig9_grok_char_verdict.json",  CVD[0], "-",  "o"),
    ("Fresh BPE (10k)",   "results/fig9_grok_bpe.json",   "results/fig9_grok_bpe_verdict.json",   CVD[1], "--", "s"),
]

def load(path):
    if not os.path.exists(R(path)):
        return None
    recs = sorted(json.load(open(R(path))), key=lambda r: r["step"])
    return {
        "step": np.array([r["step"] for r in recs], float),
        "lc":   np.array([r["lc"]["test"]["mean"] for r in recs], float),
        "adv":  np.array([r["adv_acc"] for r in recs], float),
        "acc":  np.array([r["clean_acc"] for r in recs], float),
    }

def verdict(path):
    if not os.path.exists(R(path)):
        return "(pending)"
    return json.load(open(R(path))).get("verdict", "?")

fig, (axL, axR, axT) = plt.subplots(1, 3, figsize=(16, 5),
                                    gridspec_kw={"width_ratios": [1, 1, 0.9]})

lines = []
for name, jp, vp, c, ls, mk in RUNS:
    d = load(jp)
    if d is None:
        continue
    x = np.clip(d["step"], 1, None)  # log axis; step 0 -> 1
    v = verdict(vp)
    lab = f"{name} [{ls}, {mk}]: {v}"
    axL.plot(x, d["lc"], color=c, ls=ls, marker=mk, ms=4, label=lab)
    axR.plot(x, d["adv"], color=c, ls=ls, marker=mk, ms=4, label=lab)

axL.set_xscale("log"); axL.set_xlabel("training step (log; 0 drawn at 1)")
axL.set_ylabel("test local complexity (sign crossings, 12 layers)")
axL.set_title("Grokking side: local complexity")
axL.legend(fontsize=8, loc="upper right"); axL.grid(alpha=0.3)

axR.set_xscale("log"); axR.set_xlabel("training step (log; 0 drawn at 1)")
axR.set_ylabel(r"$\epsilon=0.03$ PGD adversarial accuracy")
axR.set_title("Grokking side: delayed robustness")
axR.axhline(0.05, ls="--", color="k", lw=0.8)
axR.legend(fontsize=8, loc="upper left"); axR.grid(alpha=0.3)

# plateau side / verdict text panel
axT.axis("off")
try:
    ms = json.load(open(R("results/matthew_summary.json")))
    plateau_txt = (f"Plateau assay (reconstruction, final ckpt):\n"
                   f"  {ms['n_plateau']}/{ms['n_pairs']} pairs pass strict rule\n"
                   f"  median transition width w = {ms['median_w_monotone']:.3f}\n"
                   f"  (diagonal / no-plateau reference w = 0.8)\n"
                   f"  boundary sharpens with depth:\n"
                   f"   block0 w={ms['depth_median_w']['0']:.2f} -> logits sharper")
except Exception:
    plateau_txt = "Plateau assay summary unavailable."

vpilot = verdict("results/fig9_pilot_char_verdict.json")
vchar  = verdict("results/fig9_grok_char_verdict.json")
vbpe   = verdict("results/fig9_grok_bpe_verdict.json")
panel = (
    "JOINT VERDICT (temporal association)\n"
    "----------------------------------------\n"
    "Figure-9 grokking gate:\n"
    f"  pilot char : {vpilot}\n"
    f"  fresh char : {vchar}\n"
    f"  fresh BPE  : {vbpe}\n\n"
    + plateau_txt +
    "\n\nA joint Grokking<->plateau claim needs a\n"
    "PASSing BPE gate (Matthew's exact tokens).\n"
    "See REPORT.md for the bounded verdict."
)
axT.text(0.0, 1.0, panel, va="top", ha="left", family="monospace", fontsize=9)

fig.suptitle("Joint checkpoint timeline: Grokking (Figure-9) evolution vs plateau assay", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = R("plots/joint_timeline.png")
fig.savefig(out, dpi=130); plt.close(fig)
print("wrote", out)
print("verdicts:", vpilot, vchar, vbpe)
