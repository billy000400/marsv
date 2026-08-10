"""How big is seed noise on the median transition width, and does it explain the reported gaps?

Every claim in the frozen-block section is a difference between two runs' median width. A difference
is only informative if it is larger than the width shift a fresh initialization produces on its own.
Five conditions have now been trained twice with identical data order, schedule and freeze mask,
differing only in the model init seed (1337 vs 2024), so the across-seed spread is measurable.

Left panel:  the two seeds of each condition, at matched validation accuracy and at step 30,000.
Right panel: the size of each reported between-run gap against the largest seed spread measured.

Writes plots/seed_replication.png. CVD-safe (CLAUDE.md rule 13): seeds differ by marker AND fill AND
colour; the gap bars are one hue with hatching separating those that clear the spread from those
that do not.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cvd_style import CVD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")
C = json.load(open(os.path.join(RES, "frozen_assay_summary.json")))["summary"]["conditions"]

# (condition stem, axis label)
SEEDED = [("narrow192", "192 wide,\nnothing frozen"),
          ("frozen_early", "blocks 1-4\nfrozen"),
          ("frozen_deep", "blocks 1-7\nfrozen"),
          ("frozen_high", "blocks 0-5, 11\nfrozen"),
          ("frozen_mirror", "blocks 5-11\nfrozen")]
# (label, [A conditions], [B conditions]) -- the between-run differences the report's claims rest on.
# Where a side has two seeds, every A-B pairing is formed and the SMALLEST gap is plotted, so a claim
# is credited only with the margin its worst pair of initializations gives.
GAPS = [("blocks 6-10 vs untouched 12-block reference",
         ["frozen_high_matched", "frozen_high_s2_matched"], ["ref_matched_step"]),
        ("blocks 1-5 vs blocks 0-7 (a superset of it)",
         ["frozen_mid_low_matched"], ["frozen_late_matched"]),
        ("mid-stack window 4-8 vs bottom window 0-4",
         ["frozen_mid_matched"], ["frozen_mirror_matched", "frozen_mirror_s2_matched"]),
        ("3 mid-stack blocks vs 5 blocks at the bottom",
         ["frozen_mid3_matched"], ["frozen_mirror_matched", "frozen_mirror_s2_matched"]),
        ("5 trainable blocks by the readout vs at the bottom",
         ["frozen_deep_matched", "frozen_deep_s2_matched"],
         ["frozen_mirror_matched", "frozen_mirror_s2_matched"]),
        ("12 trainable blocks vs 8 (sharpest 8 vs bluntest 12)",
         ["ref_matched_step"], ["frozen_early_matched"])]

spread = {}
for stem, _ in SEEDED:
    for which in ("matched", "last"):
        a, b = f"{stem}_{which}", f"{stem}_s2_{which}"
        if a in C and b in C:
            spread[(stem, which)] = abs(C[a]["median_w"] - C[b]["median_w"])
max_spread = max(spread.values())

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9),
                         gridspec_kw={"width_ratios": [1.0, 1.15]})

# ---- left: the two seeds of every twice-trained condition -------------------------------------
ax = axes[0]
xs = np.arange(len(SEEDED), dtype=float)
for i, (stem, _) in enumerate(SEEDED):
    # the two axes are staggered vertically as well as horizontally so their spread labels never collide
    for dx, dy, which, mk, col in ((-0.16, 0.012, "matched", "o", CVD[0]),
                                   (0.16, 0.034, "last", "s", CVD[1])):
        a, b = f"{stem}_{which}", f"{stem}_s2_{which}"
        if a not in C or b not in C:
            continue
        v1, v2 = C[a]["median_w"], C[b]["median_w"]
        ax.plot([xs[i] + dx] * 2, [v1, v2], color="0.55", lw=1.2, zorder=1)
        ax.plot([xs[i] + dx], [v1], marker=mk, ms=9, color=col, mfc=col, ls="none", zorder=3)
        ax.plot([xs[i] + dx], [v2], marker=mk, ms=9, color=col, mfc="white", mew=1.8, ls="none",
                zorder=3)
        ax.annotate(f"{abs(v1 - v2):.3f}", (xs[i] + dx, max(v1, v2) + dy), ha="center",
                    fontsize=7.5, color="0.3")
seed_handles = [
    plt.Line2D([], [], marker="o", ms=9, color=CVD[0], mfc=CVD[0], ls="none",
               label="matched val. accuracy 0.550, seed 1337 (filled)"),
    plt.Line2D([], [], marker="o", ms=9, color=CVD[0], mfc="white", mew=1.8, ls="none",
               label="matched val. accuracy 0.550, seed 2024 (open)"),
    plt.Line2D([], [], marker="s", ms=9, color=CVD[1], mfc=CVD[1], ls="none",
               label="step 30,000, seed 1337 (filled)"),
    plt.Line2D([], [], marker="s", ms=9, color=CVD[1], mfc="white", mew=1.8, ls="none",
               label="step 30,000, seed 2024 (open)")]
ax.set_xticks(xs)
ax.set_xticklabels([lab for _, lab in SEEDED], fontsize=8)
ax.set_ylabel("median transition width $w_{10\\to90}$\n(lower = sharper plateau)")
ax.set_ylim(0.26, 0.86)  # headroom so the legend never sits on the sharpest condition
ax.grid(alpha=0.3, axis="y")
ax.set_title(f"Retraining from a second init moves $w$ by at most {max_spread:.3f}", fontsize=10)
ax.legend(handles=seed_handles, fontsize=7.5, loc="upper left", ncol=1, framealpha=0.95)

# ---- right: reported gaps against the largest seed spread --------------------------------------
ax = axes[1]
rows = []
for lab, A, B in GAPS:
    have = [(a, b) for a in A if a in C for b in B if b in C]
    if have:
        rows.append((lab, min(abs(C[a]["median_w"] - C[b]["median_w"]) for a, b in have)))
rows.sort(key=lambda r: r[1])
y = np.arange(len(rows), dtype=float)
for i, (lab, g) in enumerate(rows):
    clears = g > max_spread
    ax.barh([y[i]], [g], height=0.62, color=CVD[0] if clears else "0.72",
            hatch="" if clears else "//", edgecolor="black", linewidth=0.8)
    ax.annotate(f"{g:.3f}" + ("" if clears else "  (inside seed noise)"), (g + 0.006, y[i]),
                va="center", fontsize=8)
ax.axvline(max_spread, color=CVD[1], ls="--", lw=2)
ax.annotate(f"largest seed spread\nmeasured ({max_spread:.3f})", (max_spread + 0.004, len(rows) - 0.35),
            fontsize=8, color=CVD[1], va="top")
ax.set_yticks(y)
ax.set_yticklabels([lab for lab, _ in rows], fontsize=8)
ax.set_xlabel("difference in median $w$ between the two runs")
ax.set_xlim(0, max(g for _, g in rows) * 1.42)
ax.grid(alpha=0.3, axis="x")
n_clear = sum(1 for _, g in rows if g > max_spread)
ax.set_title(f"{n_clear} of {len(rows)} reported gaps are larger than that spread", fontsize=10)

fig.suptitle("Seed replication: five conditions trained twice, and what that error bar allows\n"
             "(150 character pairs, interpolation block 0, context \"The house was \")", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(os.path.join(PLOTS, "seed_replication.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote plots/seed_replication.png")
print(json.dumps({"spread": {f"{k[0]}_{k[1]}": round(v, 4) for k, v in spread.items()},
                  "max_spread": round(max_spread, 4),
                  "gaps": {lab: round(g, 4) for lab, g in rows}}, indent=1))
