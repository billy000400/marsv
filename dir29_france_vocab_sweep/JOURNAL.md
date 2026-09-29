# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-09-29 — iteration 1: Stage 0 verification, full sweep launched

- Located dir27 code (`../dir27_interp_dist_vs_plateau_width/experiments/common.py`, `s2_sweep.py`, `s3_plots.py`).
  `experiments/sweep.py` imports dir27's `common` module (Runner, TS = 101-point grid, y_curve, width) and only
  changes the prompt, following dir28's adaptation. Endpoint B = token ID swap, never retokenized. All 50,256
  other tokens swept (dir27 excluded none).
- **Assumption (CLAUDE.md §1):** PLAN says "block-0 resid_post"; dir27's actual site is layer 0 = input embedding
  (last-position wte row; wpe shared). Kept dir27's actual code, as PLAN also says "reuse unchanged". Width =
  dir27's W_final (block 35); PLAN's `d(t)` = dir27's progress curve y(t).
- Control: `sweep.py 256 control` reproduces dir27 shard 0 (max |ΔW_final| 6.9e-7, max |ΔW_mid| 4.6e-7).
- France sanity on first 256 IDs: prompt ids [464, 3139, 286, 4881]; median W_final 0.151, 208/256 have W ≤ 0.3.
- **Material inconsistency found:** PLAN labels `w <= 0.3` "non-plateau" and `w > 0.3` "plateau-like". In dir27
  (REPORT Methods) small W = abrupt jump (plateau-like) and W near 0.8 = straight ramp (smooth); dir28 inspected
  `w > 0.3` as the smooth set. So the PLAN's labels are inverted relative to its own stated research question
  ("which tokens produce a smooth / non-plateau transition"). Decision: do not pick one silently. Save both
  sorted lists (W ≤ 0.3 ascending, as literally requested; W > 0.3 descending = dir27's smooth side), inspect
  both ends, and let the d(t) curves show directly which side is smooth. Flag in REPORT and PLAN.
- Full sweep launched in background (`results/sweep.log`), GPU shared with dir28 (~26 min alone).

On track? yes — Stage 0 done, Section 1 sweep running (~15%); blocker: threshold-label inversion in PLAN (handled by reporting both sides).

## 2026-09-29 — iteration 2: sweep analysed, deliverables written

- Sweep finished (25 shards, 50,256 tokens). `analyze.py`: 47,073 W ≤ 0.3, 3,183 W > 0.3, median W 0.188, 537 flagged.
  Moved the histogram count labels above the bars (they overlapped).
- Narrowest end: short capitalized word starts (82/100 are space+Capital fragment vs 21% vocab), no places.
  `France` (no space) W = 0.072.
- Widest end: 53/100 places by manual reading (`experiments/categories.py`), many France-linked (Brittany, Normandy,
  Lyon, Gaul, Monaco, Algeria...), 17 scraped/glitch tokens (as in dir27), 30 other.
- `curves.py` on (a) ' Ent',' More',' the','France',' French' and (b) ' Brittany',' Spain',' Paris',' Germany',
  ' SolidGoldMagikarp'. Figure confirms small W = jump (plateau), large W = gradual.
- **Assumption:** the "complete list" of W ≤ 0.3 (47,073 rows) cannot fit in REPORT.md's 5,000-word limit. So the complete
  sorted list lives in `results/france_W_le_0.3.csv`, the report links it, and the report shows the 40 narrowest.
  Rejected alternative: printing a partial list and calling it complete.
- Did not compute a vocabulary base rate for place names, because that needs labeling the whole vocabulary (automated
  labeling is out of scope). The report names this as a limit.
- REPORT/RESULTS pass `check_render.py`. All five success-criterion items are covered, so I am creating STOP.

On track? yes — all sections done; no blocker.
