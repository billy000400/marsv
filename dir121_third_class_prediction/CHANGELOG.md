# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-07-25 — first deliverables: complete Stage-1 census + activation-region verdict (S1–S5)

Direction created from empty templates; RESULTS.md and REPORT.md written for the first time, so
nothing was superseded. All numbers below are new.

**Added to RESULTS.md and REPORT.md**
- Stage-1 census of all **45** digit transitions × **100** fixed test-image pairs × 50 interpolation
  points at direction 12's seed-0 step-30000 checkpoint (4,500 paths). Categories: 6 both / 13 stable
  third-class only / **0 sub-plateau only** / 26 neither → **19 of 45** stable third-class.
  Prevalence sensitivity 23 / 19 / 7 at 25% / 50% / 75%. Leave-one-out 100/100 on all third-class
  labels; three sub-plateau labels flagged borderline on bootstrap (0→4 0.684, 1→9 0.636, 5→7 0.895).
- Dominant third digit at seed 0 is **7 (×13) or 8 (×6)**.
- Stage-2 cross-seed confirmation at seeds 1 and 2 (same pairs, same frozen thresholds): 12 and 18
  stable third-class transitions; 27/45 three-seed agreement on the third-class label, 39/45 on the
  sub-plateau label. Dominant third digit is **seed-specific**: 7/8 (seed 0), 1 (seed 1), 2/8
  (seed 2); only **6→9 → z=8** is stable at all three seeds with the same z.
- Stage-3/4 activation-region analysis at post-ReLU `h1` for **all 19** stable transitions (no subset
  needed): pooled over 14,700 segment points on 1,376 paths, **2.5%** lie inside the real activation
  region of the predicted digit. Best case 2→9 (18.5% of points, 14.3% of paths); 15 of 19 below 5%.
  Median normalized distance to the predicted digit 1.50–3.71; to the *nearest* of all ten digits
  1.23–2.09 — outside every digit's region.
- Stage-5 three-way verdict: **19 of 19 "prediction only"**, 0 "activation-region match", 0 "mixed".
- Stage-5 controls: C1 96.0% argmin-correct / 93.6% both; C2 endpoint-plateau median 0.81 (all
  endpoint-predicted points median 0.37 — caveat stated in both deliverables); C3 91.8–96.5% across
  all ten digits; C4 max relative deviation 1.7e-7. All pass.

**Figures added (all embedded as rendered images in BOTH RESULTS.md and REPORT.md)**
`plots/s1_transition_matrix.png`, `s1_mean_curves_grid.png`, `s1_class_composition.png`,
`s1_segment_widths.png`, `s1_seed_agreement.png`, `s4_region_membership.png`, `s4_distance_view.png`,
`s4_pca_view.png`, `s5_controls.png`.

**Verification** — REPORT.md checked through `POST api.github.com/markdown`: 8 display equations all
render as `js-display-math`, 0 as `<pre lang="math">`, 9 `<img>` tags. Inline-math backslash-punctuation
grep clean in both files.

**Provenance** — no model retrained, no file written into direction 12. Every result file records the
source checkpoint path and SHA-256.

## 2026-07-25 — later-layer follow-up (h2/h3) added; plan complete

The `h1` activation-region result is a clean null, which is exactly the precondition PLAN.md sets for
its "Optional later-layer follow-up". Ran it and added a new Result 8 to REPORT.md and a new
later-layer section to RESULTS.md. Nothing earlier was superseded — all S1–S5 numbers are unchanged,
and the `h1` row of the new table (2.5% of segment points inside the predicted digit's region) is
reproduced by the new code path, confirming the two scripts agree.

**Added** (`experiments/s6_later_layers.py`, `results/s6_later_layers.json`,
`plots/s6_later_layers.png`, embedded as a rendered image in BOTH deliverables):
- Fraction of third-class segment points inside the real-z activation region by hook point:
  `h1` **2.5%**, `h2` **10.6%**, `h3` **0.2%** (same frozen segments, same rules, only the hook point
  changes).
- New metric **region occupancy** `N(u)` = how many of the ten digit regions contain a segment point,
  defined with a rendered equation in Methods. Mean over segment points: `h1` 0.08, `h2` **5.80**,
  `h3` 0.00.
- Decomposition of the inside-region criterion into its two halves: fraction with ratio to z below 1
  (`h1` 2.4%, `h2` 78.4%, `h3` 0.2%) and fraction where z is the *nearest* of the ten regions
  (`h1` 14.3%, `h2` 11.7%, `h3` 0.6%; chance among the eight non-endpoint digits is 12.5%).
- Median distance ratio to the predicted digit by layer: `h1` 1.50–3.71, `h2` 0.25–4.33,
  `h3` 174.8–471.0. Controls C1/C3 recomputed per layer: `h1` 0.936/0.946, `h2` 0.949/0.984,
  `h3` 0.950/0.966 — all pass, so the measure stays calibrated at every layer.

**Interpretation added to both deliverables:** the `h2` rise from 2.5% to 10.6% is NOT a partial
activation-region match. At `h2` the ten regions overlap so heavily under a mean-and-variance summary
that a segment point is inside 5.8 of them at once, making "ratio to z below 1" nearly automatic;
the discriminating half of the criterion is unchanged and at chance. The verdict stays 19 of 19
"prediction only".

**Also updated:** REPORT.md Summary gained a fourth finding; the Methods "Hook point" paragraph now
states that Results 1–7 are at `h1` and Result 8 at `h2`/`h3`; Conclusion limitation (iii) was
rewritten from "the `h2`/`h3` follow-up is unlocked but not yet run" to the actual `h2` region-overlap
caveat; Reproducibility lists the new script and result file. RESULTS.md headline now notes the
later-layer null.

**Verification** — REPORT.md re-checked through `POST api.github.com/markdown`: 9/9 display equations
render as `js-display-math`, 0 as `<pre lang="math">`, 10 `<img>` tags; RESULTS.md 10 `<img>` tags.
Inline-math backslash-punctuation grep clean in both.
