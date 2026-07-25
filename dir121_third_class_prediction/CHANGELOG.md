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

## 2026-07-25 — all ten figures rebuilt colour-vision-deficiency-safe (CLAUDE.md rule 13)

**No scientific result changed.** Every number in RESULTS.md and REPORT.md is byte-identical to the
previous entry: the three analysis scripts were re-run after the plotting edits and
`results/s1_classification.json`, `results/s3_s4_regions.json` and `results/s6_later_layers.json`
each `diff` clean against their previous versions. This entry is about figure accessibility and the
captions that describe the figures.

**Why** — the previous figures violated CLAUDE.md rule 13, which is a hard requirement because the
operator of this project has red-green colour deficiency. Concretely: the Stage-1 category palette used
green (`#54A24B`) against orange; the C2 control histogram used matplotlib `C2`/`C3`, i.e. green
against red, as its only distinguishing channel; the segment-width threshold line was red and the
box plot's median/mean were the default orange/green; `s4_region_membership` marked its two marker
series black-vs-red; the later-layer figure encoded `h1`/`h2`/`h3` as `C0`/`C1`/`C2` (blue/orange/
green); and `tab10` — which contains both a green and a red — encoded digit identity in three figures.
Several series were also identified by colour alone, with no linestyle, marker or hatch.

**Changed (figures)** — added `experiments/cvd_style.py` holding the mandated green-free palette
(`#0072B2`, `#D55E00`, `#CC79A7`, `#56B4E9`, `#E69F00`) plus shared category/hatch/marker maps, and
rewired all four scripts to it. Every series now carries a second, non-colour channel:
- `s1_transition_matrix.png` — categories drawn as hatched cells (`//` third-class only, `\\`
  sub-plateau only, `xx` both, unhatched pale = neither); the third-digit label is now dark text with a
  white stroke so it stays legible over hatching.
- `s1_mean_curves_grid.png` — category encoded by linestyle *and* band hatch as well as hue.
- `s1_class_composition.png` and the third-digit panel of `s1_seed_agreement.png` — the ten digits
  moved from `tab10` to the sequential `cividis` ramp (CVD-designed, monotone in grayscale), with the
  digit printed inside every band/cell large enough to hold it, so identity never rests on hue.
- `s1_segment_widths.png` — threshold line now black dashed; median/mean/flier styles set explicitly.
- `s4_distance_view.png`, `s4_pca_view.png` — the endpoint-a / endpoint-b / third-digit-z series now
  differ by linestyle and marker shape, and are role-labelled in the legend.
- `s4_region_membership.png`, `s5_controls.png`, `s6_later_layers.png` — paired bars distinguished by
  hatch, marker series by shape and linestyle; the red-vs-green C2 histogram and the blue/orange/green
  layer encoding are gone.

**Changed (deliverables)** — all ten captions in BOTH RESULTS.md and REPORT.md rewritten so that no
series is identified by its colour, as rule 13 requires: they now name the hatch, linestyle or marker
("the dashed triangle line", "the `//`-hatched bars"). Body prose that referred to "the black
diamonds", "the red triangles", "a wedge of a third colour" and "each row is dominated by one or two
colours" was rewritten the same way. RESULTS.md's figure block additionally gained one motivating
sentence before each figure (rule 12) instead of ten captions in a row. REPORT.md gained a short
**Figure conventions** subsection at the end of Methods and lists `cvd_style.py` under
Reproducibility.

**Verification** — REPORT.md re-checked through `POST api.github.com/markdown`: 9/9 display equations
render as `js-display-math`, 0 as `<pre lang="math">`, 10 `<img>` tags; RESULTS.md 10 embedded images;
no bare `(plots/*.png)` path outside an `![...]()` embed in either file; inline-math
backslash-punctuation grep clean in both; the only remaining occurrences of "red"/"green" in either
file are the sentences stating that the figures avoid a red-green contrast.

## 2026-07-25 — iter 3: PCA view replaced by an LDA plane and a margin-gradient/SVD decision slice (S4b/S7)

**Why** — PLAN.md was reopened: the two-dimensional PCA view in Result 4 was judged inadequate.
Principal components are the directions of largest overall variance, which need not separate the three
digits involved and need not be the directions in which the model's decision changes, so the figure
could not support either of the two claims it was standing next to. PLAN.md S4b specifies two
replacements, one supervised by the real class labels and one built from the model's own margins,
each with required diagnostics.

**Added (code + numbers)** — `experiments/s4b_planes.py`, writing `results/s4b_planes.json`. Rules
frozen in JOURNAL.md *before* the first run: representative path = medoid of the full 50×200 `h1`
trajectories among the paths carrying a dominant-z segment; LDA on 2,000 real training images per
class with ridge `1e-3·tr(S_W)/200`; 2 s.d. spread ellipses from held-out real points; margin
gradients over the segment ±5 alpha points; 161×161 grid, never clamped to `h ≥ 0`. Run on **all 19**
seed-0 stable transitions, not a subset.

**Removed** — the PCA figure `plots/s4_pca_view.png` and its generating block in
`experiments/s3_s4_regions.py`. That script was re-run afterwards and `results/s3_s4_regions.json`
`diff`s clean against the pre-edit copy, so **no previously reported number changed**.

**New results now in both deliverables (superseding the PCA paragraph, which made no quantitative
claim):**
- View A, pooled over the same 14,700 segment points scored in Result 4: **0.02%** lie inside the
  2 s.d. real-*z* ellipse, against **2.5%** in the full 200-d space. One transition (2→9) contributes
  every such point, at 0.3% of its own segment. Only 0.4% / 1.8% fall inside the endpoint digits'
  ellipses. Ellipse calibration 85.1%–91.0% of held-out real images inside their own ellipse (86.5%
  expected). The supervised projection is therefore *stricter* than the full-space test, not kinder.
- View B: two-axis gradient energy **96.2%–99.5%** (median 98.7%), so the slice is faithful to the
  local margin geometry; the third digit's decision region covers 1.7%–37.7% of the plotted window
  (median 31.9%); **100%** of grid cells in every transition are off the post-ReLU support (25.4%–34.9%
  of coordinates negative, 8.2%–13.4% of the norm).
- Projection honesty: median in-plane share of squared distance from the anchor is 12.6% (path) and
  4.5% (real activations), so plotted real points are shadows. Added a projected-path fidelity check —
  collapse each path point into the plane and re-classify: predictions on the segment are unchanged in
  14 of 19 transitions and ≥83% preserved in 18 of 19 (worst 1→6, 57.9%), which is what licenses the
  claim that the drawn path really crosses the drawn z-region.

**REPORT.md** — new Methods block defining the medoid rule, LDA (with `S_W`/`S_B` and the generalized
eigenproblem), the Mahalanobis ellipse test, the margin-gradient SVD plane, two-axis energy, the
off-support diagnostics, the in-plane energy share and the projected-path fidelity, all as rendered
equations at column 0. New **Result 5** with four embedded figures; former Results 5–8 renumbered 6–9
and every cross-reference updated. Summary finding 2 and the Conclusion now state the LDA number and
the classifier-region-versus-data distinction; a new limitation (v) states the plane's costs.
**RESULTS.md** — new "Two 2-D views" metric table and the four figures replacing the PCA embed.

**Figures** — `plots/s4b_feature_6to9.png` (both views, cross-seed-stable 6→9),
`plots/s4b_lda_contact.png` and `plots/s4b_margin_contact.png` (all 19 transitions),
`plots/s4b_plane_diagnostics.png`. All green-free per rule 13, with a second identity channel on every
series (linestyle/marker per digit role, `//` hatch on the third digit's decision region, drawn
outlines between decision regions, digits printed inside them).

**Verification** — REPORT.md through `POST api.github.com/markdown`: **16/16** display equations render
as `js-display-math`, 0 as `<pre lang="math">`, 0 KaTeX errors, 13 `<img>`; RESULTS.md 13 images, 0
KaTeX errors; no bare `(plots/*.png)` path outside an embed in either file; inline-math
backslash-punctuation grep clean in both.
