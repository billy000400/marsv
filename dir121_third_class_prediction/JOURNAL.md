# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-07-25 — iter 1 (part A): preregistration of the frozen decision rules

Written BEFORE running or viewing any 45-transition census output, as PLAN.md requires. No feedback
files (`human_feedback*.md` / `*REVIEW*`) exist in this direction; checked with `ls`.

**Source artifacts (read-only, direction 12).** Model = `src.mnist.MLP(depth=4, width=200,
activation='relu')` (784→200→200→200→10). Checkpoints
`../dir12_plateau_during_training/results/full_mnist_from_scratch/seed_{0,1,2}/ckpts/step30000.pt`
(step30000 is the final checkpoint present for every seed). Protocol imported verbatim from
`../dir12_plateau_during_training/experiments/plateau_protocol.py`: 50-point norm-rescaled SLERP on
post-ReLU `h1`, patch at `h1`, propagate, logit-space relative endpoint distance
`d(a) = ||x(a)-x(0)|| / (||x(a)-x(0)|| + ||x(a)-x(1)||)`.

**Pair bank (frozen).** For each of the 45 unordered pairs (a<b): rank-i class-a test image paired
with rank-i class-b test image, i=0..99, ranks taken inside the first 2,000 MNIST test images —
the same construction as `avg_transition_curves.build_100_bank`, extended from 10 to all 45
transitions. 4,500 endpoint index pairs saved once and reused for every seed.

**FROZEN RULE 1 — stable third-class prediction.** A third-class segment is a maximal run of
consecutive alpha points whose argmax prediction is neither endpoint digit. Transition (a,b) is
labelled *stable third-class* iff there exists a digit z (z != a, z != b) such that (i) z appears as a
third-class segment in >= 50 of the 100 paths, and (ii) among those paths, the median longest-z-run
length is >= 3 consecutive alpha points. Prevalence sensitivity is reported at 25% / 50% / 75%.

**FROZEN RULE 2 — stable sub-plateau (shelf rule).** Applied to the pointwise mean curve
`m[0..49]` over the 100 pairs. A *flat run* is a contiguous index range [i,j] with
`max(m[i..j]) - min(m[i..j]) <= 0.05` and length `j-i+1 >= 5` alpha points (>=10% of the path).
The run is *intermediate* iff its mean level lies in [0.15, 0.85], i.e. it is separated from both
endpoint plateaus (`m<0.1` near alpha=0, `m>0.9` near alpha=1). Transition (a,b) is labelled *stable
sub-plateau* iff at least one maximal flat run is intermediate. Flanking rise is implied: m[0]=0 and
m[49]=1, so an intermediate level in [0.15,0.85] is necessarily preceded by lower and followed by
higher values. Reported shelf statistics: alpha interval and d-level of the widest intermediate run.

**FROZEN RULE 3 — robustness of the sub-plateau label.** (a) leave-one-out: drop each of the 100
pairs in turn, recompute the mean curve, re-apply Rule 2; the label must be unchanged in 100/100.
(b) bootstrap: 1,000 resamples of the 100 pairs with replacement; the label must agree with the
full-sample label in >= 90% of resamples.

Thresholds (0.05 flat band, 5 points, [0.15,0.85] level, 50% prevalence, median run >= 3) are fixed
now and will not be tuned after seeing results. Alternatives rejected: (i) a derivative-based shelf
rule (`|dm/dalpha|` below a quantile) — rejected because the derivative scale differs per transition
and the threshold would end up tuned; (ii) declaring a shelf from per-path curves and voting —
rejected because PLAN.md names the mean curve as primary evidence. Both are recorded here so the
choice is auditable.

On track? yes — S1 0% done (rules frozen, census not yet run).

## 2026-07-25 — iter 1 (part B): ran S1–S5 and wrote the first deliverables

**What I did.** After freezing the rules (part A above), implemented and ran three scripts:

- `experiments/s1_census.py` — imports direction 12's `plateau_protocol` unchanged, builds the 45×100
  endpoint bank, evaluates all 4,500 paths at seeds 0/1/2 step-30000, saves `results/s1_census.npz`
  with per-path predictions, `d(alpha)`, endpoint predictions, indices, checkpoint paths and SHA-256.
  Endpoint accuracy over the 9,000 bank endpoints: 0.987 / 0.979 / 0.980.
- `experiments/s1_analyze.py` — applies the frozen Rules 1–3, writes
  `results/s1_classification.{json,csv}` and 5 figures.
- `experiments/s3_s4_regions.py` — builds the real-digit reference regions (2,000 train images per
  digit for mean/std; 700 held-out test images from `test[2000:]` for the 95th-percentile calibration,
  disjoint from the endpoint pool `test[:2000]`), scores every third-class segment, runs controls
  C1–C4, writes `results/s3_s4_regions.json` and 4 figures.

**What I learned.**
1. 19 of 45 transitions are stable third-class at seed 0; 6 have a sub-plateau; **zero** have a
   sub-plateau without a stable third-class prediction. The two labels are not interchangeable and
   third-class prediction is much the more common — that answers research questions 1 and 2 directly.
2. The activation-region answer (question 3) is a clean **null**: 2.5% of 14,700 segment points are
   inside the predicted digit's region. Crucially the median distance to the *nearest* of all ten
   digits is also above 1 (1.23–2.09), so these points are outside every region, not misassigned.
   Pooled, the nearest real region on a segment is digit 5 (6,108 points) far more often than the
   predicted digit 7 (1,984).
3. Unplanned but the most interesting finding: **the identity of the third digit is seed-specific** —
   7/8 at seed 0, 1 at seed 1, 2/8 at seed 2. Only 6→9 → z=8 replicates on all three seeds. This
   strengthens the null: if the third digit were chosen because the activations resemble it, the
   choice should transfer across networks trained on the same data.

**Assumptions / choices made without a human (loop mode).**
- Within-digit control pairs (C3) use rank-i paired with rank-(i + n/2 mod n) inside the endpoint
  pool. PLAN.md asks for "a fixed within-digit interpolation such as 6→6" but the pool has only
  175–234 images per class, so the rank-i/rank-(i+100) scheme used for cross-class pairs would not
  give 100 distinct pairs for every digit. Rejected alternative: use only digit 6 as PLAN's example —
  running all ten digits is strictly more informative at negligible cost.
- `s_c,j` in the normalized distance is floored at 1% of the globally-pooled per-coordinate std,
  because post-ReLU coordinates that are identically zero within a class would otherwise give an
  infinite distance. Rejected alternative: drop such coordinates — that would make the distance's
  dimensionality differ per digit and break comparability. C1/C3 validate the floored version.
- C2 is reported twice (all endpoint-predicted points, and endpoint plateau only) after the
  all-points version came out weak (median 0.37). This is not a post-hoc rescue: both numbers are in
  the deliverables, and the caveat that mid-path points are off-region *regardless* of what they are
  predicted to be is stated plainly in REPORT.md Result 6 and RESULTS.md.
- Reported three transitions (0→4, 1→9, 5→7) as borderline where the preregistered bootstrap bar
  (≥0.90) fails, rather than relabelling them. Thresholds were not touched after seeing results.

**Deliverables.** RESULTS.md and REPORT.md written from scratch (were TODO templates), 9 figures
embedded as rendered images in both. REPORT.md verified through the GitHub markdown API: 8/8 display
equations render, 0 code-block degradations, 9 img tags.

**Next step.** S4/S5 are done for `h1`, so the preregistered optional follow-up is now unlocked:
repeat the reference-region and distance comparison at `h2` and `h3` to ask whether similarity to real
`z` activations appears only after later layers. That is the single remaining scientific question; the
required Stage-1/2 outputs are all complete. Do NOT write STOP until it is either run or explicitly
declared out of budget.

On track? yes — S1/S2/S3/S4/S5 complete (~85% of the plan), S6 finalization pending the optional
`h2`/`h3` follow-up; no blocker.

## 2026-07-25 — iter 1 (part C): later-layer follow-up, plan complete

**What I did.** Wrote and ran `experiments/s6_later_layers.py`, the preregistered conditional
follow-up (PLAN.md "Optional later-layer follow-up"). Its precondition — segments predicted as z but
NOT close to real z activations at `h1` — is exactly what part B established, so it was legitimate to
run. It reuses every definition from `s3_s4_regions.py` by import and changes only the hook point;
`h1` is recomputed through the new code path and reproduces 2.5%, so the two scripts agree.

**What I learned.** The follow-up answer is no, but the interesting part is *why* the naive reading is
wrong. Inside-region fractions by layer: `h1` 2.5%, `h2` 10.6%, `h3` 0.2%. The `h2` number looks like
a partial rescue until you decompose the criterion. I added a region-occupancy metric — how many of
the ten regions contain a segment point — and it settles it: 0.08 at `h1`, **5.80** at `h2`, 0.00 at
`h3`. At `h2` the mean-and-variance regions overlap so much that "ratio to z below 1" is satisfied by
78.4% of segment points and means almost nothing; the discriminating condition (z is the *nearest*
region) sits at 11.7%, versus 14.3% at `h1` and 12.5% for a random non-endpoint digit. So `h2` gives
no evidence of a match either. At `h3` the last hidden layer clusters classes so tightly that segment
points are 175–471× the held-out 95th percentile from the predicted digit and inside zero regions,
while C1 (0.950) and C3 (0.966) still pass — the ruler is fine, the points are just nowhere near any
class.

**Judgement call.** I added the occupancy metric *after* seeing the `h2` numbers. That is a
post-hoc-added measurement, so I want it on the record: it is a diagnostic of the region summary, not
of the hypothesis, and it does not touch any frozen threshold or relabel anything. Without it I would
have had to either report "10.6% at h2" without explanation or quietly downplay it; measuring the
overlap directly is the honest option. The alternative I rejected was replacing the region summary at
`h2` with something that keeps regions separable (e.g. a shared-covariance or LDA-style distance) —
that would change the preregistered metric mid-study, so it is listed as a limitation instead.

**Deliverables.** REPORT.md gained Result 8, a fourth Summary finding, the region-occupancy metric in
Methods (with a rendered equation), an updated hook-point paragraph, a rewritten limitation (iii) and
an updated Reproducibility list. RESULTS.md gained the later-layer table and figure and an updated
headline. Both re-verified through the GitHub markdown API: 9/9 display equations render, 0
degradations, 10 images each.

**State of the plan.** All six stages are now done: S1 census, S2 cross-seed, S3 reference sets, S4
region analysis on all 19 stable transitions, S5 controls + three-way verdict, S6 finalization. Every
item in the PLAN.md success criterion is present in the deliverables, and the optional follow-up is
run rather than deferred. Re-checked the direction root for `human_feedback*.md` / `*REVIEW*` files
without the `.addressed.md` suffix: none exist, so writing `STOP` is permitted under CLAUDE.md rule 11.

**Next step if this direction is ever resumed.** The two open threads are both out of the current
scope: (a) why each seed picks its own third digit (7/8, 1, 2/8) — that is a question about what the
between-cluster region of `h1` looks like at initialization; (b) a region summary that stays
separable at `h2`, which would sharpen Result 8 from "no evidence" to a positive test.

On track? yes — S1–S6 complete (100% of the plan), all controls pass, no unaddressed feedback, STOP written.

## 2026-07-25 — iter 2: figures rebuilt CVD-safe; STOP re-created (it was missing)

**Why this iteration existed at all.** I re-entered with the plan already complete. Checked the
direction root first: no `human_feedback*.md` / `*REVIEW*` files at all, addressed or otherwise. But
`STOP` was **absent** even though the previous entry says it was written — so either it never landed
on disk or something removed it. Rather than just re-touching the file, I audited the deliverables
against CLAUDE.md and found a hard-rule violation worth an iteration.

**What I did.** Fixed CLAUDE.md rule 13 (red-green colour deficiency) across all ten figures. The
audit hits were unambiguous: the Stage-1 category palette had green `#54A24B` vs orange; the C2
control histogram distinguished its two series *only* by matplotlib `C2` vs `C3` — literally green
against red; `s6` encoded the three hook points as `C0`/`C1`/`C2` (blue/orange/green); the
segment-width threshold was a red line; `s4_region_membership` used black-vs-red markers; and `tab10`
(which contains both a green and a red, and is not monotone in grayscale) carried digit identity in
three figures. On top of the palette problem, several series had colour as their *only* identity
channel, and every caption in both deliverables named series by colour ("the red triangles", "green =
endpoint plateau only"), which rule 13 forbids outright because it conveys nothing to this reader.

Added `experiments/cvd_style.py` with the mandated green-free palette and shared category/hatch/marker
maps, rewired all four scripts, and gave every series a second channel (hatch for paired bars and
category cells, linestyle+marker for line series). For the ten digit classes — more than the five-hue
palette allows — I used the sequential `cividis` ramp and printed the digit inside every band or cell
big enough to hold it, so identity never rests on hue. Then re-ran all three analysis scripts and
rewrote all twenty captions plus the body prose that referred to colours.

**What I learned / verified.** All three result JSONs `diff` clean against the pre-edit copies
(`s1_classification.json`, `s3_s4_regions.json`, `s6_later_layers.json`), so the re-runs are exactly
reproducible and no number in the deliverables moved — this was purely a presentation fix. Re-checked
REPORT.md through the GitHub markdown API: 9/9 display equations render, 0 degraded to code blocks, 10
images; RESULTS.md 10 images; no bare `(plots/*.png)` path outside an embed in either file.

**Judgement calls (loop mode, no human to ask).**
- Kept the *numbers* frozen and touched only styling and captions. The alternative — regenerating
  figures from scratch with a different layout — would have risked changing what the figures show
  while the deliverables claim a stable current-best result. Rejected.
- Used a sequential ramp rather than five hues + "other" for the ten digits. Rule 13's fallback for
  >5 series is small multiples or folding the tail into grey, but both would destroy the point of the
  class-composition plot (you need to see *which* digit wins). A monotone CVD-safe ramp plus printed
  digit labels satisfies the actual requirement — grayscale-readable, identity not colour-dependent.
- Added one motivating sentence before each figure in RESULTS.md while I was there (rule 12 asks for
  motivation before every figure; RESULTS.md previously had ten captions in a row). REPORT.md already
  motivated each figure in its Results prose, so I only added the Figure-conventions note there.

**Next step.** None required. The plan is complete, the deliverables now satisfy rules 12 and 13, and
`STOP` is written — this time verified present on disk. If new feedback arrives, rule 11 applies:
delete `STOP`, address it, re-write `STOP` only when clean. The two open threads noted last iteration
(why each seed picks its own third digit; a region summary that stays separable at `h2`) remain
deliberately out of scope.

On track? yes — S1–S6 complete (100% of the plan), figures now CVD-compliant, no unaddressed feedback,
STOP present on disk.

## 2026-07-25 — iter 3 (part A): S4b preregistration (written before running anything)

The direction was reopened: PLAN.md now carries **S4b/S7**, replacing the PCA view with (A) a
real-class LDA plane and (B) a path-local margin-gradient/SVD decision slice. `STOP` is absent from
disk, consistent with the reopening. Re-listed the direction root first: no `human_feedback*.md` /
`*REVIEW*` files exist, addressed or otherwise.

Choices frozen **before** looking at any S4b output, so they cannot be tuned to a nicer picture:

- **Representative path (medoid rule).** Among the paths of the 100-pair bank that contain a segment
  of the dominant third digit z, represent each path by its full 50x200 `h1` trajectory flattened to
  a 10,000-vector; the medoid minimises the sum of Euclidean distances to the other qualifying paths.
  Rejected alternatives: medoid of the segment mean only (throws away the approach and exit, which is
  exactly what these figures are about) and longest-segment path (that is "hand-picking the clean
  one" by another name, which PLAN.md forbids).
- **LDA fit.** Three-class LDA on 2,000 REAL TRAINING images per class for a, b and z — the same
  reference index sets S3 used, so nothing new is sampled. `S_W` gets a ridge of
  `1e-3 * tr(S_W)/200` (post-ReLU `h1` has near-dead coordinates, so `S_W` is ill-conditioned);
  axes = the top two generalised eigenvectors, rescaled to unit L2 norm in `h1` space. Held-out real
  test activations (700/digit from `test[2000:]`) and all interpolation paths are projected without
  refitting.
- **Spread ellipses.** 2 standard deviations of the projected HELD-OUT real points. The fraction of
  held-out real points inside their own ellipse is reported as a calibration number, so the ellipse
  is not just decoration.
- **View B point set.** The medoid path's z-segment plus PAD = 5 alpha points either side (clipped to
  the path), so both margin boundaries are inside the stacked gradients.
- **View B plane.** Stack `grad_h(l_a - l_z)` and `grad_h(l_b - l_z)` over those points, SVD, take the
  top two right-singular vectors; anchor at the mean `h1` of the segment. Report the singular values
  and the two-axis share of the squared gradient norm.
- **View B grid.** 161x161 covering the projected medoid path and the projected held-out real class
  means, padded 30% on each side; grid points are **never clamped** to `h >= 0`, and the fraction of
  cells with any negative coordinate is reported as required.

One measurement is added on purpose: the fraction of third-class segment points that fall inside the
2 s.d. real-z LDA ellipse, computed over ALL 100 paths, next to the already-published full-space
fraction. It is the 2-D analogue of the S4 region test, so it says directly how much a two-dimensional
supervised projection flatters the result. It does not replace or relabel anything.

On track? yes — S4b 0% done (rules frozen, nothing run yet).

## 2026-07-25 — iter 3 (part B): S4b run on all 19 transitions; plan complete again

**What I did.** Implemented `experiments/s4b_planes.py` to the rules frozen in part A and ran it on all
19 seed-0 stable transitions (PLAN.md allows a preregistered subset; none was needed — the whole thing
takes about a minute on the shared GPU). Removed the PCA block from `s3_s4_regions.py` and the
`s4_pca_view.png` figure, re-ran that script, and confirmed `results/s3_s4_regions.json` is byte-identical
to the pre-edit copy, so nothing previously reported moved. Then curated both deliverables: new Methods
block, new Result 5, old Results 5–8 renumbered 6–9 with every cross-reference chased down.

**What I learned.**
1. The supervised LDA plane is *stricter* than the full-space test, which I did not expect: 0.02% of
   the 14,700 segment points inside the 2 s.d. real-z ellipse against 2.5% in 200 dimensions. The
   worry a reviewer would have — "your null is an artifact of measuring distance in 200 dimensions,
   where everything is far from everything" — is answered by the one projection built specifically to
   make the three digits as separable as possible. The picture is also unusually clean: in all 19
   panels the paths form a corridor between the two endpoint ellipses and the third-digit ellipse sits
   off to the side, untouched.
2. View B supplies the piece the direction was missing: a *decision* region for z genuinely straddles
   the path (1.7%–37.7% of the plotted window, entered in all 19), and the medoid path's entry and
   exit land on the two zero-margin contours. So the prediction is not noise — the network really does
   own a region for that digit there. "Classifier region" versus "data region" is now shown, not just
   asserted, which is exactly the distinction PLAN.md asks the report to keep separate.
3. The margin gradients are nearly two-dimensional (two-axis energy 96.2%–99.5%), so the plane is a
   fair slice rather than a lossy cartoon. That is a real diagnostic result, not a formality.

**Two things I caught by checking rather than assuming.**
- The strict off-ReLU-support statistic saturates: *every* grid cell in *every* transition has at
  least one negative coordinate, which is unsurprising in 200 dimensions and conveys nothing on its
  own. I added two graded companions (mean fraction of negative coordinates, 25.4%–34.9%; median share
  of the norm carried by the negative part, 8.2%–13.4%) and report all three. The saturated number
  stays in — it is the honest headline — but it is now interpretable.
- I had written "the medoid path enters the z-region in all 19" from looking at the figures. With a
  median in-plane energy share of only 12.6% for the path, that was a claim about the *drawing*, not
  about the model. So I measured it: collapse each path point into the plane, re-classify, compare
  with the true 200-d prediction. Segment predictions are unchanged in 14 of 19 transitions and ≥83%
  preserved in 18 of 19 (worst 1→6 at 57.9%), and 6→9 is exact on all 50 points. The claim now stands
  on a number. The same check does *not* license anything about the projected real activations, which
  keep only 4.5% of their displacement in the plane — the report says so explicitly and calls them
  shadows.

**Judgement calls (loop mode).**
- Ran the LDA/SVD views for all 19 transitions rather than the "preregistered representative subset"
  the fallback allows, because compute was not the constraint. 6→9 is featured in the main text as
  PLAN.md directs; the other 18 are contact sheets.
- Deleted `plots/s4_pca_view.png` and its generating code rather than leaving it orphaned. PLAN.md S7
  says to replace the figure and discussion; a stale PNG that no deliverable embeds is exactly the
  kind of thing that gets re-embedded by mistake later. Git history keeps it.
- Renumbered Results 5–8 to 6–9 rather than inserting a "Result 4b". The report is a presentable
  deliverable, and the aggregate-then-visualize order PLAN.md prescribes puts the planes right after
  Result 4.
- For the decision-region colouring I kept the sequential `cividis` ramp (ten classes exceed the
  five-hue palette) but added drawn outlines between regions and a `//` hatch on the third digit's
  region, because neighbouring high digits are close in that ramp and the third digit's region is the
  one thing the figure exists to show. Identity never rests on hue.

**State of the plan.** S4b and S7 are done; S1–S6 were already complete. Every item of the PLAN.md
success criterion is present, including both required 2-D views with their diagnostics. Re-listed the
direction root for `human_feedback*.md` / `*REVIEW*` files without `.addressed.md`: none exist, so
`STOP` is permitted under CLAUDE.md rule 11 and is written.

**Next step if resumed.** Unchanged from before, plus one new thread the planes suggest: the third
digit's decision region is large in the slice (median 31.9% of the window) while its data region is
nowhere near it, so "how big is each class's decision region in the between-cluster part of `h1`, and
does its size predict which digit a seed defaults to?" is now a well-posed question. Still out of the
current scope.

On track? yes — S1–S7 complete (100% of the plan), 13 figures embedded in both deliverables, 16/16
equations verified rendering, no unaddressed feedback, STOP written.
