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
