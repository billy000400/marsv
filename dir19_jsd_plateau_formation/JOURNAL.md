# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## Iteration 1 — 2026-08-05

**Feedback check (CLAUDE.md Part C).** Listed the direction root for `human_feedback*.md` and
`*REVIEW*` without the `.addressed.md` suffix: **none present**. (There is an unaddressed
`human_feedback_6.txt` in `dir18_continuation_jsd_plateau`, but that is another direction's file and
out of scope here — PLAN.md rule: dir18 is read-only upstream.) Proceeded with the plan.

**What I did.**
1. S1: copied the frozen manifests, corpus manifests and existing checkpoint artefacts from dir18,
   hashed all 25 into `results/INHERITED_HASHES.txt`, and re-ran the step-0 assay with this
   direction's code as a compatibility check. It reproduced dir18 **bit-for-bit** (max |Δd| = 0),
   which is the strongest form of the check the plan asked for.
2. Extended the assay with the plan's measurement 5 (neighbouring-position full-output JSD movement
   profile, computed from log-softmax and streamed — no full-vocabulary logits ever saved), plus the
   frozen-sample held-out loss and the published learning rate.
3. S2/S3/S4: ran the frozen 60-pair bank at 20 checkpoints (~165 s each: ~100 s download + ~60 s
   assay + loss), one checkpoint on disk at a time.
4. S5: ran the frozen 1,000-pair bank at step 64000 (309 s) and combined with dir18's existing
   step0/step143000 large-bank runs for the prespecified 64k → final test with an endpoint-clustered
   dyadic bootstrap.
5. Wrote `RESULTS.md` and `REPORT.md` from scratch (they were placeholders), six embedded figures,
   `check_render.py` passes.

**What I learned.**
- The headline is stronger and cleaner than the plan anticipated: the two events are **~60× apart**
  in training steps, and in the counter-intuitive order. Corpus divergence ranks the pairs by
  step 32, when median width is 0.827 against 0.831 untrained and the whole ordered spread is 0.006.
  Global shape arrives between step 1000 and 2000.
- The interval-specific test is what carries the argument. ρ(J, Δw) = −0.466 over step 8 → 32 with
  essentially no sharpening, versus +0.035 over step 512 → 1000 which produces the largest single
  sharpening event. Cross-sectional ρ alone could not have distinguished these.
- Movement concentration lands on the *shape* timeline, not the ordering timeline, and total movement
  grows 85× while concentrating — the model moves much further in a much smaller region.
- **Checkpoint QC matters.** Revision `step16` of pythia-1.4b-deduped is a mislabelled copy of the
  final model (bit-identical curves, loss 2.320 among neighbours near 9, safetensors 32 bytes
  smaller). Had I not evaluated a held-out loss per checkpoint, this would have entered the
  trajectory as a spectacular fake "phase transition" at step 16 — exactly the artefact the plan
  warns against calling a phase transition. Kept the assay output and documented it as a result.
- Incidental: step 0 and step 1 produce bit-identical curves and losses agreeing to 6 decimals. That
  is expected at lr = 1.4e−7 (one warmup step) and is not flagged as an anomaly.

**Assumptions logged (loop mode, no human to ask).**
- Primary corpus divergence = `jsd_B`, the held-out split. Rejected: `jsd_A` (the selection split,
  which would be circular). The two agree at Spearman 0.9998, so this is not a consequential choice,
  and it reproduces dir18's primary numbers exactly.
- Held-out loss sample = last 256 rows of split B truncated to 512 tokens. Rejected: full 2049-token
  rows (logits alone would be 412 MB per row at fp32, over the 7.2 GB VRAM share). Truncation is
  fine because the metric is only training-progress context, held frozen across checkpoints.
- Learning rate computed analytically from the published Pythia schedule (peak 2e−4, 1430-step linear
  warmup, cosine decay to 0.1× peak). Rejected: reading it from checkpoint optimiser state, which is
  not in the released `model.safetensors`-only downloads.
- `valid` target-ID list taken from `/tmp/dir18_data/reference_valid.npy` (50,060 IDs) because
  `reliability_bank.npz` was no longer on local disk. Verified it matches the manifest's
  `n_valid_targets = 50060`, and the bit-exact step-0 reproduction confirms it is the same list.
- step16 excluded rather than kept with a caveat, because keeping it would corrupt every trajectory
  and every onset rule. Its raw artefacts are retained, and the exclusion is itself reported as
  Result 6 with its evidence, so nothing is hidden.

**Next step.** The plan's S1–S5 are done and the success criteria are met. The remaining useful work
is resolution and robustness, in priority order: (a) fill the step 16 → 32 gap left by the excluded
revision — Pythia has no released checkpoint between them, so the honest option is to keep the
bracket as "after 8, by 32" and say so; (b) run the 1,000-pair bank at the two ordering-bracket
checkpoints (step 8 and step 32) to validate the onset at scale, which is the one prespecified
validation not yet done (~6 min of GPU each) ; (c) add step 3000/6000 zoom-ins around the shape
onset, labelled as follow-up resolution. None of these change the current verdicts, so the
deliverables are complete as they stand.

On track? yes — S1–S5 complete (~90% of plan), no blocker; remaining work is optional resolution,
chiefly the 1,000-pair validation at the step 8 / step 32 ordering bracket.

### Addendum — same iteration

Ran the 1,000-pair bank at step 8 and step 32 (~300 s each), which was item (b) on my next-step list
and the last prespecified validation the plan required. It replicates the onset bracket: ρ contains
zero at step 8 (−0.021 [−0.132, +0.104]) and excludes it at step 32 (−0.149 [−0.286, −0.011]), with
no sharpening on that bank either (median w 0.830 / 0.828 vs 0.831 at step 0).

Learned: the step-32 effect is roughly 3× weaker on the large bank than on the controlled one, though
the two agree by the final checkpoint. That is the expected direction — the large bank is unmatched
on frequency and surprisal and reuses endpoint tokens — but it is a real limit on how strong the
early-ordering claim can be, so I stated it in both deliverables rather than reporting only the
replication. Added `plots/large_bank_onset.png` as Figure 6 and renumbered the QC figure to 7.

Did NOT write `STOP`. The plan's success criteria are all met, but useful resolution work remains
(zoom-in checkpoints around the step 1000 → 2000 shape onset, e.g. step 3000/6000, labelled as
follow-up resolution), and leaving the loop open costs nothing while a premature STOP would make the
direction silently ignore any feedback dropped later.

On track? yes — S1–S5 complete plus the large-bank onset validation (~95% of plan), no blocker;
remaining work is optional zoom-in resolution around the shape onset.

**Housekeeping fix (same iteration).** While taking a final inventory I regenerated
`results/INHERITED_HASHES.txt` over *all* 97 artefacts, which destroyed its purpose (S1 asks for
hashes of the frozen upstream inputs specifically). Restored it to the 22 files copied from dir18,
and moved the full inventory to `results/ALL_ARTIFACT_HASHES.txt`. Also re-verified with `cmp` that
all 22 inherited files are still byte-identical to their dir18 originals — they are, so nothing in
the read-only upstream was touched.

---

## Iteration 2 — 2026-08-06

**Feedback check (CLAUDE.md Part C).** Listed the direction root for `human_feedback*.md` and
`*REVIEW*` lacking the `.addressed.md` suffix: **none present**. Proceeded with the plan.

**What I did.** Grepped the whole direction for "permut" and found nothing — so PLAN's S5 requirement
of an "endpoint-label permutation" on the 1,000-pair bank was the one prespecified analysis never
implemented. Every interval in the deliverables was a bootstrap. Wrote `experiments/permtest.py`
(20,000 permutations, CPU only, reads already-saved curves — no GPU, no re-assay) covering three
tests, plus `experiments/plot_perm.py` for the figure.

**What I learned.**
- The test I most wanted turned out to be the one that pays off. ρ = −0.149 at step 32 on the
  1,000-pair bank was the weakest number in the report — a clustered bootstrap CI of
  [−0.286, −0.011] that barely excluded zero. Under the endpoint-label (QAP) null it gives
  p = 0.0031, and p = 0.0082 even after paying for all five checkpoints.
- The QAP null also *quantifies* the clustering penalty, which the bootstrap could only assert:
  relabelling the 123 endpoint tokens produces |ρ| up to 0.09 by chance, against ~0.062 for 1,000
  genuinely independent pairs. So the bank behaves like ~450 effective independent pairs. That number
  is more informative than the CI it corroborates, and it is now in the report.
- The dissociation claim survives multiplicity in the strongest form available: ρ(J, Δw) at step
  8 → 32 has family-wise p = 0.0035 across all 18 intervals, while the largest sharpening event
  (512 → 1000) sits at p = 0.78. Applying ONE permutation across the whole trajectory (rather than an
  independent permutation per checkpoint) is what makes a family-wise statement legitimate here — it
  preserves the across-checkpoint dependence of the real trajectory.
- One reading had to be walked back, which is the point of running the test: step 4000 → 8000 had
  ρ = +0.258 with a bootstrap CI [+0.002, +0.483] that just excluded zero. Permutation p = 0.045,
  family-wise p = 0.55 — indistinguishable from a relabelling. Both deliverables now call that
  interval divergence-blind rather than reversed. No headline number moved.

**Assumptions logged (loop mode, no human to ask).**
- QAP null draws each pair's divergence from the *whole* 123×123 matrix, whereas the observed pairs
  were stratified into selection-split JSD quintiles, so the null's marginal J distribution is
  slightly broader than the observed one. Rejected the alternative of permuting only within strata:
  it would leave much of the endpoint→width association intact and make the test anti-conservative.
  Spearman is rank-based, so a marginal shift is second-order; the null envelope (0.09) is if
  anything conservative. Stated in Methods as a quadratic-assignment permutation.
- Scored observed and null ρ on the large bank from the same source (the stored JSD matrix) so that
  the JSON's rounding cannot differ between them; verified matrix-based ρ equals `jsd_B`-based ρ to
  6 decimals at all five checkpoints, and the matrix reproduces `jsd_B` to 5.0e−7.
- B = 20,000 permutations. Enough to resolve p down to 5e−5, which is below every p reported; a
  larger B would only sharpen the "<0.001" entries, which no claim depends on.
- Interval step 0 → 1 is degenerate (bit-identical curves, Δw ≡ 0, no rank correlation exists). It is
  marked degenerate and excluded from the family-wise maximum rather than being assigned ρ = 0, which
  would have deflated the null envelope.
- Did NOT edit `PLAN.md`: it declares itself operator-owned and read-only, which overrides the generic
  loop instruction to tick its checkboxes. Progress is recorded here and in CHANGELOG.md instead.

**Incidental fix.** Every figure in both deliverables was embedded with a visible caption, but
Figures 4–8 (REPORT.md) and 1–8 (RESULTS.md) were never cited by number in the body prose, which
rule 12 requires. Added a by-number citation to the motivating sentence that precedes each figure.

**Next step.** The plan's S1–S6 measurements and every prespecified validation are now done, and both
inference methods agree. Remaining optional work, in priority order: (a) nothing can tighten either
bracket — Pythia released no checkpoint between step 8 and 32, nor between 1000 and 2000, so the
resolution is at the limit of this trajectory and the honest brackets are the ones reported;
(b) a second Pythia size (e.g. 410M) would test whether the ~60× separation is model-specific, but
that is a new direction under this PLAN's "out of scope" note on generalising beyond this run;
(c) nothing else in the plan is outstanding. Did not write `STOP` — the deliverables are complete,
but leaving the loop open costs nothing and a premature STOP would make the direction silently ignore
any feedback dropped later.

On track? yes — S1–S6 complete including the last prespecified S5 inference (~100% of plan), no
blocker; the direction's own resolution limit, not budget, bounds any further tightening.

---

## Iteration 3 — 2026-08-06

**Feedback check (CLAUDE.md Part C).** Listed the direction root for `human_feedback*.md` and
`*REVIEW*` lacking the `.addressed.md` suffix: **none present**. Proceeded with the plan.

**What I did.** The plan's S1–S6 and every prespecified validation were already complete, so I looked
for the strongest question the deliverables left open rather than for a plan checkbox. It was this:
the report showed corpus divergence ranks the pairs by step 32, but never checked whether that
step-32 ranking is the ranking the trained model ends up with. Wrote `experiments/persistence.py`
(CPU only, reads the saved per-pair trajectories — no GPU, no re-assay) computing
π(s) = Spearman(w_s, w_final), its partial version with J removed, the 20×20 checkpoint-agreement
matrix, a paired bootstrap, and a 20,000-draw label-permutation null; plus `plot_persistence.py`.

**What I learned.**
- There are **three** clocks, not two. The per-pair ranking only becomes the final ranking between
  step 64 and step 128 (π = 0.207 → 0.437, p_fw = 0.0053), which sits between the divergence-ordering
  onset (8 → 32) and the shape onset (1000 → 2000). I did not expect a third bracket to be this clean.
- The partial correlation is what makes the result interpretable, and it is the number I would keep
  if I could keep only one: π⊥ = −0.082 at step 32. Everything the step-32 ranking shares with the
  final ranking is the divergence-aligned component; there is no pair-specific agreement at all yet.
  That *tightens* the report's claim rather than denting it — corpus divergence is not merely present
  before the plateau shape, it is the first component of the final ordering to appear.
- Running the attenuation control before writing anything was the right call. "π is low because w is
  measured on a 0.006-wide spread" is the obvious objection and it would have been fatal if true. The
  three carrier sentences agree at r̄ = 0.830 at step 32 (reliability 0.936, ceiling π_max = 0.935),
  so the disagreement with the final ranking is real. Note the reverse reading is also informative:
  even at step 0 reliability is 0.872, so the untrained network ranks pairs consistently — that
  ranking is simply not the final one.
- One Summary sentence had to be walked back: "later training ... largely preserving that early
  ranking". π = 0.161 does not support "largely preserving". What is preserved is the divergence
  alignment, not the ranking. Fixed in both deliverables; no number moved.

**Assumptions logged (loop mode, no human to ask).**
- Reference checkpoint for π is step 143000 (the final model). Rejected step 64000, which has the
  narrowest median width: "final" is the model anyone would actually deploy or interpret, and the
  late-widening result (Result 5) means the two are not interchangeable. The agreement matrix in
  Figure 8B shows the whole picture either way, so the choice only sets which column is read as 1.0.
- The permutation null relabels the 60 pairs of w_s against w_final, one permutation reused across
  the whole trajectory (same convention as `permtest.py`) so the family-wise statement respects
  across-checkpoint dependence. The reference checkpoint correlates with itself by construction and
  is excluded from the null summaries and from the family-wise maximum; including it would have
  pinned the null envelope at 1.0 and made every p-value meaningless.
- Reliability from the 3 carrier sentences, Spearman–Brown corrected to k = 3. Rejected a split-half
  over interpolation positions (the 50 positions of one curve are not independent measurements of the
  same quantity, so it would overstate reliability).
- Placed the new result as Result 8, before the step16 data-integrity finding, rather than after
  Result 3 where it belongs thematically. Reason: it uses Result 7's permutation machinery, and
  inserting it earlier would have renumbered six results and six figures for no reader benefit.
- Did NOT edit `PLAN.md` (operator-owned, declares itself read-only), same as iteration 2.

**Next step.** Nothing in the plan is outstanding and both onset brackets are at the resolution limit
of this released trajectory (no Pythia checkpoint exists between 8 and 32, 64 and 128, or 1000 and
2000). The remaining questions all need a second training run or a second model size, which this
PLAN puts out of scope. If the loop continues, the most useful in-scope work left is a robustness
check of the third clock against the reference choice — recomputing π against step 64000 instead of
step 143000 — which the agreement matrix already suggests will not move the bracket. Did not write
`STOP`: the deliverables are complete, but a premature STOP would make the direction silently ignore
any feedback dropped later, and that costs more than leaving the loop open.

On track? yes — S1–S6 complete plus a new in-scope result (~100% of plan), no blocker; the released
checkpoint spacing, not budget, bounds any further tightening.

---

## Iteration 4 — 2026-08-06

**Feedback check (CLAUDE.md Part C).** Listed the direction root for `human_feedback*.md` and
`*REVIEW*` lacking the `.addressed.md` suffix: **none present**. Proceeded with the plan.

**What I did.** The plan's S1–S6 are complete, so I looked for the weakest *evidence* in the
deliverables rather than a missing measurement. It was Result 9. The report asserts that Hugging Face
serves the wrong artefact under revision `step16` of `EleutherAI/pythia-1.4b-deduped` — a claim about
someone else's published data — on three pieces of circumstantial evidence: a loss outlier, curves
bit-identical to `step143000`, and a file 32 bytes smaller. None of that excludes a corrupted local
download, and none of it says what the artefact actually is. Wrote `experiments/step16_forensics.py`
(streams each revision's safetensors tensor payload straight from the Hugging Face CDN and hashes it,
plus per-tensor SHA-256 over a 10-tensor sample; ~11 GB streamed, nothing written to disk) and
`experiments/revision_audit.py` (header + published-digest audit of all 21 revisions, ~1 MB).

**What I learned.**
- The payload SHA-256 of `step16` **equals `step143000`'s exactly** (`fbd54ccec4e0f5ee…`), over all
  2.63 GiB; `step8` and `step32` differ. All 10 sampled tensors match `step143000` byte for byte and
  0 of 10 match either neighbour. The claim is now "it *is* the final model's parameters", not "it is
  not a step-16 model" — a stronger and much harder statement to argue with.
- The 32-byte deficit had a mundane explanation worth publishing: `step16`'s header omits the
  `__metadata__` field (`{"format": "pt"}`) every other revision carries. That makes the defect
  detectable with a **34 KB range request** — no download, no GPU. This is the part of the finding
  most useful to other people.
- The audit is exact and nearly free, which I did not expect. Because the other 20 revisions share a
  byte-identical header layout, equal weights would force equal file digests, so the Hub's published
  SHA-256 alone proves all 20 are distinct checkpoints. That closes the question a reader would ask
  next — is anything else duplicated, e.g. among the closely spaced late checkpoints where a
  loss-outlier check would not fire? — without streaming another 56 GB.
- Consequence for the headline, now stated in both deliverables: since no genuine step-16 weights are
  published, the ordering bracket **cannot** be narrowed below step 8 → 32 from released artefacts.
  That converts a nagging "why didn't you use step 16?" into a documented resolution limit.
- Method note for the future: verify claims about third-party artefacts at the source. The local
  cache had been consistent with the behavioural evidence all along, but so would a bad download.

**Assumptions logged (loop mode, no human to ask).**
- Streamed the full payload for `step8`, `step16`, `step32`, `step143000` only (~11 GB). Rejected
  streaming all 21 (~56 GB, ~35 min): the header-layout argument makes the remaining 17 exact from
  published digests alone, so the extra bandwidth would buy nothing.
- The 10 sampled tensors are the embeddings, the unembedding, the final layer norm, and the attention
  output weight and bias of blocks 0, 11 and 23 — chosen to span the depth of the network, so a
  partial match (e.g. only early blocks copied) would have shown up. The whole-payload digest makes
  the sample redundant as proof; it is there so the match can be localised and read off a figure.
- Compared payload (post-header) digests rather than whole-file digests, because `step16`'s header
  legitimately differs. Using whole-file digests would have hidden the result — the published file
  digests already differ, which is exactly why this needed streaming.
- Did NOT edit `PLAN.md` (operator-owned, declares itself read-only), same as iterations 2 and 3.

**Next step.** Running a metric-definition robustness check: every headline number rests on
$w = t(0.9) - t(0.1)$, and a reader should know the three onsets are not an artefact of the 10/90
threshold choice. Recomputing $\rho_s$, the onset brackets and median width at alternative thresholds
from the already-saved curves (CPU only). Did not write `STOP` — deliverables are complete, but a
premature STOP would make the direction silently ignore feedback dropped later.

On track? yes — S1–S6 complete plus a hardened Result 9 (~100% of plan), no blocker; the released
checkpoint spacing, now proven irreparable at step 16, bounds any further tightening.

### Addendum — same iteration

Ran the width-definition robustness check flagged as the next step above. Wrote
`experiments/threshold_robustness.py` and `experiments/plot_threshold.py`; new Result 10 / Figure 10
in both deliverables.

**What I learned.**
- The ordering onset is *completely* insensitive to the metric convention: the same step 8 → 32
  bracket at all five level pairs, with $\rho_{32}$ moving only from $-0.428$ to $-0.385$ and the
  interval statistic from $-0.466$ to $-0.452$. I expected some drift and got almost none, which is
  the strongest thing this check could have returned for the headline.
- The shape onset is *not* invariant, and that is the useful half. The three wider levels
  ($a \ge 0.20$) look only at the steep middle of the curve and therefore detect sharpening one
  checkpoint earlier, at step 512 → 1000. So "~60×" is really "31× to 62×" depending on where the
  ruler's ticks go. I qualified the Summary rather than restating the headline: the primary
  definition is still the one the whole report is computed at, and the order of events never changes.
- Third-order finding worth keeping: $\rho(J, \Delta w)$ over the biggest sharpening interval
  (512 → 1000) grows from $+0.035$ to $+0.312$ as the band narrows. It is never negative under any
  definition, so the dissociation is safe, but at wider levels that interval mildly favours the
  *low*-divergence pairs. I did not permutation-test the four alternatives and said so, rather than
  quietly implying the $p = 0.78$ from the primary definition covers them.

**Assumptions logged (loop mode, no human to ask).**
- Curve validity stays pinned to the original 0.1/0.9 criteria at every level, so only the width
  definition varies and the same 3,600 curves enter every trajectory. Rejected re-deriving validity
  per level: it would confound a change of metric with a change of sample.
- Levels below 0.10 excluded. Validity criterion V1 guarantees only $d(0) \le 0.1$ and
  $d(1) \ge 0.9$, so a 5% level need not be attained and would produce NaN widths for an unknown,
  level-dependent subset of curves. At all five levels used, 0 of 1,200 widths were NaN.
- Reused the prespecified onset rules verbatim (two-consecutive-checkpoint requirement, simultaneous
  band from the paired trajectory bootstrap) rather than inventing a robustness-specific rule, so
  each alternative definition is judged exactly as the primary one was.
- Did NOT edit `PLAN.md` (operator-owned, declares itself read-only), same as iterations 2–3.

**Next step.** Both onsets are now at the resolution limit of this released trajectory (no Pythia
checkpoint between 8 and 32, 64 and 128, or 1000 and 2000, and `step16` is proven irreparable), and
the headline survives both a chance null and a metric-definition sweep. The remaining questions —
a second model size, a second training run — are out of scope under this PLAN. If the loop continues,
the smallest useful item left is the reference-checkpoint robustness of $\pi$ (score against step
64000 instead of step 143000), which Figure 8B already suggests will not move the third bracket. Did
not write `STOP`: the deliverables are complete, but a premature STOP would make the direction
silently ignore any feedback dropped later.

On track? yes — S1–S6 complete plus a hardened Result 9 and a new Result 10 (~100% of plan), no
blocker; released checkpoint spacing bounds any further tightening.

---

## Iteration 5 — 2026-08-06

**Feedback check (CLAUDE.md Part C).** Listed the direction root for `human_feedback*.md` and
`*REVIEW*` lacking the `.addressed.md` suffix: **none present**. Proceeded with the plan.

**What I did.** Ran the item the previous iteration flagged as the smallest useful piece of work
left: the reference-checkpoint robustness of the third clock. Result 8 measures rank agreement
between each checkpoint's per-pair widths and *step 143000's*, and concludes the ranking locks in
between step 64 and step 128. Step 143000 is where the released trajectory stops, not a point where
the model has demonstrably stopped moving — and this report's own Result 5 shows the widths still
change late (median $\Delta w = +0.0158$ over step 64000 → 143000). Wrote
`experiments/persistence_ref.py` (CPU only, ~9 s: recomputes $\pi$, $\pi^{\perp}$, a 4,000-draw
paired bootstrap and a 20,000-draw label-permutation null against five references) and
`experiments/plot_persistence_ref.py`.

**What I learned.**
- The bracket is **identical under all five references** (8000, 32000, 64000, 128000, 143000):
  after step 64, by step 128. $\pi_{128}$ spans only $+0.394$ to $+0.447$, all family-wise
  significant ($p^{\mathrm{fw}} \le 0.018$), and step 64 is non-significant under every reference
  ($p^{\mathrm{fw}} \ge 0.47$). Figure 8B had suggested this and it held exactly.
- The more interesting half is the step-32 row. $\pi_{32}$ ranges $+0.077$ to $+0.200$ and stays
  inside the chance envelope for every reference, with $\pi^{\perp}_{32} \le 0$ throughout. So the
  claim "at step 32 the model holds the divergence-aligned component of its mature ordering and
  nothing more" is about the mature model in general, not about the last checkpoint. That is a
  genuinely stronger statement than the one Result 8 could make alone.
- The reason the reference does not matter is worth stating and now is: the late widening rescales
  $w$ without reshuffling the pair order, so $\pi(w_{8000}, w_{143000}) = 0.89$ and the five
  trajectories sit on top of each other from step 128 on. Magnitude drifts late, ranking does not.
- Cheapest check in the direction so far in evidence-per-second, because everything reads from
  `per_pair_trajectories.npz`; vectorising the bootstrap ranks (`rankdata(..., axis=1)`) instead of
  looping `spearmanr` made 5 references cost less than the original single-reference run.

**Assumptions logged (loop mode, no human to ask).**
- Five references, chosen as the mature checkpoints spanning the last two decades of training
  (8000 → 143000). Rejected adding early ones (step 256, 1000): scoring the ranking against a
  checkpoint that is itself inside the formation window tests a different question and would make
  the bracket search range collapse.
- Used a family-wise permutation rule with the same two-consecutive-checkpoints requirement as the
  other two prespecified onset rules, searched only over checkpoints strictly before the reference.
  At the 143000 reference it reproduces Result 8's published bracket, which is the check that the
  rule is not doing something new.
- The bootstrap CI at step 128 under the 143000 reference comes out $[+0.231, +0.630]$ here against
  $[+0.202, +0.623]$ in `persistence.json` — a different RNG stream, not a different method. The
  deliverables keep quoting the primary run's interval for that number and use this script's numbers
  only for the four alternative references, so no published figure is silently replaced.
- Did NOT edit `PLAN.md` (operator-owned, declares itself read-only), same as iterations 2–4.

**Next step.** The plan's S1–S6 are complete and all three onsets now survive a chance null, a
metric-definition sweep and a reference sweep. Every remaining question I can name (a second model
size, a second training run, checkpoints between 8 and 32) is either out of scope under this PLAN or
impossible from released artefacts. If the loop continues, the next candidate is a carrier-sentence
jackknife — recomputing $\rho_s$ and both brackets from each single sentence frame — which would test
whether the ordering onset depends on the three particular frames inherited from dir18. Did not write
`STOP`: the deliverables are complete, but a premature STOP would make the direction silently ignore
any feedback dropped later, and that costs more than leaving the loop open.

On track? yes — S1–S6 complete plus Results 8–11 beyond the plan (~100% of plan), no blocker; the
released checkpoint spacing, not budget, bounds any further tightening.

---

## Iteration 6 — 2026-08-06

**Feedback check (CLAUDE.md Part C).** Listed the direction root for `human_feedback*.md` and
`*REVIEW*` lacking the `.addressed.md` suffix: **none present**. Proceeded with the plan.

**What I did.** Ran the item iteration 5 flagged as the smallest useful piece of work left: the
carrier-sentence jackknife. Every width in this direction is a median over three fixed sentence
frames, and that averaging is load-bearing in a way the report never tested — a divergence ordering
present in only one frame would still show up in the median, and would then be published as a fact
about training rather than about one English sentence. Wrote `experiments/sentence_jackknife.py`
(CPU only, ~40 s: per-context $w$ and $E$ with no averaging, then all three prespecified onset rules
re-run per context with their own bootstrap and permutation null) and
`experiments/plot_jackknife.py`; new Result 12 / Figure 12 in both deliverables.

**What I learned.**
- The two headline brackets are **frame-invariant**: step 8 → 32 for the ordering and step 1000 →
  2000 for the shape in every single sentence. The weakest single frame still clears the simultaneous
  band at step 32 ($\rho = -0.359$ on 60 pairs), so the ~60× separation does not need pooling to be
  visible — a stronger statement than "the median of three shows it".
- The third clock is the one that moves, and it moves the way attenuation says it must: sentence 1
  alone closes at step 256 instead of 128. Worth stating precisely rather than waving at "noise" — at
  step 128 a single context's reliability ceiling is $\pi_{\max} = 0.871$ against 0.953 for the
  median of three, and step 64 → 256 is exactly where $\pi$ is climbing through the chance envelope.
  I checked this number rather than asserting "a third less reliability", which is what I had first
  written and which was wrong by a factor of four.
- Per-context agreement $\bar r$ is 0.68–0.70 at steps 0–8, 0.83 at step 32, 0.92 at step 128 and
  0.82 at the end. The three frames agree *more* about pair widths at step 128 than at the final
  checkpoint, which I did not expect and which is consistent with the late widening adding
  context-specific magnitude drift on top of a fixed ranking (Result 11's mechanism).

**Assumptions logged (loop mode, no human to ask).**
- Jackknifed by dropping to ONE context rather than leave-one-out (median of two). Median of two is
  a mean of two order statistics and would have muddled "is one frame carrying this?" with a
  different estimator; one context is the cleanest version of the question and is also the noisiest,
  i.e. the hardest test.
- Curve validity stays at the original per-curve rules; 0 of 3,600 single-context widths were NaN, so
  every context sees all 60 pairs at every checkpoint and the three trajectories are directly
  comparable.
- Each context's $\pi$ is scored against **its own** final-checkpoint widths, not against the primary
  median-of-three final widths. Scoring against the pooled reference would mix the question "does
  this frame lock in?" with "does this frame agree with the other two?".
- Reused the three onset rules verbatim, including the two-consecutive-checkpoint requirement, so
  each frame is judged exactly as the primary analysis was.
- Did NOT edit `PLAN.md` (operator-owned, declares itself read-only), same as iterations 2–5.

**Next step.** S1–S6 are complete and all three onsets now survive a chance null, a metric-definition
sweep, a reference sweep and a carrier-sentence jackknife. Everything I can still name — a second
model size, a second training run, checkpoints between 8 and 32 — is out of scope under this PLAN or
impossible from released artefacts. If the loop continues, the remaining candidate is a pair-level
leave-one-quintile-out check on $\rho_{32}$ (does the ordering survive dropping the extreme
divergence quintile, or is it driven by the tails?), which is cheap and reads from the saved
trajectories. Did not write `STOP`: the deliverables are complete, but a premature STOP would make
the direction silently ignore any feedback dropped later.

On track? yes — S1–S6 complete plus Results 8–12 beyond the plan (~100% of plan), no blocker; the
released checkpoint spacing, not budget, bounds any further tightening.

## Iteration 7 — 2026-08-06

**Feedback check (CLAUDE.md Part C).** Listed the direction root for `human_feedback*.md` and
`*REVIEW*` lacking the `.addressed.md` suffix: **none present**. Proceeded with the plan.

**What I did.** Ran the item iteration 6 flagged as the next candidate: a leave-one-quintile-out
check on the step-32 ordering. $\rho_{32} = -0.428$ is a rank correlation over a divergence range of
0.137 to 0.942 bits, and such a number can be produced by the range's extremes alone — in which case
the report's headline ("corpus divergence starts ranking pairs between step 8 and step 32") would be
dating a two-group separation, not a graded ordering. Wrote `experiments/quintile_loo.py` (60-pair
bank: full trajectory, simultaneous band, permutation null and the prespecified bracket on each
leave-one-quintile-out subset, plus a size-matched random-drop control and per-quintile
$\Delta w$), `experiments/quintile_large.py` (same subsets on the 1,000-pair bank under its
endpoint-label QAP null) and `experiments/plot_quintile.py`. New Result 13 / Figure 13 in both
deliverables.

**What I learned.**
- The ordering at step 32 is **asymmetric in a way I did not expect**: dropping the lowest quintile
  does nothing ($-0.428 \to -0.426$), dropping the highest destroys it ($\to -0.191$, band spans
  zero, bracket slides to step 64 → 128). So it is not "the tails carry it" — it is specifically the
  top of the divergence range.
- The size-matched random-drop control is what makes this reportable rather than suggestive. Dropping
  a quintile removes range *and* 12 pairs, both of which attenuate a rank correlation. Against 4,000
  random 48-pair subsets the Q5 drop was more extreme than every single draw ($u = 1.000$) while the
  Q1 drop landed on the median ($u = 0.49$). Without that control the whole result could have been
  restated as "small samples give small correlations".
- The 1,000-pair bank turned out to be the decisive instrument here, for the reason it was previously
  described as a weakness: it fills the crowded middle of the divergence range. Its 600 middle-range
  pairs give $\rho = -0.055$ ($p = 0.35$) at step 32 and $-0.300$ ($p < 0.0001$) at step 143000 — same
  pairs, same test — so "nothing in the bulk at step 32" is a fact about the model, not about power.
  I nearly skipped this run for time; it is the half of Result 13 that survives a referee.
- Per-quintile $\Delta w$ over step 8 → 32 gives the cleanest statement of the mechanism: Q1–Q4 do
  not move (all intervals cover zero), Q5 sharpens by $-0.0057$. The "ordering onset" is one quintile
  pulling away from an undifferentiated field.
- Consequence for the story (CLAUDE.md 9b/9c step 2, narrow the claim): the timing claim is
  unchanged and still replicates, but the content claim is now "the most distinguishable pairs
  separate first, the graded ordering fills in later". I edited the Summary, the onset-table note and
  the Conclusion to say that rather than leaving the qualification buried in a robustness section.

**Assumptions logged (loop mode, no human to ask).**
- Quintiles of the 60-pair bank are the inherited `bins` from the manifest (10–14 pairs each, not
  exactly 12); on the 1,000-pair bank I cut fresh quintiles at its own 20th/40th/60th/80th
  percentiles. Rejected forcing equal counts on the small bank: that would have reselected groups
  the upstream design fixed.
- The random-drop control randomises *which* pairs are removed at fixed $n$. It deliberately does not
  hold the divergence range fixed, because range restriction is part of what deleting a quintile
  means; the control isolates sample size, and the large-bank replication covers the range question.
- On the large bank the subset is fixed at what was observed and only the endpoint labels permute, so
  pairs sharing an endpoint token stay dependent — same null as `permtest.py`, restricted.
- Scored the large bank at step 32 and step 143000 only. Step 8 would have added a "still nothing"
  row, but the informative contrast is early-versus-mature on identical pairs.
- Did NOT edit `PLAN.md` (operator-owned, declares itself read-only), same as iterations 2–6.

**Next step.** S1–S6 are complete and the three onsets now survive a chance null, a metric-definition
sweep, a reference sweep, a carrier-sentence jackknife and a divergence-subset check. The natural
follow-on from Result 13 is a within-quintile analysis of *when* the bulk relation does appear on the
1,000-pair bank (it is absent at step 32 and strong at the end; the checkpoints in between are
already measured on the 60-pair bank but only steps 0, 8, 32, 64000 and 143000 exist for the large
bank, so this would need new GPU runs at intermediate checkpoints — feasible but not cheap). Did not
write `STOP`: the deliverables are complete, but a premature STOP would make the direction silently
ignore any feedback dropped later.

On track? yes — S1–S6 complete plus Results 8–13 beyond the plan (~100% of plan), no blocker; the
released checkpoint spacing, not budget, bounds any further tightening.

## Iteration 8 — 2026-08-06

**Feedback check (CLAUDE.md Part C).** Listed the direction root for `human_feedback*.md` and
`*REVIEW*` lacking the `.addressed.md` suffix: **none present**. Proceeded with the plan.

**What I did.** Ran the follow-on iteration 7 flagged: the 1,000-pair bank had no measurement between
step 32 and step 64000, so "the graded ordering across the middle of the divergence range arrives
later" was undated. Ran `scan_large.py step256 step1000 step8000` (~5 min/checkpoint on GPU, one
checkpoint on disk at a time, VRAM fraction 0.225, 2 threads) to take that bank from five to eight
checkpoints, then wrote `experiments/bulk_onset.py` and `experiments/plot_bulk.py`. New Result 14 /
Figure 14 in both deliverables, plus a Methods paragraph and a new row in the onsets table.

**What I learned.**
- **The graded ordering brackets at step 32 → 256**, not at the plateau-shape bracket. The 600
  middle-range pairs go from $\rho = -0.055$ ($p = 0.34$) to $-0.315$ ($p^{\mathrm{fw}} < 0.0001$)
  and then do not move for the remaining 142,744 steps. That was the outcome I thought least likely
  going in — I expected the bulk relation to track the sharpening, since that is where the width
  variance comes from.
- **The single most useful number of the iteration is the median width at step 256: 0.829**, against
  0.831 untrained. The full-bank $\rho$ is $-0.548$ there, i.e. already stronger than its final
  $-0.486$, while the widths have not moved at all. So the whole divergence axis — not just its top
  end — is laid down before there is any plateau. This strengthens the report's core claim rather
  than qualifying it, which is the opposite of what Result 13 did.
- **How that is possible without sharpening:** at step 256 the top quintile is at median width 0.801
  and the other four at 0.836. The bank spreads around an unchanged median. I added the group gap
  $G_s$ = median $w$(Q5) − median $w$(Q1–Q4) to state this without going through a correlation, and
  it dates both halves of the ordering on its own: $-0.0018$ ($p = 0.0040$) at step 32, $-0.0348$ at
  step 256, a factor of ~20 apart.
- **A limit I could not remove:** the step 32 → 256 window contains the step 64 → 128 ranking-lock
  window (Result 8), so the graded ordering and the ranking lock-in may be a single event. Resolving
  that needs the large bank at steps 64 and 128 (~10 min of GPU) plus a way to compare $\pi$ across
  banks; I stated it as a limitation in Result 14 instead of leaving it implicit.
- The within-top-quintile correlation (200 pairs, $J$ from 0.767 to 0.950) is weak at every
  checkpoint ($p^{\mathrm{fw}} \ge 0.05$ everywhere). That is range restriction, not evidence, so I
  computed it, kept it in `results/bulk_onset.json`, and did **not** put it in the deliverables
  (CLAUDE.md rule 12: a figure/metric no claim needs gets cut).

**Assumptions logged (loop mode, no human to ask).**
- Chose step 256, 1000 and 8000 as the three new large-bank checkpoints: 256 and 1000 bracket the
  region where the 60-pair bank shows the ranking locking in and the shape starting, and 8000 gives a
  post-sharpening anchor. Rejected 64/128 (would have resolved the ordering-vs-ranking ambiguity but
  left no measurement in the sharpening region, where the alternative hypothesis lived).
- The onset rule is the primary one verbatim (after the last non-significant checkpoint, by the first
  of two consecutive significant ones), with significance from the *simultaneous* envelope over the
  eight large-bank checkpoints under the endpoint-label null — the same null used everywhere else on
  this bank, so pairs sharing an endpoint token stay dependent.
- Quintiles are cut at the large bank's own 20/40/60/80th divergence percentiles, identical to
  `quintile_large.py`, so Result 13 and Result 14 partition the same pairs the same way.
- Left the five previously reported large-bank CIs and permutation $p$-values as they stand
  (`large_late.py` / `permtest.py`) and added the three new checkpoints from `bulk_onset.py`. Same
  estimator and same null, different RNG draw; re-running the old columns would have changed
  published third decimals for no gain.
- Did NOT edit `PLAN.md` (operator-owned, declares itself read-only), same as iterations 2–7.

**Next step.** S1–S6 are complete; Results 8–14 go beyond the plan. The one open question I can name
is whether the graded ordering (step 32 → 256) and the ranking lock-in (step 64 → 128) are one event
— which would need the 1,000-pair bank at steps 64 and 128 (~10 min GPU) and a cross-bank comparison
of $\pi$. Everything else I can think of (a second model size, a second training run, checkpoints
between 8 and 32) is out of scope under this PLAN or impossible from released artefacts. Did not
write `STOP`: the deliverables are complete, but a premature STOP would make the direction silently
ignore any feedback dropped later.

On track? yes — S1–S6 complete plus Results 8–14 beyond the plan (~100% of plan), no blocker; the
released checkpoint spacing, not budget, bounds any further tightening.
