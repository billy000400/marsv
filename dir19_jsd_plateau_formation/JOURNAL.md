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
