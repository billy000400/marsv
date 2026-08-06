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
