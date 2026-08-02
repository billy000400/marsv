# JOURNAL — Does training-corpus continuation JSD predict plateau strength?

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-08-02 — iteration 1: S1 through S5 complete in one pass

**Feedback check.** Listed the direction root for `human_feedback*.md` / `*REVIEW*` without the
`.addressed.md` suffix: **none present**. Proceeded with the plan.

**What I did.**

1. **Preflight (the plan's stated make-or-break).** Downloaded only the first 4 KB of
   `document.idx` and parsed the official Megatron header: magic `MMIDIDX`, version 1, dtype code 8 =
   `uint16`, `len = 146,432,000`, every listed size 2049. Confirmed the layout arithmetic is exact —
   predicted idx size `34 + 12L + 8D = 1,757,184,042` equals the actual file size, and the 21 `.bin`
   shards total `600,078,336,000 = 146,432,000 x 2049 x 2` bytes. **Consequence: sequence `i` starts
   at byte `4098i`, so a row-aligned sample is a plain HTTP byte range** — no unsharding, no 602 GB
   download, and no need for the official reader at run time.
2. **Corpus sample.** Two distant, row-aligned 500,000-row splits: A at global row 1,000,000
   (shard 0) and B at 73,300,000 (shard 10), each wholly inside one shard. Verified both by decoding
   rows at head/middle/tail (real Pile text; all IDs < 50,277).
3. **S1 JSD table.** Counted endpoint->successor bigrams *inside* rows only, in two halves per split
   so the same-token split-half noise floor came free (no second data pass). Both gates passed by a
   wide margin: `Spearman(JSD_A,JSD_B) = 0.9998`, noise ratio `0.072`.
4. **S2 frozen bank.** 75 endpoint-disjoint pairs, 15 per `JSD_A` quintile, balanced on frequency and
   surprisal. Frozen before viewing any curve.
5. **S3 assay + calibration.** Norm-rescaled SLERP on the post-block-0 final-position residual,
   patched, read out at the logits. Calibration gate passed (100% valid curves, `IQR(w) = 0.115`).
6. **S4/S5.** Full bank on 1.4B step143000, 1.4B step0, 410M step143000; block scan; validity checks;
   all figures and statistics.

**What I learned.**

- **The result is clean and matches the prespecified positive branch.** `rho(JSD_B, w) = -0.419`
  [-0.585, -0.222] trained versus `-0.155` [-0.368, +0.068] at step 0, replicating at 410M
  (`-0.320`). Bin medians are monotone in both trained models.
- **The step-0 control is stronger than I expected.** It is not merely "no correlation": the
  untrained network has median `w = 0.831` with `IQR = 0.004`, i.e. it sits at the linear-response
  value 0.8 with essentially *no* plateau structure and *no* variance to correlate with. That makes
  the trained-model effect much harder to attribute to tokenisation or architecture geometry.
- **The predictor is validated, so a null would have been interpretable.** `rho(JSD_B, output JSD)
  = +0.729` at step143000 — a global, context-free corpus statistic really does predict how the model
  splits these words inside one specific sentence.
- **But geometry adjustment attenuates it a lot** (`-0.419 -> -0.267`). Since block-0 endpoint
  geometry plausibly lies *on the causal path* from training targets to plateau shape, the plan's own
  guidance applies: report the total association, do not claim explanation beyond learned geometry.
- **Engineering lesson that cost the most time:** the shared MooseFS volume intermittently returns
  `EDQUOT` under 4 concurrent agents. It silently killed the downloader repeatedly, tore a
  non-atomic JSON ledger, and at one point **truncated a source file I had written from a shell
  heredoc to 0 bytes**. Fixes: bulk data moved to local `/tmp`, ledger writes made atomic via
  `os.replace`, a self-restarting supervisor around the resumable downloader, and file edits done
  through the editor tool rather than shell heredocs.

**Assumptions logged (loop mode — could not ask).**

- *Top-K endpoint filter.* The plan prespecified top-256, which yields 134 tokens = at most 67
  endpoint-disjoint pairs, short of the 75-pair target. Options were (a) shrink the bank to 67,
  (b) use a dependent all-pairs design, (c) relax to top-512. **Chose (c)** — the plan explicitly
  rejects (b), and top-512 is still the top 2.8% of 18,714 eligible word tokens, so endpoints stay
  firmly in-distribution while the independent design and the 75-pair target both survive. Rejected
  (a) because it costs power for no design benefit. Recorded in the manifest, REPORT.md Methods, and
  CHANGELOG.md; the strict top-256 subset (n = 12) is reported as an underpowered sensitivity check.
- *Counting pool.* Counted a top-1024 superset (527 tokens) in the single data pass so that the
  top-256 vs top-512 decision needed no re-count. This is a compute choice, not a selection choice —
  the bank rules were fixed before the counts were inspected.
- *`d(t)` is computed in logit space* (endpoint logit vectors restricted to the 50,060 valid target
  IDs), which is how the plan's "record final-position logits ... and compute `d(t)`" reads.

**Bugs caught before they polluted results.** (i) `build_pairs.py` indexed the per-endpoint
distribution matrix by eligible-set position while the matrix rows were pool positions — would have
scrambled every pair's JSD; fixed and rerun before the bank was frozen. (ii) The 64 MB-chunk ledger
was migrated to 16 MB units rather than silently re-downloading split A.

**Next step.** Optional formation subset from the plan's fixed setup: run the same frozen bank at
`step1000`, `step8000`, `step32000`, `step64000` to test whether the negative relationship
*strengthens* during training. Everything in the plan's definition of done is already satisfied, so
this is additive; if the intermediate checkpoints cannot be downloaded, the deliverables stand as-is
and I write `STOP`.

On track? yes — S1-S5 complete (definition of done met, ~90%), no blocker; remaining work is the
optional formation-during-training subset.

---

## 2026-08-02 — iteration 1 (continued): formation subset, and the plan's expectation is refuted

**What I did.** Ran the same frozen 75-pair bank on `pythia-1.4b-deduped` at `step1000`, `step8000`,
`step32000`, `step64000` (`experiments/formation.py` fetches each checkpoint to local `/tmp`, assays
it, then deletes it, so peak disk stays at one checkpoint — the shared volume cannot hold four).
Added `plots/formation.png` as Figure 6 in both deliverables and renumbered the block scan to
Figure 7. Ran the full 75-pair bank rather than the plan's "30 frozen pairs" — same wall-clock cost
(~1 min per checkpoint), strictly more power, and it keeps every checkpoint on one identical bank.

**What I learned — this is the interesting part.** The plan's stated expectation was that the
negative relationship would *strengthen* during training. **It does the opposite.**

| step | 0 | 1000 | 8000 | 32000 | 64000 | 143000 |
|---|---|---|---|---|---|---|
| rho(JSD_B, w) | -0.155 | **-0.660** | -0.605 | -0.524 | -0.539 | -0.419 |
| rho(JSD_B, out JSD) | -0.144 | +0.779 | +0.693 | +0.726 | +0.714 | +0.729 |
| median w | 0.831 | 0.758 | 0.624 | 0.582 | 0.541 | 0.562 |

Two things move differently, and separating them is what makes this worth reporting:

- **Sharpness keeps increasing** — median `w` falls monotonically from 0.831 (no plateau, sitting at
  the linear-response value 0.8) to ~0.55, with the IQR widening from 0.004 to ~0.11. Plateau
  structure forms early and keeps deepening.
- **The corpus predictor peaks at the earliest checkpoint I ran and then fades** — `-0.660` at step
  1000 decaying to `-0.419` at step 143000. Meanwhile its correlation with the model's own *output*
  divergence is flat from step 1000 on (+0.78 -> +0.73), so the predictor is not degrading in
  general; it specifically explains a shrinking share of *which* pairs are sharp.

My reading (stated as a suggestion, not a finding, in REPORT.md): early training is dominated by
corpus-level continuation statistics, and later training adds context-sensitive structure that a
context-free `P(y|a)` cannot capture. The obvious follow-up is a context-conditioned divergence
estimate, which this direction's plan explicitly put out of scope.

**Caveat I made sure to state in the deliverable.** One trajectory, one bank, one model, and the
earliest checkpoint sampled is step 1000 — the relationship could peak earlier still, and I have no
resolution below 1000 steps to say where it forms. So this is an observation, not a training-dynamics
result.

**Deliverable state.** RESULTS.md and REPORT.md re-curated to current-best with 7 figures embedded
in both, sequentially numbered in reading order, each with a visible caption; `check_render.py`
passes (7 display equations, 0 `<pre lang="math">`, all embeds captioned). CHANGELOG.md has a second
dated entry appended; no earlier entry was rewritten and no number was superseded.

**Next step.** The plan's definition of done is fully met and the optional item is now also done. No
unaddressed `human_feedback*.md` / `*REVIEW*` files exist, so `STOP` is written. If a future
iteration finds new feedback next to the stale `STOP`, delete `STOP`, address the feedback, and only
re-write it when clean.

On track? yes — S1-S6 complete (100%), no blocker; STOP written with zero unaddressed feedback files.
