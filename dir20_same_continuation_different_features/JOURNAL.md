# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---
## 2026-08-10 — iteration 1 (S1 + S2 + S3, direction essentially complete)

**Feedback check:** listed the direction root for `human_feedback*.md` / `*REVIEW*` without the
`.addressed.md` suffix — none present. Proceeded with the plan.

**What I did.** Built the whole pipeline from an empty direction and ran it end to end.
- `experiments/common.py` — budget-respecting setup (`set_num_threads(2)`,
  `set_per_process_memory_fraction(0.225)`), model loading, the 5 prompt pairs, tokenization
  validator.
- `experiments/run_interp.py` — S1 + S2. Validates tokenization, computes endpoint JSD from
  inference-time full-vocab softmax, SLERPs the block-0 last-token `resid_post` (linear on the L2
  norm), patches it forward in chunks of 32 alphas, records `resid_post` at every downstream block
  and the final logits, computes d(alpha) and w10-90 everywhere.
- `experiments/analyze.py` — S3. Three figures + `w_TV`, `PF`, and Spearman correlations.

**Assumptions logged (loop mode, no human to ask).**
1. `gpt2-medium` was not in the local HF cache and the default download path 404s on the xet
   endpoint; `HF_HUB_DISABLE_XET=1` fixes it, so I downloaded the planned model rather than
   substituting the cached `gpt2-large`. Rejected alternative: swap to gpt2-large and note the
   deviation — worse, since PLAN.md fixes the model.
2. Added the control pair as a fifth *row* of every figure and table instead of treating it as a
   silent sanity check. It turned out to be the load-bearing baseline, so this was the right call.
3. Added `w_TV` and `PF` beyond PLAN.md's single metric because 4/10 curves are non-monotonic and
   PLAN.md explicitly warns against forcing a verdict from w10-90 alone. w10-90 stays primary.
   Rejected alternative: report only w10-90 and flag the non-monotonic cells — that would have left
   gpt2 `Mary`/`her` (visually a strong plateau, w10-90 = 0.586 because of a dip below 0.1)
   misclassified.

**What I learned.**
- The harness is exact: patching at alpha in {0,1} reproduces the clean runs to 1e-4, so the curves
  are the model's behavior.
- The plan's hypothesis is not supported. Plateaus appear in 9/10 cells, but the *lowest*-JSD cells
  give the *widest* transitions (pythia `four`/`4`: JSD 0.056, w10-90 0.758 — the linear baseline),
  and the dissimilar control (JSD 0.66) plateaus at w10-90 0.516 / 0.425. Pooled
  rho(JSD, w10-90) = -0.37, p = 0.29, n = 10; the three sharpness statistics disagree on sign.
- The mechanism is depth, not prompt content: at block 1 every cell sits at w10-90 ~ 0.80 (exactly
  linear), and the width falls monotonically over the following 23 blocks. This is the strongest
  single piece of evidence in the direction and it reframes the report.
- So the story changed from "which pairs plateau?" to "a plateau here is not evidence of a shared
  continuation — you need a control interpolation" (CLAUDE.md rule 9b re-framing, logged above).

**Next step.** The plan's success criterion is met and a null result is explicitly complete, so the
remaining risk is statistical, not implementational: n = 10 cells cannot rule out a moderate
association. The highest-value next iteration is to mine ~100-200 prompt pairs programmatically
(same prefix, single differing final token, sampled to span a wide JSD range) and re-run the same
pipeline on them, turning Figure 2's 5-point scatter into a properly powered regression. Secondary:
repeat the sweep with the patch at a middle block to check the depth explanation directly.

On track? yes — S1/S2/S3 all complete, ~100% of PLAN.md done, no blocker; remaining work is
optional statistical strengthening of an already-complete null result.

---
## 2026-08-10 — iteration 2 (S4: mined pair bank, powered association test)

**Feedback check:** listed the direction root for `human_feedback*.md` / `*REVIEW*` without the
`.addressed.md` suffix — none present. Proceeded with the plan's stated next step.

**What I did.** Executed iteration 1's stated next step: replaced the 5-point scatter with a properly
powered regression over corpus-mined pairs.
- `experiments/mine_pairs.py` — mines 200 pairs/model from WikiText-103 validation (cached locally at
  `$HF_HOME/datasets`, run with `HF_DATASETS_OFFLINE=1`): 40 paragraphs >= 400 chars, truncated to a
  prefix of 10–40 tokens; final token A = the model's own top-1 next token, final token B = the
  rank-r token with r log-uniform in [1,5000], 5 partners per prefix. Builds `input_ids` as
  `prefix_ids + [token_id]`, which makes the "identical prefix, one differing single final token"
  validity condition exact rather than something to check. Reuses `run_interp.py`'s sweep unchanged.
- `experiments/analyze_bank.py` — prevalence stats, Spearman rho with 95% **cluster bootstrap** over
  prefixes (pairs sharing a prefix are not independent), OLS slopes, quintile means, a JSD-ceiling
  robustness split, and a partial Spearman controlling for the block-0 angle Omega. Two new figures.

**Assumptions logged (loop mode, no human to ask).**
1. Mined pairs are built by swapping the final token for a lower-ranked alternative *at the same
   position*, rather than by finding naturally occurring minimal pairs in the corpus. Rejected
   alternative: mine real minimal pairs by string matching — far fewer hits, no control over the JSD
   range, and no guarantee of single-token differences. The cost is logged as a limitation in
   REPORT.md: both continuations are ones the model itself considered plausible.
2. Used the same 40 prefixes and the same rank draws for both models (tokens differ, since each
   model's top-1 differs). Keeps the two columns comparable; rejected re-drawing per model.
3. Reported both the full-bank and the JSD<0.65 correlations rather than only the cleaner subset.
   Pre-registering the ceiling split was not possible, so showing both is the honest option.

**What I learned.**
- The n=10 null was underpowered, not a true null. At n=200 the association is real and its sign is
  the **opposite** of the direction's hypothesis: rho(JSD, w_TV) = -0.55 in gpt2-medium
  (p=6.2e-17), i.e. pairs whose continuations diverge more give *sharper* plateaus.
- pythia-410m's full-bank rho is only -0.11 because 37% of its mined pairs sit at the ln 2 JSD
  ceiling, where the independent variable stops discriminating. Below the ceiling both models agree:
  -0.61 (n=142) and -0.45 (n=127), p<1e-7. This ceiling effect is the single most useful thing I
  learned this iteration — without the split I would have written up a spurious model disagreement.
- The effect is not endpoint geometry: partial rho controlling for the block-0 angle Omega is -0.55
  (gpt2) and -0.16 (pythia), and Omega correlates with sharpness at only 0.16.
- Plateaus are the default response: 82% (gpt2) / 48% (pythia) of arbitrary mined pairs are sharp, and
  all five hand-picked pairs sit inside the bulk of their model's distribution.
- Only 7.5% of gpt2-medium's mined curves are monotonic (98% for pythia), which retroactively
  vindicates adding w_TV in iteration 1.
- Second re-framing under rule 9b (old -> new story recorded in CHANGELOG.md): from "no detectable
  association" to "the association is real and inverted".

**Next step.** The mechanism paragraph in REPORT.md now makes a testable claim I have not tested: that
the sharpening is a winner-take-all competition between two well-separated output modes. The direct
test is to re-run the mined-bank sweep with the patch at a middle block (e.g. block 12) and at a late
block, and check whether (a) the plateau survives fewer downstream layers and (b) the negative
JSD-sharpness correlation weakens with fewer layers left to compress. `mine_pairs.py` needs only a
patch-layer argument for this. Secondary: a third model family to test whether the gpt2 vs pythia gap
in prevalence (82% vs 48%) is about tokenizer or architecture.

On track? yes — PLAN.md's S1–S3 remain complete and S4 (unplanned, statistical strengthening) is done;
no blocker; the direction now has a significant positive finding rather than a null.

---
## 2026-08-10 — iteration 3 (S5: patch-depth intervention)

**Feedback check:** listed the direction root for `human_feedback*.md` / `*REVIEW*` without the
`.addressed.md` suffix — none present. Proceeded with the plan's stated next step.

**What I did.** Ran S5, the open question from iteration 2: REPORT.md claimed the sharpening is
produced by the layers below the patch, on read-out evidence only. Turned that into an intervention.
- `run_interp.py`: `sweep()` gained a `layer=0` argument (patch site and the range of recorded
  downstream blocks both keyed off it). Default preserves every earlier result exactly.
- `mine_pairs.py`: patch layer from `argv[1]`, endpoints read at that block, outputs suffixed `_L<n>`.
- `analyze_depth.py` (new): prevalence + ceiling-corrected Spearman with cluster bootstrap at each
  patch site, and `plots/depth_effect.png`.
- Ran the full 200-pair bank per model at block 12 and block 20 (~4 min/sweep set on the shared GPU;
  800 extra sweeps). JSD ranges came out identical to the block-0 bank (0.002–0.693 / 0.007–0.693),
  confirming the same pairs were used; endpoint identity error stayed <= 3.5e-4 everywhere.

**Assumptions logged (loop mode, no human to ask).**
1. Chose blocks 12 and 20 of 24 (11 and 3 blocks below) rather than a full 24-point sweep: three
   points span the range and cost 800 sweeps instead of ~9600. Rejected alternative: patch every
   block for a small subsample of pairs — smoother curve, but each point would then have too few
   pairs to estimate rho with a cluster bootstrap, which is the quantity S5 exists to test.
2. Figure 5's right panel plots the JSD<0.65 (ceiling-corrected) rho rather than the full-bank rho,
   because the full-bank number is known to be diluted for pythia-410m and the report already treats
   the unsaturated subset as the cleaner estimate. Both are in `results/depth_analysis.json`; the
   full-bank values tell the same story for gpt2 (-0.55 / -0.50 / -0.56).

**What I learned.**
- The depth claim is confirmed, and more strongly than expected: at block 20, pythia-410m has **zero**
  sharp pairs out of 200 and a median response (w_TV 0.509, w10-90 0.808) within 2% of the linear
  baseline. gpt2-medium falls 82% -> 50.5% -> 10% sharp. The plateau is not a property of the prompt
  pair; it is a property of how much network is left to process the edit.
- The prediction I expected to confirm alongside it was **wrong**: the negative JSD-sharpness
  correlation does *not* shrink with depth. In gpt2-medium it is flat (-0.61 / -0.53 / -0.53) even
  where 90% of pairs no longer plateau. In pythia-410m it only dies at block 20, where there is no
  transition shape left to modulate. So depth and divergence are separable factors — depth sets how
  much compression happens, divergence sets which pairs get compressed most, and the latter is
  resolved close to the output. That refines the report's competition account rather than confirming
  it wholesale, and it is the honest version of the mechanism paragraph.
- Non-monotonicity is a deep-stack artifact: monotonic gpt2-medium curves go 7.5% -> 33% -> 72% as
  depth is removed. Retroactive justification for `w_TV` a third time.

**Next step.** The three headline claims (plateaus ubiquitous, association inverted, depth causal) are
all now measured, so the direction's substance is complete. The largest remaining unknown is whether
the 82% vs 48% block-0 prevalence gap between the two models is tokenizer or architecture; a third
family (e.g. OPT-350m, a different tokenizer with similar depth) at block 0 and block 20 would settle
it in roughly one iteration's compute. Failing that, the finalization work is: re-read both
deliverables end to end for newcomer readability and confirm every figure is cited by number.

On track? yes — PLAN.md S1–S5 all complete, ~100% of the plan done, no blocker; remaining work is an
optional third-model generalization check.
