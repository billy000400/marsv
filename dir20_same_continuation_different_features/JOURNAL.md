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
