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

---
## 2026-08-10 — iteration 4 (S6: third model family, OPT-350m)

**Feedback check:** listed the direction root for `human_feedback*.md` / `*REVIEW*` without the
`.addressed.md` suffix — none present. Proceeded with the plan's stated next step.

**What I did.** Ran iteration 3's stated next step: added a third model family to settle the one
question the report named and did not answer (is the 82% vs 48% block-0 prevalence gap a tokenizer
effect?).
- Downloaded `facebook/opt-350m` (`HF_HUB_DISABLE_XET=1`, as for gpt2-medium in iteration 1).
- `common.py`: model entry + `m.model.decoder.layers` branch in `blocks()`. `run_interp.py` and
  `mine_pairs.py`: model list from argv; `run_interp.py` merges into `summary.json` rather than
  overwriting, so the two existing models' numbers were reused untouched instead of re-run.
- Ran the 5 hand-picked pairs plus the 200-pair mined bank at blocks 0, 12 and 20 for opt-350m
  (605 sweeps, ~9 min on the shared GPU).
- `analyze_bank.py`: new `jsd_matched()` (median w_TV inside fixed JSD bins) + `plots/jsd_matched.png`,
  and the ceiling-subset w10-90 correlation that Table 4 was quoting for two models only.

**Assumptions logged (loop mode, no human to ask).**
1. Chose OPT-350m over the alternatives because it is matched to both existing models on the two
   structural quantities Experiment 4 shows to matter (24 blocks, d_model 1024) *and* shares
   gpt2-medium's exact tokenizer — so it can falsify the tokenizer explanation. Rejected: gpt2-large
   (cached, but 36 blocks — confounds depth with family) and pythia-1.4b-deduped (cached, but same
   tokenizer *and* architecture as pythia-410m, so it tests width, not the open question).
2. Ran all three patch sites for opt-350m rather than only blocks 0 and 20 as PLAN.md suggested: the
   middle point costs ~3 min and keeps Figure 6 comparable across the three series.
3. Compared models inside fixed JSD bins rather than reweighting one bank to another's JSD
   distribution. Binning is transparent and shows the within-model trend at the same time; rejected
   propensity-style reweighting as unnecessary machinery for a 4-bin comparison.

**What I learned.**
- The headline finding replicates in a third family: below the ln 2 ceiling, rho(JSD, w_TV) = -0.57
  (n=129, p=1.3e-12) in opt-350m, between gpt2-medium's -0.61 and pythia-410m's -0.45. The inverted
  association is now 3/3 models, on both width statistics and PF.
- The depth result also replicates, and opt-350m patterns with gpt2-medium rather than pythia-410m:
  61% -> 36.5% -> 1.0% sharp as depth is removed, with the divergence correlation flat (-0.57 / -0.54 /
  -0.55) even at 1% prevalence. pythia's rho collapse at block 20 really is a floor effect of its
  response having gone fully linear, not a general depth law.
- **The tokenizer explanation is dead.** opt-350m's vocabulary is exactly gpt2-medium's 50257 token
  strings plus 8 specials and it segments the prompts identically, yet it plateaus 21 points less
  often, and at matched divergence gpt2-medium is the sharpest model in all four JSD bins (by 2-4x on
  the median). The gap is a model property; architecture, corpus and pretraining length stay
  confounded, which is the honest statement now in the limitations.
- Nice presentational bonus: in opt-350m the dissimilar *control* is the sharpest of its five
  hand-picked cells (w_TV 0.068, a near-step), which makes the report's central warning legible in a
  single panel of Figure 1.

**Next step.** PLAN.md S1-S6 are complete and the last named open question is answered as far as three
models can answer it. The remaining work is finalization: nothing in the deliverables is stale, both
pass `check_render.py`, all six figures are embedded with visible captions and cited by number. If a
further iteration runs, the highest-value additions are (a) a depth-mismatched model (e.g. a 12-block
or 36-block checkpoint) to test whether the depth curve of Figure 6 is about absolute block count or
fraction of the stack, and (b) pairs that differ at an earlier position rather than the final token.
Neither is required by PLAN.md.

On track? yes — PLAN.md S1-S5 complete plus the unplanned S6 third-model generalization; ~100% of the
plan done, no blocker; the direction's three headline claims now each rest on three model families.

---
## 2026-08-10 — iteration 5 (S7: depth-mismatched models, relative vs absolute depth)

**Feedback check:** listed the direction root for `human_feedback*.md` / `*REVIEW*` without the
`.addressed.md` suffix — none present. Proceeded with the plan's stated next step (option (a) from
iteration 4: a depth-mismatched model to test whether Figure 6's curve is about absolute block count
or fraction of the stack).

**What I did.** Ran the experiment that decides the units of "depth" in Experiment 4.
- `common.py`: added `gpt2-small` (12 blocks) and `gpt2-large` (36 blocks) plus an `N_BLOCKS` map.
  Both were already in the local HF cache. No change to `mine_pairs.py` / `run_interp.py` — the
  patch-layer argument added in iteration 3 already covered everything this needed.
- `analyze_scaling.py` (new): per-site stats for the three GPT-2 depths, the matched-level spread
  statistic under each reading, and `plots/depth_scaling.png` (two panels, same 12 runs, one x-axis
  each).
- Ran 200-pair banks at blocks 0/6/8/10 for gpt2-small (~1 min per site) and blocks 0/12/18/24/31 for
  gpt2-large (~11 min per site) — 1800 sweeps, ~55 min wall-clock, well inside the GPU budget
  (peak 4.3 GB of the 7.2 GB share).

**Assumptions logged (loop mode, no human to ask).**
1. Tested depth *within* the GPT-2 family rather than by adding another family. Holding tokenizer,
   architecture and corpus fixed is what makes the comparison interpretable; the cost is that
   residual width rises with depth (768/1024/1280), which I state as a limitation instead of trying
   to remove. Rejected alternative: pythia-1.4b-deduped (cached) — different width *and* different
   family, so a null would have been unattributable.
2. Chose patch sites to hit both matchings rather than sweeping every block: gpt2-large at block 12
   gives exactly 23 blocks below (matching gpt2-medium at block 0) and block 24 gives 11, while
   blocks 0/18/31 match the fractions. Five sites instead of 36 costs 1000 sweeps instead of 7200 and
   still lands the decisive comparison.
3. Quantified "which reading organises the data" with the across-model range of median w_TV at
   matched levels, averaged over levels. Rejected fitting a two-parameter model of sharpness in
   (blocks, fraction) — with three models and twelve runs that would over-claim; the range is
   transparent and the figure shows the raw curves anyway.

**What I learned.**
- The answer is relative depth, and it is not close. gpt2-large at block 12 and gpt2-medium at block 0
  have the *same 23 blocks* below the patch and differ by a factor of 3.2 in median w_TV
  (0.255 vs 0.080), 47% vs 82% sharp. Matching on the fraction halves the mean across-model spread
  (0.212 -> 0.104). The prediction I would have made from Figure 6 alone — absolute block count —
  was wrong.
- At 11 blocks below the patch the ordering actually *inverts* the absolute reading: the 12-block
  model is the sharpest (0.153) and the 36-block model the flattest (0.444). More blocks below the
  patch, less plateau, because those blocks are a smaller share of the network.
- This makes the depth result portable, which is the practical payoff: an experimenter can estimate
  how much plateau a patch site manufactures in an untested model from f alone.
- The inverted JSD association replicates in both new models (-0.44 and -0.64 at block 0), so it is
  now 5/5 models over three families, 124M-774M params, 12-36 blocks. gpt2-large's -0.64 is the
  strongest value in the study.
- Both new models *narrow* an earlier claim rather than confirming it. REPORT.md said the divergence
  effect is "already fully expressed by the last three blocks"; in gpt2-small (1 block below) rho goes
  to +0.04 and in gpt2-large (4 below) to -0.19, both at ~0% sharp. That matches pythia-410m and
  supports the floor-effect reading — the correlation needs some transition shape left to modulate —
  so the claim now applies only to the models that still plateau at that depth. Edited the Results
  and Summary accordingly.
- Process note: my draft numbered the two new tables 5 and 6, colliding with the existing Table 5, and
  three references in the new section pointed at Table 4 when they meant the depth table. Caught by
  grepping `Table [0-9]` across the file, not by `check_render.py`, which does not check numbering.

**Next step.** PLAN.md S1-S7 are complete and both open questions from iteration 4 are now one for
one: (a) is answered here. The single remaining item from that list is (b) pairs that differ at an
*earlier* position rather than the final token, which is the last untested generalization of the
setup and would need a modest change to `mine_pairs.py` (patch the differing position instead of the
last one, and record whether the difference propagates). Failing that, the deliverables are current,
pass `check_render.py`, and embed all seven figures with visible captions cited by number.

On track? yes — PLAN.md S1-S6 complete plus the unplanned S7 depth-scaling test; ~100% of the plan
done, no blocker; the depth finding is now stated in units that transfer to models we did not run.

---
## 2026-08-10 — iteration 5b (operator feedback: reproduction, reframing, and the real hypothesis)

**Feedback check:** `human_feedback.txt` appeared in the direction root *during* this iteration (it was
not present at the start, when I checked and found nothing). Per CLAUDE.md Part C it became the
iteration's work. Read in full, addressed all five points, renamed to
`human_feedback.txt.addressed.md`.

**What I did.** Ran the experiments the feedback demanded, then rewrote both deliverables around what
they showed.
- `common.py`: added `The house was big / large` as a sixth pair and relabelled the two `The house was`
  pairs as Matthew's plateau case and smooth case. Ran `run_interp.py` over all five models (30
  model-pair cells).
- `analyze.py` / `analyze_bank.py`: extended to five models and six pairs (6th series in gray, distinct
  linestyle+marker, no red/green).
- `feature_plateau.py` (new): the first direct test of the advisor's hypothesis — output JSD held below
  0.1, IRD (mean over blocks of 1-cos between the two clean residual streams) as the independent
  variable, IPW (longest alpha span resting at a level in [0.15,0.85] within a 0.10 band) as the
  dependent variable, with the linear response giving IPW = 0.10 by construction.
- Rewrote RESULTS.md end to end and REPORT.md end to end, including a new title.

**What I learned — and where I was wrong.**
- The feedback's central point is correct and my previous framing was wrong in a way that mattered.
  `big`/`in` is Matthew's *positive* example; I had been treating it as a dissimilar-continuation
  negative control, which inverted the meaning of every sentence built on it. With `big`/`large` added,
  his actual contrast reproduces cleanly in his actual model: gpt2-large gives w10-90 = 0.044 vs 0.592,
  a 13-fold gap, and big/large is the widest or second-widest transition in all five models.
- GPT-2 Medium really is not a reproduction: the same big/in pair scores 0.516 there, failing the
  predefined criterion. My earlier "the control plateaus as hard as the test pairs" observation was an
  artifact of running the wrong model and mislabelling the pair. The S7 relative-depth result explains
  why (f = 1 buys more compression in a longer stack), which is a satisfying consistency check between
  this iteration's two halves.
- The prevalence overstatement was real. I had been quoting "13 of 15 cells" using w_TV < 0.5, which
  only means "better than linear", not the plan's predefined w10-90 < 0.5. Under the predefined
  criterion it is 11 of 30. The mined-bank claim survives the stricter criterion (83.5% in gpt2-large),
  so the base-rate argument stands, but the hand-picked claim did not.
- "Depth produces the plateau" was too strong, and the counterexample was already in my own data once
  big/large was run: 35 blocks below the patch in gpt2-large and still smooth. Necessary, not
  sufficient.
- The hypothesis test came out null in both models (rho = +0.17, p=0.31, n=38; rho = -0.00, p=0.99,
  n=32), and notably **zero** low-JSD gpt2-large pairs show any intermediate plateau — its curves step
  once rather than pausing. I am deliberately not calling this a refutation: rho_min at n=38 is 0.32,
  and IRD is representation geometry, not a feature measurement.

**Assumptions logged (loop mode, no human to ask).**
1. Operationalised "different circuits/features" as IRD, mean cosine distance between the two clean
   residual streams across the stack. Rejected alternatives: SAE feature sets (no trained SAE available
   offline for these checkpoints within budget) and path patching (needs a per-pair circuit search,
   far beyond the remaining time). The proxy is named as a limitation and the SAE/path-patching version
   is written into the report as the next step.
2. Operationalised "different plateaus" as an intermediate resting level (IPW), not as transition
   width. This is the reading that makes the hypothesis distinguishable from what Experiments 3-5
   already measure. Rejected: counting inflection points, which non-monotonic curves make meaningless.
3. Kept the JSD-vs-width correlation in the report rather than deleting it, but demoted it with an
   explicit scope note. Rejected deleting it: it is a real, five-model regularity that a reader of this
   method will hit, and hiding it would be its own distortion. Rejected keeping it as the headline: the
   feedback is right that it tests a different claim.
4. Renamed the feedback file to `human_feedback.txt.addressed.md` rather than
   `human_feedback.addressed.md`, so the original name is preserved intact while the suffix rule is
   satisfied.

**Next step.** The hypothesis now has a stated, testable form and a first null. The highest-value next
iteration is the sharper version of Experiment 6: get a feature-level measurement of circuit difference
(a trained SAE on GPT-2 Large residuals, or path patching over attention heads for a subset of low-JSD
pairs) and re-run the IRD leg with feature-set disjointness as the independent variable. Secondary:
raise power by mining specifically for low-JSD pairs rather than filtering a general bank, which would
take n from 38 to a few hundred at the same cost.

On track? yes — all five feedback points addressed with new experiments rather than text edits alone;
both deliverables rewritten and passing `check_render.py`; no unaddressed feedback remains and no STOP
written (the hypothesis test is open work).

---

## 2026-08-10 (iteration: finish S9 curation, run S10 head ablation)

**Feedback check.** Listed the direction root: the only matching file is
`human_feedback.txt.addressed.md`, already suffixed. Nothing unaddressed, so no STOP is blocked by
feedback — and none was written, because S9/S10 opened further work.

**What I found on entry.** The previous iteration had been cut off mid-curation. `circuit_features.py`
and its plots were written at 20:46 and REPORT.md's Results section had been updated at 20:49, but
RESULTS.md, CHANGELOG.md, JOURNAL.md and PLAN.md were all still at ~19:50, and REPORT.md's own
Conclusion still reported the superseded n=38 IRD test that its Results section had just replaced.
Two sweep totals were also wrong (Methods said 4765, Results said 3645; the true count is 4750, which
I recomputed from the stored result files rather than trusting either). First half of the iteration
was making the deliverables internally consistent again.

**What I did.**
- Recomputed the endpoint identity bound over every stored sweep (bank JSONs + low-JSD rows): worst
  case 3.53e-4, so the report's "<= 4e-4" is now stated as "<= 3.6e-4" in RESULTS and "<= 3.5e-4" for
  the ablation runs.
- Rewrote RESULTS.md end to end at current-best, dropping the IRD-vs-IPW experiment in favour of the
  feature-level version, and rewrote REPORT.md's Conclusion and Limitations.
- Wrote `experiments/ablate_heads.py` and ran the S10 intervention: 3 doses x 2 models x ~380 low-JSD
  pairs, ~14 min of GPU wall-clock.
- Added the numbered figure citation to the motivating sentence of every figure in both files (five
  figures in each were previously named only inside their own caption).

**What I learned.**
- The intervention worked, and much more strongly than the pilot suggested. My n=40 pilots showed
  nothing and no dose trend; at full n the gpt2-large effect is enormous (median w_TV 0.198 -> 0.358 at
  3% of heads, -> 0.484 at 10%, p ~ 1e-43). I nearly took the pilot as the answer, and the lesson is
  that the pilot was under-powered for a paired comparison with a heavy-tailed DV, not that the effect
  was absent. Running the pre-specified dose sweep at full n rather than picking a dose from the pilot
  was the right call for a second reason: the dose-response is itself the strongest part of the
  evidence.
- The size of a correlation was a poor guide to the size of an intervention. gpt2-medium has the
  *stronger* HCD correlation (-0.36 vs -0.29) and the *weaker* intervention effect by a factor of 15.
  Worth saying out loud in the report, because the tempting move — inferring causal importance from
  correlation strength — would have got the ranking backwards.
- The matched control is what carries the claim. My first control construction drew from the low-delta
  half of the heads and removed 35% less write magnitude than the treatment (340 vs 221), which would
  have made any effect uninterpretable. Matching each treated head to its 24 nearest neighbours by
  |c^A| + |c^B| and taking the lowest-delta one brought the ratio to 1.01-1.12.

**Assumptions logged (loop mode, no human to ask).**
1. Operationalised "the circuit that differs" as the top-k heads by delta_h, HCD's own per-head
   numerator term. Rejected: path patching (needs a per-pair circuit search, out of budget) and
   ablating SAE features (public SAEs exist for gpt2-small only, which Experiment 7 does not cover).
   The consequence is stated in the report: this shows the measured construct is load-bearing, not
   that an independently-discovered circuit is.
2. Mean-ablation (head output held at its mean over 100 bank prompts) rather than zero-ablation, at
   the final token only. Zeroing is further off-distribution, and ablating at every position would
   change the prefix computation and therefore both endpoints for reasons unrelated to the switch.
3. Re-ran both endpoints inside each ablation condition so d(0)=0 and d(1)=1 hold within condition.
   The alternative — measuring d against the unablated endpoints — would confound "the switch moved"
   with "the endpoints moved".
4. Three doses fixed before looking (3/6/10%). Rejected: choosing one dose after the pilot, which
   would have been selection on the outcome.

**Next step.** Two candidates, in order. (a) Localise: the differential heads are currently a per-pair
top-k list; check whether the same heads recur across pairs in gpt2-large, which would turn "a
pair-specific set" into a named circuit and explain the 15-fold gpt2-medium gap. (b) Explain the model
gap directly by running the same intervention in gpt2-small and one non-GPT-2 model, which would say
whether the effect tracks relative depth (Experiment 5) or the family.

On track? yes — deliverables are internally consistent again, both pass `check_render.py` with all ten
figures embedded and captioned, and the iteration added a causal result rather than only repairing
text; no unaddressed feedback and no STOP (S10 opened the localisation question above).
