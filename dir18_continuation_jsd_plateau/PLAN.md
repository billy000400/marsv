# PLAN - Does continuation JSD predict plateau strength?

> Working folder: `dir18_continuation_jsd_plateau`. The agent updates "Current status" and
> "Next step" every iteration. Disk (`PLAN.md`, `JOURNAL.md`, `RESULTS.md`, `REPORT.md`,
> `CHANGELOG.md`, `../BUDGET.md`, and `../CLAUDE.md`) is the only memory.

## Research question

For endpoint token `a`, estimate its context-averaged next-token distribution from Pythia's released
training stream:

```math
p_a(y)=P_{train}(y\mid a).
```

Test whether pairs with larger `JSD(p_a,p_b)` have sharper activation plateaus, measured by smaller
final-logit transition width `w`.

This is an observational predictor test. A positive result would show that training-corpus
continuation divergence predicts learned sharp transitions. It would **not** prove that JSD causes
plateaus, that plateaus are required for low training loss, or why the separation is sharp rather
than smooth.

## Definition of done

- JSD is reliable across two independent samples of Pythia's actual tokenized training stream.
- A frozen bank of at least 75 in-distribution endpoint pairs spans five JSD bins.
- The same bank is tested on the trained model and step 0 with Matthew's block-0-to-logit assay.
- `REPORT.md` gives a clear positive, null, or invalid-metric verdict. Null results are complete.
- When all current-best artifacts are finished, write an empty `STOP` file.

## Fixed setup

- **Primary:** current `EleutherAI/pythia-1.4b-deduped`, `step143000` and `step0`; never use `-v0`.
- **Scale check:** current `EleutherAI/pythia-410m-deduped`, `step143000`.
- **Optional formation subset:** 30 frozen pairs at `step1000`, `step8000`, `step32000`, and
  `step64000`.
- **Implementation:** native Hugging Face GPT-NeoX hooks, `eval()`, `torch.inference_mode()`, float32.
- **Data:** exact rows from `EleutherAI/pile-deduped-pythia-preshuffled`.
- Read `../BUDGET.md` and `../CLAUDE.md` every iteration. Do not replace the installed CUDA/PyTorch
  stack or install TransformerLens, JAX, or Flax.

## Eight-hour feasibility boundary

Do not download or reconstruct the full 602 GB released stream; Pythia's documentation says full
reconstruction can take more than a day.

Use two distant, row-aligned samples of 500,000 rows each. Each row has 2,049 token IDs, so the total
sample is about 2.05B tokens and roughly 4.1 GB. Obtain it by byte-range download or from a mounted
shard. Verify dtype, row shape, offsets, and checksums with the official reader.

Count only the 2,048 adjacent transitions **inside** each row. Never join two rows. Save only raw
50-point `d(t)` curves and scalar summaries during the GPU run; never dump full logits or all-layer
activations.

## S1 - Build a reliable JSD table

Treat the two 500,000-row samples as selection split A and confirmation split B.

1. Count every endpoint and its immediate successor using the exact Pythia tokenizer.
2. Keep lowercase, alphabetic, word-start tokens that decode as one complete token.
3. Require at least 20,000 endpoint occurrences in **each** split.
4. Compute unsmoothed, symmetric base-2 JSD over valid target IDs. Exclude padded output IDs that
   never occur in training.
5. Use `JSD_A` to create quintiles and select pairs; use `JSD_B` as the final predictor.

Before any plateau curve is viewed, require `Spearman(JSD_A,JSD_B) >= 0.90` on a fixed 10,000-pair
reliability bank, and median same-token split-half JSD below 25% of median between-token JSD.
If this fails, raise the count threshold, then enlarge the sample if local data and time allow.

## S2 - Freeze an in-distribution pair bank

Use three natural carrier contexts:

- `The thing was`
- `They said it was`
- `I thought it was`

Keep endpoints that are among the final 1.4B model's top-256 eligible word tokens in all three
contexts. Build prompts from token IDs and assert that each pair differs only in its final token.

Freeze 15-20 pairs in each `JSD_A` quintile before running the assay:

- endpoint frequencies within a pair differ by at most a factor of two;
- frequency and endpoint surprisal are balanced across JSD bins;
- no endpoint token is reused in the primary bank.

If 75 unique pairs are impossible, reduce the bank rather than use a dependent all-pairs design.
Save the exact contexts, token IDs, counts, `JSD_A`, `JSD_B`, selection rules, revisions, and seeds in
`results/pair_manifest.json`. Never revise this bank after seeing plateau curves.

Freeze a 15-pair calibration subset (three per bin). Proceed to the full scan only if at least 80%
of its curves are valid and the final-checkpoint width has `IQR(w) >= 0.05`. Otherwise report that
this model/assay lacks enough dynamic range for the correlation test.

## S3 - Run the plateau assay

For every pair and carrier context:

1. Collect the final-position residual stream **after block 0** for both endpoint prompts.
2. Apply norm-rescaled SLERP at 50 evenly spaced points, including both endpoints. Use the true
   cosine and a tested near-collinear fallback.
3. Patch only the final position after block 0 and run the remaining blocks.
4. Record final-position logits after final LayerNorm and unembedding, restricted to valid target
   IDs, and compute:

```math
d(t)=\frac{\lVert x(t)-x_A\rVert_2}
          {\lVert x(t)-x_A\rVert_2+\lVert x(t)-x_B\rVert_2}.
```

5. Save the raw curve and the existing project summary:

```math
w=t(d=0.9)-t(d=0.1).
```

Lower `w` means a sharper transition. Matthew did not define `w` or a binary cutoff, so raw curves
remain the primary evidence. Use the median `w` over the three contexts as the pair's outcome.

Required checks:

- patched `t=0` and `t=1` reproduce direct endpoint logits;
- prefix tokens and prefix residuals match within a pair;
- reversing a fixed subset preserves `w` within grid precision;
- undefined/non-monotone curves are shown, not forced into the correlation;
- invalid-curve rates are reported by JSD bin;
- `big <-> in` and `big <-> large` are labeled Pythia references, not assumed GPT-2 replications.

On ten frozen low/high-JSD pairs only, repeat with patch blocks `0, 6, 12, 18, 23`. A downstream
plateau should generally weaken when fewer blocks remain. Do not run a full layer or component sweep.

## S4 - Primary analysis

For each checkpoint:

- plot `JSD_B` versus median `w` and show all five frozen JSD bins;
- report `Spearman(JSD_B,w)` with an endpoint-aware bootstrap;
- compute the JSD between the two model next-token distributions in each carrier context, checking
  that corpus JSD actually predicts a distinction learned by the model;
- report the unadjusted result first, then one sensitivity model including endpoint frequency,
  continuation entropy, surprisal, and block-0 cosine/distance.

Do not treat contexts, interpolation points, layers, or heavily reused endpoints as independent.
Do not hard-match final-logit distance in the primary test; it may lie on the pathway from training
targets to plateau geometry.

Expected pattern: `rho < 0` after training, but little relationship at step 0. The optional
checkpoints test whether the negative relationship strengthens during training.

## Prespecified verdicts

- **Corpus JSD predicts model-output JSD and smaller `w`; step 0 does not:** predictive divergence
  is associated with learned plateau sharpening.
- **Corpus JSD predicts model-output JSD but not `w`:** predictive states are separated, but the
  separation is not generally implemented as a plateau.
- **Corpus JSD does not predict model-output JSD:** global `P(y|token)` is too coarse; a plateau null
  is inconclusive.
- **The relationship already exists at step 0:** architecture/tokenization geometry is a likely
  confound.
- **The relationship disappears after geometry adjustment:** report the total association, but do
  not claim JSD explains sharpness beyond learned endpoint geometry.

## Required outputs

- corpus and pair manifests;
- raw `d(t)` curves and compact summaries;
- `plots/jsd_reliability.png`, `plots/jsd_vs_width.png`, `plots/width_by_jsd_bin.png`, and
  `plots/reference_curves.png`;
- current-best `RESULTS.md`, self-contained `REPORT.md`, and `CHANGELOG.md`.

## Checklist

- [x] **S1 - Preflight and JSD:** verify data access/decoding, count both splits, and pass reliability.
  *(byte-range sample verified against the official .idx; Spearman(JSD_A,JSD_B)=0.9998, noise ratio 0.072)*
- [x] **S2 - Freeze pairs:** build and save the untouched pair manifest.
  *(prespecified top-256 filter: 60 endpoint-disjoint pairs, 14/13/11/10/12 per quintile; balance
  p=0.52 log-freq, 0.21 surprisal. Cap is 61 given 123 eligible endpoints.)*
- [x] **S3 - Calibrate:** pass endpoint/self-tests and confirm the frozen pilot has usable `w` range.
  *(valid-curve rate 1.000, IQR(w)=0.109; reversal 1.1e-5, prefix residual diff exactly 0.0; strict
  validity criteria: 0/1080 curves fail, max backslide 0.0000)*
- [x] **S4 - Primary runs:** finish 1.4B final, 1.4B step 0, then 410M final.
  *(rho(JSD_B,w) = -0.525 / -0.056 / -0.512)*
- [x] **S5 - Analyze:** produce figures, primary correlation, output-JSD validation, and one adjusted
  sensitivity analysis. *(10 figures; output-JSD rho=+0.751; partial rho=-0.384)*
- [x] **S6 - Finalize:** curate reports, document the exact data/compute scope, and write `STOP`.
  *(RESULTS.md + REPORT.md current-best, 11 captioned figures in both, render check passes; raw
  curves committed and independently re-scorable)*
- [x] **Optional formation subset:** step1000/8000/32000/64000 on the same frozen bank.
  *(relationship does NOT strengthen: already at full strength at the earliest measured step, 1000,
  at -0.582, then -0.456/-0.408/-0.628/-0.525 within overlapping CIs, while median `w` falls
  0.831 -> ~0.52)*
- [x] **Feedback #3 addition — 1,000-pair generality test:** secondary bank (same 123 endpoints,
  20-use cap, 200 pairs per selection quintile), 1.4B step143000 + step0, endpoint-clustered
  inference. *(rho = -0.486, dyadic endpoint-bootstrap CI [-0.603,-0.353], endpoint-label permutation
  p < 0.00025; step 0: -0.008, CI [-0.126,+0.109]; binned medians monotone 0.649 -> 0.499)*

## Fallback

Priority: reliable JSD -> 1.4B final -> 1.4B step 0 -> 410M -> intermediate/layer subsets.

If the primary run has not started by hour 4, drop the last two items. The minimum acceptable result
is 75 unique pairs on 1.4B final and step 0 with all validity checks. Reserve the final 20 minutes for
current-best plots and reports.

If neither byte-range access nor a mounted training shard works, stop after preflight. Do not replace
the corpus with OpenWebText or model-generated JSD while calling it Pythia training-corpus JSD.

## Out of scope

- Full 602 GB reconstruction; context-conditioned trigram JSD; dependent all-pairs scans.
- Full-logit/all-layer dumps, Jacobians, splines, or MLP/attention freezing.
- Claims about semantic groups, causal necessity, training-loss necessity, grokking, or region
  migration. *Deep Networks Always Grok* measures a different quantity: local complexity around
  character embeddings, not this endpoint assay.

## On-track check

End each `JOURNAL.md` entry with:

`On track? <yes/no> - <stage, % done, blocker if any>`

## Current status

S1-S6 complete on the **prespecified top-256 bank** (60 endpoint-disjoint pairs), plus a **1,000-pair
secondary bank** added for feedback round 3; all three operator feedback files are addressed
(`human_feedback.addressed.md`, `human_feedback_2.addressed.md`, `human_feedback_3.addressed.md`).

**Primary (60 endpoint-disjoint pairs, 3 contexts, 50 positions).** Held-out corpus
immediate-next-token JSD predicts narrower 10%-90% relative-logit transitions in trained Pythia:
`rho = -0.525`, CI [-0.701,-0.304], p = 1.7e-5; not at step 0 (`-0.056`, CI includes 0, median
`w = 0.831` with `IQR = 0.006` — a restricted range near the linear-response ceiling); `-0.512` on
410M as a cross-scale check (same bank, same corpus estimates). Predictor validated against model
output JSD (`+0.751`). Using the selection split instead of the holdout split changes nothing
(`-0.526`; the two agree at 0.99972). Learned sharpening `dw = w(trained) - w(step 0)`:
`rho = -0.517`. Mediation ladder: `-0.525` total -> `-0.277` (p = 0.032, still significant) adjusting
for model output JSD -> `-0.204` (p = 0.119, **not significant**) adjusting for output JSD plus the
five covariates. Strict curve validity measured, not assumed: 0 failures in 1,080 primary-bank curves
and 3,000 secondary-bank curves, max backslide 0.0000; all raw curves committed.

**Secondary (1,000 pairs, endpoint reuse, clustered inference).** `rho = -0.486`, dyadic
endpoint-bootstrap CI [-0.603,-0.353], endpoint-label permutation p < 0.00025 (0 of 4,000); step 0
gives `-0.008`, CI [-0.126,+0.109], p = 0.86. Ten binned medians fall 0.649 -> 0.499 essentially
monotonically, so the association is neither an artefact of the small matched bank nor visibly
non-monotone. Reported as an endpoint-dependent robustness analysis, never as 1,000 independent
observations. The relaxed top-512 bank remains a clearly labelled post-hoc secondary analysis
(`-0.419 / -0.155 / -0.320`).

Prespecified verdict branch: corpus JSD predicts model-output JSD and smaller `w`; step 0 does not —
stated as **learned output separation + overall 10%-90% transition width**, not "plateau sharpening",
because `w` and edge drift (0.076 vs the 0.184 no-plateau reference) correlate at +0.971 and cannot be
separated. RESULTS.md and REPORT.md are current-best with 12 captioned figures each and pass the
GitHub render check.

The formation subset **refutes the plan's expected pattern**: the negative relationship does not
strengthen during training. It is already comparable to later checkpoints at the earliest measured
step (`step1000`, `-0.582`) and then moves within overlapping CIs (`-0.456, -0.408, -0.628, -0.525`),
while median `w` falls 0.831 -> 0.512 by step 64000 and then reverses modestly to 0.541.

## Next step

None — the plan is complete and zero unaddressed feedback files remain, so `STOP` is written. The
natural follow-up, explicitly out of scope here, is a **context-conditioned** divergence estimate. It
has two motivations: testing whether it predicts width better at the late checkpoints where the global
estimate stops improving, and attacking the mediation null — a predictor that is not simply a proxy
for the model's own output separation. If a future iteration finds a new `human_feedback*` / `*REVIEW*`
file next to the stale `STOP`, delete `STOP`, address the feedback, and only re-write `STOP` when
clean again.

## References

- Matthew's post: https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge
- Released assay: https://github.com/MShinkle/activation_plateau_mechanisms
- Pythia models/data: https://github.com/EleutherAI/pythia
- Pythia paper: https://proceedings.mlr.press/v202/biderman23a.html
