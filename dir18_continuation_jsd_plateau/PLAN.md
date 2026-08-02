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
  *(75 endpoint-disjoint pairs, 15/quintile; balance p=0.92 log-freq, 0.81 surprisal)*
- [x] **S3 - Calibrate:** pass endpoint/self-tests and confirm the frozen pilot has usable `w` range.
  *(valid-curve rate 1.000, IQR(w)=0.115; reversal 1.1e-5, prefix residual diff exactly 0.0)*
- [x] **S4 - Primary runs:** finish 1.4B final, 1.4B step 0, then 410M final.
  *(rho(JSD_B,w) = -0.419 / -0.155 / -0.320)*
- [x] **S5 - Analyze:** produce figures, primary correlation, output-JSD validation, and one adjusted
  sensitivity analysis. *(6 figures; output-JSD rho=+0.729; partial rho=-0.267)*
- [x] **S6 - Finalize:** curate reports, document the exact data/compute scope, and write `STOP`.
  *(RESULTS.md + REPORT.md current-best, 7 captioned figures in both, render check passes)*
- [x] **Optional formation subset:** step1000/8000/32000/64000 on the same frozen bank.
  *(relationship does NOT strengthen: peaks at -0.660 by step 1000, decays to -0.419, while median
  `w` falls monotonically 0.831 -> 0.562)*

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

S1-S5 complete; the definition of done is met. Corpus JSD predicts sharper plateaus in trained Pythia
(`rho = -0.419`, CI [-0.585,-0.222], n = 75) and not at step 0 (`-0.155`, CI includes 0, median
`w = 0.831` with `IQR = 0.004`), replicating at 410M (`-0.320`). Predictor validated against model
output JSD (`+0.729`). Prespecified verdict: predictive divergence is associated with learned plateau
sharpening; attenuates to `-0.267` after geometry adjustment, so the claim is a total association,
observational only. RESULTS.md and REPORT.md are current-best and pass the GitHub render check; 6
figures embedded in both. Bank frozen at top-512 rather than top-256 (documented deviation).

The optional formation subset is also done, and it **refutes the plan's expected pattern**: the
negative relationship does not strengthen during training. It is strongest at the earliest
checkpoint run (`step1000`, `rho = -0.660`) and decays to `-0.419` at `step143000`, while median `w`
falls monotonically 0.831 -> 0.562. Plateaus keep sharpening; a context-free corpus statistic
explains a shrinking share of which pairs are sharp.

## Next step

None — the plan is complete and `STOP` is written. The natural follow-up, explicitly out of scope
here, is a **context-conditioned** divergence estimate, to test whether it retains predictive power
at the late checkpoints where the global estimate fades. If a future iteration finds a new
`human_feedback*.md` / `*REVIEW*` file next to the stale `STOP`, delete `STOP`, address the feedback,
and only re-write `STOP` when clean again.

## References

- Matthew's post: https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge
- Released assay: https://github.com/MShinkle/activation_plateau_mechanisms
- Pythia models/data: https://github.com/EleutherAI/pythia
- Pythia paper: https://proceedings.mlr.press/v202/biderman23a.html
