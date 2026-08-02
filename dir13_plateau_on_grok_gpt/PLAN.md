# PLAN - Do Grokking and Matthew-style activation plateaus emerge together?

> Working folder: `dir13_plateau_on_grok_gpt`. The agent rewrites "Current status" and "Next step" and ticks stages each iteration. Disk (`PLAN.md`, `JOURNAL.md`, `RESULTS.md`, `REPORT.md`, `CHANGELOG.md`, `../BUDGET.md`, and `../CLAUDE.md`) is the only memory.

## Core question

This direction is not merely asking whether a trained GPT has sharp activation transitions. It asks:

> In a 12-layer Shakespeare GPT that actually reproduces the Grokking paper's Figure 9 phenomenon, do Matthew-style activation plateaus appear or sharpen during the same training period as the second local-complexity descent and delayed adversarial robustness?

The current 3,500-step character GPT is only a pilot. Its validation accuracy shows that it learned next-character prediction, but the existing report explicitly did not measure Figure 9's Grokking phenomenon. Until that gate is passed, its plateau curves cannot establish a relationship between Grokking and plateaus.

## Models required

Use two matched fresh training runs plus the existing checkpoint as a pilot:

1. **Character model - paper-faithful control.** A 12-layer, 12-head GeLU causal GPT trained from scratch for next-character prediction on the same Shakespeare corpus. This is the closest reconstruction of Figure 9.
2. **BPE model - primary bridge to Matthew.** The same architecture, corpus split, optimizer, schedule, and training horizon, trained from scratch with the standard GPT-2 byte-level BPE tokenizer. This model is required because Matthew's `big`, `in`, and `large` completions can be single tokens, allowing his assay to transfer without replacing words with arbitrary character transitions.
3. **Existing 3,500-step character checkpoint - pilot only.** Evaluate it with the same Grokking metrics, but do not call it a Grokking-paper model merely because ordinary validation accuracy is high.

The BPE model is an adaptation, not the model used in Figure 9. Report its vocabulary size, parameter count, token exposure, raw-character exposure, and corpus epochs so tokenization and training exposure are not silently confounded.

## Source-locked definitions

### The Grokking-paper signature

Figure 9 reports a 12-layer, 12-head GeLU GPT trained on next-character Shakespeare. The relevant signature is:

- train/test/random local complexity measured across training;
- ordinary test accuracy;
- `epsilon=0.03` `l_inf`-PGD adversarial accuracy in token-embedding space;
- a second local-complexity descent that starts before the test-accuracy peak and continues while adversarial robustness emerges.

Use the paper's documented defaults where applicable: 1,024 train/test/random points for local-complexity estimation, `r=0.005`, `P=25`, Adam, zero weight decay, and 99% confidence intervals. Record every missing or reconstructed detail. The authors did not release the exact GPT code/checkpoint, so the goal is a qualitative Figure 9 replication, not numerical identity.

### Matthew's plateau assay

Use Matthew's released repository and configuration as the source of truth:

```yaml
model_name: "gpt2-large"
shared_context: "The house was"
token_pairs:
  - ["big", "in"]
  - ["big", "large"]
n_steps: 50
```

For each pair and interpolation layer:

1. Collect final-position `resid_post` endpoints for the two full prompts.
2. Apply Matthew's `slerp_rescale`: spherical interpolation of direction and linear interpolation of L2 norm.
3. Use exactly 50 evenly spaced interpolation values including both endpoints.
4. Patch only the final sequence position at that `resid_post` layer.
5. Sweep every available interpolation layer. Record Matthew's downstream hooks: `attn_out`, `resid_mid`, `mlp_post`, `mlp_out`, `resid_post`, and final logits.
6. Compute the same relative distance:

   \[
   d(t)=\frac{\|x(t)-x_A\|_2}{\|x(t)-x_A\|_2+\|x(t)-x_B\|_2}.
   \]

Raw `d(t)` curves are the primary evidence. A plateau stays near one endpoint, changes sharply, and stays near the other. Do not replace Matthew's examples, interpolation, or 50-step grid with a new dataset or assay.

## Experiment 1 - does the trained GPT reproduce Figure 9?

This is a mandatory validity gate.

1. Run the Figure 9 measurements on every usable checkpoint from the existing character model.
2. If its 3,500-step horizon or checkpoint coverage is insufficient, say **not established** and start a fresh character run; do not extend a run whose optimizer/schedule was designed to end at 3,500 steps.
3. Train the fresh character model through approximately `10^5` optimization steps, subject to `../BUDGET.md`, saving log-spaced checkpoints and denser checkpoints around visible transitions.
4. Plot test accuracy, `epsilon=0.03` adversarial accuracy, and train/test/random local complexity on the same training axis.

Gate verdict:

- **Pass:** the qualitative temporal ordering from Figure 9 appears.
- **Fail within tested setup:** the run reaches the planned horizon with valid measurements but does not show that ordering.
- **Inconclusive:** implementation validation fails, training ends too early, or a missing paper detail materially changes the result.

If this gate does not pass, the project may still report plateaus in a Shakespeare GPT, but it must not claim evidence about the relation between Grokking and plateaus.

## Experiment 2 - train and validate the BPE Grokking model

1. Tokenize the three exact prompts and save token IDs, decoded tokens, and each completion token's frequency in the BPE training corpus. Assert that `" big"`, `" in"`, and `" large"` are each one token after the common context. If any is not one token or never appears in training, stop the exact Matthew assay and report the failed gate; do not invent a multi-token patch or silently substitute another word. If a token is merely rare, proceed but report its count prominently.
2. Train the BPE model from initialization using the same frozen setup and checkpoint schedule as the fresh character control.
3. Log optimization steps, BPE tokens, approximate raw characters, and corpus epochs. Ordinary next-character and next-BPE-token accuracies are not numerically comparable.
4. Apply the identical Figure 9 measurement pipeline without tuning attack strength or local-complexity radii after seeing the BPE result.

The BPE model must receive its own `pass`, `fail`, or `inconclusive` Figure 9 verdict. Only a passing BPE run can directly test the Grokking/plateau relationship using Matthew's exact examples.

## Experiment 3 - Matthew's exact assay across training

### Primary: BPE exact examples

Run `big/in` and `big/large` at checkpoints selected from the already-computed Grokking curves:

- initialization or earliest usable checkpoint;
- before the first local-complexity peak;
- near the local-complexity peak;
- at the start of the second local-complexity descent;
- near the onset of `epsilon=0.03` adversarial robustness;
- the final checkpoint.

Freeze checkpoint selection before inspecting plateau curves. Run the complete Matthew layer sweep at each selected checkpoint. First reproduce the original GPT-2 Large config with Matthew's unmodified code if compute permits; otherwise preserve his code path and change only the model adapter.

### Secondary: character-token control

The character model cannot represent Matthew's three words as single tokens. Retain only the two previously requested one-token controls:

```yaml
shared_context: "The house was"
token_pairs:
  - ["b", "i"]
  - ["b", "l"]
n_steps: 50
```

These are two pairwise tests, `b <-> i` and `b <-> l`, not the strings `bi` and `bl`. Label them tokenizer controls, not replications of Matthew's examples. Run them at the same training phases as the BPE assay. Do not generate 40 additional letter transitions.

## Experiment 4 - relation between the two phenomena

Put both measurements on one checkpoint timeline:

- Grokking side: test accuracy, adversarial accuracy, and train/test/random local complexity.
- Plateau side: raw layerwise `d(t)` curves for `big/in` and `big/large`; transition width may be shown only as a compact descriptive summary across checkpoints.

The final report must choose one bounded conclusion:

1. **Temporally associated:** the plateau curves sharpen during the same checkpoint interval as the second local-complexity descent and delayed robustness.
2. **Plateaus precede Grokking:** clear plateaus exist before the Figure 9 transition.
3. **Plateaus follow Grokking:** plateaus appear only after the Figure 9 transition.
4. **No visible temporal relationship:** plateau shape remains stable or changes in a different interval.
5. **Primary relationship not testable:** the BPE model does not reproduce Figure 9 or its exact-token/training-frequency gate fails. A passing character run may then provide only secondary evidence from the `b/i` and `b/l` analogues, not a Matthew-exact relationship result.

This is evidence about temporal association, not causation. One training run cannot show that Grokking creates plateaus.

## Experiment 5 - all-pairs character interpolation: does every character own a plateau?

**Operator request (2026-08-01), reopening the direction.** Feedback #3 asked one character (the
comma) against the other 64. This series generalises that to the whole vocabulary: interpolate from
*every* character to *every* other character and ask whether each character sits in its own plateau,
then say what the plateaus correspond to.

Everything below reuses the existing frozen code path - no new assay. `experiments/matthew_assay.py`
(`slerp_rescale` on the final-position `resid_post`, 50 evenly spaced `t` including both endpoints,
patch only the final position, relative distance `d(t)` in final-logit space, `w_{10->90}` on the
isotonic copy) is the source of truth, exactly as in `comma_sweep.py`/`context_sweep.py`. Do not
invent a new plateau score, a new interpolation scheme, or a new step grid.

### 5.1 The sweep

- Vocabulary: the 65-character tinyshakespeare vocabulary already restored in `comma_sweep.py`.
- Shared context: `"The house was "` (the frozen S6/feedback-#3 context), so the new numbers are
  directly comparable to the existing comma sweep and to the `b/i`, `b/l` controls.
- Pairs: all `C(65,2) = 2080` unordered character pairs, at interpolation block 0 of the final
  (step-30000) fresh character checkpoint. Endpoint order is fixed by vocabulary index (`A` = lower
  index) so the run is deterministic.
- **Symmetry check (required, cheap):** re-run 100 randomly chosen pairs with the endpoints swapped
  and report the median `|w(A,B) - w(B,A)|`. `d(t)` is not symmetric by construction, so the heatmap
  may only be drawn symmetric if this check says the asymmetry is negligible; otherwise draw the
  full ordered 65x65 matrix and say so.
- Reuse the existing diagnostics per pair: endpoint reproduction error, prefix invariance,
  `d(0)`, `d(1)`, `max_iso_dev`. A pair failing the endpoint checks is dropped and counted, never
  silently kept.

Per pair record: `w_{10->90}`, the midpoint crossing `t*` (isotonic `t` at `d = 0.5`), the strict
plateau flag from `is_plateau`, `max_iso_dev`, the logit-space endpoint separation, and the model's
next-character probabilities `p(A | context)` and `p(B | context)`.

### 5.2 The per-character question ("is each character in its own plateau?")

Make this a decidable statement rather than an impression. For character `c`, over its 64 partners:

- `med_w(c)` - median transition width. Small = every path in and out of `c` is a sharp switch.
- `flat_frac(c)` - fraction of partners for which the path stays within 0.1 of the `c` endpoint for
  at least 10% of the path (`t_lo >= 0.10` when `c` is the `A` endpoint, `t_hi <= 0.90` when it is
  `B`). This is the "`c` has a basin of its own" statistic.
- `strict_frac(c)` - fraction of partners passing the frozen `is_plateau` rule (`w <= 0.25`, both
  margins `>= 0.10`, near-monotone).

Report the joint distribution over the 65 characters and state plainly which of these holds:
(i) every character has its own plateau; (ii) only a subset does (name them and their class);
(iii) plateau-ness is a property of the *pair*, not of either character alone. Decide (iii) with a
simple variance decomposition: fit `w_ij ~ a_i + a_j` by least squares and report the fraction of
variance in `w` explained by per-character terms versus the residual (pair-specific) term. High
per-character variance supports "each character owns a region"; a dominant residual supports "the
sharpness lives in the pair".

### 5.3 What do the plateaus correspond to? (the mechanistic hooks)

The point of the sweep is to license a hypothesis, so measure the two things that can distinguish
the obvious candidate explanations. Both are cheap additions to the same forward passes:

1. **Readout-decision test.** Along each path also record `argmax` of the final logits at every `t`
   and the number of distinct `argmax` characters visited. Then compare the `d(t)` jump location
   `t*` with the first `argmax` flip `t_flip`. If `|t* - t_flip|` is small for most pairs and paths
   visit exactly 2 `argmax` regions, the plateau boundary *is* the model's next-character decision
   boundary, and a "plateau" is the set of residual states that decode to the same prediction.
2. **Where is the sharpness generated?** Re-run a random 200-pair subsample at interpolation blocks
   0, 4, 8 and 11 (block 11 leaves only `ln_f` + unembedding downstream, so it is the near-linear
   readout reference). If `w` grows toward the shallow end, the sharpness is produced by the
   intervening blocks, not by the geometry of the unembedding.

Two controls, both mandatory before the hypothesis is written:

- **Learned-vs-init.** Repeat the full 2080-pair sweep at the step-0 checkpoint. If the width
  distribution at init already matches the final one, the structure is architectural, not learned.
- **Plausibility confound.** Feedback #3 found width correlates with next-character probability
  (median Spearman `rho = -0.41` across contexts). Recompute that correlation on the all-pairs set
  using `max(p(A), p(B))` and `|log p(A) - log p(B)|`, and report partial correlations against
  endpoint separation so the report does not attribute to "plateaus" what is really "the model is
  confident about one endpoint".

### 5.4 Figures

Six figures, each motivated by a claim in 5.2-5.3 and subject to `../CLAUDE.md` rules 12 and 13
(visible bold `**Figure N.**` caption under every embed, axes and every legend entry defined in
Methods first, CVD palette, no red-vs-green, no series named by colour):

1. 65x65 heatmap of `w`, axes ordered by character class (space/newline, punctuation/digit,
   upper-case, lower-case), diagonal masked, `viridis`.
2. Per-character strip or box plot of `w` over the 64 partners, characters sorted by `med_w`, with
   `flat_frac` on a twin axis - the direct answer to "is each character in its own plateau".
3. Small multiples of raw `d(t)` curves for 6 representative characters (all 64 partners overlaid
   each): the extremes of `med_w` plus one per character class. Raw curves stay the primary evidence.
4. `t*` versus `log p(A) - log p(B)`, marker by character class - does the boundary sit where the
   two characters become equally likely?
5. Readout-decision test: histogram of `t* - t_flip` and the distribution of distinct `argmax`
   regions per path.
6. Controls in one panel: init versus final width distributions, and `w` by interpolation block on
   the 200-pair subsample.

### 5.5 The hypothesis (the actual deliverable)

`RESULTS.md` and `REPORT.md` each get a subsection **"What do the plateaus correspond to?"**
containing a **3-4 sentence hypothesis**, written from these results only. It must:

- name what a plateau region corresponds to in this model, in plain words;
- cite the specific numbers that support it (which figure, which statistic);
- name the leading alternative the data does *not* rule out;
- end with one concrete falsifiable prediction a follow-up run could test.

No more than four sentences. Speculation beyond what Figures 1-6 support does not belong there;
put longer discussion in ordinary prose around it.

## Success criterion

`RESULTS.md` and `REPORT.md` are complete only when they contain:

- a Figure 9 gate verdict for the existing character checkpoint and the fresh character run;
- a separate Figure 9 verdict for the BPE run;
- Matthew-faithful 50-step, all-layer `big/in` and `big/large` curves across selected BPE checkpoints, if the token and model gates pass;
- the two character-token controls, with no large letter-transition dataset in the headline analysis;
- one checkpoint-aligned figure showing both Grokking and plateau evolution;
- a bounded relationship verdict from the five cases above, with reconstruction limitations stated prominently;
- the Experiment 5 all-pairs series: the 2080-pair sweep with its symmetry and endpoint checks, an
  explicit per-character verdict (i/ii/iii from 5.2) backed by the `a_i + a_j` variance
  decomposition, the readout-decision and depth measurements, the init and plausibility controls,
  the six figures of 5.4, and the 3-4 sentence hypothesis of 5.5.

Null results are complete when the validity gates pass. When complete, write an empty `STOP` file.

## Required artifacts

- `MODEL_SPEC.md`: confirmed paper facts versus reconstruction choices for both tokenizers.
- `configs/grok_char.yaml`, `configs/grok_bpe.yaml`, `configs/matthew_bpe.yaml`, and `configs/matthew_char_control.yaml`.
- `tokenization_check.txt` for all exact prompts under both tokenizers.
- saved checkpoints and raw Figure 9 metrics for both fresh runs.
- raw Matthew outputs for every selected checkpoint, interpolation layer, recording layer, and hook.
- `plots/grokking_char.*`, `plots/grokking_bpe.*`, `plots/matthew_bpe_by_checkpoint.*`, and `plots/joint_timeline.*`.
- a rewritten `REPORT.md` that moves the existing 40-letter result to `CHANGELOG.md` or a clearly labeled exploratory appendix.
- Experiment 5: `experiments/allpairs_sweep.py` and `experiments/plot_allpairs.py`;
  `results/allpairs_raw.npz` (per-pair `d(t)` and per-`t` `argmax`) plus `results/allpairs_summary.json`
  (per-pair stats, per-character stats, variance decomposition, correlations, all diagnostic checks);
  `plots/allpairs_{width_matrix,width_by_char,curves_small_multiples,boundary_vs_logp,readout_decision,controls}.png`.

## Stages

- [x] **S1 - Existing reconstruction and exploratory plateau assay.** The 3,500-step character model and 40-letter exploratory result exist, but they do not answer the joint question.
- [x] **S2 - Source-lock both assays.** Port the Figure 9 measurements and Matthew's released config/code; validate endpoint fidelity and tokenization.
- [x] **S3 - Evaluate the existing checkpoint.** Pilot char Figure-9 verdict = **FAIL** (first LC descent + emerging robustness, no second descent within 3,500 steps).
- [x] **S4 - Fresh character replication.** Trained 30k steps; 14-checkpoint Figure-9 curve; verdict = **FAIL** (LC monotone to 8.1, adv→0.528, no second descent).
- [x] **S5 - Fresh BPE replication.** Trained; 10-checkpoint Figure-9 curve; verdict = **FAIL** (LC monotone to 95, adv→0.187, no second descent).
- [x] **S6 - Checkpoint-aligned plateau assays.** Ran Matthew's exact code path with `b/i`,`b/l` char controls across the 6 frozen phases (steps 0,56,831,7819,17500,30000). Plateau **emerges during the first LC descent**: block-0 final-logit width 0.80 (init) → 0.33 (step 831), flat to 30k; formed *before* robustness saturates. `plots/matthew_char_ctrl_by_checkpoint.png`, `plots/joint_timeline_char_ctrl.png`; raw `results/matthew_char_ctrl_{raw.npz,summary.json}`.
- [x] **S7 - Joint analysis.** `plots/joint_timeline.png` + bounded relationship verdict = **PLAN case 5 (primary relationship not testable)**, refined by the S6 secondary temporal observation (no coupling to grokking).
- [x] **S8 - Rewrite the report.** De-emphasised the 40-pair reconstruction dataset (now clearly-labelled *exploratory*); S6 char controls are the primary plateau evidence. STOP written.
- [x] **S9 - All-pairs character sweep (Experiment 5).** Reopened 2026-08-01 by operator request; COMPLETE.
  - [x] **S9a - Sweep.** `experiments/allpairs_sweep.py`: 2080 pairs at interpolation block 0 of the
        step-30000 char checkpoint, 50-step slerp, final-logit `d(t)`; per-`t` `argmax` recorded;
        endpoint/prefix diagnostics and the 100-pair swap-symmetry check. -> verify: every pair's
        `d(0) < 1e-3`, `d(1) > 1 - 1e-3`, and `matthew_assay.self_test()` passes before the sweep runs.
  - [x] **S9b - Per-character verdict.** `med_w`, `flat_frac`, `strict_frac` per character plus the
        `w_ij ~ a_i + a_j` variance decomposition. -> verify: one of the three verdicts (i/ii/iii in
        5.2) is stated with its supporting fraction-of-variance number.
  - [x] **S9c - Mechanism and controls.** Readout-decision test (`t*` vs `t_flip`, number of `argmax`
        regions), 200-pair depth subsample at blocks 0/4/8/11, full sweep re-run at step 0, and the
        plausibility partial correlations. -> verify: each control produces a number that either
        supports or contradicts the hypothesis, and contradictions are reported.
  - [x] **S9d - Figures + hypothesis.** The six figures of 5.4 embedded with visible `**Figure N.**`
        captions in both deliverables, and the 3-4 sentence hypothesis of 5.5. -> verify:
        `python3 experiments/check_render.py REPORT.md RESULTS.md` exits 0, and the figure/caption
        grep from `../CLAUDE.md` rule 12 matches the embed count.

- [x] **S10 - Frozen-block training test (the hypothesis's own falsifiable prediction).** Reopened
      2026-08-02. `experiments/train_frozen.py` retrains the reference character recipe from scratch
      with a block group held at its step-0 weights (`frozen_early` = blocks 1-4, the group the
      ablations implicate; `frozen_late` = blocks 8-11, the specificity control), everything else
      identical (same corpus SHA, seeds, optimizer, 30k schedule, batch, checkpoint grid).
      `experiments/frozen_assay.py` runs the frozen assay on each at its matched-accuracy checkpoint
      and its final checkpoint, against the reference run at step 0 / step 2500 / step 30000 on the
      same 150 pairs, plus the depth control at injection blocks 0/4/8. -> verify: both frozen runs
      reach the reference run's final validation accuracy (0.550) before being assayed, and the
      prediction ("frozen-early stays near the untrained width 0.80") is reported as confirmed or
      falsified with the paired per-pair shifts that decide it. **DONE 2026-08-02: verified (0.5625 /
      0.5622 final val acc; matched at steps 2750 / 2500) and the prediction is FALSIFIED** - w = 0.471
      (early) and 0.484 (late), paired dw +0.107 / +0.120 vs the trained reference, with the depth
      control showing the sharpening relocated to blocks 5-7. Figure 23 in both deliverables.

## Fallback

Prioritize in this order: Figure 9 validity gate, BPE training/validation, Matthew's exact BPE examples, then the two character controls. If either long training run ends before the relevant transition, preserve all checkpoints and report **inconclusive** rather than treating ordinary convergence as Grokking. Reserve the final 20 minutes for figures, current-best `RESULTS.md`/`REPORT.md`, `CHANGELOG.md`, and `STOP`.

## Out of scope

- No new minimal-pair dataset or 40-pair letter search in the primary *Grokking* analysis. The
  Experiment 5 all-pairs sweep is an explicitly operator-requested character-level series: it is
  reported as its own section and must not be used to restate or revise the Grokking relationship
  verdict (still case 5), which no new interpolation data can change.
- No random-direction ray assay as evidence for Matthew-style plateaus.
- No new plateau score suite, semantic clustering, steering, or manifold interpretation.
- No silent multi-token interpolation workaround.
- No causal claim from temporal correlation.
- No claim about the paper's exact checkpoint, which is unavailable.
- Do not install or replace the existing CUDA build of torch, torchvision, TransformerLens, JAX, or Flax.
- Read `../BUDGET.md` and `../CLAUDE.md` every iteration; keep current-best results in `RESULTS.md`/`REPORT.md` and history in `CHANGELOG.md`.

## On-track check

End each `JOURNAL.md` entry with: `On track? <yes/no> - <stage, % done, blocker if any>`.

## Current status

**PLAN COMPLETE (S1-S10) + operator feedback #4 (2026-08-02 file) addressed + all five PLAN-named
follow-ups DONE (denser Figure-9 grid, readout rebalancing, MLP-gain intervention, per-block scan,
frozen-block training test).** All five `human_feedback*` files are `.addressed.md`; zero
unaddressed feedback remains.

- **S10 frozen-block training test COMPLETE (2026-08-02, latest) - the hypothesis's own prediction is
  FALSIFIED.** Both frozen runs finished the full 30,000 steps at *better* validation accuracy than the
  reference (0.5625 / 0.5622 vs 0.5502; matched accuracy at steps 2750 / 2500). Prediction on record:
  frozen-early (blocks 1-4) stays near the untrained width 0.80. Outcome: median `w` **0.471**
  (IQR 0.403-0.524), i.e. 73% of the reference run's sharpening (0.803 -> 0.351) recovered without any
  trainable weights in the implicated blocks. The specificity control decides the reading: frozen-late
  (blocks 8-11) ends at **0.484**, paired median `dw` +0.120 vs +0.107 for early, so freezing *any*
  four blocks costs the same and the shortfall is generic capacity loss. Against the reference at the
  matched step (2500, 0.443) the gap is only +0.033 / +0.038 - freezing mostly *slows* the sharpening;
  what is lost is the sharp tail (strict rate 10% -> 0.7% / 0%). **Depth control shows the computation
  relocated:** injection blocks 0/4/8 give 0.351/0.761/0.805 (reference, sharpening in blocks 1-4),
  0.484/0.793/0.806 (frozen-late, same profile) and 0.471/**0.471**/0.788 (frozen-early) - zero change
  across the frozen group, so all the sharpening is now produced by blocks 5-7. Geometry otherwise
  unchanged (`t*` 0.491/0.495, endpoints differ 84%/93%, 3 argmax regions, |t*-t_flip| 0.062/0.059,
  partial rho -0.61/-0.60). **Interpretation change:** "blocks 1-4 build the sharpness" is true at
  inference but false as a training-time claim; the sharp transition is a *relocatable* computation.
  Both hypothesis paragraphs rewritten and re-ended on a new falsifiable prediction (freeze blocks 1-7,
  train only the top of the stack). New **Figure 23** in both deliverables; Limitation 6 updated,
  Limitation 7 added. `check_render.py`: ALL CHECKS PASS (26 figures / 26 captions per file).

- **Grid densified on the passing run (2026-08-02, later).** The corrected gate's LC local maximum for
  the fresh char run rested on ONE log-spaced checkpoint (step 56). The run had 35 saved checkpoints but
  only 14 evaluated, so 10 unmeasured ones (steps 1,2,6,9,23,36,88,138,339,531) were re-run through
  `experiments/fig9.py` with everything frozen - no training extended, no threshold changed. Grid
  **14 -> 24**. Result: the turnaround is real and larger than measured before. LC 491.2 @15 -> 987.7 @23
  -> **989.1 @36** -> 769.4 @56 -> 8.1, so **three** points now sit above the first minimum; local max
  **769 @56 -> 989 @36**, rise **278 -> 498** units (2.9x -> **5.1x** the frozen 5% tolerance, ~110x the
  CI); verdict stays **PASS**. Sustained robustness onset **831 -> 531** (step 531 = 0.077 was never
  measured before). Pilot char PASS and BPE FAIL unchanged. Association window restated: second descent
  **36 -> 30,000**, robustness onset **531**, so the `b/i`,`b/l` sharpening interval (56 -> 831) now
  *strictly contains* the onset; verdicts unchanged (primary case 5, character analogues case 1). The
  "local maximum = one checkpoint" caveat is now scoped to the **pilot** run only. Two incidental fixes:
  `/tmp/tinyshakespeare.txt` was re-fetched (SHA-256 verified against `train_meta_grok_char.json`), and
  `fig9.py` was run at `--vram_frac 0.225` because its 0.05 default OOMs - the PGD batch was deliberately
  NOT halved, since that would change the attack's random starts and break comparability.

- **Figure-9 gate CORRECTED (2026-08-02).** `experiments/fig9_verdict.py` had located the LC minimum
  with `np.argmin`, i.e. the GLOBAL minimum = the last checkpoint, so no rise could ever be found after
  it. Rewritten to find the first significant local minimum -> its local maximum (= second-descent
  onset) -> a sustained second descent, plus the two preregistered ordering checks that were missing
  (onset before the clean-accuracy peak; adv robustness rising >= 0.05 from its value at the onset,
  with a sustained-onset definition). Rerun on the existing JSONs only, no training extended.
  **Pilot char FAIL -> PASS** (LC 1940 -> 484 @19 -> 1043 @33 -> 68), **fresh char FAIL -> PASS**
  (1940 -> 491 @15 -> 769 @56 -> 8.1; rise 278 vs tol 96.6, CI +-3; adv 0.0006 -> 0.530, sustained
  onset 831), **fresh BPE FAIL** (only upturn 30 units = 1.4% of range, inside the frozen 5% tol).
  **Bounded verdict now split:** primary (Matthew-exact `big/in`,`big/large`, BPE-only tokens) stays
  **PLAN case 5**, because BPE is the run that still fails; the `b/i`,`b/l` character analogues sit on
  a passing run and upgrade to **PLAN case 1 (temporally associated)** - plateau width 0.80 -> 0.33
  between steps 56 and 831, inside the second-descent window (56 -> 30000) and across the sustained
  robustness onset. Caveats kept: one run, six checkpoints, onset at step 56 overlaps initial fit, and
  the LC local maximum is resolved by one checkpoint. If new feedback or a new operator request arrives:
delete `STOP`, address it, run `python3 experiments/check_render.py REPORT.md RESULTS.md` (must exit 0)
before finishing, and re-write `STOP` only when clean again.

- **S9 / Experiment 5 - all-pairs character sweep (2026-08-01).** All `C(65,2) = 2080` character pairs
  at interpolation block 0 of the step-30000 fresh-char checkpoint, same frozen code path
  (`experiments/allpairs_sweep.py` -> `analyze_allpairs.py` -> `plot_allpairs.py`; raw
  `results/allpairs_raw.npz`, stats `results/allpairs_summary.json`).
  - **Diagnostics**: 0 pairs dropped (max `d(0)` 3e-6, min `d(1)` 0.999998, prefix error exactly 0.0),
    all 2080 curves exactly monotone. Swap symmetry median = max |dw| = **0.000** - an algebraic
    identity (`d(t) -> 1-d(1-t)` on a symmetric grid) that the check confirms, so the heatmap is drawn
    symmetric.
  - **Per-character verdict = PLAN case (i)**: every character has a basin (`flat_frac` >= 0.86 for all
    65, = 1.00 for 59); per-character median widths 0.264 (`o`) - 0.590 (`3`); the additive fit
    `w_ij ~ a_i + a_j` explains **78.2%** of the variance (adjusted 77.6%) vs a 3.0% permutation null,
    ruling out cases (ii) and (iii). Median width 0.355; strict rule 182/2080 (8.8%).
  - **Mechanism**: 91% of next-character prediction changes fall inside the transition window, 79% of
    pairs have all changes inside it, 80% have single-prediction flat arms, median |t*-t_flip| = 0.045.
  - **Controls**: at init all 2080 paths are straight (median w 0.803 -> 0.355 trained, 0 strict,
    Mann-Whitney p < 1e-300) and sharpness is generated by blocks 1-4 (w 0.344/0.763/0.806/0.806 at
    blocks 0/4/8/11). Plausibility confound survives partial correlation (rho = -0.59 both ways), so it
    is reported as the live alternative.
  - **Hypothesis (5.5)**: a plateau is the set of final-position residual states that decode to the
    same next-character prediction, one basin per character, built by blocks 1-4. Six figures embedded
    in both deliverables (Figures 14-19).
- **Rule-12 fix (2026-08-01).** Both deliverables had all 16 captions in `![...]` **alt text** (invisible
  on GitHub) with out-of-order numbering. Rewritten: 22 embeds each, short alt text + a visible
  `**Figure N.**` caption under every image, sequential numbering 1-22 in reading order, each figure
  cited by number and motivated by preceding prose. `check_render.py` -> 0 problems.
- **Grokking side** as corrected above: gates PASS / PASS / FAIL, primary verdict still case 5 (BPE),
  character analogues case 1. S9 did not and cannot revise this.
- Earlier series stand as reported: comma-vs-64 sweep (median w 0.340, 1/64 strict), 8-context control
  (576 pairs, 0/576 near-linear, per-context medians 0.313-0.436, rho negative in 9/9, median -0.41),
  CVD compliance across every figure, and the `check_render.py` guard from feedback #4.

- **Readout-rebalancing intervention DONE (2026-08-02, later still).** The last PLAN-named follow-up.
  `experiments/rebalance_probe.py` adds a constant to one unembedding row (pure readout bias, every
  residual activation on the path bit-identical) and asks whether the plateau boundary follows the
  decision boundary it moves. 1,873 of 2,080 pairs (207 predict the same character at both endpoints),
  block 0, step 30000. Two findings: (a) `d(t)` is a ratio of distances *between* logit vectors, so a
  common bias cancels **exactly** - measured deviation 1.3e-6, i.e. `w` and `t*` are invariant to any
  readout bias, which also makes this test structurally unable to check the plausibility account's
  "width changes" prediction (that must be tested inside blocks 1-11); (b) the readout gap swings a
  median 21.9 nats along the path, so the boundary is stiff - 2.44 nats moves it 0.020, and the 5.28
  nats needed to force it to the midpoint moves it 0.052; median `|t* - t_gap|` 0.025 -> 0.015 -> 0.035.
  **Interpretation change:** the S9 `t* ~ t_flip` alignment is correlational, not causal - both are
  downstream of one sharp residual-stream change built by blocks 1-4. The hypothesis stands as a
  *description* of the basins, not their mechanism. New Figure 20 in both deliverables; the three
  exploratory figures renumbered 21-23.

- **MLP-gain intervention DONE (2026-08-02, latest).** The successor the rebalancing probe named.
  `experiments/mlp_gain_probe.py` scales the MLP-branch output of a block group by a gain `g` (all else
  untouched) and re-runs the frozen assay with endpoints recomputed under the modified model; 150 random
  pairs, block 0, step 30000. Early group (blocks 1–4): median `w` **0.796** (g=0) → 0.533 (g=0.5) →
  0.351 (unmodified) → **0.305** (g=1.5), strict rate 0.00/0.00/0.10/**0.30**; at g=0 the width is back
  at the untrained 0.803 and all 150 pairs widen (median Δw +0.433). Late group (blocks 8–11): 0.337 /
  0.333 / 0.380, median |Δw| ≤ 0.025. `t*` barely moves (|Δt*| ≤ 0.074). So blocks 1–4 **causally** set
  the sharpness — Experiment 5's depth *observation* upgraded to an intervention — while the
  plausibility alternative survives, now pinned to those same early weights. New **Figure 21** in both
  deliverables; the three exploratory figures renumbered 22–24.

- **Per-block MLP scan DONE (2026-08-02, latest).** The successor the MLP-gain intervention named.
  `experiments/mlp_block_scan.py` deletes each of blocks 1–4's MLP branch on its own (`g=0`) and
  re-measures both candidate mechanisms under every ablated model; same 150 pairs, block 0, step 30000.
  **(a) Distributed, front-loaded:** median `w` 0.351 (unmodified) → 0.541 / 0.478 / 0.446 / 0.402
  (block 1 / 2 / 3 / 4 alone) → 0.796 (all four); shares **41 / 28 / 18 / 11%** of the group effect,
  monotone in depth, summing to 98%, so no single block carries it. **(b) Not plausibility:** the
  partial ρ(`w`, `max_p` | separation) survives every ablation (−0.634 unmodified, reproducing
  Experiment 5's −0.587; −0.45 to −0.64 ablated), so plausibility still predicts *which* pairs are
  sharp — but it does not mediate the intervention (ρ(Δ`w`, Δ`max_p`) ≤ +0.22, median |Δ`max_p`| ≤
  0.0007 against Δ`w` = +0.433) and where it moves it moves the wrong way (median `max_p` 0.0034 →
  0.0136, the direction predicting *narrower* plateaus). **(c) Not the decision:** 80.7% of pairs still
  predict different characters at their endpoints with all four MLPs deleted (86.7% unmodified), median
  `argmax` regions unchanged at 3, yet `d(t)` is straight; `|t* − t_flip|` decouples 0.043 → 0.214.
  **Interpretation change:** the "decodes to the same prediction" clause is demoted from mechanism to
  description in both hypothesis paragraphs, which now end with the PLAN 5.5 falsifiable prediction
  (freeze blocks 1–4 at step-0 weights, train the rest to matched validation accuracy, expect straight
  paths). REPORT Limitation 6 rewritten (it still pointed at the rebalancing test as future work).
  New Figure 22 in both deliverables; exploratory figures renumbered 23–25.
  `check_render.py` ran in full for the first time (node present): **ALL CHECKS PASS**.

## Next step

**No `STOP` written this iteration** (deliberate, same reason as last): feedback #4 says "do not extend
training *yet*", so a follow-up ask is anticipated and a STOP'd direction would silently ignore it
(CLAUDE.md rule 11). Otherwise: plan complete, feedback #4 addressed, and both PLAN-named follow-ups
(denser Figure-9 grid, readout rebalancing) done. Every remaining item needs new compute: a longer run
whose second descent is separated from initial fit; the same densification applied to the *pilot* run's
local maximum (still one point); interpolation positions other than the final token; a second model.
That chain of cheap re-analyses is exhausted and the training run it pointed at (S10) has now run and
falsified its own prediction. Three candidate mechanisms have been excluded in turn - the
next-character decision (survives the ablation that destroys the plateau), endpoint plausibility (does
not mediate the intervention and moves the wrong way), and the specific weights of blocks 1-4 (freezing
them at init still yields plateaus, relocated to blocks 5-7). What those blocks actually *compute* to
produce the sharp change remains uncharacterised.

The direct successor is the new falsifiable prediction the hypothesis now ends on: **freeze blocks 1-7
at their step-0 weights and train only the top of the stack** (`train_frozen.py --freeze 1,2,3,4,5,6,7`,
same harness, ~46 min). If the sharpening simply moves to whatever blocks remain trainable, the paths
should still sharpen well below 0.80 with the width drop appearing between injection blocks 8 and 11;
if it needs several trainable blocks below the readout, the paths should stay straight. **That run was
pre-launched at 19:55 on 2026-08-02** (tag `frozen_deep`, log `/tmp/dir13_frozen/train_frozen_deep.log`,
checkpoints `/tmp/dir13_frozen/checkpoints_frozen_deep/`, ~46 min); nothing from it is in the
deliverables. Next iteration: check the log, then add `frozen_deep` to `frozen_assay.py`'s condition
list and run the depth control at injection blocks 8/10/11 as well as 0/4/8. Everything else
still open needs new compute or a new model: a longer character run whose second descent separates from
initial fit; the denser Figure-9 grid applied to the *pilot* run's local maximum; interpolation
positions other than the final token; a second model.

## Primary references

- Matthew Shinkle and StefanHex, *Activation Plateaus: Where and How They Emerge*: https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge
- Matthew's exact configuration and code: https://github.com/MShinkle/activation_plateau_mechanisms
- Humayun, Balestriero, and Baraniuk, *Deep Networks Always Grok and Here is Why*: https://arxiv.org/abs/2402.15555
- Grokking-paper repository: https://github.com/AhmedImtiazPrio/grok-adversarial