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

- [x] **S11 - Deep-freeze training test (the relocation prediction).** Reopened 2026-08-02. S10 showed
      the sharpening merely *relocated* from the frozen blocks 1-4 to blocks 5-8, so the hypothesis's
      new prediction is that freezing blocks 1-7 relocates it again into the only trainable blocks
      left (8-11). `experiments/train_frozen.py --freeze 1,2,3,4,5,6,7 --tag frozen_deep` retrains the
      reference recipe with 58% of the parameters held at step-0 weights; `experiments/frozen_assay.py`
      adds `frozen_deep` to its condition list and extends the injection-depth control to blocks
      0/4/8/10/11, since the prediction is specifically about where the width drop appears *above*
      block 8. -> verify: the run reaches the reference run's final validation accuracy (0.550) before
      being assayed, and the prediction ("paths still sharpen well below 0.80, with the width drop
      between injection blocks 8 and 11") is reported as confirmed or falsified with the depth profile
      that decides it. **DONE 2026-08-02: verified (0.5742 final val acc, the highest of any run;
      matched at step 3000) and the prediction is CONFIRMED** - median `w` 0.558 (IQR 0.471-0.621,
      149/150 pairs narrower than untrained), depth profile 0.558/0.557/0.695/0.767/0.805 at injection
      blocks 0/4/8/10/11, so the frozen blocks 1-4 contribute -0.002 and all 0.248 of sharpening sits
      in the four trainable blocks 8-11. Cross-run: the width cost tracks the *number* of frozen
      blocks (0.351 -> 0.471/0.484 -> 0.558), not their depth. Figure 23 in both deliverables.

- [x] **S12 - Mirror-image freeze (count vs depth).** Reopened 2026-08-02, the prediction S11's
      hypothesis paragraphs now end on. `train_frozen.py --freeze 5,6,7,8,9,10,11 --tag frozen_mirror`
      freezes the same *number* of blocks as `frozen_deep` but leaves the trainable capacity at the
      *bottom* of the stack. `frozen_assay.py` already lists the condition and its injection-depth grid
      is extended to blocks 0/2/4/8/10/11 (block 2 added so the drop can be resolved *inside* the
      trainable group). -> verify: the run reaches the reference run's final validation accuracy
      (0.550) before being assayed, and the prediction ("width near 0.558 again, with the entire drop
      between injection blocks 0 and 4 and nothing above") is reported as confirmed or falsified with
      the depth profile that decides it. **DONE 2026-08-02 (same iteration as S11): SPLIT verdict.**
      Verified (0.5744 final val acc, the highest of any run and within 0.0002 of frozen_deep's;
      matched at step 2750). The *location* half is CONFIRMED exactly - depth profile
      0.626/0.764/0.805/0.806/0.806/0.806 at injection blocks 0/2/4/8/10/11, so injecting at block 4
      already gives the untrained straight line and all sharpening is in blocks 1-4. The *magnitude*
      half is FALSIFIED - median `w` 0.626, not ~0.558; paired dw vs frozen_deep +0.063 (81% of pairs,
      p = 6e-17). Revised reading, now in both deliverables: trainable depth is the first-order term
      (0.351 -> 0.47 -> 0.56-0.63 for 12/8/5 trainable blocks) and position a second-order one that
      only bites once depth is scarce (worth 0.015 at eight trainable blocks, 0.068 at five, favouring
      the readout end).

- [x] **S13 - Two-block freeze (is trainable depth really the first-order term?).** The prediction the
      hypothesis paragraphs ended on. `train_frozen.py --freeze 1,2,3,4,5,6,7,8,9,10 --tag
      frozen_two` trains only blocks 0 and 11; `frozen_assay.py` gained the condition entry (the
      injection grid 0/2/4/8/10/11 already resolves both trainable blocks). -> verify: the run reaches
      the reference run's final validation accuracy (0.550) before being assayed, and the prediction
      ("straighter still, near 0.70, with the residual drop split between injection blocks 0->2 and
      10->11" versus the alternative "one trainable block beside the readout suffices, so near 0.56")
      is reported as confirmed or falsified with the depth profile that decides it. ~21 min.
      **DONE 2026-08-02: verified (0.5668 final val acc, above the reference; matched at step 7000, the
      slowest of any run) and the trainable-depth prediction is CONFIRMED** - median `w` **0.726**
      (IQR 0.642-0.802), against ~0.70 predicted and ~0.56 for the rejected one-block alternative;
      paired dw +0.160 vs frozen_deep (97% of pairs, p=7e-26) and +0.094 vs frozen_mirror (89%,
      p=3e-21). Depth profile 0.726/0.725/0.724/0.725/0.725/0.803 at injection blocks 0/2/4/8/10/11:
      the *entire* 0.077 of sharpening is produced by block 11 alone, because injecting at block 0
      overwrites block 0's output, so block 11 is the only trainable block the measurement can see (the
      predicted 0->2 half of the drop was impossible by construction - noted as a methods point). Only
      17% of the reference sharpening recovered, and this is the first run where the plateau *breaks*
      rather than blunts: 26% of pairs wider than untrained (0-1% elsewhere), |t*-t_flip| 0.146 vs
      0.043, partial rho -0.18 vs -0.63, strict_frac 0. Cross-run trainable-depth series is monotone:
      0.351 (12) -> 0.471/0.484 (8) -> 0.558/0.626 (5) -> 0.726 (1 usable). Figure 23 in both
      deliverables. New helper `experiments/frozen_pairwise.py` computes the between-run paired
      Wilcoxon shifts from `frozen_assay_raw.npz`.

- [x] **S14 - Width control (depth vs parameter count).** The prediction the hypothesis paragraphs now
      end on, and the confound S13 leaves open: frozen_two froze 82.9% of the parameters *and* left one
      usable block, so it cannot separate trainable depth from capacity. Retrain the reference recipe at
      full depth but half width (`n_embd` 192 instead of 384, nothing frozen), which removes a
      comparable parameter fraction while keeping all 12 blocks trainable. -> verify: the run reaches
      the reference run's final validation accuracy before being assayed, and the prediction ("depth is
      what matters, so ~0.35 like the reference" versus "parameter count matters, so ~0.47 like the
      eight-trainable-block runs") is reported as confirmed or falsified with the depth profile.
      **DONE 2026-08-02:** confirmed - median `w` 0.397 at matched accuracy (0.437 for the second
      seed), the depth account's range, against the capacity account's ~0.47.

- [x] **S15 - Second seed at frozen-early (error bar on the depth step).** Reopened 2026-08-03 after a
      pod reset wiped the `/tmp` scratch checkpoints. S14c gave the 12-trainable-block end of the depth
      comparison two seeds; the 8-block end still had one, so the load-bearing 0.397-vs-0.476 step could
      not be read against the seed spread. `train_frozen.py --freeze 1,2,3,4 --seed 2024 --tag
      frozen_early_s2` retrains frozen-early from a fresh initialization, everything else identical, and
      `narrow_assay.py frozen_early_s2` scores it on the same 150 pairs. -> verify: the replicate reaches
      the reference's final validation accuracy before being assayed, and the report states the paired
      per-seed shift plus a rank test over the six matched-accuracy runs.
      **DONE 2026-08-03: verified** (matched at step 2750, val 0.5529, the same step as seed 1337).
      Median `w` **0.498** against seed 1337's 0.476, per-pair widths indistinguishable (paired `dw`
      +0.001, p = 0.40); relocation signature reproduced (0.498/0.498/0.501 at injection blocks 0/2/4).
      The 12- and 8-trainable-block groups are now **disjoint** across three runs each (0.397-0.443 vs
      0.476-0.500, one-sided rank-sum p = 0.05, the floor for 3-vs-3). Figure 24 in both deliverables.
      Side fixes forced by the scratch wipe: `allpairs_sweep.load_vocab` rebuilds the 65-character
      vocabulary from the SHA-verified corpus when the pilot checkpoint is gone, and `plot_capacity.py`
      reads the narrow run's parameter count from `results/train_meta_narrow192.json` instead of a
      checkpoint - which exposed a counting error (the old value summed the state_dict, double-counting
      the tied embedding/unembedding weight: 5,584,896 -> **5,375,808**, so the narrow run has 4.0%
      *fewer* trainable parameters than frozen-early, not 0.3% more).

- [x] **S16 - Second seed at frozen-deep (error bar on the position term).** Reopened 2026-08-03, the
      step the previous `Next step` named. After S15 the only sub-claim still resting on a single pair of
      runs is the *position* term: five trainable blocks beside the readout (frozen-deep, `w` 0.558) come
      out sharper than five at the bottom of the stack (frozen-mirror, 0.626), a 0.068 gap only ~1.7x the
      seed spread S15 measured. `train_frozen.py --freeze 1,2,3,4,5,6,7 --seed 2024 --tag frozen_deep_s2`
      retrains frozen-deep from a fresh initialization, everything else identical; `narrow_assay.py
      frozen_deep_s2` scores both its checkpoints on the same 150 pairs; `frozen_pairwise.py` adds the
      paired shifts and a position-contrast summary. **Prediction on record, fixed before the run was
      scored:** the replicate lands within the measured seed spread (<=0.04) of 0.558 and therefore below
      frozen-mirror's 0.626; a replicate at or above 0.626, or shifted by more than 0.04, falsifies the
      position term. -> verify: the run reaches the reference's final validation accuracy (0.550) before
      being assayed, and the report states whether BOTH frozen-deep seeds sit below frozen-mirror,
      with the paired per-pair shifts that decide it. **DONE 2026-08-03: verified (matched at step 3000,
      val 0.5503 - the same step as seed 1337; final val 0.5730) and the prediction is CONFIRMED at both
      framings** - median `w` **0.559** at matched accuracy against seed 1337's 0.590 and **0.579** at
      step 30000 against 0.558 (spreads 0.031 / 0.021, sign-inconsistent: paired -0.016 then +0.023).
      Both seeds sit below frozen-mirror on both framings; worst-seed gaps 0.039 / 0.046; paired vs
      frozen-mirror -0.060 (21% wider, p = 5.9e-14) and -0.040 (29%, p = 3.4e-8). Relocation signature
      reproduced (injection blocks 0/2/4 -> 0.559/0.558/0.557 and 0.579/0.578/0.577). Remaining
      weakness, stated in Limitation 7: frozen-mirror is still one run, so the position gap has a seed
      spread under one side only. Figure 24 in both deliverables.

- [x] **S23 - Seed replication of the last two single-seed load-bearing runs.** DONE 2026-08-10, both
      pre-registered predictions HELD. `train_frozen.py --freeze 0,1,2,3,4,5,11 --tag frozen_high_s2
      --seed 2024` (blocks 6-10 trainable) and `--freeze 5,6,7,8,9,10,11 --tag frozen_mirror_s2 --seed
      2024` (blocks 0-4 trainable), trained concurrently to step 30,000 and scored by `narrow_assay.py`
      on the same 150 pairs. **Blocks 6-10:** 0.344 at matched accuracy (step 3,750, val 0.5530) vs
      seed 1337's 0.342 - a 0.002 spread, the smallest in the study - and 0.335 vs 0.328 at step 30,000;
      vs the untouched reference -0.071 (18.0% of pairs wider, p = 1.9e-16) and -0.021 (37.3%,
      p = 7.5e-4). **Blocks 0-4:** 0.624 at matched accuracy (step 2,750, val 0.5504) vs 0.629, and
      0.590 vs 0.626 at step 30,000; all four deep-vs-mirror seed pairings keep the near-readout window
      sharper (+0.031, +0.053 at matched accuracy; +0.038, +0.022 at step 30,000). Honest qualification
      now in both deliverables: the closest median pairing of that ordering is 0.033 (matched) and 0.010
      (final), at or inside the 0.040 largest seed spread, so it rests on the paired per-pair tests.
      New **Figure 25** (`plots/seed_replication.png`) in both deliverables; Figure 24 re-rendered with
      the two extra seed markers and leader-line labels.

- [x] **S24d - What the residual half of the responsible units are: bigram-conditioned corpus tuning.**
      DONE 2026-08-12, closing the successor S24c named. `experiments/neuron_bigram.py` tabulates each
      block-1-4 unit's mean post-GeLU activation against the (previous, current) character pair over
      the same 90% training split, then (a) decomposes each unit's table into current-character,
      previous-character and interaction shares, (b) linearizes 8 found vs 8 missed recruits at matched
      size, and (c) hands the selection to a context-matched profile (previous character = space, the
      assay's own context) and re-scores every k=32 rule on the 84 like-for-like pairs. 19 s, no
      training. **Result: the missed units are context-dependent, and the fix is refuted.** Found
      recruits are 96% current-character, missed ones 51% (interaction 18% -> 49%, p = 1.4e-186,
      population median 37%); 8 missed recruits remove 11.5% of the width gap against 29.1% for 8 found
      (p = 1.2e-20); the context-matched rule ranks better (AUROC 0.886 vs 0.869, p = 1.4e-5) and
      selects worse (21.9% vs 31.9% at k=32, p = 1.9e-11; ceiling 52.6%, random 0.6%), foretold by
      precision@32 (20.3% vs 25.6%). Figure 32 (`plots/neuron_bigram.png`) in both deliverables.

- [x] **S24b - What the early MLPs compute: chord linearization of individual units.** DONE 2026-08-12,
      the first answer to S24 item 1 (open since 2026-08-03). `experiments/neuron_path.py` replaces a
      chosen set of block-1-4 MLP hidden units' post-GeLU activations along the path by the chord
      between their own endpoint values, deleting only each unit's curvature in `t` and leaving both
      endpoints exact (worst deviation 1e-6 over every pair and condition). Same 150 pairs, block-0
      interpolation, step-30,000 `ref_pos` checkpoint as the gain/per-block interventions; three
      selection rules (per-pair top-k, one global top-k, random k) at 13 sizes; 198 s, no training.
      **Result: sparse per path, pooled across paths.** All 3,840 units -> `w` 0.743 (86.7% of the
      trained->untrained gap; MLP deletion reaches 0.796). Per-pair top-32 (0.83% of units) -> 50.9%;
      per-pair median k for half its own gap = 64 (IQR 32-128); random 32 -> 1.2% and random needs
      ~2,048 units to match 32 ranked ones. One global set of 32 -> only 19.0%, median overlap of a
      pair's top-32 with it 9/32; 668 units ever enter a top-32 and 82% of those serve >=2 pairs.
      Unpredicted: top-32 slots skew deeper (16/19/28/37% in blocks 1/2/3/4) against single-block
      deletion's front-loaded 41/28/18/11%. Figure 29 (`plots/neuron_path.png`) in both deliverables.

- [x] **S24a - Readout offset: interpolate a character the readout does not read.** DONE 2026-08-12,
      all four pre-registered predictions HELD. The reference character run was retrained
      (`train_frozen.py --tag ref_pos`, nothing frozen, seed 1337, 30,000 steps, 29.2 min) because
      `/tmp` had been wiped, then `experiments/pos_assay.py` scored step-0, matched (step 2,500,
      val 0.5522) and step-30,000 checkpoints on the same 150 pairs, with `k` = 0/1/2/4/8 filler
      characters between the patched character (position 14) and the readout. Injection at the residual
      stream entering block 0, the only exact site off the final position; anchor rows measured the
      standard way tie the sweep to the rest of the report and reproduce it (0.803 / 0.4428 / 0.3507 vs
      the reference's 0.803 / 0.443 / 0.351). **Step 30,000:** median `w` 0.243 / 0.290 / 0.249 / 0.244
      / 0.257, offsets 2/4/8 paired-indistinguishable from `k`=0 (p = 0.27, 0.43, 0.22); untrained
      0.804-0.809 at every offset (0/150 plateaus, trained-vs-init p = 2.3e-26); `read_patch` identical
      throughout (0.2427, worst endpoint error 1.9e-5). **The finding:** endpoint-decision disagreement
      falls 86.7% -> 8.7% at `k`=4 while 52.0% of pairs still meet the strict plateau rule, so the
      switch outlives the next-character decision that described it. Distance-independence is built
      late (matched accuracy still degrades: 0.328 -> 0.434 at `k`=4, p = 5.6e-20). New **Figure 28**
      (`plots/pos_offset.png`) and a REPORT.md Methods subsection in both deliverables; verdict and two
      caveats rewritten; exploratory figures renumbered 28-30 -> 29-31.

- [x] **S20 - The interior/end split's own test: a five-block window at blocks 1-5.** DONE 2026-08-03; the prediction below was FALSIFIED (it landed at 0.363, with the sharp group) and the split was withdrawn from both deliverables.
      `train_frozen.py --freeze 0,6,7,8,9,10,11 --tag frozen_mid_low` freezes the same *seven* blocks
      (58.0%) and leaves five trainable, one block *down* from S19's sharp 2-6 window, so the usable
      window touches block 1 - the first block after the patched activation. **Prediction on record
      (also in both hypothesis paragraphs and the Conclusion of the deliverables):** the interior/end
      split says it lands with the blunt group, **above 0.47**; landing near 0.365 would make the split
      a coincidence of the eight runs that generated it. This is the cheapest decisive test of the only
      surviving descriptive account. -> verify: reaches val accuracy 0.550 before assay, and the report
      states which group it joins with paired shifts against frozen-mid-off (2-6) and frozen-mirror.

- [x] **S19 - A five-block window one step off-centre.** DONE 2026-08-03. **PREDICTION FALSIFIED, and
      the falsification killed a description rather than a result.** `--freeze 0,1,7,8,9,10,11`
      (trainable 2-6, five frozen blocks below the window instead of three) was predicted to cost
      something and land at 0.40-0.45. It costs nothing: `w` = **0.365** at matched accuracy (step
      3500, val 0.5507) and **0.355** at step 30000 (val 0.5744, the highest of any run), identical to
      frozen-mid (paired +0.014, p = 0.064; +0.007, p = 0.229) and level with the fully trained
      reference (-0.009, p = 0.286). Against the end windows: -0.197 / -0.172 / -0.225 (matched) and
      -0.200 / -0.264 (final), all p <= 3.3e-25. Strict rate 21.3%. So the "distribution of frozen
      blocks around the window" description is withdrawn. What replaced it: across all eight frozen
      runs, a trainable window strictly interior to the stack (usable blocks 4-8, 2-6, 5-7) gives
      0.365 / 0.365 / 0.446 and one touching either end (5-11, 1-7, 8-11, 1-4, block 11) gives
      0.476 / 0.500 / 0.590 / 0.629 / 0.712 - disjoint, but found post-hoc, with the three-block run's
      0.030 margin inside the seed spread. S20 above is its pre-registered test.

- [x] **S18 - How few mid-stack blocks suffice? (three trainable blocks in the middle).** DONE
      2026-08-03. **PREDICTION CONFIRMED.** `w` = **0.446** (IQR 0.344-0.559) at matched accuracy (step
      7000, val 0.5518) and **0.427** at step 30000 (val 0.5711), inside the predicted 0.40-0.50 band.
      Statistically indistinguishable from the full 12-block reference (0.443; paired +0.009, 55% of
      pairs wider, p = 0.17) and clear of every five-block end window: **-0.121** vs frozen-deep seed 1
      (9.3% wider, p = 7.2e-23), **-0.090** vs seed 2 (11%, p = 1.4e-21), **-0.154** vs frozen-mirror
      (4.7%, p = 1.3e-25); at step 30000 -0.111 and -0.184. Strict rate 9.3% / 10.0% against the
      reference's 12.7% / 10.0%. Tightest relocation in the series (0.363 of 0.380 between injection
      blocks 4 and 8). Cost of shrinking the window from five mid-stack blocks to three: +0.086 of
      width (85% wider, p = 3.0e-17) and 7000 steps to matched accuracy rather than 3750. So window
      size matters but position dominates it. Both deliverables curated; Figures 23 and 24 redrawn.

- [x] **S18-launch - How few mid-stack blocks suffice? (three trainable blocks in the middle).** Launched
      2026-08-03 immediately after S17's result, which falsified the ordered position reading:
      `train_frozen.py --freeze 0,1,2,3,4,8,9,10,11 --tag frozen_mid3` freezes NINE blocks (74.2% of
      them) and leaves only blocks 5-7 trainable - a three-block window at the same mid-stack site that
      S17 showed recovers the whole plateau with five. It discriminates the two readings S17 leaves
      open. **Prediction on record, fixed before the run was scored:** if mid-stack *position* is what
      matters and the trainable-block count is secondary, three mid-stack blocks still beat five blocks
      at either end, landing below frozen-deep's 0.559-0.590 and frozen-mirror's 0.626 - expected
      `w` between frozen-mid's 0.365 and frozen-deep's 0.558, i.e. **~0.40-0.50**. If instead the count
      reasserts itself once the window shrinks below five, it lands **at or above 0.558**, which
      falsifies "position dominates count" and restores a count-plus-position reading.
      -> verify: the run reaches the reference's final validation accuracy (0.550) before being
      assayed, and the report states which side of 0.558 it falls on with the paired per-pair shifts
      against frozen-mid, frozen-deep and frozen-mirror.

- [x] **S17 - Middle-of-stack five-block freeze (the position term as an ordered three-point claim).**
      DONE 2026-08-03. **PREDICTION FALSIFIED, and the falsification is the result.** `w` = **0.365**
      at matched accuracy (step 3750, val 0.5519) and **0.331** at step 30000 (val 0.5728) - not
      between the two known five-block values but far *below* both, and below the full 12-block
      reference at its own matched step (0.443; paired -0.056, 24.7% of pairs wider, p = 2.7e-14).
      Paired against the other two five-block runs: **-0.211** vs frozen-deep seed 1 (1.3% of pairs
      wider, p = 3.3e-26), **-0.188** vs frozen-deep seed 2 (0%, p = 2.3e-26), **-0.240** vs
      frozen-mirror (0.7%, p = 2.3e-26). Strict plateau rate **24.7%** - the highest of any model in
      the series, more than double the reference's 10%. Relocation signature holds (injection blocks
      0/2/4/8 -> 0.365/0.382/0.506/0.812: the frozen blocks contribute 0.017, all sharpening in the
      trainable window 4-8). Consequence for the deliverables: the trainable-block **count** is not the
      first-order term - the three five-block runs span 0.365-0.629 - and position is not a monotone
      gradient toward the readout but has an interior optimum. Both deliverables re-framed (rule 9b).
      Launched 2026-08-03, the successor S16's hypothesis paragraphs end on. `train_frozen.py --freeze
      0,1,2,3,9,10,11 --tag frozen_mid` freezes the same *seven* blocks (58.0% of parameters) as
      frozen-deep and frozen-mirror but leaves the trainable five in the middle (blocks 4-8), the only
      one of the three five-block positions not yet run. Freezing block 0 costs the measurement nothing,
      since injecting at block 0 overwrites block 0's output anyway, so all five trainable blocks are
      downstream of the injection - the same as frozen-deep and unlike frozen-two. **Prediction on
      record, fixed before the run was scored:** if position is a genuine second-order term favouring
      the readout end, `w` lands between the two known five-block values, near 0.58-0.60; landing at or
      below frozen-deep's 0.559-0.590, or at or above frozen-mirror's 0.626, falsifies the ordered
      reading. -> verify: the run reaches the reference's final validation accuracy (0.550) before being
      assayed, and the report states where it falls in the three-point ordering with the paired
      per-pair shifts against both neighbours.

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

**S25 DONE 2026-08-12 — second seed: the structural developmental facts reproduce, the promoted-unit
describability gain does not.** `experiments/train_frozen.py --tag ref_pos_s2 --seed 2024 --freeze ""`
(23.0 min, final val acc 0.5511 vs the reference run's 0.5502) plus `experiments/neuron_seed2.py`
(66 s: importance ranking at five checkpoints, then the character-window probe refitted at each, reusing
`neuron_path.record_pair` and `neuron_probe_early.fit_probe` with `CKPT_DIR` pointed at the new run).
Reproduces: sharpening (median `w` 0.641 → 0.338 vs 0.653 → 0.351), head turnover (top-8 overlap with
the final head 1, 2, 4, 6, 8 vs 0, 2, 4, 6, 8; top-32 7, 12, 15, 22, 32 vs 6, 10, 16, 23, 32), promotion
from just below the head (median rank 126 at step 831 vs 113.5, chance 1,919.5; 50.2% inside the top 128
vs 51.8%; displaced leaders settle at 98 vs 100.5), the whole-network describability decline
(0.40 → 0.17 vs 0.39 → 0.15), the demoted fall (−0.27 vs −0.25), the stable ceiling (0.96–0.99), and both
final-checkpoint band contrasts (stable vs promoted 0.98/0.47, CLES 0.77; demoted vs never-head
0.81/0.22, CLES 0.78). Does NOT reproduce: the promoted units' absolute gain (per-unit median −0.02 vs
+0.05 from the current character; still a relative rise, 1.5× → 2.7× the background), and the
forward-looking step-831 contrast one band below the head collapses to null (0.69 vs 0.68, CLES 0.53,
p = 0.4, against 0.80 vs 0.64, p = 7e-4). Curated as **Figure 42** in both deliverables (exploratory
figures renumbered 43–45), with the Summary/Headline, the "why this matters" consequence and the caveats
narrowed to the backward-looking statement. `check_render.py` → ALL CHECKS PASS (45 figures per file).

**Operator feedback #7 addressed 2026-08-12 (pending review) — the claim is narrowed from decision
basins to character-conditioned basins.** No new experiment: the counts the operator cites were already
measured (`results/basin_decision.json`, Figure 16 — 65 endpoint characters → 15 distinct endpoint
predictions, 31.6% of paths at exactly two, median 3, 9.9% of pairs sharing an endpoint prediction).
The Summary verdict and the RESULTS.md Headline already reflected them; the **Conclusion** did not, and
still read "a plateau here is a next-character decision basin". It now reads "a character-conditioned
basin in logit space whose transition coincides with a change in the model's next-character
prediction", with the counts inline and an explicit refusal of one-basin-per-character. Also narrowed:
REPORT.md's Interpretation paragraph, Limitation 7 (re-titled "The next-character decision neither
labels the basins nor explains them"), RESULTS.md Question & verdict item 5, Figure 21's caption in both
files, and the hypothesis paragraph's "decodes to the same prediction" description. Every measurement
is unchanged. `check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS (39 figures each, 0 problems).

**S24j DONE 2026-08-12 — describability travels with head membership, not with the units.**
`experiments/neuron_head_describe.py` (no GPU, reads `results/neuron_head_identity_raw.npz`,
`neuron_bands_raw.npz`, `neuron_probe_raw.npz`) labels every unit promoted / demoted / stable /
never-head from the step-831 and step-30,000 top-8 sets and compares probe $R^2$. Unconditionally the
promoted units look like the describable ones (median $R^2$ from the current character 0.72 vs 0.41,
CLES 0.58, p = 0.013), but that contrast is confounded — promoted units have best rank ≤ 7 by
construction. Conditioning on best rank reverses it: inside the head, stable 0.97 vs promoted 0.72
(CLES 0.69, p = 3e-6); one band down, demoted 0.94 vs never-head 0.23 (CLES 0.79, p = 9e-14).
`experiments/neuron_probe_early.py` (~40 s per checkpoint on GPU, five checkpoints) then refits the
whole probe at every checkpoint and settles the reading: the median unit is **more** describable early
(all units 0.39 → 0.28 → 0.24 → 0.17 → 0.15), yet promoted units gain against that background
(0.61 → 0.64 → 0.67 → 0.73 → 0.72, per-unit median +0.05) while demoted units fall faster than it
(0.92 → 0.83 → 0.60 → 0.41 → 0.41, −0.25, most of it between steps 2,038 and 12,500) and stable units
sit at the ceiling throughout (0.96–0.98). Forward visibility at step 831 is
weak: future keepers vs future losers inside the early head 0.96 vs 0.92 (CLES 0.59, p = 0.035); future
promotions vs never-head one band down 0.80 vs 0.64 (CLES 0.63, p = 7e-4). Pipeline check: the
step-30,000 refit reproduces the published per-unit $R^2$ to 3e-14. Curated as **Figure 40** and
**Figure 41** in both deliverables; exploratory figures renumbered 42–44.

**S24i DONE 2026-08-12 — the final head units are promoted from just below the head, not recruited
from the anonymous middle.** `experiments/neuron_head_origin.py` (14 s, one recording pass per pair per
checkpoint, no ablations) reads the importance *rank* of each pair's step-30,000 top-8 units at every
earlier checkpoint and the mirror trajectory of its step-831 top-8. Median rank of the eventual head:
**113.5 → 31 → 7 → 4 → 3.5** of 3,840 across steps 831 → 30,000 (a random unit sits at 1,919.5), with
**51.8%** already inside the step-831 top 128 and **75.5%** inside its top 512; **75%** of the climb is
complete by step 2,038, while membership of the top 8 itself is the last thing to arrive (9.8 / 23.3 /
51.6 / 72.0 / 100%). The displaced step-831 leaders fall only to median rank **100.5**. This resolves
the ambiguity left by S24h's zero overlap: training re-orders a broad candidate pool that is already
recognizable early. Curated as **Figure 39** in both deliverables.

**S24h CURATED into both deliverables 2026-08-12 (later iteration).** REPORT.md gained a Methods
subsection ("Band decomposition: why the last third of the bend costs hundreds of units" — bands,
marginal-in-prefix, redundancy ratio $\Lambda$, the two random controls, best-rank unit assignment, the
checkpoint series) and a Results subsection ("The tail is weak, redundant and continuous, and training
keeps rewriting the head") carrying the three arms; RESULTS.md gained the matching self-contained
section with its methods inline. Both files: new **Figure 37** (`plots/neuron_bands.png`) and
**Figure 38** (`plots/neuron_bands_time.png`), the three exploratory figures renumbered 37–39 → 39–41,
and a Summary/Headline paragraph. `check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(41 figures each, 0 problems).

**S24h DONE 2026-08-12 — the tail is redundant, weak and continuous, not a second population.**
`experiments/neuron_bands.py` cuts the per-pair importance ranking into six bands
(0–8, 8–32, 32–128, 128–512, 512–2,048, 2,048–3,840) and linearizes each band **on its own**, plus the
nested prefix ending at each band edge, plus two size-matched random controls (drawn from all 3,840
units, and drawn only from the units ranked at or below the band's own lower edge). Same 150 pairs /
context / block-0 interpolation / step-30,000 checkpoint; 63 s of forward passes, no training.
Three findings. **(1) Not independent contributions — redundant ones.** Band-alone recovered
fractions are 27.9 / 25.9 / 24.1 / 20.4 / 10.4 / −0.1%, which sum to **111.5%** against the
all-units ceiling of **85.2%** (ratio 1.29, paired $p=6\times10^{-20}$, 86.7% of pairs above), so
bands overlap in what they bend rather than adding up. Each band alone also beats its own marginal
contribution inside the nested prefix (27.9 vs 27.9, 25.9 vs 20.0, 24.1 vs 17.9, 20.4 vs 16.5, 10.4
vs 2.7%), which is the same statement pair by pair. **(2) The ranking carries real signal to rank
2,048 and none below.** Every band except the last beats a random set of the same size drawn from its
own region — 0.15 vs 25.9%, 1.19 vs 24.1%, 2.24 vs 20.4%, 3.90 vs 10.4%, all $p\le10^{-25}$, 98–100%
of pairs — while the 2,048–3,840 band bends nothing at all (−0.1%, indistinguishable from its
control, which for that band *is* the band). The per-unit worth falls ~500-fold across the ranking
(34.8 → 0.067% per 1,000 units). The naive size-matched random control (drawn from all units)
recovers 40.6 / 48.2% for the two large bands purely because a random 1,536-unit set contains ~40% of
the top-32 units; that control answers a budget question only and is reported as such.
**(3) A continuum, not a second kind of unit.** Assigning each unit to the band of its *best* rank
over the 150 pairs and reading off `neuron_probe.py`'s fitted text description gives medians that
decline smoothly with rank — held-out $R^2$ of the full description 0.97 / 0.70 / 0.66 / 0.59 / 0.52 /
0.50, of the current-character-only description 0.91 / 0.30 / 0.22 / 0.14 / 0.12 / 0.13 — with no
break separating a tail population. Head units (best rank < 32, $n=668$) are far more describable
than tail units (best rank ≥ 512, $n=1{,}623$): $R^2$ 0.80 vs 0.51 ($p=2\times10^{-67}$) full, 0.42
vs 0.12 ($p=2\times10^{-97}$) current-character-only, Mann–Whitney over distinct units. This is an
association between rank and describability measured at one checkpoint, not a mechanism.
Raw → `results/neuron_bands_raw.npz`, stats → `results/neuron_bands_summary.json`, figure →
`plots/neuron_bands.png` (three panels: band-alone vs marginal vs control; per-unit worth on a log
axis; describability by band).

**S24h, second arm — the redundancy is not built by training; what training builds is the head.**
`experiments/neuron_bands_time.py` repeats the band-alone and all-units measurements at five
checkpoints of the same run (steps 831 / 2,038 / 5,000 / 12,500 / 30,000; 96 s of forward passes).
Early checkpoints have fewer pairs with a usable trained→untrained gap (94 / 140 / 146 / 148 / 150),
so the trend is read on the **94 pairs usable at every checkpoint**. The redundancy ratio is
**1.21 / 1.01 / 1.08 / 1.21 / 1.18** — present as soon as there is any bend to share and not growing
steadily, so the prediction that redundancy accumulates with training is **not supported**. What does
grow is how much of the bend those units carry at all: the all-units effect goes **46.2 → 46.7 → 61.9
→ 76.4 → 81.0%**, and the gain is concentrated at the top of the ranking — the eight highest-ranked
units alone go **7.2 → 23.9%** while the 1,792 lowest-ranked stay at 0 throughout. Raw →
`results/neuron_bands_time_raw.npz`, stats → `results/neuron_bands_time_summary.json`, figure →
`plots/neuron_bands_time.png` panels (a)–(c). One run, five checkpoints: this describes that run's
development, and is not evidence that training causes the redundancy.

**S24h, third arm — the head is re-selected, not amplified.** If training strengthens the head band,
the obvious reading is that the same units grow. `experiments/neuron_head_identity.py` (14 s, one
recording pass per pair per checkpoint, no ablations) refutes it: the median overlap between a
checkpoint's per-pair **top-8** set and the step-30,000 top-8 set is **0 / 2 / 4 / 6 / 8** units at
steps 831 / 2,038 / 5,000 / 12,500 / 30,000, and for the **top-32** set **6 / 10 / 16 / 23 / 32**
(chance overlap 0.02 and 0.27 units). So at step 831 — where the top eight units already remove 7.2%
of the gap — not one of them is a unit the finished network will rank in its top eight. The set also
keeps churning late: consecutive-checkpoint top-8 overlap is only 5 of 8 between steps 5,000 and
12,500 and 6 of 8 between 12,500 and 30,000. Stats →
`results/neuron_head_identity_summary.json`, raw → `results/neuron_head_identity_raw.npz`, figure →
`plots/neuron_bands_time.png` panel (d).

**Curation deferred by one iteration on purpose:** REPORT.md and
RESULTS.md are the declared outputs of feedback #7 and are awaiting the independent content review,
and inserting figures into the neuron-selection section renumbers every later figure in both files.
The next iteration embeds `plots/neuron_bands.png` and `plots/neuron_bands_time.png` as two new
figures in the S24 section of both deliverables and adds the Methods paragraph defining the band
decomposition, the two controls and the best-rank unit assignment.

**S24g DONE 2026-08-12 — unit interactions are measured and small: worth 3.4 points at $k=128$ and
nothing at $k=32$.** Zero unaddressed feedback files, so this iteration advanced the plan (PLAN's own
"Next step" named this experiment). One script, no training, 77 s of forward passes.
`experiments/neuron_greedy.py` makes selection *sequential*: build the linearized set in $R$ equal
rounds and, before each round, re-measure every unit's importance
$I^{S_r}_j=n_j\max_t\lvert a^{S_r}_j(t)-\mathrm{chord}_j(t)\rvert$ with the units already chosen
linearized, so a unit is scored by the bend still left. $R\in\lbrace 1,2,4,8\rbrace$,
$k\in\lbrace 32,128\rbrace$, same 150 pairs / context / step-30,000 checkpoint. $R=1$ is exactly the
one-shot pair-fitted rule, so it is both the control and a free reproduction check.
**P1 (greedy $R=4$ at $k=32$ gains ≥5 points) REFUTED:** 50.9% → 51.3% → **49.8%** → 49.8%, paired
$p=0.24$, $0.41$, $0.43$; only 50.7% of pairs not worse. **P3 (greedy clears the best per-unit rule's
56.6% at $k=32$) REFUTED** by the same numbers. **P2 (monotone in $R$) holds at $k=128$ only, and
there it is strong:** 68.4% → 70.7% → 71.1% → **71.8%**, paired $p=9.4\times10^{-17}$,
$1.5\times10^{-19}$, $6.1\times10^{-21}$, with $R{=}4\rightarrow8$ itself significant
($p=6.0\times10^{-6}$), 84.7% of pairs not worse, median width gain $+0.0145$. Eight rounds keeps a
median **100 of 128** picks (**26 of 32** at $k=32$), so the gain is a reallocation of a fifth of the
set, not a different circuit. Reading: joint effects close **3.4 of the 18.3 points** between the best
one-shot ranking and the 86.7% all-units ceiling, and only in the interchangeable tail — the leading
units carry the bend nearly independently. This **corrects S24f's attribution**: the blind rule beats
the fitted ranking at $k=32$ because of the per-unit score's form (endpoint displacement vs path
curvature), not because the fitted ranking misses joint structure. Free checks all exact: $R=1$
reproduces the stored pair-fitted widths (max difference 0.000000 at both $k$), baseline 0.3507 (max
difference 0.000000), worst endpoint deviation $10^{-6}$. New `experiments/neuron_greedy.py`,
`experiments/plot_neuron_greedy.py`, `results/neuron_greedy_{summary.json,raw.npz,log}`,
**Figure 35** (`plots/neuron_greedy.png`); exploratory Figures 35–37 renumbered 36–38, 38 embeds /
38 captions / sequential 1–38 in both files; `check_render.py` passes with 0 problems.

**S24f DONE 2026-08-12 — the text-only selection score is at the ceiling of its family, and two
pre-registered predictions were refuted.** Zero unaddressed feedback files, so this iteration advanced
the plan (PLAN's own "Next step" named this consolidation). One script, no training, 62 s of forward
passes. `experiments/neuron_scale.py` runs five more blind rules through the identical chord
intervention on the same 150 pairs at $k\in\lbrace 8,32,128\rbrace$: the character profile and the
fitted probe each multiplied by the write norm $n_j=\lVert W_{\mathrm{proj}}[:,j]\rVert_2$; the
*measured* endpoint swing $E_j=|a_j(1)-a_j(0)|$ (an oracle for what the corpus rules estimate) with and
without $n_j$; and $n_j$ alone as a pair-blind floor. **P1 (write norm gains ≥2 points) REFUTED — it
hurts:** character rule **56.3% → 55.4%** (paired $p=0.049$), probe 56.5% → 56.6% in median but worse
on 62% of pairs ($p=2.7\times10^{-4}$), oracle **56.6% → 55.3%** ($p=1.1\times10^{-9}$); write norms
span only **1.71×** between the 5th and 95th percentiles (median 1.66, IQR 1.49–1.82), so $n_j$ carries
almost no information. **P2 (the oracle beats the corpus rules) REFUTED on its first half — it ties
them:** 56.6% vs the probe's 56.5%, paired $p=0.27$, so at $k=32$ these rules are not
estimation-limited and the remaining distance to the fitted ranking at $k=128$ (62.9% vs 68.4%) is the
score's *form* — all of them rank by individual displacement and cannot see joint effects. P2's second
half held: weighted endpoint displacement beats the pair-fitted curvature ranking, **55.3%** vs
**50.9%** ($p=2.2\times10^{-17}$), sharing a median 20 of 32 picks. **P3 held more strongly than
predicted:** the write norm alone removes **0.3%** at $k=32$, below random's 1.2%, and 12.0% at
$k=128$ — which units write hardest says nothing about which bend a path. Free checks: baseline
reproduces per pair exactly (0.3507, max difference 0.000000), worst endpoint deviation $10^{-6}$.
New `experiments/neuron_scale.py`, `experiments/plot_neuron_scale.py`,
`results/neuron_scale_{summary.json,raw.npz,log}`, **Figure 34** (`plots/neuron_scale.png`);
exploratory Figures 34–36 renumbered 35–37, 37 embeds / 37 captions / sequential 1–37 in both files;
`check_render.py` passes with 0 problems.

**S24e DONE 2026-08-12 — the missed units are described from held-out corpus data, and the selection
limit is re-attributed from conditioning to scale.** Zero unaddressed feedback files, so this iteration
advanced the plan (PLAN's own "Next step" named this experiment: a *learned* description in place of
another hand-built conditioning). Two scripts, no training, 44 s of forward passes.
`experiments/neuron_probe.py` fits a ridge regression per block-1–4 unit predicting its post-GeLU
activation from the eight characters ending at the position plus a previous×current interaction table
(4,746 features, 3,840 units), on the model's own training split, windows split 80/10/10 so $\lambda$
is chosen on one held-out slice and every $R^2$ reported on another. **Descriptive:** the recruits the
character rule misses reach median held-out $R^2$ **0.29 → 0.53 → 0.78** for window lengths
1 → 8 → 8+interaction, against **0.92 → 0.97** for the ones it finds ($p=1.5\times10^{-116}$;
$p=8.3\times10^{-185}$ for the context gain) — "context-dependent" is literal and local, not diffuse.
**Causal:** scoring units by the probe's predicted activation difference at the assay's own context and
linearizing the top 32 removes **56.5%** of the width gap, against 28.9% (standardized character rule),
22.5% (bigram), 19.0% (global) and 1.2% (random), all paired $p\le2.3\times10^{-26}$ — and past the
pair-fitted top-32's **50.9%** ($p=2.3\times10^{-17}$), so that ranking was never a ceiling (it orders
by individual importance $I_j$). **The unexpected correction:** `experiments/neuron_probe_control.py`
decomposes the win — the character profile with per-unit standardization *removed* reaches **56.3%**
alone, the fitted context alone **34.8%**; scale is worth ~27 points and context ~6 (together 0.2 more
than scale alone, paired $p=0.0022$). S24d's noise diagnosis stands but is second-order. Free checks:
baseline reproduces per pair to 0.3507 (max difference 0.000000), worst endpoint deviation $10^{-6}$.
New `experiments/neuron_probe.py`, `experiments/neuron_probe_control.py`,
`experiments/plot_neuron_probe.py`, `results/neuron_probe_{summary.json,raw.npz,control.json,control_raw.npz}`,
**Figure 33** (`plots/neuron_probe.png`); exploratory Figures 33–35 renumbered 34–36, 36 embeds /
36 captions / sequential 1–36 in both files; `check_render.py` passes with 0 problems.

**S24d DONE 2026-08-12 — the residual half of the mechanism is characterised, and the obvious fix for
it is refuted.** Zero unaddressed feedback files, so this iteration advanced the plan (PLAN's own
"Next step" named this experiment). One script, no training, 19 s of forward passes.
`experiments/neuron_bigram.py` re-runs the corpus pass tabulating each block-1–4 unit's mean post-GeLU
activation against the (previous, current) character pair, which supports two tests at once.
**Descriptive:** a weighted two-way decomposition of each unit's bigram table shows the recruited units
the character ranking *finds* (top decile of $D_j$) are near-pure character detectors — median **96%**
of their corpus response explained by the current character alone — while the ones it *misses* sit at
**51%**, interaction share rising 18% → **49%** ($p=1.4\times10^{-186}$), against a population median
of 37%. They are not ranking noise: 8 missed recruits remove **11.5%** of the width gap against
**29.1%** for 8 found recruits at matched set size (paired $p=1.2\times10^{-20}$, 138 pairs), where 8
random units remove about 1%. **Causal, and negative:** the context-matched profile (previous character
= space, i.e. the assay's own context) *ranks* the population better — mean AUROC **0.886** vs **0.869**
on the same 84 pairs ($p=1.4\times10^{-5}$) — and *selects* worse: its top 32 remove **21.9%** of the
gap against **31.9%**, paired $p=1.9\times10^{-11}$, with the fitted ceiling at 52.6% and random at
0.6% on those pairs. Precision@32 foretells it (20.3% vs 25.6%): the bigram split estimates each cell
from ~14× fewer positions, and that noise bites hardest at the top of the ranking, which is the only
part a 32-unit intervention reads. Free checks: marginalizing the bigram table reproduces $z$ to
**0.0000**, baseline reproduces per pair to 0.3507, worst endpoint deviation $10^{-6}$. New
`experiments/neuron_bigram.py`, `experiments/plot_neuron_bigram.py`,
`results/neuron_bigram_summary.json`, `results/neuron_bigram_raw.npz`, **Figure 32**
(`plots/neuron_bigram.png`); exploratory Figures 32–34 renumbered 33–35, 35 embeds / 35 captions /
sequential 1–35 in both files; `check_render.py` passes with 0 problems.

**S24c DONE 2026-08-12 — the direction's named open problem is now answered on both halves: how many
units bend a path, and what those units detect.** Zero unaddressed feedback files, so this iteration
advanced the plan (PLAN's own "Next step" named this experiment). Two scripts, no training, 31 s of
forward passes total. `experiments/neuron_feature.py` characterises the units from a data source the
assay never touches: the model's own 90% training split, tiled into 7,842 non-overlapping 128-character
windows (941,040 scored positions), accumulating each block-1–4 hidden unit's mean post-GeLU activation
per current character and standardizing that 65-character profile within the unit ($z_{c,j}$).
**Corpus tuning predicts recruitment:** ranking units by differential tuning $|z_a-z_b|$ finds a pair's
recorded top-32 at mean **AUROC 0.847** (99% CI 0.834–0.858), precision@32 **21.6%** = **26×** chance,
against **0.562** for an overall-activity control and 0.498 for a shuffle ($p=2.3\times10^{-26}$); the
assay-derived global ranking, which has seen the experiment, reaches 0.913. Recruitment falls 4.9% →
0.09% monotonically across tuning deciles; a recruited unit's preferred character is one of the pair's
endpoints for **27.2%** of recruitments vs a 2.8% base rate (**9.8×**); the 668 pool units are the
sharply tuned population (median $\max_c|z_c|$ 5.45 vs 4.47, $p=5.8\times10^{-27}$). The most reused
unit (block 2, 88/150 pairs) is a capital-letter detector whose top corpus contexts are proper-name
onsets (`DUCHESS OF Y`, `Duke of Y`, `Henry the F`). Robustness: re-standardizing over the 62
characters with ≥100 occurrences and keeping the 143 pairs built from them gives 0.858. Then
`experiments/neuron_feature_causal.py` makes it causal and held out: linearizing the 32 units the
corpus rule selects — blind to $d(t)$, $I_j$ and the pair's curve — removes **28.9%** of the
trained→untrained width gap (0.351 → 0.482, 98% of pairs widen) against **1.2%** for 32 random units
and **19.0%** for the assay-derived global set ($p=2.7\times10^{-11}$), below the fitted per-pair
ceiling of 50.9% ($p=7.3\times10^{-26}$). Baseline reproduces per pair exactly (0.3507) and endpoints
stay exact ($10^{-6}$). Both deliverables gained two Results sections, a REPORT.md Methods subsection
(tuning profile, $z_{c,j}$, sharpness, differential/max scores, AUROC, precision@32, three baselines,
held-out selection rule) and **Figures 30–31** (`plots/neuron_feature.png`,
`plots/neuron_feature_causal.png`); the five places that named "what those units detect" as open now
carry the result with its real residual. Exploratory figures renumbered 30–32 → 32–34;
`check_render.py REPORT.md RESULTS.md` passes with 0 problems (34 embeds / 34 captions / sequential
1–34 in each). New: `results/neuron_feature_{summary.json,raw.npz,causal.json,causal_raw.npz}` and
their logs.

**S24b DONE 2026-08-12 — the direction's named open problem (S24 item 1, "what the trainable blocks
compute") has its first quantitative answer, and it needed no training.** Zero unaddressed feedback
files, so this iteration advanced the plan. `experiments/neuron_path.py` linearizes individual
block-1–4 MLP units along the interpolation path — each chosen unit's post-GeLU activation is replaced
by the chord between its own endpoint values, so the unit keeps its endpoint behaviour and loses only
its curvature in $t$, and both endpoints stay exact for any chosen set (worst deviation $10^{-6}$ over
every pair and condition). Same 150 pairs, block-0 interpolation and step-30,000 `ref_pos` checkpoint
as the gain and per-block interventions, so widths compare directly (unmodified 0.351, untrained
0.803); 198 s of forward passes. **The bend is the nonlinear-in-$t$ part of these MLPs:** linearizing
all 3,840 units gives median $w$ **0.743**, i.e. **86.7%** of the trained→untrained gap, against 0.796
for deleting the MLPs outright. **It is sparse per path:** a pair's own top-32 units (0.83% of them)
recover **50.9%**, the per-pair median for half of its own gap is **64** units (IQR 32–128), while 32
*random* units recover **1.2%** and random selection needs ~2,048 units to match 32 ranked ones.
**It is not a reusable circuit:** one fixed global set of 32 recovers **19.0%**, a typical pair shares
only **9 of its 32** units with it, and although 668 of 3,840 units ever enter a top-32 (82% of them
serving ≥2 pairs, the most reused 88/150), the subset that bends a given path is pair-dependent.
Unpredicted detail, reported rather than smoothed: the carrying units skew *deeper* inside the group
(16.0 / 18.8 / 27.8 / 37.4% of top-32 slots in blocks 1/2/3/4) while single-block MLP deletion is
front-loaded (41/28/18/11%) — consistent, because deleting block 1's MLP also changes what blocks 2–4
receive. Both deliverables gained a Results section, a REPORT.md Methods subsection (chord
substitution, importance score $I_j$, recovered fraction $\rho(S)$) and **Figure 29**
(`plots/neuron_path.png`); the four "still uncharacterised" clauses in the hypothesis, Conclusion,
Interpretation and Limitation 7 are narrowed to what the counts support, with *what those units detect*
named as the open part. Exploratory figures renumbered 29–31 → 30–32; `check_render.py REPORT.md
RESULTS.md` passes with 0 problems. New: `experiments/neuron_path.py`,
`experiments/plot_neuron_path.py`, `results/neuron_path_{summary.json,raw.npz}`, `results/neuron_path.log`.

**S24a DONE 2026-08-12 — the readout was moved off the patched character; all four pre-registered
predictions held, and the report's central description narrowed.** Zero unaddressed feedback files, so
this iteration advanced the plan. The previous iteration had left a *partial* `results/pos_assay.json`
(matched-accuracy checkpoint only) unreported on disk; `/tmp` had been wiped, so the reference
character run was retrained from scratch (`train_frozen.py --tag ref_pos`, nothing frozen, seed 1337,
30,000 steps, 29.2 min, val acc 0.5502, matched at step 2,500) and `pos_assay.py` was run over all
three checkpoints. Results at step 30,000: median $w$ = **0.243 / 0.290 / 0.249 / 0.244 / 0.257** for
$k$ = 0/1/2/4/8, with offsets 2, 4, 8 paired-indistinguishable from $k=0$ ($p$ = 0.27, 0.43, 0.22);
untrained is the straight line at every offset (0.804–0.809, 0/150 plateaus, trained-vs-init
$p=2.3\times10^{-26}$); the `read_patch` identity check is exact (0.2427 everywhere, worst endpoint
error 1.9e-5). The load-bearing number: endpoint-decision disagreement collapses 86.7% → **8.7%** at
$k=4$ while **52.0%** of pairs still meet the strict plateau rule, so the sharp switch outlives the
next-character decision that has been describing it. Unpredicted extra: the distance-independence is
built late — at matched accuracy the widths still degrade with offset (0.328 → 0.434 at $k=4$,
$p=5.6\times10^{-20}$). Free reproduction check: the fresh run's anchor rows give 0.803 / 0.4428 /
0.3507, matching the reference run quoted throughout the deliverables (0.803 / 0.443 / 0.351). Both
deliverables now carry Figure 28 (`plots/pos_offset.png`), a REPORT.md Methods subsection for the
offset assay, a rewritten verdict ("at the patched position they look like decision basins; the switch
survives where the decision does not"), and two corrected caveats — "final-position interpolation only"
is no longer true. Exploratory figures renumbered 28–30 → 29–31; `check_render.py REPORT.md RESULTS.md`
passes with 0 problems. New: `results/pos_assay.json`, `pos_assay_raw.npz`, `pos_assay.log`,
`train_ref_pos.log`.

**F6 DONE 2026-08-12 — the basin criterion was redefined and validated; `STOP` is NOT written.**
`human_feedback_6.txt` (now `.addressed.md`) reported that the old basin fraction classified the
linear null $d(t)=t$ as owning a basin, because it required $t_{lo}\ge0.10$ / $t_{hi}\le0.90$ and the
line hits those values exactly. Confirmed and fixed at the source: the basin test is now a **rest
ratio** $R=r(\delta)/\delta$, identically 1 for the line at every $\delta$, with a basin claimed at
$R\ge\kappa=2$, $\delta=0.10$. Validated against four null families through the identical code path —
exact line, line + Gaussian noise ($\sigma$ 0.01/0.02/0.05), the untrained network's 2,080 curves, the
200-pair block-11 patch — all at a **0.0%** false-positive rate (0/2, 0/12,000, 0/4,160, 0/400
endpoints) against **90.3%** for the trained network (median rest ratio 3.18 vs 0.94–1.00 for the
nulls). Headline narrowed from "every character owns a basin (65/65, min $\phi$ 0.86)" to **59/65
characters hold a basin against a majority of partners** (median $\phi$ 1.00, mean 0.90); the six
failures — `3`, `&`, `$`, `Z`, `X`, `z` — are the six rarest characters, and $\phi$ tracks training
frequency at Spearman $\rho=0.56$ ($p=1.0\times10^{-6}$, n = 65). Transition-width, strict-count,
variance-decomposition, intervention and frozen-run numbers are all unaffected (the strict rule also
demands $w\le0.25$, which the line fails at 0.80). New: `experiments/basin_criterion.py`,
`experiments/plot_basin_criterion.py`, `results/basin_criterion.json`, Figures 15 and 17
(`plots/basin_criterion.png`, `plots/basin_vs_frequency.png`); Figures 16–28 renumbered to 18–30.
`check_render.py REPORT.md RESULTS.md` passes with 0 problems. Zero unaddressed feedback files remain,
but the S24 candidates in "Next step" are open, so the loop continues.

**S23 DONE 2026-08-10 — the two remaining single-seed runs were replicated and both pre-registered
predictions held.** Zero unaddressed feedback files (all six `human_feedback*` end in `.addressed.md`),
so this iteration advanced the plan. `frozen_high_s2` (blocks 6-10 trainable) and `frozen_mirror_s2`
(blocks 0-4 trainable) were trained concurrently from model seed 2024 to step 30,000 with every other
setting identical to their seed-1337 originals, then scored on the same 150 pairs at their own matched
accuracy checkpoint and at step 30,000. Blocks 6-10 repeats at **0.344 / 0.335** against **0.342 /
0.328** (spread 0.002 and 0.007, the smallest measured anywhere here), so the study's sharpest network
- 58.0% of its parameters never moved from initialization, and sharper than the untouched 12-block
reference - is not an initialization artefact. Blocks 0-4 repeats at **0.624 / 0.590** against **0.629
/ 0.626**, and all four deep-versus-mirror seed pairings keep five trainable blocks beside the readout
sharper than five at the bottom, at both checkpoints. The one thing the replicate changes is the size
of that last ordering: its closest median pairing is 0.033 at matched accuracy and 0.010 at the end of
training, at or inside the 0.040 largest seed spread, so both deliverables now say it rests on the
paired per-pair tests rather than on the median gap. Five conditions now carry two seeds; the spread
over all nine condition-by-checkpoint pairs is 0.002-0.040 with no consistent sign. New **Figure 25**
(`plots/seed_replication.png`) in RESULTS.md and REPORT.md; Figure 24 re-rendered with the two extra
seed markers and with the crowded five-block labels moved onto leader lines.
`check_render.py REPORT.md RESULTS.md REPORT_followup.md` → ALL CHECKS PASS. No `STOP` written.

**RE-OPENED 2026-08-10 — operator feedback #5 arrived, `STOP` is gone, and the feedback is now
addressed.** `human_feedback_5.txt` asked for four analyses to be written into a **new** companion
deliverable `REPORT_followup.md` (frequency-filtered pairwise matrix + trends; example `d(t)` curves
for well-trained endpoints as a visual asymmetry check; the prompt context and per-cell sample count
stated on the character-level figures; and one well-trained letter against the others checked for
semantic grouping). All four are done, the file is written and render-verified, and the feedback file
is renamed `human_feedback_5.txt.addressed.md`. Headline new numbers: median width **0.320** over the
1,378 well-trained pairs vs **0.482** over the 702 pairs touching a character seen < 1000 times
(p = 4.0e-159); width vs log training frequency Spearman **ρ = −0.78** (n = 65), **−0.66** within the
53 kept; partner-class ordering concordant across 43 letters (Kendall **W = 0.42**, and 0.27 with the
frequency confound removed). No training and no new forward pass — the scratch checkpoints have been
wiped, which is also why the both-directions re-run was answered from the stored exact-symmetry
control instead. `check_render.py REPORT.md RESULTS.md REPORT_followup.md` → ALL CHECKS PASS. No
`STOP` written: the plan's own next step (S23) is still open.

**FINALIZED 2026-08-03 — time budget exhausted; `STOP` was written (and has since been removed).** Zero unaddressed feedback files (all
five `human_feedback*` end in `.addressed.md`). Both deliverables were re-read and verified as
current-best at S22's state: `check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS** (REPORT 29
display / 581 inline equations / 27 embedded figures with 27 visible captions; RESULTS 27 figures / 27
captions; 0 problems; no bare `(plots/*.png)` path). No experiment was left half-written — nothing was
running at close and every row in `results/` is already reflected in the deliverables. The two named
seed replications (S23) were **not** run: each needs ~21 min of training plus ~70 s of assay and only
~14 min of wall clock remained, so starting one would have left an unusable partial checkpoint. They
stand in both deliverables as an explicit limitation, not as a promise.

**S22 DONE 2026-08-03 — the pending step-30,000 row for blocks 6–10 arrived and strengthens
the claim it was pending on.** No new training: S21's chained assay finished (trainer step 30,000,
val_acc 0.5720) and gives median `w` **0.328** (IQR 0.252–0.395, strict 24.0%) against the untouched
reference's 0.351 at the *same* step — paired **−0.037**, 36.7% of pairs wider. The "training fewer
blocks can sharpen the plateau" fact therefore holds on two axes, not just at matched accuracy, and
the run kept sharpening past matched accuracy (0.342 → 0.328), so the matched-accuracy comparison is
the conservative one. Figure 23's "assay still running" scope note is retired from both deliverables.
**Next: S23 — a second seed at blocks 6–10 (the more load-bearing of the two named replications), then
a second seed at frozen-mirror.**

**S21 DONE 2026-08-03 — the pre-registered coverage test was run and REFUTED it, and the run
is the sharpest network in the study.** Five trainable blocks at 6–10 (freeze 0–5 and 11) excluded
mid-stack block 5, so coverage required **>= 0.55**. It lands at **0.342** (IQR 0.240–0.446) at matched
accuracy (step 3,750, val 0.5523) — the lowest matched width of the fifteen models here, sharper than
the two mid-stack windows it was meant to lose to (p = 0.025, 9e-4) and 0.14–0.25 sharper than every end
window — with the highest strict plateau rate measured, 28.0%. Coverage is **withdrawn**. Two post-hoc
geometric descriptions have now each died on the first experiment aimed at them, so both deliverables
state the honest reading: **ten runs support no geometric summary**, and what stands are two rule-free
two-network facts — blocks 1–5 are 0.118 sharper than blocks 0–7 which contain them, and blocks 6–10
alone (58.0% of parameters frozen at init) are **0.072 sharper than the untouched 12-block reference**
at matched accuracy (p = 8.5e-18). Its step-30,000 assay was still training at iteration close; the
chained `narrow_assay.py frozen_high` writes it for the next iteration.

**S20 DONE 2026-08-03 — the pre-registered interior/end test was run and REFUTED the split.**
Five trainable blocks at 1–5 (freeze block 0 and 6–11) had to land above 0.47; they land at **0.363**
(matched) / **0.326** (step 30,000), with the mid-stack group. Withdrawn from both deliverables and
replaced by a rule-free claim — blocks 1–5 are a strict subset of frozen-late's 0–7 and are 0.118
sharper (p = 2.2e-25), so removing trainable blocks sharpens the plateau — plus a labelled, untested
coverage description (windows over block 5: 0.363–0.500; without it: 0.559–0.712) whose own test is
S21, pre-registered below.

**S17 DONE 2026-08-03 — its prediction was FALSIFIED, and that falsification re-framed the
frozen-block conclusion (rule 9b).** Five trainable blocks in the *middle* of the stack (freeze 0–3 and
9–11) were predicted to land between the two known five-block values at 0.58–0.60. They land at
**0.365** at matched accuracy and **0.331** at step 30,000 — below both siblings, below every
eight-block run, and below the untouched 12-block reference — with a strict plateau rate of **24.7%**
against the reference's 10.0%. So the *count* of trainable blocks is not the first-order term (the
three five-block runs span 0.365–0.629, wider than the whole 12-to-5 series) and position is not a
gradient toward the readout but has an interior optimum. Both deliverables now say that where the
trainable blocks sit governs the cost, described by how the seven frozen blocks are distributed around
the trainable window (mid splits them 3/3, deep stacks seven before, mirror seven after), explicitly
labelled a description of three points. **S18 then tested that description in the same iteration and
CONFIRMED it:** shrinking the mid-stack window from five trainable blocks to three (freeze 0-4 and
8-11, 74.6% of parameters frozen) still gives **0.446**, statistically indistinguishable from the full
12-block reference (p = 0.17) and 0.09-0.18 clear of every five-block window at either end. Window size
costs something (+0.086 from five mid-stack blocks to three) but position dominates it. **S19 then
falsified the description S17 had offered:** sliding the five-block window one step off centre (blocks
2-6, five frozen blocks below it instead of three) costs nothing at all - 0.365 again, identical to
frozen-mid - so "how the frozen blocks are distributed around the window" is withdrawn. What replaced
it, stated with its caveats in both deliverables: across the eight frozen runs, a trainable window
strictly interior to the stack gives 0.365-0.446 and one touching either end gives 0.476-0.712, with no
overlap; found post-hoc, mechanism not established, and the three-block run's 0.030 margin is inside
the seed spread. Its pre-registered test is S20. All three runs curated into both deliverables;
Figures 23 and 24 regenerated (Figure 23's depth panel split into two small multiples, since nine
series exceeded the five-hue palette);
`check_render.py REPORT.md RESULTS.md` returns ALL CHECKS PASS (REPORT 29 display / 558 inline
equations / 27 figures; RESULTS 27 figures; 0 problems), 27 embeds = 27 visible captions per file.
Nothing is left uncurated on disk; the next iteration starts from S20.

**S16 DONE 2026-08-03 — the position term has an error bar, and the last single-pair
sub-claim is replicated.** A second seed of frozen-deep (blocks 1–7 frozen, seed 2024) reaches the
reference's accuracy at step 3000 and gives median `w` **0.559** at matched accuracy / **0.579** at step
30000 against seed 1337's 0.590 / 0.558 — both seeds below frozen-mirror's 0.626 / 0.626 by more than
the 0.02–0.03 seed spread, so "five trainable blocks beside the readout beat five at the bottom" is
CONFIRMED rather than a seed artefact. Two stale sentences were also fixed: both hypothesis paragraphs
still ended on the narrow-run prediction S14 answered and misstated the reference width as `n_embd` 384
(it is 240). `python3 experiments/check_render.py REPORT.md RESULTS.md` returns ALL CHECKS PASS (REPORT
29 display / 472 inline equations / 27 figures; RESULTS 27 figures; 0 problems); 27 `![…]` embeds match
27 visible `**Figure N.**` captions in each file; all 27 referenced PNGs exist; zero bare
`(plots/x.png)` paths. All five `human_feedback*` files end in `.addressed.md`; zero unaddressed
feedback remains. **No `STOP` is written while wall clock and a named next experiment remain.**

**PLAN COMPLETE (S1-S15) + operator feedback #4 (2026-08-02 file) addressed + twelve PLAN-named
follow-ups DONE (denser Figure-9 grid, readout rebalancing, MLP-gain intervention, per-block scan,
frozen-block training test, deep-freeze training test, mirror-image freeze, two-block freeze, narrow
run, narrow run scored at end of training, second narrow seed, second frozen-early seed).**

- **S15 second frozen-early seed DONE (2026-08-03, latest) - the depth step now separates cleanly with
  two seeds a side.** `--freeze 1,2,3,4 --seed 2024` reaches the reference's accuracy at step 2750 (val
  0.5529, the same step as seed 1337) and completes the full 30000 steps (val 0.5629 vs 0.5625). Median
  `w` **0.498** at matched accuracy against seed 1337's 0.476 - per-pair widths indistinguishable
  (paired +0.001, half the pairs each way, p = 0.40), deciles matching within 0.02 - and **0.445** at
  step 30000 against 0.471, i.e. 0.027 the *other* way (paired -0.030, p = 3.3e-5), so seed noise is
  <=0.04 with no consistent sign. Relocation signature reproduced at both checkpoints (injection blocks
  0/2/4 give 0.498/0.498/0.501 and 0.445/0.444/0.443 - the frozen group contributes nothing). Across
  runs, the three with 12 trainable blocks (0.443, 0.397, 0.437) are **disjoint** from the three with 8
  (0.476, 0.498, 0.500): one-sided rank-sum p = 0.05, the floor for 3-vs-3. All four
  narrow-vs-frozen-early seed combinations agree pair by pair (-0.073, -0.067, -0.044, -0.063, each
  p <= 2.7e-8). CORRECTED this iteration: the narrow run's parameter count was summing the state_dict,
  which double-counts the tied embedding/unembedding weight - 5,584,896 -> **5,375,808**, so the narrow
  run has 4.0% *fewer* trainable parameters than frozen-early rather than 0.3% more (this strengthens
  the depth conclusion: the sharper run is the one with less capacity).

- **S14c second narrow seed DONE (2026-08-02, latest) - the depth conclusion now has an across-seed
  error bar, and one sub-claim is retracted.** A second `n_embd` 192 run (model seed 2024, nothing
  frozen) reaches the reference's accuracy at the same step 2750 (val 0.5547) and gives median `w`
  **0.437** against seed 1337's 0.397, i.e. across-seed spread ~0.04 (paired +0.015, p = 0.015) - about
  half the 0.08-0.10 gap to the frozen runs. Both seeds stay below frozen-early (-0.044, p = 2.7e-8)
  and frozen-late (-0.062, p = 1.6e-16), two-seed mean 0.417 vs capacity's ~0.47. RETRACTED: the narrow
  run is NOT sharper than the full-width reference at matched accuracy (seed 2: -0.004, p = 0.17) -
  narrowing costs nothing measurable, it does not help. Only a matched-accuracy row exists for this
  seed (training stopped at that checkpoint for wall clock).

- **S14 narrow run COMPLETE (2026-08-02, latest) - the depth/capacity confound is BROKEN and the depth
  account wins.** `--n_embd 192` with nothing frozen: all 12 blocks trainable but only 5,584,896
  parameters, within 0.3% of frozen_early's 5,601,360 trainable parameters. At matched accuracy (step
  2750, val 0.5543) median `w` **0.397** (IQR 0.311-0.526) - the depth account's ~0.35-0.44, not the
  capacity account's ~0.47. Paired: **-0.073** vs frozen_early (23% of pairs wider, p=2.5e-15),
  **-0.092** vs frozen_late (13%, p=1.8e-19), **-0.014** vs the reference at its own matched step (39%,
  p=1.9e-4, i.e. slightly SHARPER). Depth profile 0.397/0.569/0.686/0.763/0.807/0.832 at injection
  blocks 0/2/4/8/10/11 (the reference's front-loaded shape); partial rho -0.65; strict_frac **0.133**,
  the only run besides the reference to keep the sharpest tail (frozen runs 0-0.007). New Figure 24.

- **S14b fully-trained row DONE (2026-08-02, latest) - both framings agree.** The narrow run finished
  and `narrow_assay.py` scored `ckpt_last.pt`: median `w` **0.332** (IQR 0.288-0.389) at step 27,143,
  val 0.5639. Paired: **-0.010** vs ref_trained's 0.351 (43% of pairs wider, p=2.1e-4, i.e. marginally
  sharper than the full-width reference), **-0.124** vs frozen_early_last's 0.471 (1.3%, p=2.6e-26),
  **-0.146** vs frozen_late_last's 0.484 (3.3%, p=3.6e-26), **-0.065** vs its own matched row
  (23%, p=3.1e-14). Front-loaded depth profile 0.332/0.626/0.746/0.794/0.802/0.808; strict_frac
  **0.120** (reference 0.100); partial rho -0.51. Caveat carried with the number: the harness time
  budget stopped it at 27,143 of 30,000 steps (lr 1.2e-4 not 1.0e-4), which can only understate its
  final sharpness since it was still sharpening. Figure 24 re-rendered to show BOTH framings per run
  (large marker = matched accuracy, small open square = end of training). Also fixed three stale
  REPORT claims the new row exposed: Limitation 7 and the Experiment-5 closing caveat both still said
  the narrower-but-full-depth run "was not performed", and the Summary never carried the
  depth-vs-capacity result at all.

- **S13 two-block freeze COMPLETE (2026-08-02, latest) - the trainable-depth prediction CONFIRMED, and
  the first run in which the plateau actually breaks.** `--freeze 1,...,10`: 82.9% of the parameters
  held at step-0 weights, only blocks 0 and 11 trainable. Final val acc **0.5668** (still above the
  reference's 0.5502) but matched at step **7000**, 2.3-2.8x slower than every other frozen run. Median
  `w` **0.726** (IQR 0.642-0.802) against the ~0.70 the trainable-depth account predicted and the ~0.56
  the "one block beside the readout suffices" alternative predicted - paired dw **+0.160** vs
  frozen_deep (97% of pairs, p=7e-26) and **+0.094** vs frozen_mirror (89%, p=3e-21). Depth profile
  0.726/0.725/0.724/0.725/0.725/**0.803** at injection blocks 0/2/4/8/10/11: the whole 0.077 of
  sharpening comes from block 11 alone. The predicted 0->2 half of the drop could not occur - injecting
  at block 0 overwrites block 0's output, so block 11 is the only trainable block downstream of the
  measurement. Only **17%** of the reference sharpening recovered, and unlike the other four runs the
  geometry degrades: 26% of pairs wider than untrained (0-1% elsewhere), |t*-t_flip| 0.146 vs 0.043,
  partial rho -0.18 vs -0.63, strict_frac 0. Cross-run series is monotone in trainable depth:
  0.351 (12 blocks) -> 0.471/0.484 (8) -> 0.558/0.626 (5) -> 0.726 (1 usable).

- **S12 mirror-image freeze COMPLETE (2026-08-02, latest) - SPLIT verdict, and it corrects S11's
  headline.** `--freeze 5,6,7,8,9,10,11`: the same 58.0% of parameters frozen and the same five
  trainable blocks as `frozen_deep`, moved from the top of the stack to the bottom. Final val acc
  **0.5744** (highest of any run, within 0.0002 of frozen_deep's), matched at step 2750. The *location*
  half of the prediction is confirmed exactly - depth profile 0.626 / 0.764 / **0.805** / 0.806 / 0.806
  / 0.806 at injection blocks 0/2/4/8/10/11, so injecting at block 4 already yields the untrained
  straight line and all sharpening is back in blocks 1-4 (0.138 in blocks 1-2, 0.042 in 3-4). That is
  the **third distinct site across four runs** (5-8, 8-11, 1-4). The *magnitude* half is falsified:
  median `w` **0.626** (IQR 0.555-0.681), not ~0.558; paired dw vs frozen_deep **+0.063** (81% of pairs,
  p = 6e-17), i.e. 39% vs 54% of the reference sharpening recovered. **S11's "the cost tracks how many
  blocks are frozen, not which" is therefore REPLACED** by a two-term reading: trainable depth is
  first-order (0.351 -> 0.471/0.484 -> 0.558/0.626 for 12/8/5 trainable blocks) and position
  second-order, worth 0.015 at eight trainable blocks and 0.068 at five, favouring the readout end.
  Incidental: adding injection block 2 to the grid shows the reference's sharpening is front-loaded
  into blocks 1-2 (0.351 -> 0.646 -> 0.761), matching the per-block MLP scan's 41/28/18/11% shares.

- **S11 deep-freeze training test COMPLETE (2026-08-02, latest) - the successor prediction is
  CONFIRMED.** With blocks 1-7 (58% of the blocks) held at their step-0 weights, the run reached the
  *highest* validation accuracy of any run here (**0.5742** vs the reference's 0.5502; matched at step
  3000) and the paths still sharpen: median `w` **0.558** (IQR 0.471-0.621), narrower than the
  untrained 0.803 for 149/150 pairs (Wilcoxon p = 2e-26), i.e. 54% of the reference sharpening
  recovered. The depth control decides it: 0.558 / 0.557 / 0.695 / 0.767 / 0.805 at injection blocks
  0/4/8/10/11, so the frozen blocks 1-4 contribute -0.002 and the entire 0.248 of sharpening sits in
  the four trainable blocks (0.139 over blocks 5-8, of which only block 8 can train; 0.071 over 9-10;
  0.039 in block 11) - the predicted signature. **Cross-run finding:** the width cost tracks *how many*
  blocks are frozen, not which - 0.351 (none) -> 0.471/0.484 (four, either end) -> 0.558 (seven), with
  frozen-deep wider than frozen-early by a paired +0.073 (p = 1e-17) and than frozen-late by +0.064
  (p = 1e-16) while the two four-block runs differ by 0.015. Geometry otherwise unchanged (`t*` 0.486,
  endpoints differ 87%, 3 argmax regions, |t*-t_flip| 0.092, partial rho -0.62); strict rate 0%.
  **Correction carried into both deliverables:** `run_pair` patches `resid_post` of the interpolation
  block, so the drop between injection points b1 < b2 comes from blocks b1+1..b2 - S10's "relocated to
  blocks 5-7" is corrected to **blocks 5-8** everywhere (numbers unchanged). Figure 23 re-rendered with
  six top-row panels and five injection depths.

- **S10 frozen-block training test COMPLETE (2026-08-02) - the hypothesis's own prediction is
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

**S25 DONE (2026-08-12) and curated — see "Current status". The second seed that every developmental
claim in this block was waiting for is measured, so the block now separates what belongs to the recipe
(turnover, promotion from a recognizable early pool, readability draining away with lost head membership)
from what belonged to one initialization (the promoted units' absolute describability gain, and any
forward prediction of which units get promoted).

**What this leaves, in order of what a reader asks next.** (i) The depth series still has one seed per
condition, so the 0.397-vs-0.476 gap that carries the depth argument has no across-seed bar; the new run
does not supply one, because it is an unfrozen reference run — a second seed at `n_embd` 192 and at one
frozen condition is still ~23 min of training plus ~70 s of assay each, and `frozen_assay_raw.npz` holds
the per-pair reference results needed to score them. (ii) The band decomposition and the flat redundancy
ratio were measured only in the reference run; re-running `neuron_bands.py` / `neuron_bands_time.py` on
the seed-2024 checkpoints now on disk would extend the second seed to them (ablations, so ~10× the cost
of `neuron_seed2.py`, roughly 10–15 min). (iii) Whether the describability coupling is specific to the
8-character window probe family — a unit that stops being readable from 8 characters may have become
readable from something else, and no probe here tests that. It needs a new probe family and is the
larger piece of work. Note the seed-2 checkpoints live in `/tmp/dir13_frozen/checkpoints_ref_pos_s2`
(1.2 GB, scratch) and vanish with `/tmp`; retraining them is 23 minutes.

**S24j DONE (2026-08-12) and curated — see "Current status". Both halves ran in one iteration: the
final-checkpoint comparison (`neuron_head_describe.py`, no GPU) and the step-831 refit that interprets
it (`neuron_probe_early.py`, 21 s). Figures 40 and 41 are embedded in REPORT.md and RESULTS.md with
their methods, captions, Summary/Headline paragraphs and updated caveats.

**The successor this names, and its cost.** The five-checkpoint curve is measured, so the coupling
between head membership and describability is a trajectory, not a two-point contrast. What remains is
(i) a **second seed**, still the honest generality check for every developmental claim in this block
(~16 min of training plus ~70 s of assay per condition, and it would put an across-seed bar on the
0.397-vs-0.476 depth gap as well); and (ii) whether the coupling is specific to the character-window
probe family — a unit that stops being readable from 8 characters may have become readable from
something else, and no probe in this direction tests that. Item (i) is what an external reader will ask
for first; item (ii) needs a new probe family and is the larger piece of work.

**S24h curation DONE and S24i DONE (2026-08-12, same iteration).** `plots/neuron_bands.png`
(Figure 37), `plots/neuron_bands_time.png` (Figure 38) and the new `plots/neuron_head_origin.png`
(Figure 39) are embedded in REPORT.md and RESULTS.md with their methods, captions and Summary/Headline
paragraphs; the exploratory figures are renumbered 40–42. S24i (`experiments/neuron_head_origin.py`,
14 s, no ablations) answered the question the head-turnover result raised — smooth climb or late
arrival — and the answer is **promotion from just below the head**: a pair's step-30,000 top-8 units sit
at median rank 113.5 of 3,840 at step 831 (random 1,919.5), 51.8% already inside that checkpoint's top
128, climbing 113.5 → 31 → 7 → 4 → 3.5 with 75% of the climb done by step 2,038, while entry into the
top 8 itself comes last (9.8 → 23.3 → 51.6 → 72.0 → 100%). The displaced step-831 leaders drift only to
median rank 100.5, an order of magnitude above chance. So the churn is a re-ordering inside a broad
candidate pool that is recognizable early, consistent with the 668-unit shared pool.

**The next question this leaves, and it is nearly free.** Are the promoted units *describable* in a way
the demoted ones are not? Every ingredient exists on disk: `results/neuron_probe_*` holds each unit's
held-out $R^2$ (full description and current character alone) and
`results/neuron_head_origin_raw.npz` holds both trajectories, so comparing the final-head units against
the step-831 head units they displaced is a Mann–Whitney over distinct units with no GPU work at all.
A positive result would say training promotes the units a short character window predicts; a null would
say the promotion is invisible to that description. Either way it stays one run, one context, and a
second seed remains the honest generality check for every developmental claim in this block.

**S24h DONE (2026-08-12) — see "Current status". The successor the three arms named. Both developmental questions this iteration raised are already
answered: redundancy does not grow with training (ratio ~1.2 → ~1.18), and the head band strengthens
by *replacing* its members (0 of the step-831 top-8 survive to step 30,000). The question that leaves
is where the final head units come from — do they climb the ranking smoothly from the start, or appear
late? The measurement is one recording pass per pair per checkpoint (~15 s, no ablations, checkpoints
on disk): record the FULL importance vector at each checkpoint and report the rank trajectory of each
pair's step-30,000 top-8 units. A smooth climb and a late jump are different developmental stories and
this distinguishes them; either way it stays a description of one run. The exhaustive one-at-a-time greedy at $k=32$ is now *less* attractive than
it looked: with band-alone effects redundant at every scale, a better batched ranking cannot be what
limits $k=32$. S24 item 3 (a longer character run whose second local-complexity descent separates from
initial fit, the denser Figure-9 grid on the pilot run's local maximum, or a second model/tokenizer)
still needs materially more compute than one 30,000-step run.**

**S24g DONE (2026-08-12) — see "Current status". The selection thread is finished. The units are
identified, described from held-out corpus text, selected better by a blind text-only rule than by the
assay-fitted one at $k=32$, shown to be near-uniform writers, and now shown to interact only weakly:
sequential re-selection gains 3.4 points at $k=128$ and nothing at $k=32$, so joint effects close about
a fifth of the distance to the all-units ceiling. What that leaves is a question of a different kind —
not *which* units are chosen but why the last third of the sharpness needs hundreds of them when a few
dozen carry the first half. The instrument for it is a saturation analysis of $\rho(S)$ against $k$
with pair-level resolution (the 13-point grid in `neuron_path.py` already holds most of the data): does
the tail behave like many small independent contributions, or like a second population of units with a
different character profile? Forward passes only. The one experiment that would raise the ceiling on
this section's own result is exhaustive one-at-a-time greedy at $k=32$, scoring each candidate by its
measured effect on $d(t)$ (~$k\times$ the passes of one ranking) — worth running only if the saturation
analysis suggests the batched approximation is what limits the $k=32$ result rather than the geometry.
S24 item 3 (a longer character run whose second local-complexity descent separates from initial fit,
the denser Figure-9 grid on the pilot run's local maximum, or a second model/tokenizer) still needs
materially more compute than one 30,000-step run.**

**S24f DONE (2026-08-12) — see "Current status". The individual-displacement family of selection rules
is closed: the write norm adds nothing (it hurts), and an oracle reading the network's own endpoint
activations ties the text-only probe, so the corpus rules are not estimation-limited. The successor is
named by that result rather than guessed at: the only score still ahead at $k=128$ is the pair-fitted
ranking, and the reason is a *joint* effect no per-unit score can see, so the direct test is a greedy /
residual-corrected selection — choose the next unit by how much of the *remaining* bend it removes
given those already linearized, and compare the resulting $\rho(S)$ curve against the ranked one at
matched $k$. Forward passes only, but roughly $k\times$ the passes of one ranking, so run it at
$k\le 32$ on a subsample of the 150 pairs first. S24 item 3 (a longer character run whose second
local-complexity descent separates from initial fit, the denser Figure-9 grid on the pilot run's local
maximum, or a second model/tokenizer) still needs materially more compute than one 30,000-step run.**

**S24e DONE (2026-08-12) — see "Current status". The "what do the responsible units compute" thread is
closed as far as text statistics reach: the units are identified, described out of sample by a fitted
probe (median held-out $R^2$ 0.78 for the previously unexplained half), and a text-only rule now
selects them better than the assay-fitted ranking at $k=32$. The successor it opens is a consolidation
rather than a new question — the scale insight applies to every earlier selection rule in the report
(the global assay-derived set, the per-block scan), which were all built on preference rather than
predicted displacement, so re-ranking them would test how far the correction reaches. S24 item 3 (a
longer character run whose second local-complexity descent separates from initial fit, the denser
Figure-9 grid on the pilot run's local maximum, or a second model/tokenizer) still needs materially
more compute than one 30,000-step run.**

**S24d DONE (2026-08-12) — see "Current status". Single-character and bigram corpus statistics have now
been taken as far as they go: the missed units are context-dependent, and sharpening the conditioning
makes the *ranking* better and the *selection* worse. The honest successor is a learned description of
the missed units rather than another hand-built conditioning — fit a probe on their corpus activations
and ask what it reads, testing whether "context-dependent" resolves into nameable features or stays
diffuse. It needs forward passes only, no training of the model. S24 item 3 (a longer character run
whose second local-complexity descent separates from initial fit, the denser Figure-9 grid on the pilot
run's local maximum, or a second model/tokenizer) still needs materially more compute than one
30,000-step run.**

**S24c DONE (2026-08-12) — S24 item 1 is answered on both halves; see "Current status". The successor
it opens is what the *other* half of the responsible units respond to: the corpus rule recovers 28.9%
of the width gap where the fitted per-pair ranking recovers 50.9%, and tuning conditioned on the
current character alone cannot describe a unit that responds to a longer pattern. Conditioning the
same corpus pass on the (previous, current) bigram, or on a short suffix, would test that directly and
needs no training. S24 item 3 (a longer character run whose second local-complexity descent separates
from initial fit, or a second model/tokenizer) still needs materially more compute than one
30,000-step run.**

**S24b DONE (2026-08-12) — superseded by S24c above; kept for the record. The successor it named was
*what those units detect*: take the 668 pool units
(`results/neuron_path_raw.npz`, `counts` and `imp_mean`) and find each one's maximally activating
characters/contexts in the SHA-verified corpus, then ask whether a unit's top-activating contexts
predict which pairs recruit it. That needs forward passes only — no training — and it would turn "a few
dozen gated units per path" into a statement about features. S24 item 3 (a longer character run whose
second local-complexity descent separates from initial fit, or a second model/tokenizer) still needs
materially more compute than one 30,000-step run.**

**S24a DONE (2026-08-12) — all four predictions below HELD; see "Current status" for the numbers.**

**S24a as pre-registered BEFORE the trained/untrained rows existed (kept for the record).** Every number in both deliverables patches
the final sequence position and reads the final logits, so "the path bends sharply" and "the readout
position's own state switches sharply" have never been separated. `experiments/pos_assay.py` puts the
varied character at `pos = 14` and appends `k` filler characters after it (`k` = 0, 1, 2, 4, 8), so the
readout sits `k` characters downstream of the patch. Injection is at the residual stream *entering*
block 0, the only site that keeps both endpoints exact at every `k`. Two readouts come from the same
forward pass: `read_final` (last position, the question) and `read_patch` (position 14, which causal
masking makes independent of the suffix — it must be bit-identical across `k`, a built-in check).
Only the matched-accuracy checkpoint was scored before the previous iteration ran out of budget; the
reference run is retraining now so that the step-0 and step-30,000 rows exist too.

Pre-registration, written with only the matched-accuracy row visible (median `w` 0.328 / 0.363 / 0.379
/ 0.434 / 0.391 for k = 0/1/2/4/8) and no untrained baseline of any kind:

1. **`read_patch` is identical for all five k** at every checkpoint (implementation check; a failure
   invalidates the sweep).
2. **The plateau survives the readout moving away:** trained median `w` stays **below 0.55** at every
   k, i.e. far from the straight line's 0.80, at the step-30,000 checkpoint.
3. **It is a training effect, not prompt geometry:** at every k the untrained (step 0) network is
   **blunter by at least 0.15** in median `w`, with a paired Wilcoxon p < 1e-6.
4. Endpoint separation at the readout falls steeply with k (it already does: 23.6 → 3.5 logit units),
   and `frac_endpoints_differ` reaches 0 by k = 4 — so if 2 and 3 hold, the sharp switch is a property
   of the network state and not of the readout position's next-character decision. If instead the
   widths climb to ~0.8, the whole plateau result is a statement about the patched position only, and
   both deliverables must say so.

**S24 — nothing in the seed queue remains; the open questions all need new compute or a new model.**
S23 closed the last two single-seed runs under load-bearing comparisons (2026-08-10), so no claim in
either deliverable now rests on a single initialization except the six conditions whose gaps are
0.14–0.26 wide, three to six times the measured 0.040 spread. Candidates, in the order they would be
worth running:

1. ~~**What the trainable blocks compute.**~~ **ANSWERED 2026-08-12 by S24b + S24c** — a few dozen
   MLP units per path carry the bend (top-32 of 3,840 recover 50.9%), drawn from a 668-unit pool, and
   those units are character detectors: corpus tuning predicts recruitment at AUROC 0.847 and units
   selected by it alone remove 28.9% of the width gap. **S24d (2026-08-12) closed the remainder as far
   as corpus statistics reach:** the units that rule misses are context-dependent (median 51% of their
   corpus response explained by the current character, against 96% for the ones it finds) and carry
   about a third as much bend each, and conditioning the profile on the previous character ranks the
   population better but selects worse. **S24e (2026-08-12) closed the learned-description question
   too:** a ridge probe over the eight characters ending at the position explains a median 78% of the
   missed units' corpus response out of sample, and used as a selection rule in raw activation units it
   removes 56.5% of the width gap — past the fitted per-pair ranking's 50.9% — with controls showing
   that dropping the per-unit standardization, not the added context, carries the gain. Original
   framing, kept for the record: five
   candidate mechanisms had been excluded in turn (the
   next-character decision, endpoint plausibility, the specific weights of blocks 1–4, any particular
   depth, and trainable parameter count), and the gap has not moved in six iterations: nothing here
   characterises the computation that bends the path, only where it can live and how much of it
   survives. This is the direction's real open problem and it needs a new probe, not another freeze.
2. ~~**Interpolation at positions other than the final token.**~~ **CLOSED 2026-08-12 by S24a** —
   the readout was moved up to eight characters downstream of the patched character and the transition
   width did not change; the decision-basin description was narrowed as a result. What is left on this
   axis is incremental (longer offsets, more contexts, a second seed) and firms up a result rather than
   answering a new question. Note the PLAN 5.5 freeze-and-retrain prediction is *not* open either:
   S10's `frozen_early` is exactly that experiment, and it was refuted — the sharpening relocated to
   blocks 5–8 rather than disappearing.
3. **A longer character run whose second local-complexity descent separates from initial fit**, the
   denser Figure-9 grid on the pilot run's local maximum, or a second model/tokenizer — each needs
   materially more compute than one 30,000-step frozen run.

Practical notes for whoever picks this up: `results/checkpoints*` are symlinks into `/tmp` and do not
survive a pod reset (S24a had to retrain the reference run for exactly this reason; it now sits at
`/tmp/dir13_frozen/checkpoints_ref_pos` with step-0, matched and step-30,000 checkpoints, until the
next reset). Re-download the corpus to `/tmp/tinyshakespeare.txt` (SHA-256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`, asserted by
`allpairs_sweep.load_vocab`) before running anything, and read reference rows from
`results/frozen_assay_summary.json` / `frozen_assay_raw.npz` rather than re-assaying vanished
checkpoints. On the figures: a seventh five-block run would break the three-panel depth split in
Figure 23, and the next division would have to be by window size rather than position.

**S23's pre-registration and outcome (kept for the record).** Both replicates used model seed 2024 with
corpus, split, data order, optimizer, schedule, batch size, checkpoint grid and freeze mask identical
to their seed-1337 originals. `frozen_high_s2` had to land within ~0.04 of 0.342 and clearly below the
reference's 0.443 → **0.344**, spread 0.002 (**held**). `frozen_mirror_s2` had to land within ~0.04 of
0.629 and above both frozen-deep seeds (0.559, 0.590) → **0.624**, spread 0.006 (**held**); at step
30,000 it gives 0.590 against the deep seeds' 0.558 and 0.579, so the ordering survives there too but
with a 0.010 margin.

**S16 IS DONE (2026-08-03) — every claim the depth/position reading rests on now has a measured seed
spread under it.** Depth step: 12-trainable-block runs (0.397-0.443) disjoint from 8-block runs
(0.476-0.500), two seeds a side. Position term: both frozen-deep seeds (0.559, 0.590 at matched
accuracy) below frozen-mirror's 0.626, spread 0.031.

**S17, S18 and S19 are all DONE and fully curated (2026-08-03); nothing is waiting on disk.**

**S20 IS DONE (2026-08-03) and it REFUTED its own pre-registered prediction.** The five-block window
at blocks 1-5 (`--freeze 0,6,7,8,9,10,11 --tag frozen_mid_low`) had to land **above 0.47** if the
interior/end split held. It lands at **0.363** (matched acc, step 3,500) and **0.326** (step 30,000) -
the sharpest final width of any of the fourteen models here - indistinguishable from the two mid-stack
five-block windows (p = 0.27, 0.23) and 0.10-0.23 clear of every end window. The split is **withdrawn**
from both deliverables.

What replaced it is deliberately not another fitted rule: blocks 1-5 are a strict **subset** of
frozen-late's trainable 0-7 and are **0.118 sharper** (p = 2.2e-25), so removing trainable blocks
sharpens the plateau - a two-network fact that survives whatever happens to the descriptions. The
geometry is demoted to a labelled description: every usable window covering mid-stack **block 5** gives
0.363-0.500, the three without it 0.559-0.712.

**Immediate next candidates (S22).** (a) Add `frozen_high`'s step-30,000 row from the chained assay to
both deliverables - no new training, one row plus a Figure 23/24 refresh, and it also restores the
final-step panel the caption currently flags as pending. (b) A **second seed at frozen-mirror**, the one
single-seed run carrying a load-bearing comparison. (c) A **second seed at blocks 6-10** to confirm the
study's sharpest network is not a seed artefact. Deliberately NOT a third fitted geometric rule: two
have now died on their first test each, so another curve through ten points earns nothing. Note for the
figure work: a seventh five-block run would break the three-panel depth split again - the next division
would have to be by window size, not position.

**Superseded next candidate (S21, now DONE and reported above): the five-block window at blocks 6-10** (`train_frozen.py --freeze
0,1,2,3,4,5,11 --tag frozen_high`). It is the pre-registered test of that description - it excludes
block 5 while touching neither end, so coverage predicts **>= 0.55** on a window the refuted rule would
have called sharp; near 0.365 refutes it too and leaves the series with no geometric summary, which is
itself the honest finding. ~21 min of training plus ~2 min of `narrow_assay.py frozen_high`, then
`frozen_pairwise.py` -> `plot_capacity.py` -> `plot_frozen.py`; all three need the tag added first
(`plot_frozen.py` STYLE + the left DEPTH_GROUPS panel, which would then hold six five-block series -
split it by window position, the five-hue budget is exhausted; `plot_capacity.py` needs a seventh nudge
slot in the 5-block column).

**The alternative** is a second seed
at frozen-mirror, the only remaining single-seed run carrying a load-bearing comparison (the
bottom-of-stack end of the position ordering); it firms up an old number rather than answering a new
question, so run it second. Everything else
open needs a longer character run whose second descent separates from initial fit, the denser Figure-9
grid on the pilot run's local maximum, interpolation at non-final positions, or a second
model/tokenizer.

**Practical note for the next agent: `/tmp` does not survive.** The pod reset before this iteration
wiped every scratch checkpoint (`/tmp/dir13_ckpt_*`, `/tmp/dir13_frozen/*`) and the corpus. Re-download
tinyshakespeare to `/tmp/tinyshakespeare.txt` (SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`,
asserted by `allpairs_sweep.load_vocab`) before running anything. Only the reference run's *per-pair*
results survive, in `results/frozen_assay_raw.npz` — which is enough to score a new run against every
existing condition, but not to re-measure an old one.

Four candidate mechanisms have now been excluded in turn: the next-character decision (survives the
ablation that destroys the plateau), endpoint plausibility (does not mediate the intervention and moves
the wrong way), the specific weights of blocks 1-4 (freezing them at init still yields plateaus,
relocated to blocks 5-8), and any particular depth at all (freezing blocks 1-7, then 5-11, relocates the
sharpening again each time). S13 added the boundary condition those four exclusions were missing: the
relocation is free in *site* but not in *amount*, and below roughly one usable block there is no
plateau left to relocate. What the responsible blocks actually *compute* to produce the sharp change
remains uncharacterised - that gap has not moved in five iterations.

**S14 is DONE and it answered the question.** The narrow run (`n_embd` 192, nothing frozen, 5.38M
parameters against frozen_early's 5.60M trainable ones - 4.0% *fewer*, corrected 2026-08-03) lands at
median `w` **0.397** at matched accuracy - the depth account's ~0.35-0.44, not the capacity account's
~0.47. Trainable depth is the variable; parameter count is not. (S14c's second seed gives 0.437, so the
spread is ~0.04; its one casualty is the claim that the narrow run is *sharper* than the reference,
which does not replicate.)

**S14b is DONE: the fully-trained row confirms it under the second framing.** The narrow run finished
(stopped by the harness time budget at step 27,143 of 30,000, lr 1.2e-4 rather than 1.0e-4 - reported
with the number) and `narrow_assay.py` scored it: median `w` **0.332**, i.e. **-0.010** against
ref_trained's 0.351 (43% of pairs wider, p = 2.1e-4) and **-0.124 / -0.146** against frozen-early's
0.471 and frozen-late's 0.484 (1.3% / 3.3% wider, p ~ 3e-26). Both framings therefore agree, and the
truncation biases against the finding rather than for it, since the run was still sharpening
(0.397 -> 0.332, p = 3.1e-14). Figure 24 now shows both framings per run.

**One smaller successor remains.** The depth series still has one seed per condition, so a second seed
at `n_embd` 192 and at one frozen condition would put an across-seed error bar on the 0.397-vs-0.476
gap that carries the argument. ~16 min of training each on this harness plus ~70 s of assay.

Everything else still open needs new compute or a new model: a longer character run whose second
descent separates from initial fit; the denser Figure-9 grid applied to the *pilot* run's local
maximum; interpolation positions other than the final token; a second model/tokenizer.

## Primary references

- Matthew Shinkle and StefanHex, *Activation Plateaus: Where and How They Emerge*: https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge
- Matthew's exact configuration and code: https://github.com/MShinkle/activation_plateau_mechanisms
- Humayun, Balestriero, and Baraniuk, *Deep Networks Always Grok and Here is Why*: https://arxiv.org/abs/2402.15555
- Grokking-paper repository: https://github.com/AhmedImtiazPrio/grok-adversarial