# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-07-15 — Complete study end-to-end; verdict: no plateaus (qualified)

**Did.** No unaddressed feedback files. Ran the whole plan in one long iteration (S1→S6):
1. S1: GitHub-API audit of `AhmedImtiazPrio/grok-adversarial` — only MNIST-MLP + CIFAR-ResNet code,
   **no GPT/Shakespeare code or checkpoint**. Wrote `MODEL_SPEC.md` (confirmed vs reconstruction tags).
2. S2: `experiments/model.py` (12L/12H GeLU GPT with residual read/replace hooks) + `train.py`.
   Trained on Tiny Shakespeare → val loss 1.494, next-char acc 0.560. Saved log-spaced checkpoints,
   `train_meta.json` (corpus SHA, seeds), `plots/training_curves.png`.
3. S3: `assay.py` — final-position residual perturbation, d_hidden/PI/sharpness/JSD, unit test
   (detects synthetic plateau PI +0.33, line PI 0.00). alpha=0 partial-forward matches full forward
   <1e-3.
4. S4/S5: `run_confirm.py` — pilot froze per-block rho_max (flip≥0.8), confirmatory swept ALL blocks
   0–10 on 48 held-out contexts × 8 dirs × 41 radii, natural + matched control.

**Learned.** median PI(natural) is **negative at every block** (−0.15…−0.30): the GPT's downstream
response to residual perturbations is concave/**saturating**, not the flat-then-steep plateau shape.
ΔPI(nat−ctrl) is positive & significant everywhere (peak +0.096, Cliff's δ +0.91, JSD agrees) but that
is a difference between two non-plateau shapes → mild on-manifold structure, not a plateau. Individual
rays confirm (no ray is flat-then-steep). Interpretation: plateaus look architecture-specific
(piecewise-linear ReLU MLP in the paper's MNIST result) vs the additive residual/GeLU/LayerNorm GPT.

**Verdict.** Success-criterion (2) NO plateaus, qualified by (3) reconstruction (paper's exact GPT
unreleased). No-go for a plateau-mapping follow-up on this model. Deliverables written current-best;
math + figure-embed checks pass. Wrote `STOP`.

**Next step.** None for this direction (complete). If revisited: test a much longer-trained checkpoint
(grokking-scale) and/or learned rather than random directions — but that overlaps the "during
training" direction and is out of this gate's scope.

On track? yes — S6 done, 100%, no blocker (calibrated negative; STOP written).

## 2026-07-16 — Rule-compliance curation; re-verify; STOP (verified on disk this time)

**Did.** Checked feedback first: no `human_feedback*`/`*REVIEW*` files in this direction (the one in
`dir12_plateau_during_training/` belongs to that direction, not this one). Found the previous
iteration's `STOP` absent from disk despite JOURNAL claiming it was written (no git trace either —
it was evidently never persisted; assumption logged: persist failure, not operator deletion, since no
feedback accompanied the relaunch). CLAUDE.md gained a new rule-9 requirement (motivate every metric
before defining it; a metric no Result uses gets cut) — brought deliverables into compliance:
narrative Methods with per-metric motivation + consuming figure, numbered figures, and made boundary
sharpness consumed by Results (new paragraph in REPORT.md + `sharp nat / ctrl` table column in
RESULTS.md from existing `confirm_summary.json`; no re-runs). All render checks pass (6/6
js-display-math, 0 pre-lang-math, clean hazard grep, all plots embedded as `![]()`).

**Learned.** Sharpness data corroborates the verdict: natural rays are *less* sharp than control at
every block (2.16–4.01 vs 2.99–4.91), and with PI < 0 the steep segment is the initial rise — no
learned wall. Also: always verify STOP exists on disk after writing it.

**Next step.** None — direction complete; STOP re-created and verified.

On track? yes — S6 done, 100%, no blocker (calibrated negative; STOP verified on disk).

## 2026-07-17 — Feedback-first iteration: human_feedback_1 addressed

**Did.** Found `human_feedback_1.txt` unaddressed (and `STOP` absent from disk again — assumption:
the wrapper/operator cleared it when dropping feedback, since a STOP'd direction would ignore the
file; alternative rejected: another persist failure — either way the correct move is identical).
Feedback asks: (1) remove the disallowed `\operatorname` macro from REPORT.md; (2) CLAUDE.md was
updated — make it more readable and concise. Actions:
1. Replaced `\operatorname{median}` with `\mathrm{median}` in the ΔPI display equation (only
   occurrence in any deliverable).
2. Rewrote REPORT.md tighter (215 → 197 lines): shorter sentences, redundancy cut, jargon still
   defined on first use, per-metric motivation kept, figure captions now name axes and legend
   entries explicitly. Interpreted "make it more readable" as referring to REPORT.md (the file the
   feedback names); RESULTS.md has no banned macros and is already table-driven and short, so left
   unchanged (logged as the rejected alternative).
3. Verified via the GitHub markdown API: 6/6 js-display-math, 0 pre-lang-math, no inline hazards,
   all plots embedded as `![]()`.
4. `mv human_feedback_1.txt human_feedback_1.txt.addressed.md` (contents untouched).

**Learned.** The operator's renderer rejects `\operatorname` even though vanilla KaTeX supports it —
prefer `\mathrm{…}` for upright operator names in this project.

**Next step.** None — plan complete, zero unaddressed feedback, STOP re-created and verified.

On track? yes — S6 done, 100%, no blocker (feedback addressed; calibrated negative stands).

## 2026-07-17 (2) — Reopened plan executed end-to-end; verdict reversed: plateaus PRESENT

**Did.** Feedback check first: only `human_feedback_1.txt.addressed.md` (addressed), no `*REVIEW*`,
no STOP on disk; PLAN.md was rewritten by the operator to REOPEN with the Matthew-style
two-natural-endpoint slerp assay (S3–S7 unchecked). Corpus was missing from /tmp (pod reset) —
re-downloaded, SHA-256 matches training provenance exactly. Then ran the whole reopened plan:
1. S3 `make_pairs.py`: froze 40 minimal pairs from val text (seed 20260717; A=observed char,
   B=model top-1; 0 degenerate; endpoint logit dist 8.7–64.4) → `results/prompt_pairs.json`.
2. S4 `matthew_assay.py`: slerp (norm-linear, clamped, collinear fallback), last-position patch via
   exact partial forward, d(t), PAVA w_10→90, frozen plateau rule. Self-tests pass (step w=0.089
   detected, line w=0.800 rejected).
3. S5/S6 `run_matthew.py`: implementation checks on real pairs all pass (endpoints <1e-3, prefix
   <1e-4, batch=single <1e-5), then primary + layerwise + depth runs (~3 min total on GPU).

**Learned.** **14/40 frozen pairs show individual plateau–boundary–plateau curves at final logits**
(median w 0.309 vs diagonal 0.8; 0 non-monotone; only #10 and #19 near-diagonal). Layerwise: median w
falls strictly monotonically with recording depth (0.777 → 0.445 → 0.309), strict rule passes only at
logits — the plateau is formed by the downstream stack. Depth control: interpolating later weakens it
monotonically to the diagonal (w 0.309 → 0.802 at block 10). So the earlier random-ray "no plateaus"
was assay-dependent: plateaus live along natural activation-to-activation directions, not random
ones. Deliverables rewritten (old assay → CHANGELOG history only); all math/embed render checks pass.

**Next step.** None — S3–S7 complete, PLAN decision rule "plateaus present" satisfied (multiple pairs,
individually visible, coherent layerwise sharpening), qualified-reconstruction scope stated. STOP
written after re-verifying zero unaddressed feedback.

On track? yes — S7 done, 100%, no blocker (plateaus present; STOP verified on disk).

## 2026-07-17 (3) — Feedback-first iteration: human_feedback_2 addressed

**Did.** Found `human_feedback_2.txt` unaddressed (and STOP absent again — same pattern as feedback #1:
operator clears STOP when dropping feedback so the loop re-enters; proceeded on that assumption).
Two asks: (1) the Summary's "two independent signatures" sentence was confusing — layers in the first
half, an undefined "diagonal" in the second; (2) the Summary never said whether our trained model
replicates the phenomenon introduced in the Grok paper. Fixes (text-only, no re-runs, numbers
unchanged):
1. REPORT.md Summary: defined the diagonal at first use (the line d = t the relative-distance curve
   traces under a uniform output morph, width 0.8), glossed d and t inline, and rewrote both
   signatures as movements of the same curve relative to that diagonal (deeper recording → away from
   it; later interpolation → back onto it).
2. REPORT.md Summary: added an explicit scope note — the grok paper's phenomenon (grokking = delayed
   adversarial robustness, defined on first use) is NOT tested/replicated; our reconstruction stops at
   ordinary convergence and the paper only specifies the model. (Alternative rejected: actually
   training to grokking scale to test replication — out of scope per PLAN "do not study
   checkpoint-to-checkpoint evolution" and far beyond remaining budget.)
3. RESULTS.md verdict paragraph updated in parallel (same definition + note).
4. Render checks pass: 4/4 js-display-math, 0 pre-lang-math, no inline hazards, no unembedded plots.
5. `mv human_feedback_2.txt human_feedback_2.txt.addressed.md`; CHANGELOG entry appended; STOP
   re-created and verified on disk.

**Learned.** Every term a figure or verdict leans on ("the diagonal") must be defined in the same
paragraph it is first used, not only in Methods; and when a source paper is used only for its model
spec, say explicitly that its headline phenomenon is not being replicated.

**Next step.** None — plan complete, zero unaddressed feedback, STOP verified.

On track? yes — S7 done, 100%, no blocker (feedback #2 addressed; plateaus-present verdict stands).

## 2026-07-17 (4) — Stale PLAN re-drop handled by full re-verification; verdict reproduced bit-exactly

**Did.** Feedback check first: only the two `.addressed.md` files, no `*REVIEW*`, nothing unaddressed.
Found STOP cleared and PLAN.md replaced (uncommitted, per git diff vs HEAD 956f54c) with the SAME
reopen-plan text executed in entry "(2)" — S3–S7 unchecked, status "No Matthew-style minimal-pair
slerp experiment has yet been run" (false on disk: `results/matthew_summary.json` exists), and the
file truncated (title/research-question/setup sections missing; starts at "Required implementation
checks"). **Assumption:** this is a stale/accidental re-drop during the operator's project-wide file
pass (run.sh + CLAUDE.md modified, new feedback dropped in dir12, none here), not a new ask — no
requirement in it differs from the plan already executed. **Alternatives rejected:** (a) blindly
re-tick the boxes without running anything — leaves the claim unverifiable this iteration; (b) redo
S3–S7 from scratch with fresh pairs — violates the frozen-pairs preregistration and answers nothing
new. **Chosen middle path:** re-run the entire pipeline end-to-end against the frozen pairs (~3 min
GPU) and compare. Corpus present, SHA-256 matches provenance. Result: **bit-exact reproduction** —
summary JSON identical (14/40 plateaus, same pair IDs, median w 0.309, depth medians identical to all
digits), all self-tests and implementation checks pass, no tracked artifact changed. Deliverables
untouched (already current-best); render checks re-pass (4/4 js-display-math, 0 pre-lang-math, no
hazards, no unembedded plots). PLAN.md restored to COMPLETE (edited in place; S3–S7 ticked).
CHANGELOG entry appended. STOP re-created and verified on disk.

**Learned.** When a plan re-drop contains no new requirement, the cheapest honest response is a full
re-verification run, not re-ticking from memory — it converts "the journal says it was done" into
"it reproduces today, bit-exactly". Also: the pipeline is fully deterministic across pod sessions
(same GPU class, same torch), which strengthens the reproducibility claim in the deliverables' favor.

**Next step.** None — plan complete, zero unaddressed feedback, STOP verified.

On track? yes — S7 done, 100%, no blocker (verdict reproduced bit-exactly; STOP verified on disk).

## 2026-07-17 (5) — Fresh-run dynamics figure + queued late-checkpoint char gate eval

**Did.** Feedback check first: only the two `.addressed.md` files, nothing unaddressed. Found the
reopened S3–S5 pipeline mid-flight in background: fresh char training (step ~21k/30k, PID 74887) and
BPE training (step ~9.5k, PID 76638) both running under budget caps; an early-checkpoint char fig9
eval (steps 0–4994, PID 84700) running; a chained BPE fig9 waiting on GPU. Inspected `fig9.py` — it
**resume-merges** into `--out` (reads existing records, skips done steps, appends). Two gaps: (1) both
queued evals only cover checkpoints ≤4994 while training reaches 20k+, so a *second* LC descent (the
whole gate) couldn't be detected; (2) no new deliverable figure this iteration.
1. Started to queue a late-checkpoint char eval, but discovered a prior iteration had ALREADY queued
   `/tmp/chain_char_late.sh` (PID 84882) doing exactly this — waits for training + early eval, then
   dynamically resume-merges late steps (incl. `MAX` and final) into `results/fig9_grok_char.json`.
   My duplicate would have been a second concurrent writer to the shared JSON, so I **killed it**
   (85258) and kept the pre-existing chain. Net: late-checkpoint char gate coverage is queued.
2. Wrote `experiments/plot_fresh_training.py`; generated `plots/fresh_training_dynamics.png` from the
   live logs. Embedded it in RESULTS.md and REPORT.md (Fig 1c) as clearly-labelled in-progress context.

**Learned.** Both fresh runs **overfit**: val loss bottoms very early (char 1.47@~3750; BPE 4.77@~750)
then climbs while train loss keeps falling; val acc plateaus (char ≈0.56, BPE ≈0.27). That is the
opposite of grokking's delayed val recovery — strong prior that the fresh gate will also FAIL within
the budgeted 30k horizon, but the per-checkpoint LC/PGD eval is the actual test and isn't done yet.

**Next step.** When char training + both fig9 evals finish: run `fig9_verdict.py` on the merged
`fig9_grok_char.json` (now with late checkpoints) → fresh char gate verdict (S4); plot its full LC/PGD
curve (`plot_fig9.py`); same for BPE (S5); then freeze 6 checkpoint phases and run the Matthew
checkpoint sweep (S6) → joint timeline (S7) → rewrite report + STOP (S8). Do NOT STOP until fresh
verdicts + joint timeline exist.

On track? yes — S3–S5 running in background, ~55% of reopened plan; blocker: awaiting background
training/eval completion (queued, no action needed until they finish).

## 2026-07-17 (6) — Killed overfitting BPE run to unblock the Fig-9 gate eval (S4/S5)

**Did.** Feedback check first: only the two `.addressed.md` files — nothing unaddressed. Found the
S3–S5 pipeline still mid-flight: fresh char training at step ~24k/30k (PID 74887), early char fig9
eval running (PID 84700) with `chain_char_late.sh` (PID 84882) queued to resume-merge late steps,
and BPE training at step ~10.3k/30k. **Decision:** killed BPE training (76638/76636) and the old
`chain_bpe_fig9.sh` (82594). Rationale: BPE val loss bottomed at step ~750 (4.77) and has risen
*monotonically* to 10.6 by step 10k — pure memorisation, no grokking possible — yet the old chain
blocked the entire BPE Figure-9 gate eval on the *full* 30k run (~87 more min of GPU for zero
scientific value), when every checkpoint the gate needs (0..10000) already exists on disk.
- **Rejected alternative:** let BPE train to 30k. Rejected — it would consume the whole remaining
  window, and the gate only needs per-checkpoint LC/PGD, computable now from saved checkpoints.
- Wrote `/tmp/chain_finalize_gates.sh` (PID 85559): waits for char training + all char fig9 activity
  to settle, then runs BPE fig9 over 0..10000 (adds late 7819/10000 vs the old chain's ≤4994 for
  second-descent coverage), writes both gate verdicts via `fig9_verdict.py`, and plots both curves
  via `plot_fig9.py` (`plots/grokking_fresh_char.png`, `plots/grokking_fresh_bpe.png`).

**Learned (in progress).** Both fresh runs are memorising, not grokking (char val bottomed ~3750 then
rose; BPE bottomed ~750 then rose) — strong prior both gates FAIL/NOT-ESTABLISHED within budget, which
would push the joint verdict toward PLAN case 5 (primary relationship not testable via a grokking BPE
model). The verdict scripts are the actual test; awaiting the finalize chain.

**Next step.** When finalize chain writes DONE: read both verdict JSONs, curate RESULTS.md + REPORT.md
with the fresh char (S4) and BPE (S5) gate verdicts, embed both new curve PNGs in BOTH files, append
CHANGELOG. Then S6/S7 (Matthew sweep at frozen phases + joint timeline) next iteration.

On track? yes — S4/S5 gate evals unblocked & running; ~60% of reopened plan; blocker: awaiting the
finalize chain (autonomous, no action needed until DONE).

## 2026-07-17 (7) — Fresh gate verdicts done (S4/S5), joint timeline + bounded case-5 verdict (S7)

**Did.** Feedback check first: only the two `.addressed.md` files — nothing unaddressed. Found the
finalize chain (85559) had run but **BPE fig9 OOM'd** at `vram_frac 0.06` (vocab-50257 head needs
>1.88 GiB) — so no BPE verdict/curve, and the char side had completed cleanly.
1. Char training reached step 30000 (all ckpts saved); its post-run metadata/plot save crashed on an
   int64 JSON error — harmless, checkpoints intact. Char-late fig9 finished the full 14-ckpt curve
   (0→30000): LC monotone 1940→8.1 (min at last ckpt), adv→0.528. `fig9_verdict.py` → **FAIL**.
2. Re-ran BPE fig9 with adequate memory (`vram_frac 0.2, pgd_bs 16, lc_bs 4`); 10 ckpts 0→10000:
   LC 2182→95 monotone, adv→0.187, clean peak 0.299@831. `fig9_verdict.py` → **FAIL**.
3. Wrote `experiments/plot_joint_timeline.py`; generated `plots/joint_timeline.png` (LC + adv vs step
   for all 3 runs + verdict/plateau text panel). Also generated `grokking_fresh_char.png`,
   `grokking_fresh_bpe.png`.
4. Curated RESULTS.md + REPORT.md: replaced the "in progress" fresh-dynamics paragraph with the
   completed 3-model gate table, three curve figures, joint timeline, and the **bounded relationship
   verdict = PLAN case 5 ("primary relationship not testable")**. Updated Summary/Conclusion/Limitations.
   Verified 6/6 display-math render, 0 broken, all 8 figures embedded as `![...]` images and on disk.

**Learned.** The fresh char run makes the null crisp: adversarial robustness climbs *higher* than the
pilot (0.53 vs 0.33) yet test LC never stops descending — so "delayed robustness" alone is NOT
grokking; the defining *second LC descent* is absent because the model memorises (val loss bottoms
early then rises) rather than groks. With no run passing the gate — and the BPE bridge to Matthew's
exact tokens failing — the joint question is not testable (case 5); plateaus stand alone.

**Next step.** Remaining plan work: S6 (Matthew `b/i`,`b/l` char controls across frozen checkpoints as
*secondary* per-checkpoint plateau evidence; big/in,big/large BPE sweep is non-decisive under case 5)
and S8 (de-emphasise the 40-pair reconstruction dataset in the headline per PLAN out-of-scope, keep it
as clearly-labelled standalone evidence). Then STOP. Do NOT STOP yet — S6/S8 pending.

On track? yes — S4/S5/S7 complete with FAIL/FAIL/FAIL gates and a bounded case-5 verdict; ~75% of the
reopened plan; blocker: none (S6/S8 are straightforward next-iteration work).

## 2026-07-17 (8) — S6 char-control per-checkpoint plateau assay + S8 rewrite; plan complete, STOP

**Did.** Feedback check first: only the two `.addressed.md` files — nothing unaddressed. Both remaining
stages closed this iteration.
1. **S6.** Generated frozen phases from the fresh-char Figure-9 curve (`freeze_phases.py` →
   `results/frozen_phases_char.json` = steps 0,56,831,7819,17500,30000; LC monotone so log-spaced
   fallback). Ran `run_matthew_ckpts.py --tok char --steps 0,56,831,7819,17500,30000 --out_tag char_ctrl
   --vram_frac 0.2` — Matthew's exact code path (context "The house was", 50-step slerp, full interp-layer
   sweep, single-position patch) on the `b↔i`,`b↔l` single-token controls. Clean run, DONE. Plotted with
   `plot_matthew_ckpts.py` → `plots/matthew_char_ctrl_by_checkpoint.png` + `plots/joint_timeline_char_ctrl.png`.
2. **Key finding.** Block-0 final-logit width: 0.80 (init, diagonal) → 0.35 (step 831) → 0.33 (step 30k).
   The plateau **emerges during the first LC descent / initial accuracy rise and is fully formed before
   ε=0.03 robustness saturates** (steps ~10³–10⁴). Depth control holds too (0.33@L0 → 0.80@L11 at 30k).
3. **S8.** Rewrote RESULTS.md + REPORT.md: the char-control per-checkpoint assay is now the **primary**
   plateau evidence; the 40-pair natural-minimal-pair sweep is demoted to a clearly-labelled
   **exploratory** section (PLAN out-of-scope forbids it in the headline). Retitled both to the joint
   question. Updated Summary/Conclusion/Limitation-4 with the secondary temporal observation. Render
   checks pass (REPORT 6/6 display-math, 0 broken, 0 stray plot paths, 0 inline hazards; all figures on disk).

**Learned.** Even in a model that never groks, the plateau is a real, timeable phenomenon — but it tracks
*initial fit* (first LC descent), not a second-descent/robustness window. This tightens the case-5 null:
not just "untestable" but "no visible temporal coupling," because the plateau is already saturated before
any grokking-like event could occur. So the bounded relationship verdict is case 5, refined toward
"no temporal relationship" for the secondary char-control evidence.

**Next step.** None — all stages S1–S8 complete, every success criterion met, `STOP` written. If new
`human_feedback*.md`/`*REVIEW*` arrives, delete STOP, address it, re-STOP when clean.

On track? yes — plan COMPLETE (100%); S6+S8 done, bounded case-5 verdict with S6 secondary evidence, STOP written; blocker: none.

## 2026-07-25 (9) — operator feedback #3: comma → all other characters

**Did.** Feedback check first: `human_feedback_3.txt` was unaddressed, so that was this iteration.
Ask: "interpolate from comma to all other characters and see if there is a plateau; add a section to
discuss the results; do not invent jargon."
1. Wrote `experiments/comma_sweep.py` reusing the existing assay functions (`run_pair`,
   `transition_width`, `is_plateau`) with no new machinery: endpoint A = `"The house was ,"`,
   endpoint B = the same context + each of the 64 other characters; 50 steps, `slerp_rescale`,
   final-position patch, final-logit `d(t)`. Ran at interpolation block 0 for the 6 frozen
   checkpoints and at every block 0–11 at step 30,000.
2. Added two extra measurements at the final checkpoint so the discussion is evidence-based rather
   than speculative: the model's next-character probability for each target character after the
   context, and the L2 distance between the two endpoints' logit vectors.
3. Wrote `experiments/plot_comma_sweep.py` → 4 figures (all 64 curves + width histogram; width per
   character coloured by character type; width vs probability / vs endpoint separation; depth and
   training panels). Embedded all four in RESULTS.md and REPORT.md with axis-describing captions.
4. Curated both deliverables (new section + discussion in each, Methods subsection with rendered
   equations for the two new quantities and Spearman ρ, Summary/Conclusion/Limitation updates),
   appended CHANGELOG, renamed the feedback file to `.addressed.md`.

**Assumptions logged (loop mode, could not ask).** (a) The corpus file `/tmp/tinyshakespeare.txt` is
gone from this pod, so the character vocabulary was restored from the pilot checkpoint's saved `stoi`
(both runs built it as `sorted(set(text))`, both are the canonical 65-character Tiny Shakespeare
vocab) — rejected alternative: re-downloading the corpus (no network guarantee, and the checkpoint
copy is exact). (b) Kept the *same* shared context as the existing controls (`"The house was "`)
rather than inventing a comma-friendly context, so widths are directly comparable to `b↔i`/`b↔l`;
rejected alternative: a context where a comma is natural (would have made the sweep incomparable).
(c) Reported the full width distribution instead of a single yes/no plateau count, since the strict
0.25 bar splits a continuum.

**Learned.** The plateau *shape* is universal in this model but its *sharpness* is graded: none of the
64 pairs is linear (median width 0.340 vs 0.80 for a straight line), yet only 1/64 clears the strict
≤0.25 bar. The strongest predictor of sharpness is how likely the model thinks the second character
is in that context (Spearman ρ = −0.74, n = 64) — letters switch sharply, punctuation and the digit
`3` drift. Endpoint separation is the weaker predictor (ρ = −0.48) and with the sign that rules out a
"the two outputs are too similar" artifact. Also reassuring: the preregistered `b↔i`/`b↔l` controls
(0.331/0.330) sit exactly at the 64-pair median, so the primary result was not a lucky pair. Both
structural controls (depth, emergence across training) replicate at n = 64.

**Next step.** None outstanding: plan stages S1–S8 complete and feedback #3 addressed, so `STOP` is
re-written. If new `human_feedback*.md`/`*REVIEW*` arrives, delete `STOP`, address it, re-STOP clean.

On track? yes — plan COMPLETE (100%) plus operator feedback #3 fully addressed with a new experiment,
4 new figures and a discussion section in both deliverables; blocker: none.

## 2026-07-25 (10) — CVD compliance sweep across all figures + context control (576 pairs)

**Did.** Feedback check first: the three `human_feedback_*.txt.addressed.md` files are all addressed;
nothing unaddressed. `STOP` was absent from disk again (same pattern as previous relaunches). Plan
stages S1–S8 were already complete, so this iteration did the two highest-value things left.

1. **Hard-rule fix (CLAUDE.md rule 13).** Audited every figure embedded in the deliverables and found
   real red-green violations: the Figure-9 panels drew local complexity for train/test/**random** in
   matplotlib's C0/C1/**C2 (green)** against a **red** dashed adversarial-accuracy line; the
   comma-sweep character classes were encoded **green** (upper-case) vs **red** (punctuation/digit)
   with colour as the *only* channel; `layerwise_emergence.png` drew the final-logit curve in **red**
   over a viridis ramp. Wrote `experiments/cvd_style.py` and patched the five plot scripts plus a new
   `plot_training_curves.py` (so `training_curves.png` could be redrawn from saved history instead of
   retraining). Every series now also carries a linestyle, marker or hatch; sequential ramps are
   viridis/cividis. Re-ran everything; `run_matthew.py` reproduced **bit-exactly** (14/40, median w
   0.309), so no result moved. Rewrote every caption in both deliverables so no series is named by
   colour, and added a "Figure conventions" Methods subsection.
2. **Context control (new experiment).** The deliverables' top caveat was that every plateau number
   came from one shared context (`"The house was "`) whose comma endpoint is implausible there.
   `experiments/context_sweep.py` repeats the comma→all-64-characters sweep in **8 further
   64-character contexts** from held-out text, picked to span p(comma) from 5e-20 to 0.997 → 576
   pairs, same settings, all implementation checks passing.

**Learned.** The shape claim got *stronger* and the correlation claim got *weaker*, which is exactly
what a control is for. **0 of 576** curves is near-linear and per-context median widths sit in a tight
band 0.313–0.436, so plateau-shaped response is a property of the model, not of the chosen context.
But the "sharper for characters the model expects" correlation is −0.74 only in the context we happened
to report first: across nine contexts it is negative every time (sign test p = 0.004, 7/9 individually
significant) with median **−0.41**, range −0.05 … −0.74, pooled −0.23. I curated that into both
deliverables as a range rather than a point estimate, and added Limitation 5 about single-context
correlations overstating effects. The implausible-endpoint caveat is retired: the context where a
comma is 99.7% likely gives median width 0.330 vs the reference 0.340, and p(comma) does not predict
sharpness across contexts (ρ = −0.32, p = 0.41, n = 9).

**Assumptions logged (loop mode, could not ask).** (a) Re-downloaded the Tiny Shakespeare corpus (SHA-256
matches `train_meta.json`/`train_meta_grok_char.json` exactly) rather than reconstructing the vocabulary
from a checkpoint, so the new contexts are genuine held-out validation text; (b) `results/train_meta_grok_char.json`
is truncated by the known post-run save crash, so the corpus SHA was verified against the intact pilot
metadata (same corpus) — rejected alternative: skipping the SHA check; (c) treated the context control
as a *control on the operator-requested sweep*, not a new headline dataset, so it does not conflict with
PLAN's "no new minimal-pair dataset in the primary analysis"; (d) reported the within-context correlation
distribution plus a sign test rather than a single pooled ρ, because pooling mixes context-level width
differences and attenuates the estimate (the pooled value is reported alongside, not instead).

**Next step.** None outstanding — plan complete, zero unaddressed feedback, `STOP` re-written and
verified on disk. If new `human_feedback*.md`/`*REVIEW*` arrives, delete `STOP`, address it, re-STOP
when clean. Natural follow-ups if the direction is reopened: other interpolation positions (not just
the final token) and a second model, the two scope limits the context control does not cover.

On track? yes — plan COMPLETE (100%); this iteration fixed a hard CVD rule violation across all 14
figures and added a 576-pair context control that refines one claim and strengthens another; blocker: none.

## 2026-07-26 (11) — operator feedback #4: REPORT.md math did not render on GitHub

**Did.** Feedback-first per CLAUDE.md Part C. `human_feedback_4.txt` was a screenshot-paste of
REPORT.md's Methods paragraph ending in *"The following macros are not allowed: operatorname"*, plus
"if you find something useful, update CLAUDE.md".

1. Diagnosed: `p(c)=\operatorname{softmax}(f(x_{ctx}))_c` at REPORT.md L140. GitHub runs KaTeX with a
   macro **denylist**; a denied macro nukes the entire equation. Fixed to `\mathrm{softmax}` and made
   the subscript upright (`x_{\text{ctx}}`). The rest of the pasted garble was just copy artefacts of
   correctly-rendered inline math (each KaTeX glyph is its own span, so `x_{ctx}` pastes as "x c t x").
2. **This was a regression of feedback #1** — the identical complaint, marked addressed 2026-07-17. It
   came back with the new Methods paragraph written for feedback #3. So a prose fix alone is worthless
   here; the failure needs a machine check.
3. Wrote `experiments/check_render.py` + `experiments/katex_compile.js`: KaTeX-compiles every display
   fence; KaTeX-compiles every inline `$…$` **after applying GitHub's backslash-before-punctuation
   stripping** (so rule-8b breaks become real errors, not silent ones); flags denylisted macros;
   confirms display-math placement via the GitHub markdown API; checks every plot path is an `![…]`
   embed. Self-tested on a file with one of each bug: 5/5 caught, exit 1. Both deliverables: 0 problems.
4. Updated `../CLAUDE.md` surgically with **8c** (never `\operatorname`/`\DeclareMathOperator`, nor the
   `\def`/`\newcommand` and `\href`/`\html*` families; use `\mathrm`/`\text`; `\max`, `\arg\max`, `\Pr`,
   `\arccos` are built in) and **8d** (run the one script; eyeballing has failed every time).
5. Recorded the check in both deliverables (REPORT.md §Figure conventions, RESULTS.md §Implementation
   checks) so the answer lives in the deliverables, then renamed the feedback file to
   `human_feedback_4.addressed.md` and logged CHANGELOG.

**Learned.** The two render checks CLAUDE.md already mandated are both *necessary and insufficient*:
the GitHub-API check only proves the delimiters were recognised (it returns the raw LaTeX inside a
`js-display-math` element — GitHub compiles client-side, so the API can never tell you the formula is
broken), and the inline-hazard grep only covers backslash-punctuation. `\operatorname` passes both and
also passes a plain KaTeX compile, because KaTeX supports it and only *GitHub* forbids it. That is
exactly the gap that let the same operator complaint recur twice, and it is why the denylist check has
to be explicit. Confirmed independently: GitHub community discussion #55368 reports `\operatorname`
not working in GitHub markdown math.

**Assumptions logged (loop mode, could not ask).** (a) The denylist beyond `\operatorname` (definition
macros, `\href`/`\html*`) is **precautionary** — only `\operatorname` is confirmed by the operator's
screenshot; I could not verify the rest because GitHub compiles math in the browser and the markdown
API returns raw LaTeX. Rejected alternative: listing only `\operatorname`, which would let the next
blocked macro through. (b) Installed `katex` via npm into `/tmp/katexcheck` (not into the repo, and not
touching the pinned torch/TransformerLens stack per PLAN out-of-scope); the script auto-installs it on
a fresh pod. (c) Did **not** fix the `\operatorname` hits I found in `dir3_manifold/PLAN.md` and
`dir4_cold_diffusion_steering/REPORT_3_external_validity.md` — other agents own those directions and
are editing them concurrently; the shared CLAUDE.md 8c/8d update is the right lever. Also left the
`\operatorname` occurrences in this direction's CHANGELOG.md/JOURNAL.md alone: those files are
append-only history and rule 7 forbids rewriting earlier entries. (d) Kept the CLAUDE.md edit additive
rather than also condensing 8a/8b, because three other agents share that file this run.

**Next step.** None outstanding — plan complete (S1–S8), all four feedback files addressed, both
deliverables render-verified, `STOP` re-written. If new `human_feedback*`/`*REVIEW*` arrives: delete
`STOP`, address it, run `python3 experiments/check_render.py REPORT.md RESULTS.md` before finishing,
re-STOP when clean. Untested scope limits if reopened: interpolation positions other than the final
token, and a second model.

On track? yes — plan COMPLETE (100%); this iteration fixed a GitHub-rendering regression that hid one
Methods equation and added a mechanical guard so it cannot recur silently; blocker: none.

## 2026-08-01 (12) — S9 / Experiment 5: all-pairs character sweep, and the rule-12 caption fix

**Did.** Checked for unaddressed feedback first (CLAUDE.md Part C): none — all four
`human_feedback*` files already carry `.addressed.md`. So the iteration is the reopened stage S9.
`STOP` was already absent, so nothing to delete.

1. **S9a — the sweep.** `experiments/allpairs_sweep.py`, built by generalising `comma_sweep.py`'s
   scaffold exactly as PLAN's Next-step said: same vocab loader, same `run_pair`, same context, same
   50-step grid, endpoint A generalised from the comma to all 65 characters. Ran all 2,080 pairs at
   interpolation block 0 of the step-30,000 checkpoint (~35 s), plus the step-0 init re-run, the
   200-pair depth subsample at blocks 4/8/11 and the 100-pair endpoint-swap check. Two minimal edits to
   `matthew_assay.py`: `run_pair` now also returns the per-`t` `argmax` (needed for the readout test),
   and the crossing interpolator was factored out as `iso_crossing` so `t*` uses the identical rule as
   `w`. `self_test()` passes unchanged; endpoint diagnostics on all 2,080 pairs are better than the
   PLAN gate demanded (max `d(0)` 3e-6 vs the 1e-3 bar).
2. **S9b — per-character verdict.** `analyze_allpairs.py`. Verdict is PLAN case **(i)**: every one of
   the 65 characters has a basin (flat_frac ≥ 0.86, =1.00 for 59), and the additive fit
   `w_ij ~ a_i + a_j` takes **78.2%** of the variance against a 3.0% permutation null, so case (iii)
   ("sharpness lives in the pair") is out. The honest qualifier is that the strict ≤0.25 rule is met by
   only 8.8% of pairs, so "basin" here means "the output rests on the endpoint", not "knife-edge step".
3. **S9c — mechanism and controls.** Readout-decision test came out stronger in its sharper form than
   in its naive one, which is worth recording: the naive version (`t*` vs the *first* argmax flip,
   "do paths visit exactly 2 argmax regions?") looks unimpressive — median path visits 3 predictions,
   only 32% visit 2. But asking *where* the flips fall gives 91% inside the transition window, 79% of
   pairs with all flips inside, 80% with single-prediction flat arms. So the plateau arms are constant-
   decision regions and the boundary is a short scramble, not a single flip. Both mandatory controls
   are decisive: init median width 0.803 with **0** strict plateaus (structure is learned, not
   architectural), and width 0.344 → 0.763 → 0.806 → 0.806 at blocks 0/4/8/11 (sharpness generated by
   blocks 1–4; the unembedding contributes none). Plausibility confound survives: partial ρ = −0.59
   both ways, so it is reported as the live alternative rather than dismissed.
4. **S9d — figures + hypothesis.** Six CVD-safe figures, plus the 4-sentence hypothesis in both
   deliverables.
5. **Rule-12 caption fix (not optional, and overdue).** Found both deliverables with 16 embeds and
   **1** visible caption — every caption was in alt text, which is exactly the failure `../CLAUDE.md`
   rule 12 names dir13 for, along with out-of-order numbering (REPORT.md ran 1, 1b, 1d, 2a, 2b, 6–11,
   3–5) and two unnumbered embeds. Rewrote both files: short alt text, visible `**Figure N.**` caption
   under every image, sequential numbering 1…22 in reading order, every figure cited by number and
   preceded by prose naming the claim it supports. Now 22 embeds / 22 captions per file.
   `check_render.py` exits 0 on both.

**Learned.** (a) The endpoint-swap symmetry check PLAN required is an *algebraic identity*, not an
empirical question: slerp is symmetric under (A↔B, t↔1−t) and the 50-point grid maps to itself, so
`d_swap(t) = 1 − d(1−t)` and `w` is exactly invariant. Getting median **and max** |Δw| = 0.000 over 100
pairs is therefore an implementation check, and I report it as such rather than as evidence about the
model — but it does license the symmetric heatmap. (b) Per-character effects dominate so heavily (78%)
that the width matrix shows visible stripes rather than a checkerboard; the figure and the statistic
agree, which is the cheapest sanity check available. (c) Choosing the right form of a preregistered
test matters: PLAN's phrasing ("do paths visit exactly 2 argmax regions?") would have read as a
*negative* result at 32%, while the same data answer the underlying question affirmatively at 91%. I
report both numbers so the weaker one is not hidden.

**Assumptions logged (loop mode, could not ask).** (i) `flat_frac` is evaluated at whichever endpoint
the character occupies (`t_lo ≥ 0.10` when it is A, `t_hi ≤ 0.90` when it is B), per PLAN §5.2 —
rejected alternative: requiring both margins, which would have conflated the basin question with the
strict plateau rule that `strict_frac` already measures. (ii) The additive fit is rank-deficient by one
(a constant can move between μ and the a_i), so I used the minimum-norm lstsq solution and reported the
design rank (65) alongside R²; rejected alternative: dropping a reference character, which changes the
coefficients but not R². (iii) I added a permutation null for R² because 65 free parameters on 2,080
points inflate R² by ~3% for free; without it "78%" would be unanchored. (iv) I fixed the rule-12
caption violation this iteration rather than deferring it: CLAUDE.md rule 12 says to re-embed and check
on *normal* iterations, the violation is named against this direction by name, and both deliverables
were being rewritten anyway.

**Next step.** S9 is complete (S9a–S9d all verified). PLAN's success criterion is now fully met, and no
unaddressed feedback file exists, so `STOP` is written. If the direction is reopened again, the
untested scope limits are unchanged: interpolation positions other than the final token, a second
model, and the unembedding-rebalancing intervention named in the hypothesis (the one experiment that
would separate the decision account from the plausibility account).

On track? yes — S9 100% done, plan COMPLETE; this iteration added the 2,080-pair sweep with a case-(i)
per-character verdict, a decision-basin hypothesis backed by four independent measurements, and fixed a
standing rule-12 violation that had left all 32 figure captions invisible; blocker: none.

## 2026-08-02 — operator feedback #4 (second file): the Figure-9 detector was wrong; two runs now PASS

**What I did.** Found `human_feedback_4.txt` unaddressed (note: a *different*, already-addressed
`human_feedback_4.addressed.md` exists from 2026-07-26; the new file reuses the number). Per CLAUDE.md
Part C this was the whole iteration.

The operator's diagnosis was right. `fig9_verdict.py` used `np.argmin(lc)` to find the "interior local
minimum", but the global LC minimum in all three runs is the **final** checkpoint — the bottom of the
second descent — so the code then looked for a rise *after the end of the run* and always concluded "no
second descent". The data had the pattern all along: fresh char 1940 → 491 @ step 15 → 769 @ 56 → 8.1;
pilot 1940 → 484 @ 19 → 1043 @ 33 → 68.

Rewrote the detector to walk the structure in order (first significant local min → its local max =
descent onset → sustained descent that does not rebound and ends below the first min), and added the two
preregistered checks that had never been implemented: onset before the clean-accuracy peak, and adv
robustness rising ≥ 0.05 from its value at the onset (with a *sustained* onset definition, so the
fresh char run's one-checkpoint adv blip at step 15 — 0.063 then back to 0.0006 — cannot count). Reran
on the existing JSONs only, as instructed; no training extended.

**Result.** Pilot char **FAIL → PASS**, fresh char **FAIL → PASS**, fresh BPE **FAIL** (unchanged: its
only upturn is 30 LC units = 1.4% of range, inside the preregistered 5% tolerance — I kept the frozen
tolerance rather than loosening it after seeing the result, and said so in the report).

**What I learned / judged.** (a) The rise is not marginal: 491.2 ± 2.7 → 769.4 ± 3.0 at 99% CI, ~90× the
CI and 2.9× the tolerance. (b) The consequence for the bounded verdict is *split*, not a wholesale flip:
Matthew's exact `big/in`/`big/large` are single tokens only under BPE, and BPE is exactly the run that
still fails — so the **primary** verdict stays PLAN case 5. The character analogues now sit on a passing
run, so their per-checkpoint evidence upgrades to **PLAN case 1 (temporally associated)**: `b↔i`/`b↔l`
width 0.80 → 0.33 between steps 56 and 831, inside the second-descent window and across the sustained
robustness onset. (c) I stated two honest caveats I could not remove: the LC turnaround happens at steps
15–56, far earlier than the paper's, so its window overlaps ordinary initial fitting and the association
cannot separate "sharpens with grokking" from "sharpens with initial fit"; and the local maximum is
resolved by a single log-spaced checkpoint in each run.

**Assumption logged (loop mode, no human to ask).** The 5% tolerance was preregistered before any curve
was seen, so I did not retune it to let BPE through — rejected alternative: a 1% tolerance, which would
flip BPE to PASS purely by post-hoc choice. The BPE near-miss is reported explicitly with its number.

**Deliverables.** Both files curated to current-best (Summary/headline, Methods verdict rule as
equations, gate table with the new landmark rows, Figures 2–5 captions, joint-timeline paragraph,
Conclusion, Limitation 4). Four PNGs regenerated: the three gate curves now annotate the detected
landmarks and print the verdict; the joint timeline panel now reads PASS / PASS / FAIL. CHANGELOG has
the old → new numbers. `check_render.py` could not run (`node` missing this session) — ran its four
checks individually instead, all clean (20/20 display equations, 0 code-block degradations, 22 embeds =
22 visible captions per file, no inline-`\_` hazards, no bare plot paths).

**Next step.** None outstanding; feedback renamed to `human_feedback_4.txt.addressed.md`. If reopened:
the sharpest follow-up is a denser checkpoint grid over steps 10–1000 to resolve the LC local maximum
with more than one point, and a longer run whose second descent is separated from initial fit.

On track? yes — S1–S9 complete plus feedback #4 (second file) addressed; grokking gate corrected, 100% done, no blocker.

## 2026-08-02 (later) — resolved the LC local maximum with a denser grid; PASS survives and strengthens

**What I did.** No unaddressed feedback (all five `human_feedback*` files end in `.addressed.md`), so I
advanced PLAN's own named next step: densify the fresh character run's Figure-9 grid so its LC local
maximum rests on more than one measured checkpoint. The key realisation was that this needed **no
training** — the run had saved 35 checkpoints but only 14 had ever been evaluated, so 10 unmeasured
checkpoints in the interesting window (steps 1–531) were sitting on disk. Ran `fig9.py` on them with
everything else frozen (same eval points, same `r`/`P`/`ε`, same detector), taking the grid to 24
points, then reran `fig9_verdict.py` and regenerated `grokking_fresh_char.png` + `joint_timeline.png`.

**Result.** The turnaround is real and was *understated* by the sparse grid. LC 491.2 @ 15 → 987.7 @ 23
→ 989.1 @ 36 → 769.4 @ 56 → … → 8.1: three points now sit above the first minimum where one did. The
detected local maximum moves 769 @ 56 → **989 @ 36**, the rise grows 278 → **498** units (2.9× → 5.1×
the frozen tolerance, ~110× the CI), and the verdict stays **PASS**. Densifying also moved the sustained
robustness onset 831 → **531**, because step 531 (adv 0.077) had never been measured. Pilot char (PASS)
and BPE (FAIL) are untouched.

**Two obstacles, both worth recording.** (a) `/tmp` had been cleared between sessions, so every script's
hard-coded `/tmp/tinyshakespeare.txt` was gone; I re-fetched it and verified `sha256` against the
`corpus_sha256` stored in `train_meta_grok_char.json` before trusting it — the character vocabulary is
derived from this file, so a different copy would have silently renumbered every token. (b) `fig9.py`'s
default `--vram_frac 0.05` (1.57 GiB) now OOMs. CLAUDE.md's rule is "halve the batch on OOM", but here
halving `--pgd_bs` would have changed how the PGD random-start generator is consumed and made the 10 new
adversarial numbers non-comparable to the existing 14; raising the fraction to the BUDGET-allowed 0.225
leaves the computation bit-identical. I chose comparability and logged the deviation.

**What I learned / judged.** (i) A landmark defined by one measured point is the weakest link in a gate
verdict, and it was cheap to remove here — worth checking for saved-but-unevaluated checkpoints before
assuming a denser grid means retraining. (ii) The consequence for the bounded verdict is a restatement,
not a change: primary stays **PLAN case 5** (BPE still FAILs, and only BPE carries Matthew's exact
tokens), character analogues stay **PLAN case 1**, but the association window is now 36 → 30,000 with
robustness onset 531, so the plateau's sharpening interval (56 → 831) *strictly contains* the onset
rather than ending at it — marginally cleaner than before. (iii) The remaining caveat is scoped
correctly now: the *pilot* run's local maximum is still a single checkpoint; the fresh run's is not.
(iv) I did not re-run the plateau assay at the new checkpoints: the six plateau phases were frozen
before any plateau curve was inspected, and re-picking them from the updated landmarks after seeing the
results would break that preregistration for no evidential gain.

**Deliverables.** RESULTS.md and REPORT.md curated to current-best everywhere the fresh-char gate
appears (summary, gate table + new "points resolving the LC local maximum" row, gate paragraph, Figure 3
caption, joint timeline, Conclusion, Limitation 4), REPORT.md Methods now documents the per-run
checkpoint grid, and CHANGELOG has the old → new table. `node` is still unavailable so `check_render.py`
cannot run; I ran its four checks individually instead.

**Next step.** Plan remains complete. The sharpest untested items are unchanged and all need new
compute: a longer run whose second descent separates from initial fit, a denser grid on the *pilot*
run's local maximum, interpolation positions other than the final token, a second model, and the
unembedding-rebalancing intervention that would separate the decision account from the plausibility
account.

On track? yes — plan complete (S1–S9) plus PLAN's named follow-up done; fresh-char gate now rests on a
24-point grid with the local maximum resolved by three checkpoints; blocker: none.

## 2026-08-02 (later still) — ran the readout-rebalancing intervention; the plateau turns out to be readout-invariant

**What I did.** No unaddressed feedback (all five `human_feedback*` files end in `.addressed.md`), so I
took PLAN's own named next step: the unembedding-rebalancing intervention, the one cheap experiment
that could separate the "decision" account of the character basins from the "plausibility" account.
Added an opt-in `return_logits` argument to `matthew_assay.run_pair` (default off, so every existing
result is untouched), wrote `experiments/rebalance_probe.py` and `experiments/plot_rebalance.py`, and
ran all 2,080 pairs at block 0 of the step-30000 checkpoint in 33 s.

**Design choice worth recording.** I defined the intervention on the *endpoint predictions* `a*`, `b*`
(the argmax at `t=0` and `t=1`), not on the endpoint characters A, B. The path patches the residual at
the final position of "The house was A", so the readout emits the character that follows A — biasing
rows A and B would not touch the decision the path actually makes. The two bias sizes (equalise the
endpoint predictions; force the boundary to the midpoint) were fixed before I looked at any output.

**The result, and the surprise.** I expected to compare a moved `t*` against an unmoved width. Instead
the intervention is *algebraically* null on `d(t)`: `d(t)` is a ratio of distances between logit
vectors, so a common additive bias cancels exactly — verified numerically at 1.3e-6. So no readout bias
of any size can change the width or `t*`. That is a real finding rather than a failed test, but it also
means PLAN's stated prediction for the plausibility account ("the width itself changes") was not
testable this way, and I said so in both deliverables instead of quietly dropping it. The empirical
half is the interesting one: the readout gap swings ~21.9 nats across the path, so the decision
boundary is very stiff — 2.44 nats moves it 0.020, and 5.28 nats (enough to put it at the midpoint)
moves it 0.052. Boundary and plateau midpoint therefore stay glued together (median |t*−t_gap| 0.025 →
0.015 → 0.035) no matter what I do to the readout.

**What I learned.** The tight `t* ≈ t_flip` alignment from S9 was being read as support for "the
decision creates the plateau". This intervention shows the arrow points the other way: both the flip
and the `d(t)` transition are downstream of one sharp residual-stream change built by blocks 1–4, and
the unembedding just reads it steeply. The S9 hypothesis sentence survives, but as a description of the
basins rather than as their mechanism — and the plausibility alternative is now pushed upstream too
(it can only act through the weights of blocks 1–11), which is a sharper statement than "not ruled
out".

**Deliverables.** New Results subsection in RESULTS.md and REPORT.md, a new Methods block in REPORT.md
defining `g(t)`, `t_gap`, `c_eq`, `c_half` with column-0 ```math fences, and Figure 20 embedded in both
with a visible caption; the three exploratory figures renumbered 21–23 to keep numbering sequential.
23 embeds and 23 `**Figure N.**` captions in each file; no bare `(plots/…)` paths, no inline-math
backslash hazards, no `\operatorname` outside code spans. `node` is still absent on this pod so
`check_render.py` cannot run its KaTeX stage; I ran its grep-level checks individually as before.

**Next step.** Everything still open needs new compute: a longer run whose second descent separates
from initial fit; the denser grid applied to the *pilot* run's local maximum; interpolation positions
other than the final token; a second model. The natural successor to today's intervention is the one
it could not perform — perturb *inside* blocks 1–4 (e.g. scale the block-1..4 MLP outputs) and see
whether the width moves, which is what would test the plausibility account properly.

On track? yes — plan complete (S1–S9) plus two PLAN-named follow-ups now done (denser Figure-9 grid,
readout-rebalancing intervention); blocker: none.

## 2026-08-02 (iteration: MLP-gain intervention)

No unaddressed `human_feedback*`/`*REVIEW*` files (all five end in `.addressed.md`), so this iteration
advanced the follow-up the previous entry named: perturb *inside* blocks 1–4 and see whether the
plateau width moves. `experiments/mlp_gain_probe.py` wraps a block's MLP so its residual-branch output
is scaled by `g` — attention, LayerNorm and all other blocks untouched — and re-runs the frozen
`run_pair` assay with endpoints recomputed under the modified model, so the assay itself is unchanged
and `d(t)` always measures the modified model's own path. Ran the early group (blocks 1–4) and the
late group (blocks 8–11) at g = 0 / 0.5 / 1.5 against the unmodified model, on a fixed 150-pair random
subsample at block 0 of the step-30000 checkpoint. 17 s on GPU at `vram_frac 0.225`, 2 threads.

**Result — the cleanest causal statement in this direction so far.** Deleting the early MLPs returns
the width to the untrained value (median 0.351 → **0.796**, vs 0.803 at step 0) with 0/150 strict
plateaus and *every* pair widening; halving them gives 0.533; amplifying them to g = 1.5 sharpens to
0.305 and triples the strict rate (10% → 30%). The dose–response is monotone across all four gains.
The same gains on blocks 8–11 move the median width by ≤ 0.043 (median paired |Δw| ≤ 0.025) — deleting
four whole top-of-stack MLPs barely registers. `t*` stays put (|Δt*| ≤ 0.074), so the intervention
changes sharpness, not location.

**What I learned.** Experiment 5's depth control was an observation about where the patch is injected;
this is an intervention on the model, and it agrees. Combined with the readout probe (no bias of any
size can move `d(t)`), the picture is: blocks 1–4 build the sharp residual-stream change, everything
downstream reads it. The plausibility alternative is now pinned to those same early weights rather
than merely "somewhere in blocks 1–11" — sharper, still alive: this experiment cannot distinguish a
decision-shaped basin from a plausibility-shaped basin that the readout thresholds.

**Deliverables.** New subsection + Figure 21 in both RESULTS.md and REPORT.md, a Methods block in
REPORT.md defining the gained-block update at column 0, and one sentence added to the REPORT Summary
and Conclusion. Exploratory figures renumbered 21–23 → 22–24; 24 embeds and 24 `**Figure N.**`
captions in each file, 0 bare `(plots/…)` paths, 0 inline-math backslash hazards, `\operatorname` only
inside code spans. `node` is still missing on this pod so `check_render.py`'s KaTeX stage cannot run;
its grep-level checks were run individually, as in the previous iterations.

**Next step.** Still open and all needing new compute: a longer character run whose second descent
separates from initial fit; the denser Figure-9 grid applied to the *pilot* run's local maximum;
interpolation at positions other than the final token; a second model. The natural successor to today's
result is to separate the two remaining accounts *inside* blocks 1–4 — e.g. ablate the MLPs
block-by-block (which of 1,2,3,4 carries it?) and test whether the width change tracks the endpoint
plausibility gap or the decision structure.

On track? yes — plan complete (S1–S9) plus three PLAN-named follow-ups done (denser Figure-9 grid,
readout rebalancing, MLP-gain intervention); blocker: none.

## 2026-08-02 (iteration: per-block MLP scan + first full render check)

No unaddressed `human_feedback*`/`*REVIEW*` files (all five end in `.addressed.md`), so this iteration
took PLAN's named successor to the MLP-gain result: ablate blocks 1–4's MLPs **one at a time** and test
whether the width change tracks the endpoint plausibility gap or the decision structure.

**Found work already half-done.** `experiments/mlp_block_scan.py` and its summary JSON existed
uncommitted from the previous iteration — the ablation had run but was never analysed, plotted, or
reported. So the useful step was to finish it rather than restart it.

**A definition error caught before it reached the deliverables.** My first pass measured the
plausibility mediator as `|log p(A) − log p(B)|` and got ρ ≈ +0.09 on the baseline, which flatly
contradicts the −0.59 the report attributes to Experiment 5. Rather than write that up as a
discrepancy I checked `analyze_allpairs.py`: the −0.59 is the **partial** Spearman of `w` against
**max(p(A), p(B))** controlling for endpoint logit separation — `|Δlog p|` is a different, weakly
positive (+0.18) quantity that PLAN 5.3 also names. Re-ran with the frozen definition and the baseline
reproduces at −0.634 on the 150-pair subsample. Worth recording as a habit: when a new measurement
disagrees with a reported one, first check that it is the *same* measurement.

**Result — the cleanest negative in this direction.** (a) *Distributed, front-loaded:* single-block
deletions give median widths 0.541 / 0.478 / 0.446 / 0.402 for blocks 1/2/3/4 against 0.351 unmodified
and 0.796 for all four — shares 41 / 28 / 18 / 11% of the group effect, monotone in depth and summing
to 98%, so no single block carries it and the four are close to additive. (b) *Not plausibility:* the
partial ρ(w, max_p | sep) survives every ablation (−0.45 to −0.64), so plausibility still predicts
*which* pairs are sharp — but it does not mediate the intervention (ρ(Δw, Δmax_p) = +0.22 at most,
median |Δmax_p| ≤ 0.0007 against Δw = +0.433), and where it does move it moves the wrong way (median
max_p rises 0.0034 → 0.0136 under the ablation, the direction that predicts *narrower* plateaus).
(c) *Not the decision:* 80.7% of pairs still predict different characters at their endpoints after all
four MLPs are deleted (86.7% unmodified) and the median number of `argmax` regions is unchanged at 3,
yet `d(t)` is a straight line — and `|t* − t_flip|` decouples 0.043 → 0.214.

**What I learned.** Both surviving accounts of the character basins are now downstream readings of the
geometry rather than its cause: you can keep the decision and lose the plateau, and you can lose the
plateau without touching the plausibility landscape. So "a plateau is the set of states that decode to
the same prediction" is right as a *description* and wrong as a *mechanism* — I demoted it explicitly
in both hypothesis paragraphs and gave them the falsifiable prediction PLAN 5.5 asks for (freeze
blocks 1–4 at step-0 weights, train the rest to the same validation accuracy, expect straight paths).
The honest remaining gap is that what blocks 1–4 *compute* is still uncharacterised; ruling out two
candidates is not identifying the mechanism.

**`check_render.py` finally ran.** `node` is present on this pod for the first time in several
iterations, so the KaTeX stage executed instead of being replaced by hand-run greps: **ALL CHECKS
PASS** (REPORT 26 display / 321 inline eqs / 25 figures; RESULTS 25 figures; 0 problems). Separately
the rule-8b grep caught two `\,` thin spaces I had just written inside inline `$…$` — KaTeX compiles
them happily, but GitHub strips the backslash and renders a stray comma. Both fixed. That is exactly
the failure mode CLAUDE.md 8b describes as silent, and it confirms the grep earns its place alongside
the compile check.

**Deliverables.** New subsection + **Figure 22** in RESULTS.md and REPORT.md, a REPORT Methods block
defining the per-block share `F_l`, the plausibility mediator and its mediation correlation, and the
three decision descriptors; Summary, Conclusion, both hypothesis paragraphs and Limitation 6 curated to
current-best; exploratory figures renumbered 22–24 → 23–25 (25 embeds and 25 captions per file).

**Next step.** Everything still open needs new compute: a longer character run whose second descent
separates from initial fit; the denser Figure-9 grid applied to the *pilot* run's local maximum;
interpolation at positions other than the final token; a second model. The direct successor to today's
result is the falsifiable prediction it ends on — freeze blocks 1–4 at initialization, train the rest
of the network to matched validation accuracy, and check whether the paths stay straight. That is a
training run (~30k steps), so it is the first follow-up here that is not a cheap re-analysis.

On track? yes — plan complete (S1–S9) plus four PLAN-named follow-ups done (denser Figure-9 grid,
readout rebalancing, MLP-gain intervention, per-block scan); blocker: none.

## 2026-08-02 (iteration: frozen-block training test — S10 finished, prediction falsified)

No unaddressed `human_feedback*`/`*REVIEW*` files (all five end in `.addressed.md`). I found the
previous iteration had been cut off mid-S10: `train_frozen.py`, `frozen_assay.py`, `plot_frozen.py` and
a REPORT Methods block existed, PLAN's S10 was already ticked, but the two training runs were still at
~7k/30k steps and neither deliverable had the Results subsection or Figure 23. So this iteration was
"finish what is running", not "start something new".

**Process note worth recording.** My first attempt to chain the assay behind the trainings used
`nohup bash -c '...' &` inside a Bash tool call; the detached shell died with its parent when the tool
call returned, so 50 minutes later `/tmp/chain_frozen.log` was still empty and no assay had run. The
trainings themselves survived (they were launched the same way by the previous iteration and had been
running for an hour) — so the failure mode is not "nohup never works" but "a wrapper whose only job is
to wait can be reaped before its child starts". Waiting in the foreground in ~9-minute Bash calls was
reliable and cost nothing. Both runs finished 30,000 steps cleanly.

**The result is a clean falsification of our own prediction, which is the point of having written it
down.** The hypothesis paragraph in both deliverables ended on: freeze blocks 1–4 at initialization,
train the rest to matched accuracy, expect straight paths (width ≈ 0.80). Outcome: median width
**0.471** — 73% of the reference run's sharpening recovered — at *better* validation accuracy than the
reference (0.5625 vs 0.5502). The specificity control decides the reading: freezing blocks 8–11, which
the ablations said contribute nothing, costs the same width (0.484, paired Δw +0.120 vs +0.107). If the
early group were special at training time, the two controls would have separated; they did not. So the
0.11 shortfall is a generic capacity cost of freezing a third of the stack.

**The depth control is what makes it more than a null.** Injecting the interpolated activation at
blocks 0/4/8 gives 0.351/0.761/0.805 for the trained reference (sharpening in blocks 1–4) and
0.484/0.793/0.806 for frozen-late — same profile. Frozen-early gives 0.471/**0.471**/0.788: injecting
after the frozen group changes the width by exactly 0.000, so those blocks contribute none of the
sharpening and all of it has moved to blocks 5–7. The computation relocated rather than disappearing.

**What I learned.** Three iterations of ablation had built toward "blocks 1–4 build the sharpness".
That statement is still true of the trained network at inference — deleting their MLPs flattens `d(t)`
completely — but it is false as a claim about training. Ablation necessity and training-time necessity
are different claims, and only the retraining test separates them. The honest summary now is that the
plateau is a robust, relocatable product of this architecture + objective, and what the responsible
blocks actually compute is still uncharacterised — three candidate mechanisms (the decision,
plausibility, those particular weights) have now been excluded in turn.

**Deliverables.** New subsection + **Figure 23** in RESULTS.md and REPORT.md; one added sentence in the
REPORT Methods block for the depth control; REPORT Summary, REPORT Conclusion, RESULTS verdict item 5
and RESULTS Headline curated to current best; both hypothesis paragraphs rewritten to report the
falsification and to end on a new falsifiable prediction (freeze blocks 1–7, train only the top of the
stack — sharpening should reappear between injection blocks 8 and 11). REPORT Limitation 6 updated,
Limitation 7 added. 26 embeds and 26 `**Figure N.**` captions per file, numbering sequential 1–26.
`check_render.py` → **ALL CHECKS PASS**.

**Next step.** The new prediction above is the cheapest meaningful follow-up (one more ~46-minute
training run, same harness, `--freeze 1,2,3,4,5,6,7`). Everything else still open needs new compute or
a new model: a longer character run whose second descent separates from initial fit; the denser
Figure-9 grid applied to the pilot run's local maximum; interpolation at positions other than the final
token; a second model/tokenizer. No `STOP` written — a follow-up operator request is still plausible
and a STOP'd direction would silently ignore it (CLAUDE.md rule 11).

On track? yes — plan complete (S1–S10), five PLAN-named follow-ups done, the hypothesis's own
falsifiable prediction tested and reported as falsified; blocker: none.

**Addendum (19:55).** I pre-launched the follow-up this entry names — `train_frozen.py --freeze
1,2,3,4,5,6,7 --tag frozen_deep --steps 30000 --max_minutes 70` — writing to
`/tmp/dir13_frozen/checkpoints_frozen_deep/` and `/tmp/dir13_frozen/train_frozen_deep.log`
(~46 min, `vram_frac 0.11`, 1 thread, so it does not crowd the other agents). It is a *pre-fetch*, not
a claimed result: nothing about it is in the deliverables. The next iteration should check the log,
and if the run finished, add `frozen_deep` to `frozen_assay.py`'s condition list (the loop already
looks for `ckpt_matched.pt` / `ckpt_last.pt` per tag) with the depth control at injection blocks 8/10/11
as well as 0/4/8, since the prediction is about where the drop appears above block 8. If feedback
arrives instead, feedback comes first — the run costs nothing to ignore or kill.

## 2026-08-02 (iteration: S11 deep-freeze training test — the relocation prediction confirmed)

No unaddressed `human_feedback*`/`*REVIEW*` files (all five end in `.addressed.md`), so this iteration
went straight to the plan. The previous iteration had pre-launched
`train_frozen.py --freeze 1,2,3,4,5,6,7 --tag frozen_deep` and had already edited `frozen_assay.py` to
know about the condition and to extend the injection-depth grid to 0/4/8/10/11, so the work here was:
wait for the run, chain the assay behind it, and curate.

**Process note that worked this time.** Last iteration's `nohup bash -c '…' &` chain was reaped before
its child started. This time I used the harness's own `run_in_background` Bash with
`while ps -p <pid>; do sleep 30; done; python3 frozen_assay.py; python3 plot_frozen.py` — it survived,
notified me on completion, and cost nothing. That is the right pattern for "run B when A finishes"
here; a detached `nohup` wrapper whose only job is to wait is not.

**Result — for once the prediction on record was confirmed, and it is the more informative outcome.**
With blocks 1–7 (58% of the stack) held at their step-0 weights, the network still reaches the best
validation accuracy of any run in this direction (0.5742 vs the reference's 0.5502) and still builds
plateaus: median width 0.558 against 0.803 untrained, narrower than untrained for 149/150 pairs. The
depth profile is the part that decides it — 0.558 / 0.557 / 0.695 / 0.767 / 0.805 at injection blocks
0/4/8/10/11. The frozen blocks 1–4 contribute −0.002 (they do nothing, as they must), and the whole
0.248 of sharpening lives in the four trainable blocks, rising monotonically as the injection point
descends from 11 to 8. Exactly the predicted signature.

**The cross-run pattern is the real finding.** Median width 0.351 (nothing frozen) → 0.471 / 0.484
(four frozen, either end) → 0.558 (seven frozen). Paired Wilcoxon puts deep above both four-block runs
(+0.073 and +0.064, p ~ 1e-17) while the two four-block runs differ by 0.015. So the cost of freezing
is a function of *how many* blocks are frozen, not which — the sharp transition is not just relocatable
but relocatable at a predictable price. Together with S10 this closes the "blocks 1–4 are special"
thread: they are special at inference in the trained network and not at all at training time.

**An off-by-one I found while writing this up.** `matthew_assay.run_pair` patches `resid_post` of the
interpolation block, so the width drop between injection points b1 < b2 is produced by blocks
b1+1 … b2. S10 had reported frozen-early's relocation as "blocks 5–7" from the 4 → 8 drop; the correct
span is **5–8**. Corrected everywhere in both deliverables and logged in CHANGELOG. The numbers were
never wrong, only the block label attached to them — but it is exactly the sort of thing that would
have propagated into the next hypothesis.

**What I learned.** Three iterations ago the honest summary was "blocks 1–4 build the sharpness". Two
retraining runs later the summary is "any four contiguous blocks below the readout will do, and losing
three more only widens the transition by ~0.09". Ablation necessity, training-time necessity, and
depth specificity are three separable claims, and only retraining separates them. What the responsible
blocks actually *compute* is still uncharacterised — that gap has not moved.

**Deliverables.** Both frozen-block subsections rewritten around a three-run prediction/outcome table;
Figure 23 re-rendered with six top-row panels and the five-depth injection profile, caption updated in
both files; REPORT Methods gained the third run, the extended depth grid and the span-attribution rule;
REPORT Summary/Conclusion/Limitations 6–7, RESULTS Headline and RESULTS verdict item 5 curated; both
hypothesis paragraphs rewritten and re-ended on a new falsifiable prediction (freeze blocks 5–11, the
mirror image — same count frozen, trainable capacity at the bottom). 26 embeds / 26 captions per file;
`check_render.py` → ALL CHECKS PASS.

**Next step.** The mirror-image run above (`--freeze 5,6,7,8,9,10,11`, ~21 min at the rate frozen_deep
managed) is the direct successor and would settle count-vs-depth outright. Everything else still open
needs new compute or a new model: a longer character run whose second descent separates from initial
fit; the denser Figure-9 grid applied to the *pilot* run's local maximum; interpolation at positions
other than the final token; a second model/tokenizer. No `STOP` written — a follow-up operator request
remains plausible and a STOP'd direction would silently ignore it (CLAUDE.md rule 11).

On track? yes — plan complete (S1–S11), six PLAN-named follow-ups done, the successor prediction tested
and confirmed; blocker: none.

**Addendum (S12, same iteration).** S11 finished with ~90 minutes left, so I ran its own successor
rather than pre-launching it for next time: `train_frozen.py --freeze 5,6,7,8,9,10,11` — the mirror
image of frozen_deep, same 58.0% of parameters frozen, same five trainable blocks, moved from the top
of the stack to the bottom. 21 minutes of training, 79 seconds of assay.

**The result is the most useful kind: half the prediction confirmed, half falsified.** The *location*
half landed exactly — the depth profile is 0.626 / 0.764 / 0.805 / 0.806 / 0.806 / 0.806 at injection
blocks 0/2/4/8/10/11, so injecting at block 4 already returns the untrained straight line and every bit
of sharpening is back in blocks 1–4. That is the third distinct site across four runs (5–8, 8–11, 1–4),
which retires "the sharpening lives in a particular place" completely. The *magnitude* half failed:
0.626, not the predicted ~0.558, with a paired median +0.063 over frozen_deep (81% of pairs,
p = 6e-17), on runs whose final validation accuracies agree to 0.0002. So the S11 summary I had just
written — "the cost tracks how many blocks are frozen, not which" — is too strong, and I replaced it
with the two-term reading the four runs actually support: trainable depth first (0.351 → 0.47 → 0.56–
0.63 for 12, 8, 5 trainable blocks), position second, and position only matters once depth is scarce
(worth 0.015 at eight trainable blocks, 0.068 at five, favouring the readout end).

**Worth recording as a methods point.** I only got a clean answer because frozen_deep and frozen_mirror
are matched on everything a capacity story can see — frozen parameter fraction, trainable block count,
final accuracy. S11's three runs confounded count with position; one extra run at 21 minutes
de-confounded them. Cheap matched controls beat more conditions.

**Also fixed while here.** Adding injection block 2 to the depth grid (needed to resolve the drop
*inside* frozen_mirror's trainable group) incidentally sharpened the reference profile: 0.351 → 0.646
at block 2 → 0.761 at block 4, i.e. the reference's sharpening is front-loaded into blocks 1–2, which
independently matches the per-block MLP scan's 41/28/18/11% shares. `plot_frozen.py` is now generic in
the number of frozen runs (it lays out one top-row panel per condition present) rather than hard-coded
to six panels.

**Next step (revised).** The hypothesis paragraphs now end on: freeze ten blocks, train only block 0 and
block 11. If trainable depth is the first-order term it should land near 0.70 with its residual drop
split between injection blocks 0→2 and 10→11; if one trainable block beside the readout suffices it
should land near 0.56. ~21 min, same harness, `frozen_assay.py` needs one more condition entry.
Everything else open still needs a longer run, a second model, or interpolation at non-final positions.

On track? yes — plan complete (S1–S12), seven PLAN-named follow-ups done, two predictions tested this
iteration (one confirmed, one split); blocker: none.

## 2026-08-02 (iteration: S13 two-block freeze — trainable depth confirmed as the first-order term)

No unaddressed `human_feedback*`/`*REVIEW*` files (all five end in `.addressed.md`), so this iteration
went straight to the plan. The previous iteration had pre-launched
`train_frozen.py --freeze 1,...,10 --tag frozen_two` and `frozen_assay.py`/`plot_frozen.py` already
knew about the condition, so the work here was: wait for the run, chain the assay, curate.

**A process bug that cost ~5 minutes and is worth not repeating.** I chained the assay behind training
with `while pgrep -f "train_frozen.py --freeze" > /dev/null; do sleep 20; done; python3 …`. That never
fires: the waiter's *own* bash command line contains the string `train_frozen.py --freeze`, so `pgrep
-f` matches the waiter itself and the loop is infinite. Worse, my status checks used
`pgrep -f "frozen_assay.py"`, which matched the same waiter's command line and cheerfully reported
"ASSAY RUNNING" for five minutes while nothing was running at all. Training had actually finished at
21:12. The tell was `nvidia-smi --query-compute-apps` showing no process on the GPU — a liveness check
should look at the *resource*, not at a pattern that can match the checker. S11's note ("use
`ps -p <pid>`, not a detached nohup") was right for a different reason than I applied it: the real rule
is **never `pgrep -f` a pattern that appears in your own command line**. Once diagnosed, I killed both
self-matching waiters and ran the assay directly.

**The result confirms the prediction on record, and adds the boundary condition four runs of
relocation had been missing.** With blocks 1–10 frozen (82.9% of parameters, only blocks 0 and 11
trainable) the network still beats the reference on the task (0.5668 vs 0.5502) but median width lands
at **0.726** — the ≈0.70 that trainable depth predicts, not the ≈0.56 that "one trainable block beside
the readout suffices" predicts. Paired: +0.160 vs frozen_deep (97% of pairs, p = 7e-26), +0.094 vs
frozen_mirror (89%, p = 3e-21). The cross-run series is now monotone in trainable depth —
0.351 (12) → 0.471/0.484 (8) → 0.558/0.626 (5) → 0.726 (1 usable) — which is as clean as this design
gets with one seed per condition.

**The more interesting half is where the prediction was structurally impossible.** I had written that
the residual drop would split between injection blocks 0→2 and 10→11. It cannot: `run_pair` injects at
`resid_post` of the interpolation block, so injecting at block 0 *overwrites* block 0's output and its
trainable weights are invisible to the measurement. Block 11 is the only trainable block downstream.
The measured profile (0.726/0.725/0.724/0.725/0.725/0.803) says exactly that — all 0.077 of sharpening
in block 11 alone. So the run measures "one usable block", not "two trainable blocks", and I have
labelled it that way throughout rather than quietly restating the prediction as if it had been about
one block all along. Same class of error as S11's off-by-one on span attribution: the injection
semantics are easy to state and easy to forget when writing the *next* prediction.

**What actually changed in the reading.** Four runs had supported "the site is contingent, only the
sharpness costs anything". Frozen-two is the first condition where the sharpness cost becomes a
qualitative failure: 26% of its pairs are *wider* than the untrained network's (0–1% in the other four
runs), the boundary comes unstuck from the prediction flip (|t*−t_flip| 0.146 vs 0.043), the
plausibility association mostly collapses (partial ρ −0.18 vs −0.63), and no pair passes the strict
rule. It is also the only run that needed materially longer to reach reference accuracy (step 7000 vs
2500–3000). So the honest summary gained a floor: the plateau relocates freely in *site*, its sharpness
scales with trainable depth, and below about one usable block there is no plateau left to relocate.

**Deliverables.** Five-run prediction/outcome table in both frozen-block subsections; new
bullet/paragraph on the two-block run; Figure 23 re-rendered with eight top-row panels and six
injection curves, caption updated in both files; the bottom-left panel title fixed because the new
condition made "…does not prevent it" an overclaim; REPORT Summary/Conclusion/Limitations 6–7, both
hypothesis paragraphs, RESULTS Headline and verdict item 5 curated. REPORT Methods already defined the
fifth run from last iteration, so it needed no edit. New helper `experiments/frozen_pairwise.py` for
between-run paired Wilcoxon shifts, validated against S12's published numbers before use. 26 embeds /
26 captions per file; `check_render.py` → ALL CHECKS PASS.

**Next step.** Frozen-two confounds trainable depth with parameter count (82.9% frozen *and* one usable
block), which is the one loose end this run creates. S14, now the prediction both hypothesis paragraphs
end on, cuts capacity while holding depth fixed: retrain at `n_embd` 192 with nothing frozen. Depth
account → ≈0.35 like the reference; parameter-count account → ≈0.47. ~20 min on this harness.
Everything else still open needs a longer character run whose second descent separates from initial
fit, the denser Figure-9 grid on the pilot run's local maximum, interpolation at non-final positions,
or a second model/tokenizer. No `STOP` written — a follow-up operator request remains plausible and a
STOP'd direction would silently ignore it (CLAUDE.md rule 11).

On track? yes — plan complete (S1–S13), eight PLAN-named follow-ups done, this iteration's prediction
tested and confirmed with its impossible half diagnosed and reported; blocker: none.

## 2026-08-02 (iteration: S14 narrow run — the depth/capacity confound broken)

No unaddressed `human_feedback*`/`*REVIEW*` files (all five end in `.addressed.md`), so this went
straight to the open plan item: S14, the prediction both hypothesis paragraphs ended on — hold trainable
depth fixed and cut capacity instead, by retraining at `n_embd` 192 with nothing frozen.

**Why this was the right next run.** Five frozen runs had produced a clean monotone series in trainable
depth (0.351 → 0.471/0.484 → 0.558/0.626 → 0.726), but every one of them cut blocks and parameters
*together*. "Width tracks trainable depth" and "width tracks trainable capacity" fit all five equally
well, and the whole relocatability story rests on which it is. The narrow run is the matched control:
5,584,896 parameters against frozen-early's 5,601,360 trainable ones — 0.3% apart, essentially a
coincidence I checked only after training started — with all 12 blocks trainable.

**Result: the depth account wins outright.** Median width **0.397** at matched accuracy, against ≈0.47
predicted by capacity and ≈0.35–0.44 by depth. Paired against the identical 150 pairs it is 0.073
narrower than frozen-early (p = 2.5e-15) and 0.092 narrower than frozen-late (p = 1.8e-19); against the
reference at *its* matched-accuracy step it is 0.014 **sharper** (p = 1.9e-4). Removing a third of the
parameters cost nothing measurable; removing a third of the trainable blocks costs 0.11–0.12. It also
kept the reference's front-loaded depth profile and, unlike all five frozen runs, the sharpest tail
(13.3% of pairs meet the strict rule vs 0–0.7%). So narrowing is not a mild freeze — on this measure it
is not a perturbation at all.

**A time-management decision worth recording.** With ~28 minutes left and training needing ~18 for the
full 30,000 steps, I did not wait for `ckpt_last.pt`. Instead I assayed `ckpt_matched.pt` — which had
already been written at step 2,750 — and framed the whole comparison at **matched validation accuracy**,
which is the frozen series' primary framing anyway and the only apples-to-apples axis across runs of
different capacity. I also wrote `narrow_assay.py` to score the one new condition rather than re-running
`frozen_assay.py` over all eight: 70 seconds instead of ~8 minutes, same functions, same pairs, merged
into the same output files. The rejected alternative was killing training early and calling the last
periodic checkpoint "final", which would have introduced a truncated-cosine-schedule confound into the
one run whose job is to remove a confound. Training was left running; its `_last` row can be added later
by re-running `narrow_assay.py`, which is idempotent per condition key.

**Deliverables.** New bullet + Figure 24 (`capacity_vs_depth.png`, median width vs trainable blocks and
vs trainable parameters, CVD-safe: filled circles = all-12-trainable, open diamonds = frozen) in
RESULTS.md; matching Methods paragraph with the two accounts' point predictions as a fenced equation,
Results paragraph and the same embedded figure in REPORT.md. Exploratory Figures 24–26 renumbered 25–27
so reading order holds. Figure 23 deliberately left alone — a seventh series would break the five-colour
CVD palette, and the narrow run is not a frozen-block run.

**Next step.** Two loose ends this run creates. (1) Re-run `narrow_assay.py` once
`checkpoints_narrow192/ckpt_last.pt` exists to add the fully-trained row; the matched row is the load-
bearing one, but the final-step comparison against ref_trained's 0.351 would complete the table.
(2) The depth series still has one seed per condition; a second seed at `n_embd` 192 and at one frozen
condition would put an error bar on the 0.397 vs 0.476 gap. Everything else still open needs a longer
character run whose second descent separates from initial fit, the denser Figure-9 grid on the pilot
run's local maximum, interpolation at non-final positions, or a second model/tokenizer. No `STOP`
written — a follow-up operator request remains plausible and a STOP'd direction would silently ignore it
(CLAUDE.md rule 11).

On track? yes — plan complete (S1–S14), nine PLAN-named follow-ups done, this iteration's prediction
tested and the depth/capacity confound removed; blocker: none.

## 2026-08-02 (iteration: S14b — the narrow run's fully-trained row, and three stale claims)

No unaddressed `human_feedback*`/`*REVIEW*` files (all five end in `.addressed.md`), so this went to the
first of the two loose ends S14 recorded: score the narrow run's final checkpoint, which did not exist
when the last iteration ran out of clock.

**Waiting correctly this time.** Training was still running when I started (launched by the previous
iteration with `--max_minutes 16`). Last iteration's process-liveness bug — `pgrep -f` matching the
waiter's *own* command line, which silently reported a job as running for five minutes — was avoided by
waiting on `ps -p 50442` for the concrete PID instead of on a pattern. The wait fired at 21:44:49 within
seconds of the training process exiting.

**Result: 0.332, i.e. the narrow model ends up marginally sharper than the full-width reference.**
Against ref_trained's 0.351 that is −0.010 paired (43% of pairs wider, p = 2.1e-4); against the
fully-trained frozen runs it is −0.124 and −0.146 with 1.3% and 3.3% of pairs wider (p ≈ 3e-26). So the
matched-accuracy conclusion from S14 (0.397 vs the capacity account's ≈0.47) is not an artifact of the
matching rule — the two framings agree, and the second one is the more conservative of the two since
the narrow run had *more* steps at its final checkpoint than at step 2,750.

**The honest caveat on this row.** The 16-minute budget stopped the run at step 27,143 of 30,000, so its
cosine schedule reached lr 1.2e-4 rather than 1.0e-4 while every frozen run's `_last` row is a clean
30,000. I report the truncation next to the number in both deliverables rather than presenting the row
as if it were schedule-matched. The direction of the bias is checkable and favourable: the run was still
sharpening (0.397 → 0.332 between the matched step and step 27,143, p = 3.1e-14), so a truncated run can
only understate how sharp the full schedule would have left it. The rejected alternative was to omit the
row entirely; that would have left S14's conclusion resting on one framing when a second was available
for 70 seconds of GPU.

**Curation caught three stale claims that the new number made obvious.** REPORT Limitation 7 still said
the depth/capacity separation "needs a narrower-but-full-depth run, which was not performed" — S14
performed it; the closing caveat of Experiment 5's "What this settles" said the same thing; and the
REPORT **Summary** never mentioned the depth-versus-capacity result at all, even though it is the
finding that makes the whole five-run frozen series interpretable. All three fixed. The lesson worth
keeping: when an iteration adds a result under time pressure, the *distant* prose that the result
contradicts is what gets missed — grepping for "not performed"/"was not run"/"future work" across the
deliverables is a cheap sweep and found two of the three.

**Figure 24 now carries both framings.** Each run keeps its matched-accuracy marker and gains a small
open square at its end-of-training width, joined by a dotted connector, so the claim "the ordering is
not an artifact of the matching rule" is visible rather than asserted. Two fixes fell out of drawing it:
the reference and narrow runs both sit at 12 trainable blocks and were overplotting, so they are now
offset along x; and my first attempt drew the trained value as a `_` tick, which is indistinguishable
from an error-bar cap — an open square is unambiguous. Legend moved to the left panel's empty corner
because it was clipping the "narrow 192" label in the right panel.

**Next step.** The remaining loose end from S14 is unchanged and is now the top open item: every point
in the depth series is a single seed, so the 0.397-vs-0.476 gap that carries the depth conclusion has no
across-seed error bar. A second seed at `n_embd` 192 and at one frozen condition (~16 min training each
on this harness plus ~70 s of assay) would give it one. Everything else still open needs a longer
character run whose second descent separates from initial fit, the denser Figure-9 grid on the pilot
run's local maximum, interpolation at non-final positions, or a second model/tokenizer. No `STOP`
written — a follow-up operator request remains plausible and a STOP'd direction would silently ignore it
(CLAUDE.md rule 11).

On track? yes — plan complete (S1–S14), ten PLAN-named follow-ups done, S14's first loose end closed and
its conclusion confirmed under a second framing; blocker: none.

## 2026-08-02 (iteration: S14c — the second narrow seed, and one sub-claim retracted)

No unaddressed `human_feedback*`/`*REVIEW*` files (all five end in `.addressed.md`), so this went to the
top open item S14b left: every point in the depth series was a single seed, so the 0.397-vs-0.476 gap
that carries the whole depth-over-capacity conclusion had no across-seed error bar.

**What I ran.** Added `--seed` to `train_frozen.py` (two lines: the flag, and `model_seed = args.seed`;
data seed left at 42 so only the initialization moves) and trained a second `n_embd` 192 run with
nothing frozen from model seed 2024. Made `narrow_assay.py` take its tag from `sys.argv` so the same
scorer handles both seeds, and scored the new run on the identical 150 pairs. ~2.5 min of training plus
~70 s of assay.

**Time management.** With ~25 min on the clock I launched training with `--max_minutes 5.5`, aiming only
at `ckpt_matched.pt` (seed 1337 hit it at step 2,750, ≈1.6 min in), and killed the process the moment
that file appeared rather than letting it keep competing for the GPU I also needed for the assay. So
this seed contributes a matched-accuracy row only — which is the right one anyway, since matched
accuracy is the only apples-to-apples axis across runs of different capacity and is the framing the
whole frozen series uses. The rejected alternative, a full 30,000-step run, needed ~18 min and would
have left no time to curate.

**Result: the conclusion holds, and one sub-claim does not.** Seed 2 reached the reference's accuracy at
the *same* step 2,750 (val 0.5547 vs 0.5543) and gave median width **0.437** against seed 1337's 0.397 —
so the across-seed spread is ≈0.04 (paired +0.015, p = 0.015). That is real but roughly half the 0.08–0.10
gap it is being used to judge: both seeds stay below frozen-early (seed 2 −0.044, p = 2.7e-8) and well
below frozen-late (−0.062, p = 1.6e-16), and the two-seed mean 0.417 sits nearer the depth account's
≈0.35–0.44 than the capacity account's ≈0.47. But S14's extra flourish — that the narrow run is
*sharper* than the full-width reference at matched accuracy (−0.014, p = 1.9e-4) — does not survive:
seed 2 is −0.004 at p = 0.17. The honest claim is "removing a third of the parameters costs nothing
measurable", not "it helps". Retracted in the Summary, in Experiment 5, and in the figure caption.

**Lesson worth keeping.** The sub-claim that died was the one with the smallest effect size in the
bullet (−0.014 against gaps of −0.073 and −0.092) and the one stated with the most confidence-sounding
p-value. A single-seed p-value measures spread across *pairs*, not across *initializations*; when the
effect is a hundredth of a unit, the second source of variance is the one that decides it. Worth
applying the same scepticism to the frozen series' smaller ordering claim (5 trainable blocks next to
the readout beat 5 at the bottom, 0.558 vs 0.626) — that gap is larger, but it is still one seed each.

**Deliverables.** New RESULTS bullet and REPORT paragraph with the seed-2 numbers; Figure 24 regenerated
with the second seed as a third filled circle at 12 trainable blocks, nudged apart on both axes (the
three all-trainable runs were overplotting, and the labels needed an explicit per-key offset table);
both captions now read the gap between the two filled circles as the across-seed spread. REPORT
Limitation 7 and the Summary updated so no stale "single seed, no error bar" text remains.
`check_render.py` passes on both files (29 display eqs, 27 embeds each, 0 problems).

**Next step.** The five frozen conditions are still one seed each; a second seed at frozen-early
(~16 min training, 70 s assay) would give the 12→8-block step its own error bar and is now the cheapest
remaining strengthening. After that, everything open needs a longer character run whose second descent
separates from initial fit, the denser Figure-9 grid on the pilot run's local maximum, interpolation at
non-final positions, or a second model/tokenizer. No `STOP` written — a follow-up operator request
remains plausible and a STOP'd direction would silently ignore it (CLAUDE.md rule 11).

On track? yes — plan complete (S1–S14), eleven PLAN-named follow-ups done, the depth conclusion now
carries an across-seed error bar and its one unsupported sub-claim is retracted; blocker: none.

## 2026-08-02 (iteration: finalization — verify, close, STOP)

~21 minutes on the clock, which the harness treats as finalization only: no new experiment, verify the
deliverables and close the direction.

**Feedback check first (CLAUDE.md Part C).** Listed the direction root: five `human_feedback*` files,
all five ending in `.addressed.md`. Zero unaddressed feedback, so rule 11's bar for writing `STOP` is
met.

**What I did.** Verified rather than rewrote. `check_render.py REPORT.md RESULTS.md` returns ALL CHECKS
PASS (REPORT 29 display / 449 inline equations / 27 figures; RESULTS 27 figures; 0 problems), so every
equation renders as `js-display-math` on GitHub, none degraded to a `<pre lang="math">` code block, no
denylisted macro (`\operatorname`) crept back in, and every inline `$…$` still compiles after GitHub
strips backslashes before punctuation. Figure hygiene: 27 embeds and 27 visible `**Figure N.**` caption
lines per file, counts equal, zero bare `(plots/x.png)` paths outside an embed, and all 27 referenced
PNGs present on disk. Structure is `Summary → Methods → Results → Conclusion` with Methods carrying
data/model/layer plus every metric and baseline defined. A staleness grep for version-history language
and for the two phrases that caught stale claims last iteration ("not performed", "was not run")
returned nothing.

**The one judgement call.** The finalization instruction says to "write a clean presentable REPORT.md",
which could be read as rewrite-from-scratch. I did not: REPORT.md is 127 KB, was curated to current-best
in S14c hours ago, and passes every mechanical check. Rewriting it in the remaining minutes has a real
downside with precedent — the `\operatorname` regression that drew operator feedback #4 was introduced
by exactly such a late rewrite — and no upside I could name. The rejected alternative was a full
re-curation pass; I chose verification plus a documented staleness sweep instead, and recorded the
verification numbers in CHANGELOG.md so the claim "verified, not rewritten" is checkable.

**One thing worth flagging for whoever reads this next.** The strongest remaining weakness in the
deliverables is not a rendering or curation problem, it is a statistical one that S14c exposed: the five
frozen conditions are one seed each, and S14c showed that across-seed spread (~0.04) is the same order
as the smaller ordering claims those single-seed runs support (5 trainable blocks next to the readout
beating 5 at the bottom, 0.558 vs 0.626, is safe; finer steps in the monotone ordering are not). The
headline depth-over-capacity conclusion is safe because its gap (0.08–0.10) is about twice the
across-seed spread, and Limitation 7 states this correctly. A second seed at frozen-early (~16 min
training + 70 s assay) remains the cheapest strengthening if this direction is ever resumed.

**Closing.** Appended the finalization entry to CHANGELOG.md, updated PLAN.md's Current status and Next
step, and wrote an empty `STOP` file — permitted here because the plan is complete and no unaddressed
feedback file remains. If an operator drops new feedback next to this `STOP`, the re-entering agent must
delete `STOP`, address the file, and only re-write it once clean (rule 11).

On track? yes — plan complete (S1–S14) plus twelve PLAN-named follow-ups, deliverables verified rendering
clean and current-best, direction closed with STOP; blocker: none.

## 2026-08-03 (iteration: S15 — second seed at frozen-early)

**Feedback check first (CLAUDE.md Part C).** Listed the direction root: five `human_feedback*` files,
all five already ending in `.addressed.md`. Zero unaddressed feedback, so this iteration advanced the
plan instead.

**Two surprises before any work.** (1) The `STOP` file written on 2026-08-02 is gone and the wrapper
relaunched the direction, so the plan is live again. (2) The pod was reset: every `/tmp` scratch
checkpoint (`/tmp/dir13_ckpt_grok_char`, `/tmp/dir13_frozen/*`) and `/tmp/tinyshakespeare.txt` are
gone. That mattered immediately, because `allpairs_sweep.load_vocab()` restored the 65-character
vocabulary from a pilot checkpoint. I re-downloaded the corpus, confirmed its SHA-256 matches the value
every `train_meta*.json` records, and made `load_vocab` fall back to rebuilding `sorted(set(text))` from
the SHA-asserted corpus — byte-identical to what the runs used, and no longer dependent on scratch.
The assumption I logged and proceeded on: a SHA match makes the rebuilt vocabulary provably the same
object, so nothing has to be re-measured. The rejected alternative was retraining the reference run to
regenerate its checkpoints (~45 min, and it would have produced a *different* reference than the one
every published number is scored against).

**What I ran.** The step the previous `Next step` named: `train_frozen.py --freeze 1,2,3,4 --seed 2024
--tag frozen_early_s2`, a second initialization of frozen-early with everything else held fixed. It
matched the reference's final validation accuracy at step 2,750 — the same step as seed 1337 — and
completed all 30,000 steps in 44.7 min (val 0.5629 vs 0.5625). `narrow_assay.py` scored both its
checkpoints on the same fixed 150 pairs; I generalised that script by two lines so it reads the frozen
block list off the checkpoint instead of hard-coding "narrow, nothing frozen".

**What it says.** Seed noise on this measure is small and directionless. At matched accuracy the two
frozen-early seeds give medians 0.476 and 0.498, but pair-by-pair they are indistinguishable (paired
+0.001, exactly half the pairs each way, p = 0.40) and their distributions agree decile by decile — the
0.022 gap in the marginal median is tail behaviour, not a shift. At step 30,000 the second seed is
0.445 against 0.471, i.e. 0.027 the *other* way. The relocation signature reproduces exactly (injecting
inside the frozen group changes the width by ≤0.003 at either checkpoint). And the comparison this was
run for now separates cleanly: the three runs with 12 trainable blocks (0.443, 0.397, 0.437) are
disjoint from the three with 8 (0.476, 0.498, 0.500), one-sided rank-sum p = 0.05 — the floor a 3-vs-3
design can reach — with all four narrow-vs-frozen-early seed combinations agreeing pair by pair.

**One error found and corrected.** Rebuilding `plot_capacity.py` without its checkpoint dependency
exposed that the narrow run's published parameter count (5,584,896) came from summing the `state_dict`,
which double-counts the tied embedding/unembedding weight and includes the causal-mask buffers. Counted
the way every other run is counted, it is **5,375,808** — so the narrow run has 4.0% *fewer* trainable
parameters than frozen-early, not 0.3% more. I checked which direction this cuts before rewriting: it
strengthens the depth conclusion, because the sharper run turns out to be the one with less capacity.
Corrected in the Summary, Methods, Results and Figure 24 caption of REPORT.md and the matching text in
RESULTS.md, and recorded as a correction in CHANGELOG.md.

**Figure work.** Figure 24 gained a sixth frozen marker and immediately became unreadable — three runs
now share x = 8 blocks and three share x ≈ 5.4–5.6M parameters, and the per-key label offsets collided.
Rather than keep nudging, I changed the convention: each *condition* is labelled once, and where a
label reads "(2 seeds)" its two markers sit adjacent. Caption rewritten in both deliverables to define
that convention and to name both across-seed gaps. I also fixed a rule-9a failure the updated
`check_render.py` flagged (the "Models actually tested" table had no prose above it).

**A trap worth recording.** I queued the post-training assay with
`while pgrep -f "train_frozen.py --freeze 1,2,3,4 --seed 2024"; do sleep 20; done`. `pgrep -f` matched
the waiter's *own* command line, so it would have spun forever; I noticed only because the log file it
was supposed to create never appeared. Any future waiter needs a pattern that cannot match itself (or
should just wait on the meta JSON the trainer writes at the end, which is what I switched to).

**Next step.** The finer ordering claim that still rests on one seed each is "5 trainable blocks next to
the readout beat 5 at the bottom" (frozen-deep 0.558 vs frozen-mirror 0.626). That 0.068 gap is only
~1.7× the seed spread measured here, making it the weakest surviving sub-claim; a second seed at
frozen-deep (~45 min training under contention plus ~2 min of scoring) would settle it. No `STOP`
written — wall clock and a named next experiment both remain.

On track? yes — plan complete (S1–S15) plus twelve PLAN-named follow-ups; the depth step now has two
seeds a side and separates disjointly, and one published parameter count was corrected; blocker: none.

## 2026-08-03 (iteration: S16 — second seed at frozen-deep, the position term)

**Feedback check first (CLAUDE.md Part C).** Listed the direction root: five `human_feedback*` files,
all five already ending in `.addressed.md`. Zero unaddressed feedback, so this iteration advanced the
plan. `/tmp` had survived this time — the corpus was present and its SHA-256 matched the value every
`train_meta*.json` records, so no re-download was needed.

**What I ran, and why that one.** The previous `Next step` named it: after S15 gave the depth step two
seeds a side, the only surviving sub-claim resting on a single pair of runs was the *position* term —
five trainable blocks beside the readout (frozen-deep, `w` 0.558) beating five at the bottom
(frozen-mirror, 0.626). A 0.068 gap against a ~0.04 measured seed spread is not a comfortable margin,
so `train_frozen.py --freeze 1,2,3,4,5,6,7 --seed 2024 --tag frozen_deep_s2`. I wrote the prediction
into PLAN.md **and** REPORT Methods before the assay ran: land within ≈0.04 of 0.558 and stay below
0.626, else the position term is falsified.

**Outcome: confirmed at both framings.** Matched at step 3,000 — the same step as seed 1337 — with
val 0.5503, finished at val 0.5730 (seed 1337: 0.5742). Median width **0.559** at matched accuracy
against 0.590, and **0.579** at step 30,000 against 0.558. Two things worth recording about that. The
first is that the seed spread is again small (0.031, 0.021) and again *sign-inconsistent*: seed 2024 is
narrower at the matched checkpoint and wider at the end, exactly the pattern frozen-early showed, which
is now two independent conditions saying the marginal median moves by ~0.02–0.03 in whichever
direction. The second is that both seeds clear frozen-mirror on both axes — the worst frozen-deep seed
is still 0.039 (matched) and 0.046 (final) narrower, and pair by pair the replicate is −0.060 and
−0.040 against it (p = 5.9e-14, 3.4e-8). The relocation signature reproduced to three decimals
(injection blocks 0/2/4 give 0.559/0.558/0.557, so the seven frozen blocks contribute nothing).

**One thing the replicate does *not* fix, and I said so in Limitation 7 rather than glossing it.** The
position comparison now has a seed spread under one of its two sides. Frozen-mirror is still a single
run, so "0.626" could itself be a high draw. What the replicate rules out is that *frozen-deep* was a
low draw, which was the more likely failure mode given the direction of the claim; the honest statement
is a one-sided error bar, and that is what both deliverables now say.

**Curation beyond the new numbers.** Both hypothesis paragraphs still ended on the narrow-run
prediction that S14 answered two iterations ago, and both misdescribed the reference width as `n_embd`
384 (it is 240) — a stale sentence that had survived several passes because it sat at the end of a long
paragraph. Replaced with the narrow-run *outcome* plus the genuinely open successor: five trainable
blocks in the middle of the stack (freeze 0–3 and 9–11 — same seven frozen blocks, neither end) should
land near 0.58–0.60 if position is a real second-order term. Figure 24 needed the same treatment as
last iteration: a seventh diamond made the "frozen 1-7" and "frozen 5-11" labels collide on both
panels, so I moved two offsets rather than adding another marker style.

**Verification.** `check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS (REPORT 29 display / 472
inline equations / 27 figures; RESULTS 27 figures; 0 problems); 27 embeds and 27 visible `**Figure N.**`
captions per file; zero bare `(plots/x.png)` paths.

**Next step.** The middle-of-stack five-block freeze the hypothesis now ends on
(`--freeze 0,1,2,3,9,10,11 --tag frozen_mid`, ~21 min training plus ~2 min scoring) is the natural
successor: it is the only one of the three five-block positions not yet run, and it turns the position
term from a two-point contrast into an ordered three-point one. A second seed at frozen-mirror would
instead close the one-sided error bar noted above; of the two, the middle-of-stack run answers a new
question and the mirror replicate only firms up an old one, so I would run the former first. No `STOP`
written — wall clock and a named next experiment both remain.

On track? yes — plan complete (S1–S15) plus thirteen PLAN-named follow-ups, S16 done and its prediction
confirmed; both terms of the depth/position reading now rest on gaps larger than a measured seed spread;
blocker: none.

## 2026-08-03 (same iteration, addendum: S17 launched — middle-of-stack five-block freeze)

With the S16 deliverables verified and wall clock left, I launched the successor the hypothesis
paragraphs now end on rather than idling: `train_frozen.py --freeze 0,1,2,3,9,10,11 --tag frozen_mid`,
which freezes the same seven blocks as frozen-deep and frozen-mirror but leaves the trainable five in
the middle (blocks 4–8). It is the only one of the three five-block positions not yet run, and it turns
the position term from a two-point contrast into an ordered three-point one. Freezing block 0 costs the
measurement nothing, since injecting at block 0 overwrites block 0's output anyway, so all five
trainable blocks sit downstream of the injection — the same as frozen-deep, unlike frozen-two.

**Prediction fixed before scoring** (PLAN S17 and the hypothesis paragraphs): `w` lands between the two
known five-block values, near 0.58–0.60; at or below frozen-deep's 0.559–0.590, or at or above
frozen-mirror's 0.626, falsifies the ordered reading. The run matched the reference's validation
accuracy at step 3,750 — later than frozen-deep's 3,000 and frozen-mirror's 2,750, which is itself a
small data point for the position account, since the middle placement optimizes more slowly.

**Code changes made while it trained** (all no-ops until its rows exist): `frozen_pairwise.py` gained
seven `frozen_mid` comparisons plus a `frozen_mid_median` / `mid_between_deep_and_mirror` field on the
position-contrast summary; `plot_capacity.py` gained the condition at five trainable blocks with the
four-marker nudge and label offsets that keeps the 5-block cluster legible.

**If this entry is the last word on frozen_mid, the run finished after the iteration did.** PLAN's
"Current status" carries the handover: the scored rows land in `results/frozen_assay_summary.json` and
`results/frozen_pairwise.json` and Figure 24 redraws itself, so the next iteration's first job is
curating the text — the numbers will already be on disk, and the prediction is on record above so the
verdict cannot be written after the fact.

On track? yes — S16 complete and verified (position term replicated), S17 launched with its prediction
pre-registered and its scoring chained; blocker: none.

## 2026-08-03 (iteration: S17 curated + S18 run — the count-first reading falls)

**Feedback check first (CLAUDE.md Part C).** Listed the direction root: five `human_feedback*` files,
all five already ending in `.addressed.md`. Zero unaddressed feedback, so this iteration curated the
run the previous iteration launched and then advanced the plan.

**S17 landed while I was reading, and it falsified its own prediction — which turned out to be the
most useful result in the frozen-block series.** Five trainable blocks in the *middle* of the stack
(freeze 0–3 and 9–11, so the frozen fraction and the trainable count match frozen-deep and
frozen-mirror exactly) were predicted to land between those two at 0.58–0.60. They land at **0.365**
at matched accuracy and **0.331** at step 30,000 — below both, below every eight-block run, and below
the untouched 12-block reference — with a strict plateau rate of **24.7%** against the reference's
10.0% and 0–0.7% for every other frozen run. Paired against its siblings: −0.211, −0.188, −0.240, with
0–1.3% of pairs going the other way.

**What that forced (rule 9b).** Both deliverables had read the series as "trainable depth first,
position a second-order correction favouring the readout", resting on the monotone 0.351 → 0.47 →
0.56–0.63 → 0.726 series. That is now wrong in its main clause: the *three five-block runs alone span
0.365–0.629*, wider than the whole 12-to-5 series. So the count is not the first-order term, and
position is not a gradient toward the readout — it has an interior optimum, which the earlier
two-point contrast could not see because it sampled only the two ends. What the three points have in
common is how the seven frozen blocks are distributed around the trainable window: mid splits them
3/3, deep stacks seven before, mirror seven after. I wrote that as an explicit description-of-three-
points rather than a law, and named the experiment that would test it.

**Then I ran that experiment rather than leaving it as a promise.** `--freeze 0,1,2,3,4,8,9,10,11`
shrinks the mid-stack window to three blocks (5–7) with 74.6% of parameters frozen, separating the
window's position from its size. Prediction fixed in PLAN and REPORT Methods before scoring: 0.40–0.50
if position dominates, ≥0.558 if the count returns. **Confirmed at 0.446** — indistinguishable from the
full 12-block reference (0.443, p = 0.17) and 0.09–0.18 clear of every five-block end window. Size is
not free (five mid-stack blocks → three costs +0.086 and doubles the steps to matched accuracy, 3,750
→ 7,000), but it is dominated. Three trainable blocks in the middle reproduce a 12-block network's
plateau geometry.

**Two things worth recording for the next agent.** (1) The figures are near their legibility ceiling:
Figure 23 now carries ten curve panels and eight series in its depth and accuracy panels, which needed
the CVD five hues plus three gray lightnesses, each with its own dash pattern and marker, and the
accuracy panel's seven per-run legend entries folded into one with the steps moved into the caption. A
ninth series needs a fourth lightness or a small-multiples split. (2) Chaining the assay off the
trainer's meta JSON (`until [ -f results/train_meta_<tag>.json ]`) worked cleanly again and avoids the
`pgrep -f` self-match trap recorded two iterations ago.

**Sweep for stale counts.** Adding two runs invalidated a dozen "five runs"/"four runs"/"the only
non-reference run" statements scattered across both files. I grepped for every numeral phrase rather
than trusting a read-through, and found one substantive error that way: the narrow run was described
as "the only non-reference run that keeps the sharpest tail (13.3%)", which frozen-mid's 24.7% now
falsifies. Corrected in both deliverables.

**Verification.** `check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS (REPORT 29 display / 536
inline equations / 27 figures; RESULTS 27 figures; 0 problems); 27 embeds and 27 visible `**Figure N.**`
captions per file; zero bare `(plots/x.png)` paths.

**Next step.** S19, the off-centre five-block window (`--freeze 0,1,7,8,9,10,11`, trainable 2–6), with
its prediction already written into PLAN and both hypothesis paragraphs: 0.40–0.45, between
frozen-mid's 0.365 and frozen-deep's 0.558. It is the first point that distinguishes "the optimum is
the middle" from "the optimum is anywhere away from the ends". A second seed at frozen-mirror is the
alternative and only firms up an old number. No `STOP` written — wall clock and a named next
experiment both remain.

On track? yes — plan complete (S1–S16) plus fifteen PLAN-named follow-ups, S17 and S18 both done and
curated in the same iteration, and the frozen-block conclusion re-framed from "trainable depth" to
"where the trainable window sits" on the strength of two new runs; blocker: none.

## 2026-08-03 (same iteration, addendum: S19 — the off-centre window, and a description retracted)

With S18 curated and ~75 minutes of wall clock left, I ran the successor the deliverables had just
committed to rather than handing it off: `--freeze 0,1,7,8,9,10,11`, a five-block trainable window at
blocks 2–6 — one step off centre, with five frozen blocks below it instead of frozen-mid's three. The
prediction was on record in PLAN and in REPORT Methods before it was scored: 0.40–0.45 if the cost
tracks how the frozen blocks are distributed around the window, with anything at or below 0.365
counting against that description.

**It landed at 0.365 — indistinguishable from frozen-mid (p = 0.064) — so the description I had
written one hour earlier is wrong, and I retracted it.** Doubling the frozen stretch below the window
cost nothing. That is worth recording as a process point: S17's "distribution of frozen blocks" reading
was explicitly labelled a description of three points rather than a law, and stating it that way is
what made retracting it cheap. Both deliverables now carry the replacement.

**The replacement is stronger than what it replaced.** Ordering the eight frozen runs by their *usable*
window — trainable blocks intersected with 1–11, since patching at block 0 overwrites block 0's output
— separates them exactly. Strictly interior windows (4–8, 2–6, 5–7) give 0.365, 0.365, 0.446; windows
touching either end (5–11, 1–7, 8–11, 1–4, block 11 alone) give 0.476, 0.500, 0.590, 0.629, 0.712. No
overlap across eight runs. I wrote it with both caveats it needs: it was found post-hoc, and its
narrowest gap (0.030, the three-block interior window against frozen-early) is inside the ≈0.04 seed
spread, while the two five-block interior windows clear every end-touching run by ≥ 0.11. And I
pre-registered its test — a five-block window at blocks 1–5, one block down from the sharp 2–6 window,
which the split says must land above 0.47.

**Figure work forced by the ninth series.** The depth panel could not hold nine series inside a
five-hue palette, so I split it into two small multiples (the four five-block windows plus the
reference; the other freeze sizes plus the reference) and moved the trained reference to neutral black
as the anchor, which freed the five hues for the runs that carry the position result. That is what
CLAUDE.md rule 13 asks for above five categories, and it also reads better — the four-position
comparison is now its own panel. A tenth series would need the same treatment again; PLAN's next-step
note says so.

**Three runs in one iteration is more than the usual step, and the reason was cheap wall clock plus
pre-registered predictions.** Each was launched only after the previous one's result was curated and
its successor's prediction was written down first, so nothing was decided after the fact.

**Verification.** `check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS (REPORT 29 display / 558
inline equations / 27 figures; RESULTS 27 figures; 0 problems); 27 embeds and 27 visible `**Figure N.**`
captions per file.

**Next step.** S20: `--freeze 0,6,7,8,9,10,11` (trainable 1–5), the pre-registered test of the
interior/end split. Above 0.47 confirms it; near 0.365 falsifies it. A second seed at frozen-mirror is
the runner-up and only firms up an old number. No `STOP` written — a named, pre-registered experiment
remains.

On track? yes — S17, S18 and S19 all done and curated this iteration, the frozen-block conclusion
re-framed once and its follow-up description falsified and replaced, with the replacement's own test
pre-registered; blocker: none.

## 2026-08-03 (iteration: S20 — the pre-registered test runs and the split falls)

**Feedback check first (CLAUDE.md Part C).** Listed the direction root: five `human_feedback*` files,
all five already ending in `.addressed.md`. Zero unaddressed feedback, so this iteration advanced the
plan. `/tmp` had survived — the corpus was present, so no re-download.

**I ran the experiment the last three iterations had been building toward, and it falsified its own
prediction.** `train_frozen.py --freeze 0,6,7,8,9,10,11 --tag frozen_mid_low` (trainable 1–5) was
pre-registered in PLAN, both hypothesis paragraphs and the Conclusion as the test of the interior/end
split: its usable window touches block 1, so it had to land **above 0.47**. It lands at **0.363** at
matched accuracy and **0.326** at step 30,000, indistinguishable from the two mid-stack five-block
windows (p = 0.27, 0.23) and 0.10–0.23 clear of every end window. The split is withdrawn. That is the
second post-hoc description of this series to die on the first experiment aimed at it, and the second
time in three iterations that pre-registering the number *before* launching the run is what made the
retraction cheap rather than embarrassing.

**The replacement is not another fitted rule, and that is deliberate.** The strongest thing this run
gives is a two-network comparison with no regularity in it: blocks 1–5 are a strict *subset* of
frozen-late's trainable 0–7, and they are 0.118 sharper (p = 2.2e-25). Removing trainable blocks
sharpens the plateau. Nine runs cannot be summarised by trainable count, and now they cannot be
summarised by window geometry either, but that one sentence stands on two runs and survives whatever
happens to the descriptions. I promoted it to the front of the frozen-block story in both deliverables
and demoted the geometry to a labelled description: every usable window covering block 5 gives
0.363–0.500, the three without it 0.559–0.712 — stated with the explicit note that it earns no credit
until tested, and with its test (a window at blocks 6–10, which excludes block 5 while touching neither
end, predicted ≥ 0.55) written into PLAN and both deliverables before it is run.

**Practical notes.** (1) The ten-series figure needed the palette re-assigned rather than extended:
the five CVD hues now map onto the five five-block windows, which are exactly the left small multiple,
and the four other freeze sizes take four gray lightnesses in the right one — so each panel is one
family and no panel exceeds five hues. A tenth *five-block* run would break this; the next split would
have to be by window position. (2) The capacity figure's 5-block column holds six markers now and
needed a finer nudge grid (±2.0 in steps of 0.8). (3) Chaining assay+plots off the trainer's meta JSON
worked again; 20.2 min for 30,000 steps at bs 48.

**Verification.** `check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS (REPORT 29 display / 569
inline equations / 27 figures; RESULTS 27 figures; 0 problems).

**Next step.** S21: `--freeze 0,1,2,3,4,5,11 --tag frozen_high` (trainable 6–10), the pre-registered
test of the coverage description — ≥ 0.55 confirms it, near 0.365 refutes it and leaves the series with
no geometric summary at all, which would itself be the honest finding. A second seed at frozen-mirror
remains the runner-up. No `STOP` written — a named, pre-registered experiment remains.

On track? yes — S20 done, its pre-registered prediction refuted and the refuted claim withdrawn from
both deliverables the same iteration, with a rule-free replacement claim (subset of blocks, sharper
plateau) promoted in its place; blocker: none.

## 2026-08-03 (iteration: S21 — the second geometric rule dies, and the sharpest network appears)

**Feedback check first (CLAUDE.md Part C).** Listed the direction root: five `human_feedback*` files,
all five already ending in `.addressed.md`. Zero unaddressed feedback, so this iteration advanced the
plan. `/tmp` had survived — corpus and prior checkpoints present, no re-download.

**What I ran.** `train_frozen.py --freeze 0,1,2,3,4,5,11 --tag frozen_high` (trainable 6–10), the
experiment PLAN, both hypothesis paragraphs, Results and the Conclusion had pre-registered as the test
of the coverage description: its usable window excludes mid-stack block 5, so coverage required
**≥ 0.55**. It lands at **0.342** — not merely below the threshold but the **sharpest matched-accuracy
width of the fifteen models in this study**, with the highest strict plateau rate measured (28.0%).
Coverage is withdrawn. That is the second post-hoc description of this series to die on the first
experiment aimed at it, and both dead rules were killed by runs whose predicted values were written
down before launch.

**The honest conclusion is now the absence of a rule, and both deliverables say it in those words.**
Ten frozen runs are ordered by neither trainable count, trainable capacity, nor window geometry. I
deliberately did not fit a third description — after two one-shot deaths, another curve through ten
points would earn nothing. What the series does establish are two direct network-to-network facts that
need no regularity: blocks 1–5 ⊂ blocks 0–7 and 0.118 sharper (from S20), and now blocks 6–10 alone —
58.0% of parameters never moved from initialization — **0.072 sharper than the untouched 12-block
reference at matched accuracy** (18.7% of pairs wider, p = 8.5e-18). Training fewer blocks sharpened
the plateau, twice, measured against two different comparators.

**A time-management decision worth recording.** With ~43 minutes and a 20-minute trainer, I launched
the run first and edited the three analysis scripts while it trained, then noticed the
matched-accuracy checkpoint (step 3,750) was already on disk at the 14-minute mark and assayed *it*
while the final steps ran. Matched accuracy is this section's primary axis, so that bought the whole
result without waiting for step 30,000. The step-30,000 assay was still training when the iteration
closed; both Figure 23 captions state that this run enters the deliverables at its matched-accuracy
checkpoint only, and the chained `narrow_assay.py frozen_high` will have written the final-step numbers
for the next iteration to pick up.

**Figure work forced by the eleventh series.** Six five-block runs no longer fit the two depth small
multiples within the five-hue palette, so the depth panel is now **three** panels split by where the
trainable window sits: upper-stack windows (6–10, 4–8, 2–6), bottom windows plus the one non-window
trainable set (1–5, 0–4, and 0&8–11), and the other freeze sizes in gray. No panel carries more than
three hues plus the black reference anchor, and `frozen_high` reuses `frozen_deep`'s hue because the
split guarantees they never share a panel. Every series keeps its own dash pattern and marker.

**Verification.** `check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS (REPORT 29 display / 577
inline equations / 27 figures; RESULTS 27 figures; 0 problems). One rule-9d contrast construction I
introduced ("seed replications rather than a third rule") was caught by the checker and rewritten.

**Next step.** S22: pick up `frozen_high`'s step-30,000 numbers from the chained assay and add them to
the deliverables (one row, no new training), then the two seed replications the deliverables now name —
a second seed at frozen-mirror, the one single-seed run carrying a load-bearing comparison, and a second
seed at blocks 6–10 to confirm the study's sharpest network is not a seed artefact. No `STOP` written.

On track? yes — S21 done, its pre-registered prediction refuted and the refuted description withdrawn
from both deliverables the same iteration, replaced by the honest "no geometric summary" reading plus a
second rule-free two-network fact; blocker: none.

## 2026-08-03 (iteration: S22 — the pending final-step row arrives and does not cost anything)

**Feedback check first (CLAUDE.md Part C).** Listed the direction root: five `human_feedback*` files,
all five already ending in `.addressed.md`. Zero unaddressed feedback, so this iteration advanced the
plan.

**What I did.** No new training. S21 had launched `train_frozen.py --freeze 0,1,2,3,4,5,11` chained to
`narrow_assay.py frozen_high`, and closed with the trainer at step 27,000 and the step-30,000 assay
still pending. That chain finished during this iteration (trainer DONE at step 30,000, val_acc 0.5720,
best 0.5764; assay merged `frozen_high_last`), and I picked the row up: median `w` **0.328**
(IQR 0.252–0.395, strict 24.0%) against the reference's 0.351 at the same step, paired **−0.037**,
36.7% of pairs wider.

**Why this mattered more than one extra row.** The load-bearing claim from S21 — training only blocks
6–10, with 58% of the parameters never moved from initialization, gives a *sharper* plateau than the
untouched 12-block network — rested entirely on the matched-accuracy checkpoint (step 3,750 vs the
reference's 30,000). A skeptic could read that as an artefact of comparing an early checkpoint against
a converged one. The final-step row removes that reading: both networks have now run the identical
30,000-step schedule and the frozen one is still sharper. It also shows the run kept sharpening after
matched accuracy (0.342 → 0.328), so the matched-accuracy comparison is the conservative one, and the
truncation direction is against the finding rather than for it — the same pattern S14b found for the
narrow run.

**Time management.** ~24 minutes on the clock with a trainer already ~18 minutes in, so the whole
iteration was budgeted around waiting for the chain rather than starting anything. Polling in a wait
loop cost more wall clock than expected (the trainer's last 3,000 steps plus a ~70 s assay ran to
about the 26-minute mark of its own log), which left only the deliverable edits — the right call was
still to wait, since the alternative was closing a second iteration with the row still pending.

**One scope note retired.** Both Figure 23 captions had carried "its step-30000 assay having still been
running when this iteration closed". That is now false, so both were rewritten to say the figure shows
the matched-accuracy checkpoint because it is the section's primary comparison axis, and to point at
the step-30,000 width in the prose. The figure itself was not regenerated — no claim depends on a
panel that does not exist.

**Verification.** `check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS (REPORT 29 display / 581
inline equations / 27 figures; RESULTS 27 figures; 0 problems).

**Next step.** S23: the two named seed replications, in order — a second seed at blocks 6–10, to check
that the study's sharpest network is not a seed artefact (it now carries a two-network fact at both
matched accuracy and step 30,000, so it is the more load-bearing of the two), then a second seed at
frozen-mirror. ~21 min of training plus ~70 s of assay each. No `STOP` written.

On track? yes — S22 done, the pending step-30,000 row collected and it strengthens rather than
qualifies the claim it was pending on; blocker: none.

## 2026-08-03 (iteration: FINALIZATION — verify, record the limitation honestly, STOP)

**Feedback check first (CLAUDE.md Part C).** Listed the direction root for `human_feedback*.md` and
`*REVIEW*`: five files, all five already ending in `.addressed.md`. **Zero unaddressed feedback**, so
`STOP` is permitted under rule 11 and this iteration was free to finalize.

**What I did.** The wrapper reported ~14 minutes of wall clock, which is under the 20-minute
finalization threshold, so I ran no experiment. Instead: confirmed nothing was still training
(`pgrep` clean), confirmed the newest artefacts in `results/` (`frozen_assay_summary.json` and
`frozen_assay_raw.npz`, both 21:02) are the S22 rows already curated into both deliverables, re-read
RESULTS.md and REPORT.md, ran the render checker, and updated PLAN/CHANGELOG/JOURNAL.

**What I decided not to do, and why.** The standing next step (S23) was a second seed at blocks 6–10.
It needs ~21 min of training plus ~70 s of assay. Starting it with 14 minutes left would have burned
the whole iteration and produced a checkpoint at roughly step 20,000 — no matched-accuracy row, no
step-30,000 row, nothing that could honestly enter a deliverable, and a half-written artefact for a
loop that is about to stop. The alternative I picked — verify and finalize — leaves the deliverables
consistent with what was actually measured.

**How the missing replication is represented.** Not as pending work. Both deliverables already carry it
as a limitation: eight frozen groups, three of which carry a second seed, with frozen-mirror and blocks
6–10 named as the two single-seed runs under load-bearing comparisons. The measured seed spread on the
conditions that *do* have two seeds is 0.031 at matched accuracy, which is what lets a reader size how
much the missing seeds could move things — the load-bearing gaps are 0.072 and 0.118, both well clear
of that spread, so the risk is real but small. That is the honest framing and it needed no edit.

**Verification.** `check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS** (REPORT 29 display /
581 inline equations / 27 figures; RESULTS 27 figures; 0 problems). Independently: 27 `![` embeds and
27 visible `**Figure` caption lines in each file, and no bare `(plots/*.png)` path outside an embed.

**Next step.** None — `STOP` written. If the loop is re-entered because new feedback lands, delete
`STOP` first (rule 11), then run the two seed replications in the order PLAN's "Next step" now gives:
blocks 6–10 first, then frozen-mirror.

On track? yes — finalized cleanly with zero unaddressed feedback, deliverables verified current-best
and fully rendered, and the one unrun experiment recorded as a limitation rather than a promise;
blocker: none (the seed replications were out of time budget, not blocked).

## 2026-08-10 (iteration: operator feedback #5 — frequency, asymmetry, context/n, semantic groups)

**Feedback check first (CLAUDE.md Part C).** Listed the direction root: six `human_feedback*` files,
five ending in `.addressed.md` and **one not** — `human_feedback_5.txt`, dropped at 19:12 today. The
previous iteration's `STOP` was already gone (removed when the feedback landed), so this iteration was
the feedback, in full, and no `STOP` was written at the end.

**What the operator asked, and what I did.** Four numbered asks (1, 2, 4, 6), all to be written into a
**new** file `REPORT_followup.md` under REPORT.md's rules. That instruction is what kept REPORT.md and
RESULTS.md untouched except for one pointer paragraph each — rule 6 says curate them to current-best,
and the current-best statement of this analysis is that it lives in the companion file the operator
named.

1. *Pairwise plot with undertrained (< 1000 occurrences) characters removed, look for trends.* 12 of 65
   characters fall below the cut. The filtered 53×53 matrix is Figure 1; the trend is Figure 2. Median
   width 0.320 (1,378 well-trained pairs) vs 0.482 (702 pairs touching a rare character), and a clean
   dose–response, Spearman ρ = −0.78 over all 65 characters, still −0.66 within the kept 53. This is the
   strongest new result: the near-linear tail of the main report's width distribution is mostly rare
   characters, and sharpening looks like something bought per character with training updates.
2. *Example `d(t)` plots between well-trained characters, visual asymmetry check, no metrics.* Figure 3.
3. *(item 4) Do the character-level figures have context; state it on the plot and give the per-cell n.*
   They do — `"The house was "` (14 characters) with only the final position patched, everywhere except
   the context-control figures (9 contexts). I added the statement to the four new figures **and**
   re-rendered the six all-pairs figures (REPORT.md Figures 14–19) with a footer line naming the context
   and the sample count behind a cell. Methods carries a table covering every character-level figure.
4. *(item 6) One well-trained letter against the others, after the frequency filter — semantic groups?*
   Yes, and it replicates: Figure 4. `e`'s row alone gives Kruskal–Wallis p = 9.2e-3; the aggregate over
   43 well-trained letters gives Friedman p = 3.8e-18, Kendall W = 0.42, and W = 0.27 after regressing
   partner frequency out of the ranks.

**A decision that shaped item 2.** The scratch checkpoint directories (`results/checkpoints*` are
symlinks into `/tmp`) have been wiped since the last iteration, and `/tmp/tinyshakespeare.txt` with
them. I re-downloaded the corpus and verified its SHA-256 against the value the training runs recorded,
which restored the frequency variable — but the 30,000-step character checkpoint is gone, so no new
forward pass was possible. I wrote `experiments/followup_asym.py` to re-run six pairs in both endpoint
orders before discovering this. It was not needed: `slerp_norm(hA,hB,t) = slerp_norm(hB,hA,1−t)`
exactly, so the reversed curve is exactly `1 − d(1−t)`, and the stored 100-pair swap control already
measures max |w(A,B) − w(B,A)| = 0.000. So the endpoint-order reading of "asymmetry" has an exact
answer with no experiment, and the informative reading is the *shape* of one curve about its midpoint —
which is what Figure 3 plots (curve vs its own mirror), and the asymmetry there is large and tracks
endpoint plausibility, matching the main report's Figure 17. The alternative I rejected was retraining
the character GPT (~20 min) to reproduce a checkpoint that would not be bit-identical to the one every
existing number came from; that would have put two slightly different models inside one report.

**Rejected alternative on placement.** I considered folding the frequency result into REPORT.md's
all-pairs section instead of the companion file, since it is a genuine finding about the main
experiment. The operator's instruction was explicit about `REPORT_followup.md`, so the finding lives
there and both deliverables carry a pointer paragraph naming it — a reader of either file is told the
follow-up exists and what it found.

**Figure work.** Six character groups exceed the five-hue CVD palette, so hue carries vowel /
consonant / punctuation (whitespace folded to gray as the tail group, per rule 13) and every group also
gets its own hatch and marker; the figures stay readable in grayscale and no meaning rides on
red-vs-green. Three rounds of layout fixes were needed for collisions (legend over data points, group
labels into the title, threshold labels on their own lines).

**Verification.** `check_render.py REPORT.md RESULTS.md REPORT_followup.md` → ALL CHECKS PASS
(REPORT 29 display / 584 inline / 27 figures; RESULTS 27 figures; REPORT_followup 4 display / 60 inline
/ 4 figures; 0 problems), and 4 embeds / 4 visible `**Figure` captions in the new file.

**Next step.** The plan's own next step is unchanged and still open: S23, the two seed replications
(blocks 6–10 first, then frozen-mirror), ~21 min of training plus ~70 s of assay each. Note for whoever
picks it up: the checkpoints are gone from scratch, so the *reference* rows must be taken from
`results/frozen_assay_summary.json` rather than re-assayed. No `STOP` written.

On track? yes — feedback #5 addressed in full and renamed to `.addressed.md`, a new companion
deliverable written and render-verified, and the one ask that could not be run as literally stated
(a fresh both-directions run) answered exactly from the stored control with the reason recorded;
blocker: none.

## 2026-08-10 (iteration: S23 — both seed replications land, both pre-registered predictions hold)

**Feedback check first (CLAUDE.md Part C).** Listed the direction root: six `human_feedback*` files,
all six ending in `.addressed.md`. Zero unaddressed feedback, so this iteration advanced the plan.

**What I walked into.** The previous iteration had launched the S23 chain (two concurrent
`train_frozen.py` runs at seed 2024, chained to two `narrow_assay.py` calls) and then run out of time
mid-curation: it had written `plot_seeds.py`, produced `plots/seed_replication.png` from the three
conditions that already had two seeds, prepared `frozen_pairwise.py`/`plot_capacity.py` for the new
tags, and **renumbered the exploratory figures 25→26, 26→27, 27→28 to make room for a Figure 25 that
was never inserted** — so both deliverables had a numbering gap and an unembedded PNG, and no
CHANGELOG/JOURNAL entry existed for that work. Picking the chain up rather than restarting it was the
obvious call: it was 12 minutes into a ~57-minute pair of runs.

**Waiting was most of the iteration.** The two trainers shared the GPU with another agent's job and
took ~57 minutes to reach step 30,000 (rate drifting 680 → 530 steps/min as the box filled up), then
~2 minutes for both assays. I spent the wait reading the regions of both deliverables the results
would touch, so the curation itself was quick once the numbers landed.

**The results.** Both predictions, written into PLAN.md before either run was scored, held.
- *Blocks 6–10* (`frozen_high_s2`): 0.344 at matched accuracy against seed 1337's 0.342 — an 0.002
  spread, the smallest measured anywhere in this study — and 0.335 vs 0.328 at step 30,000. Against the
  untouched reference, −0.071 (p = 1.9e-16) and −0.021 (p = 7.5e-4). The study's headline frozen fact
  (58.0% of parameters never moved from init, yet sharper than the untouched 12-block network) is not
  an initialization artefact.
- *Blocks 0–4* (`frozen_mirror_s2`): 0.624 at matched accuracy against 0.629, and 0.590 against 0.626
  at step 30,000. All four deep-versus-mirror seed pairings preserve the ordering at both checkpoints.

**The one thing I had to narrow, and why I did it rather than report the win.** The position ordering's
*direction* replicated, but its *margin* did not stay comfortably outside seed noise: the closest
median pairing is 0.033 at matched accuracy and only 0.010 at step 30,000, against a largest measured
seed spread of 0.040 (the narrow run's 0.397 vs 0.437). The paired per-pair tests are still strong
(p = 3.1e-3 at the tightest cell, p ≤ 3.4e-10 elsewhere), so the claim stands — but stating it on the
strength of the medians alone would have been wrong. Both deliverables now say explicitly that this
ordering rests on the per-pair tests, and Figure 25's right panel hatches the two gaps that do not
exceed the spread instead of hiding them. That is CLAUDE.md rule 9c step 2 (narrow the claim) rather
than step 7 (discuss the limitation), because the evidence still supports the narrower statement.

**Figure work.** Two problems, both from adding markers to crowded figures. (a) `plot_seeds.py`'s
legend covered the sharpest condition once I moved it out of the way of the spread labels; fixed by
adding y-headroom and staggering the two checkpoints' annotations vertically. (b) Figure 24's
five-block column now holds **nine** markers (six conditions, three of them with two seeds), so
per-marker labels no longer fit: the six five-block labels are now parked in empty parts of each panel
and joined to their marker by a thin gray leader line. Both figures were re-rendered and read back as
images to confirm nothing sits on top of anything else.

**Verification.** `check_render.py REPORT.md RESULTS.md REPORT_followup.md` → **ALL CHECKS PASS**
(REPORT 29 display / 602 inline / 28 figures; RESULTS 28 figures; REPORT_followup 4/60/4; 0 problems).
Figure numbering is contiguous 1–28 in both files again, and Figure 25 is cited by number in the prose
of both.

**What I learned that changes the plan.** The seed queue is empty: nothing in either deliverable now
rests on a single initialization except conditions whose gaps are 0.14–0.26, three to six times the
spread. So S24 is not another freeze — the standing gap is *what the trainable blocks compute*, which
five excluded mechanisms have not touched in six iterations.

**Next step.** S24 as written in PLAN: (1) a probe of the computation itself, (2) interpolation at
non-final token positions (the one Matthew-assay control never run here), (3) anything needing a longer
run or a second model. I first wrote (2) as the PLAN 5.5 freeze-and-retrain prediction and corrected it
while updating PLAN — S10's `frozen_early` already is that experiment, and it was refuted.
No `STOP` written (the plan has open candidates and the loop should stay available for feedback).

On track? yes — S23 done, both pre-registered predictions held, the one claim whose margin shrank was
narrowed in both deliverables the same iteration, and the figure gap left by the previous iteration is
closed; blocker: none.

## 2026-08-10 — finalization iteration (2 min wall-clock remaining)

**What I did.** Checked Part C first: listed the direction root for `human_feedback*.md` / `*REVIEW*`
without the `.addressed.md` suffix — none (all six are addressed). With ~2 minutes of budget left the
instructions call for finalization only, so I ran no new experiment. I read the tails of CHANGELOG.md,
JOURNAL.md and the PLAN status/next-step sections and confirmed that RESULTS.md and REPORT.md were
already curated to current-best in this morning's S23 iteration (fourteen frozen runs; the two former
single-seed conditions replicated at seed 2024; seed spread bounded at 0.040) and that the last
`check_render.py` run over REPORT.md, RESULTS.md and REPORT_followup.md passed all checks with zero
problems. Since no result was superseded, rewriting either deliverable would only have churned them,
so I left them untouched and recorded the close-out state in CHANGELOG.md instead.

**What I learned.** The finalization step is cheap when the loop has kept the deliverables curated
every iteration — there was nothing to reconcile at the end, which is the intended payoff of rule 6.

**Next step.** None inside this budget. PLAN.md "Next step" (S24) carries the three open candidates:
probe what the trainable blocks compute, interpolate at non-final token positions, and replicate on a
second model. `STOP` written — plan candidates remain open but the wall-clock budget is exhausted and
zero unaddressed feedback files exist, so stopping is permitted under rule 11. If feedback is dropped
here later, delete `STOP`, address it, and re-write `STOP` only when clean again.

On track? yes — deliverables final, all render checks passing, zero unaddressed feedback; blocker: wall-clock budget exhausted.

## 2026-08-12 — feedback iteration: the basin criterion could not fail, so it was rebuilt

**What I did.** Part C first: listed the direction root and found one unaddressed file,
`human_feedback_6.txt` (no `STOP` present — it had already been cleared for this re-entry). That was
the whole iteration. The operator's point checked out immediately on inspection of
`matthew_assay.is_plateau` / `analyze_allpairs.py`: the basin fraction counted a partner when
`t_lo >= 0.10` (character at the A end) or `t_hi <= 0.90` (B end), and the straight line `d(t)=t`
produces exactly `t_lo=0.10` and `t_hi=0.90`. The threshold was the null value itself, so the
statistic was measuring nothing and the "all 65 characters own a basin" claim had no content.

I replaced the raw-length test with a **ratio to the null**: rest length `r(δ)` (path fraction within
`δ` of the character's output, on the same isotonic copy the width uses) divided by `δ`, which is
identically 1 for the straight line at every `δ`. Basin iff that ratio ≥ κ = 2 at δ = 0.10. Then I
measured the false-positive rate on four null families through the identical code path — exact line,
line + Gaussian noise at three σ, the untrained network's own 2,080 curves, and the 200-pair block-11
patch. All four give 0.0% at κ = 2 (0/2, 0/12,000, 0/4,160, 0/400 endpoint decisions) with median rest
ratios 0.94–1.00, while the trained network passes 90.3% of 4,160 endpoints at median ratio 3.18. The
old criterion, run on the same families, passed 40.8% of untrained endpoints and about half of pure
noise-around-a-line. No GPU work was needed: everything recomputes from the stored curves in
`results/allpairs_raw.npz`.

**Assumptions logged (loop mode, no one to ask).** (a) δ = 0.10 and κ = 2 are choices. I fixed δ at
the tolerance the transition width already uses and κ at the smallest integer multiple that clears the
null, then reported the whole φ-vs-κ curve (1→5) and the count at δ ∈ {0.05, 0.10, 0.20} so the
threshold dependence is visible rather than hidden — and put it in the caveats. Rejected alternatives:
an *area-between-curve-and-line* statistic (threshold-free, but no longer a per-partner count, so the
per-character fraction φ would have had to be redefined as well and the PLAN case (i)/(ii)/(iii)
verdict re-stated); and simply moving the constant to 0.15 (clears the null by 50% but with no
principled unit, and the operator's objection would apply again at the next null).
(b) I kept φ as the reported statistic rather than switching the deliverables to the continuous rest
ratio, because the PLAN question ("does *each* character have a basin?") is a count.

**What the numbers did to the story.** 59 of 65 characters clear φ ≥ 0.5 (median 1.00, mean 0.90, 39
at exactly 1.00). Six fail: `3`, `&`, `$`, `Z`, `X`, `z` — and those are the six rarest characters in
the training text (`$` appears once, `&` three times, `3` twenty-seven times). φ correlates with
training frequency at Spearman ρ = 0.56 (p = 1.0e-6, n = 65), and every character seen ≥ 1,000 times
has φ ≥ 0.68. So the narrowed claim is better evidence than the old universal one: the criterion now
discriminates, it discriminates along an interpretable axis, and it says where the geometry a
practitioner might rely on for steering or patching stops holding. Per rule 9b I reframed the claim in
the Summary, the all-pairs table, the per-character verdict, the hypothesis block and the Conclusion of
both deliverables rather than leaving a Summary promising something the numbers no longer deliver.

**What I learned.** A threshold placed *at* a null's value is worse than no test: it produces a
number that looks like evidence (0.86–1.00 for every character) and has zero discriminative power. The
tell was visible in the old data all along — the old φ was 1.00 for 59/65 and never below 0.86 even
though the underlying widths ranged 0.264–0.590. Any per-item criterion in this project should now be
reported alongside its false-positive rate on the init checkpoint, which costs nothing because those
curves are already stored.

**Housekeeping.** Two new figures forced a renumber of Figures 16–28 → 18–30 across both deliverables;
the plural-form references ("Figures 14–15", "Figures 6, 8, 16, 20", "Figures 24 and 25") do not match
a `Figure N` regex and had to be fixed by hand — worth remembering before the next insertion. Both
files now hold 30 embeds, 30 visible captions, sequential numbering, and every figure cited by number
(Figures 3 and 4 via the range "Figures 2–4"). `check_render.py REPORT.md RESULTS.md` passes with 0
problems.

**Next step.** `human_feedback_6.txt` is renamed `.addressed.md` and no unaddressed file remains, so
the plan is live again at S24 (probe what the trainable blocks compute; interpolation at non-final
token positions; a second model). No `STOP` written — there is wall-clock left and the plan candidates
are open.

On track? yes — the operator's objection is fixed at the source rather than papered over, the replacement
criterion is validated against four null families at a 0.0% false-positive rate, and both deliverables
carry the narrowed claim plus the new frequency result; blocker: none.

## 2026-08-12 (later) — S24a: the one control Matthew's assay left untested here, run to completion

**What I did.** Part C first: listed the direction root for `human_feedback*.md` / `*REVIEW*` without
the `.addressed.md` suffix — none (all seven are addressed), so this iteration advanced the plan.
Poking at `results/` turned up something the previous iteration had left behind: `pos_assay.py`,
`plot_pos.py` and a *partial* `pos_assay.json` holding only the matched-accuracy checkpoint, written at
20:48 on 2026-08-10 and never mentioned in JOURNAL, CHANGELOG or either deliverable. The script also
carried paired-statistics code whose output was absent from the JSON, i.e. it had been edited after
that run. So the cheapest real step was not a new experiment but finishing that one: the missing rows
need an init and a step-30,000 checkpoint, and `/tmp` had been wiped again.

Re-downloaded tinyshakespeare (SHA-256 verified against `allpairs_sweep.load_vocab`), retrained the
reference character run from scratch (`train_frozen.py --tag ref_pos`, nothing frozen, seed 1337,
30,000 steps, 29.2 min, final val acc 0.5502, matched at step 2,500), then ran `pos_assay.py` over all
three checkpoints (~2 min). Before the untrained and final rows existed I wrote four predictions into
PLAN.md — the built-in `read_patch` identity check, trained width < 0.55 at every offset, untrained
blunter by ≥ 0.15 with p < 1e-6, and the decision-versus-switch reading that follows if the first three
hold. All four held.

**What the numbers say.** At step 30,000 the median transition width is 0.243 / 0.290 / 0.249 / 0.244 /
0.257 for offsets k = 0/1/2/4/8, and paired against k = 0 the offsets 2, 4 and 8 are indistinguishable
(p = 0.27, 0.43, 0.22); only k = 1 moves at all (+0.040). The untrained network is the straight line at
every offset (0.804–0.809, 0/150 plateaus, paired p = 2.3e-26). The interesting panel is the third one:
endpoint separation at the readout falls 44.5 → 16.4 logit units, but the fraction of pairs whose
endpoints predict a *different* next character collapses much faster — 86.7% → 8.7% at k = 4 — and at
that offset 52.0% of pairs still meet the strict plateau rule. That is the first evidence here that
separates the switch from the description this report has been using for it: "next-character decision
basin" is what the boundary looks like at the *patched* position, and four characters later the
decision is gone while the discrete switch is not.

A second thing fell out that I did not predict: the distance-independence is a *late-training*
property. At the matched-accuracy checkpoint the widths still degrade with offset (0.328 → 0.434 at
k = 4, p = 5.6e-20, strict rate 28.0% → 7.3%); by step 30,000 that penalty is gone. And a free
reproduction check: the fresh run's anchor rows (block-0 `resid_post`, final position) give
0.803 / 0.4428 / 0.3507, reproducing the reference numbers quoted throughout both deliverables
(0.803 / 0.443 / 0.351) from an independent training run of the same recipe.

**Assumptions logged (loop mode, no one to ask).** (a) The injection site had to change: patching
`resid_post` at a non-final position is not exact, because positions after the patch have already read
prompt A's token through block 0's attention, so t = 1 would not reproduce prompt B. I used the
residual stream *entering* block 0 for every k including k = 0, which keeps the k-axis internally
consistent and is verified per pair (worst endpoint error 1.9e-5). Rejected alternative: patching every
position downstream of the varied character, which would have made the intervention a different
experiment (it patches the model's own recomputation, not the interpolated input). The cost is that the
sweep's k = 0 row (0.243) is not directly comparable to the report's block-0 rows (0.351) — hence the
anchor row in every checkpoint, which is what ties the two together. (b) k ≤ 8 characters, one context,
one filler string, one seed; stated as scope in both deliverables rather than smoothed over.

**Deliverable work.** New Results subsection in both files with Figure 28 (`plots/pos_offset.png`),
a new Methods subsection in REPORT.md, and the Summary/Headline/Conclusion verdict rewritten from
"plateaus are next-character decision basins" to "at the patched position they look like decision
basins; the switch survives where the decision does not". Two caveats were *corrected* rather than
extended — "final-position interpolation only" had been true when written and no longer is. Exploratory
Figures 28–30 renumbered 29–31 in both files; 31 embeds / 31 captions / sequential 1–31 in each;
`check_render.py REPORT.md RESULTS.md` passes with 0 problems.

**What I learned that changes how I work here.** Two things. First, a finished experiment can sit on
disk unreported — the previous iteration's own JOURNAL entry listed this as an open candidate while its
partial output was already in `results/`. Listing `results/` against the deliverables at the start of
an iteration is cheap and would have caught it a day earlier. Second, the figure script that came with
that unreported run had a middle-panel title asserting the plateau "widens with distance but survives",
which is what the *matched-accuracy* data shows and is false at the final checkpoint the panel plots;
titles written from a partial run are a trap, and I only caught it by reading the rendered PNG back.

**Next step.** S24's remaining candidates are unchanged and all need new compute: (1) what the
trainable blocks actually compute — still the direction's real open problem, untouched by seven
iterations of freezes; (2) a second model or tokenizer; (3) a longer character run whose second
local-complexity descent separates from initial fit. The offset axis itself could go further (longer
offsets, more contexts, a second seed) but that firms up a result rather than answering a new question.
No `STOP` written — wall-clock remains and the plan has open candidates.

On track? yes — S24a complete, all four pre-registered predictions held, an experiment that had been
stranded on disk is now reported, and the report's central description is narrowed to what the evidence
actually supports; blocker: none.

## 2026-08-12 (later still) — S24b: opening the block, one unit at a time

**What I did.** Part C first: listed the direction root for `human_feedback*.md` / `*REVIEW*` without
the `.addressed.md` suffix — none of the seven, so this iteration advanced the plan. PLAN's S24 item 1
("what the trainable blocks compute") has been the named open problem since 2026-08-03 and has
survived seven iterations of freezing experiments, all of which only ever answered *where* the
computation may live. I also checked `results/` against the deliverables first — the lesson from last
iteration, when a finished experiment was found stranded on disk — and everything there is reported.

The probe is a chord linearization of individual MLP hidden units. For a set `S` of units in blocks
1–4, each unit's post-GeLU activation at the patched position is replaced along the path by the chord
between its own two endpoint values. That deletes exactly one thing — the unit's curvature in `t` —
and nothing else, and it has a property that made it worth choosing over ablation: the chord agrees
with the true activation at both ends, so the two endpoint states that `d(t)` is measured against are
untouched for any `S`. That is verified per pair (worst deviation 1e-6) and it is why the resulting
widths can be read as "how much of the bend did that set carry" rather than "how much did I break".

**What the numbers say.** Linearizing all 3,840 units of blocks 1–4 removes 86.7% of the sharpness
(0.351 → 0.743 against the untrained 0.803) — nearly as much as deleting the MLPs outright (0.796),
while keeping every unit and every endpoint. Then the concentration: a pair's own top-32 units, 0.83%
of the population, remove 50.9%; the per-pair median number of units for half of that pair's own gap
is 64 (IQR 32–128). Thirty-two random units remove 1.2%, and random selection needs ~2,048 units to
match 32 ranked ones, so the ranking carries roughly a 64× concentration. The other half of the result
is negative and I think more useful: one *fixed* global set of 32 removes only 19.0%, and a typical
pair shares 9 of its 32 units with it. 668 of the 3,840 units ever enter a top-32 and 82% of those
serve two or more pairs, so there is a shared pool of about a sixth of the early MLP units, but which
few dozen bend a given path is pair-dependent. An unpredicted detail: the carrying units skew *deeper*
within the group (16/19/28/37% of top-32 slots in blocks 1/2/3/4) while single-block deletion is
front-loaded (41/28/18/11%). Both can hold — deleting block 1's MLP also changes what blocks 2–4 see —
and I said so in the section rather than leaving it as a loose end.

**Assumptions logged (loop mode, no one to ask).** (a) The importance ranking is measured on the same
curve it is then tested on, so per-pair top-`k` is a *concentration* measure, not a held-out
prediction. I did not fix this with a train/test split over `t` (which would break the 50-point grid
the whole report shares); instead the global-set condition is the honest out-of-pair test, and it is
reported as the weaker number in both deliverables and in the caveats. (b) Importance weights the
off-chord deviation by the unit's write-vector norm, so a large swing through a small output direction
does not outrank a small swing through a large one; the unweighted alternative was rejected because it
ranks by activation scale, which is not what reaches the residual stream. (c) Linearizing units in
block 1 changes the input to blocks 2–4, so this is an intervention, not an additive decomposition —
stated in both files.

**What I learned.** Two things worth carrying. First, an intervention that preserves the endpoints is
strictly more informative here than one that does not: every earlier mechanism probe had to argue that
the endpoints it moved were still comparable, and this one simply does not have that problem — the
1e-6 check is the whole argument. Second, the "sparse or distributed?" question had a much cheaper
answer than the freezing series that preceded it: no training, 198 s of forward passes on checkpoints
that already existed, against ~30 min per frozen run. Seven iterations of retraining bought "where",
and one afternoon of hooks bought "how many".

**Deliverable work.** New Results section in both files with Figure 29 (`plots/neuron_path.png`) and a
new REPORT.md Methods subsection defining the chord substitution, the importance score and the
recovered fraction $\rho(S)$. The hypothesis paragraph, Conclusion, Interpretation and Limitation 7
each had a "still uncharacterised" clause that is no longer true; all four are narrowed to what the
counts support, with "what those units detect" named as the part that stays open. Exploratory Figures
29–31 renumbered 30–32; 32 embeds / 32 captions / sequential 1–32 in each file;
`check_render.py REPORT.md RESULTS.md` passes with 0 problems.

**Next step.** The obvious successor is what the units *detect*: take the ~668 pool units and ask what
inputs drive them (max-activating characters/contexts in the corpus), which would turn "a few dozen
gated units" into a statement about features. It needs no training either. Beyond that, PLAN S24 item 3
(a longer character run, or a second model/tokenizer) still needs materially more compute than remains.
No `STOP` written — wall-clock remains and the plan has open candidates.

On track? yes — S24b done, the direction's named open problem now has its first quantitative answer
(sparse per path, pooled across paths), and both deliverables carry it; blocker: none.

## 2026-08-12 (later still, II) — S24c: the units have names

**What I did.** Part C first: listed the direction root for `human_feedback*.md` / `*REVIEW*` lacking
`.addressed.md` — none of the seven, so this iteration advanced the plan. I also listed `results/`
against the deliverables (the lesson from two iterations ago, when a finished run sat unreported on
disk) — everything there is reported. `/tmp/dir13_frozen/checkpoints_ref_pos` and
`/tmp/tinyshakespeare.txt` had both survived, so no retraining was needed.

PLAN's own next step was the successor S24b opened: S24b said *how many* units bend a path (a few
dozen per pair, from a pool of 668) but nothing about what they are. The trap in answering that is
circularity — any description derived from the interpolation assay would be a restatement of the
ranking. So the tuning measurement uses a completely different data source: the model's own 90%
training split, tiled into 7,842 non-overlapping 128-character windows (941,040 scored positions,
first 8 of each window dropped for want of context), with each block-1–4 hidden unit's mean post-GeLU
activation accumulated per current character and standardized within the unit across the 65
characters. Then the test: rank all 3,840 units by how differently they respond to a pair's two
endpoint characters, and score that ranking against the top-32 the chord linearization recorded.

**What the numbers say.** Differential tuning predicts recruitment at mean AUROC **0.847** (99% CI
0.834–0.858), precision@32 **21.6%** — 26× the 0.83% chance rate — against 0.498 for a shuffle and,
the control that matters, **0.562** for ranking units by overall corpus activity. So it is character
tuning and not busyness. The assay-derived global importance ranking, which has seen the experiment
but not the pair, gets 0.913, which bounds how much any pair-blind ranking could achieve and says
corpus tuning recovers most of it. Three further rows: recruitment falls monotonically 4.9% → 0.09%
across deciles of differential tuning; a recruited unit's single preferred character is one of that
pair's two endpoints for 27.2% of recruitments against a 2.8% base rate (9.8×); and the 668 pool units
are the sharply tuned population (median max|z| 5.45 vs 4.47, p = 5.8e-27). The qualitative half is
the part I did not expect to be so clean: the most reused unit (block 2, 88 of 150 pairs) is a
capital-letter detector whose six highest-activating corpus contexts are proper-name onsets —
`DUCHESS OF Y`, `Duke of Y`, `Bishop of Y`, `And I the house of Y`. Ordinary character-identity
detectors, driven through their switching point by the assay.

**Assumptions logged (loop mode, no one to ask).** (a) Tuning is conditioned on the *current*
character only. That matches the assay (the interpolated position holds the varied character) and it
is why the profiles read as character identity; a unit responding to a longer pattern is summarized
crudely, and both deliverables say so. Rejected alternative: conditioning on the (previous, current)
bigram, which would have 4,225 cells and leave the rare ones unusable. (b) The differential score
$|z_a - z_b|$ was chosen as primary before looking, on the argument that a unit must distinguish the
endpoints to switch along the path; $\max(z_a,z_b)$ was computed at the same time and is reported —
they differ by 0.007 AUROC, so nothing rests on the choice. (c) Three characters occur < 100 times, so
their conditional means are noisy; rather than drop them silently I ran the whole analysis a second
time re-standardized over the 62 well-sampled characters, on the 143 pairs built from them. It comes
out slightly *higher* (0.858), which is the honest way to retire the concern.

**What I learned.** The value here came from where the evidence was measured, not from a cleverer
statistic. Every previous mechanism probe in this direction was an intervention inside the assay, so
each one could only ever bound the mechanism from inside; one pass over the training corpus — 9
seconds of forward passes — produced a description of the same units that the assay never saw, and the
agreement between the two is the result. Worth remembering the next time an open question looks like
it needs another intervention: ask first whether it needs another *data source*.

**Deliverable work.** New Results section in both files with Figure 30 (`plots/neuron_feature.png`,
four panels) and a new REPORT.md Methods subsection defining the tuning profile, the tuning score
$z_{c,j}$, sharpness, the differential and max scores, AUROC and precision@32, and the three
baselines. The four places that named "what those units detect" as the open part (REPORT Summary,
Conclusion, Limitation 7; RESULTS hypothesis paragraph and Headline) now carry the result with its
correlational caveat. Exploratory Figures 30–32 renumbered 31–33; 33 embeds / 33 captions /
sequential 1–33 in each file; `check_render.py REPORT.md RESULTS.md` passes with 0 problems. One
render bug caught by the checker and not by eye: `$32/3840=0.83\%$` inline — GitHub strips the
backslash before `%`, turning the rest of the line into a LaTeX comment (CLAUDE.md rule 8b, exactly
the documented failure). Fixed by moving the percent out of math.

**Next step.** The obvious successor is causal: high-tuning units are *predicted* to carry the bend,
so linearizing a pair's top-32 units chosen purely by corpus tuning — never having run the importance
ranking — and comparing the recovered fraction against the assay-ranked top-32 (50.9%) and random
(1.2%) would turn this correlation into a held-out causal test. It reuses `neuron_path.py`'s machinery
with a different selection rule and needs no training. Beyond that, PLAN S24 item 3 (a longer
character run, or a second model/tokenizer) still needs materially more compute than remains.
No `STOP` written — wall-clock remains and the plan has open candidates.

On track? yes — S24c done, the direction's named open problem now has both halves (how many units, and
what they detect), and the answer came from data the assay never touched; blocker: none.

## 2026-08-12 (same iteration, second half) — S24c made causal

**What I did.** With wall-clock left after the tuning result, I ran the successor the JOURNAL entry
above had named as "next step", because it was cheap (22 s) and it removes the main weakness of that
result: a ranking agreeing with another ranking is not a causal claim.
`experiments/neuron_feature_causal.py` reuses `neuron_path.py`'s chord machinery with one change —
the units to linearize are chosen by corpus tuning $|z_a - z_b|$, blind to $d(t)$, to $I_j$ and to the
pair's own curve.

**What the numbers say.** 32 corpus-selected units remove **28.9%** of the trained→untrained width gap
(0.351 → 0.482, 98% of pairs widen) against 1.2% for 32 random units. The comparison I care about is
against the assay-derived *global* set — the best previous rule that does not see the individual pair —
which removes 19.0% at the same size ($p = 2.7\times10^{-11}$): a rule fitted on ordinary text beats a
rule fitted on the assay itself, as long as neither is allowed to see the pair. And it sits below the
per-pair fitted ceiling (50.9%, $p = 7.3\times10^{-26}$), which is exactly the ordering a held-out
prediction should show against one fitted on its own test curve. Two free checks passed: the
unmodified baseline reproduced per pair to 0.0000 (same checkpoint, same 150 pairs as S24b) and the
worst endpoint deviation was 1e-6.

**What I got wrong within the same iteration.** The caveat I wrote an hour earlier — "correlation
between two measurements, not an intervention" — was true when written and false by the end of the
iteration; I replaced it in both files rather than leaving it as a hedge, and stated the residual that
actually remains (28.9% vs the fitted 50.9%, so tuning names much of the responsible population and
not all of it). Worth noting as a pattern: a caveat is a claim too, and it goes stale like any other.

**Next step.** The open half is what the *other* half of the responsible units respond to — the corpus
rule finds a bit more than half of the fitted ranking's effect, and conditioning tuning on the current
character only cannot describe a unit that responds to a longer pattern; conditioning on the
(previous, current) bigram over the same corpus pass would test that directly and needs no training.
PLAN S24 item 3 (a longer character run, or a second model/tokenizer) still needs materially more
compute than remains. No `STOP` written — wall-clock remains and the plan has open candidates.

On track? yes — the tuning result is now causal and held out, and both deliverables carry it with an
honest residual; blocker: none.

## 2026-08-12 (later still, III) — S24d: the residual has a name, and the obvious fix is refuted

**What I did.** Part C first: listed the direction root for `human_feedback*.md` / `*REVIEW*` lacking
`.addressed.md` — none of the seven, so this iteration advanced the plan. I also checked `results/`
against the deliverables (the lesson from three iterations ago, when a finished run sat unreported on
disk) — everything there is reported. `/tmp/dir13_frozen/checkpoints_ref_pos` and
`/tmp/tinyshakespeare.txt` had both survived again, so no retraining.

The target was the residual the previous iteration named as its next step: corpus character tuning
selects 32 units that remove 28.9% of the width gap, the pair's own fitted top-32 removes 50.9%, so
roughly half the effect lives in units the corpus rule does not find. The stated suspect was the
conditioning — a profile over single characters describes a unit that responds to a two-character
pattern badly. `experiments/neuron_bigram.py` tests that from one further corpus pass tabulating
activations against the (previous, current) character pair, which supports both halves at once: a
weighted two-way decomposition of each unit's bigram table (how much of a unit *is* the current
character), and a context-matched tuning profile restricted to positions after a space — which is the
assay's own context, since every interpolated character sits in `"The house was ␣X"`.

**What the numbers say.** The descriptive half confirmed the suspicion cleanly. Units the character
ranking finds are near-pure character detectors — median 96% of their corpus response explained by the
current character alone; the ones it misses sit at 51%, with the interaction share rising from 18% to
49% (p = 1.4e-186) against a population median of 37%. And they are not ranking noise: 8 missed
recruits remove 11.5% of the gap against 29.1% for 8 found recruits at matched set size (p = 1.2e-20),
where 8 random units remove about 1%. So the second population is real, context-dependent, and weaker
per unit.

The causal half refuted the fix. The context-matched rule is a *better ranking* — mean AUROC 0.886 vs
0.869 on the same 84 pairs, p = 1.4e-5 — and a *worse selector*: its top 32 remove 21.9% of the gap
against the current-character rule's 31.9% (p = 1.9e-11), on pairs where the ceiling is 52.6%. That
dissociation is the most useful thing here and it is not a contradiction: precision@32 already fell
(20.3% vs 25.6%). Conditioning on the preceding space splits the corpus roughly fourteen ways, so each
cell is estimated from far fewer positions, and the noise lands hardest at the very top of the
ranking — the only part an intervention that edits 32 units ever reads.

**Assumptions logged (loop mode, no one to ask).** (a) "Context" here is the previous character only.
A richer conditioning (more history, or a learned feature) might select the missed units well; I say so
in both files rather than concluding that context-dependence is unreachable. Rejected alternative:
conditioning on a longer suffix, which makes the sparsity problem that already sank the bigram rule
strictly worse. (b) The found/missed split cuts a graded quantity (the character ranking's own rank) at
the top decile, and the ranking that defines "found" is the same one whose misses are being described —
stated as a caveat; the matched-size ablation is what makes the split more than a definition. (c) Cells
with < 20 occurrences are dropped and only 47 of 65 characters occur ≥ 100 times after a space, so the
like-for-like comparison uses 84 of 150 pairs; I re-scored *every* k=32 rule on exactly those 84 rather
than comparing a restricted rule against a full-sample one, which would have flattered the new rule's
opponent.

**What I learned.** A ranking metric and a selection metric can disagree, and which one you believe
depends on what the downstream use reads. AUROC integrates over the whole 3,840-unit ordering;
the intervention reads only the top 32. Improving average order while degrading the top is a perfectly
coherent outcome, and it would have been invisible had I reported AUROC alone — I would have written
"bigram conditioning improves the description" and been wrong about what it buys. Worth carrying: when
a measurement exists to serve an intervention, score it the way the intervention consumes it.

**Deliverable work.** New Results section in both files with Figure 32 (`plots/neuron_bigram.png`,
three panels) and a new REPORT.md Methods subsection defining the bigram profile, the two-way variance
decomposition (current / previous / interaction shares) and the context-matched score
$D^{\sqcup}$. The two places that named the residual as unestablished (REPORT Results caveats, RESULTS
caveats) now point to the answer; REPORT Summary, Conclusion and Limitation 7 and the RESULTS
hypothesis paragraph carry it. Exploratory Figures 32–34 renumbered 33–35; 35 embeds / 35 captions /
sequential 1–35 in each file; `check_render.py REPORT.md RESULTS.md` passes with 0 problems.

**Next step.** The mechanism question is now answered on both halves as far as single-character and
bigram corpus statistics can take it. The honest successor is a *learned* description of the missed
units — e.g. fit a linear probe on their activations over the corpus and ask what it reads — which
would test whether "context-dependent" resolves into nameable features or stays diffuse; it needs no
training of the model itself. Everything else in PLAN S24 item 3 (a longer character run whose second
local-complexity descent separates from initial fit, the denser Figure-9 grid on the pilot run's local
maximum, or a second model/tokenizer) still needs materially more compute than remains. No `STOP`
written — wall-clock remains and the plan has open candidates.

On track? yes — S24d done, the residual half of the mechanism is characterised and the obvious fix for
it is tested and refuted, with both results in the deliverables; blocker: none.

---

## 2026-08-12 (S24e) — fitting the description of the missed units, and an unexpected correction

**Feedback check first.** Listed the direction root: all seven `human_feedback*` files end in
`.addressed.md`. Zero unaddressed feedback, so this iteration advanced the plan — and PLAN's own
"Next step" named exactly this experiment (a *learned* description of the context-dependent recruits
in place of another hand-built conditioning).

**What I ran.** Two scripts, no training, 44 s of forward passes in total, on the same reference
character run (`/tmp/dir13_frozen/checkpoints_ref_pos`, step 30,000, seed 1337) and the same 150 pairs
as every intervention in this series.

1. `experiments/neuron_probe.py`. One corpus pass over the model's own 90% training split accumulates
   ridge sufficient statistics for a regression predicting each block-1–4 unit's post-GeLU activation
   from one-hot codes of the eight characters ending at the position, plus a full previous×current
   interaction table (4,746 features, 3,840 units). Because every feature group is one-hot, the Gram
   matrix is assembled from `bincount` co-occurrence blocks and the cross term from `index_add`, which
   is why the whole pass takes 8 s rather than minutes of dense matmul. Windows split 80/10/10 by
   index: fit, choose $\lambda$ from $10^0..10^4$, report. Nested models $L=1,2,4,8$ are column
   prefixes of the same Gram matrix, so all five fits come from one pass. Then the fitted probe is
   evaluated at the assay's exact context and its predicted endpoint difference used as a *blind*
   selection rule fed to `neuron_path.py`'s chord linearization.
2. `experiments/neuron_probe_control.py`. The probe rule differs from the earlier character rule in
   two ways at once (it knows the context, and it is in raw activation units rather than $z$-scored),
   so this runs the two rules that isolate each: the character profile with standardization removed,
   and the probe's prediction re-standardized.

**What came out.** (i) The missed recruits are *describable*: median held-out $R^2$ 0.29 (current
character) → 0.53 (eight characters) → 0.78 (plus interaction), against 0.92 → 0.97 for the found
ones. They read short, local character context. (ii) The probe rule's top-32 removes **56.5%** of the
width gap — beating the pair-fitted top-32's 50.9% (paired $p=2.3\times10^{-17}$). A rule blind to
$d(t)$ beating one fitted to it is only a contradiction if you call the fitted ranking optimal; it
ranks by *individual* importance $I_j$, so a jointly better set is reachable. (iii) **The correction
I did not expect:** the controls say scale, not context, carries the win — the plain character profile
with per-unit standardization removed already reaches 56.3%, while the fitted context alone reaches
34.8%. S24d's diagnosis (the bigram rule fails on estimation noise at the top of the ranking) is not
wrong but is second-order: both earlier rules were handicapped mainly by $z$-scoring, which asks which
character a unit prefers when the intervention cares how many activation units the swap moves.

**Assumption logged.** I compared rules on all 150 pairs rather than S24d's 84-pair well-sampled
subset, because the fitted probe is defined for every character and restricting would only weaken $n$;
the rejected alternative was re-scoring on the 84 pairs for continuity with the previous section. Both
figures state which pair set each number uses.

**Deliverable work.** New Results section in both files with **Figure 33** (`plots/neuron_probe.png`,
three panels) and a new REPORT.md Methods subsection defining the probe, held-out $R^2$, the context
and interaction gains, the probe selection score $D^{\mathrm{probe}}$ and the two decomposition
controls. REPORT Summary, Conclusion and Limitation 7 and the RESULTS hypothesis paragraph and
Headline carry the result and the corrected reading of S24d. Exploratory Figures 33–35 renumbered
34–36; 36 embeds / 36 captions / sequential 1–36 in each file; `check_render.py REPORT.md RESULTS.md`
passes with 0 problems.

**Next step.** The "what do the responsible units compute" thread is now closed as far as text
statistics reach: the units are identified, described out of sample, and a text-only rule selects them
better than the assay-fitted ranking at small $k$. The natural successor is the same scale insight run
backwards — rank units by predicted displacement rather than by preference for *every* earlier rule in
the report (the global set, the per-block scan) and see whether other conclusions shift — but that is a
consolidation, not a new question. Everything else open is PLAN S24 item 3 (a longer character run
whose second local-complexity descent separates from initial fit, the denser Figure-9 grid on the pilot
run's local maximum, or a second model/tokenizer) and needs materially more compute than one
30,000-step run. No `STOP` written — the plan retains open candidates.

On track? yes — S24e done, the missed units are described from held-out corpus data and the selection
limit is re-attributed from conditioning to scale, with both results in the deliverables; blocker: none.

---

## 2026-08-12 — S24f: does the intervention read activation units or residual displacement?

**Feedback check first.** `ls` of the direction root shows seven `human_feedback*` files, all already
ending in `.addressed.md`. Zero unaddressed, so this iteration advanced the plan — and PLAN's own
"Next step" named exactly this consolidation (rank by predicted *displacement* rather than preference,
for the rules the report already carries, and see whether other conclusions shift).

**Why this experiment.** S24e ended with "keep each unit's own activation scale", which left two
loose ends that both have an obvious next move. (1) *Which* scale: a hidden unit only reaches the
residual stream through its write vector, so the physically natural score is
$|\Delta a_j|\cdot\lVert W_{\mathrm{proj}}[:,j]\rVert$, and every blind rule so far ignores that norm.
(2) *Is the corpus estimate the limit?* The corpus rules predict an endpoint swing that can simply be
read off the network — an oracle for the same quantity with the estimation error removed.

**Pre-registered before running** (in the script docstring, `experiments/neuron_scale.py`): P1 the two
write-norm-weighted rules each gain ≥2 points at $k=32$; P2 the oracle beats every corpus-estimated
rule and the pair-fitted 50.9%; P3 the write norm alone recovers <20%.

**What came out — P1 and P2 refuted, and that is the result.** Write-norm weighting *hurts*: character
rule 56.3% → 55.4% (paired p = 0.049), probe 56.5% → 56.6% in median but worse on 62% of pairs
individually (p = 2.7e-4), oracle 56.6% → 55.3% (p = 1.1e-9). The reason is in the weights themselves —
write norms span a factor of only 1.71 between the 5th and 95th percentiles, so $n_j$ carries almost no
information and mostly reshuffles the top of the ranking. The oracle *ties* the text-only probe
(56.6% vs 56.5%, p = 0.27): a rule built from Shakespeare character statistics is as good as one that
watches the network do the thing, so at $k=32$ these rules are not estimation-limited. P2's second half
did hold — weighted endpoint displacement beats the pair-fitted curvature ranking, 55.3% vs 50.9%
(p = 2.2e-17), while sharing a median 20 of 32 picks, so endpoint displacement is a better score than
maximum deviation from the chord. P3 held far more strongly than predicted: the write norm alone
removes 0.3% at $k=32$, *below* random's 1.2%.

**What I learned.** The selection thread has a ceiling, and it is not where I expected. The limit on a
text-only rule is the *form* of the score — every rule here ranks units by individual displacement and
none can see that a set works better jointly than its members do apart — not the quality of the corpus
estimate. Also a clean negative worth keeping: the "a few loud units dominate the residual stream"
intuition is false in this model; the units are near-uniform writers and what distinguishes the ones
that carry a plateau boundary is entirely what they detect.

**Assumption logged.** I scored all five rules at the same three set sizes (8, 32, 128) used by the
S24e control rather than the full 13-point $k$ grid of `neuron_path.py`, because the comparisons that
matter are paired against rules already measured at those three sizes; the rejected alternative was the
full grid, which would have cost ~4× the compute to add points no claim uses. Consequence: the $k=128$
statements are read off three points, not a curve.

**Deliverable work.** New Results section in both files with **Figure 34** (`plots/neuron_scale.png`,
three panels) and a new REPORT.md Methods subsection defining $n_j$, the weighted rules, and the
endpoint-swing oracle $E_j$ with its relation to the pair-fitted $I_j$. REPORT Summary, Conclusion and
Limitation 7, and the RESULTS.md hypothesis paragraph and Headline, carry the ceiling result.
Exploratory Figures 34–36 renumbered 35–37; 37 embeds / 37 captions / sequential 1–37 in each file;
`check_render.py REPORT.md RESULTS.md` passes with 0 problems.

**Next step.** The individual-displacement family is closed. The one score still ahead at $k=128$ is
the pair-fitted ranking, and the reason is now named rather than guessed: it is a *joint* effect that
no per-unit score can see. The honest successor is therefore a greedy or residual-corrected selection —
pick the next unit by how much of the *remaining* bend it removes given the ones already linearized —
which would test directly whether joint effects account for the whole gap; it needs forward passes
only, but ~k× the passes of one ranking. Everything else open is PLAN S24 item 3 (a longer character
run whose second local-complexity descent separates from initial fit, the denser Figure-9 grid on the
pilot run's local maximum, or a second model/tokenizer) and needs materially more compute than one
30,000-step run. No `STOP` written — the plan retains open candidates.

On track? yes — S24f done, the text-only selection score is shown to be at the ceiling of its family
with two pre-registered predictions refuted and the remaining limit identified as joint rather than
individual effect; blocker: none.

## 2026-08-12 — S24g: are the units interacting? Residual-corrected (greedy) selection

**Feedback check first.** `ls` of the direction root shows seven `human_feedback*` files, all already
ending in `.addressed.md`. Zero unaddressed, so this iteration advanced the plan — and PLAN's own
"Next step" named exactly this: a greedy or residual-corrected selection that picks the next unit by
how much of the *remaining* bend it removes, to test directly whether joint effects account for the
gap left by the per-unit rules.

**Why this experiment.** Every selection rule in this series scores units one at a time and takes the
top $k$. S24f ended by naming that as the one remaining explanation for why no rule reaches the 86.7%
that linearizing all 3,840 units removes — and it was an explanation, not a measurement. Making the
selection sequential turns it into a measurement: build the set in $R$ rounds and re-measure every
unit's importance with the units already chosen straightened, so a unit is scored by the bend still
left for it to carry. $R=1$ is *identical* to the one-shot pair-fitted rule, so it is simultaneously
the control and a free reproduction check, and the only thing that changes as $R$ grows is that later
rounds see the network after the earlier picks are linearized. Any gain is therefore a joint effect and
nothing else.

**Implementation note.** `experiments/neuron_greedy.py` subclasses `neuron_path.ChordMLP` with one
extra mode, `ablate_record`, that linearizes the current set *and* records the resulting activations,
so one forward pass per round yields the residual curvature of every remaining unit. The chords never
need refreshing: the chord equals the true activation at $t=0$ and $t=1$, so linearizing a set leaves
both endpoints — and hence every other unit's chord — untouched. Whole run: 77 s, 150 pairs,
$R\in\{1,2,4,8\}$ at $k\in\{32,128\}$.

**Pre-registered before running** (in the script docstring): P1 greedy $R=4$ at $k=32$ gains ≥5 points
over $R=1$; P2 recovery is non-decreasing in $R$ at both $k$; P3 greedy $R=8$ at $k=32$ clears the
56.6% of the best per-unit rule.

**What came out — P1 and P3 refuted, P2 only at the larger set.** At $k=32$ sequential selection buys
nothing: 50.9% → 51.3% → 49.8% → 49.8%, paired $p=0.24/0.41/0.43$, and 50.7% of pairs not worse — a
coin flip. At $k=128$ it is real, monotone and broad: 68.4% → 70.7% → 71.1% → **71.8%**, every step
significant ($p$ down to $6.1\times10^{-21}$, and $R{=}4\rightarrow8$ alone at $p=6.0\times10^{-6}$),
84.7% of pairs not worse, median width gain +0.0145. Eight rounds keeps a median 100 of 128 picks
(and 26 of 32), so the gain is a reallocation of about a fifth of the set, not a different circuit.
Free checks all exact: $R=1$ reproduced `neuron_path.py`'s per-pair widths to 0.000000 at both $k$,
baseline to 0.000000, worst endpoint deviation $10^{-6}$.

**What I learned.** Joint effects exist but are worth about a fifth of the remaining distance to the
ceiling, and only in the tail of the ranking. The contrast between the two set sizes is the real
content: the leading units carry the bend nearly independently of one another — which is why scoring
them alone is already near the best a size-32 set can do — while the tail is close to interchangeable,
so re-measuring can reallocate it. That also **corrects last iteration's reading**. S24f attributed the
$k=32$ gap (blind rule 56.5% beating the fitted ranking's 50.9%) to the fitted ranking being blind to
joint structure; a rule that sees joint structure perfectly does no better at $k=32$, so the cause is
the per-unit score's *form* — endpoint displacement versus path curvature — which S24f had itself
already isolated. The mechanism thread now closes with a bounded number rather than an open
possibility: what stays uncharacterised is not *which* units get chosen but why so much of the
sharpness needs hundreds of them.

**Assumption logged.** I used equal-sized rounds up to $R=8$ rather than exhaustive one-at-a-time
greedy, because full greedy scores each candidate by its measured effect on $d(t)$ and costs ~$k$×
more forward passes than the whole batched sweep; the rejected alternative was full greedy at $k=32$
only, which would have answered the question at the one set size where the batched version already
shows a flat curve. Consequence, stated in both deliverables: 3.4 points is a *lower bound* on what
joint selection can reach, not the maximum attainable by any 128 units.

**Deliverable work.** New Results section in both files with **Figure 35** (`plots/neuron_greedy.png`,
three panels) and a new REPORT.md Methods subsection defining the residual importance $I^{S}_j$, the
round schedule, and the set-overlap diagnostic. REPORT Summary, Conclusion and Limitation 7, and the
RESULTS.md Headline, carry the bounded joint-effect result and the corrected reading of S24f.
Exploratory Figures 35–37 renumbered 36–38; 38 embeds / 38 captions / sequential 1–38 in each file;
`check_render.py REPORT.md RESULTS.md` passes with 0 problems.

**Next step.** The selection thread is finished: the units are identified, described from held-out
corpus text, selected better by a blind rule than by the assay-fitted one at small $k$, shown to be
near-uniform writers, and now shown to interact only weakly. The one question it leaves is different in
kind — why hundreds of units are needed for the last third of the sharpness when a few dozen carry the
first half — which is a question about the *distribution* of the bend, not about ranking, and the
honest instrument for it is a saturation curve of recovered fraction against $k$ with pair-level
resolution (already partly in `neuron_path.py`'s 13-point grid). Everything else open is PLAN S24
item 3 (a longer character run whose second local-complexity descent separates from initial fit, the
denser Figure-9 grid on the pilot run's local maximum, or a second model/tokenizer) and needs
materially more compute than one 30,000-step run. No `STOP` written — the plan retains open candidates.

On track? yes — S24g done, joint unit interactions are bounded at 3.4 points (≈ a fifth of the
remaining gap) at $k=128$ and zero at $k=32$, with two pre-registered predictions refuted and last
iteration's attribution corrected in both deliverables; blocker: none.

## 2026-08-12 — operator feedback #7: narrowing the claim from decision basins to character basins

**Feedback check first.** `ls` shows `human_feedback_7.txt` without `.addressed.md`, so this iteration
is feedback work, not plan work. `.tasks/human_feedback_7.txt.manifest.json` arrived in state `triage`.

**Triage (no research).** The feedback names no output file. The claim it targets — "one next-character
decision basin per character" — is a headline/conclusion claim of the two curated deliverables that
carry it, and both have the literal sections the operator names: RESULTS.md has `## Headline` (and the
`## Question & verdict` summary at the top), REPORT.md has `## Conclusion` (and the `## Summary`
verdict). So the routing is REPORT.md + RESULTS.md, and there is nothing materially ambiguous to block
on. `REPORT_followup.md` is feedback #5's deliverable (frequency / asymmetry / character-class
re-analysis); it makes no decision-class claim, so it stays unchanged. Manifest set to `in_progress`
with those routing fields recorded.

**What I found.** The supporting analysis the operator quotes already exists in both files — the
Results subsection "What labels a basin: the endpoint character, not the decision class" with Figure 16
and `results/basin_decision.json` (15 distinct endpoint predictions, 31.6% of paths at exactly two,
median 3, 9.9% of pairs sharing an endpoint prediction). It arrived in the previous iteration and,
oddly, was never journaled or changelogged, so I recorded its provenance in the CHANGELOG entry. The
Summary verdict and the RESULTS.md Headline had already been narrowed with those counts. What had
**not** been narrowed was the Conclusion, which still read "Our reading: **a plateau here is a
next-character decision basin**" and, one paragraph later, "what the basins *index* is the model's
next-character decision". That is exactly the conflation the operator is objecting to, so no new
measurement was needed — only the claim had to come down to the evidence.

**What I changed.** REPORT.md Conclusion: the reading is now a character-conditioned logit-space basin
whose transition coincides with a prediction change, with the three counts stated inline and an
explicit refusal of the one-basin-per-character phrasing. The Interpretation paragraph now says the
basins are indexed by the endpoint character and keeps the 91% coincidence as a coincidence between two
labellings. Limitation 7 was re-titled from ""Decision basin" is a description, not a demonstrated
mechanism" to "The next-character decision neither labels the basins nor explains them", with the
labelling failure first and the (unchanged) mechanism argument second. RESULTS.md Question & verdict
item 5 gained the three counts. Consistency edits in both files: Figure 21's caption said the plateau
boundary *is* the decision boundary — now *coincides with* a prediction change, pointing at Figure 16;
"the decision reading survives" → "the coincidence survives"; and the hypothesis paragraph's "decodes
to the same prediction … stands as a description" is now "at best a loose description", because one
prediction can label the basins at both ends of a path and the typical path visits a third.

**What I did not change.** Every number. The rest-ratio criterion and its null families, 59/65
characters holding a basin, the frequency correlation, the 78% per-character variance share, every
transition width, intervention, frozen-block run and unit-selection result stand as measured — the
operator anticipated this, and it is true here for the same reason it was true for feedback #6: all of
those are computed from the geometry of $d(t)$, not from the decision labelling.

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 49 display / 1,059 inline equations / 39 figures; RESULTS 39 figures; 0 problems). Manifest set
to `review_pending`. No `STOP` written and no feedback file renamed — the wrapper does that after the
independent content review.

On track? yes — feedback #7 answered by narrowing the Conclusion, Interpretation, Limitation 7 and the
RESULTS verdict to what the prediction counts support, with no measurement affected; blocker: none.

## 2026-08-12 — S24h: band decomposition of the unit ranking (why the tail costs hundreds of units)

**Feedback check first.** `human_feedback_7.txt` is still un-renamed, but its manifest is
`review_pending` with the single checklist item `done` — the wrapper's independent reviewer has not run
yet and there is nothing for me to repair. I re-verified the fix rather than redoing it: `grep` over
REPORT.md / RESULTS.md finds no surviving identity claim (the only "decision basin" hits are the two
sentences that explicitly *refuse* the phrasing, plus the readout `t_gap` decision boundary, which is a
different, separately defined object). So this iteration advanced the plan.

**What PLAN's "Next step" asked for.** A saturation analysis of $\rho(k)$ with pair-level resolution:
does the tail of the ranking behave like many small independent contributions, or like a second
population of units with a different character profile? The nested-prefix curve in `neuron_path.py`
cannot answer it — $\rho(k)$ confounds "units out there are individually weaker" with "units out there
only work together".

**Design (`experiments/neuron_bands.py`, 63 s of forward passes, no training).** Cut the per-pair
importance ranking into six bands (0–8, 8–32, 32–128, 128–512, 512–2,048, 2,048–3,840) and linearize
each band **alone**; also the nested prefix at each band edge (a free reproduction of `neuron_path`)
and two size-matched random controls. The second control is the one that matters and it is new: draw
the random set only from the units ranked at or below the band's own lower edge, so for the
512–2,048 band it is 1,536 units drawn from the 3,328 ranked ≥ 512. My first run used only the
draw-from-all control and it was misleading — a random 1,536-unit set contains ~40% of the top-32
units, so it "recovers" 40.6% and makes the tail band look worse than chance. Same 150 pairs /
context / block 0 / step 30,000 as the rest of the section.

**Result 1 — redundant, not independent.** Band-alone: 27.9 / 25.9 / 24.1 / 20.4 / 10.4 / −0.1%.
Sum 111.5% against the 85.2% all-units ceiling (ratio 1.29, paired $p=6\times10^{-20}$, 86.7% of
pairs). Every band alone exceeds its marginal contribution inside the prefix. So the answer to the
first half of the question is neither "independent" nor "jointly necessary": the bands overlap.

**Result 2 — the ranking has signal to rank 2,048 and none below.** Each band beats a same-size draw
from its own region (0.15 / 1.19 / 2.24 / 3.90% for bands 2–5 against 25.9 / 24.1 / 20.4 / 10.4%,
$p\le10^{-25}$, 98–100% of pairs), while the last 1,792 units bend nothing (−0.1%). Per-unit worth
falls ~500-fold down the ranking (34.8 → 0.067% per 1,000 units).

**Result 3 — a continuum, not a second population.** Assign each unit to the band of its *best* rank
over the 150 pairs (mean rank was useless: a unit that matters for one pair sits near the middle on
average, and my first version put zero units in the head), then read off `neuron_probe.py`'s fitted
text description. Held-out $R^2$ declines smoothly: full description 0.97 / 0.70 / 0.66 / 0.59 / 0.52 /
0.50; current-character-only 0.91 / 0.30 / 0.22 / 0.14 / 0.12 / 0.13. Head (best rank < 32, n=668) vs
tail (best rank ≥ 512, n=1,623): 0.80 vs 0.51 ($p=2\times10^{-67}$) and 0.42 vs 0.12
($p=2\times10^{-97}$), Mann–Whitney over distinct units, no pseudo-replication across pairs. This is a
rank–describability association at one checkpoint, not a mechanism, and it is written that way.

**Assumption logged (rule 1).** Band edges are the dyadic-ish edges the existing $k$ grid already
uses, so the prefix column reproduces `neuron_path`'s published $\rho$ at $k=8\ldots3{,}840$
(0.279 / 0.473 / 0.656 / 0.825 / 0.853 / 0.852 here against 0.300 / 0.509 / 0.684 / 0.836 / 0.868 /
0.867 there — same curve, small differences from the per-pair rather than median-of-medians
definition of $\rho$). Rejected alternative: a per-pair parametric saturation fit, which would have
put a model between the reader and the measurement for no gain.

**Curation deferred by one iteration, deliberately.** REPORT.md and RESULTS.md are feedback #7's
declared outputs and are awaiting the independent content review; adding a figure to the S24 section
renumbers every later figure in both files, which would land a large unrelated diff in the middle of
that review. PLAN's "Next step" now carries the curation as step (i), ahead of the successor
experiment. The figure and both result files are on disk (`plots/neuron_bands.png`,
`results/neuron_bands_{raw.npz,summary.json,log}`).

On track? yes — the saturation question in PLAN's "Next step" is answered and its successor is named
(does the redundancy ratio grow with training? repeat the decomposition on an early checkpoint and on
the frozen-early run); blocker: none, but the S24h figure is not yet in the deliverables by design.

**S24h, second and third arms (same iteration).** With the one-checkpoint decomposition done and time
left, I ran the developmental version PLAN's successor named, then the question it in turn raised.

*Second arm — `experiments/neuron_bands_time.py` (96 s).* Band-alone and all-units runs at steps
831 / 2,038 / 5,000 / 12,500 / 30,000. Early checkpoints have fewer pairs with a usable
trained→untrained gap (94 / 140 / 146 / 148 / 150 pass the 0.10 filter), so the per-checkpoint rows
are not comparable as-is; the script therefore also reports the trend on the **94 pairs usable at
every checkpoint**, and that is the row I read. Redundancy ratio: **1.21 / 1.01 / 1.08 / 1.21 / 1.18**
— flat, with a dip at step 2,038. So the prediction I wrote into PLAN last iteration ("does redundancy
grow with training?") is **not supported**: the overlap between bands is there as soon as there is a
bend to share. What does grow is how much of the bend those units carry at all (all-units effect
46.2 → 81.0%), concentrated at the top of the ranking (top-8 band 7.2 → 23.9%, deep tail 0 throughout).

*Third arm — `experiments/neuron_head_identity.py` (14 s, no ablations).* The obvious reading of "the
head strengthens" is that the same units grow, and it is wrong. Median overlap of a checkpoint's
per-pair top-8 with the step-30,000 top-8: **0 / 2 / 4 / 6 / 8** units (top-32: 6 / 10 / 16 / 23 / 32;
chance 0.02 / 0.27). At step 831 the top eight units already remove 7.2% of the gap and **none** of
them is a unit the finished network ranks in its top eight. Consecutive-checkpoint overlap (3 / 3 / 5 /
6 of 8) says the churn continues late. Figure panel (d) added to `plots/neuron_bands_time.png`.

*Framing (rule 9b / rule 7).* Two of the three arms came back against the guess I had written down,
and I let the evidence set the framing rather than the reverse: the S24h story is now "the tail is
weak, redundant and continuous; the redundancy is not what training builds; what training builds is a
head whose membership it keeps rewriting", not "redundancy accumulates". All three arms are
descriptions of one training run at five checkpoints — stated that way in PLAN and to be stated that
way in the deliverables at curation.

On track? yes — PLAN's saturation question answered plus two successors run; blocker: none, the three
S24h figures/panels are on disk and curation into REPORT.md/RESULTS.md is the next iteration's first
item, deferred only because those two files are feedback #7's declared outputs and are in review.
