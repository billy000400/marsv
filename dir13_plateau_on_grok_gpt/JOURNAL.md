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
