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
