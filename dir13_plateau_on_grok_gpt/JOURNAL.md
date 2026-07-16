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
