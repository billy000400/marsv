# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-07-26 — iteration 1: S1+S2+S3 complete in one pass

**Feedback check:** listed the direction root for `human_feedback*.md` / `*REVIEW*` without the
`.addressed.md` suffix — **none present**, so this iteration advanced the plan.

**What I did.**
- `experiments/common.py`: shared corrupted-MNIST dataset (fixed-seed sigma=0.3 Gaussian noise,
  clipped; clean 7x7 avg-pooled target), generic-head 4-layer MLP (dir12 config), dir12's SLERP +
  relative-distance code, and the frozen 90-pair bank (2 pairs per unordered digit pair; replica 0
  reproduces dir12's exact pair so `6 -> 7` etc. are the same images).
- `experiments/train.py`: trains classifier and regressor per seed with bit-identical init in the
  three shared layers (asserted at runtime) and identical batch order. Seeds 0/1/2, ~90 s per model.
- `experiments/probe.py`: runs the frozen probe (101-point SLERP at h1) for both models, all seeds.
- `experiments/analyze.py`: aggregates, bootstraps, writes all 8 PNGs.
- `experiments/control_earlystop.py`: the early-stopped-classifier control.

**What I learned.**
- The effect is much larger and cleaner than expected: 4.4-5.9x lower linearity deviation for the
  regressor, significant at every layer, 90/90 pairs, tight across seeds. No pair-level heterogeneity
  to report — the plan's "results vary strongly by pair or seed" branch does not apply.
- The `6 -> 7` output-space figure is the most legible evidence: the regressor morphs the digit
  continuously while the classifier snaps 6 -> 9 -> 7 with flat regions between.
- The regressor does **not** overfit under matched training (validation loss falls monotonically to a
  flat floor), so the plan's "slight overfitting" adequacy criterion is met by the classifier only.
  **Assumption logged:** I kept the training budget matched (same 30,000 steps) rather than training
  the regressor longer to force overfitting, because matched optimization is the more important
  control for this comparison. *Rejected alternatives:* (a) train the regressor to overfitting on a
  reduced subset — breaks the "same inputs, same data order" requirement; (b) train the regressor for
  more steps — breaks matched step count and would confound the comparison. Reported plainly as
  Limitation 2.
- The confound a reviewer would raise first — the classifier memorizes (train MSE ~1e-7) while the
  regressor does not — is now ruled out by the early-stop control rather than only discussed.
- Two metric-definition choices logged: PLAN.md's endpoint-normalized `d(alpha)` is reported as
  primary (it is what the plan literally specifies); dir12's fraction form is computed alongside as a
  robustness check and gives the same verdict.

**Next step.** The plan's success criterion is met (both models trained and verified, hand-selected
and aggregate d(alpha) results, reconstructions along the path, 3 seeds, clear verdict). Remaining
optional polish only: nothing is blocking. Since S1-S3 are all complete and no feedback file is
outstanding, this direction writes `STOP`.

On track? yes — S3 (aggregate & report) complete, 100% done, no blocker.

## 2026-07-26 — iteration 2: operator feedback #1 (best-val-loss checkpoint for interpolation)

**Feedback check:** found `human_feedback_1.txt` (no `.addressed.md` suffix) — "When you evaluate
interpolation, can you pick the checkpoint that has the best val_loss. Make that explicit in the
REPORT.md." Addressing it WAS this iteration. Note it is a `.txt`, not `.md`; treated as operator
feedback anyway and renamed to `human_feedback_1.addressed.md` per CLAUDE.md Part C.

**What I did.**
- `train.py`: track the running best validation loss (already evaluated every 100 steps) and save
  `{kind}_best.pt`, plus best-checkpoint test metrics in summary.json. Retrained all 3 seeds
  (~2 min/seed); the final-step numbers reproduced exactly (seed0 clf test_loss 0.006114,
  acc 0.9615 as before), confirming training is deterministic on this box, so nothing else moved.
- `probe.py`: probes both `{kind}_best.pt` and the final `{kind}.pt`.
- `analyze.py`: all headline stats + all 8 figures now come from the best-val checkpoints; the old
  `fig_control` (early-stopped classifier) became `fig_ckpt_control`, which re-runs the comparison on
  the step-30,000 weights for BOTH models. Figure 1 now marks the probed checkpoint on every curve.
- Deleted `experiments/control_earlystop.py` + `clf_early.pt` / `probe_clf_early.npz` /
  `control_earlystop.json` / `plots/control_earlystop.png`: the early-stopped classifier is now the
  main analysis, so keeping the old control would have been the same experiment twice (CLAUDE.md
  rule 6).
- REPORT.md: new Methods subsection **"Checkpoint selection"** — the rule, its argmin equation, the
  selected steps, and the reason (final iterate is a different kind of solution for each model).
  Both deliverables rewritten to current-best; render check passes (16 embeds, 16 captions, 0 issues).

**What I learned.**
- Selected steps: classifier 7,500 / 16,200 / 14,400; regressor 29,800 for every seed. The
  regressor's best checkpoint is essentially its last one, so only the classifier moves.
- The effect shrinks slightly but survives intact: LD ratio 4.8/4.4/5.9x -> 4.5/4.3/5.9x, all CIs
  still far from 0. Pairs where the regressor is smoother went 90/90 -> 89/90 at every layer.
- Checkpoint choice matters little in absolute terms (<=6% on classifier LD), which is itself the
  cleanest answer to the "is this just overtraining?" objection — it is now built into the main
  result rather than argued in a control.
- Assumption logged: "best val_loss" = minimum validation MSE against each model's own target
  (classifier val loss is MSE-to-one-hot, not accuracy). Rejected alternative: select the classifier
  by max validation accuracy — that would be a different criterion for the two models and would
  break the matching. Worth noting the two disagree slightly: at the val-loss minimum seed 1's test
  accuracy is 96.39% vs 96.54% at step 30,000.

**Next step.** None required — the plan's success criterion is met and the feedback is addressed, so
`STOP` is written. If new feedback arrives, delete `STOP`, address it, re-write `STOP` when clean.

On track? yes — S3 (aggregate & report) complete, 100% done, no blocker.
