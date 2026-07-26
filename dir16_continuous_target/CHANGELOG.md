# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-07-26 — first full result: continuous targets sharply reduce plateaus

**RESULTS.md** and **REPORT.md**: written from the TODO templates to a complete current-best
deliverable (previously both were empty placeholders — no numbers superseded).

New in the deliverables:
- Matched classifier/regressor training on corrupted MNIST (sigma=0.3), 3 seeds, 30,000 steps each.
  Classifier test accuracy 96.15 / 96.54 / 96.38%; regressor test MSE 0.001124 / 0.001114 / 0.001112
  vs baselines 0.02907 (mean target) and 0.01292 (pooled corrupted input).
- Main comparison, linearity deviation (mean_alpha |d(alpha) - alpha|), 90 fixed cross-digit pairs
  x 3 seeds, paired difference classifier - regressor with 95% bootstrap CI over pairs:
  hidden 2  0.1253 vs 0.0263 -> diff 0.0990 [0.0867, 0.1118] (4.8x)
  hidden 3  0.1345 vs 0.0304 -> diff 0.1041 [0.0920, 0.1159] (4.4x)
  output    0.1801 vs 0.0304 -> diff 0.1497 [0.1405, 0.1592] (5.9x)
  Regressor smoother on 90/90 pairs at every layer.
- Second scalar, max normalized jump (plateau-and-cliff signature): output layer 6.64 vs 1.21
  (diff 5.42 [5.10, 5.75]).
- Control: classifiers re-trained and early-stopped at their own validation-loss minimum
  (steps 7,500 / 16,200 / 14,400) are still plateaued (LD 0.1182 / 0.1295 / 0.1792 across layers,
  at most 6% below the fully-trained classifier), so the gap is not an overtraining artifact.
- Robustness: dir12's alternative normalization d^frac gives the same verdict
  (hidden 3: 0.0970 vs 0.0238, diff 0.0732 [0.0663, 0.0805]).
- Eight figures added and embedded as rendered images in BOTH deliverables:
  training_curves, reconstructions, path_reconstructions, hand_selected_curves, mean_curves,
  paired_difference, per_pair_scatter, control_earlystop.

Verdict recorded: POSITIVE — continuous, information-preserving supervision reduces (but does not
abolish) activation plateaus.

**Correction within the same iteration** (before publication, no reader saw the earlier text): the
classifier-accuracy reference was quoting our 10,000-image test accuracy against dir12's
2,000-image-pool number. Both deliverables now give the like-for-like figure: ours 94.45 / 94.75 /
94.85% on test[0:2000] (corrupted inputs) vs dir12's 97.9-98.1% on the same pool (clean inputs);
the 10k-test-set accuracy 96.15 / 96.54 / 96.38% is kept as the headline.

## 2026-07-26 — operator feedback #1: interpolation now evaluated at the best-val-loss checkpoint

**Addressed `human_feedback_1.txt`** ("When you evaluate interpolation, can you pick the checkpoint
that has the best val_loss. Make that explicit in the REPORT.md.") — renamed to
`human_feedback_1.addressed.md`.

What changed in the deliverables:
- `train.py` now saves each model's best-validation-loss checkpoint (`{kind}_best.pt`, val loss
  evaluated every 100 steps on test[2000:10000]); `probe.py` probes it. Selected steps: classifier
  7,500 / 16,200 / 14,400 (seeds 0/1/2), regressor 29,800 (all seeds). All 3 seeds retrained —
  final-step metrics reproduced bit-for-bit, so no other setting changed.
- **REPORT.md has a new Methods subsection "Checkpoint selection"** stating the rule explicitly, with
  its equation, the selected steps, and why (the final iterate is a different kind of solution for
  the two models: the classifier memorizes, the regressor does not).
- Every headline number and all 8 figures are now computed from the best-val checkpoints
  (previously: final step-30,000 weights). Main comparison, linearity deviation, old -> new:
    hidden 2  clf 0.1253 -> 0.1182, reg 0.0263 -> 0.0263; diff 0.0990 [0.0867, 0.1118] ->
              0.0918 [0.0804, 0.1036]; ratio 4.8x -> 4.5x
    hidden 3  clf 0.1345 -> 0.1295, reg 0.0304 -> 0.0304; diff 0.1041 [0.0920, 0.1159] ->
              0.0990 [0.0874, 0.1107]; ratio 4.4x -> 4.3x
    output    clf 0.1801 -> 0.1792, reg 0.0304 -> 0.0304; diff 0.1497 [0.1405, 0.1592] ->
              0.1487 [0.1395, 0.1580]; ratio 5.9x -> 5.9x
  Pairs where the regressor is smoother: 90/90 -> 89/90 (all three layers).
  Max normalized jump, old -> new: h2 2.12/1.14 -> 2.01/1.14 (diff 0.98 -> 0.87);
  h3 2.43/1.21 -> 2.31/1.21 (diff 1.22 -> 1.11); output 6.64/1.21 -> 6.35/1.21 (diff 5.42 -> 5.13).
  d^frac robustness, old -> new (h3): 0.0970/0.0238 diff 0.0732 -> 0.0923/0.0238 diff 0.0685.
  Classifier training quality now quoted at the probed checkpoint: 10k test accuracy
  96.15/96.54/96.38% -> 96.34/96.39/96.37%; dir12 2k pool 94.45/94.75/94.85% -> 95.00/94.50/94.65%.
  Regressor test MSE unchanged (0.001124 / 0.001114 / 0.001112) — its best checkpoint is step 29,800.
- The "early-stopped classifier" control is **superseded**: early stopping is now the main analysis,
  so the control was inverted into a **checkpoint control** (both models re-probed at step 30,000).
  `plots/control_earlystop.png` -> `plots/checkpoint_control.png` (Figure 8 in both deliverables);
  `experiments/control_earlystop.py` and its artifacts deleted as redundant. New control numbers:
  paired difference at step 30,000 is 0.0990 [0.0868, 0.1119] / 0.1041 [0.0927, 0.1161] /
  0.1497 [0.1403, 0.1590] at hidden 2 / hidden 3 / output — same verdict, at most 6% apart.
- Figure 1 (training curves) now marks the best-val checkpoint on every curve and shows both models
  in the rescaled panel (previously classifier only).

Verdict unchanged: POSITIVE — continuous, information-preserving supervision reduces (but does not
abolish) activation plateaus.
