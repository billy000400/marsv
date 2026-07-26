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
