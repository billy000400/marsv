# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-07-15 — Initial result: plateau emergence over training (3 seeds)

First populated RESULTS.md and REPORT.md (both were TODO templates → full current-best deliverables).

- **New experiment.** Trained the image-models d4/w200 ReLU MNIST MLP (n=1000, AdamW, MSE-on-one-hot,
  100k steps) for seeds 0/1/2, snapshotting 13 log-spaced checkpoints, then ran the frozen plateau
  protocol (perturb `h1`, measure `L3` displacement; matched-random control; plateau contrast on
  ρ∈[0,0.2]; agglomerative-clustering validated stable-region count).
- **Headline numbers (mean over 3 seeds):** plateau contrast rises 0.42 (step 100) → 0.80 (step 100k),
  *after* test accuracy peaks (0.90 by step ~300, decaying to ~0.87). Validated stable-region count
  converges to **10** by step ~300 in every seed. Confident-wrong contrast 0.73 vs confident-correct
  0.85 vs uncertain 0.49 at 100k → confidence, not correctness, drives the plateau.
- **Resolved candidate transient:** the step-10k contrast dip (0.30) is present only in seed 0; seeds
  1–2 rise monotonically (0.56, 0.45). Classified as seed noise, not a split/merge → no escalation.
- **Verdict recorded:** expected monotonic emergence, replicated across 3 seeds.
- **Figures added (all embedded in RESULTS.md + REPORT.md):** training_dynamics.png,
  plateau_curves_by_stage.png, plateau_contrast_and_region_count.png (3-seed band), contrast_by_group.png.
- Confidence metric chosen as max raw output (softmax saturates near 0.23 under MSE-to-one-hot);
  documented in Methods.
