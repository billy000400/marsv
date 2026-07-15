# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-07-15 — Iter 1: train checkpoint sweep + plateau analysis, 3 seeds

**Feedback check:** globbed for `human_feedback*`/`*REVIEW*` without `.addressed.md` — none. Proceeded.

**Did.**
- Located the shared codebase at `/workspace/mars-plateaus-image` (dir11 referenced a stale
  `/network/...` path). Reused `src.mnist.MLP` / `load_mnist`, `forward_from(h1, layer=1)`.
- `experiments/train_checkpoints.py`: exact-config training (seed defines subset via manual_seed →
  seed 0 matches the existing endpoint subset), snapshots 13 log-spaced checkpoints + history
  (loss/acc/confidence). Ran seeds 0, 1, 2.
- `experiments/analyze_sweep.py`: frozen protocol — fixed 500-example class-balanced eval set, 16
  shared directions, ρ grid [0,0.6], plateau contrast on [0,0.2] with bootstrap CI, norm+sparsity
  matched-random control, confidence/correctness groups, and validated stable-region count
  (agglomerative + silhouette, cosine & euclidean; size≥20, purity≥0.9, contrast CI>0).
- `experiments/make_plots.py`: 4 figures (auto-uses all available seeds).

**Learned.**
- Softmax confidence is useless here (MSE-to-one-hot → max softmax ≈ 0.23 for all). Max raw output is
  the correct confidence; threshold 0.7 gives clean groups. Logged assumption in Methods.
- Core result replicates cleanly across 3 seeds: contrast 0.42→0.80 while test acc *declines* after its
  step-~300 peak (plateau lags generalization); region count → 10; confidence-not-correctness.
- The step-10k dip is seed-0-only → seed noise, not a real transient. Escalation criterion not met.

**Next step.** Add the remaining required figure #4 (region_composition_and_lineage): predicted-label /
confidence composition of clusters through training + a compact membership-overlap split/merge heatmap
aligning the same eval examples across adjacent checkpoints. After that the deliverable is complete →
finish REPORT.md and write STOP.

On track? yes — S1–S3 done, S4 partial (region counts+composition done, lineage heatmap pending), ~85% done, no blocker.
