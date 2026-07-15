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

## 2026-07-15 — Iter 2: figure #4 (region composition + lineage), finalize, STOP

**Feedback check:** globbed dir root for `human_feedback*`/`*REVIEW*` without `.addressed.md` — none. Proceeded.

**Did.**
- `experiments/lineage.py`: for seed 0, recomputed per-example L3 cluster labels at all 13 checkpoints
  under the frozen protocol; reused sweep_seed0.json's per-cluster `valid` flags. Saved
  `results/lineage_seed0.json` (per-example labels + preds + validated digits per checkpoint).
- `experiments/make_lineage_plot.py`: figure #4 = (a) digit×step grid of validated-region presence with
  per-column validated counts, (b) birth-transition (100→300) and (c) late (75k→100k) membership-overlap
  heatmaps; plus a printed split/merge audit across every adjacent transition.
- Embedded fig #4 in RESULTS.md + REPORT.md; added Methods paragraph defining the overlap matrix + split/
  merge/escalation rule; updated finding #2 and limitations. Re-ran the GitHub markdown-API render check
  (3/3 js-display-math, 0 pre-lang-math) and the inline-`$…$` backslash-hazard grep (clean).

**Learned.**
- No predicted digit ever hosts ≥2 validated regions at any checkpoint (max=1) → the key escalation
  signal is absent; monotonic emergence confirmed by membership, not just by the count trajectory.
- The raw split/merge audit shows nonzero counts (total 9 splits / 3 merges) but they are all either
  early region *formation* (births as k grows 2→10) or late silhouette k oscillating 10↔12 — a transient
  sub-threshold split of the uncertain/mixed cluster that is never validated and never persists across two
  adjacent checkpoints. So the "validated-region lineage" is clean (0/0) even though raw-k lineage wobbles.
  Made the figure/caption state this precisely rather than overclaiming "no splits/merges anywhere".

**Next step.** None — all 5 stages complete, verdict is the preregistered expected monotonic emergence,
deliverables current-best with all 4 figures embedded. Wrote empty `STOP`.

On track? yes — S1–S5 all done, 100% complete, no blocker; STOP written (no unaddressed feedback).
