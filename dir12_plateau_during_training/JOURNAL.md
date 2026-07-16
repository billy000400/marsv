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

## 2026-07-16 — Iter 3: operator feedback addressed (docs-only iteration)

**Feedback check:** found `human_feedback_07161227.txt` (unaddressed; no stale STOP present). Per
CLAUDE.md Part C this WAS the iteration. Four points:

1. *"add the numerical value of epsilon"* → grepped `analyze_sweep.py`: `EPS = 1e-8`, used in both the
   R_t denominator and the AUC ratio. Stated $10^{-8}$ in the equation + a sentence on why it never
   affects reported values.
2. *"define positive entries, and why it matters"* → the phrase lived in the matched-random-control
   paragraph. Now defined (post-ReLU ⇒ entries ≥ 0; positive entries = active neurons = sparsity) and
   motivated (unmatched norm/sparsity ⇒ contrast could be a scale artifact, undermining the headline
   claim that flatness is learned).
3. *"a lot of in-line symbols are not rendered correct"* → root cause: symbols were in backtick code
   spans (render as literal code), incl. a raw `\arg\max_i M_{ij}`. Converted to inline `$…$` math in
   REPORT.md + RESULTS.md using 8b-safe macros (`\lbrace/\rbrace` instead of `\{`). Verified via the
   GitHub markdown API: 34 inline spans render, 3/3 display blocks, 0 pre-lang-math, hazard grep clean.
4. *"reads like a list of definitions"* → rewrote Methods/Metrics as narrative per CLAUDE.md rule 9:
   opening paragraph maps the research questions to the measurements; each metric gets
   motivation-before-definition ("the obvious measure fails because …") and names the figure that
   consumes it.

No experiments re-run; no numbers changed. Renamed feedback file to
`human_feedback_07161227.txt.addressed.md`; appended CHANGELOG; updating PLAN status; re-writing STOP
(plan complete AND zero unaddressed feedback).

**Learned.** The `js-display-math` API check doesn't catch backtick-wrapped pseudo-math — visually it
"renders" as code and silently reads as unrendered symbols to a human. Worth converting ALL inline
symbols to `$…$` at first writing, not just display blocks.

**Next step.** None — direction complete, feedback addressed, STOP re-written.

On track? yes — S1–S5 done + feedback addressed, 100% complete, no blocker; STOP written.

## 2026-07-16 — Iter 4: redirected PLAN executed end-to-end (SLERP movie), feedback 07161151, STOP

**Feedback check:** found `human_feedback_07161151.txt` unaddressed ("Use the first 2,000 of the
10K test images to test") and a rewritten PLAN.md (new primary experiment: checkpointed
interpolation animation; STOP had been deleted). Both drove this iteration.

**Did.** Built the frozen protocol lib (`plateau_protocol.py`): 55-pair deterministic bank (cross
pair (a,b) = rank-b image of class a + rank-a image of class b; within-class = ranks 10/11 — all
within test[:233] ⊂ test[:2000] per feedback), vectorized SLERP (validated vs `slerp_path`,
9.5e-7), 50 points, patch at h1, record h2/h3/logits + d(α) + preds. S1 on the existing 100k
checkpoint passed (endpoint reproduction 3.7e-4; 24/45 pairs clean plateaus). Trained seed 0 with
205 checkpoints (138 s) + seeds 1–2 with 56 each; manifest check 317/317. Rendered 205-frame GIF,
static frames, step×α heatmap, layerwise plot, seed comparison (one transparent summary: plateau
fraction, floor ≈0.2), training context. Dense 50-step rerun of steps 82,000–82,500 reproduced the
movie records BIT-EXACTLY and resolved the largest late jump (pair 5→6) to a ~150-step boundary
flip. Rewrote RESULTS.md/REPORT.md around the new result; old perturbation study kept as the
plan's optional secondary control. Renamed feedback to `.addressed.md`; wrote STOP.

**Assumptions logged (loop mode, no human to ask).**
- Feedback interpreted as: endpoints AND test metrics restricted to test[:2000] (old work already
  complied, so no rerun of the perturbation sweep was needed).
- Storage: full 50-point h1/h2/h3 raw arrays saved at 16 anchor steps; every checkpoint saves
  state_dict + endpoint activations + logits + d-curves, so every frame is regenerable without
  retraining (verified bit-exact determinism makes even full regeneration cheap). Alternative
  (raw arrays at all 205 ckpts, ~800 MB more in git) rejected to keep the shared repo pushable.
- Seeds 1–2 use the PLAN's fallback 2,000-step density (primary gets the full 500-step movie);
  rejected running them at full density — the comparison only needs emergence timing.
- GIF (pillow) instead of MP4 — no ffmpeg on the box.
- Animation subset preregistered by digit identity: (0,1),(2,3),(4,5),(6,7),(8,9),(0,8),(1,7),
  (3,5),(4,9),(2,6) — every digit exactly twice.

**Learned.** (1) Plateaus are genuinely absent at init — d(α) is the diagonal, so the phenomenon
is 100% learned. (2) Emergence is gradual and asynchronous across pairs; no grokking-style jump in
this metric. (3) The most surprising bit: at step ~82k (train acc 1.0 for ~82k steps) a boundary
can still sweep across the entire path in ~150 steps — late training keeps rearranging region
geometry while test accuracy only wobbles by ~0.03. (4) Within-class paths cross boundaries only
when an endpoint is genuinely misclassified — a nice internal consistency check of the
region-per-predicted-class picture.

**Next step.** None — plan complete, verdict written, STOP present.

On track? yes — S1–S6 all done, 100% complete, no blocker; STOP written (no unaddressed feedback).
