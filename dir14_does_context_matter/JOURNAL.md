# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---
## 2026-07-25 — iteration 1: S1–S5 in one pass (assay built, both experiments run)

**Feedback check:** listed the direction root; no `human_feedback*.md` / `*REVIEW*` files present, so
this iteration advanced the plan.

**Did.**
1. Confirmed the environment: torch 2.8.0.dev+cu128, transformers 5.14.1, GPT-2 Large downloads and
   caches to `HF_HOME=/workspace/hf_home`. TransformerLens 3.5.1 is installed but I used plain HF
   `GPT2LMHeadModel` + forward hooks — fewer API-version risks, and the hook points I need
   (`attn`, `ln_2` pre-hook for `resid_mid`, `mlp.act`, `mlp`, block output) map one-to-one.
2. `experiments/assay.py` — slerp-rescale interpolation, relative endpoint distance, patch hook at
   `resid_post` of block L (final position only), recorders at all five sites of every downstream
   block + logits, and the frozen curve summaries (width, location, plateau rule) with self-tests on
   synthetic step/line curves.
3. `experiments/manifest.py` — preregistered prompt bank frozen to `results/manifest.json` before any
   curve was looked at: endpoints ` big` (1263), ` large` (1588), ` in` (287), all single-token;
   13 contexts (none / 4 random seed 0 / 4 unrelated / 4 relevant), all prefixes exactly 3 tokens by
   a selection rule fixed in advance.
4. `run_exp1.py` (2 pairs × 36 blocks, 24 s), `run_exp2.py` (13 contexts × 2 pairs × 36 blocks,
   95 s), `analyze.py` (tables, threshold sensitivity, rank-sum tests, geometry control),
   `plot_exp1.py` / `plot_exp2.py` (7 CVD-safe figures).
5. Curated RESULTS.md and REPORT.md from the empty templates; verified with the GitHub markdown API
   (6/6 display equations render, 0 code-block fallbacks, 0 KaTeX errors) and grepped for
   unembedded `(plots/*.png)` paths — none.

**Learned.**
- The reference assay reproduces cleanly and the two endpoint pairs separate hugely under the same
  prefix: `big → in` width 0.050 vs `big → large` 0.592 (straight line 0.800), final logits, patch
  at block 0. So plateau *presence* is mostly an endpoint-pair property.
- Because the preregistered Experiment-2 pair (`big → large`) has no plateau in any context, the
  context question is only answerable on a pair that plateaus. I added `big → in` across the same 13
  contexts as a positive control (same cost, same code) and report `big → large` beside it.
  Alternatives rejected: (a) run only the preregistered pair and report "no context effect", which
  would have been an uninformative null caused by a floor effect; (b) swap the preregistered pair
  silently, which would have broken the preregistration.
- Context effect is real and ordered: none 0.575 ≫ random 0.105 > unrelated 0.074 > relevant 0.048.
  With n = 4 per class only relevant-vs-random separates (exact rank-sum p = 0.029). Context changes
  sharpness, not the transition location (0.437–0.470 everywhere).
- Plateaus are built gradually downstream of the patch: width falls monotonically with recording
  depth (0.71 → 0.07 for `big → in` patched at block 0) and no single sub-layer type owns it.
- Threshold caveat found while writing the summary code: the plateau boolean with a 10%-of-grid
  minimum run accepts the straight line d(t)=t, since that line spends exactly 10% of the path
  within 0.1 of each endpoint. Raised the minimum run to 20% *before* looking at any GPT-2 curve
  (calibrated only on synthetic step/line), and report full threshold sensitivity.
- Endpoint geometry co-varies with the context effect (width vs endpoint cosine, ρ = +0.49,
  p = 0.09). Reported as an open confound rather than dismissed.

**Next step.** The weakest claim is relevant-vs-unrelated (p = 0.49, n = 4). Freeze a second,
larger prefix bank (declared before running, kept separate from bank 1) and re-test the class
ordering as a confirmatory replication; then finalize.

On track? yes — S1–S5 done, S6 (finalization) partly done; blocker: none.
## 2026-07-25 — iteration 1 (continued): confirmatory bank-2 replication, then finalization

**Did.** Froze `experiments/bank2.py` -> `results/manifest_bank2.json` (8 new prefixes per class,
seed 1, asserted disjoint from bank 1, same 3-token rule) and ran the identical assay
(`run_bank2.py`, 24 contexts x 2 pairs x 36 blocks, 2m55s), then `analyze_bank2.py` and
`plot_bank2.py`. Re-curated RESULTS.md and REPORT.md around the replication and re-verified both
with the GitHub markdown API (8/8 figures render in each, 6/6 display equations, 0 KaTeX errors).

**Learned.**
- The bank-1 story was subtly wrong in emphasis. Bank 2 (n = 8/class) separates natural language
  from random tokens decisively (relevant vs random p = 1.6e-4, unrelated vs random p = 3.1e-4) but
  still cannot separate relevant from unrelated (p = 0.083). Pooled over both banks: natural vs
  random p = 8e-7 (medians 0.054 vs 0.141), relevant vs unrelated p = 0.045 (0.049 vs 0.063).
  So the robust ladder is **no context (0.575) >> random tokens (0.141) >> natural text (0.054)**,
  with topical relevance a weak add-on. Updated the verdict in both deliverables and recorded the
  old -> new numbers in CHANGELOG.md.
- Practical note: `np.load` on the 152k-curve npz takes minutes, so `sweep.load_meta()` now reads
  only the summary JSON for analysis scripts that do not need raw curves.
- Bank 2 also shows the `big -> large` (non-plateau) pair going the other way — relevant contexts
  are slightly *wider* (0.585 vs 0.499, p = 0.002). Reported as exploratory, uncorrected.

**Next step.** None — the plan's success criteria are met: source reproduction, endpoint-pair
comparison, four context classes with a preregistered bank plus a disjoint replication bank, all-layer
sweeps, tokenization/endpoint/determinism checks, threshold sensitivity, geometry control, and a
bounded verdict. No unaddressed feedback file exists, so this direction writes STOP.

On track? yes — S1-S6 complete, 100% done, STOP written; blocker: none.
