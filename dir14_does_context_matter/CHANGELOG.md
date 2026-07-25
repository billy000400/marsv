# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---
## 2026-07-25 — first results: both deliverables written from empty templates

**RESULTS.md / REPORT.md — created (previously TODO templates, no numbers).**

- **New (S2 reference reproduction, S3 endpoint-pair comparison).** GPT-2 Large, prefix
  `The house was`, 50-step slerp-rescale interpolation of the final-position `resid_post`
  activation, all 36 patch blocks, downstream sites `attn_out`/`resid_mid`/`mlp_post`/`mlp_out`/
  `resid_post` + logits. Final logits, patch at block 0: `big → in` transition width
  w = 0.050 (plateau), `big → large` w = 0.592 (no plateau); straight-line reference w = 0.800.
  Plateau-rule threshold sensitivity: `big → in` 9/9 settings, `big → large` 2/9.
- **New (S4 context comparison).** Same `big → in` / `big → large` transitions under 13 frozen
  contexts (1 none + 4 random + 4 unrelated + 4 relevant, 3 tokens each). `big → in` width medians:
  none 0.575, random 0.105 [0.069, 0.245], unrelated 0.074 [0.045, 0.139], relevant 0.048
  [0.040, 0.068]. Exact rank-sum: relevant vs random p = 0.029; relevant vs unrelated p = 0.49.
  Transition location stable at 0.437–0.470 across classes.
- **New (S5 robustness).** Endpoint fidelity max |Δlogit| = 9.2e-5 (relative 2.4e-6), d(0) and
  1−d(1) ≤ 5.9e-6, deterministic re-run bit-identical (diff 0.0), reference condition run by two
  independent scripts bit-identical. Endpoint-geometry control: width vs endpoint cosine
  Spearman ρ = +0.49 (p = 0.09, n = 13), reported as an open confound.
- **Figures added and embedded in BOTH deliverables:** `fixed_context_endpoint_pairs.png`,
  `exp1_width_by_layer.png`, `exp1_depth_emergence.png`, `exp1_site_types.png`,
  `fixed_transition_contexts.png`, `context_width_summary.png`, `context_effect_by_layer.png`.
- No result was superseded (both files were empty templates before this entry).

## 2026-07-25 — confirmatory replication (bank 2) supersedes the bank-1 context interpretation

**RESULTS.md / REPORT.md — Experiment-2 verdict revised; new section 4 and new figure.**

- **Why.** Bank 1 (4 prefixes per class) could separate `relevant` from `random` (p = 0.029) but not
  the two adjacent contrasts, so it could not say whether the driver was naturalness or topical
  relevance. A second bank of 8 NEW prefixes per class (seed 1, no string shared with bank 1) was
  frozen in `results/manifest_bank2.json` and run with the identical assay before its curves were
  examined.
- **Superseded claim (old -> new).** OLD (bank 1 only): "relevant context sharpens the plateau more
  than random tokens, median 0.048 vs 0.105, p = 0.029; relevant vs unrelated undecided."
  NEW (bank 1 + bank 2, 36 prefixes): the replicated driver is **natural language vs random tokens**
  — pooled medians 0.054 vs 0.141, exact rank-sum p = 8e-7 (n = 24 vs 12); topical relevance is a
  small extra effect, 0.049 vs 0.063 pooled (p = 0.045), not significant within either bank
  (bank 1 p = 0.49, bank 2 p = 0.083). Bank 2 alone: relevant vs random p = 1.6e-4, unrelated vs
  random p = 3.1e-4.
- **Headline numbers updated.** "deleting the context nearly abolishes the plateau (0.575 vs
  <= 0.245 for every 3-token prefix)" -> "0.575 vs <= 0.349 across 36 frozen prefixes"; the
  headline sentence now cites the natural-vs-random effect (~2.6x) instead of the bank-1
  relevant-vs-random effect.
- **Figure added and embedded in BOTH deliverables:** `context_bank2_replication.png`
  (bank 1 vs bank 2 widths by class; bank-2 class medians across the layer sweep).
- Bank-1 numbers are retained in the deliverables as the first-bank result they are, not as a
  competing version: the pooled/replicated statistic is the one the verdict rests on.
- **Exploratory note added:** for the non-plateauing pair `big -> large`, bank 2 shows relevant
  contexts are slightly WIDER than unrelated (0.585 vs 0.499, p = 0.002), flagged as an
  uncorrected secondary comparison.

