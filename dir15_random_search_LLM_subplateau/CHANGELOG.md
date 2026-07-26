# CHANGELOG — Direction 15: Random search for LLM activation sub-plateaus

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-07-26 — first full screen; RESULTS.md and REPORT.md written from empty templates

**What changed.** Both deliverables went from the empty scaffold to a complete current-best report of
the random-pair `A|C|B` screen (stages S1-S6 of PLAN.md, all run this iteration). No previous numbers
existed, so nothing is superseded.

**New results now in RESULTS.md / REPORT.md.**
- Primary screen (frozen): GPT-2 Large, WikiText-103 validation, 32-token windows, blocks 0/2/4/6,
  50 alphas, 1,000 random pairs x 4 blocks x 2 conditioning contexts = 8,000 paths, 7,611 eligible.
  **1,290 candidates = 16.9% of eligible paths, 95% Wilson CI [16.1%, 17.8%]**; pair-level 610/991 =
  61.6% [58.5%, 64.5%]; clean `A,C,B` 283 (21.9% of candidates, 3.7% of eligible paths).
- Per block: 8.2% / 15.4% / 16.4% / 27.7% at blocks 0 / 2 / 4 / 6. Conditioning on context A 17.2%
  vs context B 16.7%.
- Validation bank (300 disjoint pairs, rule unchanged): 401/2,261 = **17.7% [16.2%, 19.4%]**.
- Controls: self-pairs 0 eligible / 0 candidates; linear interpolation 16.1% [14.5%, 17.8%];
  same-prediction pairs 11.1% [9.5%, 12.9%]; foreign-endpoint transfer 17.6%, rate on the
  transfer-consistent subset 14.0% [12.3%, 15.9%].
- C-region character: top-1 prob 0.227 vs 0.323 at endpoints; entropy 6.97 vs 5.70 bits; 39.9% of
  candidates have margin > 0.05, 3.6% > 0.2; 32.3% of C tokens are among the 10 commonest endpoint
  tokens.
- Nearest natural activations (2,000 held-out contexts): median cosine distance 0.160 (C-region) vs
  0.140 (A), 0.153 (B), 0.086 (natural query); top-10 neighbour label agreement 4.5% vs 8.1% / 8.1% /
  14.1%.
- Continuations (6 frozen inspected candidates): all six produce fluent C-region English; identical
  greedy prefix across the C run of 20, 20, 8, 1, 1, 1 tokens; in 6/6 the same C activation under the
  other endpoint's context reverts to that context's own continuation.
- Validity: endpoint fidelity max|dlogit| 1.5e-05 with top-1 reproduced on 100% of paths; 20-path
  re-run 0/1,000 top-1 mismatches; batching invariance 0 top-1 changes.

**Figures added (all embedded in BOTH deliverables).** `candidate_prevalence_by_layer.png`,
`top_candidate_probability_paths.png`, `segment_width_margin_distribution.png`,
`c_region_confidence.png`, `intermediate_token_census.png`, `threshold_sensitivity.png`,
`natural_neighbor_comparison.png`, `continuation_stability.png`.

**Correction made before publication (never shown in a deliverable).** The first analysis pass mixed
up which end of a path is the "own" endpoint under each conditioning context, reporting an endpoint
match rate of 52.4% and a transfer rate of 8.2%. Recomputed per conditioning: own-endpoint match
**100%**, foreign-endpoint transfer **17.6%**. Only the corrected numbers appear in RESULTS.md /
REPORT.md.

**Verdict recorded.** Robust but mostly fragile third output region: common at the level of top-1
labels, usually low-margin, higher-entropy, generic-token and off-manifold; a small minority (~3-4%)
behaves like the crisp MNIST-style third state.
