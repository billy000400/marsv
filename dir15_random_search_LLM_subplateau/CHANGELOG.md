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

## 2026-07-26 — operator feedback `human_feedback_1`: worked examples + Matthew-style plateau curves

**Why.** Operator feedback (`human_feedback_1.txt`, now `human_feedback_1.addressed.md`) asked:
*"Can you show some examples that sub-plateau shows up? what the sequence? interpolate from where to
where? Show plots in Mathew's plateau lesswrong post style."* Four asks: concrete examples, the top-1
sequence along each path, the two source texts of each interpolation, and figures in the format of
Matthew Shinkle & StefanHex's *Activation Plateaus* post.

**What changed in the deliverables.** Nothing was superseded — every previously reported number is
unchanged. New material added to BOTH RESULTS.md and REPORT.md:

- **New Methods subsection** (REPORT.md, "Is the third region a *plateau*? Matthew-style output
  geometry") defining, with rendered equations: the relative output distance
  `d(α) = ||x(α)-x_A|| / (||x(α)-x_A|| + ||x(α)-x_B||)` on the final logits; the flatness
  `ρ` of the C window (its range of `d` ÷ its width in `α`; ρ = 1 is the no-plateau diagonal);
  the matched non-candidate control; and Matthew's transition width `w(10→90)`.
- **New Results section 4** ("Worked examples: which two texts, which sequence, and is it really a
  sub-plateau?"); previous sections 4–7 renumbered to 5–8. Contains the full worked example
  (contexts A and B in full, endpoint tokens, the complete top-1 run sequence with alpha ranges, the
  C-region continuation, the geometry) plus two more from the flat tail.
- **New quantitative result — the sub-plateau rate.** Median flatness of a candidate C run
  **ρ = 2.05** (IQR 1.15–3.38) vs **1.09** (IQR 0.47–2.99) for width- and position-matched windows on
  non-candidate paths. ρ < 1 for 20.2% of candidates (**3.43% of eligible paths**, CI [3.04, 3.86]);
  **ρ < 0.5 for 8.2% of candidates = 1.39% of eligible paths, CI [1.15%, 1.68%]** — the true
  sub-plateau rate, ~1 path in 72. Mean output distance across the C run `d̄_C`: median 0.518, 97.3%
  inside (0.2, 0.8). Whole-path transition width `w(10→90)`: 0.459 (candidates) vs 0.302
  (non-candidates). Median ρ falls 2.65 → 0.93 from the lowest to the highest decile of the
  *pre-frozen* candidate score (Spearman −0.34, p ≈ 2e−36). Median ρ by block 0/2/4/6:
  2.52 / 2.58 / 2.38 / 1.54; 55 of the 106 sub-plateaus are at block 6; their C runs average 8.1 of 50
  grid points (vs 5.2 for all candidates) and 16.0% are clean `A,C,B` (vs 21.9%).
- **Summary and Conclusion updated** in REPORT.md and the Headline in RESULTS.md to carry the
  plateau-geometry split (common as a *label* event, rare as a *plateau*); a new limitation (v) states
  that the ρ < 0.5 cut is post hoc, descriptive and feeds no prevalence estimate.

**Figures added (embedded in BOTH deliverables).** `matthew_dt_frozen.png` (d(t) for the six
pre-frozen inspection paths, plateau-post format), `matthew_dt_gallery.png` (the six flattest
sub-plateaus), `subplateau_dwell.png` (ρ vs matched control; ρ vs frozen-score decile; `d̄_C`).

**Code/data added.** `experiments/matthew_examples.py` (re-runs the 1,290 candidates + 1,290 matched
controls, 289 s, and records d(t); no new pairs, no rule change), `experiments/plot_matthew.py`,
`results/matthew_examples.json`, `results/matthew_d_curves.npz`, `results/matthew_gallery.json`.

## 2026-07-26 — exploratory depth sweep added (blocks 12/18/24/30)

**Why.** Inside the preregistered window (blocks 0–6) both the third-token rate (8.2% → 27.7%) and the
plateau flatness (median ρ 2.52 → 1.54) improved monotonically with depth, so the obvious question —
does it keep improving? — was still open. `experiments/depth_extension.py` re-ran the **same 1,000
primary pairs with the same frozen detector** at blocks 12, 18, 24 and 30 (block set fixed before
running, 8,000 further paths, 935 s), and recomputed d(t)/ρ for every candidate found.

**What changed in the deliverables.** Nothing superseded; all previously reported numbers stand. Added
as REPORT.md Results section 9 ("Where in the network does this live?") and a table + figure in
RESULTS.md, both labelled exploratory and excluded from the headline (they reuse the primary pairs).

**New numbers.** Third-token rate by block 0/2/4/6/12/18/24/30 = 8.2 / 15.4 / 16.4 / **27.7** / 22.8 /
13.6 / 5.9 / **1.7** %; true sub-plateau rate (ρ < 0.5) = 0.95 / 1.16 / 0.58 / **2.87** / 0.10 / 0.00 /
0.00 / 0.00 %; median ρ = 2.52 / 2.58 / 2.38 / 1.54 / 2.07 / 2.03 / 1.47 / 1.24. Eligible paths per
exploratory block: 1,956 / 1,974 / 1,987 / 1,999. Clean `A,C,B` share of candidates rises with depth
(22 / 28 / 31 / 45 % at blocks 12–30) even as the plateaus vanish.

**Interpretation added.** The depth trend turns over: the phenomenon is early-to-mid network, maximal
near block 6 of 36 and gone by block 18, because a late patch leaves too few blocks to fold the
interpolant into a discrete state and the output curve converges on the no-plateau diagonal.
REPORT.md Summary, Conclusion, Methods (hook point) and Limitation (i) updated accordingly; RESULTS.md
Headline updated.

**Figure added (embedded in BOTH deliverables).** `depth_sweep.png`.

**Code/data added.** `experiments/depth_extension.py`, `experiments/plot_depth.py`,
`results/depth_extension.json`, `results/depth_extension_rho.npz`.
