# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-07-02 — Experiment 1: motivating off-manifold phenomenon (first result)
- Populated RESULTS.md and REPORT.md (both were TODO templates) with the first quantitative
  result for the ColdSteer direction.
- **Setup:** GPT-2 small, resid_post block 6; DiffMean sentiment steering vector (`|v|=11.1`,
  mean `|h|=112.2`); Gaussian manifold fit on 49,218 clean FineWeb tokens; ΔLM on 100 held-out docs.
- **Result (new):** raw steering `z=h+α·v` goes monotonically off-manifold as α grows.
  α=0→8: Mahalanobis `D_M` 27.3→49.0 (real-act ref 27.3); norm ratio 0.98→1.30;
  ΔLM 0.00→+2.78 nats (≈16× perplexity).
- Added figure `plots/01_offmanifold_phenomenon.png` (3 panels: D_M, norm inflation, ΔLM vs α).
- REPORT Methods define all three metrics with rendered `math` fences (verified via GitHub
  markdown API: 5/5 js-display-math, 0 broken).

## 2026-07-02 — Experiment 2: projection-preserving corrector (analytic) — negative result
- Added Experiment 2 to RESULTS.md and REPORT.md: tested the ColdSteer parameterization
  `ĥ = z + P_{v⊥}r` with the analytic optimal Gaussian correction
  `Δ = Σv̂·α|v|/(v̂ᵀΣv̂)` (min whitened-movement shift at matched projection), plus
  norm-clip and naive-inversion baselines.
- **Result (new):** the corrector LOWERS off-manifold distance (`D_M` 49.0→38.1 at α=8) and
  preserves the steering projection exactly (retention 88.6 = raw), but WORSENS LM loss:
  ΔLM +4.20 nats vs raw +2.78 at α=8, and +3.31 vs +0.08 at α=1. Norm-clip gives ~no ΔLM
  gain and inflates `D_M` on clean acts. Decisive finding: statistical on-manifold distance
  and real LM damage are DECOUPLED (D_M down while LM loss up ~40× at low α) — the
  Mahalanobis-minimizing direction `Σv̂` loads onto GPT-2 high-variance outlier dims the LM is
  most sensitive to.
- Implication captured in REPORT Conclusion: corrector must be trained on the DOWNSTREAM LM
  objective, not a manifold-distance surrogate. This reframes/upgrades the motivation.
- New figure `plots/02_corrector.png` (D_M, ΔLM, projection retention vs α, 4 methods).
- New code: `experiments/projections.py` (utilities + `cov_aligned_shift`, unit tests PASS),
  `experiments/02_corrector.py`; results in `results/02_corrector.json`.
- REPORT math re-verified via GitHub API: 8/8 js-display-math, 0 broken, 0 inline hazards.
