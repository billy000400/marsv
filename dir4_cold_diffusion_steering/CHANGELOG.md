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

## 2026-07-02 — Experiment 3: learned LM-supervised corrector — POSITIVE result (direction's payoff)
- Added Experiment 3 to RESULTS.md and REPORT.md: the ColdSteer parameterization
  `ĥ = z + P_{v⊥}r_θ` with `r_θ` a 4-layer MLP (4.46M params) trained END-TO-END against the
  frozen model's downstream next-token cross-entropy (h detached, LM weights frozen), α~U(0.5,8)
  sampled per step, light minimal-correction penalty λ_near=0.05. 6 epochs / ~230 steps on 300
  FineWeb docs; evaluated on the SAME held-out 100 docs as Exp 1/2. Matched projection (retention
  = α|v|, identical to raw and cov_corr).
- **Result (new):** the learned corrector BEATS raw steering at every α at matched projection.
  ΔLM at α=8: raw +2.78 → learned **+0.44 nats (84% reduction)**; α=1: raw +0.08 → learned −0.07;
  α=2 −0.05; α=4 +0.06; α=6 +0.22. It achieves this while moving FURTHER off the Gaussian manifold
  than raw (`D_M` 49.0→79.5 at α=8) — the mirror image of Exp 2's cov_corr (which moved toward the
  manifold and broke the LM). Confirms the decoupling constructively: the LM-safe correction is
  off-Gaussian-manifold and only a downstream-LM objective finds it.
- Reframed REPORT Summary/Conclusion from a two-step (phenomenon + negative) story to the full
  three-step thesis (phenomenon → surrogate fails → downstream-supervised corrector works). Exp 2's
  "corrector fails" framing narrowed to "manifold-surrogate fails"; Exp 3 supplies the working method.
- New code: `experiments/03_learned_corrector.py`; results in `results/03_learned_corrector.json`.
- New figure `plots/03_learned_corrector.png` (ΔLM, D_M, projection retention vs α; raw / analytic
  cov-aligned / learned).
- REPORT math re-verified via GitHub API: 9/9 js-display-math, 0 broken `<pre lang=math>`, 0 inline hazards.
