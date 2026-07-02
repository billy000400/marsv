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

## 2026-07-02 — Experiment 4: generalization / α-extrapolation of the learned corrector (S4)
- Added Experiment 4 to RESULTS.md and REPORT.md: the Exp-3 learned corrector (trained with
  α~U(0.5,8)) evaluated UNCHANGED at α=10 and α=12 — strictly beyond its training range — on the
  same held-out 100 docs, matched projection.
- **Result (new):** the corrector extrapolates. Fluency recovered vs raw steering: α=8 (boundary)
  84%; α=10 (extrap.) 77% (ΔLM raw +3.31 → learned +0.76); α=12 (extrap.) 60% (ΔLM raw +3.74 →
  learned +1.50). Recovery declines smoothly (84→77→60%) — graceful degradation, not collapse.
  In-range α (1–8) reproduce Exp 3 to the digit (same seed/data), confirming reproducibility.
  D_M learned rises above raw throughout (91.2, 101.2 at α=10,12) — same off-Gaussian-manifold
  signature as Exp 3.
- Interpretation captured: the 4.46M-param MLP learned a transferable correction rule, not a
  lookup over the trained α grid — a sanity check before trusting the method past fitted strengths.
- REPORT Limitation (3) refined: strength-generalization now demonstrated (α up to 12); held-out
  vector / prompt-family / multi-layer still open.
- New code: `experiments/04_generalization.py` (reuses Exp-3 Corrector/training/eval via import).
- New figure `plots/04_generalization.png` (ΔLM and D_M vs α, α>8 shaded as extrapolation region);
  results in `results/04_generalization.json`.
- REPORT math re-verified via GitHub API: 9/9 js-display-math, 0 broken, 0 inline hazards.

## 2026-07-02 — Experiment 5: held-out steering vector / cross-direction generalization (S4b)
- Added Experiment 5 to RESULTS.md and REPORT.md: built a SECOND DiffMean steering vector v₂ for
  an unrelated concept (formality, |v₂|=34.0, cos(v₁,v₂)=0.014 — nearly orthogonal), and compared,
  on v₂ at matched projection α|v₂|, three methods: raw; TRANSFER (Exp-3 corrector trained on
  sentiment v₁, applied unchanged); NATIVE (identical recipe retrained on v₂, the oracle).
- **Result (new):** two findings. (1) The correction is DIRECTION-SPECIFIC — the sentiment-trained
  corrector does NOT transfer to formality (ΔLM transfer ≈ raw at every α; recovery ≈0%, e.g. α=8
  raw +6.49 → transfer +6.53). Confirms proposal Failure Mode 4 (overfits to one vector). (2) The
  RECIPE generalizes — retraining the same 4-layer MLP on v₂ recovers 83–104% of raw's fluency
  damage (α=8 raw +6.49 → native +1.12; α=1 104%, α=2 97%, α=4 92%, α=6 87%, α=8 83%), reproducing
  Exp 3 on a different/larger/near-orthogonal behavior family, again moving FURTHER off the Gaussian
  manifold (D_M 66.6→123.1 at α=8).
- Practical implication captured in REPORT: ColdSteer must be instantiated per steering direction
  (or made direction-conditional / trained on a vector bank), not reused frozen across concepts.
- REPORT Summary + Conclusion + Limitation (3) updated (direction-generalization now shown;
  multi-layer/multi-model/prompt-family and a direction-conditional corrector remain open).
- New code: `experiments/05_heldout_vector.py` (reuses Exp-3 Corrector/training/eval via import;
  builds formality vector, persisted to `data/formality_vec_layer6.npy`).
- New figure `plots/05_heldout_vector.png` (ΔLM and D_M vs α on v₂: raw / transfer / native);
  results in `results/05_heldout_vector.json`.
- REPORT math re-verified via GitHub API: 9/9 js-display-math, 0 broken, 0 inline hazards.

## 2026-07-02 — Experiment 6: direction-conditional corrector on a vector bank (S4c)
- Added Experiment 6 to RESULTS.md and REPORT.md: the direct fix for Exp 5's transfer failure —
  make the corrector CONDITIONAL on the direction (`r_θ(h,z,v̂,α)`, feed the unit vector v̂ as input;
  arch 3d+1, 5.25M params) and train ONE such model on a BANK of 3 DiffMean directions
  {sentiment |v|=11.1, formality 34.0, concreteness 64.5}, sampling (direction, α~U(0.5,8)) per step,
  8 epochs, same frozen-LM objective/seed/data. A 4th direction (certainty, |v|=32.8) is HELD OUT.
  Cosines: sentiment ⟂ all (|cos|≤0.03); formality/concreteness/certainty share a subspace
  (|cos| 0.76–0.82) so the held-out certainty lies largely IN the bank's span.
- **Result (new):** (1) ONE conditional model corrects every in-bank direction at once — recovery at
  α=8: sentiment 55%, formality 70% (ΔLM +6.49→+1.95), concreteness 17% (but 70% at α=2). Cost of
  sharing vs a dedicated single-vector corrector: sentiment 84%→55%, formality 83%→70% at α=8 (capacity
  interference; concreteness weakest at strong steering). (2) Conditioning + bank PARTIALLY transfers
  to the held-out certainty: recovery 51% @α=1 → 7% @α=8 — a real gain over Exp 5's frozen single-vector
  transfer (≈0% at every α), but far below the native oracle retrained on certainty (78% @α=8, 141% @α=1).
  A 3-vector bank does not yet solve held-out transfer at strong steering; scaling the bank is indicated.
- Practical framing captured in REPORT: replaces "one model per vector" (Exp 5) with "one model per
  bank," and the path to a reusable corrector is a LARGER bank, not a frozen operator.
- Updated RESULTS Headline + REPORT Summary/Conclusion/Limitation(3) (direction-conditional/vector-bank
  now shown; larger bank + multi-layer/model/prompt-family still open).
- New code: `experiments/06_conditional_bank.py` (reuses Exp-3 machinery + Exp-5 diffmean via import;
  new CondCorrector + train_cond; builds concreteness/certainty vectors, persisted to
  `data/{concreteness,certainty}_vec_layer6.npy`). New figure `plots/06_conditional_bank.png`
  (per-direction recovery bars @α=8; held-out certainty ΔLM sweep raw/bank/native).
  Results in `results/06_conditional_bank.json`.
- REPORT math re-verified via GitHub API: 9/9 js-display-math, 0 broken, 0 inline hazards.
