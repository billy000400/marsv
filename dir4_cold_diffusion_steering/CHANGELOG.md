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

## 2026-07-02 — Experiment 7: scaling the vector bank / does a denser bank close the held-out gap? (S4c follow-up)
- Added Experiment 7 to RESULTS.md and REPORT.md: directly tests Exp 6's closing prescription
  ("scaling the bank is the indicated path"). Held out `certainty` as before; trained the SAME
  direction-conditional corrector (5.25M params, identical recipe/seed/data/8 epochs) on NESTED
  training banks of size 1 [sentiment], 3 [sentiment,formality,concreteness = Exp 6's bank], and 5
  [+ politeness |v|=15.6, + complexity |v|=58.4 — two new DiffMean dirs, 16 pairs each]. Cosines to
  certainty: complexity −0.80 (strong), politeness −0.35 (weak), formality +0.77, concreteness −0.82,
  sentiment +0.03.
- **Result (new, corrective):** enlarging the bank does NOT close the held-out gap — at fixed model
  capacity it makes transfer WORSE. Held-out `certainty` recovery is non-monotone in bank size and
  PEAKS at size 3, not size 5: α=1 14%/51%/−1% (bank 1/3/5), α=8 0%/7%/3%. Even though the size-5
  bank adds a strongly-correlated direction (complexity, |cos|=0.80), transfer dropped at every α.
  Corroborating in-bank signal: under the size-5 model, per-direction recovery @α=8 is LOWER than the
  size-3 model gave (formality 70%→45%, concreteness 17%→13%; new dirs politeness 72%, complexity
  41%, sentiment 57%). ⇒ capacity interference between directions competing for the shared 5.25M MLP,
  not coverage, is the binding constraint. Native oracle retrained on certainty still recovers 78–142%,
  so the direction is fully correctable — the gap is a cost of amortization.
- Size-3 bank reproduces Exp 6 exactly (held-out recovery [51,42,21,12,7]; raw ΔLM
  [0.22,0.99,2.62,3.35,3.71]), confirming reproducibility.
- **Superseded framing:** Exp 6's "path to a reusable corrector is a LARGER bank" → corrected to
  "more MODEL CAPACITY and/or a bank CURATED toward the target subspace, not simply more directions."
  Updated RESULTS Headline + Exp-6 closing sentence; REPORT Summary + Conclusion (Exp 6 paragraph) +
  Limitation (3).
- New code: `experiments/07_bank_scaling.py` (reuses Exp 6 CondCorrector/train_cond/make_hat_cond +
  Exp 3 LM-loss/Gaussian/Mahalanobis via import; builds politeness/complexity vectors, persisted to
  `data/{politeness,complexity}_vec_layer6.npy`). New figure `plots/07_bank_scaling.png` (held-out
  recovery vs α per bank size + oracle; held-out recovery @α=1,8 vs bank size). Results in
  `results/07_bank_scaling.json`.
- REPORT math re-verified via GitHub API: 10/10 js-display-math (added the recovery-fraction eq),
  0 broken, 0 inline hazards.

## 2026-07-02 — Experiment 8: scaling MODEL CAPACITY on a fixed bank (S4c follow-up #2)
- Added Experiment 8 to RESULTS.md and REPORT.md: directly tests Exp 7's causal claim ("capacity
  interference between directions competing for a fixed 5.25M MLP binds"), which Exp 7 never varied.
  Held the BANK fixed at Exp 7's size-5 set {sentiment, formality, concreteness, politeness, complexity}
  (its WORST-transfer bank) and scaled corrector WIDTH hidden∈{1024,2048,4096} = 5.2M/14.7M/46.2M params
  (9× range), identical recipe/seed/data/8 epochs. Native oracle (retrained on certainty, 5.25M) = ceiling.
- **Result (new, corrective):** more capacity does NOT close the held-out gap either — simple width
  scaling is not the fix. (1) Mean in-bank recovery @α=8 SATURATES at ~45% across the 9× range
  (45.4%→43.8%→46.3%) — the MLP was not width-starved. (2) Held-out `certainty` transfer @α=8 is
  flat-to-falling (3%→2%→1%) and at weak steering the widest model OVERFITS, actively harming the unseen
  direction: α=1 recovery −1%→−22%→−146% (46.2M model adds +0.32 nats to a near-harmless weak steer).
  ⇒ the amortization ceiling is set by the TRAINING SIGNAL (bank composition / conditioning / objective),
  NOT by parameter count. Native oracle unchanged (78–142%); correction is fundamentally direction-specific.
- hidden=1024 point reproduces Exp 7's size-5 model to the digit (held-out rec [-1,9,6,4,3]; in-bank@8
  {sent57,form45,conc13,pol72,cplx41}) — built-in reproducibility check passed.
- **Superseded framing:** Exp 7's "closing the gap needs more MODEL CAPACITY and/or a curated bank" →
  corrected to "needs bank CURATION toward the target subspace and/or a stronger training signal — NOT
  simply a bigger model (Exp 8) or more directions (Exp 7)." Updated RESULTS Headline + Exp-7 closing
  sentence; REPORT Summary + Exp-7 interpretation + Conclusion + Limitation (3).
- New code: `experiments/08_capacity_scaling.py` (reuses Exp 6 CondCorrector[hidden]/train_cond/
  make_hat_cond + Exp 3 LM-loss/layer via import; loads all 5 persisted pool vectors + certainty).
  New figure `plots/08_capacity_scaling.png` (held-out recovery vs α per capacity + oracle; recovery @α=8
  vs capacity for held-out + mean in-bank). Results in `results/08_capacity_scaling.json`.
- REPORT math re-verified via GitHub API: 10/10 js-display-math, 0 broken (<pre lang=math>), 0 inline hazards.

## 2026-07-02 — Experiment 9: curating the bank TOWARD the target subspace (S4c follow-up #3)
- Added Experiment 9 to RESULTS.md and REPORT.md: directly tests the open path that BOTH Exp 7 and Exp 8
  named ("curate the bank toward the held-out target's subspace"), which neither had varied. Held bank
  SIZE fixed at 3 and corrector CAPACITY fixed at 5.25M (hidden=1024); varied only WHICH 3 of the 5 pool
  directions are trained, by mean |cos| to held-out `certainty`: diffuse {sentiment,politeness,formality}
  0.38 / exp6 {sentiment,formality,concreteness} 0.54 / curated {formality,concreteness,complexity} 0.80.
  diffuse & curated share exactly one member (formality) — controlled contrast. Identical recipe/seed/data.
- **Result (new, corrective — third negative in a row):** curating TOWARD the target subspace does NOT
  close the gap; it makes transfer CATASTROPHICALLY worse. Held-out `certainty` recovery is non-monotone
  in bank→target alignment and COLLAPSES at the most-aligned bank: curated is net-negative at every
  strength (α=1 rec −183%: adds +0.40 nats to a +0.22-nat steer; α=8 −12%), while the moderately-aligned,
  angularly DIVERSE exp6 bank transfers BEST (51/42/21/12/7). Mechanism from in-bank recovery @α=8: it
  FALLS as the bank's own directions grow internally correlated — diffuse 67% (sent65,pol74,form60) >
  exp6 48% (sent55,form70,conc17) > curated 30% (form37,conc17,cplx35). The curated members are pairwise
  near-collinear (|cos| 0.76–0.82), so the conditional corrector can't disambiguate them from v̂ and can't
  specialize. ⇒ the lever is bank ANGULAR DIVERSITY (separability), NOT coverage of the target subspace;
  curating toward the target is exactly the wrong move.
- exp6 bank reproduces Exp 6/7's size-3 model to the digit (rec 51/42/21/12/7) — reproducibility check.
  Native oracle unchanged (78–142%); correction remains fully available per-direction.
- **Superseded framing:** Exp 7/8's parting "curate the bank toward the target subspace is the open path"
  → corrected to "curating TOWARD the target backfires; bank DIVERSITY (not target alignment) governs
  transfer." Updated RESULTS Headline + Exp-8 closing; REPORT Summary + Conclusion + Limitation (3).
- New code: `experiments/09_curated_bank.py` (reuses Exp 6 CondCorrector/train_cond/make_hat_cond + Exp 3
  LM-loss/layer via import; loads all 5 pool vectors + certainty). New figure `plots/09_curated_bank.png`
  (held-out recovery vs α per bank + oracle; held-out recovery @α=1,8 vs bank |cos| alignment). Results in
  `results/09_curated_bank.json`.
- REPORT math re-verified via GitHub API: 10/10 js-display-math, 0 broken (<pre lang=math>), 0 inline hazards.
