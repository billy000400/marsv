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

## 2026-07-02 (Experiment 10 — behavioral reality-check: matched projection ≠ matched steering in generation)
- **New experiment (corrective/caveat to the flagship fluency story).** All prior experiments score the
  corrector on teacher-forced `ΔLM` at *matched projection at one layer* — a proxy that never checked
  whether the corrected steer, used to GENERATE, still steers the text. Exp 10 tests it directly.
- **Setup:** flagship sentiment corrector (Exp 3, retrained identically). Greedy-generate 30 tokens from
  48 held-out 12-token prompts with the steer applied at resid_post block 6 at every position, raw vs
  corrected. On a CLEAN re-encode of the output: sentiment effect `B(α)−B(0)` (proj onto v̂; baseline
  B0=+0.34) and degeneration distinct-2 (unique-bigram ratio; baseline 0.70).
- **Result:** the corrector's fluency win is NOT free — it trades away the behavioral steer, which the
  matched-projection metric hid. Raw steers hard (effect +2.97 @α=2) then collapses into repetition
  (distinct-2 0.78→0.32 @α=8; "the second-t-t-t-t-t-t"). The corrector STAYS fluent at all α (distinct-2
  0.64–0.72, near baseline 0.70) but is only weakly steered (effect +0.15–0.48, ~1/6 of raw's). Neither
  method dominates the effect-vs-fluency Pareto. Mechanism: P_{v⊥}r is orthogonal to v in ACTIVATION
  space but NOT to the downstream sentiment READOUT, so minimizing LM loss yields near-normal,
  lightly-steered text.
- **Superseded framing:** Exp 3/Summary/Conclusion "recovers 84% ... with the steering edit fully intact"
  → qualified to "with the layer-6 steering PROJECTION intact"; the behavioral edit in generation is
  weakened. Limitation (2) corrected: it previously claimed "concept strength is held fixed by
  construction" (FALSE for generation) → now states matched layer-6 projection ≠ matched behavioral
  steering, and behavioral effect must be measured directly. RESULTS Headline + REPORT Summary/Conclusion
  gain a behavioral-caveat paragraph.
- New code: `experiments/10_behavioral_pareto.py` (imports Exp 3 Corrector/train_corrector/make_hat/
  FuncPatcher via importlib). New figure `plots/10_behavioral_pareto.png` (effect vs α; distinct-2 vs α;
  effect-vs-fluency Pareto). Results in `results/10_behavioral_pareto.json` (incl. sample generations).
- REPORT math re-verified via GitHub API: 12/12 js-display-math (added B(α) + distinct-2 fences),
  0 broken (<pre lang=math>), 0 inline hazards.

## 2026-07-02 — Experiment 11: behavioral-preservation term pushes the Exp 10 Pareto outward
- **Motivation:** Exp 10 found the flagship sentiment corrector under-steers in generation because its
  layer-6 correction, though ⟂ v, is NOT ⟂ the downstream concept readout. Exp 11 acts on PLAN Next-step
  (i): add a term that preserves the downstream readout and test whether the effect-fluency Pareto moves out.
- **Setup:** identical Exp 3 corrector/recipe/seed/data + ONE extra loss term. During teacher-forced
  training also read out the sentiment projection at downstream layer L2=11 (final resid_post; DiffMean
  ŵ, |w|=3.87) and push corrected p_corr toward RAW steering's p_raw via λ_b·⟨((p_corr−p_raw)/100)²⟩.
  Train family λ_b∈{0,10,40} (λ_b=0 = Exp 10 corrector) and score each on the IDENTICAL Exp 10 generation
  protocol (48 prompts, 30 greedy tokens; effect B(α)−B(0) and distinct-2 on clean re-encode).
- **Result (new):** the behavioral term recovers 2–6× more behavioral effect while staying fluent, and
  pushes the frontier OUTWARD at the fluent end. Generated effect rises from Exp 10's +0.15–0.48 (λ_b=0)
  to +0.8–1.3 (λ_b=10/40); distinct-2 stays 0.52–0.73 (raw collapses to 0.32 @α=8). NEW finding vs Exp 10
  ("neither dominates"): the corrector now Pareto-DOMINATES raw at moderate steering — λ_b=40 @α=2 reaches
  effect +0.99 at distinct-2 0.73 (≈baseline), whereas raw only reaches effect that low (+1.77 @α=8) after
  collapsing to 0.32. CEILING: no λ_b lifts effect past ≈+1.3 (λ_b 10→40 stops raising it; even falls @α=6
  +0.93→+0.84) — matching the teacher-forced downstream readout (training behav loss →~0.005, p_corr≈p_raw)
  only PARTIALLY transfers to autoregressive generation. λ_b=0 reproduces Exp 10 to the digit (reproducibility).
- **Deliverable deltas:** RESULTS.md +Exp 11 (table + reading) + figure entry + Headline "Partial fix"
  paragraph. REPORT.md +Exp 11 Methods (behavioral term equation + downstream readout) + Results section +
  Summary/Conclusion updates; Limitation (2) updated (the "explicit behavioral objective is the natural next
  step" is now DONE — pushes frontier out but does not erase it). No prior result superseded (Exp 11 is new;
  Exp 10 numbers unchanged, reproduced by λ_b=0).
- New code `experiments/11_behavioral_corrector.py` (imports exp01 POS/NEG, exp03 Corrector/make_hat/
  FuncPatcher, exp10 generate/effect/distinct2). New figure `plots/11_behavioral_corrector.png` (effect vs
  α; distinct-2 vs α; effect-vs-fluency Pareto with λ_b family). Results `results/11_behavioral_corrector.json`.
- REPORT math re-verified via GitHub API: 13/13 js-display-math (added the behavioral-loss fence), 0 broken
  (<pre lang=math>), 0 inline hazards.

## 2026-07-02 — Experiment 12: layer robustness (blocks 3, 6, 9) — the fluency result is not a block-6 artifact
- **Motivation:** every prior experiment hooks resid_post block 6. A reviewer's first question is whether
  the two headline facts — raw steering breaks the LM, the LM-supervised corrector recovers it — are
  specific to that layer. Acts on PLAN Next-step (iii, layer generality).
- **Setup:** replicated the EXACT Exp 3 pipeline at blocks 3 (early), 6 (mid = Exp 3), 9 (late); only the
  hook layer changes. Per layer: rebuild DiffMean sentiment v (|v| = 6.75 / 11.08 / 23.16), fit clean
  Gaussian on 400 docs, train the identical 4-layer corrector on the same 300 docs vs downstream LM loss
  (same seed/hyperparams/α∼U(0.5,8)), eval ΔLM / D_M / retention on the same held-out 100 docs at matched
  projection α|v|.
- **Result (new):** both facts replicate at every depth. Fluency recovery @α=8 = 90% / 84% / 76% (blocks
  3/6/9), ≥91% @α=4, ΔLM near zero at weak steering. The corrected activation sits FURTHER off the Gaussian
  manifold than raw at every layer (D_M corrected > raw), so "LM-safe but off-Gaussian" is layer-robust.
  Recovery declines mildly with depth (fixed-capacity corrector faces larger |v| toward the output). Block 6
  reproduces Exp 3 TO THE DIGIT (raw +2.78 → learned +0.44, 84%) — built-in reproducibility check of the
  refactored layer-swept pipeline. No prior result superseded (Exp 12 is new; layer-6 row = Exp 3 reproduced).
- **Deliverable deltas:** RESULTS.md +Exp 12 (table + reading) + figure entry + Headline layer-robustness
  sentence. REPORT.md +Exp 12 Methods (recovery-fraction equation) + Results section (table + interpretation)
  + Summary and Conclusion layer-robustness sentences.
- New code `experiments/12_layer_robustness.py` (reuses exp03 Corrector/train_corrector/eval helpers by
  swapping module-global LAYER; POS/NEG from exp01). New figure `plots/12_layer_robustness.png` (ΔLM raw vs
  corrected per layer; recovery vs α per layer; D_M raw vs corrected per layer). Results
  `results/12_layer_robustness.json`.
- REPORT math re-verified via GitHub API: 14/14 js-display-math (added the recovery fence), 0 broken
  (<pre lang=math>), 0 inline hazards.

## 2026-07-02 — Experiment 13: cross-model generality (GPT-2 medium) — POSITIVE
- Added Experiment 13 to RESULTS.md and REPORT.md: replicated the flagship Exp-3 pipeline
  UNCHANGED on GPT-2 **medium** (355M, 24 blocks, d=1024), steering/correcting at its mid layer
  (block 12 of 24 — depth analogue of block 6 of 12 in small). Same DiffMean sentiment prompts,
  400-doc Gaussian fit, 300-doc train / held-out 100-doc eval, 4-layer corrector (now d=1024,
  5.25M params), seed, α∼U(0.5,8), hyperparams; only the MODEL changes (|v|=19.6, mean|h|=226.2,
  clean D_M=31.5). Reuse: loaded medium once and installed it in common's model cache so the
  imported Exp-3 helpers run on it; corrector trained at batch 4 for the VRAM budget.
- **Result (new):** both headline facts replicate on the larger model — NOT a GPT-2-small artifact.
  (P) raw steering breaks the LM: ΔLM +0.04/+0.15/+0.74/+2.72 at α=1/2/4/8, D_M 31.5→55.1.
  (C) the identical LM-supervised corrector recovers it at matched projection: ΔLM
  −0.12/−0.09/−0.01/+0.30, **recovery 89% @α=8, 101% @α=4** (>100% at α≤2 = learned ΔLM slightly
  below the unsteered baseline, as on small; raw denominator near zero). Signature decoupling holds:
  corrected D_M > raw at every α (79.9 vs 55.1 @α=8) — off-Gaussian-but-LM-safe on medium too.
  α=8 recovery (89%) ≈ small's 84%.
- Added figure `plots/13_cross_model.png` (ΔLM raw vs corrected; recovery vs α; D_M raw vs corrected).
- RESULTS.md: +Exp 13 block + figure entry + Headline "model-robust" sentence.
  REPORT.md: +Exp 13 Methods (cross-model setup; reuses Exp 12 recovery equation) + Results
  (table + interpretation) + Summary/Conclusion "model-robust" sentences + Limitation (3) updated
  (multi-model generalization now DONE; only held-out-prompt-family and still-larger models open).
- Code: `experiments/13_cross_model.py`; results `results/13_cross_model.json`.
- REPORT math re-verified via GitHub API: 14/14 js-display-math (no new equations; Exp 13 reuses the
  recovery fence), 0 broken (<pre lang=math>), 0 inline hazards.

## 2026-07-02 — Experiment 14: direct/causal confirmation of the bank-DIVERSITY lever (S4c follow-up #4)
- Added Experiment 14 to RESULTS.md and REPORT.md: turns Exp 9's *correlational* claim ("bank angular
  diversity, not target-subspace coverage, drives a conditional corrector's recovery") into a *controlled*
  one. Exp 9 could not isolate diversity from target-alignment because the held-out `certainty` lives
  inside the collinear cluster, so alignment and internal collinearity co-varied. Exp 14 removes the
  confound with a CONTROLLED THIRD-MEMBER SWAP: three size-3 banks, capacity fixed 5.25M, all sharing the
  anchor pair {sentiment, formality}; only the THIRD member varies in collinearity with formality —
  div=+politeness (|cos| 0.07, internal D=0.13), mid=+complexity (0.57, D=0.21), coll=+concreteness
  (0.76, D=0.26). Decisive control: `sentiment` is ⟂ every direction AND ⟂ the held-out target
  (|cos|≤0.03), so its recovery can only depend on the bank's internal separability.
- **Result (new, POSITIVE — the positive counterpart to Exp 7/8/9's three negatives):** bank angular
  diversity is a CAUSAL lever. Two monotone signals @α=8: (1) the swapped 3rd member's OWN recovery
  collapses as it collinearizes with formality — politeness 69% → complexity 40% → concreteness 17%
  (α=4: 75/57/34) — a member confusable with a neighbor can't be specialized (corrector gets v̂, can't
  separate near-parallel dirs); (2) the confound-free isolate `sentiment` is corrected WORSE in more
  collinear banks — 63% → 61% → 55% — which cannot be a target-coverage effect (sentiment ⟂ target).
  `formality` (the anchor that gains the collinear neighbor) holds ~69–70%: the corrector collapses the
  near-parallel pair onto the dominant larger-norm member, so the neighbor loses recovery, the anchor
  keeps it. Held-out certainty transfer flat (9/5/7%) as designed (this varies internal separability, not
  target coverage). No prior result superseded (Exp 14 is new; corroborates & causally upgrades Exp 9).
- **Deliverable deltas:** RESULTS.md +Exp 14 (table + reading) + figure entry + Headline diversity-lever
  sentence now cites the causal confirmation. REPORT.md +Exp 14 Methods (controlled-swap design +
  confound-free-isolate rationale) + Results (table + interpretation) + Conclusion (Exp 9 sentence now
  "confirmed causally by Exp 14") + Limitation (3) updated.
- New code `experiments/14_diversity_lever.py` (reuses Exp 6 CondCorrector/train_cond/make_hat_cond +
  Exp 3 LM-loss/LAYER via import; loads all 6 persisted pool vectors). New figure
  `plots/14_diversity_lever.png` (anchor-pair recovery @α=8 vs internal collinearity; sentiment recovery
  vs α per bank). Results `results/14_diversity_lever.json`.
- REPORT math re-verified via GitHub API: 14/14 js-display-math (Exp 14 reuses the recovery fence, no new
  equation), 0 broken (<pre lang=math>), 0 inline hazards.

## 2026-07-02 — Experiment 15: held-out prompt-family generalization (S4 follow-up) — POSITIVE
- Added Experiment 15 to RESULTS.md and REPORT.md: every prior experiment both TRAINS and EVALUATES on
  FineWeb web text, so the corrector could be overfit to that prompt distribution. Tested directly.
  Trained the flagship sentiment corrector EXACTLY as Exp 3 (same vector/seed/recipe/300 FineWeb train docs),
  then evaluated its fluency recovery UNCHANGED, at matched projection α|v|, on three held-out prompt families
  of increasing distribution shift from FineWeb: fineweb (in-dist, = Exp 3 held-out 100 docs), markdown (100
  chunks of this project's own .md research prose), code (100 chunks of numpy/torch/transformers Python source).
  Quantified each family's shift by the mean Mahalanobis distance of its CLEAN activations under the FineWeb
  Gaussian: D_M 27.5 (fineweb) / 30.1 (markdown) / 37.4 (code).
- **Result (new, POSITIVE):** the corrector is NOT overfit to the FineWeb prompt distribution — it transfers
  to genuinely different families and degrades gracefully with distribution shift. Recovery @α=8: fineweb 84%,
  markdown 77%, code 60% (@α=4: 95/87/78%). Recovery tracks the activation shift monotonically (D_M 27.5→30.1
  →37.4 ⇒ 84→77→60%), i.e. smooth graceful degradation (as for strength extrapolation in Exp 4), not collapse.
  The in-distribution fineweb row reproduces Exp 3 TO THE DIGIT (raw +2.78 → learned +0.44, 84%) — built-in
  reproducibility check. No prior result superseded (Exp 15 is new; fineweb row = Exp 3 reproduced).
- **Deliverable deltas:** RESULTS.md +Exp 15 (two tables + reading) + figure entry + Headline
  "prompt-family-robust" sentence. REPORT.md +Exp 15 Methods (held-out-prompt-family setup; reuses Exp 12
  recovery equation and the Exp-1 D_M definition — no new display equation) + Results (two tables +
  interpretation) + Summary/Conclusion "prompt-family-robust" sentences + Limitation (3) updated
  (held-out-prompt-family generalization now DONE; only still-larger models open).
- New code `experiments/15_prompt_family.py` (imports exp03 Corrector/train_corrector/make_hat/lm_loss_fn/
  gaussian_stats/mahalanobis/LAYER; builds markdown+code corpora from local files). New figure
  `plots/15_prompt_family.png` (ΔLM raw vs corrected per family; recovery vs α per family; clean-activation
  shift bar). Results `results/15_prompt_family.json`.
- REPORT math re-verified via GitHub API: 14/14 js-display-math (Exp 15 reuses the recovery fence, no new
  equation), 0 broken (<pre lang=math>), 0 inline hazards.

## 2026-07-06 — Experiment 16: is the "manifold" Gaussian? Intrinsic dimension + Gaussianity — NEW (acts on human feedback)
- Acts on human feedback (2026-07-06): "I'm not sure if Gaussian manifold is valid here. Look up recent
  literature about recovering smooth manifold from discrete points and evaluate what kind of manifold we
  are dealing with. I doubt it is not Gaussian." Tested directly, no steering, on the clean layer-6 FineWeb
  activations used throughout (49,218 tokens).
- **Method (new):** two standard manifold-from-discrete-points intrinsic-dimension estimators — TwoNN
  (Facco et al. 2017) and the Levina–Bickel MLE (2004) — plus PCA participation ratio; and a Gaussianity
  test comparing held-out D_M^2 to its chi^2_768 law (moments + Wilson–Hilferty QQ) with per-dimension
  excess kurtosis. All implemented in numpy/torch (no scipy/sklearn available). Deterministic (seed 0).
- **Result (new):** the activation cloud is NOT a single 768-d Gaussian. (1) LOW-DIMENSIONAL: intrinsic
  dim ~8–34 (TwoNN 11.4 raw / 8.1 z-scored; MLE 25–34) vs 768 ambient. (2) EXTREMELY ANISOTROPIC: PCA
  participation ratio 1.1, ~90% of variance in ONE PC and 95% in three (GPT-2 outlier/rogue dims).
  (3) HEAVY-TAILED: held-out D_M^2 spread 6.7× the Gaussian chi^2_768 (variance ~45× too big), skew 0.45
  vs 0.10, 14 dims with excess kurtosis >1 (max 118). Mean of D_M^2 matches 768 but is non-diagnostic.
- **Framing (no prior result superseded):** this SHARPENS the thesis. It is the concrete mechanism behind
  Exp 2's negative result (the Gaussian piles its "volume" into high-variance rogue dims, so the
  D_M-minimizing correction Σv̂ moves there — cheap in D_M, destructive to the LM), and it reframes "off
  the Gaussian manifold" (Exp 3/5/12/13) as "off a crude fit," confirming D_M is a diagnostic, never a
  training target. All existing LM-loss-based numbers unchanged.
- **Deliverable deltas:** RESULTS.md +Exp 16 (two tables + reading) + figure entry + Headline
  "On the 'manifold' itself" paragraph. REPORT.md +Methods "Manifold geometry" subsection (TwoNN /
  Levina–Bickel / participation-ratio / chi^2 equations, 4 new display-math blocks) + Results Exp 16
  (two tables + interpretation + figure) + Summary Step-2 clause + Conclusion clause + Limitation (1)
  upgraded (now quantified; future work notes a mixture/flow/diffusion prior).
- New code `experiments/16_manifold_geometry.py`. New figure `plots/16_manifold_geometry.png` (QQ vs
  chi^2 / PCA cumulative variance / intrinsic-dim bars). Results `results/16_manifold_geometry.json`.
- REPORT math re-verified via GitHub API: 18/18 js-display-math (14 prior + 4 new), 0 broken
  (<pre lang=math>), 0 inline hazards.

## 2026-07-06 — Experiment 17: a REAL diffusion corrector (Cold-Diffusion) vs one-shot MLP vs GLP Gaussian prior — NEW (acts on human feedback #1)
- Acts on human feedback (2026-07-06): "build an actual diffusion-model corrector like the GLP arxiv paper,
  not the one-shot MLP; name it explicitly diffusion; compare Pareto to the one-shot MLP and to a generic
  Gaussian-noise GLP-style teacher." This was the central critique of the direction (named after Cold
  Diffusion but the flagship corrector is a one-shot MLP).
- **Method (new):** three correctors compared at MATCHED steering projection α|v| on the SAME held-out
  FineWeb eval (GPT-2 small, block 6, sentiment vector), all reusing the Exp-3 pipeline. (1) one-shot MLP
  (Exp 3 incumbent, 4.46M). (2) COLD-DIFFUSION iterative (NEW): same-capacity weight-shared step-conditioned
  velocity field g_θ(h,x,α,t), integrated over K=8 projection-preserving steps (each increment ⟂v so the
  steer is preserved at every step), trained by UNROLLING the K steps and backpropping the frozen upper-LM
  next-token CE (iterative analogue of Exp 3; 4.46M). (3) GLP Gaussian prior (NEW baseline): a real DDPM
  (cosine schedule, ε-prediction, 2.69M) trained on CLEAN standardized activations with GAUSSIAN-noise
  corruption, pure MSE, NO LM in the loop; corrects a steered z by SDEdit (noise to t_start=0.15 chosen by
  steelmanning, DDIM-denoise back), projection re-imposed for the matched-ΔLM comparison.
- **Result (new):** recovery of raw steering's fluency damage @α=8 — one-shot MLP **84%** (ΔLM +0.435),
  cold-diffusion iterative **85%** (ΔLM +0.419), GLP Gaussian prior **−5%** (ΔLM +2.925, WORSE than raw
  +2.778). Three answers: (RQ1) the Cold-Diffusion CORRUPTION MODEL is what matters — LM-supervised training
  on the actual steering corruption recovers 84–85% while the generic Gaussian-noise "denoise back to the
  manifold" prior has NEGATIVE recovery at every α; (RQ2) the iterative diffusion structure ~TIES the
  one-shot MLP (85 vs 84% @α=8, a small consistent edge; iter D_M 75.2 vs one-shot 79.5 — slightly closer to
  Gaussian) so iteration is not the source of the benefit; (RQ3) the unconditional GLP prior ERASES the
  steer (as-is projection retention 10.6/83.1 vs target 11.1/88.6 @α=1/8, ~5–6% lost). No prior result
  superseded — this is an added comparison; the one-shot MLP's 84% @α=8 is unchanged.
- **Deliverable deltas:** RESULTS.md +Exp 17 (7-column comparison table + reading) + figure entry +
  Headline "A real diffusion corrector" paragraph. REPORT.md +Methods "A real diffusion corrector (Exp 17)"
  subsection (3 new display-math: one-shot form, iterative update, DDPM forward+MSE) + Results Exp 17 (table
  + interpretation + figure) + Summary diffusion paragraph + Conclusion clause + Limitation (1) updated (the
  "better manifold model = diffusion prior" future-work item is now TESTED and does not help).
- New code `experiments/17_diffusion_corrector.py`. New figure `plots/17_diffusion_corrector.png`
  (ΔLM vs α / recovery vs α / projection retention). Results `results/17_diffusion_corrector.json`.
- REPORT math re-verified via GitHub API: 21/21 js-display-math (18 prior + 3 new), 0 broken
  (<pre lang=math>), 0 inline hazards.

## 2026-07-06 — Experiment 18: beyond hand-built DiffMean — steering-vector-family robustness (acts on human feedback #3)
- **Why:** the last open human-feedback ask (#3). Every steering vector in Exp 1–17 (6 of them) is a
  DiffMean direction built from ~20 HAND-WRITTEN contrastive sentences, so the flagship result could be an
  artifact of that one extraction method and/or the hand-built prompts. Tests whether the recipe generalizes
  to a genuinely different steering FAMILY and DATA SOURCE.
- **Method (new):** built the sentiment steering vector from a REAL downloaded dataset (SST-2, 500 pos + 500
  neg movie-review sentences, mean-pooled block-6 activations) via the three canonical linear-steering
  families — (1) DiffMean μ⁺−μ⁻; (2) logistic-regression probe (L2, weight mapped from standardized to raw
  activation coords, discriminative); (3) PCA-contrast (top PC of centered pos−neg pair differences, RepE,
  unsupervised). Sign-aligned to +sentiment and RESCALED to a common norm |v|=11.0 (flagship scale) so the
  ONLY variable across families is the direction. Ran the identical flagship recipe (Exp 3 corrector, per
  direction) on each at matched projection α|v|.
- **Result (new):** the three directions are genuinely different (cos to DiffMean 1.00 / 0.40 / 0.30), and the
  SST-2 DiffMean agrees with the original hand-built DiffMean only at cos 0.49. All three break the LM under
  raw steering (ΔLM @α=8 = +3.41 / +2.63 / +2.27) and the identical LM-supervised corrector recovers each:
  **recovery @α=8 = 86% / 84% / 101%** (98/95/118% @α=4), matched projection. The DiffMean family reproduces
  the flagship Exp 3 (raw +3.41→+0.47, 86% ≈ 84%) from real data. PCA-contrast is especially telling: its raw
  steering leaves D_M FLAT at the clean value 27.3 (ON the Gaussian manifold) yet still breaks the LM (+2.27),
  and the corrector fixes it by moving OFF the manifold (27.3→47.5) — off-Gaussian distance is neither
  necessary nor sufficient for LM damage. No prior result superseded; this is an added external-validity axis.
  The core result now holds on SIX axes (strength/direction/layer/model/prompt-family/steering-family).
- **Deliverable deltas:** RESULTS.md +Exp 18 (family table + reading) + figure entry + Headline
  steering-family clause (now "six axes"). REPORT.md +Methods "Steering-vector families (Experiment 18)"
  subsection (3 new display-math: DiffMean, logistic-probe objective, PCA-contrast) + Results Exp 18 (table +
  interpretation + figure) + Summary six-axes clause + Conclusion clause.
- New code `experiments/18_steering_family.py` (run with dir9's cupenv python — the shared conda `transformers`
  had disappeared this iteration, cupenv has torch+CUDA+transformers). New figure `plots/18_steering_family.png`
  (ΔLM / recovery / D_M vs α per family). Results `results/18_steering_family.json`, `data/sst2_train.tsv`.
- REPORT math re-verified via GitHub API: 24/24 js-display-math (21 prior + 3 new), 0 broken
  (<pre lang=math>), 0 inline hazards.

## 2026-07-07 — Experiment 19: model-scaling to GPT-2 large (774M) — the third model-scale point
- **Why:** Exp 13 showed the flagship result survives one step up in model size (GPT-2 medium, 355M). The
  natural next external-validity point is a still-larger model, to check whether amortized correction
  quality erodes with scale. Adds a third scale so the model axis spans a 6× parameter range
  (124M → 355M → 774M) — the one untested model-scale point flagged as optional in PLAN.md.
- **Method (new):** replicated the EXACT flagship Exp-3 pipeline UNCHANGED on GPT-2 large (774M, 36 blocks,
  d=1280) at the mid layer block 18/36 (depth analogue of block 6/12 small, block 12/24 medium). Only the
  model changes; same DiffMean sentiment prompts, 400-doc Gaussian fit, 300-doc training set, held-out
  100-doc eval, 4-layer projection-preserving corrector (now 6.03M params at d=1280), seed, α∼U(0.5,8),
  hyper-parameters. Reused Exp-3 helpers via the shared model cache (as Exp 13). Batch 2 for training to fit
  the 774M model in the ~4.3 GB per-agent VRAM share (no OOM). Ran with dir9's cupenv python (shared conda
  transformers still absent).
- **Result (new):** both headline facts replicate. Raw steering breaks the LM (ΔLM@α=8 = +2.47 nats, D_M
  35.2→66.0) and the identical LM-supervised corrector recovers it at matched projection: **recovery @α=8 =
  84%** (ΔLM +2.47→+0.39), **95% @α=4** (ΔLM +0.73→+0.03), free-or-better at weak α (ΔLM −0.05 to −0.07).
  Corrected activation sits FURTHER off the Gaussian manifold than raw at every α (D_M learned 96.8 > raw
  66.0 @α=8) — the Exp-2/3 decoupling holds a third time. Retention matched α|v|=16.8→134.0 exactly.
  Model-scaling trend @α=8 recovery is FLAT across the 6× range: small 84% / medium 89% / large 84% — quality
  does not erode as the model grows. No prior result superseded; this is an added external-validity axis.
  Core result now model-robust across GPT-2 small/medium/large.
- **Deliverable deltas:** RESULTS.md +Exp 19 (table + reading) + figure entry + Headline model-robust clause
  extended to three scales (124M→355M→774M, 84/89/84%). REPORT.md +Methods "Model scaling to GPT-2 large
  (Experiment 19)" subsection + Results Exp 19 (table + interpretation + figure) + Summary model-robust
  clause + Conclusion clause (two spots) extended to three scales.
- New code experiments/19_gpt2_large.py, figure plots/19_gpt2_large.png, results results/19_gpt2_large.json,
  log results/19_run.log. Downloaded gpt2-large weights into the shared HF cache (3.1 GB).
- REPORT math re-verified via GitHub API: 24/24 js-display-math (no new display-math added), 0 broken
  (<pre lang=math>), 0 inline hazards.

## 2026-07-07 — Experiment 20: differentiable-generation behavioral supervision (breaks Exp 11's ceiling)
- **Why:** the single substantive open lever flagged since Exp 11 (PLAN Next-step (i)). Exp 11's
  behavioral-preservation term matched the corrector's downstream sentiment readout on a TEACHER-FORCED
  pass and hit a ceiling (generated effect never past ≈+1.3), diagnosed as a proxy gap — a teacher-forced
  readout only partially transfers to autoregressive generation. Exp 20 supervises the readout on the
  corrector's OWN generated continuation to close that gap.
- **Method (new):** a DIFFERENTIABLE soft-token rollout. From P=8 real prompt tokens, roll out K=8 steps
  with the steer applied at LAYER at every position; read the downstream sentiment projection at L2 for the
  produced position; feed the softmax-weighted expected embedding softmax(ℓ/τ)·Wₑ (τ=1) back as the next
  input (fully differentiable in r_θ). Push the corrected rollout's readout toward RAW steering's own no-grad
  rollout, weight λ_g, backpropped through the K-step unroll. Total loss = teacher-forced LM CE (Exp 3) +
  λ_near·‖P_{v⊥}r‖²/100² + λ_g·mean_K((p_corr−p_raw)/100)². Trained λ_g∈{0,40,160} (λ_g=0 = Exp 10/11 base),
  scored on the identical Exp 10 protocol (48 prompts, 30 greedy tokens, effect B(α)−B(0) + distinct-2).
  GEN_B=4 rollout batch (VRAM), no OOM. Ran with dir9's cupenv python (shared conda transformers still absent).
- **Result (new):** PARTIAL POSITIVE — the generation-aware signal breaks Exp 11's ≈+1.3 effect ceiling.
  λ_g=0 reproduces Exp 10/11 to the digit (eff +0.17/+0.19/+0.15/+0.48). λ_g=40: eff +1.01/+1.40/+1.30/+1.72,
  d2 0.67/0.67/0.54/0.47 — at α=8 effect +1.72 (vs Exp 11's best +1.23/+1.08) at d2 0.47 (vs raw's collapsed
  0.32), nearly matching raw's already-collapsed +1.77. λ_g=160 at α=2 reaches eff +1.61 at near-baseline
  d2 0.71 (dominates Exp 11's +0.99@0.73), but OVER-weighting collapses at strong steering: destabilizes
  training (one LM step spiked to ~20) and degenerates like raw at α≥6 (eff +0.61 then −0.22 @α=8, d2→0.32,
  sample repeats "the Southern-the-Beal and the Southern-the-Beal…"). So supervising on the autoregressive
  distribution is a strictly better lever than teacher-forced at moderate steering and raises the achievable
  strong-α effect (+1.08→+1.72 @α=8), but the strong-effect-AND-fluent corner still eludes. Frontier pushed
  out a SECOND time, not erased. No prior result superseded (Exp 10/11 unchanged; Exp 20 is a new follow-up).
- **Deliverable deltas:** RESULTS.md +Exp 20 (table + reading) + figure entry + Headline "Second fix (Exp 20)"
  clause on the behavioral paragraph. REPORT.md +Methods "Differentiable-generation behavioral supervision
  (Experiment 20)" subsection (2 new display-math: soft-token feedback + total loss) + Results Exp 20 (table +
  interpretation + figure) + Summary clause + Conclusion clause.
- New code experiments/20_diff_generation.py, figure plots/20_diff_generation.png, results
  results/20_diff_generation.json, log results/20_run.log.
- REPORT math re-verified via GitHub API: 26/26 js-display-math (2 new), 0 broken (<pre lang=math>), 0 inline hazards.

## 2026-07-07 — Experiment 21: cross-ARCHITECTURE generality (Qwen3-1.7B, non-GPT-2)
- **Why:** Exps 13/19 scaled the model (124M→355M→774M) but every point stayed inside the GPT-2 family
  (learned positions, LayerNorm, dense MHA, GELU). The remaining external-validity gap was whether the
  flagship result is a GPT-2-*architecture* artifact. Exp 21 tests a genuinely different architecture.
- **Method (new):** replicated the EXACT flagship Exp-3 pipeline UNCHANGED on Qwen3-1.7B (28 blocks,
  d=2048) at mid layer block 14/28 — a modern architecture differing from GPT-2 on every structural axis:
  RMSNorm (not LayerNorm), rotary position embeddings (not learned), SwiGLU MLP (not GELU), grouped-query
  attention (16 query / 8 KV heads, not dense MHA). Same DiffMean sentiment prompts, 400-doc Gaussian fit,
  300-doc train, held-out 100-doc eval, 4-layer projection-preserving corrector vs downstream LM loss,
  seed, α~U(0.5,8). Only the model changes (|v|=38.1, mean|h|=301.9, clean D_M=44.7; corrector 8.39M @
  d=2048). Qwen3 loaded bf16 for the ~4.3 GB VRAM share; corrector fp32 with a bf16 boundary at the patch
  hook; train batch 2, EVAL batch 1 (full 151,936-token vocab logits dominate memory at d=2048);
  expandable_segments alloc. Ran with dir9's cupenv python (shared conda transformers still absent).
- **Result (new):** POSITIVE — both headline facts replicate on a non-GPT-2 architecture. Raw steering
  breaks the LM (ΔLM@8 +3.43, D_M 44.7→77.8); the identical corrector recovers it at matched projection —
  recovery @α=8 = **94%** (ΔLM +3.43→+0.19), **108%** @α=4 (ΔLM even below clean baseline, free-or-better
  weak-α as on every GPT-2 scale), retention matched α|v| (38.1→304.8). Corrected activation FURTHER off the
  Gaussian manifold than raw at every α (122.2 vs 77.8 @α=8; Exp-2/3 decoupling holds a 4th time). 94% @α=8
  edges GPT-2 small's 84%. ⇒ architecture-robust across LayerNorm↔RMSNorm, learned↔rotary, GELU↔SwiGLU,
  dense↔grouped-query attention. No prior result superseded (Exp 21 is a new follow-up).
- **Deliverable deltas:** RESULTS.md +Exp 21 block (table + reading) + figure entry + Headline
  model-robustness clause extended to architecture. REPORT.md +Methods "Cross-architecture generality
  (Experiment 21)" subsection (no new display math — reuses Exp 12's recovery equation) + Results
  "Experiment 21" (table + Observation/Interpretation/Limitations/Next-check) + Summary clause + Conclusion
  clauses (2).
- New code experiments/21_cross_arch.py, figure plots/21_cross_arch.png, results
  results/21_cross_arch.json, checkpoint results/21_corr.pt, log results/21_run.log.
- REPORT math re-verified via GitHub API: 26/26 js-display-math (unchanged), 0 broken (<pre lang=math>),
  0 inline hazards.

## 2026-07-07 — Experiment 22: behavioral reality-check on Qwen3 (is Exp 21's 94% recovery honest?)
- **Why:** Exp 21's 94% recovery on Qwen3-1.7B is a TEACHER-FORCED ΔLM at matched layer-14 projection. Exp 10
  showed on GPT-2 that this proxy is partly misleading — the corrector's fluency win came at the cost of a
  WEAKER propagated behavioral edit in generation (correction ⟂ v in activation space but not ⟂ the downstream
  readout). That caveat had never been checked off GPT-2 (Exp 21's own "Next check"). Exp 22 tests it directly.
- **Method (new):** ran the IDENTICAL Exp 10 protocol on Qwen3-1.7B, reusing the EXACT Exp 21 corrector
  (checkpoint results/21_corr.pt, no retraining). Greedy-generate 30-token continuations from 48 held-out
  12-token prompts with the steer applied at block 14 every position, raw vs corrected; on a CLEAN re-encode
  measure sentiment effect B(α)−B(0) (proj of continuation resid_post@14 onto v̂; baseline B0=+28.6) and
  distinct-2 (unique-bigram ratio; baseline 0.875). Only model/corrector differ from Exp 10; vector, prompt
  split, α grid, metrics identical → directly comparable. GEN_BATCH 8 (VRAM), empty_cache between chunks; no
  OOM. Ran with dir9's cupenv python (shared conda transformers still absent).
- **Result (new):** the Exp 10 under-steering caveat REPLICATES on Qwen3 — 94% is honest as a teacher-forced
  metric but is again partly bought by a weaker behavioral edit. Effect raw/corr: +5.22/+0.53 (α=2), +7.31/
  +0.77 (α=4), +7.64/+0.98 (α=6), +8.01/+2.31 (α=8) — corrector effect only 10–29% of raw's (cf. ~1/6 on GPT-2
  Exp 10). Distinct-2: raw 0.886→0.761, corrector flat 0.825–0.843. KEY DIFFERENCE from GPT-2: raw steering
  degenerates FAR LESS on Qwen3 (distinct-2 0.76 vs GPT-2's 0.32 collapse @α=8), so raw is a STRONGER baseline
  here (steers hard AND stays fairly fluent) and the corrector's fluency edge is smaller (0.06 @α=8) — the
  effect-vs-fluency Pareto is shallower than on GPT-2. ⇒ "matched projection ≠ matched behavioral steering" is
  architecture-robust; the Exp 11/20 behavioral-preservation terms (GPT-2-tested) are the indicated fix if
  strong behavioral steering is required on Qwen3. No prior result superseded (Exp 22 is a new follow-up;
  Exp 21's ΔLM numbers unchanged — Exp 22 measures a different, behavioral quantity).
- **Deliverable deltas:** RESULTS.md +Exp 22 (table + reading) + figure entry + Headline behavioral-caveat
  paragraph now notes architecture-robustness (Exp 22). REPORT.md +Results "Experiment 22" (table +
  Observation/Interpretation/Limitations/Next-check; reuses Exp 10's behavioral-metric definitions, no new
  display math) + Exp 21 Limitations/Next-check updated (behavioral check now DONE in Exp 22) + Summary
  behavioral clause + Conclusion behavioral clause + Limitation (2) all note the caveat is architecture-robust.
- New code experiments/22_behavioral_qwen.py, figure plots/22_behavioral_qwen.png, results
  results/22_behavioral_qwen.json, log results/22_run.log.
- REPORT math re-verified via GitHub API: 26/26 js-display-math (unchanged — Exp 22 adds no equation),
  0 broken (<pre lang=math>), 0 inline hazards.

## 2026-07-08 — Experiment 23: does the GPT-2 behavioral-preservation fix (Exp 11) transfer to Qwen3?
- **Why:** Exp 22 showed the Exp 10 under-steering caveat replicates on Qwen3-1.7B and named the Exp 11/20
  behavioral-preservation term as the indicated fix, but never tested whether the fix transfers across the
  architecture boundary (Exp 22's own "Next check"; PLAN Next-step ii). This completes an experiment a prior
  iteration had started (script + λ_b=10 checkpoint present, but no JSON/plot — the run was interrupted).
- **Method (new):** reused the exact Exp 21/22 Qwen3 pipeline + the identical Exp 22 generation protocol, and
  added the Exp 11 behavioral term at a DOWNSTREAM Qwen3 layer L2=27 (last decoder block, DiffMean ŵ |w|=12.9):
  push the corrected activation's downstream sentiment readout p_corr toward RAW steering's p_raw, weight
  λ_b∈{0,10,40}. λ_b=0 LOADS the exact Exp 21 checkpoint (= Exp 22 corrector, reproducibility anchor); λ_b∈{10,40}
  trained fresh with the Exp 21 recipe/seed/data. Scored each on the Exp 22 protocol (48 prompts × 30 greedy
  tokens; effect B(α)−B(0), distinct-2 on clean re-encode; baselines B0=+28.6, distinct2=0.875). GEN_BATCH 8, no OOM.
- **Result (new, nuanced/corrective):** the fix's MECHANISM transfers, its PARETO ADVANTAGE does not. Adding λ_b
  lifts the generated effect from the base corrector's +0.53/+0.77/+0.98/+2.31 (10–29% of raw's, = Exp 22) to
  λ_b=40 +4.06/+5.87/+6.35/+4.21 — 53–83% of raw's effect at α≤6, a 2–8× increase, exactly the Exp 11 lever. BUT on
  Qwen3 the corrector does NOT beat raw: at λ_b=40 its distinct-2 (0.875→0.673) sits slightly BELOW raw's
  (0.886→0.761) at every α while its effect is also below raw's, so raw weakly dominates at matched α. Reason: on
  GPT-2 the term won by dominating a COLLAPSED raw (distinct-2 0.32); Qwen3's raw does not collapse (0.761 @α=8), so
  there is no degenerate baseline to beat. Strong-steering wobble replicates (λ_b=40 @α=8 effect drops to +4.21 <
  its α=6 peak +6.35, distinct-2 0.673) — the Exp 20 λ_g=160 over-steer instability. ⇒ the behavioral fix is
  architecture-robust as a lever on generated effect; its payoff is GATED by whether the raw baseline degenerates.
  Closes the behavioral arc (Exp 10→11→20→22→23). λ_b=0 reproduces Exp 22 to the digit (reproducibility check).
  No prior result superseded (Exp 23 is new; Exp 22's λ_b=0 numbers reproduced).
- **Deliverable deltas:** RESULTS.md +Exp 23 (table + reading) + figure entry + Headline behavioral-fix-transfer
  clause. REPORT.md +Methods "Behavioral-fix transfer across the architecture boundary (Experiment 23)" subsection
  (no new display math — reuses Exp 11's behavioral-loss equation) + Results "Experiment 23" (table +
  Observation/Interpretation/Limitations/Next-check) + Exp 22 Next-check marked done + Summary clause + Conclusion
  behavioral clause + Limitation (2) clause.
- New code experiments/23_behavioral_qwen_fix.py, figure plots/23_behavioral_qwen_fix.png, results
  results/23_behavioral_qwen_fix.json, checkpoints results/23_corr_lamb{10,40}.pt, log results/23_run.log.
- REPORT math re-verified via GitHub API: 26/26 js-display-math (unchanged — Exp 23 adds no equation),
  0 broken (<pre lang=math>), 0 inline hazards.

## 2026-07-08 — Experiment 24: second non-GPT-2 architecture (Pythia-410m / GPT-NeoX) — architecture SWEEP
- **Why:** Exp 21 crossed the GPT-2 architecture boundary ONCE (to Qwen3-1.7B). A single point off the
  GPT-2 family is a weak basis for "architecture-robust." Added a THIRD, structurally distinct family so
  the axis is a genuine sweep. Pythia-410m = GPT-NeoX: shares rotary with Qwen3 and LayerNorm/GELU/dense-MHA
  with GPT-2, but its block uses a PARALLEL residual (attention + MLP from the same input, summed) unlike
  BOTH GPT-2's and Qwen3's serial residual — the untested structural axis.
- **Setup:** replicated the EXACT flagship Exp-3 pipeline UNCHANGED on Pythia-410m (24 blocks, d=1024),
  steer & correct at mid layer block 12/24; same DiffMean sentiment prompts / 400-doc Gaussian fit /
  300-doc train / held-out 100-doc eval / 4-layer corrector (5.25M) / seed / α∼U(0.5,8) / objective.
  Only the model changes (|v|=3.29, mean|h|=35.3, clean D_M=31.3). Ran fp32 in the ~4.3 GB VRAM share,
  batch 4 train+eval (small 50304 vocab, no eval bottleneck).
- **Result (new):** both headline facts replicate. (P) raw steering breaks the LM — ΔLM +3.10 @α=8,
  D_M 31.3→52.3. (C) identical corrector recovers **81% @α=8, 81% @α=4** (71% @α=2), matched projection
  (retention α|v| exactly 3.29→26.29), corrected activation FURTHER off the Gaussian manifold at every α
  (89.4 vs 52.3 @α=8 — decoupling holds a 5th time). α=1 recovery 41% is noise-dominated (raw damage only
  +0.06 nats there). Architecture axis is now a 3-family SWEEP: GPT-2 small/med/large 84/89/84%, Qwen3 94%,
  GPT-NeoX 81% — all 81–94% @α=8.
- **RESULTS.md:** +Experiment 24 (table + reading) + figure entry (plots/24_cross_arch_pythia.png) +
  Headline architecture clause upgraded from "single boundary crossing" to a 3-family sweep (81–94% band).
- **REPORT.md:** +Methods "A second non-GPT-2 architecture (Experiment 24)"; +Results "Experiment 24"
  (Observation/Interpretation/Limitations/Next-check); Exp 21 Limitations/Next-check updated (sweep concern
  now addressed); Summary + Conclusion (two clauses) architecture claims upgraded to a 3-family sweep.
  No result superseded (Exp 24 is new; Exp 21's Qwen3 numbers unchanged).
- New code experiments/24_cross_arch_pythia.py, figure plots/24_cross_arch_pythia.png, results
  results/24_cross_arch_pythia.json, checkpoint results/24_corr.pt, log results/24_run.log.
- REPORT math re-verified via GitHub API: 26/26 js-display-math (unchanged — Exp 24 reuses Exp 3/12
  ΔLM/recovery/D_M definitions, adds no equation), 0 broken (<pre lang=math>), 0 inline hazards.

## 2026-07-08 — Experiment 25: behavioral reality-check on Pythia-410m (Exp 24's Next check) — NUANCED POSITIVE
- **Why:** Exp 24's 81% recovery on Pythia-410m/GPT-NeoX is a TEACHER-FORCED ΔLM. Exp 10 (GPT-2) and Exp 22
  (Qwen3) both showed matched layer projection can hide a weaker propagated behavioral edit in generation. This
  is Exp 24's own "Next check", and closes the behavioral check on the third architecture. A prior iteration had
  written the script (experiments/25_behavioral_pythia.py) but never run it (no JSON/plot/log); this iteration ran it.
- **Method:** identical Exp 10/22 generation protocol on Pythia-410m, reusing the EXACT Exp 24 corrector
  (results/24_corr.pt, block-12 sentiment, no retraining): greedy-generate 30 tokens from 48 held-out 12-token
  prompts, steer at block 12 every position, raw vs corrected; on a clean re-encode measure sentiment effect
  B(α)−B(0) (baseline B0=−4.77) and distinct-2 (baseline 0.77). GEN_BATCH 8, fp32, no OOM.
- **Result (new):** the under-steering caveat is MILDER than on GPT-2/Qwen3. Corrected effect
  +0.90/+0.80/+0.93/+0.98 (α=2/4/6/8) is ABOVE raw's +0.17/+0.40 at α≤4 and 84–92% of raw's +1.01/+1.17 at α≥6 —
  not the ~1/6 shortfall of GPT-2 (Exp 10) or 10–29% of Qwen3 (Exp 22). At α=8 the corrector Pareto-DOMINATES raw
  (effect +0.98 at distinct-2 0.72, vs raw effect +1.17 but collapsed distinct-2 0.38). Mechanism: raw steering
  itself propagates weakly on Pythia at these α (effect peaks +1.17), so there is little behavioral effect for the
  corrector to lose — the size of the "matched projection ≠ matched steering" penalty tracks how strongly raw
  steering propagates in a given model. Limitation: small effect magnitudes (low-signal regime); best read as
  "penalty mild here," not "corrector steers more than raw in general." No prior result superseded (Exp 25 is new;
  reuses Exp 24 checkpoint).
- **Deliverable deltas:** RESULTS.md +Exp 25 (table + reading) + figure entry + Headline behavioral-caveat
  Pythia sentence. REPORT.md +Exp 25 Results subsection (Observation/Interpretation/Limitations/Next check; reuses
  Exp 10 behavioral-metric definitions, NO new equation).
- New outputs: results/25_behavioral_pythia.json, plots/25_behavioral_pythia.png, results/25_run.log.
- REPORT math re-verified via GitHub API: 26/26 js-display-math (unchanged — Exp 25 adds no equation), 0 broken
  (<pre lang=math>), 0 inline hazards.

## 2026-07-08 — Finalization pass (no deliverable numbers changed)
- Verification-only iteration: with the direction complete on all seven external-validity axes
  (strength/direction/layer/model-scale/architecture/prompt-family/steering-family) and the full
  behavioral arc closed on all three architectures (GPT-2 Exp 10 / Qwen3 Exp 22 / Pythia Exp 25),
  confirmed RESULTS.md and REPORT.md are current-best: both carry every experiment through Exp 25,
  all plots/2[1-5]_*.png present.
- REPORT math re-verified via GitHub API: 26/26 js-display-math, 0 broken (<pre lang=math>),
  0 inline-math hazards. No result superseded; no equation added. STOP file created.

## 2026-07-09 — REPORT restructure per human feedback (no result numbers changed)
- **Trigger:** operator feedback (`human_feedback_07082201.md`): "the report is too long and too much back and
  forth and I cannot understand it" → asked to disassemble REPORT.md into 2–4 topic-focused mini reports.
- **Change:** the 1744-line monolithic REPORT.md was split into a short **index** + **four self-contained parts**,
  each a clean linear narrative on ONE topic (Summary → Methods → Results → Conclusion; Methods gives Data/Model/
  Layer and defines every metric & baseline with rendered equations):
  - `REPORT.md` (was 1744 lines → now 108): overall Summary, the takeaway equation, a headline-numbers table, a
    limitations overview, and links to the four parts.
  - `REPORT_1_core_correction.md` — Exp 2,3,4,5,16,17 (Gaussian corrector backfires; LM-supervised fix recovers
    84% @α=8 by moving further off-manifold; why the Gaussian is the wrong yardstick; diffusion-framing ablation).
  - `REPORT_2_amortization.md` — Exp 6,7,8,9,14 (direction-conditional corrector on a bank; scaling/curation
    fail; bank-diversity lever confirmed causally).
  - `REPORT_3_external_validity.md` — Exp 12,13,19,21,24,15,18 (layer / model-scale / architecture-sweep /
    prompt-family / steering-vector-family robustness; 81–94% @α=8 across three architectures).
  - `REPORT_4_behavioral.md` — Exp 10,11,20,22,23,25 (matched projection ≠ matched steering; readout-preservation
    and differentiable-generation fixes across three architectures).
- **Fidelity:** all Methods/Results blocks copied VERBATIM from the source — no number, equation, or figure
  reference altered; only each part's Summary/Conclusion is newly written (topic-scoped, no cross-topic
  back-and-forth). Added one previously-orphaned figure reference (`plots/25_behavioral_pythia.png`) to Part 4.
- **Verification:** GitHub-API math check on all five files — 42 js-display-math total (index 1 / P1 16 / P2 6 /
  P3 9 / P4 10), 0 broken (`<pre lang=math>`), 0 inline-math hazards. Every part has all four top-level sections.
- **RESULTS.md:** left unchanged (a distinct per-experiment results log; the feedback targeted the narrative
  report's length/back-and-forth). No experiment re-run; no result superseded.

## 2026-07-09 — Experiment 26: seed robustness / confidence interval on the flagship recovery (new result)
- **Trigger:** every prior experiment is a single run at `SEED=0`, so the headline "84% recovery @α=8"
  (Exp 3) had no error bar. CLAUDE.md rule 10 names *seed* as a control a trustworthy metric must survive,
  and it was the one axis never varied (strength/direction/layer/model/prompt/steering-family/architecture
  all were).
- **Change (new, supersedes nothing):** added **Experiment 26** to RESULTS.md — re-ran the EXACT flagship
  Exp-3 pipeline (same vector `|v|=11.08`, 400-doc Gaussian fit, 300-doc train, held-out 100-doc eval,
  4.46M corrector, recipe, `α∼U(0.5,8)`) at **5 seeds (0–4)** and reported mean ± sd of the fluency recovery.
  Result: recovery **83.3 ± 2.0% @α=8** (per-seed 84.3/84.5/84.6/83.0/80.0%), **96.2 ± 0.8% @α=4**,
  **90.0 ± 0.6% @α=6**; ΔLM learned @α=8 = +0.464 ± 0.054 nats vs raw +2.778. Seed 0 reproduces Exp 3 to the
  digit (84.3% ≈ 84%), a built-in check. Wide bar only at α=1 (196 ± 19%) — a ratio artifact of raw's
  near-zero +0.076-nat damage there, not real spread (absolute ΔLM_learned tight at −0.073 ± 0.014).
- **Deliverables:** RESULTS.md gained the Exp-26 section + figure entry; the Headline now carries the seed CI
  ("83.3 ± 2.0% across 5 training seeds"). No prior result number changed (Exp 26 is additive; Exp 3's 84%
  is confirmed as representative). Figure `plots/26_seed_robustness.png`; artifacts
  `experiments/26_seed_robustness.py`, `results/26_seed_robustness.json`, `results/26_run.log`.
- **Limitation recorded:** varies only the training seed (corrector init + α-sampling/data-shuffle RNG);
  eval set, Gaussian fit, and steering vector held fixed — bounds optimization variance, not eval-doc or
  vector-construction sampling variance.
- **Ops:** the shared conda `transformers` was still absent, but `/opt/conda/bin/python` (transformers
  5.13.0, torch 2.9 cu130, matplotlib 3.11, on LOCAL disk) is available and imports in seconds — vastly
  faster than dir9's `cupenv` on the contended `/mars-vol` network volume (a cold import there stalled ~30 min
  in `folio_wait_bit_common` grinding scipy/sklearn). Switched to `/opt/conda/bin/python`; run completed in
  ~15 min. Recommend this env for future iterations.
