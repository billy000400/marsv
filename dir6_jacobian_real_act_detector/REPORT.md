# REPORT — Direction #6: What makes real activations "real"?

**Model/data:** GPT-2 small, residual stream `resid_post` (= `hidden_states[L+1]`), layers {3,6,9},
all token positions pooled, FineWeb. Real activations REUSED from the Direction-3 cache
(`../dir3_manifold/data/acts_layer{3,6,9}.npy`, [200000,768] float16). Environment had only numpy
(no sklearn/scipy); all metrics (AUROC/AUPRC, logistic, MLP, kNN, PCA, Mahalanobis) are pure-numpy.
The functional probe uses `transformers` (reinstalled `--no-deps`; torch 2.9.0+cu130 / numpy 2.3.3
left untouched) on the A10 GPU.

## Question
Do real residual activations have local statistical / geometric / functional properties beyond their
first two moments, and do those properties generalize across hard synthetic negatives? (Hypothesis H1.)

## Benchmark
- **Positives:** real activations, split into contiguous TRAIN/EVAL blocks with a 5k-row gap (a
  leakage-*reduction* heuristic — the Direction-3 cache stores activations in document order but
  per-row document IDs were not saved, so this is "contiguous split + gap", not a verified
  document-level split); negatives derived from EVAL reals only; statistics fit on TRAIN only.
- **Negative families (7):** isotropic Gaussian, covariance-matched Gaussian (Cholesky of shrinkage
  cov), coordinate-shuffle (norm preserved exactly), norm-preserving perturbation, **convex
  interpolation between two reals** (norm-matched), top-PCA-tangent perturbation, orthogonal-complement
  perturbation. The last three are norm-matched, manifold-aware hard negatives.
- **Scores (10):** norm, mean-L2, Mahalanobis (shrinkage), PCA reconstruction error, coordinate-quantile,
  kNN distance (statistical); next-token entropy, max-softmax-prob, plateau-KL, max-logit (functional).
  NOTE: the *discrimination* functional probe (Phase 3, `functional_features.py`) continues the model
  from the activation as a SINGLE-POSITION input, so it measures intrinsic model sensitivity, not
  in-context fidelity. The *prediction* and *causal* phases (5 & 6) use the corrected genuinely
  in-context method (forward hook over the full prompt; see Correction section).
- **Protocols:** per-family AUROC/AUPRC; leave-one-corruption-family-out (LOFO) for learned and combined
  detectors; cross-depth replication at layers 3/6/9.

## Key results
1. **No single statistic is "realness."** Norm is a shortcut (AUROC 0.50 on the two norm-matched
   families). Each strong baseline has a structural blind spot.
2. **Local density (kNN) and global covariance (Mahalanobis) are complementary and depth-stable.**
   kNN macro 0.913 at L6; it is the only method catching covariance-matched Gaussians (0.97, which
   Mahalanobis cannot see by construction, 0.53). Mahalanobis owns low-variance-direction moves
   (shuffle 1.0, orth_pert 0.88–0.92, norm_pert 0.86). Together they cover Gaussian/shuffle/perturbation
   negatives at AUROC 0.86–1.0 across all three layers.
3. **Learned discriminative detectors memorize generators and do NOT generalize.** Logistic/MLP trained
   on three families and tested on a held-out fourth give LOFO macro 0.68/0.74 — *below* the
   unsupervised kNN baseline (0.91) — collapsing to ~chance on unseen covariance-matched Gaussian
   (0.48–0.51) and isotropic Gaussian (logreg 0.49). Realness is better modeled as a one-class/density
   property than a classification boundary.
4. **The subtle hard core — interpolation — is anomalous by being too CENTRAL, not too far.** A convex
   combination of two real activations (renormed to original norm) is invisible to one-sided "too-far"
   anomaly scores and to local density (kNN 0.50, since it sits among real neighbors). But it has
   *lower* Mahalanobis distance than real (658 vs 803; norm exactly matched): a **two-sided** global
   Mahalanobis detects it at ~0.68. Real activations occupy a characteristic-distance *shell*;
   averaging falls into the over-typical interior (high-dimensional Gaussian-annulus intuition). An
   independent **functional probe** (entropy / plateau-KL) also catches interpolation at ~0.61,
   evidencing a genuine functional component of realness.
5. **Opposite-direction anomalies break combined detectors.** Because interpolation is anomalous in the
   opposite direction from ordinary corruptions, a combined logistic over {Mahalanobis, kNN, entropy,
   plateau-KL} trained LOFO generalizes to covariance-matched Gaussian (0.999) but FAILS on interpolation
   (0.54) — a concrete, reproducible generalization limit for any single realness score. (Caveat: the
   per-family combined LOFO AUROC is reported with held-out-label orientation, `max(au, 1-au)`; this is
   an optimistic diagnostic of separability, not a strictly orientation-fixed deployed-detector number.
   The qualitative split — generalizes to cov_gauss, fails on interp — is robust to orientation.)
6. **The functional axis PREDICTS downstream degradation; the density axis is a proximity proxy.** In a
   context-aware sweep, plateau-KL/entropy predict true in-context KL(clean‖corrupt) beyond
   distance-to-original (partial ρ +0.17…+0.57, positive in both noise and interp sweeps), while
   Mahalanobis/kNN add nothing over proximity. So the property best for *discriminating* interpolations
   (two-sided Mahalanobis) differs from the property best for *predicting* downstream harm (functional).
7. **No realness score is a valid CAUSAL repair objective (negative result).** Gradient descent on
   Mahalanobis or plateau-KL to "repair" a corrupted activation makes downstream KL WORSE than the
   corrupted start and worse than a matched random move (Mahalanobis → over-central shell-trap;
   plateau-KL → Goodhart degenerate region), while the oracle move toward the true clean activation
   recovers behavior (in-context KL 0.78→0.009). Discriminative/predictive ≠ causal; naive descent
   reward-hacks these.

## Correction (post-review, codex 20260623T024606Z): genuinely in-context re-run of Phases 5 & 6
An external review correctly flagged that the first Phase-5/6 scripts continued the forward pass by
feeding a **single position** `[B,1,768]` through the late blocks, so later-layer attention could not
attend to the prompt — i.e. the "in-context KL" was actually single-position late-block continuation.
Phases 5 and 6 were **re-run with a genuine in-context method** (`context_validation_v2.py`,
`causal_repair_v2.py`): a forward hook overwrites ONLY the last-token resid_post@L6 during a FULL
model forward, so all other positions keep their true context. Sanity: severity-0 corruption gives
mean downstream KL = 0.0000 exactly (the hook injects the true clean residual). **Both conclusions
survive the correction:**
- **Claim 3 holds (in-context).** Functional plateau-KL/entropy keep positive partial-ρ over the
  distance control in BOTH sweeps (noise: plateau-KL +0.51, entropy +0.35; interp: plateau-KL +0.25,
  entropy +0.19); raw plateau-KL Spearman actually *rises* in-context (0.55→0.79 noise, 0.13→0.52
  interp). Density scores (Mahalanobis/kNN) keep NEGATIVE partials on the noise sweep — proximity
  proxies, as before. The headline magnitudes are essentially unchanged.
- **Claim 4 still fails (in-context), now with properly move-matched controls** (addressing the review's
  second point that `func_descent` lacked a move-matched random control): maha_descent in-context KL
  0.78→3.33 vs its matched random move 1.99; func_descent →8.82 vs its OWN func-budget-matched random
  move 2.20; oracle clean-direction move →0.009. Both descents are worse than their *exactly* matched
  random controls. (The in-context corrupted-start KL is 0.78, lower than the buggy 2.20, because
  context anchors the prediction — the single-position version overstated corruption impact — but the
  qualitative verdict is identical.)

## Verdict (5-claim ladder)
- **Discrimination — YES**, but only via a multi-axis score (density ⊕ two-sided covariance ⊕
  functional); no single statistic suffices.
- **Generalization — PARTIAL.** Generalizes across Gaussian/shuffle/perturbation families and across
  layers 3/6/9, but not to opposite-direction interpolation negatives.
- **Prediction — PARTIAL/YES, functional axis only.** In a *genuinely in-context* corruption-severity
  sweep (corrupt last-position resid_post of 400 real prompts via a forward hook, continue the FULL
  forward, measure true KL(clean‖corrupt); severity-0 → KL 0 exactly), the functional scores
  plateau-KL/entropy predict downstream KL *beyond* the distance-to-original control (partial Spearman
  +0.19…+0.51, positive across both noise and interp sweeps); statistical density scores add no value
  over proximity (negative partials on the noise sweep). The *functional* axis carries the incremental
  predictive signal, the *density* axis is a proximity proxy. (Verified on the corrected in-context
  re-run, `context_validation_v2.py`.)
- **Causality — NO (negative result).** Gradient-descending a corrupted activation to improve EITHER a
  density (Mahalanobis) or functional (plateau-KL) realness score makes the true downstream KL WORSE
  than the corrupted start and worse than a *move-matched* random move: Mahalanobis-descent collapses
  into the over-central interior (the shell-distance trap), func-descent Goodharts the flatness score
  into a degenerate region. The oracle move toward the true clean activation recovers behavior
  (in-context KL 0.78→0.009 at the same budget), so the failure is the score, not the optimizer. These
  realness scores are valid *detectors/predictors* but invalid *causal repair objectives*. (Confirmed
  on the corrected in-context re-run with per-descent move-matched random controls,
  `causal_repair_v2.py`.)
- **Steering preservation — NOT TESTED**, but (4)'s failure implies a steering-repair regularizer built
  on these scalar scores would degrade behavior; a usable objective must respect the data shell.
- **H1 — SUPPORTED.** Real activations carry geometric (shell-distance) and functional (model-sensitivity)
  structure beyond first/second moments. The effect is real but smaller and more orientation-dependent
  than a naive "interpolations fool everything" reading; the honest framing is that statistics need a
  *two-sided* view and a complementary functional probe to capture realness.

## Limitations
- The discrimination functional probe (Phase 3) treats each activation as a single-position input
  (intrinsic model sensitivity, not in-context fidelity). Phases 5 & 6 (PREDICTION, CAUSAL) now use a
  genuinely in-context method (forward hook over the full prompt); a fully context-aware
  *discrimination* benchmark (negatives generated in-context) is still future work.
- The combined-detector LOFO AUROC uses held-out-label orientation (diagnostic, not orientation-fixed).
- Partial-correlation evidence for prediction uses approximate rank-residual partials; the consistent
  positive functional partials are robust, the exact magnitudes are not.
- Activations pool all token positions, including pos-0 "attention-sink" tokens (norm ≈ 3000), an
  un-stratified confound for norm/Mahalanobis. Token-position stratification is future work.
- GPT-2 small only; single model. No cross-model transfer.
- N (≈2k–10k per family) gives stable AUROCs but bootstrap CIs were not computed.

## Implications for Direction 1 (steering)
Two-sided global Mahalanobis is a good *diagnostic* for the "too-central"/over-shrunk regime that
naive small-alpha steering produces (real activations occupy a characteristic-distance shell, not the
interior). BUT Phase 6 is a clear warning: do NOT use these scalar realness scores as *optimization
objectives* for steering repair — gradient descent on Mahalanobis or plateau-KL reward-hacks them and
degrades behavior below even a random move. A usable causal objective must constrain movement toward
the data shell/manifold (e.g. a learned generative/denoising prior, or projection onto the local
data tangent), and any steering correction must be evaluated against in-context downstream KL at
*matched achieved steering effect*, with reward-hacking checked via metrics outside the objective.

## Reproduce
`experiments/mvp_benchmark.py` (Phase 2 L6) · `train_detectors.py` (LOFO detectors) ·
`baselines_layers.py` (cross-depth + hard negatives) · `functional_features.py` (functional probe) ·
`combined_score.py` (combined + interp correction) · `context_validation_v2.py` (Phase-5 CORRECTED
in-context prediction) · `causal_repair_v2.py` (Phase-6 CORRECTED in-context causal repair, negative
result). The pre-correction single-position versions (`context_validation.py`, `causal_repair.py`) are
retained for provenance. Results in `results/*.csv`, `results/*.json` (the `_v2` files are canonical
for Phases 5/6).
