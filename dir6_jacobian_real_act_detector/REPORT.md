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
- **Positives:** real activations, split into contiguous TRAIN/EVAL blocks with a gap (document-leakage
  control); negatives derived from EVAL reals only; statistics fit on TRAIN only.
- **Negative families (7):** isotropic Gaussian, covariance-matched Gaussian (Cholesky of shrinkage
  cov), coordinate-shuffle (norm preserved exactly), norm-preserving perturbation, **convex
  interpolation between two reals** (norm-matched), top-PCA-tangent perturbation, orthogonal-complement
  perturbation. The last three are norm-matched, manifold-aware hard negatives.
- **Scores (10):** norm, mean-L2, Mahalanobis (shrinkage), PCA reconstruction error, coordinate-quantile,
  kNN distance (statistical); next-token entropy, max-softmax-prob, plateau-KL, max-logit (functional,
  from continuing the model forward from the activation as a single-position input — verified to
  reconstruct true logits exactly when given full context).
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
   (0.54) — a concrete, reproducible generalization limit for any single realness score.
6. **The functional axis PREDICTS downstream degradation; the density axis is a proximity proxy.** In a
   context-aware sweep, plateau-KL/entropy predict true in-context KL(clean‖corrupt) beyond
   distance-to-original (partial ρ +0.17…+0.57, positive in both noise and interp sweeps), while
   Mahalanobis/kNN add nothing over proximity. So the property best for *discriminating* interpolations
   (two-sided Mahalanobis) differs from the property best for *predicting* downstream harm (functional).
7. **No realness score is a valid CAUSAL repair objective (negative result).** Gradient descent on
   Mahalanobis or plateau-KL to "repair" a corrupted activation makes downstream KL WORSE than the
   corrupted start and worse than a matched random move (Mahalanobis → over-central shell-trap;
   plateau-KL → Goodhart degenerate region), while the oracle move toward the true clean activation
   recovers behavior (KL 2.20→0.03). Discriminative/predictive ≠ causal; naive descent reward-hacks these.

## Verdict (5-claim ladder)
- **Discrimination — YES**, but only via a multi-axis score (density ⊕ two-sided covariance ⊕
  functional); no single statistic suffices.
- **Generalization — PARTIAL.** Generalizes across Gaussian/shuffle/perturbation families and across
  layers 3/6/9, but not to opposite-direction interpolation negatives.
- **Prediction — PARTIAL/YES, functional axis only.** In a context-aware corruption-severity sweep
  (corrupt last-position resid_post of 400 real prompts, continue in-context, measure true
  KL(clean‖corrupt)), the functional scores plateau-KL/entropy predict downstream KL *beyond* the
  distance-to-original control (partial Spearman +0.17…+0.57, positive across both noise and interp
  sweeps); statistical density scores add no value over proximity (negative/near-zero partials). The
  *functional* axis carries the incremental predictive signal, the *density* axis is a proximity proxy.
- **Causality — NO (negative result).** Gradient-descending a corrupted activation to improve EITHER a
  density (Mahalanobis) or functional (plateau-KL) realness score makes the true downstream KL WORSE
  than the corrupted start and worse than a matched-distance random move: Mahalanobis-descent collapses
  into the over-central interior (the shell-distance trap), func-descent Goodharts the flatness score
  into a degenerate region. The oracle move toward the true clean activation recovers behavior (KL
  2.20→0.03 at the same budget), so the failure is the score, not the optimizer. These realness scores
  are valid *detectors/predictors* but invalid *causal repair objectives*.
- **Steering preservation — NOT TESTED**, but (4)'s failure implies a steering-repair regularizer built
  on these scalar scores would degrade behavior; a usable objective must respect the data shell.
- **H1 — SUPPORTED.** Real activations carry geometric (shell-distance) and functional (model-sensitivity)
  structure beyond first/second moments. The effect is real but smaller and more orientation-dependent
  than a naive "interpolations fool everything" reading; the honest framing is that statistics need a
  *two-sided* view and a complementary functional probe to capture realness.

## Limitations
- The discrimination functional probe (Phase 3) treats each activation as a single-position input
  (intrinsic model sensitivity, not in-context fidelity). Phase 5 addresses this for PREDICTION by
  corrupting the last position of real prompts and continuing in-context; a fully context-aware
  *discrimination* benchmark (negatives generated in-context) is still future work.
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
`combined_score.py` (combined + interp correction) · `context_validation.py` (Phase-5 in-context
prediction) · `causal_repair.py` (Phase-6 causal repair, negative result). Results in `results/*.csv`,
`results/*.json`.
