# JOURNAL — Direction: TODO — describe this direction

Append-only. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## Iter 1 (2026-06-23) — Audit + MVP benchmark + Phase-2 baselines
**Did:** Audited env & nearby assets. Env is stripped: numpy 2.3.3 only — NO
transformers/sklearn/scipy/datasets (shared base env was reset, per dir9 notes; do NOT
change torch/numpy). GPU is an A10 (sm_86), CUDA works. Found CACHED real GPT-2 resid acts
in `../dir3_manifold/data/acts_layer{3,6,9}.npy` [200000,768] f16 (resid_post, all positions
pooled, FineWeb) — reused them as real positives so no model load needed for the MVP.
Wrote `experiments/mvp_benchmark.py` (pure numpy: AUROC/AUPRC self-implemented): contiguous
TRAIN/EVAL split (doc-leakage control), 4 negative families (iso_gauss, cov_gauss [chol of
shrinkage cov], shuffle_coord, norm_pert — last two preserve norm exactly), 6 baselines
(norm, mean_l2, mahalanobis, pca_recon, coord_quantile, knn_distance). Ran in 32s. Wrote
results/audit.json, dataset_summary.json, negative_family_summary.csv, baseline_metrics.csv.
**Learned:** kNN local density is the best baseline (macro AUROC 0.913) and the ONLY one that
catches cov-matched Gaussians (0.976) — real acts have local-density structure beyond global
2nd moments (Mahalanobis=0.53 on cov_gauss by construction). Norm is a partial shortcut, fully
defeated (0.50) by the two norm-preserving families. No single baseline dominates; norm_pert is
the hardest family (max 0.86, Mahalanobis). Gate 2: baselines do NOT solve everything → richer
scores justified, bar to beat = kNN macro 0.913 (esp. norm_pert headroom).
**Next:** Train a learned detector (logistic reg / small MLP, numpy) with LEAVE-ONE-FAMILY-OUT
splits to test generalization vs shortcut memorization (Gate 4); use standardized raw acts +
optionally the 6 baseline scores as features; compare held-out-family AUROC against kNN. Then
add layers 3 & 9. (Geometry/Jacobian features need the model → reinstall transformers --no-deps
only when reaching Phase 3.)
On track? yes — S4 done (baselines), at ~5% of full ladder; blocker: none.

## Iter 2 (2026-06-23) — Learned detectors, leave-one-family-out (Gate 4)
**Did:** Wrote `experiments/train_detectors.py` (pure numpy logreg + 1-hidden-layer MLP). Leakage
control: positives and the reals used to derive negatives are DISJOINT row blocks; detector sees
only standardized raw acts. LOFO protocol: train on 3 families, test on held-out 4th. Ran 31s.
Wrote results/detector_metrics.csv, detector_summary.json.
**Learned:** LOFO macro logreg 0.676, mlp 0.735 — BOTH below the unsupervised kNN baseline 0.913.
Detectors collapse to ~chance on unseen cov_gauss (0.48/0.51) and iso_gauss (logreg 0.49); only
shuffle_coord transfers (0.999, broken covariance is obvious even unseen). Held-IN all-4 is higher
(mlp 0.85) than LOFO — the classic memorization gap. So a discriminative real-vs-fake classifier
learns generator shortcuts and does NOT generalize. The realness property that DOES transfer is
local density (kNN). Reframes D6: realness ≈ high local density on the real-activation manifold
(one-class/density), not a learned boundary. Gate 4 fails for learned discriminative detectors.
**Next:** Pursue density/manifold/reconstruction one-class scores and test their held-out-family
transfer. Concretely: (a) extend the kNN/density + PCA-recon baselines to layers 3 & 9 to check the
local-density story holds across depth; (b) add a harder norm-matched negative family (e.g. real+
along-tangent vs orthogonal perturbations, interpolations between reals) to stress kNN — currently
norm_pert is the only family with headroom (0.68). Defer Jacobian/geometry (needs model) until a
density-vs-functional comparison is set up.
On track? yes — S6 done (LOFO detectors; informative negative result), ~12% of ladder; blocker: none.

## Iter 3 (2026-06-23) — Baselines across layers {3,6,9} + harder negatives
**Did:** Wrote `experiments/baselines_layers.py`. Extended the 6 baselines to layers 3/6/9 and
added 3 harder NORM-MATCHED negatives: `interp` (convex combo of two distinct reals, renormed),
`tangent_pert` (perturb in top-50 PCA subspace), `orth_pert` (perturb orthogonal to it). Ran 151s
(pure numpy). Wrote results/baseline_layers_metrics.csv. (Fixed an `np.eye(D, np.float32)` arg bug.)
**Learned:** BIG: `interp` defeats EVERY statistical baseline at every depth — AUROC ≈ 0.44–0.54
(chance), kNN included. Averages of real activations are statistically indistinguishable from real.
This is the strongest evidence for H1 so far: realness has structure NOT captured by moments or
local density. Also confirmed kNN(density) ⊕ Mahalanobis(covariance) are complementary across all
layers — neither alone is complete (kNN owns cov_gauss, Mahalanobis owns orth/norm/shuffle). Story
is depth-stable. tangent_pert also hard (0.66–0.75); orth_pert caught by Mahalanobis (0.88–0.92).
**Next:** Test a FUNCTIONAL probe where statistics fail (interp/tangent_pert). Need the model:
`pip install --no-deps transformers tokenizers safetensors regex` (NEVER touch torch/numpy/cuda;
HF cache has gpt2 already). Then `experiments/functional_features.py`: treat each activation as
resid_post at the last position, run GPT-2 blocks L+1..11 + lnf + unembed -> logits; compute
entropy, MSP/max-prob, and plateau-KL (output KL under a small activation perturbation, Hutchinson-
style). Compare real vs interp vs tangent_pert AUROC. Reuse dir9 plateau_score.py patterns. If
functional features catch interp where stats can't -> H1 confirmed via functional geometry.
On track? yes — Phase 2 fully done across depth incl. the decisive `interp` hole, ~20% of ladder;
blocker: env needs transformers reinstall (--no-deps) for the functional/Jacobian phase.

## Iter 4 (2026-06-23) — Functional probe where statistics fail
**Did:** Reinstalled transformers 5.12.1 via `pip install --no-deps` (already present; verified
torch 2.9.0+cu130 / numpy 2.3.3 UNCHANGED before+after). Verified my "continue-forward-from-
activation" reconstructs true GPT-2 logits exactly (max abs diff 0.0) with full context. Wrote
`experiments/functional_features.py` (GPU A10, VRAM 0.45): treat each activation as single-position
resid_post@L6, run blocks h[7:]+ln_f+lm_head -> logits; features entropy/msp/plateau_kl/logit_max.
Ran 2s for N=2000 x 5 families. Wrote results/functional_metrics.csv, functional_summary.json.
**Learned:** On `interp` (≈0.50 for ALL statistical baselines) functional entropy/plateau_kl reach
~0.61 — weak but the ONLY signal beating chance. Direct evidence of a FUNCTIONAL component of realness
not in any statistic. But functional features are complementary, not dominant: ≈kNN on cov_gauss
(0.95) yet beaten by Mahalanobis/kNN on norm_pert/tangent_pert. `interp` stays the hard core: best
detector across ALL methods ~0.61. Wrote the RESULTS.md Headline (full verdict).
**Next:** Optional strengthening before finalize: (a) quick incremental-value check — does a logistic
model over [kNN, Mahalanobis, entropy, plateau_kl] beat the best single score per family, esp. lifting
interp above 0.61? (Phase-3 gate / added value). (b) sensitivity of interp-functional AUROC to plateau
eps and to using in-context continuation. If time-limited, go straight to REPORT.md. Core scientific
verdict is already established and written.
On track? yes — Phases 2-5 covered with a clear verdict incl. the decisive interp/functional finding,
~45% of ladder (Phases 6-7 causal/steering are optional extensions); blocker: none.

## Iter 5 (2026-06-23) — Combined score + IMPORTANT interp correction
**Did:** Wrote `experiments/combined_score.py` (mahalanobis,knn,entropy,plateau_kl on shared L6 eval;
COMBINED = logistic LOFO). Ran 9s. Wrote results/combined_metrics.csv, combined_summary.json.
**Learned (corrects iter3):** `interp` is NOT undetectable — my iter3 "≈chance for all baselines"
used a fixed ONE-SIDED orientation (anomaly=far). Verified directly: interp has LOWER Mahalanobis than
real (658 vs 803, norm exactly matched), directed AUROC 0.317 → TWO-SIDED Mahalanobis catches interp at
~0.68. interp is anomalous by being too CENTRAL (averaging lands in the over-typical interior; real
acts occupy a characteristic-distance shell, high-dim annulus intuition). kNN still 0.50 (interp is
near real neighbors) → the signal is GLOBAL not local. Functional plateau_kl independently ~0.61.
Combined LOFO catches cov_gauss (0.999) but FAILS interp (0.54) because interp's anomaly direction is
OPPOSITE to standard corruptions a combined detector trains on. Rewrote RESULTS Headline with the
corrected verdict + 5-claim-ladder assessment (H1 supported but gap smaller/orientation-dependent).
**Next:** Fallback deliverable is fully met (L6+3,9 benchmark, 7 families, 6 stat + 4 functional
scores, LOFO, functional vs statistical comparison, depth check). Write REPORT.md with verdict,
limitations (single-position functional probe, pooled positions incl. pos-0 sink tokens, GPT-2 small
only, Phases 6-7 untested), and Direction-1 implications. Optionally attempt a small Phase-6 causal
sanity check if budget remains; else finalize + STOP.
On track? yes — fallback criterion met; core multi-axis verdict established and corrected; ~55% of
ladder, Phases 6-7 optional; blocker: none.

## Iter 6 (2026-06-23) — Context-aware downstream prediction (Phase 5, claim 3) + REPORT.md
**Did:** Wrote REPORT.md (full verdict). Then wrote `experiments/context_validation.py`: corrupt
layer-6 resid_post at the LAST position of 400 real FineWeb prompts (dir9 fineweb_sample.txt),
continue the IN-CONTEXT forward, measure true KL(clean‖corrupt). Two norm-matched severity sweeps
(noise, interp). Spearman + PARTIAL Spearman controlling dist_to_orig. Ran 21s GPU. Wrote
results/functional_prediction.csv, functional_prediction_summary.json.
**Learned:** KL rises monotonically with severity (probe valid). dist_to_orig dominates raw prediction
(ρ 0.87/0.92). KEY: FUNCTIONAL scores predict downstream KL BEYOND distance — plateau_kl partial +0.57
(noise)/+0.20 (interp), entropy +0.49/+0.17, positive in all sweeps. Statistical density scores do NOT
add value over proximity (maha/knn partials negative on noise sweep, collinear with distance). So the
property best for DISCRIMINATING interp (two-sided Mahalanobis, Phase 3) differs from the one best for
PREDICTING downstream harm (functional plateau-KL/entropy). Claim 3 = PARTIAL/YES for the functional
axis. Updated RESULTS.md + REPORT.md verdicts and limitations accordingly.
**Next:** Deliverable now covers claims 1-3 with evidence; REPORT.md complete. Optional remaining:
Phase 6 causal repair (now tractable on the Phase-5 inject-and-continue machinery + differentiable
functional score) — gradient-descend a corrupted activation to lower the functional anomaly with a
move-distance penalty, test whether true downstream KL drops MORE than matched-distance shrinkage/random
controls (claim 4). If pursued, guard against reward-hacking by reporting an external metric
(next-token loss vs clean) not in the objective. Else finalize + STOP.
On track? yes — claims 1-3 evidenced, REPORT.md written; ~70% of ladder, Phases 6-7 optional high-
compute extensions; blocker: none.

## Iter 7 (2026-06-23) — Causal repair (Phase 6, claim 4) — clean NEGATIVE result + FINALIZE
**Did:** Wrote `experiments/causal_repair.py`: corrupt last-pos resid_post (noise s=1, norm-matched)
of 300 prompts; repair by Adam gradient descent on a realness score (Mahalanobis distance; plateau-KL
functional anomaly); compare to matched-L2-distance controls (shrink_mean, shrink_clean oracle,
random). External objective-free metrics: KL(clean‖x), NLL of clean argmax. Ran 5s GPU. Wrote
results/repair_metrics.csv, repair_summary.json.
**Learned:** Claim 4 FAILS decisively. maha_descent KL 2.20→6.87 (moves into over-central interior,
dist_to_mean 82→19 = shell-distance trap); func_descent KL→14.58 (Goodharts flatness into a degenerate
region). BOTH worse than corrupted start AND worse than a random move of equal size (3.61). Only the
oracle (move toward true clean act) recovers (KL 0.03) — so the failure is the SCORE, not the optimizer.
These realness scores are valid detectors/predictors but invalid causal objectives; naive descent
reward-hacks them. Strong Direction-1 caution: don't regularize steering toward low Mahalanobis /
low functional-sensitivity. Updated RESULTS.md + REPORT.md (claim-4 verdict, key result #7, D1
implications reconciled).
**Next:** FINALIZE. Deliverable now spans the full ladder: claims 1 (yes, multi-axis), 2 (partial),
3 (partial/functional), 4 (NO/reward-hacks), 5 (untested, blocked by 4). REPORT.md + RESULTS.md
complete and self-consistent. Creating STOP. Future work if resumed: a manifold/denoising-prior
causal objective; in-context discrimination benchmark; cross-model transfer; token-position
stratification (pos-0 sink confound).
On track? yes — FULL 5-claim ladder addressed with an honest mixed/negative verdict; deliverable
complete; creating STOP. Stage S10 done.

## Iter 8 (2026-06-23) — External review corrections: genuinely IN-CONTEXT Phases 5 & 6
**Did:** A codex review (codex_review_20260623T024606Z.md) appeared (STOP had been removed). Triaged
its 5 findings. The decisive one (#1): the Phase-5/6 "in-context" continuation actually fed a SINGLE
position `[B,1,768]` through the late GPT-2 blocks, so later-layer attention never saw the prompt —
the "in-context KL" was single-position late-block continuation. Re-implemented both phases genuinely
in-context with a FORWARD HOOK that overwrites ONLY the last-token resid_post@L6 during a FULL model
forward (`experiments/context_validation_v2.py`, `causal_repair_v2.py`). Gradients for func_descent
flow through the late blocks to the injected residual. Also fixed finding #2 (func_descent's random
control was matched to the maha move budget, not its own) by adding a func-budget-matched random
control. Corrected wording for #3 (combined LOFO uses held-out-label orientation = diagnostic), #4
(single-position probe phrasing), #5 (split is contiguous+gap leakage-reduction, not verified doc-level).
Ran both (77s + 109s GPU). Wrote functional_prediction_v2.csv/.json, repair_metrics_v2.csv/.json.
**Learned:** SANITY severity-0 → mean KL = 0.0000 EXACTLY → hook injects the true clean residual, the
in-context machinery is correct. BOTH conclusions SURVIVE the correction. Claim 3: functional plateau-KL/
entropy keep POSITIVE partial-ρ beyond distance in both sweeps (noise +0.51/+0.35, interp +0.25/+0.19);
raw plateau-KL Spearman even RISES in-context (0.55→0.79 noise, 0.13→0.52 interp) — context makes the
functional signal MORE predictive. Density scores (maha/knn) keep negative partials on noise = proximity
proxies. Claim 4: still FAILS, now rigorously — maha_descent in-context KL 0.78→3.33 vs its matched
random 1.99; func_descent →8.82 vs its OWN func-matched random 2.20; oracle →0.009. Both descents worse
than their EXACTLY move-matched random controls → reward-hacking is real, not a move-budget artifact.
Note in-context corrupted-start KL (0.78) ≪ buggy single-position (2.20): context anchors the prediction,
so the old version overstated corruption impact — but the verdict is identical. Updated REPORT.md +
RESULTS.md (new in-context tables, Correction section, all 5 findings addressed) + PLAN.md.
**Next:** Project re-finalized; the external review's substantive concerns are resolved and the core
verdict is confirmed (not overturned) under the corrected in-context method. Re-creating STOP. Genuine
remaining future work (unchanged): a manifold/denoising-prior causal objective (Phase 6 alternative),
an in-context DISCRIMINATION benchmark (Phase 3 negatives generated in-context), cross-model transfer,
token-position stratification (pos-0 sink), and bootstrap CIs.
On track? yes — review corrections applied, claims 3 & 4 re-verified in-context (conclusions unchanged),
deliverable complete & self-consistent; re-creating STOP. Stage S10 done (re-finalized).

## Iter 9 (2026-06-30) — Figures for every result + CHANGELOG backfill (deliverable-hygiene finalize)
**Did:** Re-audited on resume (STOP had been removed again). Science was complete after iter8, but two
operator-rule (CLAUDE.md) gaps remained: `plots/` was EMPTY (rule: "visualize every reported result")
and `CHANGELOG.md` was a 4-line stub (rule 7: all history lives there). Env has NO matplotlib (only
numpy + PIL 11.3; transformers/scipy/pandas also absent on this resume). Rather than risk a --no-deps
matplotlib install (compiled deps: kiwisolver/contourpy), wrote `experiments/make_plots.py` rendering
figures with pure PIL (grouped_bars supporting negative values + dashed reference lines, and a
diverging heatmap; Pillow 11.3 `load_default(size)` for scalable text). Generated 8 PNGs from the
cached result CSVs (no GPU/model needed, CPU-only, <1s): fig1 Phase-2 baselines @L6, fig2 Phase-4 LOFO
detectors vs kNN, fig3 Phase-2b L6 heatmap (7 fam × 6 base), fig4 interp one-sided AUROC across layers,
fig5 functional probe, fig6 combined-score capstone, fig7 Phase-5 partial-ρ, fig8 Phase-6 causal-repair
KL. Verified figures render correctly (read 4 of them back as images; fixed em-dash missing-glyph boxes
→ '-', and a y-label/tick overlap). Embedded fig1–8 in RESULTS.md beside each table; added a Figures
section to REPORT.md. Backfilled CHANGELOG.md with the iter1–9 deliverable history (old→new numbers).
**Learned:** No NEW science — all numbers verified to match the curated RESULTS.md tables exactly
(read every source CSV: baseline_metrics, baseline_layers, detector_metrics, functional_metrics,
combined_metrics, functional_prediction_v2, repair_metrics_v2). The deliverable is now figure-complete
and the change history is recorded as CLAUDE.md requires. PIL is a viable matplotlib substitute for
bar/heatmap result figures in this stripped env.
**Next:** Project finalized and now fully rule-compliant. Genuine remaining future work (unchanged):
manifold/denoising-prior causal objective (Phase-6 alternative), in-context DISCRIMINATION benchmark
(Phase-3 negatives generated in-context), cross-model transfer, token-position stratification (pos-0
sink confound), bootstrap CIs. Re-creating STOP.
On track? yes — deliverable complete, figure-complete, and history recorded; S10 done (re-finalized
with figures+CHANGELOG); blocker: none.
