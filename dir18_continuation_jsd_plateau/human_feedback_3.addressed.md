I reread the latest report. The central association appears robust, but several definitions, captions, and claims remain ambiguous or stronger than the measurements support. Please address the following points:

* Define the corpus metric explicitly as **context-averaged immediate-next-token JSD**, not general “continuation divergence”:

  [
  \widehat J_{\mathrm{holdout}}(u,v)
  ==================================

  JSD!\left(
  \widehat P_{\mathrm{holdout}}(y_{i+1}\mid y_i=u),
  \widehat P_{\mathrm{holdout}}(y_{i+1}\mid y_i=v)
  \right).
  ]

  It uses only the token immediately following each endpoint, not a multi-token continuation sequence.

* Rename corpus splits A/B to `selection`/`holdout`, and use (u,v) for the two endpoints. The current notation conflicts with endpoint labels such as (x_A,x_B).

* Replace ambiguous labels such as `JSD (split B, bits)` with:

  > **Held-out corpus next-token JSD, (\widehat J_{\mathrm{holdout}}(u,v)) [bits]**

* Explain why only the holdout JSD is used in the primary analysis: the selection split defines the strata and pair bank, while the holdout split supplies the predictor used for testing. Do not imply that the holdout split is completely untouched, because its counts contribute to eligibility filtering and A+B counts contribute to frequency matching.

* Add selection-split JSD as a sensitivity check. The results are nearly identical:

  [
  \rho(J_{\mathrm{selection}},w)=-0.5260,\qquad
  \rho(J_{\mathrm{holdout}},w)=-0.5247,
  ]

  with (\rho(J_{\mathrm{selection}},J_{\mathrm{holdout}})=0.99972).

* Clarify Figure 4 completely. Each dot is one endpoint pair. Its (w) value is the median across three carrier contexts, and each context-specific (w) is calculated from 50 interpolation positions. The analysis therefore contains only 60 pair-level observations; the three panels reuse the same pair identities.

* Explain why only 60 pairs were assayed: the strict top-256 filter leaves 123 eligible endpoints, and prohibiting endpoint reuse permits at most 61 disjoint pairs. State that this design avoids direct dependence from endpoint reuse; do not claim that the pairs are fully statistically independent.

* In Figure 4, clarify that colors represent selection-split JSD strata, whereas the crosses and dashed line are medians after re-binning by holdout JSD. These are five non-overlapping binned medians, not additional observations or a running median.

* Add a larger secondary analysis using approximately 1,000 pairs. Keep the current 60 endpoint-disjoint pairs as the primary confirmatory bank. Select the additional pairs without examining (w), stratify them across selection-split JSD, and prevent a few endpoints from dominating the sample.

* Because 1,000 pairs necessarily reuse endpoints, do not treat them as 1,000 independent observations or report a naïve Spearman (p)-value. Use dyadic/endpoint-clustered inference, an endpoint-level bootstrap or permutation procedure, or a cross-classified model with random effects for both endpoints. Run the final 1.4B checkpoint first; additional checkpoints can be optional.

* Replace “complete words” with **single-token word-start endpoints**. The current filter admits fragments such as `un`, so it does not establish that every endpoint is a complete word.

* Replace “every prompt is in-distribution” with **the endpoints are model-plausible under the three carrier contexts**. Token frequency and endpoint probability do not prove that the exact prompts are in-distribution.

* Clarify what the plateau measurement establishes. A flat

  [
  d(t)=\frac{|z(t)-z_u|}
  {|z(t)-z_u|+|z(t)-z_v|}
  ]

  means that the relative endpoint-distance coordinate changes little. It does not prove that the complete logit vector or output distribution “stays put.” Use “relative-logit-coordinate plateau” unless adjacent-logit movement is measured directly.

* Do not call (w) generic “transition strength.” It is specifically the 10%–90% transition width, with smaller values indicating sharper transitions. It does not independently isolate plateau flatness, especially because (w) and edge drift are highly correlated.

* Narrow “predicted from the training corpus alone” to:

  > “The JSD predictor itself is computed from corpus statistics.”

  Endpoint filtering and matching also use trained-model probabilities and surprisal.

* Remove the Figure 1 claim that “93% is real signal.” The ratio of same-token noise JSD to between-token JSD is not a valid additive decomposition of signal and noise.

* Replace “the bins are indistinguishable” with “we detected no significant imbalance.” A nonsignificant Kruskal–Wallis result does not prove equality.

* Describe step-0 (w) as having a restricted range or a near-ceiling effect, not a floor effect.

* Describe the 410M result as a **cross-scale robustness check**, not an independent replication, because it uses the same corpus estimates and pair bank.

* Define (JSD_{\mathrm{out}}) more precisely. State that it is aggregated across the three carrier contexts and that the output distribution is restricted to the target IDs observed in the sampled corpus, if that remains the implementation.

* Replace “almost nothing survives adjustment” with:

  > “The association is attenuated after adjustment, and the fully adjusted estimate is not statistically significant.”

  Controlling only for output JSD still gives (\rho=-0.277,\ p=0.032); nonsignificance appears after adding all planned covariates.

* Do not claim “full strength by step 1,000, then no further change.” Step 1,000 is the earliest measured checkpoint. The supported statement is:

  > “The association is already comparable to later checkpoints at the earliest measured checkpoint.”

* Correct Figure 2’s scope. Clearly distinguish the curves displayed in the figure from the larger set used for QC across checkpoints; do not describe aggregated or selected curves as “every raw curve.”

* State that the block scan uses only 10 extreme pairs and one carrier context. Its result is consistent with a role for downstream computation, but it does not establish that downstream blocks are generally required for the effect.

* Narrow the prespecification claim to:

  > “The top-256 selection rules were prespecified, and the exact-pair curves were not used during pair selection.”

  Avoid saying that everything was frozen before any related curve had been examined.

* Use a more precise headline:

  > “Within a stratified bank of high-frequency, model-plausible single-token endpoint pairs, held-out corpus immediate-next-token JSD is associated with narrower median 10%–90% relative-logit transitions.”

The 1,000-pair analysis would materially strengthen the generality claim and reveal whether the relationship is nonlinear or driven by the particular 60-pair matching bank. However, it should be presented as an endpoint-dependent robustness analysis, not as 1,000 independent confirmations.
