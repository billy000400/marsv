# What explains transition-width variation beyond successor JSD?

## Summary

Two token pairs can have almost the same successor Jensen–Shannon divergence (JSD) yet very different transition widths because successor JSD describes only the difference between corpus endpoints. It does not describe how this model travels between the corresponding internal states. In Pythia-1.4B, most of the remaining predictable variation is an additive contribution from the two individual tokens. A smaller, reproducible pair-specific remainder is related to the geometry of the pair's block-0 states.

This answer is supported by seven steps. The width measurement is reproducible after excluding pairs whose endpoints barely differ. Direct matched examples show that equal endpoint statistics can hide different curve shapes. Adding one effect per token raises held-out prediction from $R^2=0.149$ for successor JSD alone to $0.578$. Measuring each token against six separate anchors recovers that effect and predicts pairs of unseen tokens. The residual after additivity is reproducible and endpoint geometry raises held-out $R^2$ to $0.723$. Most decisively, transplanting the block-0 multilayer perceptron (MLP) output from one token to another transfers the donor's width with slope $0.913$. The width difference is therefore mainly encoded in what each token contributes at the first MLP, rather than determined by successor JSD alone.

The conclusion is model-specific. It reproduces across larger Pythia models, but Pythia-160M lacks clear plateau structure and GPT-2 learns a different, less reliable ordering. Detailed robustness checks, secondary probes, failed interventions, and all 42 recorded patterns remain in [RESULTS.md](RESULTS.md) and [TECHNICAL_APPENDIX.md](TECHNICAL_APPENDIX.md).

## Methods

### Data and model

The main experiment uses `pythia-1.4b-deduped` at its final checkpoint. It starts from 1,000 unordered pairs drawn from 123 single-token continuations. Each pair is placed into three sentence frames. At the final token position, the residual state after transformer block 0 is interpolated between the pair's endpoints with 50 norm-rescaled spherical-linear interpolation steps. The model's next-token distribution is recorded at every step. Analyses of the central claim use the 929 pairs that pass the endpoint-movement gate defined below.

The transfer test measures each of the 123 tokens against six anchor tokens absent from the 1,000-pair bank. The forward test uses 40 further tokens absent from that bank and evaluates 718 of their pairs that pass the same gate. The mechanistic transplant uses 12 tokens spanning narrow to wide measurements, six anchors, and one sentence frame, for 132 cross-token donor–recipient interventions plus self-transplant controls.

### Metrics

**Successor JSD.** For tokens $u$ and $v$, $J(u,v)$ is the Jensen–Shannon divergence, in bits, between their empirical immediate-successor distributions in the training corpus. Larger values mean the tokens occur before more different distributions of next tokens. It describes the corpus endpoints, not the path between model states.

**Normalised output distance.** At interpolation position $t$, let $p_t$ be the model's next-token distribution. The distance from the first endpoint is normalised by the total endpoint movement:

```math
d(t)=\frac{\mathrm{JSD}(p_t,p_0)}{\mathrm{JSD}(p_1,p_0)}.
```

Thus $d(0)=0$ and $d(1)=1$ when the endpoints differ. This curve shows whether output changes steadily or remains flat and then switches sharply.

**Transition width.** The width $w$ is the distance in interpolation position between the first crossings of $d=0.1$ and $d=0.9$:

```math
w=t_{0.9}-t_{0.1}.
```

A small $w$ is a sharp transition; a large $w$ is gradual. A proportional response has $w=0.8$.

**Endpoint-movement gate.** Width is interpreted only when the endpoint output distributions differ by at least 0.2 bits of JSD. Otherwise normalising a tiny movement can make an uninformative path look wide.

**Predictive accuracy and reproducibility.** Held-out $R^2$ compares squared prediction error with predicting the test-set mean; higher is better and zero means no improvement over that mean. Spearman's $\rho$ measures rank agreement. Agreement of widths across the three sentence frames gives a reproducibility ceiling of $R^2=0.934$: variation above this ceiling cannot be predicted consistently from these measurements.

**Token-additive model.** The central model gives each token one fitted effect and adds the two effects to the successor-JSD prediction:

```math
\widehat w(u,v)=\mu+f(J(u,v))+a_u+a_v.
```

Here $f$ is the fitted successor-JSD term, while $a_u$ and $a_v$ are token effects. Five-fold cross-validation holds out pairs before scoring, so the comparison tests prediction rather than training fit.

**Anchor width.** A token's directly measured effect, $\widehat w_u$, is its median transition width against six fixed anchors over three frames. The anchors occur in none of the original pairs. This makes the measurement independent of the partners used to fit $a_u$.

### Baselines

The primary baseline predicts width from corpus successor JSD alone. Additional comparisons use model-output endpoint JSD and five pair-level covariates; these test whether an endpoint or simple pair statistic replaces the token effects. The mechanistic transplant has two controls: self-transplanting restores the original width exactly, while donor and recipient correlations distinguish whether the overwritten MLP output or the untouched recipient state determines the result.

## Results

### Observed patterns

This section reports direct evidence only. Interpretation is separated into **Candidate hypotheses** below.

#### Width is meaningful, but successor JSD is incomplete

Across all 1,000 pairs, widths agree across sentence frames with mean correlation $0.825$, producing the $R^2=0.934$ reproducibility ceiling. Seventy-one pairs fail the 0.2-bit endpoint gate. They are not unusually noisy, but they are systematically wide because their endpoints barely move. On the remaining 929 pairs, successor JSD still correlates with width at $\rho=-0.409$, but its held-out $R^2$ is only $0.149$.

Figure 1 shows both the overall relation and why the gate is needed.

![Width against corpus successor JSD and endpoint movement](plots/scatter_and_gate.png)

**Figure 1.** Transition width $w$ against corpus successor JSD (left) and endpoint output movement (right) for 1,000 pairs. Filled circles pass the 0.2-bit gate; open squares fail it. Black linked diamonds are matched pairs of pairs with similar successor JSD and endpoint movement but widths differing by as much as 0.44. Axes are labelled in the plot; marker shape as well as colour identifies gate status.

The mismatch is visible directly, in many cases rather than one anecdote. A matched narrow-versus-wide contrast is a pair of gated pairs selected by three criteria: their successor JSD values differ by at most 0.02 bits, their endpoint output movements differ by at most 0.05 bits, and one pair's width exceeds the other's by at least 0.15 in every one of the three sentence frames separately. Under these criteria 1,529 contrasts exist among the 929 gated pairs, and only 21 of them share a token between the two sides, so the effect is not one unusual token dragging its partners along. The seven largest are listed below; they are the concrete cases the rest of the report tries to explain.

| narrow pair | $w$ | wide pair | $w$ | $\Delta w$ | $J$ (bits) | endpoint movement, narrow / wide (bits) |
|---|---|---|---|---|---|---|
| ` her` / ` when` | 0.34 | ` kind` / ` wrong` | 0.77 | 0.44 | 0.70 | 0.86 / 0.90 |
| ` our` / ` very` | 0.32 | ` never` / ` nothing` | 0.69 | 0.37 | 0.73 | 0.84 / 0.82 |
| ` from` / ` one` | 0.43 | ` kind` / ` wrong` | 0.77 | 0.35 | 0.70 | 0.90 / 0.90 |
| ` our` / ` very` | 0.32 | ` most` / ` now` | 0.67 | 0.35 | 0.72 | 0.84 / 0.83 |
| ` one` / ` when` | 0.38 | ` hard` / ` kind` | 0.72 | 0.34 | 0.74 | 0.91 / 0.96 |
| ` completely` / ` interesting` | 0.44 | ` kind` / ` wrong` | 0.77 | 0.34 | 0.70 | 0.90 / 0.90 |
| ` because` / ` being` | 0.45 | ` kind` / ` wrong` | 0.77 | 0.33 | 0.69 | 0.90 / 0.90 |

**Table 1.** The seven largest matched contrasts. Each row holds corpus successor JSD $J$ and endpoint output movement approximately fixed while width differs by at least 0.33. The narrow side is built mostly from function words (determiners, possessives, prepositions, ` when`), the wide side mostly from evaluative adjectives and adverbs. The same tokens recur on the same side across rows, which is the observation the per-token analysis below follows up.

The strongest contrast compares ` her` / ` when`, with $w=0.34$, against ` kind` / ` wrong`, with $w=0.77$, although both have successor JSD near 0.70 bits and endpoint movements of 0.86 and 0.90 bits. The full curves show a sharp middle switch for the first pair and an almost proportional change for the second.

![Three matched narrow-versus-wide transition curves](plots/contrast_curves.png)

**Figure 2.** Normalised output distance $d(t)$ against interpolation position $t$ for three matched contrasts and three frames per pair. Solid curves are narrow pairs and dashed curves are wide pairs. Horizontal guides at 0.1 and 0.9 define $w$. Line style distinguishes the conditions without relying on red–green colour.

#### Each token contributes much of the missing width

A per-token additive term explains the main gap. The token term alone reaches held-out $R^2=0.365$, already above successor JSD alone at $0.149$ and model-output endpoint JSD at $0.187$. Combining successor JSD with the two token effects reaches $0.578$, or 62% of the reproducible variance. Adding model-output JSD reaches $0.648$; adding block-0 geometry reaches $0.723$.

![Held-out accuracy for models of transition width](plots/cv_r2.png)

**Figure 3.** Five-fold held-out $R^2$ for models of transition width. Hatched bars use pair-level predictors; solid bars include $a_u+a_v$. The vertical dashed line is the $0.934$ reproducibility ceiling. The plot labels every model and value, and hatching makes the grouping colour-independent.

This is not merely a flexible fit to the original pair bank. A token's anchor width predicts its fitted effect at Spearman $\rho=0.70$. At the pair level, the sum of two measured anchor widths reaches held-out $R^2=0.350$, close to the 123-parameter token-only fit at $0.365$. Adding successor JSD raises the measured version to $0.452$. Because the anchors appear in none of the fitted pairs, the same token contribution survives a complete change of partner.

![Anchor measurements recover fitted token effects](plots/transfer.png)

**Figure 4.** Fitted token effect against independently measured anchor width (left), a failed basin-radius comparison (middle), and held-out pair prediction (right). The main evidence is the left and right panels: measured token widths recover the fitted effects and nearly match their predictive accuracy. Shapes and labels distinguish all conditions.

The strongest predictive test uses 40 tokens never seen in the original bank. A slope and intercept learned from the bank, applied to the two anchor measurements, predict all gated pairs among those tokens without refitting. The result is $R^2=0.397$, $\rho=0.66$, and mean absolute error 0.047 over observed widths from 0.34 to 0.78. Median observed width rises from 0.50 to 0.57 to 0.62 across predicted terciles.

![Forward prediction for pairs of unseen tokens](plots/forward_screen.png)

**Figure 5.** Observed width against prediction from two anchor measurements for 718 pairs of 40 unseen tokens (left), and observed width by predicted tercile (right). The diagonal is exact prediction. Distinct box hatching identifies terciles without requiring colour.

Together, Figures 3–5 answer why equal successor JSD does not imply equal width: the tokens themselves bring different, transferable contributions to the curve.

#### A reproducible pair-specific remainder also remains

Additivity is dominant, not complete. When the additive model is fitted separately in independent sentence frames, its pair residuals correlate at $r=0.67$. The leftover is therefore not just measurement noise. Pair-level geometry at block 0 helps: adding endpoint norms, cosine, and distances raises held-out $R^2$ from $0.648$ for token effects plus both JSD measures to $0.723$. This result supports a smaller relational contribution: how the two endpoint states sit relative to one another changes width after their individual effects are accounted for. It does not identify one geometric variable as a causal mechanism.

#### The first MLP output carries the token effect

The causal evidence comes from overwriting the block-0 MLP output of a recipient token with the corresponding vector from a donor token. The recipient's post-transplant width follows the donor at $\rho=0.968$, with regression slope $0.913$, while correlation with the recipient's untouched state is $-0.104$. Between-donor variance is 66 times between-recipient variance. A self-transplant reproduces baseline width to four decimal places.

![Block-0 MLP probes and donor-recipient transplants](plots/mlp_read.png)

**Figure 6.** From left to right: held-out probes from three token representations; the 12-by-12 transplant matrix ordered narrow to wide; transplanted width against donor width; and donor-versus-recipient rank agreement. The heat map and marker shapes show that changing the donor changes width while changing the recipient does little. Axes and conditions are labelled directly.

In this architecture, the block-0 MLP vector is computed from the token embedding before attention has added context; its cosine across the three frames is 1.0000. The transplant therefore explains why the same token effect recurs across partners. The intervention is large—median endpoint output movement is 0.738 bits—so the justified claim is that this vector is sufficient to carry the width-relevant content inside this setup, not that a small or one-dimensional edit can steer width.

#### Generalization is limited to models with the same learned ordering

The per-token ordering reproduces across Pythia-410M, 1B, and 1.4B at raw $\rho=0.88$–$0.90$, close to each measurement's reliability ceiling. Absolute widths still sharpen with model size. Pythia-160M is different: its median edge drift is 0.183, near the 0.2 value of a straight ramp, so it largely lacks the plateau structure whose width is being explained.

GPT-2 small has plateau-shaped curves at the median, but 88.8% of its block-0 curves are non-monotone under the original width definition. Filtering to plateau-shaped curves improves split-half reliability to 0.66, yet its token ordering still disagrees with Pythia at $\rho=-0.19$. Thus the Pythia ordering belongs to tokens as learned in that model family and corpus; strings do not carry a universal width independent of training.

![Curve shape and reliability across GPT-2 and Pythia](plots/edgedrift.png)

**Figure 7.** Edge-drift distributions across GPT-2 sites and Pythia sizes (left), and GPT-2 reliability and agreement with Pythia before and after filtering to plateau-shaped curves (right). A straight response has edge drift 0.2. Line styles and direct labels make model comparisons readable without colour.

## Candidate hypotheses

Everything above is measurement. This section is interpretation: three hypotheses about *why* the observed patterns hold, ranked by how well they fit that evidence, each with the cheapest experiment that would separate it from its main alternative.

**H1 — Width is a per-token trait that a token carries into any pairing.** This fits the evidence best. It predicts the abundance of matched contrasts at fixed successor JSD (Table 1), the accuracy of the additive model (Figure 3), and the part it was actually tested on and passed: a token's width measured against six strangers predicts its fitted effect and the widths of pairs of unseen tokens (Figures 4 and 5). What it does not explain is the reproducible pair-specific remainder, roughly a third of the explainable variance, and the fact that a token's anchor width shifts somewhat with the choice of anchors. The competing reading is that anchor width is just "how unlike a typical token this token is" — a similarity statistic wearing a trait's clothes. Replacing the anchors with a disjoint set from a different word class still recovers the fitted effects at $\rho\approx0.6$, which argues against the pure-similarity reading without settling it. *Discriminating experiment:* measure $\widehat w_u$ using a single anchor averaged over many sentence frames instead of six anchors over three frames. A token property should survive that change; a relational statistic should degrade.

**H2 — The trait reflects how sharply the model reads the token, not the string itself.** Five weak correlations point the same way: tokens that are frequent ($\rho=-0.33$ between a token's fitted effect and corpus log-frequency), that precede low-entropy continuations ($-0.24$), that are surprising in the frame ($+0.26$), and that produce low-entropy model outputs ($-0.30$) tend toward narrower transitions. All of these are correlations across tokens at one checkpoint, and none is strong enough to be the mechanism. The cross-model result is the strongest support: the token ordering survives across Pythia sizes but not into GPT-2 (Figure 7), so the trait belongs to the token *as this model family learned it*. The competing reading is that these five statistics are proxies for a function-word versus content-word split, which Table 1 also shows. *Discriminating experiment:* hold the token fixed and change only the sentence frame so that its surprisal moves by several bits, then re-measure $\widehat w_u$. Manipulating context within a token separates "how the model reads it here" from a property of the string.

**H3 — The pair-specific remainder is endpoint geometry at the interpolation site.** Adding the endpoint norms, their cosine, and their distance at block 0 raises held-out $R^2$ from $0.648$ to $0.723$, and it cuts the across-frame agreement of the leftover residuals from $0.67$ to $0.54$, so some of the reproducible pair-specific part is geometric. This is the weakest of the three: the evidence is predictive, not causal, and the geometric variables might partly re-express successor JSD, although they add accuracy on top of it. *Discriminating experiment:* hold the token pair fixed and change only the path — interpolate through a third state, or use straight-line instead of spherical interpolation — and see whether width tracks the geometry of the new path.

One hypothesis was tested and dropped. The idea that each token owns a region of output insensitivity whose size sets the width predicts that tokens with larger such regions have wider transitions; the measured relation ran in the opposite direction along anchor directions ($\rho=+0.39$) and was absent along random directions ($\rho=-0.02$). The evidence is in [RESULTS.md](RESULTS.md).

## Recommended next experiment

The single most informative next experiment asks **where in the network the width becomes explicit**, using measurements that already exist. The transplant shows that the block-0 MLP output vector carries the trait between tokens, yet no probe reads a number off that vector beyond what the curve's overall shape already gives — transport and decodability came apart. That negative was established on 123 measured tokens. Enlarging the pool to 250 tokens and refitting one representation (the residual state after block 6) reversed it: the width-specific part, meaning width after the component shared with overall curve shape is removed, became readable at $+0.265$ against a permutation bar of $+0.131$. The other five representations were only ever tested at the sample size now known to be too small.

So refit all six — the static embedding row, the block-0 MLP output, the post-block-0 residual state, and the states after blocks 6, 12 and 18 — on the same 250 tokens, with identical splits, permutation nulls and within-sample controls. The curves are already measured and the features are single-token forward passes, so the cost is minutes of GPU time. The outcome is decisive either way. If the static embedding row clears its null, the screen becomes a vocabulary-wide lookup table requiring no forward pass, which is the cheapest useful form this result could take. If readability instead begins at block 6, the screen costs one partial forward pass, and the depth where the profile turns is a fact about where the model computes the crossing width.

## Conclusion

Pairs with similar successor JSD have different transition widths mainly because successor JSD omits a stable contribution from each individual token. In Pythia-1.4B, adding the two token effects raises held-out $R^2$ from 0.149 to 0.578; independent anchor measurements recover those effects and predict pairs of unseen tokens at $R^2=0.397$. A smaller reproducible pair-specific component remains and is partly captured by block-0 endpoint geometry. The block-0 MLP transplant supplies the strongest mechanistic evidence: replacing one token's MLP output with another's transfers 91% of the donor-width relation and removes dependence on the recipient ordering.

The practical lesson is to screen tokens, not every token pair: measure a token once against fixed anchors, then combine two measurements with successor JSD. The scientific limit is equally important. This is not a universal property of token strings, and the large transplant does not isolate a low-dimensional causal feature. GPT-2 and very small Pythia models require their own reliability and plateau-shape checks before this explanation can be applied.
