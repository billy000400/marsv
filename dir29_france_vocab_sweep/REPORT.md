# REPORT — Direction: TODO — describe this direction

> Final, presentable, current-best only (no history — see CHANGELOG.md). Read before rewriting.

## Summary
TODO — 2-4 sentences: the question, the headline result, the verdict.

## Methods
### Data & Model
TODO — dataset, model (e.g. GPT-2 small, 124M), exact layer(s)/hook point, sample sizes.

### Metrics
TODO — define EACH metric with a rendered equation. Example:
$$\mathrm{AUROC} = \Pr\big(s(x^{+}) > s(x^{-})\big)$$
State exactly what `s(x)` scores and which direction means "more anomalous".

### Baselines
TODO — name and define EACH baseline. Example (Mahalanobis distance):
$$d_M(x) = \sqrt{(x-\mu)^{\top}\,\Sigma^{-1}\,(x-\mu)}$$

## Results
TODO — current-best numbers only (one row per experiment), referencing figures in plots/.

## Conclusion
TODO — what the result implies; limitations.
