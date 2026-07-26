# RESULTS — Random search for LLM activation sub-plateaus (`A | C | B`)

> CURRENT-BEST ONLY. One row per experiment. No history (see CHANGELOG.md).
> Full method, definitions and equations: **REPORT.md**.

**Question.** Interpolate between the final-position residual-stream activations of two *random*
held-out natural contexts at an early GPT-2 Large block and patch the interpolant back in. How often
does the model's next-token prediction pass through a **persistent third token** `C` on the way from
`A` to `B` — the language-model analogue of the `A → C → B` paths seen earlier in MNIST?

**Setup in one line.** GPT-2 Large (774M, 36 blocks), 32-token WikiText-103 validation windows,
`resid_post` at blocks 0/2/4/6, `slerp_rescale` over 50 alphas, 1,000 random primary pairs × 4 blocks
× 2 conditioning contexts = 8,000 paths; frozen rule (C top-1 for ≥3 consecutive alphas, beating both
endpoint tokens at every point, with a real distribution change at both boundaries).

## Metrics

### Prevalence of `A | C | B` (primary screen, frozen rule)

| quantity | value | 95% CI (Wilson) |
|---|---|---|
| paths screened / eligible (`A≠B`) | 8,000 / 7,611 (95.1%) | — |
| **candidate paths** | **1,290 → 16.9% of eligible paths** | [16.1%, 17.8%] |
| pairs with ≥1 candidate path (of 8) | 610 / 991 → 61.6% | [58.5%, 64.5%] |
| **clean** `A,C,B` (exactly 3 top-1 runs) | 283 → 21.9% of candidates, 3.7% of eligible paths | — |
| complex with a persistent C region | 1,007 → 78.1% of candidates | — |
| endpoint fidelity, own-context end | max&#124;Δlogit&#124; = 1.5e-05; top-1 reproduced on 100% of paths | — |
| determinism (20 paths re-run, different batch layout) | 0 / 1,000 top-1 mismatches; max Δp = 2.7e-06 | — |
| batching invariance (32 contexts, singly vs batched) | 0 top-1 changes; max&#124;Δlogit&#124; = 4.2e-05; no padding used | — |

### Confirmation on the disjoint validation bank (rule applied unchanged, nothing retuned)

| bank | eligible paths | candidates | rate | 95% CI |
|---|---|---|---|---|
| primary (1,000 pairs) | 7,611 | 1,290 | 16.9% | [16.1%, 17.8%] |
| **validation (300 disjoint pairs)** | 2,261 | 401 | **17.7%** | [16.2%, 19.4%] |

### Controls

| control | eligible paths | rate | 95% CI | reading |
|---|---|---|---|---|
| primary screen | 7,611 | 16.9% | [16.1%, 17.8%] | reference |
| self-pairs (context with itself) | **0** | 0% | — | detector cannot fire on a constant path |
| same-prediction pairs (different contexts, same unpatched top-1) | 1,284 | 11.1% | [9.5%, 12.9%] | two thirds of the rate survives *without* any A/B disagreement |
| linear interpolation instead of `slerp_rescale` | 1,904 | 16.1% | [14.5%, 17.8%] | not a spherical-geometry artefact |
| foreign endpoint reproduces its home prediction | 1,409 of 8,000 (17.6%) | 14.0% | [12.3%, 15.9%] | rate barely moves on the transfer-consistent subset |

### Is the third region a *confident* state? (1,290 candidates)

| quantity | C-region centre | path endpoints (mean of α=0,1) |
|---|---|---|
| top-1 probability | 0.227 ± 0.165 | 0.323 ± 0.182 |
| predictive entropy (bits) | 6.97 ± 1.99 | 5.70 ± 1.82 |
| candidates where C is sharper than the endpoints | 26.8% | — |
| minimum dominance margin > 0.05 / > 0.2 | 39.9% / 3.6% | — |
| C is one of the 10 most common endpoint tokens (' the', '.', …) | 32.3% | — |

### Do C-region activations look natural? (2,000 held-out reference contexts, exact cosine search)

| query type | median cosine distance to nearest natural activation | fraction of top-10 neighbours predicting the query's own top-1 token |
|---|---|---|
| natural context (control) | **0.086** [0.061, 0.131] | **14.1%** [11.6%, 16.8%] |
| A-region point | 0.140 [0.120, 0.160] | 8.1% [7.1%, 9.1%] |
| B-region point | 0.153 [0.133, 0.169] | 8.1% [7.1%, 9.1%] |
| **C-region point** | **0.160** [0.154, 0.166] | **4.5%** [3.8%, 5.3%] |

### Continuations from the C region (6 frozen inspected candidates: 3 top-scoring + 3 random)

| quantity | value |
|---|---|
| greedy C-region continuations that are fluent, context-appropriate English | 6 / 6 |
| identical greedy tokens across the first/middle/last alpha of the C run (of 20) | 20, 20, 8, 1, 1, 1 |
| C activation inserted into the *other* endpoint's context | reverts to that context's own unpatched continuation in 6 / 6 |

## Figures

The screen's headline number and where it comes from — the rate rises monotonically with block depth
and is identical under either conditioning context, while the self-pair control is exactly zero:

![A|C|B rate per eligible path by interpolation block (left; blocks 0/2/4/6 of GPT-2 Large) and by conditioning context and control condition (right). Bars are rates per eligible path, error bars are 95% Wilson intervals.](plots/candidate_prevalence_by_layer.png)

Candidates are extremely heterogeneous, so a single example would misrepresent them. The top-ranked
paths show wide, confident third plateaus with two clearly separated distribution changes; randomly
drawn qualifying paths show narrow, low-probability blips:

![Next-token probability of the A, C and B tokens versus the interpolation coefficient alpha for the three top-scoring candidates (upper row) and three randomly drawn candidates (lower row). Grey band = the detected C run; dotted grey line on the right axis = Jensen-Shannon divergence between neighbouring alphas, in bits.](plots/top_candidate_probability_paths.png)

That heterogeneity is the rule, not the exception — most C segments are only 3–5 of the 50 grid
points wide and win by a small margin, and both transitions sit in the middle of the path:

![Distributions over the 1,290 candidate paths: C-segment width as a fraction of the alpha grid (left), minimum dominance margin of C over both endpoint tokens (middle), and entry versus exit alpha of the C run (right; dashed line is the diagonal).](plots/segment_width_margin_distribution.png)

If the third region were a genuine extra state we would expect it to be at least as confident as the
endpoints. It is not — it is flatter and less certain:

![Top-1 probability (left) and predictive entropy in bits (right) at the centre of the C region versus the mean of the two path endpoints, over the 1,290 candidate paths. Solid = C-region centre, dashed = endpoints.](plots/c_region_confidence.png)

And the C tokens themselves are not distinctive concepts: they are drawn from the same generic
high-frequency pool as the endpoint predictions:

![Left: the 15 most common intermediate (C) tokens across candidate paths. Right: the 15 most common endpoint (A) tokens across eligible paths. Bars count paths.](plots/intermediate_token_census.png)

Because the headline rate depends on two frozen thresholds, we show what happens when they move —
the effect degrades smoothly and never collapses to zero:

![A|C|B rate per eligible path versus the persistence threshold (2, 3 or 5 consecutive alpha points), for three minimum-dominance-margin floors. The dotted vertical line marks the frozen default (persistence 3, margin > 0).](plots/threshold_sensitivity.png)

Finally, two probes of whether a C region behaves like a real model state. Its points sit *further*
from real activations than the endpoint-region points do, and their natural neighbours rarely predict
C:

![Left: distribution of cosine distance to the nearest of 2,000 held-out natural activations, for A-region, C-region and B-region interpolation points and for natural contexts themselves. Right: fraction of the 10 nearest natural neighbours whose own unpatched top-1 next token equals the query's top-1 token, with 95% bootstrap intervals.](plots/natural_neighbor_comparison.png)

Yet the text the C region produces is fluent, and in a third of inspected cases it is reproducible
across the whole C run rather than at a single grid point:

![Number of leading greedy-decoded tokens (out of 20) that are identical across continuations generated at the first, middle and last alpha of the C run, for each of the six inspected candidates. Labels give the C token and the interpolation block; the dotted line at 1 is the trivial floor (the first token is C by construction).](plots/continuation_stability.png)

## Headline

**The `A → C → B` phenomenon is real and common in GPT-2 Large — about 1 in 6 random interpolation
paths (16.9%, CI [16.1, 17.8]) shows a third token holding top-1 for ≥3 consecutive alphas while
beating both endpoints, reproduced at 17.7% on a disjoint bank — but the typical third region is a
weak, high-entropy band whose token is a generic frequency default, sits further off the natural
activation manifold than the endpoints, and appears almost as often (11.1%) when the two contexts do
not even disagree.** Only a small minority looks like a genuine third *state*: 3.7% of eligible paths
are clean `A,C,B`, 3.6% of candidates hold a margin above 0.2, and 2 of 6 inspected candidates keep
an identical 20-token continuation across their whole C run. The verdict is therefore **"robust third
output region, mostly fragile"** — not a null result, and not the crisp third-class plateau seen in
MNIST.
