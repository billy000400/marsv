# PLAN — Delayed-Successor Plateau

> Working folder: `delayed_successor_plateau`. Rewrite “Current status” and “Next step” after every iteration. Read `../BUDGET.md` and `../CLAUDE.md` every iteration.

## Research question

Can an (A\rightarrow B) interpolation produce a plateau in a future prediction, even though both endpoints currently predict the same next token?

## Fixed example

| Role                            | Text/token                                       |
| ------------------------------- | ------------------------------------------------ |
| Prefix (P)                      | `The capital of France is Paris. The capital of` |
| (A)                             | ` Japan`                                         |
| (B)                             | ` Germany`                                       |
| Shared successor (S)            | ` is`                                            |
| Expected prediction after (A+S) | ` Tokyo`                                         |
| Expected prediction after (B+S) | ` Berlin`                                        |

The two complete endpoint inputs are:

```text
The capital of France is Paris. The capital of Japan is
The capital of France is Paris. The capital of Germany is
```

Preliminary GPT-2 Large results to reproduce:

* After ` Japan` and ` Germany`, both predict ` is` with approximately 0.94 probability.
* Immediate endpoint JSD is approximately 0.0014 bits.
* After the shared ` is`, the predictions diverge to ` Tokyo` and ` Berlin`.
* Delayed endpoint JSD is approximately 0.99 bits.

## Success criterion

`REPORT.md` must give one of these conclusions:

1. **Delayed plateau:** the immediate prediction remains approximately unchanged, while the delayed output stays near ` Tokyo`, switches sharply, and then stays near ` Berlin`.
2. **No delayed plateau:** the endpoint conditions hold, but the delayed output changes smoothly.
3. **Invalid example:** the required GPT-2 Large endpoint behavior cannot be reproduced.

A negative result counts as complete.

## Fallback

If time runs short, finish the endpoint validation and delayed interpolation, save the two main figures, state the result, and create `STOP`.

## Setup

* Model: pretrained `openai-community/gpt2-large`.
* Evaluation mode; no sampling, training, or fine-tuning.
* Use 101 evenly spaced interpolation points (t\in[0,1]).
* Confirm that ` Japan`, ` Germany`, ` is`, ` Tokyo`, and ` Berlin` are each one GPT-2 token.
* Interpolate the token embeddings of ` Japan` and ` Germany`.
* Follow [Matthew Shinkle’s interpolation procedure](https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge): use shortest-arc spherical interpolation for direction, then rescale each interpolated vector so its norm changes linearly between the endpoint norms.
* Insert the interpolated embedding at the (A/B) token position before the first transformer layer. Keep the prefix, shared successor, and positional embeddings fixed.
* For every (t), run:

```text
P + [interpolated Japan→Germany embedding] + " is"
```

* From the same forward pass, record:

  * **Immediate logits:** logits at the interpolated (A/B) position, which predict ` is`.
  * **Delayed logits:** logits at the shared ` is` position, which predict ` Tokyo` or ` Berlin`.

## Stages

* [x] **S1 — Validate endpoints**

  * Reproduce the two endpoint inputs.
  * Record top-5 predictions and full probability distributions at both readout positions.
  * Verify:

    * both immediate top-1 predictions are ` is`;
    * delayed top-1 predictions are ` Tokyo` and ` Berlin`.
  * Compute immediate and delayed endpoint JSD in bits.
  * If either requirement fails, document the result and stop.

* [x] **S2 — Run interpolation**

  * Run all 101 interpolation points.
  * At the immediate readout, save:

    * (p(\texttt{ is}));
    * top-1 token.
  * At the delayed readout, save:

    * full logits;
    * (p(\texttt{ Tokyo}));
    * (p(\texttt{ Berlin}));
    * top-1 token.

* [x] **S3 — Measure the delayed transition**

  * Let (z(t)) be the delayed logit vector and (z_A,z_B) the endpoint delayed logits.
  * Compute:

[
d(t)=
\frac{\lVert z(t)-z_A\rVert_2}
{\lVert z(t)-z_A\rVert_2+\lVert z(t)-z_B\rVert_2}.
]

* Compute transition width:

[
w=t_{0.9}-t_{0.1},
]

where (t_q) is the first point at which (d(t)\ge q).

* Save:

  * `plots/immediate_prediction.png`: (p(\texttt{ is})) across (t).
  * `plots/delayed_distance.png`: delayed (d(t)), including the reference line (d=t).
  * `plots/delayed_tokens.png`: (p(\texttt{ Tokyo})) and (p(\texttt{ Berlin})) across (t).

* Call the result a clear delayed plateau only if:

  * the immediate prediction remains ` is` throughout most or all of the path;
  * delayed (d(t)) has visibly stable regions near both endpoints;
  * the delayed transition is concentrated in a narrow interval, with (w<0.5).

* [x] **S4 — Write the verdict**

  * If a delayed plateau appears, conclude:

> GPT-2 Large can preserve a discrete, future-relevant distinction between two contexts even when their immediate next-token outputs are almost identical.

* If the delayed transition is smooth, conclude:

> Future output divergence alone is insufficient to produce an activation plateau in this example.

* Do not generalize beyond this single example.

## Deliverables

* `RESULTS.md`: endpoint predictions, JSD values, transition width, and figure paths.
* `REPORT.md`: research question, method, three figures, and concise verdict.
* `plots/`: all saved figures.
* Empty `STOP` file when complete.

## Out of scope

* No additional prompts or token pairs.
* No other models.
* No training or fine-tuning.
* No layerwise, attention, MLP, neuron, or Jacobian analysis.
* Do not modify the prompt to rescue a failed endpoint.
* Do not interpret the immediate normalized-distance curve: its endpoints are nearly identical, making that normalization uninformative.

## On-track check

End each `JOURNAL.md` entry with:

```text
On track? <yes/no> — <stage, % done, blocker if any>
```

## Current status

Complete. S1–S4 done in one iteration. All four endpoint checks pass (immediate top-1 ` is` at both
endpoints, 0.944/0.940; delayed top-1 ` Tokyo` 0.928 / ` Berlin` 0.848; endpoint JSD 0.0014 bits
immediate, 0.9945 bits delayed — matching the preliminary numbers in this plan). The 101-point sweep
gives the verdict **conclusion 1, delayed plateau**: the immediate top-1 is ` is` at every `t`
(p in 0.931–0.944) while the delayed `d(t)` is monotone with width `w = 0.28` (linear null 0.80),
midpoint `t₅₀ = 0.48`, and the delayed top-1 flips ` Tokyo`→` Berlin` at `t = 0.49`. RESULTS.md and
REPORT.md are curated to this verdict; the three named figures are saved and embedded;
`check_render.py` passes on both; `STOP` written.

## Next step

None — the plan is complete and `STOP` exists. If a human drops feedback later, delete `STOP`,
address it, and re-write `STOP` only when clean again.
