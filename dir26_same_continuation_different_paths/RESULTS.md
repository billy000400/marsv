# RESULTS — Same answer, different route: activation plateaus on GPT-2 Large and Qwen2.5-3B

> CURRENT-BEST ONLY. Detailed evidence record behind `REPORT.md`. History lives in CHANGELOG.md.

## What was run

**Models.** GPT-2 Large (`gpt2-large`, 774M parameters, 36 blocks, float32) and Qwen2.5-3B
(`Qwen/Qwen2.5-3B`, 3.09B parameters, 36 blocks, bfloat16). Both base models, greedy decoding, single
GPU. Qwen2.5-3B was added in response to `human_feedback.txt`, which asked for a bigger model; it is
the largest Qwen2.5 base model that fits this project's GPU share, and bfloat16 is required for the
same reason. Select the model with `DIR26_MODEL=gpt2|qwen`.

**S1 — completion screen.** Each of the 8 prompts is continued greedily for 6 new tokens
(`do_sample=False`), and the next-token distribution at the end of the prompt is recorded: top-1 token
with its probability, plus the top 5. A pair passes only if the intended answer appears in both
continuations. Script `experiments/s1_completions.py`; raw output
`results/s1_completions_{gpt2,qwen}.json`.

**S1b — the single authorized wording repair.** PLAN.md allows at most one wording repair and names
the wording to try for pair 3 (`The capital city of France is`). Script `experiments/s1b_repair.py`;
raw output `results/s1b_repair_{gpt2,qwen}.json`.

**S2 — interpolation.** For each passing, length-matched pair: hook the output of block 0
(`resid_post` after block 0), take the whole residual-stream sequence for prompt A and for prompt B,
interpolate them position by position (SLERP on the direction, linear interpolation of the L2 norm),
patch the full interpolated sequence back in, and read the final-position logits. 50 interpolation
positions, `t = numpy.linspace(0, 1, 50)`. Scripts `experiments/s2_interp.py` and
`experiments/s2_plot.py`; raw output `results/<pair>_interp_<model>.npz`,
`results/s2_summary_{gpt2,qwen}.json`.

**Metric.** Relative logit distance, with `x(t)` the final-position logit vector at interpolation
position `t` and `A = x(0)`, `B = x(1)` the endpoints:

```math
d(t) = \frac{\lVert x(t) - A \rVert}{\lVert x(t) - A \rVert + \lVert x(t) - B \rVert}
```

`d = 0` is the prompt-A output, `d = 1` the prompt-B output. The shape of the curve across `t` is the
result; a flat stretch followed by a sharp rise is a plateau.

## S1 completion screen on Qwen2.5-3B — all 8 exact prompts plus the repair

In every table below `\n` stands for a literal newline and a leading space inside backticks is part of
the token. Three pairs pass on Qwen2.5-3B. Pair 2 fails on the A side only; the top-5 column shows why,
with three blank-filler tokens above ` seven`. Pair 4 passes on the letter of the screen — the intended
answer appears in both continuations — but its immediate next token is a blank-filler, so its endpoints
describe a fill-in-the-blank state.

| Pair | Side | Exact prompt | Tokens | Top-1 next token (prob) | Top-5 next tokens (prob) | Greedy 6-token continuation | Intended |
|---|---|---|---|---|---|---|---|
| 1 Copy vs recall | A | `The capital of France is Paris. The capital of France is` | 12 | ` Paris` (0.62) | ` Paris` 0.62, ` also` 0.07, ` the` 0.04, ` located` 0.03, ` a` 0.02 | ` Paris. The capital of France` | Paris |
| 1 Copy vs recall | B | `The capital of Spain is Madrid. The capital of France is` | 12 | ` Paris` (0.95) | ` Paris` 0.95, ` ____` 0.01, ` __` 0.01, ` ______` 0.00, `____` 0.00 | ` Paris. The capital of Germany` | Paris |
| 2 Successor vs predecessor | A | `The number after six is` | 5 | ` __` (0.19) | ` __` 0.19, ` ____` 0.13, ` ` 0.12, ` seven` 0.07, ` ______` 0.06 | ` ____\nA. 5` | seven |
| 2 Successor vs predecessor | B | `The number before eight is` | 5 | ` seven` (0.18) | ` seven` 0.18, ` __` 0.16, ` ____` 0.11, ` ` 0.10, ` ______` 0.07 | ` seven. The number after eight` | seven |
| 3 Different relation | A | `The capital of France is` | 5 | ` Paris` (0.45) | ` Paris` 0.45, ` __` 0.06, ` ______` 0.04, `____` 0.04, ` located` 0.04 | ` Paris. The capital of the` | Paris |
| 3 Different relation | B | `The largest city in France is` | 6 | ` Paris` (0.26) | ` Paris` 0.26, ` __` 0.16, ` ____` 0.08, `____` 0.08, `\n` 0.07 | ` Paris. The largest city in` | Paris |
| 4 Different entity | A | `The author of Hamlet is` | 6 | ` ______` (0.18) | ` ______` 0.18, ` __` 0.16, ` ____` 0.11, `\n` 0.08, `____` 0.06 | ` ______.\nA. William Shakespeare` | Shakespeare |
| 4 Different entity | B | `The author of Macbeth is` | 6 | ` ______` (0.14) | ` ______` 0.14, ` __` 0.10, ` ____` 0.09, ` William` 0.09, `\n` 0.07 | ` ______.\nA. William Shakespeare` | Shakespeare |
| 3 repaired | A | `The capital city of France is` | 6 | ` Paris` (0.43) | ` Paris` 0.43, ` __` 0.08, `____` 0.05, ` ______` 0.05, ` located` 0.04 | ` Paris. The capital city of` | Paris |
| 3 repaired | B | `The largest city in France is` | 6 | ` Paris` (0.26) | ` Paris` 0.26, ` __` 0.16, ` ____` 0.08, `____` 0.08, `\n` 0.07 | ` Paris. The largest city in` | Paris |

## S1 completion screen on GPT-2 Large — the same 8 prompts plus the repair

GPT-2 Large fails in a different way. Its top-1 predictions on the failing prompts are generic function
words at probabilities near 0.1, and the intended answers are absent or low in the top 5, so the
failures are missing knowledge rather than formatting. The repaired pair-3 A prompt continues
` home to the world's largest`, so the repair fixes the token-length mismatch without fixing the answer.

| Pair | Side | Exact prompt | Tokens | Top-1 next token (prob) | Top-5 next tokens (prob) | Greedy 6-token continuation | Intended |
|---|---|---|---|---|---|---|---|
| 1 Copy vs recall | A | `The capital of France is Paris. The capital of France is` | 12 | ` Paris` (0.65) | ` Paris` 0.65, ` the` 0.02, ` France` 0.02, ` not` 0.02, ` New` 0.02 | ` Paris.\n\nThe capital` | Paris |
| 1 Copy vs recall | B | `The capital of Spain is Madrid. The capital of France is` | 12 | ` Paris` (0.94) | ` Paris` 0.94, ` Lyon` 0.01, ` B` 0.01, ` T` 0.01, ` Mont` 0.00 | ` Paris. The capital of Germany` | Paris |
| 2 Successor vs predecessor | A | `The number after six is` | 5 | ` the` (0.28) | ` the` 0.28, ` a` 0.11, ` not` 0.03, ` an` 0.03, ` for` 0.02 | ` the number of times the player` | seven |
| 2 Successor vs predecessor | B | `The number before eight is` | 5 | ` the` (0.27) | ` the` 0.27, ` a` 0.10, ` not` 0.03, ` an` 0.02, ` called` 0.02 | ` the number of times the player` | seven |
| 3 Different relation | A | `The capital of France is` | 5 | ` the` (0.10) | ` the` 0.10, ` Paris` 0.07, ` a` 0.07, ` also` 0.04, ` located` 0.04 | ` the capital of France.\n` | Paris |
| 3 Different relation | B | `The largest city in France is` | 6 | ` the` (0.10) | ` the` 0.10, ` Paris` 0.09, ` also` 0.07, ` not` 0.04, ` a` 0.03 | ` the capital, Paris. It` | Paris |
| 4 Different entity | A | `The author of Hamlet is` | 6 | ` a` (0.13) | ` a` 0.13, ` dead` 0.10, ` not` 0.06, ` the` 0.03, ` also` 0.02 | ` a man of the world,` | Shakespeare |
| 4 Different entity | B | `The author of Macbeth is` | 7 | ` a` (0.11) | ` a` 0.11, ` not` 0.05, ` the` 0.05, ` also` 0.03, ` an` 0.02 | ` a man of the world,` | Shakespeare |
| 3 repaired | A | `The capital city of France is` | 6 | ` home` (0.08) | ` home` 0.08, ` the` 0.07, ` a` 0.06, ` also` 0.03, ` known` 0.03 | ` home to the world's largest` | Paris |
| 3 repaired | B | `The largest city in France is` | 6 | ` the` (0.10) | ` the` 0.10, ` Paris` 0.09, ` also` 0.07, ` not` 0.04, ` a` 0.03 | ` the capital, Paris. It` | Paris |

Screen outcome: on Qwen2.5-3B pairs 1, 3 (repaired) and 4 are interpolated; pair 2 is not testable. On
GPT-2 Large only pair 1 is interpolated.

## S2 endpoint sanity check

Patching the full block-0 sequence overrides every downstream computation, so at `t = 0` and `t = 1`
the patched logits must reproduce the clean forward pass. They do, to within the arithmetic precision
used: the residual is about 3% of the endpoint separation in bfloat16 on Qwen2.5-3B and about 2e-6 of
it in float32 on GPT-2 Large. The endpoints are far apart in all cases, so `d(t)` measures a real gap.
The last column records that the model's predicted token never changes along the path, which is why the
vector-level metric is needed.

| Pair | Prompt tokens | Endpoint separation L2(A,B) | Patched vs clean L2 at t=0 | at t=1 | Top-1 token at all 50 positions |
|---|---|---|---|---|---|
| 1 Copy vs recall | 12 | 543.2 | 15.5877 | 12.9093 | ` Paris` |
| 4 Different entity | 6 | 321.5 | 12.8374 | 12.8844 | ` ______` |
| 3 repaired | 6 | 377.5 | 10.2595 | 11.7624 | ` Paris` |

On GPT-2 Large, run in float32, the same check is four orders of magnitude tighter, confirming that
the residual on Qwen2.5-3B above is bfloat16 rounding and not a bug in the patching.

| Pair | Prompt tokens | Endpoint separation L2(A,B) | Patched vs clean L2 at t=0 | at t=1 | Top-1 token at all 50 positions |
|---|---|---|---|---|---|
| 1 Copy vs recall | 12 | 445.2 | 0.0009 | 0.0012 | ` Paris` |

## S2 relative logit distance `d(t)`

The curve shape is the verdict; the numbers below are readings off the curves in Figure 1 and Figure 2,
not a fitted score. "Steepest 20%-wide window" is the interval of `t` of width 0.2 over which `d(t)`
gains the most, and the next column is how much of the endpoint gap that interval covers. Pair 1
concentrates roughly two thirds (Qwen2.5-3B) to three quarters (GPT-2 Large) of the whole journey into
a fifth of the path; pairs 3 and 4 spread it out.

| Pair (model) | d(0.2) | d(0.4) | d(0.5) | d(0.6) | d(0.8) | Steepest 20%-wide window of t | Share of the gap it covers | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 Copy vs recall (Qwen2.5-3B) | 0.056 | 0.210 | 0.427 | 0.854 | 0.952 | [0.39, 0.59] | 65% | **plateau visible** |
| 3 repaired (Qwen2.5-3B) | 0.108 | 0.216 | 0.373 | 0.615 | 0.868 | [0.45, 0.65] | 44% | no clear plateau |
| 4 Different entity (Qwen2.5-3B) | 0.072 | 0.205 | 0.367 | 0.569 | 0.824 | [0.43, 0.63] | 37% | no clear plateau |
| 1 Copy vs recall (GPT-2 Large) | 0.039 | 0.100 | 0.141 | 0.248 | 0.950 | [0.55, 0.75] | 77% | **plateau visible** |

Figure 1 shows the three Qwen2.5-3B curves together.

![relative logit distance against interpolation position for three prompt pairs on Qwen2.5-3B](plots/qwen_dt_three_pairs.png)

**Figure 1.** Qwen2.5-3B, block-0 residual stream, full-sequence interpolation, 50 positions.
x: interpolation position `t` (0 = prompt A endpoint, 1 = prompt B endpoint). y: relative logit
distance `d(t)` (0 = output equals the prompt-A endpoint, 1 = equals the prompt-B endpoint). Series:
pair 1 copy-available versus factual-recall (solid, circles); pair 3 repaired (dashed, squares);
pair 4 Hamlet versus Macbeth (dotted, triangles). Dotted horizontal line marks `d = 0.5`. Pair 1 is
flat-then-steep; pairs 3 and 4 rise steadily throughout.

Figure 2 checks whether pair 1's shape depends on the model. Pair 1 is the only pair testable on both.

![relative logit distance for pair 1 on Qwen2.5-3B and GPT-2 Large](plots/p1_gpt2_vs_qwen_dt.png)

**Figure 2.** Pair 1 on both models, same axes as Figure 1. Qwen2.5-3B (solid, circles), GPT-2 Large
(dashed, squares). Both are flat-then-steep; the GPT-2 Large transition is sharper and later along the
path.

## Per-position cosine between the two prompts' block-0 residual vectors

This records how much of the sequence actually moves during interpolation: a cosine of 1.00 means the
two prompts have an identical residual vector at that position, so interpolating it does nothing. In
pair 1 only two of twelve positions differ substantially, the ones holding ` France`/` Spain` and
` Paris`/` Madrid`. Positions are in reading order.

On Qwen2.5-3B, pairs 3 and 4 move a larger share of their (shorter) sequences than pair 1 does.

| Pair | Per-position cosine between the two prompts' block-0 residual vectors |
|---|---|
| 1 Copy vs recall | 1.00, 1.00, 1.00, 0.67, 0.99, 0.57, 0.99, 0.99, 1.00, 1.00, 0.99, 1.00 |
| 4 Different entity | 1.00, 1.00, 1.00, 0.38, 0.31, 0.97 |
| 3 repaired | 1.00, 0.20, 0.77, 0.32, 0.98, 0.99 |

On GPT-2 Large the pair-1 cosines are close to the Qwen2.5-3B ones, so the two models see a similar
amount of movement along the interpolation path.

| Pair | Per-position cosine between the two prompts' block-0 residual vectors |
|---|---|
| 1 Copy vs recall | 1.00, 1.00, 1.00, 0.65, 1.00, 0.54, 0.97, 0.94, 0.99, 0.96, 0.99, 0.99 |

## Verdicts

- **Pair 1, copy-available vs factual-recall** — testable on both models; **plateau visible** on both
  (Figure 2).
- **Pair 2, successor vs predecessor** — **not testable** on either model. GPT-2 Large answers ` the`
  on both sides; Qwen2.5-3B answers ` seven` for the B prompt but ranks blank-filler tokens above it
  for the A prompt.
- **Pair 3, different relation** — not testable on GPT-2 Large; testable on Qwen2.5-3B after the one
  authorized wording repair, verdict **no clear plateau** (Figure 1).
- **Pair 4, different entity** — not testable on GPT-2 Large; testable on Qwen2.5-3B, verdict **no
  clear plateau** (Figure 1), with the caveat that both endpoints sit in a fill-in-the-blank state.
