# Do "same continuation, different phrasing" prompt pairs produce an activation plateau in GPT-2 Large?

## Summary

When you take the internal state a language model builds for one prompt, the state it builds for a
second prompt, and slide continuously from the first to the second, the model's output does not
always change smoothly along the way. Sometimes the output barely moves for most of the path and then
flips over a short stretch. That flat stretch is what this project calls an **activation plateau**.
Plateaus are interesting for safety work because they suggest a model's behaviour can be insensitive
to large internal changes and then very sensitive to a small one, which is exactly the regime where
edits, steering, or distribution shift produce surprises.

The concrete research question here is narrow and empirical: **do four specific hand-designed prompt
pairs — pairs that should give the model the same next word by two different routes — produce a
visible plateau in GPT-2 Large?**

The answer is: mostly the pairs do not even get off the ground, and the one that does shows a
plateau. Three of the four pairs fail before any interpolation happens, because GPT-2 Large does not
actually produce the intended answer for both prompts. The surviving pair — asking for "Paris" when
the answer is already sitting in the context, versus asking for it from stored knowledge — shows a
clearly visible plateau: the output creeps 17% of the way across the gap over the first 57% of the
path, then covers 71% of the gap in the next 10% (Figure 1). Throughout the whole path the model's
top predicted token stays ` Paris`.

## Methods

### Data & Model

**Model.** GPT-2 Large (`gpt2-large`, 774M parameters, 36 transformer blocks), float32, greedy
decoding, no sampling. There is no dataset: the inputs are eight hand-written prompts, listed
verbatim in the Results section.

**Hook point.** All interpolation happens at the *residual stream* — the running vector, one per
token position, that each transformer block reads from and writes back to. We take `resid_post`
after block 0, i.e. the output of the first of the 36 blocks.

**What is interpolated.** The two prompts in a pair differ at more than one token position, so we
interpolate the **entire residual-stream sequence** (all 12 token positions for the pair that
qualified), not just the final token. For each position we interpolate the direction of the vector
with SLERP (spherical linear interpolation — moving along the arc between two unit vectors rather
than the chord, so intermediate vectors stay on the sphere) and interpolate the L2 norm linearly.
This is the interpolation used by the activation-plateau reference implementation this project
follows. We then write the interpolated sequence back into block 0's output and read the logits at
the final token position.

**Sample size.** 50 interpolation positions, `t` evenly spaced from 0 to 1 inclusive, for the single
pair that passed the screen. One pair, one layer, one model — this is a small hand-designed probe,
not a survey.

**Screening rule.** Before interpolating, each prompt is continued greedily for 6 tokens. A pair is
tested only if both prompts give the intended answer **and** the two prompts have the same GPT-2
tokenized length (unequal lengths make a position-by-position interpolation ill-defined). One
minimal wording repair was permitted for a length mismatch; it was used once, on pair 3.

### Metrics

We need one number per interpolation position that says "how far along are we, in output space?" The
obvious choice, the model's predicted word, is useless here by construction: every pair is designed
so that both prompts predict the *same* word, so the argmax never moves and would report nothing.
Instead we measure the position of the whole final-position logit vector relative to the two
endpoints. Writing `x(t)` for the final-position logit vector at interpolation position `t`, and
`A = x(0)` and `B = x(1)` for the two endpoint logit vectors, the relative logit distance is:

```math
d(t) = \frac{\lVert x(t) - A \rVert}{\lVert x(t) - A \rVert + \lVert x(t) - B \rVert}
```

Read it as a fraction of the journey: `d = 0` means the output is exactly the prompt-A output,
`d = 1` means it is exactly the prompt-B output, and `d = 0.5` means it is equidistant from both.
Neither direction is "better" — the shape of the curve is the result, not its level. A smooth
diagonal means the output tracks the internal interpolation proportionally; a flat stretch followed
by a sharp rise is a plateau. This metric is consumed by Figure 1, the only quantitative result in
this report.

**Plateau verdict.** By design this is a visual call with no fitted threshold: we call a curve a
plateau only if `d(t)` visibly hugs one endpoint for an extended part of the path and then changes
much faster over a shorter region, and otherwise report "no clear plateau". No change point is
fitted, no plateau score is computed, and no statistical test is run — the question asked here is
whether the effect is visible at all in these four hand-built pairs.

### Baselines

This experiment has no baseline method and no comparison condition. It is a screening question
about four specific prompt pairs, so the reference point is the qualitative alternative to a
plateau — a `d(t)` curve that rises steadily across the whole path — rather than a competing
technique. The endpoint sanity check reported below serves as the correctness control.

## Results

### Three of four pairs never reach the interpolation stage

The screen is unforgiving, and that is the main finding. Two pairs fall over because GPT-2 Large
simply does not know the answer in the requested form: asked `The number after six is`, it continues
` the number of times the player`, and asked `The author of Hamlet is`, it continues ` a man of the
world,`. Two pairs also have mismatched tokenized lengths, which alone would block a
position-by-position interpolation. The table below gives every prompt and continuation verbatim
(`\n` stands for a newline character).

| Pair | Side | Exact prompt | Tokens | Greedy 6-token continuation | Intended | Verdict |
|---|---|---|---|---|---|---|
| 1 Copy vs recall | A | `The capital of France is Paris. The capital of France is` | 12 | ` Paris.\n\nThe capital` | Paris | **pass** |
| 1 | B | `The capital of Spain is Madrid. The capital of France is` | 12 | ` Paris. The capital of Germany` | Paris | **pass** |
| 2 Successor vs predecessor | A | `The number after six is` | 5 | ` the number of times the player` | seven | fail |
| 2 | B | `The number before eight is` | 5 | ` the number of times the player` | seven | fail |
| 3 Different relation | A | `The capital of France is` | 5 | ` the capital of France.\n` | Paris | fail |
| 3 | B | `The largest city in France is` | 6 | ` the capital, Paris. It` | Paris | (length mismatch) |
| 4 Different entity | A | `The author of Hamlet is` | 6 | ` a man of the world,` | Shakespeare | fail |
| 4 | B | `The author of Macbeth is` | 7 | ` a man of the world,` | Shakespeare | fail |

Pair 3 received the one permitted wording repair, `The capital city of France is` in place of
`The capital of France is`. That equalises the lengths at 6 tokens but the A side then continues
` home to the world's largest`, so the pair still fails the answer check and was dropped. Pair 4 was
not repaired, because its problem is the answer rather than the length.

So the per-pair verdicts are: pair 1 **plateau visible** (below); pairs 2, 3 and 4 **not testable**,
because GPT-2 Large does not give the intended completion for both prompts.

### Pair 1 shows a clear plateau

Pair 1 contrasts two ways of arriving at "Paris". In prompt A the answer is already present in the
context and can be copied; in prompt B the context supplies a parallel example about Spain, and the
answer for France has to come from stored knowledge. Both prompts end with the identical string
`The capital of France is`, and both greedily continue with ` Paris`.

To check that the interpolation machinery is doing what we claim, note that patching the full
block-0 sequence overrides every downstream computation, so the endpoints must reproduce the clean
runs. They do: at `t = 0` the patched logits differ from the clean prompt-A logits by L2 = 0.0009,
and at `t = 1` from the clean prompt-B logits by L2 = 0.0012, against an endpoint separation of
L2 = 445.2. The endpoints are far apart, so `d(t)` is measuring a real gap.

Figure 1 answers the pair's question directly: does the output move steadily as the internal state
slides from A to B, or does it wait and then jump?

![relative logit distance against interpolation position for pair 1](plots/p1_copy_vs_recall_dt.png)

**Figure 1.** GPT-2 Large, pair 1, block-0 residual stream, full-sequence interpolation.
x: interpolation position `t` (0 = prompt A endpoint, 1 = prompt B endpoint). y: relative logit
distance `d(t)` as defined in Methods (0 = output equals the prompt-A endpoint, 1 = equals the
prompt-B endpoint). Dotted horizontal line marks `d = 0.5`. One solid curve with round markers,
the 50 measured positions; no second series. Verdict: **plateau visible**.

The curve is flat-then-steep by a wide margin. Over the first 57% of the path (`t = 0` to
`t = 0.571`) the output covers only 17% of the distance between the endpoints. Over the next 10% of
the path (`t = 0.571` to `t = 0.673`) it covers 71%. The remaining third of the path carries the
last 12%. In other words, the steepest tenth of the path does about four times as much output
movement as the first six-tenths combined.

Two details make the plateau more striking. First, the top-1 predicted token is ` Paris` at both
endpoints and at all 50 interpolation positions: the jump is a reorganisation of the full logit
vector underneath a prediction that never changes, which is precisely why the vector-level metric
was needed. Second, only two token positions genuinely differ between the two prompts — the ones
holding ` France`/` Spain` and ` Paris`/` Madrid` (per-position cosine 0.65 and 0.54; every other
position is above 0.94). The plateau appears while moving those two positions.

### What this does and does not show

The useful part of this result is a practical one for anyone designing plateau experiments on
language models: **the binding constraint is the completion screen, not the interpolation**. Three of
four intuitively reasonable "same answer, different route" contrasts died because a 774M-parameter
model does not answer `The number after six is` with `seven`. Any study that assumes hand-designed
semantic contrasts will behave as intended should screen them first; here the screen removed 75% of
the candidates. Pair 1 is the design that survived, and its ingredient — keeping the surface form of
the shared final clause identical and varying only the earlier context — is the property worth reusing.

Two limits are worth stating plainly. This is one pair, one layer, one model, so nothing here
establishes how common plateaus are. And a visible plateau does not demonstrate that the model uses
two different computational pathways for the two prompts; the experiment measures only the shape of
the output curve along an interpolated path, and a mechanism claim would need interventions this
study did not run.

## Conclusion

Of the four hand-designed same-continuation prompt pairs, only the copy-available versus
factual-recall pair (pair 1) passed the completion screen on GPT-2 Large, and it shows a plateau:
`d(t)` moves 17% of the way over the first 57% of the interpolation and then 71% over the next 10%
(Figure 1), with the predicted token fixed at ` Paris` throughout. Pairs 2, 3 and 4 are negative
results at the screening stage — GPT-2 Large does not produce the intended answer for both prompts,
and the one permitted wording repair on pair 3 fixed its length mismatch without fixing its answer.
The headline for follow-up work is that candidate prompt pairs must be screened on the actual model
before any interpolation is attempted, and that the plateau observed here is a statement about the
shape of one output curve, not evidence of distinct internal pathways.

Full per-pair evidence, raw continuations, endpoint checks and per-position cosines are in
`RESULTS.md`.
