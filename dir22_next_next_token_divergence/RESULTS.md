# RESULTS — Delayed-Successor Plateau (GPT-2 Large)

> CURRENT-BEST ONLY. One row per experiment. No history (that lives in CHANGELOG.md).

## Headline

**Verdict: delayed plateau (PLAN conclusion 1).** Moving the token embedding continuously from
` Japan` to ` Germany` inside `The capital of France is Paris. The capital of X is` leaves the
*immediate* next-token prediction untouched — ` is` is the top-1 token at all 101 interpolation
positions, with probability never leaving 0.931–0.944 — while the prediction *one token later*
holds ` Tokyo`, switches abruptly near the middle of the path, and then holds ` Berlin`. The
delayed transition width is `w = 0.28`, far narrower than the linear reference `w = 0.80`.

Both endpoint conditions the plan required were reproduced exactly, so the example is valid.

## Metrics — endpoint validation (S1)

Every one of the five tokens is a single GPT-2 token, and all four endpoint predictions match the
plan. The two contexts are almost indistinguishable now and almost disjoint one token later.

| Check | Sequence | Top-1 | p(top-1) | Required token | Pass |
|---|---|---|---|---|---|
| Immediate | `…The capital of Japan` | ` is` | 0.944 | ` is` | ✓ |
| Immediate | `…The capital of Germany` | ` is` | 0.940 | ` is` | ✓ |
| Delayed | `…The capital of Japan is` | ` Tokyo` | 0.928 | ` Tokyo` | ✓ |
| Delayed | `…The capital of Germany is` | ` Berlin` | 0.848 | ` Berlin` | ✓ |

Endpoint Jensen–Shannon divergence (JSD, bits): immediate **0.0014**, delayed **0.9945** — a
factor of 690. Runner-up predictions are also sensible (` Kyoto` 0.025, ` Osaka` 0.019 after
` Japan`; ` Munich` 0.067, ` Frankfurt` 0.031 after ` Germany`), so the delayed divergence is a
genuine capital-city lookup rather than one lucky token.

## Metrics — interpolation sweep (S2/S3, 101 points)

The delayed readout satisfies all three plateau criteria the plan set: the immediate prediction
never changes, `d(t)` is visibly flat near both endpoints, and the transition is concentrated in a
narrow interval well under the 0.5 threshold.

| Quantity | Value |
|---|---|
| Immediate top-1 = ` is` | 101 / 101 positions (100%) |
| Immediate p(` is`) range | 0.931 – 0.944 |
| Delayed transition width `w = t₀.₉ − t₀.₁` | **0.28** (t₀.₁ = 0.34, t₀.₉ = 0.62) |
| Linear reference width | 0.80 |
| Delayed midpoint `t₅₀` | 0.48 |
| Delayed top-1 flip (` Tokyo` → ` Berlin`) | t = 0.49 |
| `d(t)` monotone in `t` | yes |
| Delayed endpoint separation ‖z_A − z_B‖₂ | 462.5 |

Outside the transition the delayed distribution barely moves: `d(t) ≤ 0.077` for `t ≤ 0.30` and
`d(t) ≥ 0.89` for `t ≥ 0.60`. p(` Tokyo`) is still 0.902 at `t = 0.45` and has collapsed to 0.070
by `t = 0.50`; p(` Berlin`) reaches 0.833 by `t = 0.55` and is flat thereafter.

## Figures

The first question is whether the immediate prediction really stays fixed — without that, a delayed
switch would be unremarkable.

![Probability of ' is' at the interpolated position across the interpolation](plots/immediate_prediction.png)

**Figure 1.** The immediate prediction is effectively constant. x: interpolation position `t`
(0 = ` Japan`, 1 = ` Germany`); y: probability of ` is` at the interpolated position, linear scale
0–1. The curve stays within 0.931–0.944 and ` is` is the top-1 token at every `t`.

The plateau claim is about the shape of the delayed response, so we plot its relative logit distance
against the linear reference. To check whether the model's delayed prediction switches where the
logits do, the top panel shows the delayed top-1 token (argmax over the full 50,257-token vocabulary
of `results/logits_delayed.npy`) at every `t`. Only ` Tokyo` (`t = 0.00–0.48`) and ` Berlin`
(`t = 0.49–1.00`) ever win, so there is exactly one switch, between `t = 0.48` (`d = 0.52`) and
`t = 0.49` (`d = 0.58`). `d(t)` first reaches 0.5 at `t₅₀ = 0.48`, so the token switch lags the logit
midpoint by one 0.01 grid step and lies inside the transition interval 0.34–0.62.

![Delayed top-1 token and relative logit distance at the delayed readout versus interpolation position](plots/delayed_distance.png)

**Figure 2.** The delayed logits are flat, switch sharply, then flat again; the delayed top-1 token
switches once, just after the logit midpoint. Top panel: delayed top-1 token at each of the 101
values of `t` (circles, upper row = ` Tokyo`; squares, lower row = ` Berlin`). Bottom panel: x:
interpolation position `t`; y: relative logit distance `d(t)` (0 at the ` Japan` endpoint, 1 at the
` Germany` endpoint). Solid with triangles = delayed readout after ` is`; dotted gray = linear
reference `d = t`; open diamond = first `t` with `d ≥ 0.5` (`t₅₀ = 0.48`). Thin horizontal lines mark
the 0.1 and 0.9 levels; the shaded band is the transition interval of width `w = 0.28`. The
dash-dotted vertical line in both panels marks the observed top-1 switch, between `t = 0.48` and
`t = 0.49`.

Finally, whether that geometric switch is also a behavioural one: which city the model actually
predicts.

![Probability of ' Tokyo' and ' Berlin' at the delayed readout across the interpolation](plots/delayed_tokens.png)

**Figure 3.** The predicted capital swaps within a few interpolation steps. x: interpolation
position `t`; y: probability at the delayed readout. Solid with circles = p(` Tokyo`), dashed with
squares = p(` Berlin`). The dash-dotted vertical line marks `t = 0.49`, where the top-1 token flips.
