# PLAN — Do Different Downstream Readouts Share the Same Plateau Boundary?

> Working folder: `shared_entity_transition`. Rewrite “Current status” and “Next step” after every iteration. Read `../BUDGET.md` and `../CLAUDE.md` every iteration.

## Research question

When interpolating the same `Japan → Germany` token embedding, do different downstream property readouts—capital, continent, currency, and language—switch at the same interpolation location?

The immediate next-token prediction after `Japan` and `Germany` is nearly identical. Country-specific information only becomes visible after a later readout suffix is provided.

If several readouts switch at approximately the same (t), this is consistent with a shared transition from a Japan-like future-relevant entity state to a Germany-like state.

If they switch at different (t), the plateau boundaries are more likely readout-specific.

This single example cannot prove that plateaus correspond to general semantic groups.

## Success criterion

`REPORT.md` must:

1. Show an individual normalized-distance (d(t)) plot for every readout.
2. Report (t_{10}), (t_{50}), (t_{90}), and transition width (w) for every readout.
3. Compare the transition locations of Capital, Continent, Currency, and Language.
4. Give one of three conclusions:

   * **Aligned transitions:** the downstream readouts switch at similar (t).
   * **Readout-specific transitions:** their transition locations clearly differ.
   * **Mixed or inconclusive:** some curves lack a clear transition or cannot be compared reliably.

Negative or mixed results count as complete.

## Fixed setup

* Model: pretrained `openai-community/gpt2-large`.
* Evaluation mode; no sampling, training, or fine-tuning.
* Use 101 evenly spaced interpolation points (t\in[0,1]).
* Use the following prefix exactly, including indentation and blank lines:

```text
Country: France
 Capital: Paris
 Continent: Europe
 Currency: euro
 Language: French
 Type: country

Country:
```

* Endpoint (A): ` Japan`
* Endpoint (B): ` Germany`

Append each of these readout suffixes after the endpoint:

```text
\n Capital:
\n Continent:
\n Currency:
\n Language:
\n Type:
```

Each suffix is exactly three GPT-2 tokens:

1. newline;
2. one property-name token;
3. colon.

The five inputs are therefore:

```text
P + [Japan→Germany interpolation] + "\n Capital:"
P + [Japan→Germany interpolation] + "\n Continent:"
P + [Japan→Germany interpolation] + "\n Currency:"
P + [Japan→Germany interpolation] + "\n Language:"
P + [Japan→Germany interpolation] + "\n Type:"
```

Use exactly the same prefix, endpoint position, and interpolated embedding for all five readouts.

## Preliminary endpoint behavior to reproduce

These values were previously measured using pretrained GPT-2 Large:

| Readout   | Japan-side prediction | Germany-side prediction | Endpoint JSD |
| --------- | --------------------: | ----------------------: | -----------: |
| Capital   |       ` Tokyo`, 0.925 |        ` Berlin`, 0.743 |   0.991 bits |
| Continent |        ` Asia`, 0.743 |        ` Europe`, 0.863 |   0.885 bits |
| Currency  |         ` yen`, 0.584 |          ` euro`, 0.388 |   0.915 bits |
| Language  |    ` Japanese`, 0.950 |        ` German`, 0.891 |   0.969 bits |
| Type      |     ` country`, 0.727 |       ` country`, 0.823 |   0.111 bits |

Before adding a readout suffix:

* both country endpoints predict newline as top-1;
* (p(\text{newline})) is approximately 0.929 after `Japan`;
* (p(\text{newline})) is approximately 0.945 after `Germany`;
* immediate endpoint JSD is approximately 0.0076 bits.

Confirm that ` Japan`, ` Germany`, and all expected answer strings are single GPT-2 tokens.

## Interpolation

* Interpolate only the input token embeddings of ` Japan` and ` Germany`.
* Follow Matthew Shinkle’s interpolation procedure:

  * use shortest-arc spherical interpolation for direction;
  * rescale each interpolated vector so its norm changes linearly between the endpoint norms.
* Insert the interpolated embedding before the first transformer layer.
* Keep positional embeddings, prefix tokens, suffix tokens, and model weights fixed.
* Construct the 101 interpolated embeddings once and reuse them for all five readouts.

## Normalized distance

For readout (r), let (z_r(t)) be the full logit vector after the final colon in its suffix.

Let:

[
z_{r,A}=z_r(0),
\qquad
z_{r,B}=z_r(1).
]

Compute:

[
d_r(t)=
\frac{\lVert z_r(t)-z_{r,A}\rVert_2}
{\lVert z_r(t)-z_{r,A}\rVert_2+
\lVert z_r(t)-z_{r,B}\rVert_2}.
]

This measures how the downstream readout moves from its Japan-side behavior to its Germany-side behavior.

It does not measure the immediate next-token prediction after the country token.

## Transition location

For each readout, compute:

[
t_{10}: d_r(t)=0.1,
\qquad
t_{50}: d_r(t)=0.5,
\qquad
t_{90}: d_r(t)=0.9.
]

Use linear interpolation between adjacent sampled points.

Define:

[
w_r=t_{90}-t_{10}.
]

Use (t_{50}) as the transition location and (w_r) as the transition width.

If a curve is strongly non-monotonic or crosses a threshold multiple times, report this explicitly. Do not smooth the curve or fit a new score to force a transition location.

## Stages

* [x] **S1 — Reproduce endpoint behavior**

  * Confirm all tokenizations.
  * Reproduce the shared immediate newline prediction.
  * Reproduce all five downstream readout predictions.
  * Compute immediate and readout endpoint JSD in bits.
  * Record top-5 predictions and target-token probabilities.
  * If any of the four divergent readouts fails to produce the expected top-1 predictions, document the discrepancy and stop.

* [x] **S2 — Run the shared interpolation**

  * Construct the 101 `Japan → Germany` interpolated embeddings.
  * For every (t), run the same embedding through all five readout suffixes.
  * At the immediate country-token position, record:

    * (p(\text{newline}));
    * top-1 token.
  * For every downstream readout, record:

    * full logits;
    * top-1 token;
    * probabilities of both expected endpoint tokens.
  * For Type, additionally record (p(\texttt{country})).

* [x] **S3 — Plot every (d(t)) curve**

  * Save:

    * `plots/distance_capital.png`
    * `plots/distance_continent.png`
    * `plots/distance_currency.png`
    * `plots/distance_language.png`
    * `plots/distance_type.png`
  * Every individual plot must:

    * show (d_r(t));
    * show the linear reference (d=t);
    * use (t\in[0,1]) and (d\in[0,1]);
    * mark (t_{10}), (t_{50}), and (t_{90});
    * report (w_r) in the title or caption;
    * use the same visual style and axis limits.
  * Also save:

    * `plots/distance_overlay.png`: all five curves on the same axes;
    * `plots/immediate_prediction.png`: (p(\text{newline})) across (t).

* [x] **S4 — Compare transition locations**

  * Create `plots/transition_comparison.png`.
  * For every readout:

    * plot (t_{50}) as a point;
    * plot ([t_{10},t_{90}]) as a horizontal interval.
  * Use Capital, Continent, Currency, and Language as the primary comparison.
  * Show Type in gray as a secondary, non-discriminating readout.
  * Include this table in `RESULTS.md`:

| Readout   | Endpoint JSD | (t_{10}) | (t_{50}) | (t_{90}) | (w) | Monotonic? |
| --------- | -----------: | -------: | -------: | -------: | --: | ---------- |
| Capital   |              |          |          |          |     |            |
| Continent |              |          |          |          |     |            |
| Currency  |              |          |          |          |     |            |
| Language  |              |          |          |          |     |            |
| Type      |              |          |          |          |     |            |

* For the four primary readouts, compute:

[
\Delta t_{50}
=============

## \max_r t_{50,r}

\min_r t_{50,r}.
]

* Treat (\Delta t_{50}\leq0.05) as descriptively aligned at the resolution of this experiment.

* Do not present this threshold as a statistical significance test.

* [x] **S5 — Write the verdict**

  * If the four primary readouts have clear transitions and (\Delta t_{50}\leq0.05), conclude:

> Capital, continent, currency, and language become Germany-like at approximately the same interpolation location. This is consistent with a shared transition in the future-relevant country representation that is subsequently accessed by different downstream readouts.

* If their (t_{50}) values clearly differ, conclude:

> Different property readouts become Germany-like at different interpolation locations. This supports readout-specific boundaries rather than a single shared entity-state transition.

* If some curves are smooth, weak, or non-monotonic, give a mixed or inconclusive verdict.
* Discuss Type separately:

  * both endpoints predict ` country`;
  * its full next-token distributions are not identical;
  * it is therefore a non-discriminating top-1 readout, not a perfect identical-distribution control.
* Do not call the delayed predictions the model’s immediate outputs.
* Do not claim that GPT-2 is explicitly planning the answer before seeing the readout suffix.
* Do not generalize from this pair to semantic groups in general.

## Interpretation boundary

The strongest supported conclusion from aligned transitions would be:

> The model preserves a future-relevant country representation whose change is reflected consistently across several later readouts.

Do not strengthen this into:

> The model has proven discrete semantic groups.

The experiment distinguishes:

* **shared entity-state transition**, versus
* **readout-specific transition**.

It does not fully distinguish all possible mechanisms producing either pattern.

## Deliverables

* `RESULTS.md`: endpoint checks, JSD values, transition table, and figure paths.
* `REPORT.md`: research question, method, every individual (d(t)) plot, transition comparison, and concise verdict.
* `plots/`: all figures.
* Machine-readable interpolation results in `.csv`, `.json`, or `.npz`.
* Empty `STOP` file when complete.

## Fallback

If time runs short:

1. finish all five interpolations;
2. save all five individual (d(t)) plots;
3. save the transition comparison plot;
4. report the transition table;
5. write the limited verdict;
6. create `STOP`.

## Out of scope

* No additional countries or token pairs.
* No other models.
* No training or fine-tuning.
* No new plateau score.
* No attention, MLP, neuron, Jacobian, probe, or layerwise analysis.
* Do not alter the prompts after seeing interpolation results.
* Do not smooth curves to make transition locations appear more aligned.
* Do not treat this experiment as proof of general semantic categories.

## On-track check

End every `JOURNAL.md` entry with:

```text
On track? <yes/no> — <stage, % done, blocker if any>
```

## Current status

**Complete — verdict: aligned transitions.** S1–S5 all done in one iteration.

* S1 reproduced every preliminary endpoint value exactly (immediate JSD 0.0076 bits, p(newline)
  0.929/0.945; Capital 0.991, Continent 0.885, Currency 0.915, Language 0.968, Type 0.111 bits;
  all tokenizations single-token, every suffix exactly 3 tokens).
* S2 ran the shared 101-point slerp interpolation of the ` Japan`→` Germany` input embedding through
  all five readout suffixes. All five d(t) curves are monotonic with exactly one crossing of each of
  0.1/0.5/0.9.
* S3/S4 produced all eight figures. t50 = 0.454 (Capital), 0.444 (Continent), 0.443 (Currency),
  0.450 (Language), 0.438 (Type); widths 0.255–0.279 against 0.80 for a linear change.
  **Δt50 across the four primary readouts = 0.011**, well inside the 0.05 descriptive-alignment
  threshold. Top-1 answers flip at t = 0.44–0.47.
* S5 verdict written in REPORT.md: aligned transitions, with the Type control discussed separately as
  a non-discriminating top-1 readout (0.111 bits of residual endpoint divergence), and the scope
  limits (one pair, one prompt, one model, no mechanism claim) stated.

Deliverables: REPORT.md (7 figures, ~2.9k words), RESULTS.md, plots/ (8 PNGs),
results/{s1_endpoints.json, interp.csv, interp.npz, transitions.json}.

## Next step

None — success criterion met. Direction finalized and STOP written.
