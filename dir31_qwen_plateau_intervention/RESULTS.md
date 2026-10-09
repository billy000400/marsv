# RESULTS — Does the Japan-to-Germany plateau follow country identity or the later answer? (Qwen3-0.6B)

Detailed evidence record for this direction. `REPORT.md` is the short synthesis. All numbers come from
`results/` and can be regenerated with `bash experiments/run_all.sh`.

## Headline

We could not separate the two explanations. No edit of transcoder features at the country word reversed the
capital answer at natural strength. The only edit that did (layer 16, 10 features, twice the natural strength)
also changed the country itself: the country readout flipped and the model's currency and language answers
switched to the other country. Under this edit the answer boundary, the readout crossing and most of the
late-layer plateau boundary disappear together, so the plan's verdict is **not disentangled**.

## Terms used below

- **Country token**: the token ` Japan` or ` Germany` in the prompt. **Bridge**: the text ` is` (or
  ` is the city of`) that we append after the country token so the model has to name the capital next.
- **State**: the residual-stream vector at one token after one of the 28 transformer blocks (width 1024).
- **Path**: 41 states at the layer-0 block output of the country token, from the Japan state (`t = 0`) to
  the Germany state (`t = 1`), by spherical interpolation of direction and linear interpolation of length.
  Everything downstream is recomputed for each `t`.
- **Adjacent-step change**: L2 distance between the states at two neighbouring `t` values, at a given layer.
  **Peak / mean** is its largest value divided by its mean over the 40 steps (1.0 for a path that moves at
  constant speed). **Boundary** is the `t` where the adjacent-step change is largest.
- **Relative distance `d(t)`**: distance to the unedited `t = 0` state divided by the sum of the distances to
  the unedited `t = 0` and `t = 1` states. **Width 10–90** is the range of `t` over which `d` goes from 0.1 to
  0.9 (0.8 for a straight path).
- **Margin**: logit of ` Tokyo` minus logit of ` Berlin` at the last bridge token. Positive means Tokyo is
  preferred. **Answer boundary** is the `t` where the margin crosses zero.
- **Country readout**: a linear score fitted on 32 non-capital prompts (16 per country); +1 is the mean Japan
  state and −1 the mean Germany state. It is fitted separately per layer and never sees capital prompts or
  edited states.
- **Feature**: one unit of a transcoder, a sparse dictionary that imitates one MLP block. **Decoder
  direction**: the vector that a feature adds to the MLP output.
- **Edit**: `MLP_new = MLP_native + beta * sum_i (a_donor_i − a_current_i) * decoder_i` at one layer and token.
  `a_current` is the feature's value in the run being edited and `a_donor` its stored value in the other
  country's tuning prompt. `beta = 1` replaces the features by the donor's values; `beta = 2` goes twice as
  far and is labelled extrapolation.
- **Random control**: the same coefficients placed on 10 random decoder directions of the same dictionary and
  rescaled to the real edit's L2 norm (3 seeds).

## 1. Baseline: prompts, plateau and readout

All six capital prompts are answered correctly, and ` Japan`, ` Germany`, ` is`, ` Tokyo`, ` Berlin` are single
tokens. The table lists the unedited behaviour. "P(` is`)" is read at the country token; the other columns are
read after the bridge.

| Prompt | Role | P(` is`) | Top token | Its probability | Margin | Greedy continuation |
|---|---|---:|---|---:|---:|---|
| `The capital city of Japan is` | tuning | 0.409 | `Tokyo` | 0.183 | +9.53 | `Tokyo. The capital city of China is` |
| `The capital city of Germany is` | tuning | 0.601 | `Berlin` | 0.252 | −9.17 | `Berlin. The capital city of France is` |
| `The capital of Japan is` | reserved (original) | 0.693 | `Tokyo` | 0.402 | +10.15 | `Tokyo. The capital of the United States` |
| `The capital of Germany is` | reserved (original) | 0.751 | `Berlin` | 0.477 | −10.23 | `Berlin. The capital of France is Paris` |
| `The capital of Japan is the city of` | reserved | 0.693 | `Tokyo` | 0.356 | +11.58 | `Tokyo. The capital of the United States` |
| `The capital of Germany is the city of` | reserved | 0.751 | `Berlin` | 0.416 | −11.01 | `Berlin. The capital of the United States` |

**Path endpoints.** The two layer-0 states have cosine 0.665 (angle 0.844 rad) and norms 10.71 and 10.93. The
path uses the Japan prompt's tokens for all `t`. At `t = 0` it reproduces the native Japan run (largest logit
difference 4e-5). At `t = 1` the country-token states equal the native Germany run (relative difference 1e-6),
and the after-bridge logits differ from it by at most 0.13 (P(` Berlin`) 0.4776 against 0.4770), because the
` is` token's own layer-0 output still comes from the Japan prompt.

**Where the plateau is.** At the country token the top next token is ` is` at all 41 points (probability
0.69–0.76), and the next-token distributions of the two endpoints differ by only 0.033 bits of Jensen–Shannon
divergence, against 0.673 bits after the bridge. The hidden state nevertheless develops a boundary. The table
shows that the path moves at constant speed in the early layers and that from about layer 20 the movement
concentrates near `t = 0.51`: at layer 26 the largest step is 2.65 times the mean and the ten outermost steps
average 0.44 times the mean. In layers 0–7 the largest step is the first or last one and there is no central
boundary. We treat layers 20–27 as plateau-bearing; layer 26 has the sharpest boundary and
is used for the path comparison in Section 3. This is a moderate plateau: the 10–90 width shrinks from 0.80 to
0.64.

| Layer | Width 10–90 | Peak / mean step | Mean of 10 outer steps / mean | Boundary `t` | `t` where `d = 0.5` | `t` where readout = 0 | Readout held-out accuracy | Smallest signed held-out score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.80 | 1.01 | 1.00 | 0.988 | 0.503 | 0.510 | 1.00 | 0.93 |
| 4 | 0.83 | 1.28 | 1.09 | 0.013 | 0.462 | 0.487 | 1.00 | 0.86 |
| 8 | 0.82 | 1.12 | 0.95 | 0.562 | 0.498 | 0.505 | 1.00 | 0.86 |
| 12 | 0.78 | 1.43 | 0.79 | 0.512 | 0.514 | 0.508 | 1.00 | 0.83 |
| 16 | 0.77 | 1.40 | 0.79 | 0.512 | 0.517 | 0.506 | 1.00 | 0.84 |
| 18 | 0.76 | 1.58 | 0.72 | 0.537 | 0.523 | 0.517 | 1.00 | 0.85 |
| 20 | 0.74 | 1.81 | 0.66 | 0.512 | 0.536 | 0.529 | 1.00 | 0.86 |
| 22 | 0.70 | 2.13 | 0.55 | 0.512 | 0.523 | 0.512 | 1.00 | 0.78 |
| 23 | 0.67 | 2.36 | 0.50 | 0.512 | 0.518 | 0.508 | 1.00 | 0.81 |
| 24 | 0.67 | 2.38 | 0.50 | 0.512 | 0.521 | 0.505 | 1.00 | 0.80 |
| 25 | 0.67 | 2.42 | 0.49 | 0.512 | 0.522 | 0.509 | 1.00 | 0.81 |
| 26 | 0.64 | 2.65 | 0.44 | 0.512 | 0.515 | 0.501 | 1.00 | 0.75 |
| 27 | 0.65 | 2.61 | 0.45 | 0.512 | 0.513 | 0.497 | 1.00 | 0.76 |

**After the bridge.** P(` Tokyo`) stays near 0.40 until `t = 0.40` and is below 0.01 from `t = 0.60`. The margin
crosses zero at `t = 0.531`. The top token is ` Tokyo` up to `t = 0.500`, ` located` at 0.525 and 0.550, and
` Berlin` from 0.575. The hidden-state boundary (0.512), the readout crossing (0.501 at layer 26) and the answer
boundary (0.531) therefore lie within about one grid step (0.025) of each other. This coincidence is an
observation about the unedited model; it cannot say which of the two the plateau follows.

**Readout validation.** Training prefixes come in four families of four templates each (currency, language,
identity, general), for example `The official currency of {country}`, `People in {country}`,
`I was born in {country}`, `The flag of {country}`. Leaving one family out and scoring it with a readout
fitted on the other three gives the right sign for every held-out prompt at every layer (accuracy 1.00), with
a smallest signed score of 0.74. On the unedited capital prompts, which the readout never saw, the Japan
prefixes score +0.89 to +1.11 and the Germany prefixes −0.97 to −1.25 across layers. The readout is invalid at
the first token of a prompt: there it returns about +1.3 for both countries.

To show where the plateau sits before any edit, Figure 1 plots the adjacent-step change in every layer, the
distance curves at three layers, and the next-token probabilities along the path.

![Baseline plateau: step heatmap, distance curves and probabilities](plots/fig1_baseline_plateau.png)

**Figure 1.** Unedited path. (a) Adjacent-step change at the country token divided by that layer's mean
step; x: interpolation position `t`, y: layer. Brighter cells mean faster movement. (b) Relative distance
`d(t)` at the country token for layer 0 (dotted, circles), layer 16 (dashed, squares) and layer 26 (solid,
triangles). (c) Probability of ` Tokyo` (solid, circles) and ` Berlin` (dashed, squares) after the bridge, and
of ` is` at the country token (dotted, triangles).

## 2. Finding an edit that changes the answer

### 2.1 Scan on the tuning pair

Features were ranked on the tuning pair `The capital city of {country} is` by a first-order estimate of how
much moving each one to the donor's value would shift the margin (gradient of the margin with respect to the
MLP output, projected on the feature's decoder direction, times the donor-minus-recipient activation). The
top 1, 5 or 10 features with a helpful estimate were then really patched at `beta` 0, 0.5, 1 and 2, in both
directions, at every layer. The table gives the best single layer for each site. "Shift" is the change in
margin toward the donor's capital; a shift larger than the baseline margin (9.53 for the Japan prompt, 9.17
for the Germany prompt) reverses the preference.

| Site | Pushed toward | Best layer at `beta = 1` | Shift at `beta = 1` | Best layer at `beta = 2` | Shift at `beta = 2` | Settings that reverse |
|---|---|---:|---:|---:|---:|---|
| ` is` token (positive control), layers 0–27 | Berlin | 22 | 4.38 | 22 | 7.77 | none |
| ` is` token (positive control), layers 0–27 | Tokyo | 23 | 4.39 | 23 | 8.20 | none |
| country token, layers 0–26 | Berlin | 16 | 7.06 | 16 | 16.04 | layer 16, 5 or 10 features, `beta = 2` |
| country token, layers 0–26 | Tokyo | 16 | 4.13 | 16 | 12.95 | layer 16, 10 features, `beta = 2` |
| country token, two adjacent layers | Berlin | 16+17 | 7.38 | 16+17 | 15.99 | pairs containing layer 16, `beta = 2` |
| country token, two adjacent layers | Tokyo | 16+17 | 8.07 | 16+17 | 15.97 | pairs containing layer 16, `beta = 2` |

Three things follow from the scan.

- **No natural-strength edit reverses the answer.** At `beta <= 1` no setting at either site changes the sign
  of the margin, including the two-layer groups.
- **The positive control is only partial.** Editing at the ` is` token, where the answer is produced, moves
  the margin by at most 4.4 logits at `beta = 1` and never reverses it. These transcoders reconstruct the MLP
  output with a relative error of 0.21–0.71 at the country token (median 0.49) and 0.22–0.80 at the ` is` token
  (median 0.64), and only about 7 features are active per layer at the country token, so ten features at one
  layer give limited leverage.
- **Layer 16 at the country token stands out.** Its shift at `beta = 1` is 7.06 and 4.13; the next best single
  layers reach 2.77 (layer 7) and 1.89 (layer 17). All four settings that reverse the margin in both directions
  include layer 16 and use `beta = 2`.

### 2.2 The frozen edit

The selection rule was written down before the scan (JOURNAL.md): country token, layers up to 26, margin
reversed in both directions at the same setting, natural strength preferred, then the fewest features. It
selects **layer 16, top 10 features, `beta = 2`**. This is an extrapolated edit: its L2 norm is 60.8
(Berlin-directed) and 50.4 (Tokyo-directed) on the original prompts, against a native layer-16 MLP output
norm of 31–34 at that token. Features, donor activations and strength were frozen at this point.

The table lists the frozen features. The last two columns count on how many of the 16 non-capital readout
prefixes per country each feature is active at the country token. The features that carry most of the
estimated effect are active on every prefix of one country and on none of the other: 126668, 94001 and 50675
for Germany; 121007, 139039, 23125, 115996 and 52503 for Japan. They respond to the country word whether or
not the prompt is about a capital.

| Pushed toward | Rank | Layer-16 feature | Activation in recipient | Activation in donor | Estimated margin shift | Active on Japan prefixes | Active on Germany prefixes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Berlin | 1 | 126668 | 0.00 | 16.52 | 0.904 | 0/16 | 16/16 |
| Berlin | 2 | 94001 | 0.00 | 13.17 | 0.814 | 0/16 | 16/16 |
| Berlin | 3 | 50675 | 0.00 | 7.47 | 0.231 | 0/16 | 16/16 |
| Berlin | 4 | 121007 | 9.79 | 0.00 | 0.203 | 16/16 | 0/16 |
| Berlin | 5 | 9693 | 0.07 | 2.54 | 0.034 | 6/16 | 7/16 |
| Berlin | 6 | 23125 | 6.06 | 0.00 | 0.028 | 16/16 | 0/16 |
| Berlin | 7 | 115996 | 6.06 | 0.00 | 0.027 | 16/16 | 0/16 |
| Berlin | 8 | 13235 | 10.83 | 11.54 | 0.006 | 16/16 | 16/16 |
| Berlin | 9 | 139039 | 7.06 | 0.00 | 0.002 | 16/16 | 0/16 |
| Berlin | 10 | 70797 | 0.00 | 0.56 | 0.001 | 2/16 | 1/16 |
| Tokyo | 1 | 121007 | 0.00 | 9.79 | 0.893 | 16/16 | 0/16 |
| Tokyo | 2 | 139039 | 0.00 | 7.06 | 0.568 | 16/16 | 0/16 |
| Tokyo | 3 | 23125 | 0.00 | 6.06 | 0.191 | 16/16 | 0/16 |
| Tokyo | 4 | 115996 | 0.00 | 6.06 | 0.191 | 16/16 | 0/16 |
| Tokyo | 5 | 50675 | 7.47 | 0.00 | 0.146 | 0/16 | 16/16 |
| Tokyo | 6 | 52503 | 0.00 | 3.96 | 0.081 | 16/16 | 0/16 |
| Tokyo | 7 | 126668 | 16.52 | 0.00 | 0.043 | 0/16 | 16/16 |
| Tokyo | 8 | 108301 | 3.68 | 2.16 | 0.034 | 12/16 | 14/16 |
| Tokyo | 9 | 9693 | 2.54 | 0.07 | 0.018 | 6/16 | 7/16 |
| Tokyo | 10 | 4835 | 1.70 | 3.52 | 0.008 | 12/16 | 12/16 |

### 2.3 Capital prompts under the frozen edit

On the reserved prompts the frozen edit reverses the margin in all four cases and makes the other capital the
top token, with fluent continuations. The three random controls of the same size never reverse it. P(` is`) at
the country token is preserved (0.69 to 0.73 and 0.75 to 0.69). The readout at layer 26 moves with the answer:
from +1.05 to −0.84 on the Japan prompts and from −1.21 to +0.09 on the Germany prompts, while the controls
stay between +0.72 and +1.13, and between −1.14 and −0.86. The two reserved templates share the prefix
`The capital of {country}`, so their country-token columns are identical. `beta = 0` and removing the hooks
reproduce the unedited outputs exactly.

| Prompt | Role | Pushed toward | Margin, no edit | Margin, edit | Margin, 3 controls | Top token (probability), edit | P(` is`), no edit → edit | Readout L26, no edit → edit | Readout L26, 3 controls | Greedy continuation, edit |
|---|---|---|---:|---:|---|---|---|---|---|---|
| `The capital city of Japan is` | tuning | Berlin | +9.53 | −6.52 | +7.05 to +9.36 | `a` (0.09) | 0.41 → 0.51 | +1.06 → −0.90 | +0.74 to +1.07 | `a city that has been continuously developing for` |
| `The capital city of Germany is` | tuning | Tokyo | −9.17 | +3.78 | −8.91 to −6.92 | `Tokyo` (0.13) | 0.60 → 0.50 | −1.20 → +0.16 | −1.13 to −0.87 | `Tokyo, and the capital city of France` |
| `The capital of Japan is` | reserved (original) | Berlin | +10.15 | −6.61 | +7.96 to +10.44 | `Berlin` (0.28) | 0.69 → 0.73 | +1.05 → −0.84 | +0.72 to +1.13 | `Berlin, and the capital of France is` |
| `The capital of Germany is` | reserved (original) | Tokyo | −10.23 | +3.32 | −9.73 to −6.85 | `Tokyo` (0.27) | 0.75 → 0.69 | −1.21 → +0.09 | −1.14 to −0.86 | `Tokyo, and the capital of the United` |
| `The capital of Japan is the city of` | reserved | Berlin | +11.58 | −6.69 | +9.68 to +11.12 | `Berlin` (0.34) | 0.69 → 0.73 | +1.05 → −0.84 | +0.72 to +1.13 | `Berlin. The capital of the United States` |
| `The capital of Germany is the city of` | reserved | Tokyo | −11.01 | +2.34 | −10.02 to −8.84 | `Tokyo` (0.27) | 0.75 → 0.69 | −1.21 → +0.09 | −1.14 to −0.86 | `Tokyo, and the capital of the United` |

The next table varies the strength of the same frozen feature set on the reserved original pair. Margin and
readout move together at every strength: at `beta = 1` the margin has covered about a third (Berlin-directed)
to a fifth (Tokyo-directed) of the distance to the other country's unedited margin, and the readout a similar
fraction of the distance to the other country's score.

| Pushed toward | `beta` | Edit L2 | Margin, no edit | Margin, edit | Margin, 3 controls | Top token, edit | Readout L26, no edit | Readout L26, edit | Readout L26, 3 controls |
|---|---:|---:|---:|---:|---|---|---:|---:|---|
| Berlin | 0.5 | 15.2 | +10.15 | +8.29 | +9.75 to +10.36 | `Tokyo` | +1.05 | +0.81 | +1.00 to +1.08 |
| Berlin | 1.0 | 30.4 | +10.15 | +3.64 | +9.29 to +10.52 | `Tokyo` | +1.05 | +0.33 | +0.94 to +1.10 |
| Berlin | 2.0 | 60.8 | +10.15 | −6.61 | +7.96 to +10.44 | `Berlin` | +1.05 | −0.84 | +0.72 to +1.13 |
| Tokyo | 0.5 | 12.6 | −10.23 | −8.69 | −10.31 to −9.80 | `Berlin` | −1.21 | −1.07 | −1.22 to −1.13 |
| Tokyo | 1.0 | 25.2 | −10.23 | −5.81 | −10.25 to −9.12 | `Berlin` | −1.21 | −0.79 | −1.21 to −1.03 |
| Tokyo | 2.0 | 50.4 | −10.23 | +3.32 | −9.73 to −6.85 | `Tokyo` | −1.21 | +0.09 | −1.14 to −0.86 |

To show both the scan and the effect of strength at a glance, Figure 2 plots the margin shift per layer for
the two sites and the margin against `beta` for the frozen edit and its controls.

![Scan of margin shifts by layer and effect of edit strength](plots/fig2_intervention.png)

**Figure 2.** (a, b) Tuning pair, top-10 features, `beta = 1`. x: edited layer; y: margin shift toward the
donor's capital in logits. Solid line with circles: edit at the country token. Dashed line with squares: edit
at the ` is` token. The dotted horizontal line is the shift needed to reverse the answer. (c) Reserved original
pair, frozen layer-16 feature set. x: edit strength `beta`; y: margin toward the donor's capital, so values
above the dotted zero line mean the other capital is preferred. Filled circles on a solid line: Berlin-directed
edit on the Japan prompt. Filled squares on a dashed line: Tokyo-directed edit on the Germany prompt. Open
markers: mean of the three random controls for the matching edit, with bars from the smallest to the largest
control.

### 2.4 Other country attributes under the frozen edit

The same frozen edit was applied at the country token of the currency and language prompts that Qwen answers
correctly (from dir30), pushing each prompt toward the other country. On all six prompts where the country
word sits mid-sentence, the top answer switches to the other country's attribute (`German mark`, `German`,
`Japanese`) and the layer-26 readout changes sign. None of the 18 control runs produces the other country's
attribute; 16 keep the original top token and two (one seed, the two Japan language prompts) change it to `a`
and `English`. On the four prompts that start with the country word the edit has no effect. Those four are
uninformative: the first token has a very large residual norm in this model, dir30 found that the transcoders
fail there, and the readout is invalid at that position.

| Prompt | Country token position | Pushed toward | P(correct answer), no edit → edit | Top token, edit | Greedy continuation, edit | Top token, 3 controls | Readout L26, no edit → edit |
|---|---|---|---|---|---|---|---|
| `The official currency of Japan is the` | mid | Berlin | `yen` 0.85 → 0.01 | `German` | `German mark, which is the same as` | `yen`, `yen`, `yen` | +0.94 → −0.89 |
| `What is the official currency of Japan? It is called the` | mid | Berlin | `yen` 0.38 → 0.00 | `German` | `German mark. What is the official currency` | `yen`, `yen`, `yen` | +0.82 → −0.84 |
| `People in Japan speak` | mid | Berlin | `Japanese` 0.36 → 0.00 | `German` | `German, but they are not German.` | `a`, `Japanese`, `Japanese` | +0.99 → −0.47 |
| `In Japan, the primary language is` | mid | Berlin | `Japanese` 0.72 → 0.00 | `German` | `German, and the primary language of the` | `English`, `Japanese`, `Japanese` | +1.05 → −0.73 |
| `People in Germany speak` | mid | Tokyo | `German` 0.58 → 0.09 | `Japanese` | `Japanese as a second language, and Japanese` | `German`, `German`, `German` | −0.81 → +0.38 |
| `In Germany, the primary language is` | mid | Tokyo | `German` 0.86 → 0.18 | `Japanese` | `Japanese, and the primary language of the` | `German`, `German`, `German` | −1.03 → +0.18 |
| `Japan's official currency is the` | first | Berlin | `yen` 0.89 → 0.88 | `yen` | `yen, and the official currency is the` | `yen`, `yen`, `yen` | invalid at first token |
| `Japan's official currency is called the` | first | Berlin | `yen` 0.34 → 0.34 | `yen` | `yen. The yen is a currency used` | `yen`, `yen`, `yen` | invalid at first token |
| `Japan, where most people speak` | first | Berlin | `Japanese` 0.44 → 0.45 | `Japanese` | `Japanese, is a country that has a` | `Japanese`, `Japanese`, `Japanese` | invalid at first token |
| `Germany, where most people speak` | first | Tokyo | `German` 0.62 → 0.64 | `German` | `German, and the country is known for` | `German`, `German`, `German` | invalid at first token |

**Gate.** The plan asks for an early edit that changes the answer while an independent country readout stays
stable. The frozen edit changes the answer, the readout and two other country attributes together, so it is a
country swap. The plan's stop condition ("effective early edits also change country identity") applies.

## 3. The path under the frozen edit

The path was still rerun under the frozen edit, because it shows directly what "not disentangled" looks like.
The same 41 points were run with no edit, the Tokyo-directed edit, the Berlin-directed edit, and the three
random controls for each. Feature activations are recomputed at every `t`; nothing is reselected. Distances use
the unedited `t = 0` and `t = 1` states as fixed references.

**Answer.** With no edit the margin crosses zero at `t = 0.531`. Under the Tokyo-directed edit the margin stays
positive at every `t` (10.11 down to 3.31; P(` Tokyo`) 0.41 down to 0.27). Under the Berlin-directed edit it
stays negative at every `t` (−6.52 to −10.18; P(` Berlin`) between 0.22 and 0.47). Neither has an answer
boundary. The six controls keep it, at `t` = 0.503 to 0.538. P(` is`) at the country token stays between 0.65
and 0.79 under both edits.

**Readout and hidden state.** The table gives, for the country token at layers downstream of the edit, the
largest adjacent-step change, its size relative to the unedited path, where it occurs, and the readout at both
ends. Under either edit the readout no longer crosses zero, and the largest step falls to 34–41% of its
unedited size at layers 23–27. The controls leave both in place: their largest step is 89–128% of the unedited
one at layers 23–27, at `t` = 0.46–0.51, and their readout crosses zero at `t` = 0.46–0.51.

| Layer | Condition | Largest step (L2) | Relative to no edit | Peak / mean | Boundary `t` | Readout at `t = 0` | Readout at `t = 1` | `t` where readout = 0 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 17 | no edit | 2.9 | 1.00 | 1.59 | 0.512 | +1.09 | −1.13 | 0.513 |
| 17 | Tokyo-directed | 2.3 | 0.80 | 1.48 | 0.537 | +1.06 | +0.57 | never |
| 17 | Berlin-directed | 2.6 | 0.90 | 1.59 | 0.512 | −1.05 | −1.15 | never |
| 20 | no edit | 6.3 | 1.00 | 1.81 | 0.512 | +1.05 | −1.10 | 0.528 |
| 20 | Tokyo-directed | 3.5 | 0.55 | 1.48 | 0.537 | +1.03 | +0.59 | never |
| 20 | Berlin-directed | 4.3 | 0.68 | 1.60 | 0.512 | −0.80 | −1.10 | never |
| 23 | no edit | 13.1 | 1.00 | 2.36 | 0.512 | +1.04 | −1.14 | 0.507 |
| 23 | Tokyo-directed | 4.9 | 0.37 | 1.46 | 0.662 | +1.04 | +0.32 | never |
| 23 | Berlin-directed | 5.3 | 0.40 | 1.52 | 0.512 | −0.85 | −1.15 | never |
| 26 | no edit | 16.0 | 1.00 | 2.65 | 0.512 | +1.05 | −1.21 | 0.501 |
| 26 | Tokyo-directed | 5.5 | 0.34 | 1.52 | 0.662 | +1.04 | +0.09 | never |
| 26 | Berlin-directed | 6.2 | 0.39 | 1.58 | 0.512 | −0.84 | −1.22 | never |
| 26 | Tokyo-matched controls (3) | 14.3–18.4 | 0.90–1.15 | 2.20–2.61 | 0.487–0.512 | +1.02 to +1.06 | −0.86 to −1.13 | 0.462–0.503 |
| 26 | Berlin-matched controls (3) | 17.9–19.8 | 1.12–1.24 | 2.46–2.68 | 0.512 | +0.72 to +1.13 | −1.19 to −1.22 | 0.492–0.507 |
| 27 | no edit | 17.4 | 1.00 | 2.61 | 0.512 | +1.03 | −1.25 | 0.496 |
| 27 | Tokyo-directed | 6.0 | 0.34 | 1.52 | 0.662 | +1.03 | +0.07 | never |
| 27 | Berlin-directed | 6.5 | 0.37 | 1.52 | 0.512 | −0.85 | −1.26 | never |

**What remains.** A smaller peak survives under both edits (5.5 and 6.2 at layer 26, against 2.6–3.4 at the
ends of the path). It sits at `t = 0.512` under the Berlin-directed edit and at `t = 0.662` under the
Tokyo-directed edit. Under these edits neither the answer nor the readout crosses, so this remainder cannot be
assigned to either explanation. The output distribution still drifts: under the Berlin-directed edit P(` Berlin`)
dips from 0.28 to 0.22 near `t = 0.40` and then rises to 0.47.

**Absolute positions.** The relative distance `d(t)` alone can mislead under an edit, so we also report raw
distances at layer 26, where the two unedited endpoints are 177.3 apart. At `t = 0` the Berlin-directed edit
puts the state 163.6 from the unedited Japan state and 85.9 from the unedited Germany state. A random control
of the same size puts it 123.6 from Japan and 197.3 from Germany. At `t = 1` the Tokyo-directed edit puts the
state 94.9 from Japan and 118.3 from Germany, and its control 186.0 and 111.8. The real edits move the state
toward the other country's state without reaching it; the controls move it a comparable distance in an
unrelated direction.

**The ` is` token.** The hidden state of the ` is` token shows the same pattern. Its largest step at layer 26
is 22.8 with no edit, 6.0 under the Tokyo-directed edit, 5.2 under the Berlin-directed edit, and 18.8–26.1
under the six controls.

Figure 3 puts the three quantities on a common `t` axis: the layer-26 hidden-state curves, the readout, and
the answer probabilities.

![Path under no edit, both directed edits and random controls](plots/fig3_path_comparison.png)

**Figure 3.** Japan-to-Germany path, x: interpolation position `t` in every panel. Solid line with circles: no
edit. Dashed line with squares: Tokyo-directed edit. Dash-dot line with triangles: Berlin-directed edit. Thin
lines without markers: random controls, dotted for those matched to the Tokyo-directed edit and long-dashed
for those matched to the Berlin-directed edit. Panel 1a: relative distance `d(t)` of the layer-26 country-token state,
with unedited references. Panel 1b: adjacent-step L2 change of the same state. Panel 2a: country readout at
layer 26 (+1 Japan, −1 Germany). Panel 2b: L2 norm of the vector added at layer 16; each control has the same
norm as its edit by construction, so those lines coincide. Panels 3a and 3b: probability of ` Tokyo` and of
` Berlin` after the bridge.

## 4. Verdict against the plan's table

The plan fixed four possible readings in advance. The table states which one the data match.

| Planned observation | Seen? | Evidence |
|---|---|---|
| Readout stays stable; answer preference and plateau boundary move together | No | Readout flips under the edit (Sections 2.3, 2.4) |
| Answer changes but plateau does not | No | Largest step falls to 34–41% at layers 23–27 (Section 3) |
| One city stays preferred but the plateau still tracks the country switch | No clean case | A reduced peak remains, but the readout does not cross either (Section 3) |
| Country and answer both change, or controls have comparable effects | **Yes: both change** | Margin, readout, currency and language answers all switch; controls do not (Sections 2.3, 2.4, 3) |

## Files

- `results/s1_endpoints.json`, `s1_layers.csv`, `s1_path.npz`, `probe.npz`: baseline answers, per-layer plateau
  summary, raw path curves, readout weights and validation.
- `results/s2_slices.csv`, `slices/layer_<L>.pt`: active-feature counts, reconstruction errors and the
  encoder/decoder rows used.
- `results/s2_ranking.csv`, `s2_scan.csv`, `frozen_edit.json`, `frozen_features.csv`: feature ranking, full
  scan, the frozen edit and its features.
- `results/s2_frozen_eval.csv`, `s2_context.csv`: capital and context prompts under the edit and controls.
- `results/s3_path.npz`, `s3_summary.csv`: raw path curves and per-layer summaries for all nine conditions,
  for both the country token and the ` is` token.
- `results/run_config.json`: pinned revisions, settings and the selection rule.
- Rerun: `bash experiments/run_all.sh`.
