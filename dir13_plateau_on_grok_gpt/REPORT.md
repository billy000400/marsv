# REPORT — Do Grokking and Matthew-style activation plateaus emerge together?

> Final, presentable, current-best only (history in CHANGELOG.md).

## Summary

Matthew Shinkle & StefanHex's post *Activation Plateaus: Where and How They Emerge* reports a striking
geometry inside trained transformers: take two inputs, interpolate between their internal activations,
and the network's output does **not** morph gradually. Instead it stays locked to the first input's
output, snaps across a narrow boundary, and locks to the second input's output — a
**plateau–boundary–plateau** curve. If real, this matters for safety-relevant interpretability: it
means the network's computation is organized into discrete basins, so activation-space edits
(steering, patching) behave predictably inside a basin and abruptly across one, and jailbreak- or
backdoor-style behavior switches may live at such boundaries.

This direction asks the cheap gating question for one specific model: does the **12-layer, 12-head
character-level Shakespeare GPT** from *Deep Networks Always Grok and Here is Why* (Figure 9) show
Matthew-style plateaus? The paper's GPT code and checkpoint are not public, so we trained a faithful
reconstruction (next-char accuracy 0.56 ≈ 37× chance) and ran the two-natural-endpoint interpolation
assay with everything frozen before any curve was inspected. One scope note up front: the grok
paper's own headline phenomenon is **grokking** — `ε=0.03`-PGD adversarial robustness emerging long
after training accuracy saturates, alongside a **second local-complexity descent**. We treat this as
an explicit **validity gate** (Methods §Figure-9 gate, Results §Figure-9 gate). **All three models we
trained FAIL this gate** within budget: the existing 3,500-step pilot, a fresh 30k-step character run,
and a fresh BPE run (stopped at 10k steps once it was plainly memorising) each show a *first*
local-complexity descent and emerging `ε=0.03`-PGD
robustness, but **no second descent** — the ordering that defines grokking. The consequence is the
**bounded relationship verdict = "primary relationship not testable"** (PLAN case 5): because no run —
in particular the BPE run needed for Matthew's exact `big/in`, `big/large` tokens — reproduces
Figure 9, we cannot test a Matthew-exact Grokking↔plateau relationship on a grokking model. The plateau
phenomenon itself is nonetheless **present** in the character reconstruction and stands on its own
(below), not as evidence about grokking. The paper's role here is only to specify the model under
test; the phenomenon under test is Matthew's activation plateaus.

**Result: plateaus are present, and we can time their emergence.** Each interpolation curve plots the
output's relative closeness to endpoint B (call it $d$, from 0 = "still A's output" to 1 = "B's
output"; defined precisely in Methods) against the interpolation step $t$; the no-plateau reference is
the **diagonal** $d = t$ (transition width 0.8 — no flat plateau segments). Our **primary plateau
evidence** runs Matthew's own code path with his exact context and two preregistered single-token
character controls (`b↔i`, `b↔l`) across six frozen training checkpoints. The plateau is **absent at
initialization** (curve is the diagonal, width ≈ 0.80) and **emerges during training**: by step ~831
it is a sharp plateau–boundary–plateau sigmoid (width ≈ 0.33) and stays there to step 30k. Crucially
that emergence happens during the model's **first** local-complexity descent and initial accuracy
rise, and is **fully formed before** adversarial robustness saturates — so in this (non-grokking)
model the plateau tracks *initial fit*, not the (absent) grokking transition. A larger **exploratory**
40-pair natural-minimal-pair sweep (labelled as such, out of the headline per PLAN scope) corroborates
the two structural signatures: **(1)** reading the curve at successively deeper layers moves it
monotonically *away* from the diagonal (the boundary sharpens layer by layer), and **(2)** moving the
interpolation point later collapses it back *onto* the diagonal. A further sweep, requested by an
operator, fixes one endpoint at the **comma** and interpolates to **all 64 other characters**: no pair
responds linearly (median width 0.340 vs the diagonal's 0.80), but only 1/64 clears the strict
≤ 0.25 plateau bar, and the sharpness of the switch tracks how likely the model thinks that character
is in the context (rank correlation −0.74). Because all of that came from one shared context, we
repeated the sweep in **8 further contexts** from held-out text (576 pairs): the shape result
replicates exactly — **0/576** curves are near-linear and per-context median widths stay in the band
0.313–0.436 — while the probability effect replicates only in **direction** (negative in 9/9
contexts, sign test p = 0.004) with a much more modest typical size (median ρ = −0.41). **Verdict:
plateaus are real in this model** — qualified, because we tested a reconstruction rather than the
paper's exact checkpoint, and because the sharpness is graded rather than step-like.

## Methods

### Data & Model

- **Task/data.** Next-character prediction on **Tiny Shakespeare** (`input.txt`, 1,115,394 chars,
  SHA-256 `86c4e6…565ed`); first 90% train, last 10% validation; character-level tokens (vocab 65).
- **Model.** A nanoGPT-style decoder-only GPT: **12 blocks, 12 heads, GeLU MLPs** (the paper's
  confirmed Figure-9 facts), pre-norm, learned positions, weight-tied head. Reconstruction choices
  (unspecified by the paper): `d_model = 240`, MLP hidden `4·d_model`, context 128, dropout 0.2,
  8.38M params. Every field is tagged confirmed-vs-reconstructed in `MODEL_SPEC.md`.
- **Why a reconstruction.** The official repo `AhmedImtiazPrio/grok-adversarial` (audited via the
  GitHub API, 2026-07-15) contains no GPT/Shakespeare code or checkpoint. All conclusions are
  explicitly about this reconstruction (plan success-criterion 3).
- **Training.** AdamW (betas 0.9/0.99, weight decay 0.1), peak LR 1e-3, 100-step warmup + cosine
  decay, batch 48×128, fp32, 3,500 steps → **val loss 1.494, val next-char accuracy 0.560**. Seeds
  and provenance in `results/train_meta.json`; curves in Figure 1.
- **Hook point.** We intervene on the **residual stream** — the running hidden vector that each
  transformer block reads from and adds to — at the **final sequence position**, after block $L$
  (`resid_post`). Because attention is causal, replacing only the final position's vector and
  re-running blocks $L{+}1..11$ is an exact continuation of the forward pass (verified below). The
  primary interpolation point is **block 0**, leaving 11 of 12 blocks downstream. **Logits** are the
  model's 65 raw pre-softmax output scores at the final position.
- **Sample sizes.** 40 frozen pairs × 101 interpolation steps; recording at 11 downstream residual
  points + final logits; depth comparison over interpolation blocks {0, 2, 4, 6, 8, 10}.

### Constructing natural minimal pairs (frozen before any curve was seen)

The question is about interpolating between **two natural activations**, so each pair must be two
real, plausible inputs whose activations differ as little as possible — we use equal-length sequences
`prefix + char_A` vs `prefix + char_B`, identical except the final input character. Selection never
looks at interpolation curves (that would bias the frozen set toward or away from plateaus):

- 40 shared prefixes of length 127 sampled from held-out validation text (seed 20260717,
  deduplicated), giving full sequences of the model's context length 128.
- `char_A` = the character actually observed after the prefix in the corpus (guaranteed natural);
  `char_B` = the model's highest-probability next character, or its second if the top choice equals
  `char_A` (both endpoints plausible; median model probability of `char_B` is 0.146). This is
  option 2 of the plan; option 1 (two continuations of the same 127-char prefix observed in text) is
  infeasible — such long prefixes are unique in a 1.1M-char corpus.
- **Degeneracy exclusion (frozen threshold):** a pair would be dropped only if its two endpoint logit
  vectors were numerically indistinguishable (L2 distance < 1e-3). None were: endpoint distances span
  8.7–64.4 (median 24.7). All pair metadata is in `results/prompt_pairs.json`.

### Matthew-faithful character-token controls across training (primary plateau assay)

The **primary** plateau evidence follows Matthew's released config/code path
(`experiments/run_matthew_ckpts.py`, `configs/matthew_char_control.yaml`) so it transfers his assay
with only the model adapter changed: shared context `"The house was"`, **exactly 50** evenly spaced
interpolation values including both endpoints, `slerp_rescale` (spherical direction + linear norm;
same equations below), patch **only the final sequence position**, and sweep **every** interpolation
layer (`resid_post` blocks 0–11), recording Matthew's downstream hooks (`attn_out`, `resid_mid`,
`mlp_post`, `mlp_out`, `resid_post`) plus final logits. Because the character model cannot represent
Matthew's `big/in/large` as single tokens, we use his two preregistered single-**character** controls
`b↔i` and `b↔l` (labelled tokenizer controls, *not* replications of his word examples). We run them at
**6 checkpoint phases frozen before any plateau curve was inspected** (`experiments/freeze_phases.py`
→ `results/frozen_phases_char.json`; the Figure-9 LC curve is monotone so the rule falls back to
log-spaced picks): steps **0, 56, 831, 7819, 17500, 30000**. This lets us plot plateau width *against*
the Grokking metrics on one training-step axis (Results §Primary plateau evidence).

### Comma against every other character (operator-requested sweep)

The two controls above are only two pairs, so an operator asked whether the plateau holds when one
endpoint is held fixed and the other is swept over the whole alphabet. We fix endpoint A at the comma
and use every other character as endpoint B, giving **64 pairs** (`experiments/comma_sweep.py`).
Everything else is unchanged from the primary assay: the same fresh character GPT and its saved
checkpoints, shared context `"The house was "`, endpoint A = context + `,`, endpoint B = context +
one other character, **50** evenly spaced interpolation values, `slerp_rescale`, patch of the final
position only, and $d(t)$ read in final-logit space. Sample sizes: 64 pairs × 50 steps at
interpolation block 0 for each of the **6 frozen checkpoints** (steps 0, 56, 831, 7,819, 17,500,
30,000), plus 64 pairs × 50 steps at **every** interpolation block 0–11 at the final checkpoint. The
same metrics are reused — $d(t)$, the transition width $w_{10\to 90}$, and the frozen plateau rule —
so the numbers are directly comparable to the `b↔i`/`b↔l` controls.

Two extra quantities are measured at the final checkpoint, only to ask *why* some pairs switch more
sharply than others. The first asks whether sharpness tracks how ordinary the second character is in
this context. With $x_{ctx}$ the context `"The house was "` and $f$ the model, the model's
**next-character probability** for character $c$ is its softmax score at the final position:

```math
p(c)=\operatorname{softmax}\big(f(x_{ctx})\big)_c .
```

The second is a control against a trivial explanation — that flat curves merely mean the two
endpoints' outputs are hard to tell apart. **Endpoint separation** is the plain distance between the
two endpoint logit vectors $\ell_{,}$ (comma prompt) and $\ell_{c}$:

```math
s(c)=\lVert \ell_{,}-\ell_{c}\rVert_2 .
```

Both are related to width by the **Spearman rank correlation** $\rho$ — the ordinary correlation
computed on ranks rather than raw values, so it measures "does one go up when the other goes down?"
without assuming a straight-line relation. With $R_i$ and $S_i$ the ranks of the two quantities for
pair $i$ and $n=64$:

```math
\rho = 1-\frac{6\sum_{i=1}^{n}(R_i-S_i)^2}{n\,(n^2-1)} .
```

$\rho$ runs from −1 (perfect opposite ordering) through 0 (no monotone relation) to +1. These two
quantities are consumed by Results §"Comma against every other character" and by the context control
that follows it.

### Context control: the same sweep in eight further contexts

Every plateau number above is measured in the one shared context `"The house was "`, whose comma
endpoint is also an implausible continuation ($p = 1.0\times10^{-7}$). Both facts are candidate
confounds, so we repeat the whole comma sweep in **8 additional contexts** (`experiments/context_sweep.py`).
Contexts are 64-character windows sampled from held-out validation text (seed 20260725, 256
candidates), then chosen at nine evenly spaced ranks of $p(\texttt{,})$ — the model's probability of
a comma in that slot, from the same equation as above — so the set spans "a comma is impossible here"
to "a comma is almost certainly next". Adding the reference context gives 9 contexts × 64 pairs =
**576 pairs**, all at the final checkpoint (step 30,000), interpolation block 0, final logits, with
every other setting unchanged.

Two things are then asked of the data. First, **does the shape claim survive the change of context?**
— answered by the per-context width distribution and the count of near-linear curves. Second, **does
the width-vs-probability correlation replicate?** — answered by computing $\rho$ *within* each
context. Since nine correlations of varying strength cannot be summarized by their mean, we report
the median and range, and test only the direction with a **sign test**: under the null that sharpness
is unrelated to the model's probability, each context's $\rho$ is negative with probability $1/2$, so
observing $k$ negatives out of $n$ has two-sided p-value

```math
p = 2^{1-n}\sum_{j=k}^{n}\binom{n}{j} .
```

Finally, to test the implausible-endpoint worry directly, we correlate each context's **median width**
with its $p(\texttt{,})$ across the nine contexts. These are consumed by Results §"Does the plateau
depend on the context?".

### Spherical interpolation and patching

A straight line between two activations cuts through low-norm regions the model never produces, which
would confound "off-distribution activation" with "between two inputs". Following Matthew's post we
therefore **slerp** (spherically interpolate) the directions and linearly interpolate the norms: for
$t \in [0,1]$, with $\theta$ the angle between $h_A$ and $h_B$,

```math
\hat h(t)=\frac{\sin((1-t)\theta)}{\sin\theta}\,\frac{h_A}{\lVert h_A\rVert}
+\frac{\sin(t\theta)}{\sin\theta}\,\frac{h_B}{\lVert h_B\rVert},
\qquad
h(t)=\Big[(1-t)\lVert h_A\rVert+t\lVert h_B\rVert\Big]\,\hat h(t),
```

```math
\theta=\arccos\!\left(\frac{h_A^{\top} h_B}{\lVert h_A\rVert\,\lVert h_B\rVert}\right)
\quad\text{(cosine clamped to } [-1,1]\text{; if } \theta<10^{-4}\text{, fall back to normalized linear interpolation).}
```

Each $h(t)$ is patched into the final position of the block-$L$ residual stream (all earlier positions
untouched — they are identical between A and B anyway, verified below) and the remaining blocks are
run. We use 101 evenly spaced $t$ values including both endpoints, identical for all pairs.

### Metrics

**Relative distance $d(t)$** — *is the downstream output near endpoint A, near endpoint B, or in
between?* Raw distances are not comparable across pairs (endpoint separations vary 8.7–64.4), so we
use Matthew's normalized form, where $x(t)$ is the recorded downstream vector (final logits, or a
later block's final-position residual) and $x_A, x_B$ are the endpoints' vectors at the same point:

```math
d(t)=\frac{\lVert x(t)-x_A\rVert_2}{\lVert x(t)-x_A\rVert_2+\lVert x(t)-x_B\rVert_2}.
```

Read it as: $d \approx 0$ means "output still looks like A", $d \approx 1$ "like B". A
**plateau–boundary–plateau** curve hugs 0, crosses quickly, then hugs 1; a no-plateau response is
roughly the diagonal $d = t$. By construction $d(0)=0$ and $d(1)=1$. The raw individual curves are
the primary evidence (Figures 2–4).

**Transition width $w_{10\to 90}$** — *how narrow is the boundary?* Eyeballing 40 curves invites
cherry-picking, so we summarize each curve with one boundary-position-invariant scalar: the fraction
of the path over which $d$ climbs from 0.1 to 0.9,

```math
w_{10\rightarrow 90}=t(d=0.9)-t(d=0.1),
```

with the crossing points read off an **isotonic copy** of the curve (a least-squares monotone fit via
the pool-adjacent-violators algorithm) so that small non-monotonic wiggles cannot create spurious
crossings; plots always show the raw curve. Smaller is sharper. The diagonal scores $w = 0.8$; our
synthetic step curve scores 0.089. Consumed by the per-pair panel titles (Figure 2) and the layerwise
and depth tables/figures (Figures 3–4). Curves whose raw-vs-isotonic deviation exceeds 0.10 would be
reported separately as non-monotone and excluded from width statistics (none occurred: 0/40).

**Candidate-plateau rule (frozen).** A pair counts as a plateau iff $w_{10\to 90} \le 0.25$ **and**
the transition both starts after 10% and ends before 90% of the path ($t_{lo} \ge 0.10$,
$t_{hi} \le 0.90$ — i.e. the curve visibly rests near each endpoint) **and** the curve is
near-monotone (isotonic deviation ≤ 0.10). This yields the headline count `14/40` in Results.

### Baselines

**Diagonal (no-plateau) reference.** The straight line $d = t$ is what a downstream map that morphs
uniformly between the two outputs would produce; it scores $w_{10\to 90} = 0.8$. It is drawn as the
gray dashed reference line in every figure, and the depth-comparison test checks whether curves
collapse onto it.

**Synthetic calibration (assay unit test).** A synthetic step-like path (sharp sigmoid, boundary at
$t = 0.5$) must be detected as a narrow transition and a synthetic linear path must not:
measured $w = 0.089$ (detected) vs $w = 0.800$ (rejected). This shows the pipeline *can* find a
plateau if one exists and does not hallucinate one from a line.

### Figure-9 grokking gate (validity gate for any joint claim)

PLAN forbids joining the plateau result to a Grokking claim unless the model qualitatively reproduces
*Deep Networks Always Grok* Fig. 9. We measure Fig. 9's three quantities on log-spaced checkpoints with
a pipeline **source-locked** to the official repo (`experiments/fig9.py`; our forward reimplementation
matches the repo's to 0.0 logit error). **Data/model/layer:** the same reconstruction GPT, evaluated at
its saved checkpoints; local complexity is read from the 12 GeLU pre-activations.

**Local complexity (LC)** — *how many piecewise-linear regions does the network fold near the data?* For
each of the 12 GeLU layers we count, along short random line segments through the input, how many times
that layer's pre-activations change sign (a proxy for region boundaries crossed), and sum over layers.
With $N_{seg}$ segments of radius $r$ around a base point and $z_{\ell}(u)$ the layer-$\ell$
pre-activation at point $u$:

```math
\mathrm{LC} = \sum_{\ell=1}^{12} \mathbb{E}\big[\,\#\{\text{sign changes of } z_{\ell} \text{ along the segment}\}\,\big].
```

We report LC on 1,024 **train**, 1,024 **test**, and 1,024 **random** base points (`r=0.005`, `P=25`
samples per segment, 99% CIs) — the paper's defaults. Fig. 9's signature is a **second LC descent** that
begins before test accuracy peaks.

**Adversarial accuracy** — *does the model resist small input perturbations?* Next-token accuracy under an
`ε=0.03` `ℓ∞`-PGD attack in token-embedding space:

```math
\mathrm{adv\_acc} = \Pr\nolimits_{(x,y)}\Big[\ \arg\max \, f\big(x + \delta^\star\big) = y\ \Big],
\qquad \delta^\star = \arg\max_{\lVert\delta\rVert_\infty \le 0.03} \mathcal{L}\big(f(x+\delta), y\big).
```

Grokking = this rising **long after** clean accuracy saturates ("delayed robustness").

**Preregistered verdict rule** (`experiments/fig9_verdict.py`, applied identically to every run). Let the
LC-range tolerance be $\mathrm{tol} = 0.05\thinspace(\mathrm{LC}_{\max}-\mathrm{LC}_{\min})$. **PASS** iff test-LC
has an interior minimum then a rise above tol then a fall below tol (a genuine *second* descent) **and**
final adv accuracy $\ge 0.05$ **and** that descent's onset precedes the clean-accuracy peak. **NOT
ESTABLISHED** iff robustness never emerges *and* LC is still in its first monotone descent (or accuracy
still climbing) at the last checkpoint — the horizon was too short to decide. **FAIL** otherwise: valid
measurements at the planned horizon, but the Fig. 9 ordering is absent. This is the gate the pilot fails.

### Figure conventions

Every figure uses a colour-vision-deficiency-safe encoding (`experiments/cvd_style.py`): the
categorical palette is green-free, red-versus-green contrasts are never used, and **no series is
identified by colour alone** — each also carries a distinct linestyle, marker or hatch, which the
captions name. Continuous quantities use the `viridis` or `cividis` ramps, which stay monotone in
lightness and so remain readable in grayscale. Two reference lines recur: the gray dashed
no-plateau diagonal ($w = 0.8$) and the black dotted strict plateau bar ($w = 0.25$).

### Implementation checks (all passed before the full run)

- **Endpoint fidelity:** patched $t{=}0$ / $t{=}1$ forwards reproduce the direct unpatched forwards
  of A and B (max abs logit error < 1e-3), and $d(0) < 10^{-4}$, $d(1) > 1-10^{-4}$ for every pair.
- **Minimal-pair validity:** sequences differ only at the final character; the final-position patch
  is exact because all earlier-position activations of A and B match at every block (max abs
  difference < 1e-4).
- **Batching:** batched interpolation matches a single-example reference to < 1e-5.
- **Slerp:** endpoints reproduced exactly; interpolated norms linear; near-collinear fallback tested.

## Results

**Training.** Val loss 1.494 / accuracy 0.560 — a clearly trained network, not a random one.

![Figure 1 — Training curves for the pilot character GPT. Left panel: cross-entropy loss in nats (y) vs training step (x) for the train split (solid) and the validation split (dashed, square markers); validation loss falls to ≈1.49. Right panel: validation next-character accuracy (y) vs training step (x), rising to 0.56.](plots/training_curves.png)

**Figure-9 gate — pilot FAILs within its 3,500-step horizon.** Across 13 log-spaced checkpoints the
pilot's clean next-char accuracy climbs to 0.564 (peak at the last checkpoint) and `ε=0.03` PGD
adversarial accuracy rises to 0.327 — so *delayed robustness does emerge* — but test LC falls
monotonically from 1940 to its minimum (68) **at the final checkpoint**: there is **no second LC
descent**. Under the preregistered rule this is a **FAIL** (valid measurements, Fig. 9 ordering absent
within the tested horizon), not "not established", because robustness rose rather than staying flat.
The honest reading: the 3,500-step horizon ends inside the *first* LC descent, so the pilot cannot
support a joint Grokking↔plateau claim.

**Figure-9 gate — both fresh runs also FAIL (S4, S5).** We trained two matched fresh runs on a 30k-step
schedule (budget-capped below the paper's ~1e5; the BPE run was stopped at 10k steps once its
validation loss had been rising monotonically for 9k steps) and ran the identical
LC/PGD pipeline across log-spaced checkpoints. Both reproduce the pilot's pattern — emerging
robustness but no second LC descent:

| Figure-9 quantity | Pilot char (3.5k) | **Fresh char (30k)** | **Fresh BPE (10k)** |
|---|---|---|---|
| checkpoints evaluated | 13 | 14 | 10 |
| clean acc (peak / final) | 0.564 / 0.564 | 0.568 @ 4994 / 0.554 | 0.299 @ 831 / 0.274 |
| `ε=0.03` PGD adv acc (final) | 0.327 | **0.528** | **0.187** |
| test LC (first → min → final) | 1940 → 68 → 68 | 1940 → **8.1** → 8.1 | 2182 → **95** → 95 |
| LC minimum at… | last ckpt (3500) | last ckpt (30000) | last ckpt (10000) |
| second LC descent? | No | No | No |
| delayed robustness emerged? | Yes | Yes | Yes |
| **preregistered verdict** | **FAIL** | **FAIL** | **FAIL** |

The fresh character run is instructive: adversarial accuracy climbs *higher* than the pilot (0.53 vs
0.33), so "delayed robustness" is unambiguously present — yet test LC descends **monotonically** to its
minimum at the final checkpoint (8.1), never rising to produce a second descent. Robustness here tracks
the *first* fold-collapse of a memorising network, not the second-descent generalization event of
grokking. The BPE run behaves the same at a shorter horizon (LC still descending at 10k). Both fresh
runs **overfit**: validation loss bottoms early (character ≈step 3,750, BPE ≈step 750) then rises while
train loss keeps falling — the opposite of grokking's delayed val-loss recovery.

![Figure 1b — Pilot char (3.5k) Figure-9 curves. Left y-axis = local complexity (sign-crossing units summed over the 12 GeLU layers) for the train (solid), test (dashed) and random (dash-dot) base-point sets, each with a 99% CI band; right y-axis = next-token accuracy, plotted in black with circle markers for clean test accuracy and black dotted with square markers for ε=0.03 PGD adversarial accuracy; x-axis = training step (log scale, step 0 drawn at 1). LC descends monotonically to the horizon (no second descent) while adversarial accuracy rises to 0.33 — verdict FAIL.](plots/grokking_pilot_char.png)
![Fresh char (30k) Figure-9 gate, same axes and line styles as Figure 1b — LC monotone to 8.1, adversarial accuracy to 0.53, no second descent → FAIL.](plots/grokking_fresh_char.png)
![Fresh BPE (10k) Figure-9 gate, same axes and line styles as Figure 1b — LC monotone to 95, adversarial accuracy to 0.19, no second descent → FAIL.](plots/grokking_fresh_bpe.png)

**Joint checkpoint timeline and bounded relationship verdict (S7).** Putting all three runs on one
training-step axis (below), the Grokking side is uniform: LC falls monotonically and PGD robustness
rises, with **no second descent in any run**. Because the primary bridge to Matthew — a BPE model that
reproduces Figure 9 — does **not** pass the gate (and neither character run does either), the bounded
relationship verdict is **PLAN case 5: "primary relationship not testable."** We cannot claim the
plateau assay sharpens *during* a second-descent/robustness window, because no run exhibits that window.
What we *can* say standalone: the character reconstruction **does** show Matthew-style plateaus (next
sections), and both fresh runs develop adversarial robustness — but robustness and plateaus here are
properties of trained/memorising networks, not evidence of the specific grokking ordering.

![Figure 1d — Joint checkpoint timeline. Left panel: test local complexity (y) vs training step (x, log scale) for the pilot char run (dotted, triangles), the fresh char run (solid, circles) and the fresh BPE run (dashed, squares); each legend entry gives that run's Figure-9 gate verdict. Middle panel: ε=0.03 PGD adversarial accuracy (y) vs training step (x, log), same three line styles; the horizontal dashed line marks the 0.05 robustness threshold used by the verdict rule. Right panel: text summary of the three gate verdicts (all FAIL), the plateau-assay reference from the reconstruction, and the bounded relationship verdict. No run shows a second LC descent, so the joint verdict is "primary relationship not testable."](plots/joint_timeline.png)

**Primary plateau evidence: the Matthew-faithful char controls show the plateau emerging during
training (S6).** Running Matthew's code path (context `"The house was"`, 50-step slerp grid, full
interpolation-layer sweep) with the two frozen single-token controls `b↔i` and `b↔l` at the six frozen
checkpoint phases, the final-logit transition width at interpolation block 0 evolves as:

| training step | `b↔i` width | `b↔l` width | plateau? |
|---:|---:|---:|---|
| 0 (init) | 0.802 | 0.802 | no — diagonal |
| 56 | 0.771 | 0.814 | no — diagonal |
| 831 | 0.348 | 0.674 | forming |
| 7,819 | 0.364 | 0.326 | **yes** |
| 17,500 | 0.336 | 0.338 | **yes** |
| 30,000 | 0.331 | 0.330 | **yes** |

At init and step 56 the curve is the diagonal (width ≈ 0.80, no plateau); it collapses to a sharp
sigmoid (≈ 0.33) by step ~831 and holds flat to 30k. That collapse happens **during the first LC
descent and the initial clean-accuracy rise, and is fully formed before `ε=0.03` robustness saturates**
(steps ~10³–10⁴). So even though this model never groks, the plateau still appears — but tied to
*initial fit*, with no temporal coupling to a second-descent/robustness window (which never opens).
The depth control holds here too: at step 30000, `b↔i` widens 0.33 (block 0) → 0.72 (block 3) → 0.80
(block 11) as fewer downstream layers remain.

![Figure 2a — Matthew-faithful char-control d(t) (y), interpolation block 0, final logits, one panel per frozen checkpoint (steps 0→30000; x = interpolation step t within each panel). The b↔i pair is the solid line with circle markers, b↔l the dashed line with square markers, and the gray dashed straight line is the diagonal d = t. Both curves lie on the diagonal at init and step 56, and are sharp plateau–boundary–plateau sigmoids by step 831, stable thereafter.](plots/matthew_char_ctrl_by_checkpoint.png)

![Figure 2b — Grokking metrics vs plateau width on one timeline (fresh char run). Top: left y = local complexity for the train (solid), test (dashed) and random (dash-dot) base-point sets with 99% CI bands; right y = next-token accuracy, black with circles = clean, black dotted with squares = ε=0.03 PGD adversarial; x = training step (log). Bottom: transition width w_10→90 (y) for b↔i (solid, circles) and b↔l (dashed, squares) vs training step (log); the gray dashed line is the diagonal 0.8 and the black dotted line the plateau bar 0.25. Width hits its floor by step ~831 — during the first LC descent, before robustness rises.](plots/joint_timeline_char_ctrl.png)

**Comma against every other character: the shape holds for all 64 pairs, but sharpness is graded.**
Holding endpoint A at the comma and sweeping endpoint B over the other 64 characters at the final
checkpoint (interpolation block 0, final logits), the median transition width is **0.340**
(inter-quartile range 0.305–0.409), against 0.80 for a straight line. Every curve is monotone
(isotonic deviation exactly 0 for all 64) and every curve rests near both endpoints (the transition
starts at median $t = 0.252$ and ends at median $t = 0.603$; no pair starts before $t=0.10$ or ends
after $t=0.90$). Nothing is near the straight line: the widest pair is 0.665 (`3`), the narrowest
0.245 (`c`). But under the strict frozen rule only **1 of 64** pairs qualifies as a plateau
(33/64 pass at $w \le 0.35$, 52/64 at $w \le 0.45$). The two preregistered controls `b↔i` (0.331) and
`b↔l` (0.330) land exactly at this sweep's median — they were typical pairs, not favourable ones.

![Figure 6 — All 64 comma→character curves at step 30,000. Left: relative distance d(t) (y; 0 = output still looks like the comma prompt, 1 = looks like the other character's prompt) vs interpolation step t (x); one thin line per pair, shaded on the viridis scale by that pair's transition width (see colour bar); the thick black line is the median over the 64 pairs and the gray dashed line is the straight line d = t expected with no plateau. Right: histogram of transition width (x) against number of pairs (y); the black dotted vertical line marks the strict plateau rule 0.25, the gray dashed line the straight-line value 0.80, and the thick black line the median 0.34.](plots/comma_all_chars_curves.png)

The spread across characters is systematic rather than noisy: lower-case letters give the sharpest
switches (median width 0.313, n = 26), upper-case letters follow (0.355, n = 26), space and newline
sit between them (0.336, n = 2), and punctuation or the digit `3` are clearly the flattest (0.564,
n = 10).

![Figure 7 — Transition width (y) for each comma→character pair (x: one bar per character, sorted sharpest to flattest; ␣ = space, \n = newline) at the final checkpoint, interpolation block 0, final logits. Each character type has its own bar hatch as well as its own colour: lower-case letter (//), upper-case letter (\\), space/newline (xx), punctuation or digit (..). The black dotted horizontal line marks the strict plateau rule 0.25 and the gray dashed line the straight-line value 0.80.](plots/comma_width_by_char.png)

**What predicts sharpness.** Width falls as the model's own probability for that character after
`"The house was "` rises: Spearman $\rho = -0.74$ (p = 2.7e-12, n = 64). Endpoint separation explains
much less ($\rho = -0.48$, p = 5.6e-5) and with the sign that rules out the trivial reading —
*wider*-separated endpoints switch *faster*, so flat curves are not "the two outputs are too similar
to distinguish". The comma endpoint is itself an implausible continuation here (model probability
1.0e-7), so the sharp cases are not driven by both endpoints being common inputs; what varies is the
other character.

![Figure 8 — Left: transition width (y) vs the model's probability of the other character after "The house was " (x, log scale); one point per pair, with a distinct marker shape per character type — circle = lower-case letter, square = upper-case letter, triangle = space/newline, diamond = punctuation or digit; Spearman ρ = −0.74. Right: transition width (y) vs the L2 distance between the two endpoints' final-logit vectors (x), same markers; Spearman ρ = −0.48. In both panels the black dotted horizontal line marks the strict plateau rule 0.25 and the gray dashed line the straight-line value 0.80.](plots/comma_width_vs_endpoints.png)

**Both structural controls replicate with 32× more pairs.** Moving the interpolation point deeper
flattens the curve back onto the straight line — median width 0.34 (block 0), 0.51, 0.65, 0.72, 0.77,
0.79, then ≈0.80 for blocks 6–11 — and across training the transition narrows early and then stops
changing: 0.799 (init) → 0.751 (step 56) → 0.524 (831) → 0.328 (7,819) → 0.367 (17,500) → 0.340
(30,000). Both trends match the `b↔i`/`b↔l` result above.

![Figure 9 — Left: median transition width over the 64 pairs (y, solid line with circle markers) vs interpolation block (x, 0–11; the residual stream after this block is the one replaced); the hatched band is the inter-quartile range; the gray dashed horizontal line is the straight-line value 0.80 and the black dotted line the strict plateau rule 0.25. Right: median transition width (y, dashed line with square markers) vs training step (x, log scale, step 0 drawn at 1) at interpolation block 0, over the six frozen checkpoints; hatched band = inter-quartile range; same two reference lines.](plots/comma_depth_and_training.png)

**Discussion of this sweep.** Five points, in order of how much they change the picture.
*(1)* The plateau-like shape is the rule, not the exception in this model: with one endpoint fixed
and all 64 alternatives tried, no pair behaves linearly — the downstream stack always holds the
output near one endpoint, switches, and holds near the other. That it survives an exhaustive sweep is
what this experiment adds over hand-chosen pairs. *(2)* Sharpness is a continuum and the strict bar
sits near its edge: 1/64 at $w \le 0.25$ but 33/64 at $w \le 0.35$, so any count of "how many
plateaus" in this model is mostly a statement about the threshold — we therefore report the full
distribution. *(3)* The switch is sharpest for characters the model actually expects there. A plain
reading of $\rho = -0.74$: when the second endpoint is a continuation the model has a confident,
well-practised output for, the downstream layers snap between two familiar outputs; when it is a
character the model essentially never predicts in that slot (`3`, `&`, `!`, `:`, `z`), the output
drifts across the path instead. This is a correlation over 64 characters in one context, not a causal
test. *(4)* It does not rescue the joint question: these pairs are measured on the same non-grokking
run at the same frozen checkpoints, they reach their final sharpness by step ~7,800, and the bounded
relationship verdict stays PLAN case 5. *(5)* Caveats: one model, interpolation at the final token
only, and single characters as endpoints. The two context-related worries — one shared context, and a
comma endpoint that is itself an unlikely input — are tested next.

**Does the plateau depend on the context? No — 0 of 576 curves is linear across nine contexts.**
Repeating the sweep in 8 further held-out contexts spanning $p(\texttt{,})$ from $5\times10^{-20}$ to
0.997 leaves the shape result untouched: **not one of the 576 curves is near the straight line**
($w \ge 0.70$), per-context median widths stay in the narrow band **0.313–0.436** (reference context
0.340; pooled median 0.381), and the strict bar stays hard to clear (11/576 at $w \le 0.25$; 198/576
at $w \le 0.35$). The implausible-endpoint worry is also settled: the context where a comma is nearly
certain (probability 0.997) gives median width 0.330 — indistinguishable from the reference — and
across the nine contexts the comma's own probability does not predict sharpness ($\rho = -0.32$,
p = 0.41, n = 9).

![Figure 10 — Left: transition width $w_{10\to90}$ (y) for the 64 comma→character pairs of each context (x, one box per context, ordered by the model's probability of a comma there which is printed under each box; "ref" = "The house was ", the context behind every earlier plateau number, drawn with a cross hatch; the 8 held-out contexts use a diagonal hatch). Boxes give the inter-quartile range with the median as a horizontal bar, whiskers 1.5×IQR, dots outliers. Gray dashed = straight-line value 0.80, black dotted = strict plateau rule 0.25. Right: each context's median width (y) vs its comma probability (x, log scale); circles = held-out contexts, diamond = the reference context; same two reference lines.](plots/context_widths.png)

**The width-vs-probability predictor replicates in direction, not in size.** Within each context the
rank correlation between transition width and the model's probability of the target character is
**negative in all nine cases** (sign test p = 0.004; individually significant at p < 0.05 in 7 of 9),
so "the switch is sharper for characters the model expects there" is a real repeatable tendency. Its
strength, however, ranges from −0.05 to −0.74 with median **−0.41**, and pooling all 576 pairs gives
$\rho = -0.23$. The context we reported first is the strongest of the nine, so the earlier −0.74
should be read as the top of a range rather than a typical value — a case where a single context
overstated an effect that is real but modest.

![Figure 11 — Left: Spearman ρ between transition width and the model's probability of the target character (x) for each context (y, ordered by that context's comma probability; reference context cross-hatched, held-out contexts diagonally hatched); the dash-dot vertical line marks the median over contexts (−0.41). Right: transition width (y) vs the model's probability of the target character in its own context (x, log scale) for all 576 pairs; circles = the 8 held-out contexts, diamonds = the reference context; gray dashed = straight-line value 0.80, black dotted = strict plateau rule 0.25.](plots/context_rho.png)

**Exploratory corroboration: 14/40 natural minimal pairs are plateaus; almost all curves are sigmoid.**
*(Labelled exploratory and kept out of the headline — PLAN scope forbids a new 40-pair dataset in the
primary analysis. Retained because its layerwise and depth controls corroborate the above at larger
`n`.)* With interpolation after block 0 and recording at final logits, 14 of 40 pairs meet the strict
frozen rule
(IDs 0, 4, 5, 6, 7, 9, 14, 20, 21, 22, 28, 34, 36, 37); 24/40 have $w \le 0.35$; only 2/40 are
near-diagonal (#10, #19, $w \ge 0.6$); 0/40 are non-monotone. Median width is 0.309 (range
[0.110, 0.773]) against the diagonal's 0.8. The structure is visible pair by pair — no averaging is
involved:

![Figure 3 (exploratory) — Raw relative distance d(t) (y) vs interpolation step t (x) in final-logit space, one panel per frozen pair; panel titles give pair ID, the two endpoint characters, and the transition width w. Gray dashed = diagonal d = t. Most curves hug d≈0, cross rapidly, then hug d≈1; two (#10, #19) track the diagonal.](plots/pair_curves_logits.png)

This is heterogeneity worth stating plainly: the strict 0.25 bar splits a continuum — the model's
typical curve is strongly sigmoid (three times sharper than the diagonal) rather than every pair
being an extreme step. No pair had to be discarded, and endpoint separation does not obviously
predict which pairs pass.

**The boundary sharpens with depth, exactly as Matthew observed.** Fixing interpolation after
block 0 and recording $d(t)$ at each later block's final-position residual: median width falls
strictly monotonically from 0.777 (block 1) to 0.445 (block 11) and 0.309 at the logits; the strict
rule is passed only at the logits (14 pairs), never at intermediate residuals. The plateau is
*formed* by the downstream stack, not present in the interpolated activation itself:

![Figure 4 (exploratory) — Layerwise emergence for four fixed representative pairs (IDs 0–3, frozen before inspection): d(t) (y) vs interpolation step t (x). Thin lines are the recording blocks, shaded on the cividis scale from block 1 (dark) to block 11 (light) per the colour bar; the thick black line is the final logits and the gray dashed line the diagonal. Early-block curves are near-diagonal and progressively sharpen into plateau–boundary–plateau by the output.](plots/layerwise_emergence.png)

**Later interpolation kills the plateau — the predicted control.** If downstream layers create the
plateau, interpolating later (fewer layers left) must weaken it. It does, monotonically: median
$w_{10\to 90}$ = 0.309, 0.564, 0.647, 0.733, 0.757, 0.802 for interpolation blocks 0, 2, 4, 6, 8, 10
— reaching the diagonal reference 0.8 when only one block remains:

![Figure 5 (exploratory) — Left: median final-logit d(t) (y) vs interpolation step t (x) per interpolation block, shaded on the cividis scale from block 0 (dark) to block 10 (light) as given in the legend; the block-0 curve is strongly sigmoid and later blocks collapse onto the gray dashed diagonal. Right: median transition width w_10→90 (y; bars = inter-quartile range across the 40 pairs, solid line with circle markers) vs interpolation block (x); the black dotted horizontal line is the plateau bar 0.25 and the gray dashed line the diagonal reference 0.8.](plots/interpolation_layer_comparison.png)

Tidy per-curve data: `results/matthew_tidy.csv`; per-pair summary: `results/matthew_summary.json`.

## Conclusion

The reconstructed 12-layer character-level Shakespeare GPT **shows Matthew-style activation
plateaus**. Under a fully frozen assay, 14/40 natural minimal pairs produce individual
plateau–boundary–plateau curves in final-logit space; the typical pair is a clear sigmoid three times
sharper than the no-plateau diagonal; and both predicted structural signatures hold — monotone
sharpening across the 11 downstream blocks and monotone weakening as the interpolation point moves
later. Sweeping one fixed endpoint (the comma) against all 64 other characters shows the same shape
for every pair — none linear, median width 0.340 — while showing that the sharpness is graded (1/64
at the strict ≤ 0.25 bar) and largest for characters the model expects in that context. Repeating
that sweep in eight further held-out contexts (576 pairs) shows the shape claim is not an artifact of
the chosen context — **0/576** near-linear curves, per-context medians 0.313–0.436 — and that the
fixed endpoint's own plausibility is irrelevant; it also puts the probability effect in proportion
(negative in 9/9 contexts, median ρ = −0.41, not the −0.74 of the first context alone). The plateau
gate is **go**: mapping and interpreting these plateau basins (where boundaries lie, what they
correspond to linguistically, how they evolve over training) is warranted on this model.

**Joint Grokking↔plateau verdict: primary relationship not testable (PLAN case 5).** The mandatory
validity gate — reproducing *Deep Networks Always Grok* Figure 9 — is **FAILed by all three models we
trained** (pilot char 3.5k, fresh char 30k, fresh BPE 10k): each develops adversarial robustness but none
shows the defining *second* local-complexity descent within budget (the fresh runs overfit instead).
Because the BPE model required for Matthew's exact `big/in`, `big/large` tokens does not reproduce
Figure 9, we **cannot** test whether plateaus sharpen during a second-descent/robustness window — that
window never opens. This is a bounded, honest null on the *relationship*, not on plateaus. **Secondary
temporal observation:** running Matthew's exact assay with the `b↔i`/`b↔l` character controls across
six frozen checkpoints, the plateau **emerges early** — absent at init (width ≈ 0.80), sharp by step
~831 (width ≈ 0.33), then flat to 30k — during the *first* LC descent and initial fit, and fully
formed **before** adversarial robustness saturates. So in this model the plateau shows **no visible
temporal coupling** to the grokking signature; it is a property of the trained downstream stack that
appears with initial fit. The plateau phenomenon itself is clearly present and stands independently.
Closing
the relationship question would require a training setup that actually groks (far longer horizon, weight
decay tuned for delayed generalization, or the paper's exact recipe) — outside this run's compute
budget (~30k vs the paper's ~1e5 steps).

**Interpretation.** The interpolated block-0 activation itself carries a nearly linear image of the
input mixture; the downstream stack then collapses it toward one of the two endpoint computations,
with the decision boundary near $t \approx 0.5$. That the effect strengthens with every additional
downstream layer suggests plateaus here are an emergent, distributed property of the whole stack
rather than a single layer's thresholding. Note this coexists with an earlier finding on the same
model (see CHANGELOG, 2026-07-15) that responses to *random-direction* perturbations are smooth and
saturating — that assay answered a different question; the plateau structure lives specifically along
natural activation-to-activation directions.

**Limitations.**
1. **Reconstruction, not the paper's checkpoint.** The verdict applies to a faithful, standard build
   of the Figure-9 architecture and training recipe, not the authors' exact weights.
2. **Strictness of the bar is arbitrary.** 14/40 pass at $w \le 0.25$; 24/40 at 0.35. The
   comma-against-everything sweep makes this sharper still: 1/64 pass at 0.25 but 33/64 at 0.35, even
   though not one of the 64 curves is linear (across the nine-context control, 11/576 and 198/576).
   We report the full width distribution so readers can apply their own threshold; the depth and
   layerwise trends do not depend on any threshold.
3. **Scope.** One model size, one training length (accuracy 0.56, not grokking-scale), final-position
   interpolation only, and endpoint pairs differing in exactly one character. Plateaus between more
   distant natural inputs are untested here. Context dependence *has* now been tested (nine contexts,
   576 pairs) and the shape result holds; what remains untested is other models and other
   interpolation positions.
4. **No grokking model, so the per-checkpoint Matthew sweep is secondary, not a joint result.**
   Because every trained run FAILs the Figure-9 gate, the checkpoint-aligned Matthew sweep on a
   *grokking BPE* model (Matthew's exact `big/in`, `big/large` tokens) is not decisive. We instead ran
   Matthew's exact assay with the `b↔i`/`b↔l` character controls across six frozen checkpoints (S6) as
   *secondary* per-checkpoint plateau evidence: it shows the plateau emerging with initial fit, not
   with grokking — but this is temporal association in a non-grokking model, not the intended joint
   result, and not a causal claim. The joint verdict remains the bounded null of PLAN case 5.
5. **Single-context correlations can overstate an effect.** The width-vs-probability rank correlation
   was −0.74 in the first context we measured but has median −0.41 (range −0.05 … −0.74) across nine
   contexts. The direction is solid (9/9 negative, sign test p = 0.004); the magnitude should be
   quoted as a range, and the same caution applies to any other single-context number here.
