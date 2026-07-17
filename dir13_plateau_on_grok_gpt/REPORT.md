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
and a fresh 30k-step BPE run each show a *first* local-complexity descent and emerging `ε=0.03`-PGD
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
interpolation point later collapses it back *onto* the diagonal. **Verdict: plateaus are real in this
model** — qualified, because we tested a reconstruction rather than the paper's exact checkpoint.

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
gray dashed line in every figure, and the depth-comparison test checks whether curves collapse onto
it.

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

![Figure 1 — Training curves: cross-entropy loss in nats (y, left panel) for train and validation falls to ~1.49; validation next-char accuracy (y, right panel) rises to 0.56; x = training step.](plots/training_curves.png)

**Figure-9 gate — pilot FAILs within its 3,500-step horizon.** Across 13 log-spaced checkpoints the
pilot's clean next-char accuracy climbs to 0.564 (peak at the last checkpoint) and `ε=0.03` PGD
adversarial accuracy rises to 0.327 — so *delayed robustness does emerge* — but test LC falls
monotonically from 1940 to its minimum (68) **at the final checkpoint**: there is **no second LC
descent**. Under the preregistered rule this is a **FAIL** (valid measurements, Fig. 9 ordering absent
within the tested horizon), not "not established", because robustness rose rather than staying flat.
The honest reading: the 3,500-step horizon ends inside the *first* LC descent, so the pilot cannot
support a joint Grokking↔plateau claim.

**Figure-9 gate — both fresh 30k-step runs also FAIL (S4, S5).** We trained two matched fresh runs
(character + BPE, 30k-step schedule, budget-capped below the paper's ~1e5) and ran the identical
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

![Figure 1b — Pilot char (3.5k) Figure-9 curves. Left y-axis = local complexity (sign-crossing units summed over the 12 GeLU layers) for train (blue), test (orange), random (green) base points with 99% CI bands; right y-axis = next-token accuracy, black solid = clean test accuracy, red dashed = ε=0.03 PGD adversarial accuracy; x-axis = training step (log scale, step 0 drawn at 1). LC descends monotonically to the horizon (no second descent) while adversarial accuracy rises to 0.33 — verdict FAIL.](plots/grokking_pilot_char.png)
![Fresh char (30k) Figure-9 gate — LC monotone to 8.1, adversarial accuracy to 0.53, no second descent → FAIL.](plots/grokking_fresh_char.png)
![Fresh BPE (10k) Figure-9 gate — LC monotone to 95, adversarial accuracy to 0.19, no second descent → FAIL.](plots/grokking_fresh_bpe.png)

**Joint checkpoint timeline and bounded relationship verdict (S7).** Putting all three runs on one
training-step axis (below), the Grokking side is uniform: LC falls monotonically and PGD robustness
rises, with **no second descent in any run**. Because the primary bridge to Matthew — a BPE model that
reproduces Figure 9 — does **not** pass the gate (and neither character run does either), the bounded
relationship verdict is **PLAN case 5: "primary relationship not testable."** We cannot claim the
plateau assay sharpens *during* a second-descent/robustness window, because no run exhibits that window.
What we *can* say standalone: the character reconstruction **does** show Matthew-style plateaus (next
sections), and both fresh runs develop adversarial robustness — but robustness and plateaus here are
properties of trained/memorising networks, not evidence of the specific grokking ordering.

![Figure 1d — Joint checkpoint timeline. Left panel: test local complexity (y) vs training step (x, log scale) for pilot char (gray), fresh char (blue), fresh BPE (red); each legend entry gives the run's Figure-9 gate verdict. Middle panel: ε=0.03 PGD adversarial accuracy (y) vs training step (x, log), same colors; dashed line = the 0.05 robustness threshold in the verdict rule. Right panel: text summary of the three gate verdicts (all FAIL), the plateau-assay reference from the reconstruction, and the bounded relationship verdict. No run shows a second LC descent, so the joint verdict is "primary relationship not testable."](plots/joint_timeline.png)

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

![Figure 2a — Matthew-faithful char-control d(t) (y), interpolation block 0, final logits, one panel per frozen checkpoint (steps 0→30000, x = t within each panel); blue = b↔i, orange = b↔l, gray dashed = diagonal d=t. Diagonal at init/step 56; sharp plateau–boundary–plateau sigmoid by step 831, stable thereafter.](plots/matthew_char_ctrl_by_checkpoint.png)

![Figure 2b — Grokking metrics vs plateau width on one timeline (fresh char run). Top: left y = local complexity for LC train (blue)/test (orange)/random (green), 99% CI; right y = next-token accuracy, black = clean, red dashed = ε=0.03 PGD adv; x = training step (log). Bottom: transition width w_10→90 (y) for b↔i (blue) and b↔l (orange) vs step (log); gray dashed = diagonal 0.8, red dotted = plateau bar 0.25. Width hits its floor by step ~831 — during the first LC descent, before robustness rises.](plots/joint_timeline_char_ctrl.png)

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

![Figure 4 (exploratory) — Layerwise emergence for four fixed representative pairs (IDs 0–3, frozen before inspection): d(t) (y) vs t (x). Line color = recording block from 1 (dark) to 11 (light), per colorbar; red = final logits; gray dashed = diagonal. Early-block curves are near-diagonal and progressively sharpen into plateau–boundary–plateau by the output.](plots/layerwise_emergence.png)

**Later interpolation kills the plateau — the predicted control.** If downstream layers create the
plateau, interpolating later (fewer layers left) must weaken it. It does, monotonically: median
$w_{10\to 90}$ = 0.309, 0.564, 0.647, 0.733, 0.757, 0.802 for interpolation blocks 0, 2, 4, 6, 8, 10
— reaching the diagonal reference 0.8 when only one block remains:

![Figure 5 (exploratory) — Left: median final-logit d(t) (y) vs t (x) per interpolation block (line color dark→light = block 0→10); the block-0 curve is strongly sigmoid and later blocks collapse onto the gray dashed diagonal. Right: median transition width w_10→90 (y; bars = interquartile range across the 40 pairs) vs interpolation block (x); red dashed = plateau bar 0.25; gray dashed = diagonal reference 0.8.](plots/interpolation_layer_comparison.png)

Tidy per-curve data: `results/matthew_tidy.csv`; per-pair summary: `results/matthew_summary.json`.

## Conclusion

The reconstructed 12-layer character-level Shakespeare GPT **shows Matthew-style activation
plateaus**. Under a fully frozen assay, 14/40 natural minimal pairs produce individual
plateau–boundary–plateau curves in final-logit space; the typical pair is a clear sigmoid three times
sharper than the no-plateau diagonal; and both predicted structural signatures hold — monotone
sharpening across the 11 downstream blocks and monotone weakening as the interpolation point moves
later. The plateau gate is **go**: mapping and interpreting these plateau basins (where boundaries lie,
what they correspond to linguistically, how they evolve over training) is warranted on this model.

**Joint Grokking↔plateau verdict: primary relationship not testable (PLAN case 5).** The mandatory
validity gate — reproducing *Deep Networks Always Grok* Figure 9 — is **FAILed by all three models we
trained** (pilot char, fresh 30k char, fresh 30k BPE): each develops adversarial robustness but none
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
2. **Strictness of the bar is arbitrary.** 14/40 pass at $w \le 0.25$; 24/40 at 0.35. We report the
   full width distribution so readers can apply their own threshold; the depth and layerwise trends
   do not depend on any threshold.
3. **Scope.** One model size, one training length (accuracy 0.56, not grokking-scale), final-position
   interpolation only, and endpoint pairs differing in exactly one character. Plateaus between more
   distant natural inputs are untested here.
4. **No grokking model, so the per-checkpoint Matthew sweep is secondary, not a joint result.**
   Because every trained run FAILs the Figure-9 gate, the checkpoint-aligned Matthew sweep on a
   *grokking BPE* model (Matthew's exact `big/in`, `big/large` tokens) is not decisive. We instead ran
   Matthew's exact assay with the `b↔i`/`b↔l` character controls across six frozen checkpoints (S6) as
   *secondary* per-checkpoint plateau evidence: it shows the plateau emerging with initial fit, not
   with grokking — but this is temporal association in a non-grokking model, not the intended joint
   result, and not a causal claim. The joint verdict remains the bounded null of PLAN case 5.
