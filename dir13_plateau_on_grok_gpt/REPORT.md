# REPORT — Does the 12-layer Shakespeare GPT show activation plateaus?

> Final, presentable, current-best only (history in CHANGELOG.md). Read before rewriting.

## Summary

The paper *Deep Networks Always Grok and Here is Why* (Humayun, Balestriero, Baraniuk, ICML 2024)
argues that trained networks develop **activation plateaus** — flat regions where a hidden activation
can be perturbed a long way before the network's downstream output changes, followed by a sharp
transition at a region boundary. This "flat-then-steep" geometry is offered as a mechanism behind
delayed generalization (grokking) and adversarial robustness. This direction is a cheap **go/no-go
gate**: before anyone maps or interprets plateaus in the paper's **Figure-9 model** — a 12-layer,
12-head, character-level Shakespeare GPT with GeLU MLPs — do plateaus even exist there?

We first audited the paper's official repository: it releases training code only for the MNIST-MLP and
CIFAR-ResNet experiments; **the GPT training code and checkpoint are not public**. We therefore built a
faithful reconstruction (12 blocks, 12 heads, GeLU MLP, char-level Shakespeare — the paper's confirmed
facts) and trained it to a genuinely-trained state (next-char accuracy 0.56, ≈37× chance). Applying a
pre-registered, control-calibrated plateau assay to the final-position residual stream, **we find no
plateaus at any of the 11 intervention blocks**: the downstream response is smooth and *saturating*
(concave), the opposite of a plateau. Real activations differ significantly from a norm-matched random
control (they saturate slightly less), but that is a difference between two non-plateau shapes.
**Verdict: no-go for a plateau-mapping follow-up on this model — qualified, because we tested a
reconstruction rather than the paper's exact checkpoint.**

## Methods

### Data & Model

- **Task/data.** Next-character prediction on **Tiny Shakespeare** (`input.txt`, 1,115,394 chars,
  SHA-256 `86c4e6…565ed`), first-90%/last-10% train/val split, character-level tokenization
  (vocabulary size 65).
- **Model.** A nanoGPT-style decoder-only causal GPT — **12 transformer blocks, 12 heads/block, GeLU
  MLPs** (the paper's confirmed Figure-9 facts), pre-norm, learned absolute positions, weight-tied
  embedding/head. Reconstruction choices (not specified by the paper): `d_model = 240` (head dim 20),
  MLP hidden `4·d_model`, context length 128, dropout 0.2, 8.38M parameters. Every field is tagged
  confirmed-vs-reconstructed in `MODEL_SPEC.md`.
- **Why a reconstruction.** The official repo `AhmedImtiazPrio/grok-adversarial` (git tree audited via
  the GitHub API on 2026-07-15) contains only `train_mlp_mnist.py` and `train_resnet18_cifar10.py`; no
  GPT/transformer/Shakespeare code or checkpoint exists. Per the plan's success-criterion (3) we keep
  all conclusions explicitly about this reconstruction.
- **Training.** AdamW (betas 0.9/0.99, wd 0.1), peak LR 1e-3 with 100-step warmup + cosine decay,
  batch 48×128, fp32, 3,500 steps. Final **val loss 1.494**, **val next-char accuracy 0.560**. Seeds,
  versions, and log-spaced checkpoints saved (`results/train_meta.json`, `results/checkpoints/`).
- **Hook point.** Unit of intervention = the residual stream at the **final sequence position** after
  transformer block `l`. Because attention is causal, modifying only the final position at block `l`
  changes only the final position downstream, so a forward pass from block `l+1` using the clean
  residual with a replaced final position is exact (verified below). Downstream measurement = the
  final-position residual immediately before the LM head (post final LayerNorm).
- **Sample sizes.** 48 held-out contexts (disjoint 16-context pilot), 8 fixed random unit directions
  per context, 41 radii per ray, both natural and matched-control basepoints, all 11 blocks 0–10.
  Contexts are pre-registered to those where the model's next character is correct.

### Metrics

We perturb a final-position activation `h` at block `l` along a fixed unit direction `u` and scale the
step by the layer's natural scale `s_l` (the median L2 distance between random pairs of held-out
final-position activations at that block), so `alpha = rho * s_l` is comparable across layers.

**Downstream response** — how far the pre-head residual `z` moves for a perturbation of size `alpha`,
normalized by width so it is comparable across layers:

```math
d_{\mathrm{hidden}}(\alpha) = \frac{\lVert z(h + \alpha u) - z(h) \rVert_2}{\sqrt{d_{\mathrm{model}}}}
```

**Plateau index (PI)** — the shape statistic. Normalize each ray to `x = rho/rho_max` and
`y = d(rho)/d(rho_max)` (both run 0→1) and integrate the gap to the straight line:

```math
\mathrm{PI} = \int_0^1 \big[\, x - y(x) \,\big]\, dx
```

Read it as: `PI > 0` means the response is **delayed** relative to a straight line (flat-then-steep =
a **plateau**); `PI = 0` is linear; `PI < 0` means the response is **front-loaded / saturating**
(steep-then-flat). Higher is more plateau-like.

**Boundary sharpness** — the steepness of the sharpest transition relative to the average slope, so a
plateau must be both delayed *and* have a sharp edge:

```math
\mathrm{sharp} = \frac{\max_i \big[ \Delta y_i / \Delta x_i \big]}{\overline{\Delta y / \Delta x}}
```

**Functional response (JSD)** — the same experiment scored on the model's *output* instead of its
hidden state, via the Jensen–Shannon divergence between the baseline and perturbed next-character
distributions `p_0, p`:

```math
\mathrm{JSD}(p_0 \Vert p) = \tfrac{1}{2}\,\mathrm{KL}\!\big(p_0 \Vert m\big) + \tfrac{1}{2}\,\mathrm{KL}\!\big(p \Vert m\big), \quad m = \tfrac{1}{2}(p_0 + p)
```

We compute a PI on the JSD curve as well; the hidden-state and JSD conclusions must agree. We also
record the fraction of rays whose top-1 next character flips at the largest radius (a calibration that
the perturbation range is functionally large).

**Group comparison.** The plateau effect is the control-calibrated difference of medians, with a
hierarchical bootstrap (resample contexts, then directions) for the 95% interval:

```math
\Delta \mathrm{PI} = \operatorname{median}\big(\mathrm{PI}_{\text{natural}}\big) - \operatorname{median}\big(\mathrm{PI}_{\text{control}}\big)
```

Effect size is Cliff's delta between the pooled natural and control PI values (`|delta| > 0.474` is a
"large" effect). A tiny-but-significant `ΔPI` is treated as inconclusive on its own — the **sign of PI
itself** decides whether a plateau exists.

### Baselines

**Matched off-distribution control.** For each natural basepoint we draw a control from the empirical
per-block diagonal Gaussian (mean `mu_l`, per-coordinate std `sigma_l` of the held-out final-position
activations) and rescale it to the natural activation's norm, keeping the surrounding sequence fixed:

```math
h_{\text{ctrl}} = \big(\, \sigma_l \odot \varepsilon + \mu_l \,\big)\cdot \frac{\lVert h \rVert_2}{\lVert \sigma_l \odot \varepsilon + \mu_l \rVert_2}, \qquad \varepsilon \sim \mathcal{N}(0, I)
```

This isolates *on-manifold structure*: a plateau specific to trained representations should be stronger
for natural `h` than for this norm-and-statistics-matched off-manifold point.

**Linear reference.** The straight line `y = x` (a purely linear downstream response). `PI` is defined
as the signed area between a ray and this reference, so the baseline is built into the metric and shown
as the dashed line in the individual-ray figure.

### Calibration checks (pre-registered; all passed)

- **Assay can detect a plateau:** on a synthetic flat-then-steep curve `PI = +0.33`; on a linear curve
  `PI = 0.00`; sharpness of the delayed curve is 3.2× the linear one (unit test in `assay.py`).
- **`alpha = 0` fidelity:** the partial forward from block `l+1` reproduces the full unmodified forward
  pass to max logit error `< 1e-3` at blocks 0/5/10 (also validates batched interventions across
  mini-batch boundaries).
- **Range is large enough:** radius 0 gives distance 0 by construction; at the frozen `rho_max` the
  top-1 next character flips for ≥81% of rays at every block.
- **Not an averaging artifact:** individual rays are inspected (figure below).
- **Metric agreement:** hidden-state and JSD scores give the same qualitative conclusion.

## Results

**Training.** The model trains normally to val loss 1.494 / accuracy 0.560 — clearly a trained network,
not a random one.

![Training curves: cross-entropy falls to ~1.49 on validation; next-char accuracy rises to 0.56.](plots/training_curves.png)

**No plateau at any block.** Across all 11 intervention blocks the median `PI` of *natural* activations
is **negative** (−0.15 to −0.30): the downstream response rises quickly and then saturates, the
opposite of the flat-then-steep plateau shape. The response curves sit **above** the linear diagonal
(concave), for both natural and control basepoints.

![Downstream response vs perturbation radius, per block. Natural (blue) and matched-control (red) curves are both concave and saturating, above the linear diagonal — no plateau at any block.](plots/response_by_layer.png)

**The natural-vs-control difference is real but not a plateau.** `ΔPI` is positive at every block with
95% CIs excluding zero (peaking at `ΔPI = +0.096`, Cliff's δ = +0.91 around blocks 2–3), meaning
natural activations saturate *slightly less* than the random control. The JSD-based `ΔPI` agrees in
sign at every block. But because *both* PIs are negative, this is a difference between two non-plateau
shapes — it signals mild on-manifold structure, not a plateau.

![Plateau index by block. Left: PI is negative (saturating) for both natural and control at all 11 blocks. Right: ΔPI (nat−ctrl) is small, positive, and significant, decaying from early to late blocks.](plots/plateau_score_by_layer.png)

**Not an averaging artifact.** Every individual ray is concave; none shows a flat region followed by a
steep edge. An average therefore cannot be hiding a plateau.

![Individual rays (blue=natural, red=control, dashed=linear reference). Each ray is concave/saturating; no ray is flat-then-steep.](plots/individual_curves.png)

Full numbers per block are in `RESULTS.md`; the tidy per-ray table is `results/tidy_results.csv` (one
row per context × direction × block × basepoint × radius).

## Conclusion

Under a pre-registered, control-calibrated assay that provably *can* detect a plateau, the
reconstructed 12-layer character-level Shakespeare GPT shows **no activation plateaus** at its
final-position residual stream: downstream responses are smooth and saturating at every block 0–10. The
statistically significant natural-vs-random difference (`ΔPI > 0`) reflects that real activations
saturate a little less sharply than matched noise — mild on-manifold structure — not a plateau. This is
a **no-go** for a plateau-mapping follow-up on this model.

**Interpretation.** A plateau (locally flat, low-sensitivity region) is a natural consequence of
piecewise-linear MLPs with saturating ReLU regions, where the MNIST result was found. A residual GPT
with GeLU MLPs and LayerNorm passes perturbations through an approximately additive residual stream,
which tends to produce linear-to-saturating downstream responses rather than flat regions — consistent
with what we observe. So the plateau phenomenon may be architecture-specific rather than universal.

**Limitations.**
1. **Reconstruction, not the paper's checkpoint.** The paper's exact GPT is unreleased; a negative on
   our reconstruction cannot prove the paper's exact model lacks plateaus. It does show that a
   faithful, standard build of that architecture does not exhibit them.
2. **Training length.** We trained to a solid but not extreme point (accuracy 0.56) under a shared-GPU
   budget; the paper's phenomena can emerge with much longer training. Whether plateaus appear only
   after extended grokking-scale training is a separate question (the remit of the "during training"
   direction), not this gate.
3. **Scope.** We probed the final-position residual after each block with random directions; a plateau
   confined to other positions, specific learned directions, or a different hook point could be missed,
   though the individual-ray and multi-block evidence argues against a strong effect.
