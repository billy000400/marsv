# REPORT — Does the 12-layer Shakespeare GPT show activation plateaus?

> Final, presentable, current-best only (history in CHANGELOG.md).

## Summary

*Deep Networks Always Grok and Here is Why* (Humayun, Balestriero & Baraniuk, ICML 2024) claims that
trained networks develop **activation plateaus**: perturb a hidden activation along a line and the
network's output barely moves for a while (a flat region), then changes sharply at a region boundary.
The paper offers this flat-then-steep geometry as a mechanism behind **grokking** (delayed
generalization long after the training loss converges) and adversarial robustness. Its **Figure-9
model** is a 12-layer, 12-head character-level Shakespeare GPT with GeLU MLPs. Before anyone spends
effort mapping or interpreting plateaus in that model, this direction asks the cheap gating question:
**do plateaus exist there at all?**

The paper's GPT code and checkpoint are not public (the official repo covers only the MNIST-MLP and
CIFAR-ResNet experiments), so we trained a faithful reconstruction — 12 blocks, 12 heads, GeLU MLPs,
character-level Shakespeare, next-char accuracy 0.56 ≈ 37× chance — and ran a pre-registered,
control-calibrated plateau assay on its residual stream. **Result: no plateaus at any of the 11
intervention blocks.** The downstream response is smooth and *saturating* (concave) — the opposite of
flat-then-steep. Natural activations do differ significantly from a norm-matched random control (they
saturate slightly less), but that is a difference between two non-plateau shapes. **Verdict: no-go
for a plateau-mapping follow-up on this model — qualified, because we tested a reconstruction rather
than the paper's exact checkpoint.**

## Methods

### Data & Model

- **Task/data.** Next-character prediction on **Tiny Shakespeare** (`input.txt`, 1,115,394 chars,
  SHA-256 `86c4e6…565ed`); first 90% train, last 10% validation; character-level tokens (vocab 65).
- **Model.** A nanoGPT-style decoder-only GPT: **12 blocks, 12 heads, GeLU MLPs** (the paper's
  confirmed Figure-9 facts), pre-norm, learned positions, weight-tied head. Reconstruction choices
  (unspecified by the paper): `d_model = 240`, MLP hidden `4·d_model`, context 128, dropout 0.2,
  8.38M params. Every field is tagged confirmed-vs-reconstructed in `MODEL_SPEC.md`.
- **Why a reconstruction.** The official repo `AhmedImtiazPrio/grok-adversarial` (audited via the
  GitHub API, 2026-07-15) contains no GPT/Shakespeare code or checkpoint. Per the plan's
  success-criterion (3), all conclusions are explicitly about this reconstruction.
- **Training.** AdamW (betas 0.9/0.99, weight decay 0.1), peak LR 1e-3, 100-step warmup + cosine
  decay, batch 48×128, fp32, 3,500 steps → **val loss 1.494, val next-char accuracy 0.560**. Seeds
  and provenance in `results/train_meta.json`.
- **Hook point.** We intervene on the **residual stream** (the running hidden vector each block adds
  to) at the **final sequence position** after block `l`. Causal attention means this changes only
  the final position downstream, so a forward pass from block `l+1` with the one vector replaced is
  exact (verified below). We measure the response at the final-position residual just before the
  language-model head.
- **Sample sizes.** 48 held-out contexts (disjoint from a 16-context pilot), 8 fixed random unit
  directions per context, 41 radii per ray, natural and matched-control basepoints, all blocks 0–10.
  Contexts are pre-registered to those the model predicts correctly.

### Metrics

We perturb a final-position activation `h` at block `l` along a fixed unit direction `u`, with step
size `alpha = rho * s_l`, where `s_l` (the median distance between random pairs of held-out
activations at that block) makes radii comparable across layers. Each metric answers one question in
the chain "is there a plateau?".

**Downstream response** — *how much did the perturbation change the computation?* Logit or loss
changes conflate this with how the head weighs directions, so we measure movement of the pre-head
residual `z` itself, normalized by width. This is the y-axis of Figures 2 and 4:

```math
d_{\mathrm{hidden}}(\alpha) = \frac{\lVert z(h + \alpha u) - z(h) \rVert_2}{\sqrt{d_{\mathrm{model}}}}
```

**Plateau index (PI)** — *is the curve flat-then-steep?* Averaging curves can hide or manufacture
shape, so we score **each ray**. Normalize it to `x = rho/rho_max`, `y = d(rho)/d(rho_max)` (both run
0→1) and integrate the gap to the straight line:

```math
\mathrm{PI} = \int_0^1 \big[\, x - y(x) \,\big]\, dx
```

Read it as: `PI > 0` = response **delayed** relative to a line (flat-then-steep = **plateau**);
`PI = 0` = linear; `PI < 0` = **front-loaded/saturating**. Higher is more plateau-like. The sign of
the median PI is the headline result (Figure 3 left; per-block table in `RESULTS.md`).

**Boundary sharpness** — *is there a sharp wall?* PI alone cannot separate a gentle S-curve from a
flat region ending at a hard boundary, so we also score each ray's steepest segment relative to its
mean slope:

```math
\mathrm{sharp} = \frac{\max_i \big[ \Delta y_i / \Delta x_i \big]}{\overline{\Delta y / \Delta x}}
```

A line scores 1.0; our synthetic flat-then-steep test curve scores 3.2. Sharpness signals a plateau
**only together with** `PI > 0` (the steep segment must come *late*). Consumed in the sharpness
paragraph of Results.

**Functional response (JSD)** — *does the conclusion hold for the model's output, not just its hidden
state?* A hidden state could move in directions the head ignores. So we re-score every ray on the
**Jensen–Shannon divergence** (a symmetric, bounded measure of distance between probability
distributions) between baseline and perturbed next-character distributions `p_0, p`:

```math
\mathrm{JSD}(p_0 \Vert p) = \tfrac{1}{2}\,\mathrm{KL}\big(p_0 \Vert m\big) + \tfrac{1}{2}\,\mathrm{KL}\big(p \Vert m\big), \quad m = \tfrac{1}{2}(p_0 + p)
```

We compute PI on the JSD curve too; hidden-state and JSD conclusions must agree (`ΔPI (JSD)` column
in `RESULTS.md`). We also record the fraction of rays whose top-1 prediction flips at the largest
radius — the calibration that the radius range is functionally large.

**Group comparison** — *is any shape difference specific to trained representations?* A raw PI could
reflect generic architecture geometry, so the pre-registered effect is a difference of medians
against the matched control (below), with a hierarchical bootstrap (contexts, then directions) for
the 95% CI:

```math
\Delta \mathrm{PI} = \mathrm{median}\big(\mathrm{PI}_{\mathrm{natural}}\big) - \mathrm{median}\big(\mathrm{PI}_{\mathrm{control}}\big)
```

Effect size is Cliff's delta (fraction of natural-vs-control pairs where natural is larger, minus the
reverse; `|delta| > 0.474` = "large"). A tiny-but-significant `ΔPI` is inconclusive on its own — the
**sign of PI itself** decides whether a plateau exists. Consumed in Figure 3 (right) and the
`RESULTS.md` table.

### Baselines

**Matched off-distribution control.** For each natural basepoint we draw a random vector from the
per-block diagonal Gaussian (mean `mu_l`, per-coordinate std `sigma_l` of held-out activations) and
rescale it to the natural activation's norm, keeping the rest of the sequence fixed:

```math
h_{\mathrm{ctrl}} = \big(\, \sigma_l \odot \varepsilon + \mu_l \,\big)\cdot \frac{\lVert h \rVert_2}{\lVert \sigma_l \odot \varepsilon + \mu_l \rVert_2}, \qquad \varepsilon \sim \mathcal{N}(0, I)
```

A plateau specific to *trained* representations should be stronger for natural `h` than for this
norm-and-statistics-matched random point.

**Linear reference.** The straight line `y = x`. PI is defined as the signed area against this
reference, so it is built into the metric; it appears as the dashed line in Figure 4.

### Calibration checks (pre-registered; all passed)

- **The assay can detect a plateau:** synthetic flat-then-steep curve → `PI = +0.33`; linear curve →
  `PI = 0.00`; the delayed curve is 3.2× sharper (unit test in `assay.py`).
- **`alpha = 0` fidelity:** the partial forward from block `l+1` matches the full forward pass to
  max logit error `< 1e-3` (blocks 0/5/10; also validates batching).
- **Radius range large enough:** at the frozen `rho_max`, top-1 flips for ≥81% of rays at every block.
- **Not an averaging artifact:** individual rays inspected (Figure 4).
- **Metric agreement:** hidden-state and JSD scores agree qualitatively.

## Results

**Training.** Val loss 1.494 / accuracy 0.560 — clearly a trained network, not a random one.

![Figure 1 — Training curves: cross-entropy (y, left) falls to ~1.49 on validation; next-char accuracy (y, right) rises to 0.56; x = training step.](plots/training_curves.png)

**No plateau at any block.** At all 11 blocks the median `PI` of natural activations is **negative**
(−0.15 to −0.30): the response rises quickly, then saturates — the opposite of flat-then-steep. Both
natural and control curves sit **above** the linear diagonal (concave).

![Figure 2 — Downstream response d_hidden (y) vs normalized radius rho (x), per block. Natural (blue) and matched-control (red) curves are both concave/saturating, above the linear diagonal — no plateau at any block.](plots/response_by_layer.png)

**The natural-vs-control difference is real but not a plateau.** `ΔPI` is positive at every block
with 95% CIs excluding zero (peak `+0.096`, Cliff's δ `+0.91`, blocks 2–3): natural activations
saturate *slightly less* than random controls. The JSD-based `ΔPI` agrees in sign everywhere. But
both PIs are negative, so this is a difference between two non-plateau shapes — mild on-manifold
structure, not a plateau.

![Figure 3 — Plateau index by block (x = block). Left: median PI (y) is negative for both natural and control at all 11 blocks. Right: ΔPI = nat − ctrl (y) is small, positive, significant, and decays with depth.](plots/plateau_score_by_layer.png)

**Sharpness shows no late wall.** Mean sharpness of natural rays is 2.2–4.0 (linear = 1.0; synthetic
plateau = 3.2), but with `PI < 0` the steepest segment is the *initial* rise near `rho = 0`, not a
late boundary. Natural rays are also *less* sharp than control at every block (e.g. 2.75 vs 3.51 at
block 0; 4.01 vs 4.91 at block 10) — no trained-representation-specific wall.

**Not an averaging artifact.** Every individual ray is concave; none is flat-then-steep.

![Figure 4 — Individual rays: response (y) vs normalized radius (x); blue = natural, red = control, dashed = linear reference. Every ray is concave/saturating.](plots/individual_curves.png)

Per-block numbers are in `RESULTS.md`; the per-ray table is `results/tidy_results.csv`.

## Conclusion

Under a pre-registered assay that provably *can* detect a plateau, the reconstructed 12-layer
Shakespeare GPT shows **no activation plateaus** at its final-position residual stream: responses are
smooth and saturating at every block. The significant natural-vs-random difference reflects slightly
weaker saturation for real activations — mild on-manifold structure, not a plateau. **No-go** for a
plateau-mapping follow-up on this model.

**Interpretation.** Plateaus are natural in piecewise-linear ReLU MLPs (where the paper's MNIST
result lives): a saturated linear region is literally flat. A residual GPT with GeLU MLPs and
LayerNorm passes perturbations through an approximately additive residual stream, which produces
linear-to-saturating responses — exactly what we observe. The plateau phenomenon may be
architecture-specific rather than universal.

**Limitations.**
1. **Reconstruction, not the paper's checkpoint.** A negative on our reconstruction cannot prove the
   paper's exact model lacks plateaus; it does show a faithful, standard build of that architecture
   has none.
2. **Training length.** We trained to a solid but not grokking-scale point (accuracy 0.56) under a
   shared-GPU budget. Whether plateaus emerge only after much longer training is the remit of the
   "during training" direction, not this gate.
3. **Scope.** We probed the final position with random directions; a plateau confined to other
   positions, learned directions, or another hook point could be missed, though the per-ray and
   multi-block evidence argues against a strong effect.
