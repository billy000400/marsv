# RESULTS — Does the 12-layer Shakespeare GPT show Matthew-style activation plateaus?

> CURRENT-BEST ONLY. History lives in CHANGELOG.md. Read before rewriting.

## Question & verdict

**Question (a go/no-go gate).** Interpolate between the last-position activations of two natural
inputs that are identical except for their final character. Does the downstream output stay close to
endpoint A, cross a boundary rapidly, then stay close to endpoint B — the **activation plateau**
phenomenon of Matthew Shinkle & StefanHex's post *Activation Plateaus: Where and How They Emerge* —
in the 12-layer, 12-head character-level Shakespeare GPT of *Deep Networks Always Grok* (Figure 9)?

**Verdict: YES — Matthew-style plateaus are present (qualified).** In a faithful reconstruction of
the Figure-9 model (the paper's exact GPT code/checkpoint is **not publicly released** — audited
2026-07-15), **14/40 frozen minimal pairs** show plateau–boundary–plateau structure in raw individual
final-logit curves under a strict preregistered rule, and most remaining pairs show the same sigmoid
shape with a wider boundary. Both predicted plateau signatures hold, judged on the same `d(t)` curve
against the **diagonal** `d = t` — the straight line the curve traces when the output morphs
uniformly with no plateaus (transition width 0.8): recording the curve at successively **deeper
layers moves it monotonically away from the diagonal** (the boundary sharpens with depth), while
moving the **interpolation point later** (fewer layers left downstream to build the plateau)
**collapses it back onto the diagonal**. "Qualified" because we tested a reconstruction, not the
paper's exact checkpoint. The grok paper's own phenomenon — **grokking**, i.e. `ε=0.03`-PGD
adversarial robustness emerging long after training accuracy saturates, together with a **second
local-complexity descent** — is enforced as an explicit **validity gate** (next section). **All three
models we trained FAIL that gate** (the 3,500-step pilot, a fresh 30k-step character run, and a fresh
30k-step BPE run): each shows a first LC descent and emerging robustness, but **no second descent**.
So the **bounded joint verdict is "primary relationship not testable" (PLAN case 5)** — no run
reproduces Figure 9 (in particular the BPE run needed for Matthew's exact tokens), so plateaus cannot
be tied to a grokking transition. The plateau result below therefore **stands on its own** and is
**not** joined to a Grokking claim.

## Model actually tested

Reconstruction 12-layer/12-head GeLU GPT (`d_model=240`, context 128, 8.38M params), trained on Tiny
Shakespeare to **val loss 1.494, next-char accuracy 0.560** (≈37× the 1/65 chance rate). Full
provenance (corpus SHA-256, seeds, config) in `results/train_meta.json`; confirmed-vs-reconstructed
fields in `MODEL_SPEC.md`.

![Training curves: cross-entropy loss falls to ~1.49 on validation (left); next-char accuracy rises to 0.56 (right); x = training step.](plots/training_curves.png)

## Figure-9 grokking gate (validity gate — S3 pilot + S4/S5 fresh runs: all FAIL)

PLAN makes a **validity gate** mandatory before any *joint* Grokking↔plateau claim: the model must
qualitatively reproduce *Deep Networks Always Grok* Fig. 9 — a **second local-complexity (LC) descent**
that begins before the test-accuracy peak, together with **delayed adversarial robustness** (`ε=0.03`
`ℓ∞`-PGD accuracy rising while that second descent continues). LC is the paper's sign-crossing count
summed over the 12 GeLU layers on 1,024 train/test/random points (`r=0.005`, `P=25`, 99% CIs); the
pipeline is source-locked to the official repo (`experiments/fig9.py`, 0.0 logit-reimplementation
error). The preregistered PASS/FAIL/NOT-ESTABLISHED rule is in `experiments/fig9_verdict.py`.

**Pilot (existing 3,500-step char GPT, 13 log-spaced checkpoints): verdict = FAIL within its horizon.**

| Figure-9 quantity (pilot) | value |
|---|---|
| clean next-char accuracy (final / peak) | 0.564 / 0.564 (peak at last ckpt, step 3500) |
| `ε=0.03` PGD adversarial accuracy (final / max) | **0.327 / 0.327** — delayed robustness *did* emerge |
| test LC (first ckpt → minimum) | 1940 → 68 at step 3500 (minimum at the **last** checkpoint) |
| second LC descent present? | **No** — LC is still in its first monotone descent at the horizon |
| **Preregistered verdict** | **FAIL** (valid measurements, Figure-9 ordering absent within 3,500 steps) |

Read this honestly: the pilot shows the *first* LC descent and emerging robustness, but the horizon
ends before any *second* descent could appear, so it cannot support a joint claim.

**Fresh 30k-step char + BPE runs — both also FAIL (S4, S5).** We trained two matched fresh runs and
ran the identical LC/PGD pipeline across log-spaced checkpoints. Both reproduce the pilot's pattern:

| Figure-9 quantity | Pilot char (3.5k) | **Fresh char (30k)** | **Fresh BPE (10k)** |
|---|---|---|---|
| checkpoints evaluated | 13 | 14 | 10 |
| clean acc (peak / final) | 0.564 / 0.564 | 0.568@4994 / 0.554 | 0.299@831 / 0.274 |
| `ε=0.03` PGD adv acc (final) | 0.327 | **0.528** | **0.187** |
| test LC (first → min → final) | 1940 → 68 → 68 | 1940 → **8.1** → 8.1 | 2182 → **95** → 95 |
| second LC descent? | No | No | No |
| delayed robustness emerged? | Yes | Yes | Yes |
| **preregistered verdict** | **FAIL** | **FAIL** | **FAIL** |

The fresh char run reaches *higher* adversarial accuracy than the pilot (0.53 vs 0.33), so robustness
clearly emerges — but test LC descends **monotonically** to its minimum (8.1) at the last checkpoint,
never rising to produce a second descent. Both fresh runs **overfit** (val loss bottoms early — char
≈step 3,750, BPE ≈step 750 — then rises while train loss falls), the opposite of grokking's delayed
val-loss recovery. Robustness here is a property of the first fold-collapse of a memorising network,
not the grokking second-descent event.

![Pilot char (3.5k) Figure-9 curves. Left y-axis: local complexity (sign-crossings summed over 12 GeLU layers) for train (blue), test (orange), random (green), 99% CI bands. Right y-axis: next-token accuracy — black = clean test accuracy, red dashed = ε=0.03 PGD adversarial accuracy. x-axis: training step (log scale, step 0 at 1). LC monotone to the 3,500-step horizon (no second descent); adv accuracy climbs to 0.33; verdict FAIL.](plots/grokking_pilot_char.png)
![Fresh char (30k) Figure-9 gate. Same axes as the pilot panel. LC monotone to 8.1 while adversarial accuracy climbs to 0.53; no second descent → FAIL.](plots/grokking_fresh_char.png)
![Fresh BPE (10k) Figure-9 gate. Same axes. LC monotone to 95 while adversarial accuracy climbs to 0.19; no second descent → FAIL.](plots/grokking_fresh_bpe.png)

**Joint timeline & bounded relationship verdict (S7) = "primary relationship not testable" (PLAN case
5).** On one training-step axis the Grokking side is uniform across runs — LC falls monotonically and
PGD robustness rises, with no second descent anywhere. Because the BPE bridge to Matthew's exact tokens
does not reproduce Figure 9 (nor does either character run), we cannot test whether the plateau assay
sharpens during a second-descent/robustness window: that window never opens. The plateau phenomenon is
present in the char reconstruction (below) and stands alone; it is not tied to grokking here.

![Joint checkpoint timeline. Left: test local complexity (y) vs training step (x, log) for pilot char (gray), fresh char (blue), fresh BPE (red); legend gives each run's Figure-9 verdict. Middle: ε=0.03 PGD adversarial accuracy (y) vs step (x, log), same colors; dashed = 0.05 robustness threshold. Right: text summary of the three FAIL verdicts, the plateau reference from the reconstruction, and the bounded relationship verdict. No second LC descent in any run → primary relationship not testable.](plots/joint_timeline.png)

## Assay (frozen before any curve was inspected)

- **Pairs:** 40 minimal pairs from held-out (val) Shakespeare; each is `prefix + char_A` vs
  `prefix + char_B`, length 128, differing only in the final character. `char_A` = the character
  actually observed after the prefix in text; `char_B` = the model's top next-char prediction
  (top-2 if it equals `char_A`). Seed 20260717; prefixes deduplicated; 0 pairs excluded by the frozen
  degeneracy threshold (endpoint logit distance < 1e-3); endpoint logit distance median 24.7, range
  [8.7, 64.4]. Frozen in `results/prompt_pairs.json`.
- **Interpolation:** norm-interpolating slerp between final-position `resid_post` activations at
  block 0, patched into the final position, 101 evenly spaced `t` in [0,1], recording `d(t)` =
  Matthew's relative distance in final-logit space (definitions in REPORT.md Methods).
- **Plateau rule (frozen):** transition width `w_10→90 ≤ 0.25` AND the curve spends ≥10% of the path
  near each endpoint (`t_lo ≥ 0.10`, `t_hi ≤ 0.90`) on a near-monotone curve (max isotonic deviation
  ≤ 0.10). Diagonal reference `d = t` has `w = 0.8`.

## Main result — individual final-logit curves (interpolation after block 0)

| Statistic | Value |
|---|---|
| Pairs meeting the strict plateau rule | **14 / 40** (IDs 0, 4, 5, 6, 7, 9, 14, 20, 21, 22, 28, 34, 36, 37) |
| Pairs with `w_10→90 ≤ 0.35` | 24 / 40 |
| Near-diagonal pairs (`w ≥ 0.6`) | 2 / 40 (#10, #19) |
| Non-monotone curves | 0 / 40 |
| Median `w_10→90` (all pairs) | 0.309 (range [0.110, 0.773]) |

Every curve is individually sigmoid-or-diagonal — the plateau count is not an averaging artifact:

![Raw Matthew relative distance d(t) (y-axis) vs interpolation step t (x-axis) in final-logit space for every frozen pair (one panel per pair; title = pair ID, endpoint characters, transition width w). Gray dashed line = diagonal d = t (non-plateau reference). Most curves hug d≈0, cross rapidly near t≈0.5, then hug d≈1.](plots/pair_curves_logits.png)

## Layerwise emergence — the boundary sharpens with depth

Interpolation fixed after block 0; `d(t)` recorded at the final position of every later block
(`resid_post`) and at final logits. Median transition width **falls monotonically with depth**, and
only the final logits pass the strict rule:

| Recording point | median `w_10→90` | pairs passing plateau rule |
|---|---:|---:|
| resid_post block 1 | 0.777 | 0 |
| resid_post block 3 | 0.722 | 0 |
| resid_post block 5 | 0.647 | 0 |
| resid_post block 7 | 0.581 | 0 |
| resid_post block 9 | 0.527 | 0 |
| resid_post block 11 | 0.445 | 0 |
| **final logits** | **0.309** | **14** |

![Layerwise emergence for four fixed representative pairs (IDs 0–3, frozen before inspection): d(t) (y) vs t (x); line color = recording block (dark = early, light = late, per colorbar); red = final logits; gray dashed = diagonal. Curves start near-diagonal at early blocks and sharpen into a plateau–boundary–plateau shape by the final logits.](plots/layerwise_emergence.png)

## Depth comparison — later interpolation weakens the plateau (as predicted)

Recording fixed at final logits; interpolation moved across blocks {0, 2, 4, 6, 8, 10}. Median
`w_10→90` over the 40 pairs rises toward the diagonal reference (0.8) as fewer layers remain
downstream:

| Interpolation block | 0 | 2 | 4 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| median `w_10→90` | 0.309 | 0.564 | 0.647 | 0.733 | 0.757 | 0.802 |

![Left: median final-logit d(t) (y) vs t (x) for each interpolation block (color = block, dark = 0, light = 10); the block-0 curve is strongly sigmoid, later blocks approach the gray dashed diagonal. Right: median transition width w_10→90 (y, IQR bars) vs interpolation block (x); red dashed = plateau bar 0.25, gray dashed = diagonal reference 0.8.](plots/interpolation_layer_comparison.png)

## Implementation checks (all passed)

- `t=0` / `t=1` patched forwards reproduce the direct unpatched endpoint forwards (max logit error
  < 1e-3); `d(0) < 1e-4`, `d(1) > 1 − 1e-4` for every pair.
- Prefix positions differ only at the final character; all earlier-position activations of A and B
  match at every block (max abs diff < 1e-4).
- Batched interpolation matches a single-example reference to < 1e-5.
- Synthetic step-like path detected as narrow-transition (w = 0.089); synthetic linear path rejected
  (w = 0.800).
- Slerp endpoints exact; norms interpolate linearly; documented near-collinear fallback.

## Headline

The reconstructed 12-layer character-level Shakespeare GPT **does show Matthew-style activation
plateaus**: 14/40 frozen natural minimal pairs give raw individual plateau–boundary–plateau curves in
final-logit space (median transition width 0.309 vs diagonal 0.8), the boundary sharpens monotonically
through the 11 downstream blocks, and it weakens toward the diagonal when interpolation happens later.
The plateau-mapping follow-up **is warranted** for this model (qualified: a reconstruction, not the
paper's exact checkpoint).
