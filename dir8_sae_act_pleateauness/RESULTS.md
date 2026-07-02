# RESULTS — Direction 8: Plateau-ness as a test for SAE activation validity

## Setup (current best)
- **Model:** GPT-2 small (124M), HuggingFace `gpt2`.
- **Hook / layer:** `blocks.6.hook_resid_pre` (= output of transformer block 5), last
  non-padding token, captured via a full in-context forward hook.
- **SAE:** `jbloom/GPT2-Small-SAEs-Reformatted`, `blocks.6.hook_resid_pre`, d_in=768,
  d_sae=24576. Encoder convention chosen empirically: subtract `b_dec` before encoding
  (mean recon error 27.7 vs 380.8 — so `apply_b_dec_to_input=True`).
- **Data:** FineWeb text cache (`../dir3_manifold/data/fineweb_texts.json`), seq len 64,
  prompts with ≥16 real tokens. **N = 200** source prompts.
- **Plateau metric (primary):** `plateau_auc_low` — isotropic perturbation
  `x(r)=x+r||x||d`, in-context downstream `KL(p_x || p_x(r))` at the last token,
  monotone-cumulative-max over `r`, area of `clip(1 − KL_mono/τ, 0, 1)` vs radius,
  averaged over directions. Radius grid `r ∈ {0, .0025, .005, .01, .02, .04, .08}`.
  Higher = wider low-response plateau.
- **τ calibration:** `τ = median real KL @ r=0.02`. Stage A used all real (smoke);
  **Stage B uses a held-out real split** (calibrate on sources `<N/2`, score on sources
  `≥N/2`), so τ never sees the scored activations. `τ_heldout = 1.33e-4`.
- Conditions paired by source prompt: **real**, SAE **recon**, **naive** independent latent
  composition (k∼empirical L0, idx∼activation frequency, coef∼empirical active coefficients,
  decoded), **sparse_match** (naive but `k=`source L0 and coefficients rescaled to the
  source's active-coefficient RMS), **norm_rand** (random direction at the real norm), and
  the **iso_displace** reference family (below).

## Stage A — real / reconstruction vs naive synthetic plateau (N=200, 8 directions)

| Condition | plateau_auc_low (median) | norm (median) | L0 (median) |
|---|---|---|---|
| real      | **0.200** | 79.5 | 32.5 |
| recon     | **0.162** | 68.4 | 32.5 |
| naive     | 0.066 | 60.0 | 34.5 |
| norm_rand | 0.035 | 79.5 | 0 |

Paired difference vs real (median paired gap, 95% paired-bootstrap CI over source prompts):

| real − condition | median gap | 95% CI | verdict |
|---|---|---|---|
| real − recon     | 0.017 | [0.010, 0.024] | small but nonzero — recon slightly less flat |
| real − naive     | 0.122 | [0.108, 0.138] | large — naive compositions far less flat |
| real − norm_rand | 0.160 | [0.148, 0.176] | largest — norm-matched random least flat |

**Reproduced (H1):** real and SAE reconstructions have substantially wider downstream-response
plateaus than naive independent SAE-latent compositions and norm-matched random activations.
Real ≳ recon ≫ naive > norm_rand; every contrast CI excludes zero. The effect is **not a norm
artifact**: norm_rand carries the *same* norm as real (79.5) yet is flattest, and within the
synthetic conditions the lower-norm `naive` is *flatter* than the higher-norm `norm_rand` —
the opposite of a norm shortcut. Pooled Spearman(plateau, norm) = +0.06.

![Stage A: plateau curves and per-condition AUC](plots/plateau_stageA.png)

## Stage B — distance-to-source-matched control (the decisive test)

Stage A's gap is tightly coupled to **distance to the source real activation**: recon sits
at median distance 25, naive at 64, norm_rand at 113. We therefore ask: at a *matched*
distance from a real activation, do SAE-decoded conditions plateau more than a purely
**random displacement** of a real activation?

**iso_displace reference** — `x_real + δ·d` (isotropic unit `d`; distance ≡ δ exactly).
This traces plateau as a function of distance for random off-manifold displacement of a real
activation (N_eval=100, 6 directions, held-out τ):

| δ (distance) | 15 | 30 | 60 | 120 |
|---|---|---|---|---|
| iso_displace plateau_auc_low | 0.184 | 0.173 | 0.128 | 0.078 |

Plateau falls monotonically with distance for random displacement alone — distance-to-source
**by itself** reproduces most of the Stage A ordering.

**Distance-matched residual** = (condition plateau) − (iso_displace reference interpolated at
that condition's own distance). `>0` means flatter than a random point at equal distance;
`<0` means less flat. Median over eval sources, 95% bootstrap CI:

| condition | median dist | plateau | reference @ dist | residual | 95% CI | verdict |
|---|---|---|---|---|---|---|
| recon        | 25.1 | 0.157 | 0.176 | **−0.016** | [−0.021, −0.003] | ≈ on the random-displacement curve (tiny deficit) |
| naive        | 64.0 | 0.065 | 0.124 | **−0.058** | [−0.065, −0.053] | well **below** random displacement |
| sparse_match | 64.1 | 0.068 | 0.124 | **−0.063** | [−0.067, −0.049] | well **below** random displacement |

Pooled Spearman(plateau, distance) over the SAE + iso conditions (eval half) = **−0.64**.

**Readings:**
- **No SAE-decoded condition plateaus *above* the random-displacement reference at matched
  distance.** The Stage A "real/recon plateau more than naive" gap is fully accounted for by
  (i) distance-to-source plus (ii) a naive-specific deficit — not by any special plateau
  *validity* of SAE activations.
- **recon** sits essentially *on* the reference curve (residual −0.016): its plateau is
  explained by how close the reconstruction is to the real activation, with no extra credit.
- **naive** and **sparse_match** sit clearly *below* the reference: at the same distance from
  a real activation, an SAE independent composition plateaus *less* than a random point — the
  composition is actively more downstream-sensitive, not flatter.
- **Sparsity/coefficient matching does not help** (sparse_match ≈ naive, residual −0.063 vs
  −0.058). The naive deficit is **not** caused by marginal L0/coefficient mismatch.

![Stage B: plateau vs distance and distance-matched residual](plots/plateau_stageB.png)

## Stage B-dir — is the below-random deficit isotropic-only? (direction-family robustness)

Stage B used **isotropic** perturbation directions. If SAE-decoder directions *reversed* the
finding (SAE-decoded conditions flatter than random along their own feature directions), the
metric would be direction-dependent and the null would need scoping. We recompute the exact
Stage B distance-matched residual under three perturbation-direction families, apples-to-apples
in one run (N=200, N_eval=100, 8 directions, held-out τ per family):

- **iso** — isotropic Gaussian unit directions (primary; reproduces Stage B);
- **sae_single** — each direction is a single unit-normed SAE decoder column `W_dec[j]`, `j`
  drawn from real-code-active features by frequency;
- **sae_sparse** — each direction is a normalized signed sum of 8 random active decoder columns.

Distance-matched residual `ρ_c` (median over eval sources, 95% bootstrap CI). `<0` = **less**
flat than a random displacement at equal distance:

| family | recon `ρ` | naive `ρ` | sparse_match `ρ` |
|---|---|---|---|
| iso (primary) | −0.015 [−0.025, +0.003] | −0.061 [−0.068, −0.057] | −0.062 [−0.069, −0.052] |
| sae_single | −0.016 [−0.029, −0.003] | −0.066 [−0.071, −0.058] | −0.062 [−0.068, −0.052] |
| sae_sparse | −0.015 [−0.032, +0.006] | **−0.077** [−0.084, −0.065] | **−0.071** [−0.076, −0.063] |

Pooled Spearman(plateau, distance) over the SAE + iso conditions: iso −0.64, sae_single −0.60,
sae_sparse −0.62 — distance dominates plateau under every family.

**Reading:** the finding is **direction-family robust**. Across all three families the ordering
is identical — recon sits essentially *on* the random-displacement curve (residual ≈ −0.015,
CI straddling or barely below 0), while **naive and sparse_match sit clearly below random**. No
family reverses the conclusion; if anything the naive below-random deficit is *larger* along SAE
decoder directions (sae_sparse −0.077 vs iso −0.061), the opposite of SAE-specific plateau
validity. The Stage B null is therefore not an isotropic artifact.

![Stage B-dir: distance-matched residual by perturbation-direction family](plots/plateau_stageB_dir.png)

## Stage C — do *improved* synthetic codes recover plateau? (H3)

Stage B left one escape hatch: maybe naive compositions plateau below random only because they
lack **higher-order** latent structure. Stage C asks whether **co-occurrence-aware** or
**cycle-consistent** synthetic codes climb *above* the same random-displacement reference at
matched distance (N=200, N_eval=100, 6 directions, held-out τ). Three new conditions:

- **cooc** — take a *real* example's active feature **set** (support), assign coefficients from
  the empirical active-coefficient marginal (exactly as `naive`). Isolates **support
  co-occurrence** as the only change over `naive` (which draws indices independently by frequency).
- **cycle_consistent** — **filter** naive candidates to encode–decode self-consistency: keep only
  those with relative cycle error `||encode(decode(z))−z|| / ||z||` below the **75th percentile of
  real-code cycle error** (τ_cyc = 0.342). Real codes: median cycle error 0.239; naive: 0.780;
  pass rate 0.56% (36 000 generated → 200 kept, selected median 0.307).
- **cooc_full** — a genuine **real-derived** code (SAE reconstruction of *another* real example),
  paired to the source only for distance. A positive control: an actual on-manifold activation
  placed *far* from the paired source.

Distance-matched residual `ρ_c` = (condition plateau) − (random-displacement reference at that
condition's distance). `>0` = flatter than a random point at equal distance. Median, 95% CI:

| condition | median dist | plateau | ref @ dist | residual `ρ_c` | 95% CI | verdict |
|---|---|---|---|---|---|---|
| recon (ref) | 25.1 | 0.164 | 0.176 | −0.012 | [−0.028, −0.004] | ≈ on the random curve |
| naive (ref) | 64.0 | 0.068 | 0.121 | −0.054 | [−0.061, −0.048] | **below** random |
| **cooc** | 67.3 | 0.070 | 0.117 | **−0.044** | [−0.049, −0.036] | **below** random |
| **cycle_consistent** | 64.9 | 0.077 | 0.120 | **−0.043** | [−0.049, −0.040] | **below** random |
| **cooc_full** | 69.8 | 0.159 | 0.115 | **+0.043** | [+0.035, +0.056] | **ABOVE** random |

**Readings:**
- **Neither improved construction recovers plateau.** Co-occurrence support (`cooc`, −0.044) and
  encode–decode cycle-consistency (`cycle_consistent`, −0.043) both stay **clearly below** the
  random-displacement curve — only marginally above `naive` (−0.054) and nowhere near it. Enforcing
  realistic support co-occurrence or self-consistency under the SAE map is **not** enough to make a
  synthetic code as downstream-flat as a random point at equal distance. **H3 is negative** for
  constructible synthetic codes.
- **The one thing that works is a genuine real activation.** `cooc_full` — a real-derived SAE
  reconstruction — sits **above** the random reference (+0.043), even at a large distance from the
  paired source. This is the positive control Stage B lacked: a real, downstream-valid activation is
  flatter than a random displacement at *any* matched distance, so plateau tracks genuine-activation
  validity, not merely distance to the specific source.
- **The missing ingredient is real-activation (manifold) membership, which code-space constraints do
  not synthesize.** Consistent with the PLAN's H3 null interpretation: the gap "may require
  higher-order model-computation compatibility," not latent marginal realism or self-consistency.
  Also consistent with Stage D (a real activation occupies a locally-robust region; a constructed
  code does not).

![Stage C: improved synthetic codes vs random-displacement reference](plots/plateau_stageC.png)

## Stage D — does plateau predict downstream validity *beyond baselines*? (the project gate)

**Independent downstream-validity target.** For each candidate activation `x_c` paired to a
source prompt, overwrite the last-token residual in full context and measure
`output_kl = KL(p_real || p_candidate)`, where `p_real` is the next-token distribution when
the *real* source activation sits in context and `p_candidate` when `x_c` does. **Low
`output_kl` = downstream-valid** (the candidate makes the model behave like the real
activation). This target is independent of provenance labels.

**Pool.** Every candidate condition × source as one row: recon, naive, sparse_match, and the
iso_displace random-displacement family at distances 15/30/60/120 (7 conditions × N=200 = 1400
rows). **Split by source prompt** (train sources `<N/2`, test sources `≥N/2`) so no source
leaks across. Predict `log₁₀ output_kl` with plain linear least squares; report **held-out
test R²**. Baselines: distance-to-source, norm. A **local-sensitivity** baseline `locsens` =
`log₁₀` of the fixed-radius mean KL at `r=0.02` (Direction-6 `plateau_kl`) is added as the
decisive discriminator: plateau itself is a local-sensitivity measure, so the real question is
whether its plateau *shape* (the AUC) adds anything beyond a single local-sensitivity number.

| model (features) | held-out test R² | Spearman(pred, true) |
|---|---|---|
| baseline (dist, norm) | 0.795 | 0.874 |
| plateau only | 0.498 | 0.851 |
| **+plateau** (dist, norm, plateau) | **0.869** | 0.915 |
| baseline + locsens (dist, norm, locsens) | 0.873 | 0.924 |
| **+plateau** (dist, norm, locsens, plateau) | **0.878** | 0.924 |

| added-value test | ΔR² | partial Spearman (held-out) |
|---|---|---|
| plateau beyond **{dist, norm}** | **+0.073** | **−0.65** |
| plateau beyond **{dist, norm, locsens}** | **+0.005** | **−0.16** |

Marginal held-out Spearman vs `log₁₀ output_kl`: plateau **−0.85**, locsens **+0.84**,
dist +0.75, norm −0.22.

**Readings:**
- **Beyond distance+norm, plateau *does* predict downstream validity** (ΔR²=+0.073, partial
  ρ=−0.65): flatter candidates are more downstream-valid at matched distance/norm. This
  replicates Direction 6's "plateau predicts downstream KL beyond movement distance".
- **But that predictive value is entirely local sensitivity.** A single fixed-radius KL
  (`locsens`) already captures it (marginal ρ=+0.84), and once `locsens` is in the model
  plateau adds **essentially nothing** (ΔR²=+0.005, partial ρ=−0.16). Plateau's AUC *shape*
  carries no downstream-validity information beyond one local-sensitivity number.
- **Verdict: robustness, not interpretability validity** (decision-table row: *"Primary metric
  adds little beyond local sensitivity → report as robustness, not interpretability validity"*).

![Stage D: validity target vs plateau, and held-out R² with/without local sensitivity](plots/plateau_stageD.png)

## Current verdict (project-level)
- **H1 (Stage A):** supported — real/recon activations plateau more than naive synthetic
  compositions, and this is not a norm artifact (norm-matched random is *flattest*; pooled
  Spearman(plateau, norm)=+0.06).
- **H2 (Stage B):** the plateau gap **does not survive distance-to-source matching as an
  SAE-validity signal.** recon's advantage is explained by closeness to the real activation;
  no SAE-decoded condition is flatter than a random displacement at equal distance; and
  sparsity/coefficient matching does not recover plateau. **This is direction-family robust**
  (Stage B-dir): the naive/sparse below-random deficit persists — slightly stronger — along
  single- and sparse-sum SAE-decoder directions, so it is not an isotropic artifact.
- **H3 (Stage C):** **improved synthetic codes do not recover plateau.** Neither co-occurrence
  support matching nor encode–decode cycle-consistency lifts a synthetic code above the
  random-displacement reference at matched distance (both ≈ −0.04, still below random). Only a
  genuine real-derived code (`cooc_full`) plateaus *above* random (+0.043) — so the missing
  ingredient is real-activation manifold membership, not latent-code marginal realism or
  self-consistency.
- **H4 (Stage D):** plateau predicts the independent downstream-validity target beyond
  distance+norm, **but this is fully explained by local sensitivity** — plateau adds nothing
  beyond a single fixed-radius KL. So plateau-ness measures **local robustness**, not
  SAE-specific downstream validity.

**Project conclusion (null, and it names the failing notion).** In this GPT-2-small
resid_pre@6 / jbloom-SAE setup, plateau-ness is **(i) a closeness-to-real proxy and (ii) a
local-robustness / local-sensitivity measure** — *not* an independent SAE interpretability-
validity diagnostic. Of the four candidate notions (synthetic-provenance detection,
off-distribution detection, local downstream-invalidity detection, mere local robustness), the
evidence says plateau-ness is **mere local robustness** plus distance-to-real. This is
consistent with Direction 9 (plateau-as-OOD weak) and Direction 6 (plateau predicts downstream
KL but only as local sensitivity).

_Scope / open:_ one primary metric, one layer/SAE. Direction-family robustness is **confirmed**
(Stage B-dir). **Improved synthetic constructions are tested and do not overturn the null**
(Stage C: co-occurrence-aware and cycle-consistent codes stay below random; only genuine
real-derived codes plateau above it). Not run: an alternate-layer generalization (Stage E) — given
the local-sensitivity, direction-robustness, and improved-code results, it would *scope* the
conclusion but is not expected to overturn the project-level null.

_Artifacts:_ `results/plateau_metrics.csv`, `results/plateau_summary.json` (Stage A);
`results/stageB_metrics.csv`, `results/stageB_summary.json` (Stage B);
`results/stageB_dir_metrics.csv`, `results/stageB_dir_summary.json` (Stage B-dir);
`results/stageC_metrics.csv`, `results/stageC_summary.json` (Stage C);
`results/stageD_metrics.csv`, `results/stageD_summary.json` (Stage D).
