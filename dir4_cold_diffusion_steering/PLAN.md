# PLAN — Direction: ColdSteer — on-manifold correction for activation steering

> Working folder: `dir4_cold_diffusion_steering`. Agent REWRITES "Current status"/"Next step" + ticks stages each
> iteration. Disk (PLAN/JOURNAL/RESULTS/CHANGELOG + ../BUDGET.md + ../CLAUDE.md) is the only memory.
> Full research proposal is preserved verbatim below the plan sections.

## Success criterion (definition of "done")
RESULTS.md/REPORT.md show a **projection-preserving corrector** `ĥ = z + P_{v⊥}r_θ` that, at a
matched steering projection along `v`, reduces off-manifold damage (Mahalanobis `D_M` and ΔLM loss)
versus raw steering `h+αv` — with a clear verdict, Methods+equations, and figures. A well-supported
negative result ("corrector cannot beat raw steering at matched projection") also counts as done.

## Fallback (if time runs short)
The already-delivered **Experiment 1** (off-manifold phenomenon + 3 metrics + figure) is a
self-contained result. Minimum acceptable = that, finalized in REPORT.md.

## Setup (fixed / self-contained — NO external GLP repo)
- **Model:** GPT-2 small (124M), HF `transformers`. Import `from transformers import GPT2LMHeadModel,
  GPT2TokenizerFast` (top-level `import transformers` is broken: hf_hub version skew). Weights cached.
- **Hook:** resid_post block 6 = `hidden_states[7]`. CUDA works on this A10 (use it, frac 0.18).
- **Data:** reuse `../dir3_manifold/data/fineweb_texts.json` (1500 docs). Steering = DiffMean sentiment.
- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md` — read both every iteration.**
- **Deliverable hygiene:** RESULTS.md/REPORT.md = current-best only; CHANGELOG.md = history.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, flax.**

## Stages (checklist)
- [x] S1 — Motivating phenomenon: raw steering `h+αv` goes off-manifold (D_M, norm, ΔLM vs α). DONE.
- [x] S2 — `projections.py` (project_orthogonal / retain_projection_update / cov_aligned_shift) +
        unit tests DONE. (ColdSteerResidualCorrector MLP class deferred to S3-learned.)
- [x] S3 — Corrector evaluation at matched projection DONE. (a) ANALYTIC Gaussian-optimal corrector
        lowers `D_M` but WORSENS `ΔLM` (decoupling/negative). (b) LEARNED `r_θ` MLP trained on the
        DOWNSTREAM LM loss BEATS raw at every α — ΔLM +2.78→+0.44 at α=8 (84% recovery), matched
        projection, moving FURTHER off the Gaussian manifold. Decisive POSITIVE; success criterion met.
- [~] S4 — Generalization + Pareto. (a) α-EXTRAPOLATION DONE: corrector trained on α~U(0.5,8)
        evaluated at α=10,12 (beyond range) still recovers 77%/60% of raw's ΔLM damage — graceful
        degradation, not collapse; in-range α reproduce Exp 3 to the digit. (b) held-out vector /
        concept-strength Pareto still open.
  (each reported metric: produce + save figure to plots/ + define it in REPORT.md Methods)

## Out of scope (do NOT)
- Cloning/installing the GLP repo or its billion-activation datasets (too heavy for our VRAM share);
  GLP-distillation (Strategy 2) is optional and only if S2–S4 land with time to spare.
- Multi-layer / multi-model scaling before a single-layer single-vector result works. No other directions.

## On-track check (required every iteration)
End each JOURNAL.md entry with: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status
S1 + S2 + S3 complete + S4(a) α-extrapolation delivered — success criterion MET.
**S4(a):** the Exp-3 learned corrector (trained α~U(0.5,8)) generalizes BEYOND its training range:
evaluated unchanged at α=10,12 it recovers 77% / 60% of raw steering's ΔLM damage (raw +3.31→+0.76,
+3.74→+1.50), recovery declining smoothly (84→77→60%) — graceful degradation, not collapse; in-range
α reproduce Exp 3 to the digit. Artifacts: `experiments/04_generalization.py`,
`results/04_generalization.json`, `plots/04_generalization.png`. RESULTS/REPORT/CHANGELOG curated;
REPORT math verified (9/9).
<!-- prior -->
S1 + S2 + S3 complete — success criterion MET. Full three-experiment arc: (1) raw steering goes
off-manifold and breaks the LM (ΔLM +2.78 at α=8); (2) analytic Gaussian-optimal corrector lowers
`D_M` but WORSENS `ΔLM` to +4.20 (decoupling/negative); (3) a LEARNED 4-layer MLP `r_θ` trained on
the DOWNSTREAM LM loss (frozen LM, h detached, α~U(0.5,8), matched projection) BEATS raw at every α
— ΔLM +2.78→**+0.44** at α=8 (84% recovery), while moving FURTHER off the Gaussian manifold
(`D_M` 49.0→79.5). The LM-safe correction is off the statistical manifold; only a downstream
objective finds it. Artifacts: `experiments/{projections.py(tests PASS),02_corrector.py,
03_learned_corrector.py}`, `results/03_learned_corrector.json`, `plots/03_learned_corrector.png`.
RESULTS/REPORT/CHANGELOG curated to three-experiment current-best; REPORT math verified (9/9).

## Next step
S4 remaining polish (core + α-extrapolation already delivered). Best next options, any one a clean
iteration: (i) held-out steering vector / second behavior family — build a second DiffMean concept
vector and test whether the sentiment-trained corrector still helps at matched projection on that
direction (tests overfit to one v; the corrector sees v only implicitly through z, so this is a real
generalization probe and either outcome is informative); (ii) text-level concept-strength readout so
the frontier is behavior-vs-fluency, not just ΔLM (projection along v is fixed by construction, so
concept strength is controlled — measure generated-text repetition/quality alongside ΔLM).

# Research Proposal: Cold-Steer â Steering-Corruption Meta-Models for On-Manifold Activation Steering

## 1. Motivation

The GLP paper trains a diffusion-style meta-model over LLM residual-stream activations using Gaussian/flow-matching corruption, then uses the learned prior to post-process steered activations back toward the activation manifold. Its reported results show improved fluency for activation steering, and the authors explicitly note that GLP is unconditional and that conditioning on the clean activation could reduce information loss for steering.

The proposed direction is to replace the generic âadd noise, denoiseâ corruption with the actual corruption that steering creates:

\[
z = h + \alpha v
\]

where \(h\) is a clean activation, \(v\) is a steering direction, and \(\alpha\) is steering strength. This is Cold-Diffusion-like in the sense that Cold Diffusion showed that diffusion-style models can be built around non-Gaussian, even deterministic, degradations rather than only stochastic noise.

The key distinction: **we should not train the model to reconstruct \(h\) from \(h + \alpha v\)**. That would simply learn to remove the steer. Instead, we want a learned correction operator:

\[
C_\theta(h, h+\alpha v, v, \alpha) \rightarrow \hat{h}_{\text{good}}
\]

where \(\hat{h}_{\text{good}}\) is close to the steered activation, preserves the intended semantic shift, but lies in a region that behaves well under the downstream LLM.

## 2. Central Hypothesis

A corrector trained directly on steering-like corruptions will produce a better concept-strength/fluency Pareto frontier than:

1. raw linear steering \(h + \alpha v\);
2. generic GLP post-processing using Gaussian noising plus denoising;
3. simple projection or norm-clipping baselines.

The expected gain is largest at high steering strengths, where raw steering tends to push activations off-manifold.

## 3. Main Research Questions

1. **Does steering-corruption training outperform generic GLP denoising?**  
   Evaluate at matched fluency and matched concept strength.

2. **Can the model preserve the steering direction by construction?**  
   Test hard projection-preserving parameterizations versus soft losses.

3. **Does the corrector generalize?**  
   Hold out prompts, steering strengths, steering vectors, and possibly behavior families.

4. **What supervision works best?**  
   Compare paired activation targets, GLP-distilled pseudo-targets, and direct downstream objectives.

5. **Is the method useful enough to justify extra inference cost?**  
   Measure quality per extra forward pass versus GLP sampling and raw steering.

## 4. Proposed Method

### 4.1 Notation

Let:

\[
h \in \mathbb{R}^d
\]

be a clean activation at layer \(\ell\), token position \(t\). Let \(v_j\) be a normalized steering vector for concept or behavior \(j\). Work in standardized activation coordinates, following the GLP preprocessing convention of subtracting activation mean and dividing by activation standard deviation.

Naive steering gives:

\[
z = h + \alpha v_j
\]

The learned corrector outputs:

\[
\hat{h} = C_\theta(h, z, v_j, \alpha)
\]

or, preferably for the first version:

\[
\hat{h} = z + P_{v_j^\perp} r_\theta(h, z, v_j, \alpha)
\]

where \(P_{v_j^\perp}\) projects the correction onto the subspace orthogonal to \(v_j\). This hard-constrains the model to preserve the steering projection:

\[
\langle \hat{h} - h, v_j \rangle = \langle z - h, v_j \rangle
\]

up to numerical error. This avoids the most obvious failure mode: learning to undo the steer.

### 4.2 Model Families to Test

#### A. ColdSteer-Residual: Projection-Preserving Residual Corrector

This is the primary MVP.

Input:

\[
[h, z, v, \alpha]
\]

Output:

\[
r_\theta \in \mathbb{R}^d
\]

Final activation:

\[
\hat{h} = z + P_{v^\perp} r_\theta
\]

This makes the model responsible only for the âmake it behave wellâ correction, not for deciding whether to keep the semantic steering component.

#### B. ColdSteer-Soft: Soft Projection Retention

Allow the model to adjust the steering component, but penalize erasure:

\[
L_{\text{retain}} =
\left(
\langle \hat{h} - h, \hat{v} \rangle - \alpha
\right)^2
\]

This may outperform hard preservation if optimal on-manifold projections need a small change along \(v\).

#### C. ColdSteer-Iterative: Cold-Diffusion-Style Multi-Step Correction

Use a schedule:

\[
\alpha_0 > \alpha_1 > \dots > \alpha_K
\]

but **do not** run the usual Cold Diffusion inverse that subtracts away steering. Instead, interpret the schedule as correction strength, not semantic strength. The model repeatedly predicts a corrected activation while preserving a target steering projection \(\alpha_\star\).

MVP should be one-shot. Iterative correction is a stretch goal.

## 5. Supervision Strategies

### Strategy 1: Paired Activation Targets

Construct contrastive prompt pairs that differ mainly in a target attribute: positive/negative sentiment, polite/rude, sycophantic/non-sycophantic, truthful/hallucinatory, etc. Persona Vectors is a useful source of steering-style behavior families, since it extracts activation directions for traits such as evil, sycophancy, and hallucination and validates them through steering.

For pair \((x^-, x^+)\), extract activations:

\[
h^- = M_\ell(x^-), \quad h^+ = M_\ell(x^+)
\]

Train:

\[
z = h^- + \alpha v
\]

\[
C_\theta(h^-, z, v, \alpha) \approx h^+
\]

Use losses that avoid forcing unrelated content changes:

\[
L =
\lambda_\perp \|P_{v^\perp}(\hat{h} - h^+)\|^2
+
\lambda_{\text{retain}} L_{\text{retain}}
+
\lambda_{\text{near}} \|\hat{h} - z\|^2
\]

This is the cleanest âlearn the correct shifted targetâ version, but it depends on high-quality paired data.

### Strategy 2: GLP-Distilled Pseudo-Targets

Use the released GLP model as a teacher. The official GLP repository includes code, pretrained weights, a demo notebook, on-manifold steering integration, and 1M activation sanity datasets; it also reports that most demo scripts fit under 24GB VRAM.

For each \(z = h + \alpha v\):

1. Run GLP post-processing with multiple \(t_{\text{start}}\), step counts, and seeds.
2. Score candidates by:
   - low Delta LM loss / perplexity impact;
   - high steering projection retention;
   - low orthogonal distance from \(z\);
   - high concept score if available.
3. Select the best candidate \(\tilde{h}\).
4. Train ColdSteer to imitate \(\tilde{h}\), optionally after projecting out any teacher correction that erases \(v\).

This is likely the fastest path to an MVP because it reuses GLP infrastructure.

### Strategy 3: Direct Downstream Training

Train \(C_\theta\) through a frozen LLM using a combined objective:

\[
L =
\lambda_{\text{LM}} L_{\text{fluency}}
+
\lambda_{\text{concept}} L_{\text{concept}}
+
\lambda_{\text{retain}} L_{\text{retain}}
+
\lambda_{\text{near}} \|\hat{h} - z\|^2
\]

This is more expensive and more brittle, but it directly optimizes the desired behavior.

Use as a second-stage finetune after Strategy 1 or 2.

### Strategy 4: Negative Control â Naive Inversion

Train:

\[
C_\theta(h+\alpha v, v, \alpha) \rightarrow h
\]

This should preserve fluency but erase steering. It is a useful sanity check: if this performs well on concept strength, the evaluation is broken.

## 6. Evaluation Plan

### 6.1 Baselines

Compare against:

1. no steering;
2. raw steering \(h + \alpha v\);
3. raw steering with norm clipping;
4. raw steering with projection onto PCA/activation-statistics ellipsoid;
5. GLP post-processing;
6. ColdSteer-Residual;
7. ColdSteer-Soft;
8. naive-inversion negative control.

### 6.2 Metrics

Activation-level:

\[
\text{projection retention}
=
\langle \hat{h} - h, \hat{v} \rangle
\]

\[
\text{orthogonal displacement}
=
\|P_{v^\perp}(\hat{h} - z)\|
\]

Also measure:

- Frechet Distance to real activations;
- Delta LM Loss from replacing clean activations with corrected activations;
- next-token KL versus base model and versus raw-steered model;
- activation norm and layernorm-stat drift.

Behavior-level:

- concept strength;
- fluency;
- repetition rate;
- refusal or collapse rate;
- matched-fluency concept gain;
- matched-concept fluency gain;
- area under the concept/fluency Pareto frontier.

AxBench is a natural benchmark candidate because it was introduced specifically for evaluating language-model steering and concept detection methods at scale.

### 6.3 Primary Success Criterion

ColdSteer is successful if, on held-out prompts and steering strengths, it improves the Pareto frontier over both raw steering and GLP post-processing.

Concrete MVP success target:

> At matched fluency, ColdSteer improves concept score by at least 10â20% over raw steering and by a measurable margin over GLP post-processing on at least two behavior families.

## 7. Implementation Plan for Claude Code

Use the official GLP repository as the starting point. The repo already contains PyTorch implementation, pretrained GLP loading, training code, activation datasets, steering integration, and scalar probing scripts.

### Phase 0 â Repository Inspection and Smoke Test

Ask Claude Code to:

1. Clone or open the GLP repo.
2. Install the environment exactly as the README specifies.
3. Run the demo notebook or convert the core demo cells into a smoke-test script.
4. Load a pretrained GLP, preferably the Llama1B model first.
5. Confirm that an activation batch can be:
   - loaded;
   - standardized;
   - passed through GLP;
   - injected back into the LLM.

Deliverable:

```text
reports/00_glp_smoke_test.md
```

with environment notes, GPU memory, model used, and any required patches.

### Phase 1 â Implement Steering Corruption Utilities

Create:

```text
glp/cold_steer/corruptions.py
glp/cold_steer/projections.py
tests/test_cold_steer_corruptions.py
```

Required functions:

```python
def normalize_vector(v, eps=1e-8):
    ...

def apply_steering(h, v, alpha):
    # h: [batch, d]
    # v: [batch, d] or [d]
    # alpha: scalar or [batch]
    ...

def project_orthogonal(x, v):
    ...

def projection_along(x, v):
    ...

def retain_projection_update(z, residual, v):
    # returns z + P_{v_perp}(residual)
    ...
```

Unit tests:

1. \(\alpha = 0\) returns the original activation.
2. Orthogonal projection has near-zero dot product with \(v\).
3. Projection-preserving update keeps \(\langle \hat{h} - h, v\rangle\) unchanged.
4. Vector normalization is stable for batched and unbatched vectors.

### Phase 2 â Build a Steering Vector Bank

Create:

```text
glp/cold_steer/vector_bank.py
scripts/build_vector_bank.py
configs/cold_steer/vector_bank_llama1b.yaml
```

Start with simple DiffMean-style vectors:

\[
v = \mathbb{E}[h^+] - \mathbb{E}[h^-]
\]

Initial vector families:

1. sentiment: positive versus negative;
2. refusal/compliance if safe and available;
3. persona-style traits if using compatible models;
4. SAE feature directions from GLP examples as an optional baseline.

Store:

```text
artifacts/vector_banks/{model_name}/{layer}.pt
```

with metadata:

```python
{
    "model_name": ...,
    "layer": ...,
    "activation_scaler": ...,
    "vectors": {
        "positive_sentiment": {
            "v": tensor,
            "norm": float,
            "source_dataset": str,
            "num_pos": int,
            "num_neg": int,
        }
    }
}
```

### Phase 3 â Build the ColdSteer Dataset

Create:

```text
glp/cold_steer/datasets.py
configs/cold_steer/train_mvp.yaml
```

Each sample should return:

```python
{
    "h": clean_activation,
    "z": h + alpha * v,
    "v": steering_vector,
    "alpha": alpha,
    "target": optional_target,
    "concept_id": concept_id,
    "metadata": ...
}
```

Support three modes:

```yaml
target_mode: none
target_mode: paired
target_mode: glp_distilled
```

For the first MVP, use:

```yaml
target_mode: glp_distilled
model_family: llama1b
layer: 7
num_activations: 1_000_000
alpha_distribution:
  type: uniform
  min: 0.0
  max: 8.0
```

### Phase 4 â Implement the Corrector Model

Create:

```text
glp/cold_steer/models.py
glp/cold_steer/losses.py
scripts/train_cold_steer.py
```

MVP architecture:

```python
class ColdSteerResidualCorrector(nn.Module):
    def __init__(self, d_model, hidden_mult=4, n_layers=4):
        ...
    def forward(self, h, z, v, alpha):
        ...
        residual = ...
        return z + project_orthogonal(residual, v)
```

Recommended inputs:

\[
[z, h, v, z-h, \alpha\text{-embedding}]
\]

Losses:

```python
L_target = mse(y_hat, target)                         # if target exists
L_orth_target = mse(P_perp(y_hat), P_perp(target))    # paired or distilled
L_retain = (dot(y_hat - h, v_hat) - alpha) ** 2
L_near = mse(y_hat, z)
L_norm = activation_norm_penalty(y_hat)
```

For the hard projection-preserving model, `L_retain` should be logged but not needed.

### Phase 5 â Add GLP-Distillation Teacher

Create:

```text
scripts/build_glp_distillation_targets.py
configs/cold_steer/distill_glp_llama1b.yaml
```

Candidate generation:

```python
for t_start in [0.05, 0.1, 0.2, 0.3]:
    for num_steps in [4, 8, 16, 32]:
        for seed in seeds:
            y_candidate = glp_postprocess(z, t_start, num_steps, seed)
```

Score:

\[
S(y) =
\lambda_{\text{retain}} |\langle y-h, \hat v\rangle-\alpha|
+
\lambda_{\text{near}} \|P_{v^\perp}(y-z)\|
+
\lambda_{\text{lm}} \Delta \text{LM Loss}(y)
-
\lambda_{\text{concept}} \text{ConceptScore}(y)
\]

Select the lowest-score candidate as the pseudo-target.

Important: also log the raw GLP teacherâs projection loss. If GLP often removes the steering projection, that supports the motivation for ColdSteer.

### Phase 6 â Evaluation Harness

Create:

```text
scripts/eval_cold_steer.py
glp/cold_steer/eval.py
configs/cold_steer/eval_llama1b.yaml
```

The evaluation script should sweep:

```yaml
methods:
  - no_steer
  - raw_steer
  - raw_steer_norm_clip
  - glp_postprocess
  - cold_steer_residual
  - cold_steer_soft
  - naive_inversion_negative_control

alphas: [0, 1, 2, 4, 6, 8, 10, 12]
prompts: heldout
num_generations_per_setting: 100
```

Outputs:

```text
results/cold_steer/{run_id}/metrics.jsonl
results/cold_steer/{run_id}/pareto_frontier.png
results/cold_steer/{run_id}/sample_generations.jsonl
results/cold_steer/{run_id}/summary.md
```

Primary plots:

1. concept strength versus fluency;
2. projection retention versus alpha;
3. Delta LM Loss versus alpha;
4. orthogonal correction norm versus alpha;
5. GLP teacher versus ColdSteer student.

### Phase 7 â Ablations

Run these ablations before scaling:

1. **Hard versus soft projection retention**
   - Does strict preservation hurt fluency?
2. **Input conditioning**
   - \(C(z,\alpha)\)
   - \(C(z,v,\alpha)\)
   - \(C(h,z,v,\alpha)\)
3. **Target source**
   - paired targets;
   - GLP-distilled targets;
   - direct downstream finetuning;
   - naive inversion.
4. **Generalization**
   - held-out alpha;
   - held-out prompts;
   - held-out vector;
   - held-out behavior family.
5. **Sampling cost**
   - one-shot ColdSteer versus multi-step GLP.

## 8. Concrete Claude Code Task Prompt

Give Claude Code something close to this:

```text
You are working in the GLP repository. Implement an MVP of âColdSteer,â a steering-corruption activation corrector.

Goal:
Train and evaluate a small residual corrector that takes a clean activation h, a naively steered activation z = h + alpha*v, the steering vector v, and steering strength alpha, then outputs a corrected activation y_hat that preserves the projection along v while correcting only the orthogonal component.

Do not train the model to reconstruct h from z. That is a negative control only.

Implementation steps:
1. Read the repo README, glp_demo.ipynb, GLP model loading utilities, and existing on-manifold steering code.
2. Add glp/cold_steer/ with:
   - corruptions.py
   - projections.py
   - datasets.py
   - models.py
   - losses.py
   - eval.py
3. Implement apply_steering(h, v, alpha), project_orthogonal(x, v), and retain_projection_update(z, residual, v).
4. Add unit tests showing retain_projection_update preserves dot(y_hat - h, v_hat).
5. Implement ColdSteerResidualCorrector:
   - input: h, z, v, alpha
   - output residual r
   - final y_hat = z + P_v_perp(r)
6. Build a minimal training script scripts/train_cold_steer.py.
7. First training target mode: GLP-distilled pseudo-targets.
   - Generate candidate GLP postprocessed activations for z.
   - Score candidates by projection retention, orthogonal distance, and Delta LM Loss if available.
   - Train ColdSteer to imitate the best candidate.
8. Add scripts/eval_cold_steer.py comparing:
   - no steering
   - raw steering
   - GLP postprocessing
   - ColdSteerResidualCorrector
   - naive inversion negative control
9. Produce a report with:
   - setup details
   - metrics table
   - Pareto frontier plot
   - sample generations
   - known bugs or failure modes

Start with Llama1B/layer 7 and a small activation subset. Keep configs in configs/cold_steer/.
```

## 9. Expected Failure Modes and Mitigations

### Failure Mode 1: The Corrector Erases Steering

Mitigation: use the projection-preserving parameterization first. Treat naive inversion as a negative control.

### Failure Mode 2: Paired Targets Change Content Too Much

Mitigation: use orthogonal-only target loss, nearest-neighbor pairing, or GLP-distilled pseudo-targets.

### Failure Mode 3: GLP Teacher Already Dominates

Mitigation: evaluate inference cost. A one-shot ColdSteer student may still be useful if it approximates or improves GLP post-processing with fewer steps.

### Failure Mode 4: Method Overfits to One Vector

Mitigation: train on a vector bank and report held-out-vector generalization separately.

### Failure Mode 5: Activation Correction Looks Good Locally but Hurts Generation

Mitigation: always include downstream generation metrics. Activation-level metrics are necessary but not sufficient.

## 10. Recommended MVP Scope

Do not start with full multi-layer modeling or billion-activation training. Start with:

```text
Model: Llama1B-compatible GLP
Layer: middle residual layer
Vectors: 1â3 simple vectors
Training data: 100kâ1M activations
Corrector: 4-layer MLP
Correction: one-shot hard projection-preserving residual
Baselines: raw steering, GLP postprocessing, negative-control inversion
Metrics: projection retention, Delta LM Loss, concept/fluency Pareto
```

A good first paper-quality result would be:

> âTraining on steering-shaped corruptions yields a correction operator that preserves the intended steering projection better than generic GLP post-processing, while recovering much of GLPâs fluency benefit over raw steering.â

That would validate the core idea without needing to solve every target-construction problem.

## References

- GLP paper: <https://arxiv.org/abs/2602.06964>
- Cold Diffusion paper: <https://arxiv.org/abs/2208.09392>
- GLP repository: <https://github.com/g-luo/generative_latent_prior>
- Persona Vectors paper: <https://arxiv.org/abs/2507.21509>
- AxBench paper: <https://arxiv.org/abs/2501.17148>

