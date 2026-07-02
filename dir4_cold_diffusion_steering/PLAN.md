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
- [x] S4 — Generalization. (a) α-EXTRAPOLATION DONE: corrector trained on α~U(0.5,8) evaluated at
        α=10,12 (beyond range) still recovers 77%/60% of raw's ΔLM damage — graceful degradation.
        (b) HELD-OUT VECTOR DONE: built formality v₂ (cos(v₁,v₂)=0.014); a single sentiment-trained
        corrector does NOT transfer (recovery ≈0%) but retraining the recipe on v₂ recovers 83–104%
        — correction is direction-specific, method generalizes.
        (c) DIRECTION-CONDITIONAL + VECTOR BANK DONE: `r_θ(h,z,v̂,α)` trained on a 3-vector bank
        {sentiment,formality,concreteness} is ONE model correcting all in-bank dirs (55–70% @α=8) and
        PARTIALLY transfers to held-out certainty (51%→7% recovery weak→strong; vs ≈0% frozen
        single-vector). "One model per vector" → "one model per bank".
        (c-follow-up) BANK-SCALING DONE (Exp 7): a LARGER bank at fixed capacity does NOT close the
        held-out gap — transfer peaks at bank size 3, drops at 5 (coverage is not the constraint).
        (c-follow-up #2) CAPACITY-SCALING DONE (Exp 8): scaling the corrector 9× wider (5.2M→46.2M) on a
        fixed bank ALSO does not close the gap — in-bank recovery saturates ~45%, held-out overfits at
        weak α (rec −1%→−146%). ⇒ ceiling is the TRAINING SIGNAL, not directions or params.
        (c-follow-up #3) CURATED-BANK DONE (Exp 9): curating the bank TOWARD the target subspace at fixed
        size(3)/capacity(5.25M) BACKFIRES — the most target-aligned bank (|cos|0.80) transfers
        CATASTROPHICALLY (α=1 rec −183%, net-neg every strength); the diverse moderately-aligned exp6 bank
        transfers best (51→7%). In-bank recovery falls with internal correlation (diffuse67>exp6 48>cur30).
        ⇒ bank ANGULAR DIVERSITY, not target coverage, drives transfer; per-direction native corrector
        (78–142%) remains the reliable route. Third corrective negative (dirs/params/curation all fail).
        (d) BEHAVIORAL text Pareto DONE (Exp 10): scored the flagship sentiment corrector on GENERATED
        text (not teacher-forced ΔLM) — sentiment effect B(α) on a clean re-encode + distinct-2
        degeneration. CORRECTIVE: matched layer-6 projection ≠ matched behavioral steering. Raw steers
        hard then collapses to repetition (distinct-2 0.78→0.32 @α=8); the corrector stays fluent
        (0.64–0.72) but is only weakly steered (effect ~1/6 of raw). P_{v⊥}r is ⟂v in activation space
        but not to the downstream readout ⇒ the ΔLM win partly reflects a weaker propagated edit. Exp 3's
        "full edit intact" qualified to "layer-6 projection intact"; Limitation (2) fixed.
        (e) BEHAVIORAL-PRESERVATION term DONE (Exp 11): added one loss term matching the corrector's
        DOWNSTREAM sentiment readout (final resid_post, L2=11) to raw steering's, weight λ_b∈{0,10,40}.
        PARTIAL POSITIVE: recovers 2–6× more generated effect (+0.15–0.48→+0.8–1.3) while staying fluent
        (distinct-2 0.52–0.73 vs raw's 0.32 collapse), turning Exp 10's "neither dominates" into outright
        Pareto DOMINANCE over raw at moderate steering. Ceiling: no λ_b reaches raw's strong pre-collapse
        effect (≈+2.5) — matching the teacher-forced readout only partially transfers to generation. Frontier
        pushed out, not erased. λ_b=0 reproduces Exp 10 to the digit.
        (f) LAYER ROBUSTNESS DONE (Exp 12): replicated the EXACT flagship Exp 3 pipeline at blocks 3/6/9
        (only the hook layer changes). POSITIVE generality — both headline facts replicate at every depth:
        fluency recovery @α=8 = 90/84/76% (≥91% @α=4), corrected activation off the Gaussian manifold at
        EVERY layer (Exp 2/3 decoupling is layer-robust). Block 6 reproduces Exp 3 to the digit. NOT a
        block-6 artifact.
        (g) CROSS-MODEL generality DONE (Exp 13): replicated the EXACT flagship Exp 3 pipeline on GPT-2
        MEDIUM (355M, 24 blocks, d=1024) at mid layer block 12/24 (only the model changes). POSITIVE — both
        headline facts replicate: recovery 89% @α=8 / 101% @α=4, corrected off the Gaussian manifold at every
        α (79.9 vs 55.1 @α=8); ≈ small's 84%. NOT a GPT-2-small artifact. Result is layer- AND model-robust.
  (each reported metric: produce + save figure to plots/ + define it in REPORT.md Methods)

## Out of scope (do NOT)
- Cloning/installing the GLP repo or its billion-activation datasets (too heavy for our VRAM share);
  GLP-distillation (Strategy 2) is optional and only if S2–S4 land with time to spare.
- Multi-layer / multi-model scaling before a single-layer single-vector result works. No other directions.

## On-track check (required every iteration)
End each JOURNAL.md entry with: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status
S1+S2+S3 + S4(a) strength-extrap + S4(b) held-out-vector + S4(c) direction-conditional-bank +
S4(c-follow-ups) bank/capacity/curated SCALING (Exp 7/8/9) + S4(d) BEHAVIORAL text Pareto (Exp 10) +
S4(e) BEHAVIORAL-PRESERVATION term (Exp 11) + S4(f) LAYER ROBUSTNESS (Exp 12) + S4(g) CROSS-MODEL
generality (Exp 13) delivered — success criterion MET; direction complete on all planned axes plus
behavioral-tradeoff + layer- + model-generality follow-ups (~99%).
**S4(g) Experiment 13 (new):** answered the second obvious external-validity question — is the result a
GPT-2-*small* artifact? Replicated the EXACT flagship Exp 3 pipeline on GPT-2 MEDIUM (355M, 24 blocks,
d=1024) at mid layer block 12/24, changing ONLY the model (reused the Exp-3 helpers by installing medium in
common's model cache; Corrector at d=1024, batch 4 for VRAM). POSITIVE, clean generality result — both
headline facts replicate: raw steering breaks the LM (ΔLM→+2.72 @α=8, D_M 31.5→55.1) and the identical
LM-supervised corrector recovers it at matched projection (recovery **89% @α=8, 101% @α=4**; ΔLM slightly
negative at α≤2 = free-or-better weak-α, as on small), with the corrected activation sitting FURTHER off the
Gaussian manifold than raw at every α (79.9 vs 55.1 @α=8). α=8 recovery (89%) ≈ small's 84%. The core result
is layer- AND model-robust. Artifacts: `experiments/13_cross_model.py`, `results/13_cross_model.json`,
`plots/13_cross_model.png`. RESULTS/REPORT/CHANGELOG curated; REPORT math verified (14/14 js-display-math,
0 broken, 0 inline hazards).
<!-- prior: S4(f) layer robustness Exp 12 -->
**S4(f) Experiment 12:** answered the most obvious external-validity question — is the whole result a
block-6 artifact? Replicated the EXACT flagship Exp 3 pipeline (same prompts/data/seed/recipe) at blocks 3
(early), 6 (mid = Exp 3), 9 (late), changing ONLY the hook layer (reused exp03 helpers by swapping the
module-global LAYER; POS/NEG from exp01). POSITIVE, clean generality result: both headline facts replicate
at every depth — fluency recovery @α=8 = 90% / 84% / 76% (≥91% @α=4), ΔLM≈0 at weak steering, and the
corrected activation sits FURTHER off the Gaussian manifold than raw at EVERY layer (the Exp 2/3 "LM-safe
but off-Gaussian" decoupling is layer-robust). Recovery declines mildly with depth (|v| grows 6.75→11.08→
23.16 toward the output). Block 6 reproduced Exp 3 to the digit (raw +2.78 → learned +0.44, 84%) — built-in
reproducibility check. Artifacts: `experiments/12_layer_robustness.py`, `results/12_layer_robustness.json`,
`plots/12_layer_robustness.png`. RESULTS/REPORT/CHANGELOG curated; REPORT math verified (14/14
js-display-math, 0 broken, 0 inline hazards).
<!-- prior: S4(e) behavioral-preservation term Exp 11 -->
S1+S2+S3 + S4(a) strength-extrap + S4(b) held-out-vector + S4(c) direction-conditional-bank +
S4(c-follow-ups) bank/capacity/curated SCALING (Exp 7/8/9) + S4(d) BEHAVIORAL text Pareto (Exp 10) +
S4(e) BEHAVIORAL-PRESERVATION term (Exp 11) delivered — success criterion MET; direction complete on all
planned axes plus the behavioral-tradeoff follow-up (~99%).
**S4(e) Experiment 11 (new):** acted on the highest-value open follow-up — attack Exp 10's discovered
effect-fluency tradeoff. Kept the Exp 3 corrector/recipe/seed/data and added ONE loss term matching the
corrector's DOWNSTREAM sentiment readout (final resid_post L2=11; DiffMean ŵ |w|=3.87) to RAW steering's,
weight λ_b∈{0,10,40} (λ_b=0 = Exp 10 corrector). Scored each on the IDENTICAL Exp 10 generation protocol.
PARTIAL POSITIVE (first non-negative follow-up in a while): the term recovers 2–6× more behavioral effect
(gen effect +0.15–0.48→+0.8–1.3) while keeping generation fluent (distinct-2 0.52–0.73 vs raw's 0.32
collapse @α=8), turning Exp 10's "neither dominates" into outright Pareto DOMINANCE over raw at moderate
steering (λ_b=40 @α=2: effect +0.99 at distinct-2 0.73≈baseline; raw only reaches that effect after
collapsing). CEILING: no λ_b reaches raw's strong pre-collapse effect (≈+2.5); λ_b 10→40 stops raising it
and only raises training LM loss — matching the TEACHER-FORCED downstream readout (behav loss→~0.005,
p_corr≈p_raw) only PARTIALLY transfers to autoregressive generation (a second layer of the same proxy
gap). Frontier pushed OUT, not erased. Artifacts: `experiments/11_behavioral_corrector.py`,
`results/11_behavioral_corrector.json`, `plots/11_behavioral_corrector.png`. RESULTS/REPORT/CHANGELOG
curated; REPORT math verified (13/13 js-display-math, 0 broken, 0 inline hazards).
<!-- prior: S4(d) behavioral text Pareto Exp 10 -->
S1 + S2 + S3 complete + S4(a) strength-extrapolation + S4(b) held-out-vector + S4(c)
direction-conditional-bank + S4(c-follow-ups) bank/capacity/curated SCALING (Exp 7/8/9) + S4(d) BEHAVIORAL
text Pareto (Exp 10) delivered — success criterion MET; direction complete on all planned axes (~99%).
**S4(d) Experiment 10 (new):** the one unmeasured axis — all 9 prior experiments scored ΔLM (fluency) at
matched projection; none checked behavioral steering on GENERATED text. Greedy-generated under raw vs the
flagship sentiment corrector; scored sentiment effect B(α) (clean re-encode proj onto v̂) + distinct-2
degeneration. CORRECTIVE finding: matched layer-6 projection ≠ matched behavioral steering. Raw steers
hard (effect +2.97 @α=2) then collapses to repetition (distinct-2 0.78→0.32 @α=8); the corrector stays
fluent at all α (0.64–0.72 ≈ baseline 0.70) but is only weakly steered (effect +0.15–0.48, ~1/6 of raw).
Mechanism: P_{v⊥}r ⟂ v in activation space but NOT to the downstream sentiment readout, so minimizing LM
loss yields near-normal lightly-steered text ⇒ the ΔLM win of Exp 3–9 partly reflects a weaker propagated
edit, not costless cleanup. Qualified Exp 3's "full edit intact"→"layer-6 projection intact"; fixed
Limitation (2)'s false "concept strength held fixed by construction". Artifacts:
`experiments/10_behavioral_pareto.py`, `results/10_behavioral_pareto.json`, `plots/10_behavioral_pareto.png`.
RESULTS/REPORT/CHANGELOG curated; REPORT math verified (12/12 js-display-math, 0 broken, 0 inline hazards).
<!-- prior: S4(c follow-up #3) curated-bank Exp 9 -->
S1 + S2 + S3 complete + S4(a) strength-extrapolation + S4(b) held-out-vector + S4(c)
direction-conditional-bank + S4(c-follow-up) bank-SCALING (Exp 7) + S4(c-follow-up #2) capacity-SCALING
(Exp 8) + S4(c-follow-up #3) CURATED-BANK (Exp 9) delivered — success criterion MET; direction complete
on all planned axes (~99%).
**S4(c follow-up #3) Experiment 9 (new):** tested the standing open hypothesis of BOTH Exp 7 and Exp 8
("curate the bank toward the held-out target's subspace"), which neither varied. Held bank SIZE=3 and
capacity=5.25M FIXED; varied only WHICH 3 of the 5 pool dirs train, by mean |cos| to held-out certainty:
diffuse{sent,pol,form}0.38 / exp6{sent,form,conc}0.54 / curated{form,conc,cplx}0.80 (diffuse↔curated
share exactly formality — controlled). Result — CORRECTIVE (3rd negative in a row): curating TOWARD the
target BACKFIRES. Held-out recovery non-monotone in alignment, COLLAPSES at most-aligned bank — curated
net-negative every strength (α=1 −183%, α=8 −12%); diverse exp6 best (51/42/21/12/7). Mechanism: in-bank
recovery @α=8 falls with internal correlation (diffuse67>exp6 48>curated30); curated's 3 members pairwise
collinear (|cos|0.76–0.82) → corrector can't disambiguate from v̂ → can't specialize → over-fires on
unseen dirs. ⇒ lever is bank ANGULAR DIVERSITY, not target-subspace coverage. exp6 reproduced Exp 6/7 to
the digit; native oracle unchanged 78–142%. Corrects Exp 7/8's "curate toward subspace" prescription.
Artifacts: `experiments/09_curated_bank.py`, `results/09_curated_bank.json`, `plots/09_curated_bank.png`.
RESULTS/REPORT/CHANGELOG curated; REPORT math verified (10/10 js-display-math, 0 broken, 0 inline hazards).
<!-- prior: S4(c follow-up #2) capacity-scaling Exp 8 -->
S1 + S2 + S3 complete + S4(a) strength-extrapolation + S4(b) held-out-vector + S4(c)
direction-conditional-bank + S4(c-follow-up) bank-SCALING (Exp 7) + S4(c-follow-up #2) capacity-SCALING
(Exp 8) delivered — success criterion MET; direction complete on all planned axes (~99%).
**S4(c follow-up #2) Experiment 8 (new):** tested Exp 7's causal claim (capacity interference binds) by
holding the size-5 bank fixed and scaling corrector WIDTH hidden∈{1024,2048,4096}=5.2M/14.7M/46.2M params
(9× range), identical recipe/seed/data. Result — CORRECTIVE: more capacity does NOT close the held-out
gap. In-bank recovery @α=8 SATURATES ~45% across 9× params (45.4→43.8→46.3); held-out `certainty` @α=8
flat-falling 3→2→1%; at weak steering the 46M model OVERFITS (α=1 rec −1→−22→−146%). ⇒ ceiling is the
TRAINING SIGNAL (bank composition/conditioning/objective), not parameter count. hidden=1024 reproduces
Exp 7's size-5 to the digit. Native oracle unchanged 78–142% (direction fully correctable). Corrects Exp
7's "scale the model" prescription. Artifacts: `experiments/08_capacity_scaling.py`,
`results/08_capacity_scaling.json`, `plots/08_capacity_scaling.png`. RESULTS/REPORT/CHANGELOG curated;
REPORT math verified (10/10 js-display-math, 0 broken, 0 inline hazards).
<!-- prior: S4(c follow-up) bank-scaling Exp 7 -->
S1 + S2 + S3 complete + S4(a) strength-extrapolation + S4(b) held-out-vector + S4(c)
direction-conditional-bank + S4(c-follow-up) bank-SCALING delivered — success criterion MET; direction
complete on all planned axes (~99%).
**S4(c follow-up) Experiment 7 (new):** directly tested Exp 6's parting prescription ("scale the bank").
Held `certainty` out; trained the SAME conditional corrector (5.25M, identical recipe) on NESTED banks
of size 1/3/5 (size 5 adds new DiffMean politeness |v|=15.6, complexity |v|=58.4). Result — CORRECTIVE:
enlarging the bank does NOT close the held-out gap; at fixed capacity it makes transfer WORSE. Held-out
recovery is non-monotone and PEAKS at size 3 (α=8: bank1 0% / bank3 7% / bank5 3%), even though size-5
adds a strongly-correlated direction (complexity |cos|=0.80). In-bank per-direction recovery @α=8 also
dropped size3→size5 (formality 70%→45%). ⇒ capacity interference, not coverage, binds; the route to a
reusable corrector is more MODEL CAPACITY and/or a CURATED bank, not more directions. Native oracle
78–142% (direction fully correctable). Size-3 reproduces Exp 6 exactly. Artifacts:
`experiments/07_bank_scaling.py`, `results/07_bank_scaling.json`, `plots/07_bank_scaling.png`,
`data/{politeness,complexity}_vec_layer6.npy`. RESULTS/REPORT/CHANGELOG curated; REPORT math verified
(10/10 js-display-math, 0 broken, 0 inline hazards).
<!-- prior: S4(c) direction-conditional bank -->
S1 + S2 + S3 complete + S4(a) strength-extrapolation + S4(b) held-out-vector + S4(c)
direction-conditional-bank delivered — success criterion MET; direction near-complete (~98%).
**S4(c) (new):** made the corrector direction-conditional (`r_θ(h,z,v̂,α)`, 5.25M params) and trained
ONE on a 3-vector bank {sentiment,formality,concreteness}; held out certainty (|v|=32.8). Cosines:
sentiment ⟂ all (|cos|≤0.03), the other three share a subspace (|cos| 0.76–0.82). Result: one model
corrects every in-bank direction at once (α=8 recovery sentiment 55%, formality 70%, concreteness 17%
/70% @α=2) at a per-direction cost vs a dedicated corrector; and PARTIALLY transfers to held-out
certainty (51% @α=1 → 7% @α=8, vs ≈0% for Exp-5's frozen single-vector; native oracle 78% @α=8). A
3-vector bank starts to generalize across directions but doesn't yet solve held-out transfer at strong
steering. Artifacts: `experiments/06_conditional_bank.py`, `results/06_conditional_bank.json`,
`plots/06_conditional_bank.png`, `data/{concreteness,certainty}_vec_layer6.npy`. RESULTS/REPORT/
CHANGELOG curated; REPORT math verified (9/9).
<!-- prior: S4(b) held-out vector -->
S1 + S2 + S3 complete + S4(a) strength-extrapolation + S4(b) held-out-vector delivered — success
criterion MET; direction near-complete (~95%).
**S4(b) (new):** built a second DiffMean concept vector v₂ (formality, |v₂|=34.0, cos(v₁,v₂)=0.014
— nearly orthogonal). On v₂ at matched projection: the sentiment-trained corrector does NOT transfer
(ΔLM ≈ raw, recovery ≈0%), but retraining the SAME recipe on v₂ recovers 83–104% of raw's fluency
damage (α=8 +6.49→+1.12). ⇒ the correction is direction-specific, the METHOD generalizes → train
per-vector (or condition on v / vector-bank). Artifacts: `experiments/05_heldout_vector.py`,
`results/05_heldout_vector.json`, `plots/05_heldout_vector.png`, `data/formality_vec_layer6.npy`.
RESULTS/REPORT/CHANGELOG curated; REPORT math verified (9/9).
<!-- prior: S4(a) α-extrapolation -->
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
Core arc + all generalization axes + behavioral axis + behavioral-preservation follow-up + LAYER-ROBUSTNESS
(Exp 12: 90/84/76% @α=8 at blocks 3/6/9) + CROSS-MODEL generality (Exp 13: GPT-2 medium, 89% @α=8 / 101% @α=4,
off-Gaussian at every α) delivered — flagship fluency result now layer- AND model-robust. Amortization story
CLOSED on three scaling axes (Exp 7/8/9 negative); flagship story has its behavioral reality-check (Exp 10),
constructive follow-up (Exp 11), layer-generality (Exp 12), and model-generality (Exp 13) checks. Optional
remaining polish, any one a clean iteration: (i) push the Exp 11 ceiling by supervising the behavioral readout
THROUGH sampled/differentiable generation rather than teacher-forced; (ii) confirm the bank-diversity lever
directly (max-orthogonal 3-bank vs collinear curated); (iii) held-out-prompt-family generalization, or a
still-larger model (GPT-2 large). All optional; success criterion long met.

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

