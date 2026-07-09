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
        (g-follow-up) MODEL-SCALING to GPT-2 LARGE DONE (Exp 19): replicated the EXACT flagship Exp-3 pipeline
        on GPT-2 LARGE (774M, 36 blocks, d=1280) at mid layer block 18/36 (only the model changes). POSITIVE —
        both headline facts replicate: recovery 84% @α=8 / 95% @α=4, corrected off the Gaussian manifold at
        every α (96.8 vs 66.0 @α=8). Model axis now spans 6× params (124M→355M→774M) with FLAT α=8 recovery
        (84/89/84%) — amortized correction quality does not erode with scale. Model-robust at three scales.
        (h) BANK-DIVERSITY LEVER — CAUSAL confirmation DONE (Exp 14): controlled third-member swap removes
        Exp 9's confound (all size-3 banks share {sentiment,formality} anchor; only 3rd member's collinearity
        varies). POSITIVE: the swapped member's OWN recovery collapses as it collinearizes (politeness 69%→
        complexity 40%→concreteness 17% @α=8), and the confound-free isolate `sentiment` (⟂ all dirs AND ⟂
        target) is corrected WORSE in more collinear banks (63→61→55% @α=8) — pure separability, cannot be
        target coverage. Turns Exp 9's correlation into a controlled causal result; positive counterpart to
        the Exp 7/8/9 scaling negatives.
        (i) PROMPT-FAMILY robustness DONE (Exp 15): trained the flagship Exp-3 sentiment corrector on FineWeb,
        evaluated it UNCHANGED on 3 held-out prompt families of increasing distribution shift (fineweb in-dist /
        markdown research prose / Python code). POSITIVE: recovery @α=8 = 84/77/60% (95/87/78% @α=4), tracking
        the clean-activation shift under the FineWeb Gaussian (D_M 27.5→30.1→37.4) — graceful degradation, not a
        FineWeb-prompt artifact. fineweb row reproduces Exp 3 to the digit. Flagship result now robust on 5 axes
        (strength/direction/layer/model/prompt).
- [x] S5 — MANIFOLD GEOMETRY (Exp 16, human feedback #2) DONE: intrinsic dim (TwoNN/Levina–Bickel MLE/PCA
        participation ratio) + Gaussianity (held-out D_M² vs χ²₇₆₈) on clean layer-6 activations. Cloud is
        LOW-DIM (~8–34 ≪ 768), ANISOTROPIC (PR 1.1), HEAVY-TAILED (D_M² spread 6.7× Gaussian) — NOT a single
        Gaussian. Sharpens the thesis (mechanism for Exp 2). Human asks #1 (diffusion corrector) + #3 (other
        steering family) remain — see Next step.
- [x] S6 — REAL DIFFUSION CORRECTOR (Exp 17, human feedback #1) DONE: built the actual iterative machinery
        and compared three correctors at matched projection — one-shot MLP (Exp 3), cold-diffusion iterative
        K=8 (step-conditioned velocity field, LM-supervised through the unroll), GLP Gaussian-noise DDPM prior
        (SDEdit, no LM). Recovery @α=8: 84% / 85% / −5%. The Cold-Diffusion CORRUPTION MODEL + LM supervision
        carries the result (not iteration: iter ~ties one-shot); the generic Gaussian-noise prior is WORSE
        than raw and ERASES the steer. Validates ColdSteer's design. Only human ask #3 (other steering
        family) remains — see Next step.
- [x] S4(j) — STEERING-FAMILY robustness (Exp 18, human feedback #3) DONE: rebuilt the sentiment steering
        vector from a REAL downloaded dataset (SST-2) via the three canonical linear-steering families —
        DiffMean, logistic-regression probe, PCA-contrast (cos to DiffMean 1.00/0.40/0.30) — rescaled to a
        common norm and ran the identical flagship recipe per family. POSITIVE + family-robust: recovery@8 =
        86%/84%/101% (all three genuinely different directions), DiffMean reproduces flagship Exp 3 from real
        data; PCA-contrast raw steering is ON the Gaussian manifold (D_M flat 27.3) yet breaks the LM.
        Core result now robust on SIX axes; all three human asks done.
- [x] S4(k) — DIFFERENTIABLE-GENERATION behavioral supervision (Exp 20, PLAN Next-step (i)) DONE: the last
        substantive open lever. Exp 11 matched the downstream sentiment readout TEACHER-FORCED and hit a
        ≈+1.3 effect ceiling (teacher-forced readout only partially transfers to generation). Exp 20
        supervises the readout on the corrector's OWN generated continuation via a differentiable K=8
        soft-token rollout (softmax(ℓ/τ)·Wₑ feedback), pushing toward raw's own rollout, weight λ_g.
        PARTIAL POSITIVE: breaks the ceiling — λ_g=40 @α=8 effect +1.08(Exp11)→+1.72 at d2 0.47 (raw 0.32),
        λ_g=160 @α=2 effect +1.61 at near-baseline d2 0.71 — but over-weighting collapses at strong steering
        (λ_g=160: eff −0.22, d2 0.32 @α=8, degenerates like raw). Frontier pushed out a SECOND time, not
        erased; strong-effect-and-fluent corner still eludes. λ_g=0 reproduces Exp 10/11 to the digit.
- [x] S4(l) — CROSS-ARCHITECTURE generality (Exp 21) DONE: Exp 13/19 scaled the model but stayed in the
        GPT-2 family. Replicated the EXACT flagship Exp-3 pipeline UNCHANGED on Qwen3-1.7B (28 blocks, d=2048,
        block 14/28) — a non-GPT-2 architecture (RMSNorm/rotary/SwiGLU/grouped-query attention). POSITIVE:
        raw breaks the LM (ΔLM@8 +3.43, D_M 44.7→77.8); identical corrector recovers 94% @α=8 / 108% @α=4 at
        matched projection, corrected activation further off the Gaussian manifold at every α (122.2 vs 77.8
        @α=8). 94% edges GPT-2 small's 84%. Architecture-robust; result now spans 7 axes (strength/direction/
        layer/model-scale/architecture/prompt-family/steering-family).
- [x] S4(m) — BEHAVIORAL check on Qwen3 (Exp 22) DONE: Exp 21's 94% recovery is a TEACHER-FORCED ΔLM; ran the
        identical Exp-10 behavioral generation protocol on Qwen3-1.7B reusing the exact Exp-21 corrector. The Exp-10
        under-steering caveat REPLICATES: corrector generated sentiment effect only 10–29% of raw's (raw +5.2–8.0,
        corr +0.53–2.31). So 94% is honest as a fluency metric but partly bought by a weaker propagated edit — as on
        GPT-2. KEY DIFFERENCE: raw degenerates far less on Qwen3 (distinct-2 0.76 vs GPT-2's 0.32 @α=8), so raw is a
        stronger baseline and the Pareto is shallower. "matched projection ≠ matched behavioral steering" is
        architecture-robust.
- [x] S4(n) — BEHAVIORAL-FIX transfer to Qwen3 (Exp 23, PLAN Next-step ii, closes Exp 22's Next check) DONE:
        re-fit the Exp 11 behavioral-preservation term (downstream readout at Qwen3 L2=27, family λ_b∈{0,10,40})
        on the Exp 21/22 Qwen3 pipeline; λ_b=0 loads the Exp 21 checkpoint (reproduces Exp 22 to the digit).
        NUANCED/CORRECTIVE: the fix's MECHANISM is architecture-robust — λ_b=40 recovers 53–83% of raw's generated
        effect (vs base 10–29%), a 2–8× jump — but its PARETO ADVANTAGE is NOT: on Qwen3 raw does not collapse
        (distinct-2 0.76 not 0.32), so raw weakly dominates at matched α; the λ_b sweep approaches raw's frontier
        without passing it. Exp 20's over-steer wobble replicates (λ_b=40 @α=8). ⇒ behavioral fix is a robust lever
        on effect; its payoff is GATED by whether the raw baseline degenerates. Closes the behavioral arc
        (Exp 10→11→20→22→23).
- [x] S4(l-follow-up) — SECOND non-GPT-2 architecture → architecture SWEEP (Exp 24, 2026-07-08) DONE:
        Exp 21 crossed the GPT-2 boundary once (Qwen3). Added a THIRD, structurally distinct family — Pythia-410m
        (GPT-NeoX): shares rotary w/ Qwen3 and LayerNorm/GELU/dense-MHA w/ GPT-2 but uses a PARALLEL residual
        (attn+MLP from same input, summed) unlike BOTH. Replicated the EXACT flagship Exp-3 pipeline unchanged at
        mid layer block 12/24. POSITIVE — both facts replicate: raw breaks LM (ΔLM +3.10@8, D_M 31.3→52.3);
        identical corrector recovers 81%@8 / 81%@4 (71%@2), matched projection, corrected FURTHER off Gaussian at
        every α (89.4 vs 52.3@8, decoupling holds a 5th time). Architecture axis now a 3-family SWEEP with a tight
        81–94% @α=8 band (GPT-2 84/89/84%, Qwen3 94%, GPT-NeoX 81%). Not a single-boundary-crossing claim anymore.
- [x] S4(l-follow-up) — BEHAVIORAL check on Pythia (Exp 25, Exp 24's own Next check, 2026-07-08) DONE: ran the
        identical Exp 10/22 generation protocol on Pythia-410m reusing the exact Exp 24 corrector (no retrain).
        NUANCED POSITIVE — the under-steering caveat is MILDER here: corrected effect +0.90/+0.80/+0.93/+0.98
        (α=2/4/6/8) is ABOVE raw's at α≤4 and 84–92% of raw at α≥6 (vs ~1/6 GPT-2, 10–29% Qwen3), and at α=8 the
        corrector Pareto-DOMINATES raw (eff +0.98 @ d2 0.72 vs raw +1.17 @ collapsed 0.38). Reason: raw steers
        Pythia weakly here (eff peaks +1.17), so little effect for the corrector to lose ⇒ penalty size tracks how
        strongly RAW steering propagates. Behavioral arc now closed on all THREE architectures (GPT-2/Qwen3/Pythia).
- [x] S7 — SEED robustness (Exp 26, 2026-07-09) DONE: every prior experiment is a single run at SEED=0, so
        the flagship "84% @α=8" had no error bar — the one control CLAUDE.md rule 10 names that no axis varied
        (strength/direction/layer/model/prompt/steering-family/architecture all did). Re-ran the EXACT flagship
        Exp-3 pipeline at 5 seeds (0–4). POSITIVE + tight: recovery 83.3±2.0% @α=8 (per-seed 84/84/85/83/80%),
        96.2±0.8% @α=4, 90.0±0.6% @α=6; ΔLM learned @α=8 +0.464±0.054 vs raw +2.778. Seed 0 reproduces Exp 3 to
        the digit (84.3%). Wide bar only @α=1 (196±19%) = ratio artifact of raw's near-zero +0.076-nat damage.
        Flagship result now robust on a 7th axis; headline confirmed representative, not a lucky init.
  (each reported metric: produce + save figure to plots/ + define it in REPORT.md Methods)

## Out of scope (do NOT)
- Cloning/installing the GLP repo or its billion-activation datasets (too heavy for our VRAM share);
  GLP-distillation (Strategy 2) is optional and only if S2–S4 land with time to spare.
- Multi-layer / multi-model scaling before a single-layer single-vector result works. No other directions.

## On-track check (required every iteration)
End each JOURNAL.md entry with: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status
**S7 SEED robustness — Experiment 26 (new, 2026-07-09):** picked the single highest-value remaining rigor
gap. Every prior experiment (incl. the flagship Exp 3) is a SINGLE run at SEED=0, so "84% recovery @α=8" had no
error bar — and *seed* is the one control CLAUDE.md rule 10 explicitly names that no axis varied (strength/
direction/layer/model/prompt/steering-family/architecture all were). Re-ran the EXACT flagship Exp-3 pipeline
(same DiffMean sentiment vector |v|=11.08, 400-doc Gaussian fit, 300-doc train, held-out 100-doc eval, 4.46M
projection-preserving corrector, recipe, α∼U(0.5,8)) at 5 seeds (0–4); seed varies corrector init + α-sampling/
data-shuffle RNG; raw ΔLM is seed-independent (computed once). POSITIVE + TIGHT: recovery 83.3±2.0% @α=8
(per-seed 84.3/84.5/84.6/83.0/80.0%), 96.2±0.8% @α=4, 90.0±0.6% @α=6; ΔLM learned @α=8 +0.464±0.054 nats vs raw
+2.778. Seed 0 reproduces Exp 3 to the digit (84.3%) — built-in check. The only wide bar, α=1 (196±19%), is a
ratio artifact of raw's near-zero +0.076-nat damage there (absolute ΔLM_learned tight −0.073±0.014). So the
headline 84% is reproducible to ±2 points — representative, not a lucky init — closing a 7th robustness axis.
Limitation: varies only the TRAINING seed on the flagship setup (eval set/fit/vector fixed; cross-model checks
still single-seed) — bounds optimization variance, not eval-doc or vector-construction sampling variance. No
prior result superseded (Exp 26 additive; Exp 3's 84% confirmed). Artifacts: `experiments/26_seed_robustness.py`,
`results/26_seed_robustness.json`, `results/26_run.log`, `plots/26_seed_robustness.png`. Curated RESULTS.md
(Exp-26 section + figure + Headline seed CI), REPORT_3 (Methods+Results+Conclusion for the seed axis) + index
(Summary/Part-3 blurb/headline table). CHANGELOG appended. REPORT math re-verified on the 2 touched files
(index 1 / Part 3 9 js-display-math, 0 broken, 0 inline hazards — Exp 26 reuses Exp 12's recovery equation, no
new equation). OPS: dir9 `cupenv` on /mars-vol stalled ~30 min in `folio_wait_bit_common` on a cold scipy/sklearn
import (disk contention); switched to `/opt/conda/bin/python` (transformers 5.13.0, torch 2.9 cu130, matplotlib
3.11, LOCAL disk) which imports in seconds — RECOMMENDED for future iters.

<!-- prior: REPORT restructure -->
**REPORT RESTRUCTURE (human feedback, 2026-07-09):** the operator said REPORT.md was "too long and too much
back and forth" to follow, and asked to disassemble it into 2–4 topic-focused mini reports. Done: the 1744-line
monolith is now a 108-line **index** (`REPORT.md`: overall Summary + the takeaway equation + headline-numbers
table + limitations overview + links) plus **four self-contained parts**, each with its own Summary → Methods
(Data/Model/Layer + every metric & baseline defined with rendered equations) → Results → Conclusion:
`REPORT_1_core_correction.md` (Exp 2–5,16,17: the negative Gaussian corrector + the LM-supervised fix + why the
Gaussian is the wrong yardstick + the diffusion-framing ablation), `REPORT_2_amortization.md` (Exp 6–9,14:
direction-conditional corrector / bank scaling / capacity / curation / the diversity lever),
`REPORT_3_external_validity.md` (Exp 12,13,19,21,24,15,18: layer / model-scale / architecture-sweep / prompt-family
/ steering-vector-family robustness), `REPORT_4_behavioral.md` (Exp 10,11,20,22,23,25: the matched-projection≠
matched-steering caveat + the readout-preservation & differentiable-generation fixes across three architectures).
All Methods/Results blocks were copied VERBATIM from the source (no number/equation/figure altered); each part's
Summary/Conclusion is newly written to give that topic a clean linear narrative (no cross-topic back-and-forth).
Math re-verified via the GitHub API on all five files: 42 js-display-math total (index 1 / P1 16 / P2 6 / P3 9 /
P4 10), 0 broken (`<pre lang=math>`), 0 inline hazards; every part carries its topic's figures (P4 now also
references the previously-orphaned `plots/25_behavioral_pythia.png`). RESULTS.md left as-is (a per-experiment
results log, a distinct deliverable from the narrative REPORT; the feedback was specifically about the report's
length/back-and-forth). No experiment re-run; no result number changed. CHANGELOG appended.

<!-- prior status below -->
**S4(l-follow-up) Experiment 25 (new, BEHAVIORAL check on Pythia-410m = Exp 24's own Next check, 2026-07-08):**
completed an experiment a prior iteration left half-done (script `experiments/25_behavioral_pythia.py` present — a
clean adaptation of Exp 22 to the Exp-24 Pythia pipeline — but never run: no JSON/plot/log). Exp 24's 81% recovery
is a TEACHER-FORCED ΔLM; Exp 10 (GPT-2) and Exp 22 (Qwen3) both showed matched layer projection can hide a weaker
propagated behavioral edit. Ran the IDENTICAL Exp 10/22 generation protocol on Pythia-410m reusing the EXACT Exp 24
corrector (`results/24_corr.pt`, no retrain): greedy 30 tokens from 48 held-out 12-token prompts, steer block 12
every position, raw vs corrected; clean-re-encode sentiment effect B(α)−B(0) (B0=−4.77) + distinct-2 (baseline
0.77). GEN_BATCH 8, fp32, no OOM. NUANCED POSITIVE — the under-steering caveat is MILDER than on GPT-2/Qwen3:
corrected effect +0.90/+0.80/+0.93/+0.98 (α=2/4/6/8) is ABOVE raw's +0.17/+0.40 at α≤4 and 84–92% of raw's
+1.01/+1.17 at α≥6 (not the ~1/6 shortfall of GPT-2 or 10–29% of Qwen3), and at α=8 the corrector Pareto-DOMINATES
raw (eff +0.98 @ distinct-2 0.72 vs raw +1.17 @ collapsed 0.38). Mechanism: raw steering propagates weakly on
Pythia here (eff peaks +1.17), so there is little behavioral effect for the corrector to lose ⇒ the size of the
"matched projection ≠ matched steering" penalty tracks how strongly RAW steering propagates in a given model.
Limitation: small-magnitude low-signal regime — read as "penalty mild here," not "corrector out-steers raw in
general." The behavioral arc is now closed on all THREE architectures (GPT-2 Exp 10 / Qwen3 Exp 22 / Pythia Exp 25).
No prior result superseded (Exp 25 is new; reuses Exp 24 checkpoint). Ops note: cold Pythia load ran ~5 min under
/mars-vol disk contention (CUDA idle until load finished) then the retrain-free sweep completed in seconds; `setsid`
full detach used again. Artifacts: `experiments/25_behavioral_pythia.py`, `results/25_behavioral_pythia.json`,
`results/25_run.log`, `plots/25_behavioral_pythia.png`. RESULTS/REPORT/CHANGELOG curated; REPORT math verified
(26/26 js-display-math, 0 broken, 0 inline hazards — Exp 25 reuses Exp 10's behavioral-metric definitions, no new
equation). ENV: dir9's cupenv python (shared conda `transformers` still absent).
<!-- prior: S4(l-follow-up) SECOND non-GPT-2 architecture Exp 24 -->
**S4(l-follow-up) Experiment 24 (new, SECOND non-GPT-2 architecture → architecture SWEEP, 2026-07-08):** picked
the single highest-value remaining external-validity point. Exp 21 crossed the GPT-2 architecture boundary only
ONCE (Qwen3), a weak "architecture-robust" claim. Added a THIRD, structurally distinct family — **Pythia-410m
(GPT-NeoX)** — whose block uses a **parallel residual** (attention+MLP from the same input, summed) unlike BOTH
GPT-2 and Qwen3 (serial), while sharing rotary with Qwen3 and LayerNorm/GELU/dense-MHA with GPT-2. Replicated the
EXACT flagship Exp-3 pipeline UNCHANGED (same prompts/fit/train/eval/corrector/seed/α∼U(0.5,8)/objective) at mid
layer block 12/24; only the model changes (|v|=3.29, mean|h|=35.3, clean D_M=31.3; corrector 5.25M @ d=1024;
fp32 in the 4.3GB share, batch 4). POSITIVE — both headline facts replicate: raw steering breaks the LM (ΔLM
+3.10@α=8, D_M 31.3→52.3) and the identical corrector recovers **81% @α=8 / 81% @α=4** (71%@2) at matched
projection (retention α|v| exactly 3.29→26.29), with the corrected activation FURTHER off the Gaussian manifold
at every α (89.4 vs 52.3 @α=8 — the Exp 2/3 decoupling holds a 5th time). α=1 recovery 41% is noise-dominated
(raw damage only +0.06 nats). The architecture axis is now a genuine **3-family SWEEP** with a tight **81–94%
@α=8** band: GPT-2 small/medium/large 84/89/84%, Qwen3 94%, GPT-NeoX 81%. No prior result superseded (Exp 24 is
new; Exp 21's Qwen3 numbers unchanged). Ops note: the run was silently killed twice when launched via `nohup &`
(child died with the shell process group when the Bash tool returned); fixed with `setsid` full detach. Eval slow
under GPU contention but completed clean. Artifacts: `experiments/24_cross_arch_pythia.py`,
`results/24_cross_arch_pythia.json`, `results/24_corr.pt`, `results/24_run.log`, `plots/24_cross_arch_pythia.png`.
RESULTS/REPORT/CHANGELOG curated; REPORT math verified (26/26 js-display-math, 0 broken, 0 inline hazards — Exp 24
reuses Exp 3/12 definitions, no new equation). ENV: dir9's cupenv python (shared conda `transformers` still absent).
<!-- prior: S4(n) behavioral-fix transfer to Qwen3 Exp 23 -->
**S4(n) Experiment 23 (new, BEHAVIORAL-FIX transfer to Qwen3, 2026-07-08):** completed an experiment a prior
iteration had left half-done (script + λ_b=10 checkpoint present, but empty run log, no JSON/plot — interrupted).
Acted on PLAN Next-step (ii) / Exp 22's own "Next check": re-fit the Exp 11 behavioral-preservation term on Qwen3
to test whether the GPT-2 fix transfers across the architecture boundary. Reused the exact Exp 21/22 Qwen3
pipeline + Exp 22 generation protocol; added the Exp 11 term at a DOWNSTREAM Qwen3 layer L2=27 (last decoder
block, DiffMean ŵ |w|=12.9) pushing the corrected downstream readout toward raw's, family λ_b∈{0,10,40}. λ_b=0
LOADS the Exp 21 checkpoint (= Exp 22 corrector; reproduced B0=+28.633, distinct2=0.875 and λ_b=0 effect/d2 to the
digit — anchor); only λ_b=40 trained fresh (~900 steps, batch 2, no OOM); λ_b=10 reused the existing checkpoint.
NUANCED / CORRECTIVE result — the fix's MECHANISM transfers, its PARETO ADVANTAGE does not. Adding λ_b lifts the
corrected generation's sentiment effect from the base corrector's +0.53/+0.77/+0.98/+2.31 (10–29% of raw's, = Exp
22) to λ_b=40 +4.06/+5.87/+6.35/+4.21 (53–83% of raw's at α≤6, a 2–8× jump) — exactly the Exp 11 lever, so the
correction's non-orthogonality to the downstream readout AND the readout-preservation fix are architecture-robust.
BUT on Qwen3 the corrector does NOT beat raw: at λ_b=40 its distinct-2 (0.875→0.673) sits slightly below raw's
(0.886→0.761) at every α while its effect is also below raw's, so raw weakly dominates at matched α. Reason (= Exp
22): GPT-2's term won by dominating a COLLAPSED raw (distinct-2 0.32); Qwen3's raw does not collapse, so there is
no degenerate baseline to beat — the λ_b sweep approaches raw's frontier without passing it. Exp 20's λ_g=160
over-steer wobble replicates (λ_b=40 @α=8 effect drops to +4.21 < its α=6 peak +6.35, d2 0.673). ⇒ the behavioral
fix is a robust lever on generated effect; its payoff is GATED by whether the raw baseline degenerates. Closes the
full behavioral arc (Exp 10→11→20→22→23): matched projection ≠ matched steering everywhere; the readout-
preservation fix transfers everywhere; the size of its payoff depends on the baseline's failure mode. No prior
result superseded (Exp 23 is new; λ_b=0 reproduces Exp 22). Artifacts: `experiments/23_behavioral_qwen_fix.py`,
`results/23_behavioral_qwen_fix.json`, `results/23_corr_lamb{10,40}.pt`, `results/23_run.log`,
`plots/23_behavioral_qwen_fix.png`. RESULTS/REPORT/CHANGELOG curated; REPORT math verified (26/26 js-display-math,
0 broken, 0 inline hazards — Exp 23 reuses Exp 11's behavioral-loss equation, no new equation). ENV: dir9's cupenv
python (shared conda `transformers` still absent).
<!-- prior: S4(m) behavioral check on Qwen3 Exp 22 -->
**S4(m) Experiment 22 (new, BEHAVIORAL check on Qwen3, 2026-07-07):** picked the highest-value remaining point —
the honesty check Exp 21 itself flagged as its "Next check". Exp 21's headline 94% recovery on Qwen3-1.7B is a
TEACHER-FORCED ΔLM at matched layer-14 projection; Exp 10 taught (on GPT-2) that this proxy can hide a weaker
propagated behavioral edit in generation. Ran the IDENTICAL Exp-10 behavioral generation protocol on Qwen3-1.7B,
reusing the EXACT Exp-21 corrector checkpoint (results/21_corr.pt, no retraining): greedy-generate 30 tokens from
48 held-out 12-token prompts, steer at block 14 every position, raw vs corrected; on a clean re-encode measure
sentiment effect B(α)−B(0) (baseline B0=+28.6) and distinct-2 (baseline 0.875). CORRECTIVE / honest result — the
Exp-10 under-steering caveat REPLICATES: corrector effect only 10–29% of raw's (raw +5.22/+7.31/+7.64/+8.01 vs
corr +0.53/+0.77/+0.98/+2.31 @α=2/4/6/8), so 94% is honest as a fluency metric but partly bought by a weaker
propagated edit — exactly as on GPT-2 (~1/6 there). KEY DIFFERENCE: raw steering degenerates FAR LESS on Qwen3
(distinct-2 0.886→0.761 @α=8 vs GPT-2's collapse to 0.32), so raw is a STRONGER baseline here and the corrector's
fluency edge is small (0.06 @α=8) — the effect-vs-fluency Pareto is shallower than on GPT-2. ⇒ "matched projection
≠ matched behavioral steering" is architecture-robust; the Exp 11/20 behavioral-preservation terms (GPT-2-tested)
are the indicated fix if strong behavioral steering is required. No prior result superseded (Exp 22 measures a
behavioral quantity; Exp 21's ΔLM unchanged). No OOM (GEN_BATCH 8, empty_cache between chunks). Artifacts:
`experiments/22_behavioral_qwen.py`, `results/22_behavioral_qwen.json`, `results/22_run.log`,
`plots/22_behavioral_qwen.png`. RESULTS/REPORT/CHANGELOG curated; REPORT math verified (26/26 js-display-math, 0
broken, 0 inline hazards — Exp 22 reuses Exp 10's behavioral-metric definitions, no new equation). ENV: dir9's
cupenv python (shared conda `transformers` still absent).
<!-- prior: S4(l) cross-architecture Exp 21 -->
**S4(l) Experiment 21 (cross-ARCHITECTURE generality, 2026-07-07):** picked the single highest-value
remaining external-validity point — every model tested so far (Exp 13/19: GPT-2 small/medium/large) is the
SAME GPT-2 architecture, so the flagship result could be a GPT-2-architecture artifact. Replicated the EXACT
flagship Exp-3 pipeline UNCHANGED on Qwen3-1.7B (28 blocks, d=2048) at mid layer block 14/28 — a modern
architecture differing from GPT-2 on EVERY structural axis: RMSNorm (not LayerNorm), rotary position
embeddings (not learned), SwiGLU MLP (not GELU), grouped-query attention (16 query / 8 KV heads, not dense
MHA). Only the model changes (|v|=38.1, mean|h|=301.9, clean D_M=44.7; corrector 8.39M @ d=2048; Qwen3 bf16
for the ~4.3 GB share, corrector fp32 with a bf16 hook boundary, train batch 2 / EVAL batch 1). POSITIVE —
both headline facts replicate: raw steering breaks the LM (ΔLM@8 **+3.43**, D_M 44.7→77.8) and the identical
LM-supervised corrector recovers it at matched projection — **recovery @α=8 = 94%** (ΔLM +3.43→+0.19), **108%
@α=4** (ΔLM even below clean baseline = free-or-better weak-α, as on every GPT-2 scale), retention matched
α|v| (38.1→304.8), with the corrected activation FURTHER off the Gaussian manifold than raw at every α (122.2
vs 77.8 @α=8; Exp-2/3 decoupling holds a 4th time). 94% @α=8 edges GPT-2 small's 84%. ⇒ ARCHITECTURE-robust
across LayerNorm↔RMSNorm, learned↔rotary positions, GELU↔SwiGLU, dense↔grouped-query attention; the flagship
result now spans SEVEN axes (strength/direction/layer/model-scale/architecture/prompt-family/steering-family).
Debugging: shared /mars-vol disk contention made the cold Qwen3 load slow (~8 min tensor-by-tensor; ~21 MB/s;
re-run instant via OS page cache); first run OOM'd in eval at the float cast of Qwen3's 151,936-token vocab
logits — fixed with EVAL batch 1 + expandable_segments + empty_cache + a corrector checkpoint (results/21_corr.pt).
No prior result superseded. Artifacts: `experiments/21_cross_arch.py`, `results/21_cross_arch.json`,
`results/21_corr.pt`, `results/21_run.log`, `plots/21_cross_arch.png`. RESULTS/REPORT/CHANGELOG curated;
REPORT math verified (26/26 js-display-math, 0 broken, 0 inline hazards). ENV: dir9's cupenv python (shared
conda `transformers` still absent).
<!-- prior: S4(k) differentiable-generation Exp 20 -->
**S4(k) Experiment 20 (differentiable-generation behavioral supervision, 2026-07-07):** closed the last
substantive open lever (PLAN Next-step (i)). Exp 11's behavioral term matched the corrector's downstream
sentiment readout on a TEACHER-FORCED pass and hit a ≈+1.3 generated-effect ceiling — a proxy gap
(teacher-forced ≠ autoregressive). Exp 20 supervises the readout on the corrector's OWN generated
continuation via a DIFFERENTIABLE K=8 soft-token rollout (softmax(ℓ/τ)·Wₑ feedback, differentiable in r_θ),
pushing the corrected rollout's readout toward RAW steering's own rollout, weight λ_g∈{0,40,160}; everything
else is the Exp 11 recipe; scored on the identical Exp 10 protocol. GEN_B=4 (VRAM), no OOM. PARTIAL POSITIVE:
supervising on the autoregressive distribution BREAKS Exp 11's ceiling — λ_g=40 @α=8 raises the achievable
effect +1.08→**+1.72** at distinct-2 0.47 (raw's collapsed 0.32), and λ_g=160 @α=2 reaches effect **+1.61 at
near-baseline fluency 0.71** (dominates Exp 11's +0.99@0.73) — but OVER-weighting (λ_g=160) destabilizes and
collapses to raw-like repetition at strong steering (effect −0.22, d2 0.32 @α=8). Frontier pushed out a second
time, still not erased; the strong-effect-and-fluent corner remains genuinely hard. λ_g=0 reproduces Exp 10/11
to the digit. No prior result superseded. Artifacts: `experiments/20_diff_generation.py`,
`results/20_diff_generation.json`, `results/20_run.log`, `plots/20_diff_generation.png`. RESULTS/REPORT/
CHANGELOG curated; REPORT math verified (26/26 js-display-math, 0 broken, 0 inline hazards). ENV: dir9's
cupenv python (shared conda `transformers` still absent).
<!-- prior: S4(g-follow-up) model-scaling Exp 19 -->
**S4(g-follow-up) Experiment 19 (model-scaling to GPT-2 large, 2026-07-07):** picked the untested
optional external-validity point (a still-larger model) to strengthen the model-robustness axis from two
scales to three. Replicated the EXACT flagship Exp-3 pipeline UNCHANGED on GPT-2 LARGE (774M, 36 blocks,
d=1280) at mid layer block 18/36 — only the model changes. Downloaded gpt2-large (3.1 GB, previously
uncached); corrector 6.03M params at d=1280; trained batch 2 for VRAM (~4.3 GB share, no OOM). POSITIVE —
both headline facts replicate: raw steering breaks the LM (ΔLM@8 +2.47, D_M 35.2→66.0) and the identical
LM-supervised corrector recovers it at matched projection — **recovery @α=8 = 84%** (ΔLM +2.47→+0.39),
**95% @α=4**, free-or-better at weak α, with the corrected activation FURTHER off the Gaussian manifold than
raw at every α (96.8 vs 66.0 @α=8; Exp-2/3 decoupling holds a third time). The model axis now spans a 6×
parameter range (124M→355M→774M) with **FLAT α=8 recovery (84% / 89% / 84%)** — amortized correction quality
does not erode with model size. Retention matched α|v| exactly. No prior result superseded. Artifacts:
`experiments/19_gpt2_large.py`, `results/19_gpt2_large.json`, `plots/19_gpt2_large.png`, `results/19_run.log`.
RESULTS/REPORT/CHANGELOG curated; REPORT math verified (24/24 js-display-math, 0 broken, 0 inline hazards).
ENV: ran with dir9's cupenv python (shared conda `transformers` still absent).
<!-- prior: S4(j) steering-family Exp 18 -->
**S4(j) Experiment 18 (acts on human feedback #3, 2026-07-06):** tested the LAST open human-feedback ask
— a genuinely different steering FAMILY, beyond the 6 hand-built DiffMean concepts. Changed BOTH data source
and extraction method on the same concept (sentiment): built the vector from a REAL downloaded dataset (SST-2,
500 pos + 500 neg movie-review sentences) via the three canonical linear-steering families — DiffMean (μ⁺−μ⁻),
logistic-regression probe (discriminative), PCA-contrast (top PC of centered pos−neg pair diffs, RepE,
unsupervised). Sign-aligned + rescaled all to a common norm |v|=11.0 so ONLY the direction varies; ran the
identical flagship Exp-3 recipe per family at matched projection. POSITIVE + family-ROBUST: the three
directions are genuinely different (cos to DiffMean 1.00/0.40/0.30), all break the LM under raw steering
(ΔLM@8 +3.41/+2.63/+2.27), and the identical LM-supervised corrector recovers each — recovery@8
**86%/84%/101%** (98/95/118% @α=4). DiffMean reproduces flagship Exp 3 (86%≈84%) from real data (concept vector
only cos-0.49 reproducible across data sources, yet recipe works on both). BONUS: PCA-contrast aligns with
GPT-2's dominant high-variance axis (Exp 16), so its raw steering leaves D_M FLAT at the clean 27.3 (ON the
Gaussian manifold) yet still breaks the LM — off-Gaussian is neither necessary nor sufficient for LM damage.
Core result now robust on SIX axes (strength/direction/layer/model/prompt-family/steering-family). ALL THREE
human asks now done. No prior result superseded. Artifacts: `experiments/18_steering_family.py`,
`results/18_steering_family.json`, `plots/18_steering_family.png`, `data/sst2_train.tsv`. RESULTS/REPORT/
CHANGELOG curated; REPORT math verified (24/24 js-display-math, 0 broken, 0 inline hazards). ENV note: shared
conda `transformers` vanished this iter; ran with dir9's `cupenv` python (superset env, no state modified).
<!-- prior: S6 real diffusion corrector Exp 17 -->
**S6 Experiment 17 (acts on human feedback #1, 2026-07-06):** built the REAL diffusion machinery the
direction is named after and settled whether "diffusion" adds anything over the one-shot MLP. Three
correctors at matched projection α|v| on the same held-out FineWeb eval (GPT-2 small, block 6, sentiment):
(1) one-shot MLP (Exp 3, 4.46M), (2) COLD-DIFFUSION iterative K=8 (step-conditioned velocity field,
projection-preserving every step, LM-supervised through the K-step unroll, 4.46M), (3) GLP Gaussian-noise
DDPM prior (SDEdit, no LM, 2.69M). RESULT: recovery @α=8 = **84% / 85% / −5%**. Three answers — (RQ1) the
Cold-Diffusion CORRUPTION MODEL is what matters: LM-supervised training on the real steering corruption
recovers 84–85%, but the generic Gaussian-noise "denoise back to the manifold" prior has NEGATIVE recovery
(worse than raw); (RQ2) the iterative structure ~TIES the one-shot MLP (85 vs 84%) so iteration isn't the
source; (RQ3) the unconditional prior ERASES the steer (as-is retention 10.6/83.1 vs target 11.1/88.6
@α=1/8). Validates ColdSteer's design (condition on clean activation + LM supervision). No prior result
superseded. Artifacts: `experiments/17_diffusion_corrector.py`, `results/17_diffusion_corrector.json`,
`plots/17_diffusion_corrector.png`. RESULTS/REPORT/CHANGELOG curated; REPORT math verified (21/21
js-display-math, 0 broken, 0 inline hazards). Only human ask #3 (a different steering family) remains queued.
<!-- prior: S5 manifold geometry Exp 16 -->
**S5 Experiment 16 (acts on human feedback 2026-07-06):** the human doubted the Gaussian-manifold
assumption behind `D_M` and asked to characterize the real manifold via manifold-recovery literature.
Tested directly on clean layer-6 activations (49,218 tokens, no steering): intrinsic dimension (TwoNN —
Facco 2017; Levina–Bickel MLE — 2004; PCA participation ratio) + Gaussianity (held-out D_M² vs χ²₇₆₈).
RESULT: the cloud is NOT a single 768-d Gaussian — LOW-DIM (intrinsic dim ~8–34 ≪ 768), EXTREMELY
ANISOTROPIC (participation ratio 1.1, ~90% var in 1 PC), HEAVY-TAILED (held-out D_M² spread 6.7× the
Gaussian, 14 dims excess-kurt>1 max 118). SHARPENS the thesis: concrete mechanism for Exp 2's negative
(D_M piles volume into rogue dims → D_M-minimizing correction goes there → LM breaks); reframes "off the
Gaussian manifold" as "off a crude fit." No prior (LM-loss) number changes. Artifacts:
`experiments/16_manifold_geometry.py`, `results/16_manifold_geometry.json`, `plots/16_manifold_geometry.png`.
RESULTS/REPORT/CHANGELOG curated; REPORT math verified (18/18 js-display-math, 0 broken, 0 inline hazards).
Two human asks remain queued (see Next step): #1 a real diffusion-model corrector, #3 a different steering family.
<!-- prior: S4(i) prompt-family Exp 15 -->
S1+S2+S3 + S4(a) strength-extrap + S4(b) held-out-vector + S4(c) direction-conditional-bank +
S4(c-follow-ups) bank/capacity/curated SCALING (Exp 7/8/9) + S4(d) BEHAVIORAL text Pareto (Exp 10) +
S4(e) BEHAVIORAL-PRESERVATION term (Exp 11) + S4(f) LAYER ROBUSTNESS (Exp 12) + S4(g) CROSS-MODEL
generality (Exp 13) + S4(h) BANK-DIVERSITY causal confirmation (Exp 14) + S4(i) PROMPT-FAMILY robustness
(Exp 15) delivered — success criterion MET; direction complete on all planned axes, the flagship fluency
result now robust on FIVE generalization axes (strength/direction/layer/model/prompt-family) and the
amortization story closed with 3 scaling negatives + 1 controlled positive lever (~99%).
**S4(i) Experiment 15 (new):** acted on Next-step (ii) — held-out-prompt-family generalization, the last
untested external-validity axis. Every prior experiment both trains AND evaluates on FineWeb web text, so
the corrector could be overfit to that prompt distribution. Trained the flagship Exp-3 sentiment corrector on
FineWeb, evaluated it UNCHANGED (matched projection) on 3 held-out families of increasing distribution shift:
fineweb (in-dist), markdown (this project's research prose), code (numpy/torch/transformers Python source).
POSITIVE: NOT overfit — recovery @α=8 = 84/77/60% (95/87/78% @α=4), and recovery tracks each family's
clean-activation Mahalanobis shift under the FineWeb Gaussian MONOTONICALLY (D_M 27.5→30.1→37.4 ⇒ 84→77→60%)
— graceful degradation, the prompt-axis analogue of Exp 4's strength extrapolation. fineweb row reproduces
Exp 3 to the digit (raw +2.78 → learned +0.44, 84%). Artifacts: `experiments/15_prompt_family.py`,
`results/15_prompt_family.json`, `plots/15_prompt_family.png`. RESULTS/REPORT/CHANGELOG curated; REPORT math
verified (14/14 js-display-math, 0 broken, 0 inline hazards).
<!-- prior: S4(h) bank-diversity causal Exp 14 -->
**S4(h) Experiment 14 (prior):** acted on Next-step (ii) — confirm the bank-diversity lever directly. Exp 9
only *inferred* that bank angular diversity (not target-subspace coverage) drives conditional-corrector
recovery, because its banks confounded alignment with internal collinearity (held-out `certainty` lives in
the collinear cluster). Exp 14 removes the confound with a CONTROLLED THIRD-MEMBER SWAP: three size-3 banks,
capacity fixed 5.25M, all sharing the {sentiment,formality} anchor; only the 3rd member's collinearity with
formality varies (div=+politeness 0.07 / mid=+complexity 0.57 / coll=+concreteness 0.76). POSITIVE, causal:
(1) the swapped member's OWN recovery collapses as it collinearizes — 69%→40%→17% @α=8 (a member confusable
with a neighbor can't be specialized); (2) the confound-free isolate `sentiment` (⟂ every dir AND ⟂ target)
is corrected WORSE in more collinear banks — 63%→61%→55% @α=8 — which can ONLY be reduced separability, not
target coverage. `formality` (gains the collinear neighbor) holds ~69–70% (corrector collapses the pair onto
the dominant larger-norm member). Turns Exp 9's correlation into a controlled causal result; the positive
counterpart to Exp 7/8/9's three scaling negatives. Artifacts: `experiments/14_diversity_lever.py`,
`results/14_diversity_lever.json`, `plots/14_diversity_lever.png`. RESULTS/REPORT/CHANGELOG curated; REPORT
math verified (14/14 js-display-math, 0 broken, 0 inline hazards).
<!-- prior: S4(g) cross-model Exp 13 -->
**S4(g) Experiment 13 (prior):** answered the second obvious external-validity question — is the result a
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
**S7 seed robustness DONE (Exp 26).** The last-named review control (seed) is now covered for the flagship:
83.3±2.0% @α=8 across 5 seeds. Success criterion long met; result robust on SEVEN axes (strength/direction/
layer/model-scale/architecture/prompt-family/steering-family) PLUS seed for the flagship. Only very-low-value
optional points remain: (i) run the 5-seed control on a cross-model/architecture check (e.g. Qwen3) to give
those numbers error bars too — modest value, the flagship CI already establishes the recipe is stable; (ii) a
FURTHER architecture family (state-space/MoE) for a fuller sweep; (iii) finer λ_b + Exp-20 differentiable-
generation ON Qwen3 (Exp 23's Next check). ENV: use `/opt/conda/bin/python` (transformers 5.13.0, LOCAL disk,
imports in seconds) — the dir9 `cupenv` on /mars-vol stalls ~30 min on cold imports under disk contention.
Verify any REPORT edit with the GitHub-API math check on the touched report files.

<!-- prior next step -->
**Report restructure DONE (2026-07-09 human feedback).** REPORT.md is now an index + four topic-focused parts
(see Current status). Optional follow-ups if a future iter wants them: (a) apply the same split to RESULTS.md if
the operator finds it too long too (currently left as a single per-experiment results log); (b) the research
follow-ups below remain low-value and optional. Verify any REPORT edit with the GitHub-API math check on ALL FIVE
report files, not just REPORT.md.

<!-- prior next-step below -->
**All three human-feedback asks DONE; optional model-scale (Exp 19), differentiable-generation (Exp 20),
cross-ARCHITECTURE (Exp 21), the Qwen3 behavioral honesty-check (Exp 22), the Qwen3 behavioral-fix transfer
(Exp 23), the SECOND non-GPT-2 architecture / architecture SWEEP (Exp 24), AND the Pythia behavioral honesty-check
(Exp 25) all DONE.** Success criterion long met; direction complete on all planned axes plus the FULL behavioral
arc (Exp 10→11→20→22→23) now closed on all THREE architectures (GPT-2 Exp 10 / Qwen3 Exp 22 / Pythia Exp 25),
robust on SEVEN axes (strength/direction/layer/model-scale/architecture/prompt-family/steering-family) — the
architecture axis now a genuine 3-family SWEEP (GPT-2 / Qwen3 / GPT-NeoX, 81–94% recovery @α=8; Exp 21+24), the
behavioral caveat shown architecture-robust AND its severity shown to track how strongly raw steering propagates
(mild on Pythia, Exp 25), and the behavioral-fix mechanism shown to transfer across the architecture boundary while
its Pareto payoff is gated by baseline degeneration (Exp 23). No substantive open lever remains.
Only very-low-value untested points are left, all optional:
(i) a FURTHER architecture family (state-space / MoE, e.g. Mamba/Mixtral) for a fuller sweep beyond three;
(iii) a FINER λ_b sweep + the Exp-20 differentiable-generation term ON Qwen3 to map how close the corrected
frontier can get to raw's strong-and-fluent corner (Exp 23's own Next check) and check whether the α=8 wobble
is a schedule artifact;
(iv) a harder differentiable-generation objective (Gumbel-softmax hard samples, longer rollouts) for the
strong-effect-and-fluent corner Exp 20 left open; (v) GPT-2 XL (same architecture, adds no axis).
ENV: run experiment scripts with `/mars-vol/marsv/dir9_ood/cupenv/bin/python` until the shared conda
`transformers` is restored (still absent as of iter 21). Note: shared /mars-vol disk is contended — a cold
large-model load can take several minutes; a warm re-run is instant via OS page cache.

Prior (all delivered, success criterion long met): Core arc + all generalization axes + behavioral axis +
behavioral-preservation follow-up + LAYER-ROBUSTNESS
(Exp 12: 90/84/76% @α=8 at blocks 3/6/9) + CROSS-MODEL generality (Exp 13: GPT-2 medium, 89% @α=8 / 101% @α=4)
+ BANK-DIVERSITY causal confirmation (Exp 14) + PROMPT-FAMILY robustness (Exp 15: FineWeb-trained corrector
recovers 84/77/60% @α=8 on fineweb/markdown/code) delivered — the flagship fluency result is now robust on
FIVE generalization axes (strength/direction/layer/model/prompt-family), and the amortization story is closed
with THREE scaling negatives (Exp 7/8/9) AND one controlled positive lever (Exp 14). Optional remaining polish,
any one a clean iteration: (i) push the Exp 11 ceiling by supervising the behavioral readout THROUGH
sampled/differentiable generation rather than teacher-forced (the one substantive open lever left);
(ii) a still-larger model (GPT-2 large). All optional; success criterion long met.

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

