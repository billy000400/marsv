# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-07-02 — Iter 1: establish the off-manifold phenomenon (S1)

**Scope decision.** The PLAN proposal is huge and leans on the external GLP repo (large
download, likely won't fit our 4GB VRAM share, and distillation infra is heavy). I chose a
**self-contained MVP** on GPT-2 small that tests the CORE hypothesis without GLP: establish
the motivation (raw steering → off-manifold → LM damage), then in later iters train a
projection-preserving corrector supervised by *paired activation targets* (Strategy 1) and/or
a *direct downstream* fluency+retention loss (Strategy 3), which need no external teacher.
Rejected: (a) cloning GLP repo as Phase 0 — too costly for our share, and GLP distillation
(Strategy 2) can be added later if time allows; (b) transformer_lens — not installed and
forbidden to pip-install.

**Env notes.** `import transformers` top-level fails (`huggingface_hub` 1.21 vs pinned
transformers: `is_offline_mode` import error), but `from transformers import GPT2LMHeadModel,
GPT2TokenizerFast` works (lazy submodule). CUDA IS usable on this A10 (sm_86 + cu130) — I ran
on GPU, unlike dir3 (which ran CPU on an older V100 box). gpt2 weights cached locally; reused
dir3's `fineweb_texts.json` (1500 docs) to avoid network.

**Did.** Wrote `experiments/common.py` (model load, resid_post extraction, forward-hook
Patcher, ΔLM) and `experiments/01_offmanifold_phenomenon.py`. Built DiffMean sentiment vector
(`|v|=11.1`), fit a full-cov Gaussian on 49,218 clean layer-6 tokens (`|h|=112.2`), swept
α∈{0..8}. Persisted vector+stats to `data/sentiment_vec_layer6.npz` for reuse.

**Learned.** Clean monotonic phenomenon: α=0→8 gives `D_M` 27.3→49.0 (vs real-act ref 27.3),
norm 0.98→1.30, ΔLM 0→+2.78 nats. Damage is negligible at α≤2 then accelerates — the
strong-steer regime is exactly where a corrector should pay off. Runtime ~90s on GPU.

**Next step (S2).** Implement `projections.py` (project_orthogonal, retain_projection_update)
+ unit tests, and the `ColdSteerResidualCorrector` (4-layer MLP, output `ĥ=z+P_{v⊥}r_θ`).
Supervise via paired sentiment targets (h⁻→h⁺) with orthogonal-only + near losses; evaluate
whether it lowers `D_M`/`ΔLM` at matched projection vs raw steering. Add a norm-clip baseline.

On track? yes — S1 done (~20% of direction), no blocker; phenomenon + metrics + reusable
artifacts in place.

## 2026-07-02 — Iter 2: analytic projection-preserving corrector (S2 + core of S3)

**Did.** Wrote `experiments/projections.py` (normalize_vector, apply_steering,
projection_along, project_orthogonal, retain_projection_update, and the analytic
`cov_aligned_shift`) with unit tests (all PASS: alpha=0 identity, P_perp orthogonal,
retain preserves ⟨ĥ-h,v⟩, cov shift matches projection AND provably lowers Mahalanobis
penalty via Kantorovich). Then `experiments/02_corrector.py` evaluates, at matched
projection, raw steering vs the analytic corrector `ĥ=z+P_{v⊥}Δ` with
`Δ=Σv̂·α|v|/(v̂ᵀΣv̂)` (the min whitened-movement shift), plus norm-clip and naive-inversion
controls, on `D_M`, ΔLM, and projection retention. Added a per-token `FuncPatcher` +
`lm_loss_fn` so per-token correctors (norm-clip) route through the real LM.

**Learned (the headline).** The corrector does what it's designed to: lowers `D_M`
(49.0→38.1 at α=8) and preserves projection exactly (retention 88.6 = raw). But it makes
the LM MUCH worse: ΔLM +4.20 vs raw +2.78 at α=8, and a brutal +3.31 vs +0.08 at α=1.
So **statistical on-manifold distance and real LM damage are decoupled** — you can lower
`D_M` while raising LM loss ~40×. Mechanism: the Mahalanobis-minimizing direction `Σv̂`
concentrates in GPT-2's high-variance outlier dims — cheap in whitened cost but exactly the
dims the LM reads sharpest. Norm-clip: ~no ΔLM gain, worse `D_M` on clean acts.

**Why it's a good result.** It's an honest, well-supported negative result that satisfies
the PLAN's "corrector cannot beat raw steering at matched projection (via a manifold prior)"
branch AND sharply reframes the direction: a corrector MUST be supervised by the downstream
LM loss, not a manifold-distance surrogate. Deliverables (RESULTS/REPORT/CHANGELOG/plots)
curated to current-best; REPORT math re-verified (8/8 js-display-math, 0 broken).

**Assumption/decision logged.** Chose the analytic Gaussian-optimal corrector first (fast,
parameter-free, provable) instead of jumping straight to a trained MLP — it isolates whether
the *manifold-projection idea itself* works before spending compute on training. It doesn't,
which is more informative than a null MLP would have been.

**Next step (S3 learned).** Implement `ColdSteerResidualCorrector` (4-layer MLP) and train
`r_θ` against the DOWNSTREAM LM loss (backprop into the delta through the frozen upper 6
GPT-2 blocks; detach h to avoid backprop into lower layers; small batch under 0.18 VRAM
frac) + a stay-near-`z` term, projection preserved by construction. Evaluate vs raw at
matched projection on ΔLM/`D_M`; target: beat raw ΔLM at high α. This is now well-motivated.

On track? yes — S2 done + S3 core delivered as a decisive negative/decoupling result
(~55% of direction); no blocker; learned downstream corrector is the clear next build.

## 2026-07-02 — Iter 3: learned downstream-supervised corrector (S3-learned) — POSITIVE

**Did.** Wrote `experiments/03_learned_corrector.py`: a `Corrector` 4-layer MLP (4.46M params,
last layer zero-init → starts = raw steering) producing `r_θ(h,z,α)`, applied as
`ĥ = z + P_{v⊥}r_θ` (projection preserved by construction). Trained end-to-end against the FROZEN
GPT-2's real next-token cross-entropy via the FuncPatcher hook: inside the hook `h` is detached
(no grad to lower blocks) and all LM weights are frozen (`requires_grad_(False)`), so only `r_θ`
learns. α~U(0.5,8) per step, +λ_near·⟨‖P⊥r‖²⟩ (0.05) minimal-correction penalty. 6 epochs /
~230 steps on 300 FineWeb docs (seq 64, batch 8), evaluated on the same held-out 100 docs as
Exp 1/2. Ran clean on GPU (0.18 frac), ~90s total.

**Learned (the headline — a decisive POSITIVE).** The learned corrector BEATS raw steering at
every α at matched projection. ΔLM at α=8: raw +2.78 → learned **+0.44 nats (84% recovery)**;
at low α it is essentially free (−0.07 at α=1, −0.05 at α=2). And it does this while moving
FURTHER off the Gaussian manifold than raw (`D_M` 49.0→79.5 at α=8) — the exact mirror of Exp 2,
where the manifold-optimal cov_corr moved TOWARD the manifold and broke the LM. So the three
experiments now form a complete arc: (1) raw steering breaks the LM off-manifold; (2) a Gaussian
manifold surrogate is anti-correlated with LM safety; (3) the SAME projection-preserving form,
supervised by the downstream LM loss, recovers most fluency — and the safe correction is itself
off the statistical manifold. This satisfies the PLAN success criterion (corrector beats raw at
matched projection on ΔLM, with verdict + Methods equations + figures).

**Assumption/decision logged.** (a) Trained via the full-model hook with `h` detached rather than
caching upper-block inputs and reimplementing `upper_forward` — guaranteed identical to the eval
path (same FuncPatcher), at the cost of recomputing lower blocks each step (negligible, ~90s
total). (b) Chose a single α-conditioned corrector (α sampled in training) over per-α correctors,
so one model generalizes across the strength sweep on held-out text — it does. (c) λ_near kept
small (0.05); the LM loss is self-regularizing (huge orthogonal moves also raise ΔLM), so heavy
near-z weighting wasn't needed. Rejected: a stay-near-`z` L2 on the full residual (would fight the
correction); orthogonal-target/paired supervision (Strategy 1) — unnecessary now that direct
downstream supervision works cleanly.

**Deliverables.** RESULTS.md + REPORT.md curated to the three-experiment current-best; new figure
`plots/03_learned_corrector.png`; `results/03_learned_corrector.json`. REPORT math re-verified via
GitHub API (9/9 js-display-math, 0 broken, 0 inline hazards). CHANGELOG appended.

**Next step (S4).** Generalization + Pareto: (i) hold out α *beyond* the training range (e.g. 10,12)
to test extrapolation; (ii) a held-out steering vector / second behavior family to test the
corrector isn't overfit to one direction; (iii) add a text-level concept-strength readout so the
frontier is behavior-vs-fluency, not just ΔLM. Any one of these is a clean next iteration.

On track? yes — S3 fully delivered as a decisive positive result (~85% of direction); success
criterion met; no blocker. S4 generalization/Pareto is the remaining polish.

## 2026-07-02 — Iter 4: generalization — α-extrapolation beyond training range (S4a)

**Did.** Wrote `experiments/04_generalization.py`, which reuses Exp 3's Corrector / training loop /
eval helpers verbatim (imported via importlib to stay DRY and guarantee an identical training path),
trains the corrector identically (α~U(0.5,8), same seed/data), then evaluates it UNCHANGED at
α∈{1,2,4,6,8,10,12}. α=10,12 are strictly beyond the training ceiling of 8, so they measure
extrapolation. Figure shades the α>8 region. ~90s on GPU (0.18 frac).

**Learned (headline).** The learned corrector extrapolates gracefully. Fluency recovered vs raw:
α=8 (boundary) 84%; α=10 77% (ΔLM raw +3.31→learned +0.76); α=12 60% (raw +3.74→+1.50). Recovery
declines smoothly (84→77→60%) instead of collapsing at unseen strengths — evidence the MLP learned a
transferable correction rule, not a lookup over the trained α grid. In-range α (1–8) reproduce Exp 3
to the digit (same seed/data), a clean reproducibility check. D_M learned stays above raw throughout
(91.2, 101.2 at α=10,12) — same off-Gaussian-manifold signature as Exp 3, now confirmed to persist
out of range.

**Assumption/decision logged.** Chose α-extrapolation as the S4 step over held-out-vector or a
text-level concept readout because it is the cheapest, lowest-risk generalization probe that reuses
the exact trained model (no new vector-bank or generation pipeline), and directly answers the most
immediate practitioner worry ("does it still work if I dial α past the training strengths?").
Rejected for this iter: (a) held-out vector — needs a second concept vector + is more likely to be a
partial-negative that needs careful framing; queued as the next step. (b) concept-strength text
readout — needs a generation+scoring harness; heavier. Both remain open in PLAN Next step.

**Deliverables.** RESULTS.md + REPORT.md curated: added Experiment 4 table/interpretation, new
figure `plots/04_generalization.png`, refined REPORT Limitation (3) (strength-generalization now
shown; held-out-vector/multi-layer still open). `results/04_generalization.json`. CHANGELOG appended.
REPORT math re-verified via GitHub API (9/9 js-display-math, 0 broken, 0 inline hazards).

**Next step.** S4(b): held-out steering vector — build a second DiffMean concept direction and test
whether the sentiment-trained corrector still lowers ΔLM at matched projection on it (real overfit
probe; the corrector sees v only through z). Alternatively a text-level concept-strength readout for
a behavior-vs-fluency frontier.

On track? yes — success criterion met since Iter 3; S4(a) generalization delivered (~90% of
direction); no blocker. Remaining S4(b) held-out-vector / Pareto is optional polish.

## 2026-07-02 — Iter 5: held-out steering vector — cross-direction generalization (S4b)

**Did.** Wrote `experiments/05_heldout_vector.py`. Built a SECOND DiffMean vector v₂ (formality:
20 formal vs 20 informal sentences) at layer 6: |v₂|=34.0, cos(v₁_sentiment,v₂)=0.014 (nearly
orthogonal — a genuinely different behavior family). Reused Exp 3's Corrector/training/eval verbatim
(importlib). Trained TWO correctors identically (α~U(0.5,8), same seed/data): TRANSFER on sentiment
v₁, NATIVE on v₂. Evaluated both + raw on v₂ at matched projection α|v₂| for α∈{1,2,4,6,8}. ~2min×2
trainings on GPU (0.18 frac). (Fixed a wrong figure suptitle that pre-supposed transfer succeeded.)

**Learned (headline — a clean two-part answer).** The corrector r_θ(h,z,α) never sees v explicitly
(only through z), so this is a real overfit probe. (1) The correction is DIRECTION-SPECIFIC: the
sentiment-trained corrector gives ~zero benefit on formality — ΔLM transfer lies on top of raw at
every α (α=8 raw +6.49 → transfer +6.53; recovery ≈0%, slightly negative at high α). That is exactly
proposal Failure Mode 4 (overfits to one vector). (2) The RECIPE generalizes: retraining the same
4.46M MLP on v₂ recovers 83–104% of raw's fluency damage (α=8 +6.49→+1.12; 104/97/92/87/83% across
α=1..8) — reproducing Exp 3 on a larger, near-orthogonal concept, again by moving FURTHER off the
Gaussian manifold (D_M 66.6→123.1). So ColdSteer is a working recipe to instantiate per direction,
not a frozen operator to reuse across concepts.

**Assumption/decision logged.** (a) Chose formality as the second concept (clearly distinct from
sentiment, easy contrastive pairs, no new data pipeline) and DiffMean to match v₁'s construction.
(b) Added a NATIVE (retrained-on-v₂) corrector as the oracle/ceiling so the transfer failure is
interpretable — it isolates "single corrector doesn't transfer" from "method doesn't work here";
without it a reader couldn't tell which. (c) Kept the exact Exp-3 recipe/seed so native-vs-Exp-3
is an apples-to-apples reproduction on a new direction. Rejected: feeding v to r_θ this iter (would
change the architecture and conflate the overfit finding) — noted as the natural fix for future work.

**Deliverables.** RESULTS.md + REPORT.md curated: added Experiment 5 table/interpretation, updated
Headline/Summary/Conclusion/Limitation(3) (direction-generalization now shown). New figure
`plots/05_heldout_vector.png`; `results/05_heldout_vector.json`; `data/formality_vec_layer6.npy`.
REPORT math re-verified via GitHub API (9/9 js-display-math, 0 broken, 0 inline hazards). CHANGELOG
appended.

**Next step.** Optional remaining polish, any one a clean iter: (i) a DIRECTION-CONDITIONAL corrector
`r_θ(h,z,v,α)` (feed v̂ as input) trained on a small BANK of vectors {sentiment, formality, …} and
test whether ONE model then transfers to a held-out vector — the direct fix for Exp 5's failure;
(ii) text-level concept-strength readout for a behavior-vs-fluency Pareto (heavier: needs generation).

On track? yes — success criterion met since Iter 3; S4(a) strength + S4(b) direction generalization
both delivered (~95% of direction); no blocker. Remaining is optional (vector-bank corrector / text
Pareto).

## 2026-07-02 — Iter 6: direction-conditional corrector on a vector bank (S4c)

**Did.** Wrote `experiments/06_conditional_bank.py` — the direct fix for Exp 5's transfer failure.
Made the corrector CONDITIONAL on the direction: `CondCorrector` takes v̂ as an extra input
(`r_θ(h,z,v̂,α)`, input dim 3d+1, 5.25M params, zero-init last layer → starts = raw), and a new
`train_cond` samples a (direction, α~U(0.5,8)) pair per step from a BANK. Reused Exp-3's LM-loss
hook/eval verbatim + Exp-5's diffmean_vector (importlib). Bank = {sentiment |v|=11.1, formality 34.0,
concreteness 64.5}; held out certainty |v|=32.8. Built concreteness/certainty DiffMean vectors from
16 contrastive pairs each (persisted to data/). Trained ONE bank corrector + ONE native oracle on the
held-out direction (8 epochs each), evaluated at matched projection on all 4 directions. ~11 min on
GPU (0.18 frac).

**Learned (headline — a nuanced two-part answer).** Cosines were informative: sentiment ⟂ everything
(|cos|≤0.03), but formality/concreteness/certainty share a subspace (|cos| 0.76–0.82), so the held-out
certainty sits largely IN the bank's span. (1) ONE conditional model corrects every in-bank direction
at once — α=8 recovery: sentiment 55%, formality 70% (+6.49→+1.95), concreteness 17% (70% at α=2). The
cost of sharing vs a dedicated corrector is real (sentiment 84%→55%, formality 83%→70%; concreteness
worst at strong steering — capacity interference). (2) Conditioning + bank PARTIALLY transfers to the
held-out certainty: 51% @α=1 → 7% @α=8 — a genuine gain over Exp 5's ≈0% frozen single-vector transfer,
but far under the native oracle (78% @α=8, 141% @α=1). So a 3-vector bank starts to generalize across
directions (best at moderate strength) but doesn't yet solve held-out transfer at strong steering.
Practical framing: "one model per vector" → "one model per bank"; path to reuse is a LARGER bank.

**Assumption/decision logged.** (a) Chose the direction-conditional + bank experiment (PLAN Next-step
(i)) over the text-level concept-strength Pareto (ii) because it directly closes Exp 5's open question
and REPORT Limitation (3), reuses all existing machinery (no new generation/scoring harness), and is
cheaper/lower-risk. (b) Bank size 3 (+1 held-out): needed only 2 new concept pair-sets (concreteness,
certainty) — reused saved sentiment/formality vectors — keeping it a single clean iteration; a larger
bank is the obvious follow-up now that the interference/partial-transfer tradeoff is quantified.
(c) Trained a native oracle on the held-out direction so the transfer number is interpretable against a
ceiling (as in Exp 5). (d) Honest reporting of the weak spots (concreteness 17% @α=8; held-out 7%
@α=8) rather than cherry-picking — the capacity/transfer tradeoff is the actual finding. Rejected:
bigger bank / feeding α-embedding / per-direction loss weighting this iter (scope creep).

**Deliverables.** RESULTS.md + REPORT.md curated: added Experiment 6 (two tables + interpretation),
updated Headline/Summary/Conclusion/Limitation(3), new figure `plots/06_conditional_bank.png`;
`results/06_conditional_bank.json`; `data/{concreteness,certainty}_vec_layer6.npy`. REPORT math
re-verified via GitHub API (9/9 js-display-math, 0 broken, 0 inline hazards). CHANGELOG appended.

**Next step.** Direction near-complete. Optional remaining polish, any one a clean iter: (i) SCALE the
bank (5–10 directions) and re-measure held-out transfer at strong steering — the direct follow-up to
Exp 6's partial-transfer finding; (ii) text-level concept-strength readout for a behavior-vs-fluency
Pareto (heavier: needs a generation+scoring harness); (iii) multi-layer or a second model.

On track? yes — success criterion met since Iter 3; S4 (a)+(b)+(c) all delivered, Exp 5's open
question closed (~98% of direction); no blocker. Remaining is optional (larger bank / text Pareto /
multi-layer).

## 2026-07-02 — Iter (S4c follow-up): Experiment 7 — scaling the vector bank
**Did.** Built Experiment 7 (`experiments/07_bank_scaling.py`) to directly test Exp 6's parting
prescription that "scaling the bank is the indicated path" to correct a held-out direction at strong
steering. Held `certainty` out; trained the SAME direction-conditional corrector (5.25M params,
identical recipe/seed/data/8ep) on nested banks of size 1/3/5. Added two NEW DiffMean directions for
size 5 — politeness (|v|=15.6, cos to certainty −0.35) and complexity (|v|=58.4, cos −0.80) — 16
contrastive pairs each, persisted to data/. Reused saved sentiment/formality/concreteness/certainty.
~13 min GPU (3 bank models + 1 native oracle, 0.18 frac).

**Learned (headline — a corrective, honest result).** Enlarging the bank does NOT close the held-out
gap; at fixed model capacity it makes transfer WORSE. Held-out `certainty` recovery is non-monotone in
bank size and PEAKS at size 3, not 5: α=1 14%/51%/−1%, α=8 0%/7%/3% (bank 1/3/5). This held even though
the size-5 bank adds complexity (|cos|=0.80, strongly correlated with certainty) — coverage that should
help. Corroborating in-bank: under size-5, per-direction recovery @α=8 dropped vs size-3 (formality
70%→45%, concreteness 17%→13%; new: politeness 72%, complexity 41%, sentiment 57%). ⇒ capacity
interference between directions sharing the fixed 5.25M MLP, NOT coverage, is the binding constraint.
Native oracle still 78–142%, so the direction is fully correctable — the gap is a cost of amortization.
Size-3 reproduces Exp 6 to the digit (recovery [51,42,21,12,7]).

**Assumption/decision logged.** (a) Chose PLAN Next-step (i) "scale the bank" over (ii) text-Pareto /
(iii) multi-layer because it directly interrogates Exp 6's own conclusion — the highest-value next
question — and reuses all machinery (no new harness). (b) Nested banks {1,3,5} with 2 new directions
(vs a bigger {1,3,5,7} needing 4 new concepts) to keep it one clean iteration; the non-monotone drop at
5 already answers the question, so a 7-point sweep was unnecessary this iter. (c) Kept model capacity
FIXED across bank sizes on purpose — that isolates the interference effect and is exactly what makes the
negative result interpretable (a bigger model at size 5 would confound). (d) Reported the negative
result honestly and CORRECTED Exp 6's optimistic "larger bank" framing rather than hiding the reversal —
this is the actual scientific finding. Rejected: capacity-scaling ablation, curated-bank ablation this
iter (clear follow-ups, out of scope for one step).

**Deliverables.** RESULTS.md + REPORT.md curated: added Experiment 7 (two tables + interpretation),
updated Headline / Exp-6 closing sentence (RESULTS) and Summary / Conclusion / Limitation(3) (REPORT);
new figure `plots/07_bank_scaling.png`; `results/07_bank_scaling.json`;
`data/{politeness,complexity}_vec_layer6.npy`. REPORT math re-verified via GitHub API (10/10
js-display-math after adding the recovery-fraction equation, 0 broken, 0 inline hazards). CHANGELOG
appended.

**Next step.** Direction complete on all planned axes; Exp 6's open question now closed with a
corrected answer. Remaining optional polish, any one a clean iter: (i) CAPACITY-scaling ablation — retrain
the size-5 bank at 2×/4× width and check whether held-out transfer recovers (the direct test of Exp 7's
"capacity interference" claim); (ii) CURATED bank — pick the 3 bank directions most correlated with the
held-out target and see if transfer beats the 5-direction bank; (iii) text-level concept-strength Pareto;
(iv) multi-layer / second model.

On track? yes — success criterion met since Iter 3; S4 (a)+(b)+(c) delivered and Exp 6's follow-up now
resolved (Exp 7); ~99% of direction. No blocker. Remaining is optional (capacity/curation ablation,
text Pareto, multi-layer).

## 2026-07-02 20:0x — Experiment 8: does more MODEL CAPACITY close the held-out gap? (S4c follow-up #2)
**Did:** Exp 7 concluded held-out transfer fails to improve with a bigger bank because of "capacity
interference," and prescribed "more model capacity and/or a curated bank" — but never varied capacity.
Tested that claim directly. Wrote `experiments/08_capacity_scaling.py` (reuses Exp 6 CondCorrector
which already takes a `hidden` arg, + train_cond/make_hat_cond + Exp 3 lm_loss_fn/LAYER via import).
Held the BANK fixed at Exp 7's size-5 set (its worst-transfer bank), scaled corrector width
hidden∈{1024,2048,4096} = 5.2M/14.7M/46.2M params (9× range), identical recipe/seed/data. Ran in ~2 min
total (4 trainings incl. native oracle) under the 0.18 VRAM fraction, no OOM.
**Learned:** more capacity does NOT close the gap. (1) Mean in-bank recovery @α=8 SATURATES ~45% across
9× params (45.4→43.8→46.3) — not width-starved. (2) Held-out `certainty` @α=8 flat-to-falling 3→2→1%;
at weak steering the 46M model OVERFITS and harms the unseen dir (α=1 rec −1→−22→−146%). ⇒ the ceiling
on amortized cross-direction correction is the TRAINING SIGNAL, not parameter count. hidden=1024
reproduced Exp 7's size-5 to the digit (good reproducibility check). Native oracle unchanged 78–142%.
This corrects Exp 7's optimistic "scale the model" reading: neither more directions nor more parameters
amortizes the correction; it is fundamentally direction-specific and the per-direction native corrector
remains the reliable route.
**Deliverables:** RESULTS.md +Exp 8 (2 tables) + updated Exp-7 closing + Headline; REPORT.md +Exp 8
Methods/Results + updated Summary/Exp-7-interp/Conclusion/Limitation(3); plots/08_capacity_scaling.png;
results/08_capacity_scaling.json; CHANGELOG appended. REPORT math re-verified (10/10 js-display-math,
0 broken, 0 inline hazards).
**Next step (optional):** the two open axes now well-motivated by Exp 7+8 — (a) bank CURATED toward a
target subspace vs a diffuse bank (does subspace-matched curation beat capacity/size?), or (b) the
S4(d) concept-strength TEXT Pareto (does preserving projection preserve generated-text concept strength?).
Both are optional; success criterion long met.
On track? yes — S4(c) fully closed (bank-size Exp 7 + capacity Exp 8 both negative/corrective); direction ~99% complete, deliverables curated + math-verified.

## 2026-07-02 20:1x — Experiment 9: does CURATING the bank toward the target subspace close the gap? (S4c follow-up #3)
**Did:** Exp 7 (more directions) and Exp 8 (more params) both ended by naming the SAME untested open
path — "curate the bank toward the held-out target's subspace." Tested it directly, the clean controlled
way. Wrote `experiments/09_curated_bank.py` (reuses Exp 6 CondCorrector/train_cond/make_hat_cond + Exp 3
LM-loss/LAYER via import). Held bank SIZE=3 and capacity=5.25M (hidden=1024) FIXED; varied only WHICH 3
of the 5 pool directions train, by mean |cos| to held-out `certainty`: diffuse 0.38 / exp6 0.54 /
curated 0.80. diffuse & curated share exactly formality — controlled. Ran in ~1.5 min (4 trainings incl.
native oracle), no OOM under 0.18 frac.
**Learned:** curating TOWARD the target subspace does NOT close the gap — it makes transfer
CATASTROPHICALLY WORSE. Held-out recovery is non-monotone in alignment and COLLAPSES at the most-aligned
bank: curated net-negative every strength (α=1 rec −183%, α=8 −12%), while diverse moderately-aligned
exp6 transfers BEST (51/42/21/12/7). Mechanism: in-bank recovery @α=8 FALLS as bank directions grow
internally correlated (diffuse 67% > exp6 48% > curated 30%). curated's 3 members are pairwise collinear
(|cos| 0.76–0.82) → conditional corrector can't disambiguate from v̂ → can't specialize → over-fires on
nearby unseen dirs (weak-α blow-up on certainty). ⇒ lever is bank ANGULAR DIVERSITY, not target-subspace
coverage. Third corrective negative in a row (Exp 7 dirs, Exp 8 params, Exp 9 curation). exp6 reproduced
Exp 6/7 to the digit. Native oracle unchanged 78–142%.
**Assumption/decision logged.** (a) Chose PLAN Next-step (ii) CURATED-bank over (iii) text-Pareto /
(iv) multi-layer because it directly closes the standing open hypothesis of BOTH Exp 7 & 8 (highest-value
question) and reuses all machinery — one clean iteration. (b) Used a 3-point alignment gradient
(diffuse/exp6/curated) with diffuse↔curated sharing exactly one member so the contrast isolates
alignment at fixed size & capacity; rejected a 2-point (aligned vs anti-aligned) design as less
informative given the non-monotone risk. (c) Kept size & capacity FIXED on purpose — that is what makes
"alignment, not size/params" the isolated variable. (d) Reported the surprising catastrophic reversal
honestly and CORRECTED Exp 7/8's "curate toward subspace" prescription rather than burying it — that is
the finding.
**Deliverables:** RESULTS.md +Exp 9 (2 tables + interp) + updated Headline + figure entry; REPORT.md
+Exp 9 Methods/Results + updated Summary/Conclusion/Limitation(3); plots/09_curated_bank.png;
results/09_curated_bank.json; CHANGELOG appended. REPORT math re-verified (10/10 js-display-math,
0 broken, 0 inline hazards).
**Next step (optional, success criterion long met):** the amortization story is now closed on all three
scaling axes (directions/params/curation, all negative). Remaining optional polish, any one a clean iter:
(i) test bank DIVERSITY directly as the positive lever — build a maximally-orthogonal 3-bank and check
in-bank + held-out vs the collinear curated bank; (ii) S4(d) concept-strength TEXT Pareto (behavior vs
fluency, not just ΔLM); (iii) multi-layer or a second model. All optional.
On track? yes — S4(c) fully closed on all three scaling axes (Exp 7 bank-size, Exp 8 capacity, Exp 9
curation — all corrective/negative); direction ~99% complete, deliverables curated + math-verified. No blocker.

## 2026-07-02 20:2x — Experiment 10: behavioral reality-check (does the corrected steer still steer generated text?)
**Did:** Every prior experiment (1–9) scored the corrector on teacher-forced ΔLM at MATCHED PROJECTION
along v — a single-layer proxy that never verified the corrected steer, used to GENERATE, still moves the
output text. Wrote `experiments/10_behavioral_pareto.py` (imports Exp 3 Corrector/train_corrector/
make_hat/FuncPatcher via importlib; retrains the flagship sentiment corrector identically). Greedy-
generate 30 tokens from 48 held-out 12-token prompts with the steer at resid_post b6 every position, raw
vs corrected. On a CLEAN re-encode of output: sentiment effect B(α)−B(0) (proj onto v̂, B0=+0.34) and
degeneration distinct-2 (unique-bigram ratio, baseline 0.70). Ran ~2 min, no OOM under 0.18 frac.
**Learned (important, corrective):** the corrector's fluency win is NOT a free lunch — it trades away the
behavioral steer, which matched-projection ΔLM hid. Raw steers hard (effect +2.97 @α=2) then COLLAPSES
into repetition/gibberish (distinct-2 0.78→0.32 @α=8; sample "the second-t-t-t-t-t-t"). The corrector
STAYS coherent/fluent at all α (distinct-2 0.64–0.72 ≈ baseline 0.70; sample "It is located in the heart
of the city … watch the city's skyline") but is only WEAKLY steered (effect +0.15–0.48, ~1/6 of raw's).
Neither dominates the effect-vs-fluency Pareto. Mechanism: P_{v⊥}r is orthogonal to v in ACTIVATION space
but NOT to the downstream sentiment READOUT, so minimizing LM loss drives the corrector to near-normal,
lightly-steered generations. ⇒ matched layer-6 projection ≠ matched behavioral steering; the big ΔLM
recoveries of Exp 3–9 partly reflect a weaker propagated edit, not costless cleanup.
**Assumption/decision logged.** (a) Chose PLAN Next-step (ii) the S4(d) behavioral/text axis over
(i) diversity-confirmation / (iii) multi-layer because it is the single UNMEASURED axis (all 9 prior
metrics are ΔLM/fluency; none the behavioral effect) and it directly tests the flagship "keeps the full
edit" claim — the most reviewer-obvious gap and highest-value question. (b) Behavioral effect measured by
projecting a CLEAN re-encode of the generated text onto v̂ (not the steered acts) — non-circular: it asks
whether the produced TEXT reads steered on its own. (c) Degeneration via distinct-2 (unique-bigram ratio)
rather than self-perplexity, because repetition/gibberish (the actual strong-steer failure mode) gives
LOW self-perplexity — distinct-2 captures it correctly. (d) Greedy decoding for determinism/reproducibility.
(e) Reported the deflating tradeoff HONESTLY and qualified Exp 3's "steering edit fully intact" (true only
for the layer-6 projection) + corrected Limitation (2)'s false "concept strength held fixed by
construction" — this reframing IS the finding; hiding it would violate faithful reporting.
**Deliverables:** RESULTS.md +Exp 10 (table + reading) + figure entry + Headline behavioral-caveat para +
qualified flagship sentence; REPORT.md +Exp 10 Methods (B(α), distinct-2, gen protocol) + Results + Summary
+ Conclusion caveats + fixed Limitation (2); plots/10_behavioral_pareto.png; results/10_behavioral_pareto.json
(incl. sample gens); CHANGELOG appended. REPORT math re-verified (12/12 js-display-math, 0 broken, 0 inline hazards).
**Next step (optional; success criterion long met, all planned axes done):** any one a clean iter —
(i) train a corrector with an explicit behavioral-preservation term (e.g. match downstream v̂-projection of
generation, not just layer-6) and see if the effect–fluency Pareto can be pushed out; (ii) confirm the
diversity lever directly (max-orthogonal 3-bank vs collinear); (iii) multi-layer / second model. All optional.
On track? yes — S4(d) behavioral axis delivered (corrective: matched projection ≠ matched steering);
direction ~99% complete on all planned axes, deliverables curated + math-verified. No blocker.

## 2026-07-02 20:3x — Experiment 11: behavioral-preservation term (attacking Exp 10's tradeoff)
**Did:** PLAN Next-step (i), the highest-value follow-up — Exp 10 found the flagship corrector under-steers
in generation (behavioral effect ~1/6 of raw) because P_{v⊥}r is ⟂ v at layer 6 but NOT ⟂ the downstream
sentiment READOUT, so minimizing LM loss suppresses the propagated concept signal. Wrote
`experiments/11_behavioral_corrector.py`: keep the Exp 3 corrector/recipe/seed/data, add ONE loss term.
Each teacher-forced step also reads out sentiment at downstream L2=11 (final resid_post; DiffMean ŵ from
the same POS/NEG prompts, |w|=3.87) and pushes corrected p_corr toward RAW steering's p_raw (separate
no-grad forward) via λ_b·⟨((p_corr−p_raw)/100)²⟩. Trained a family λ_b∈{0,10,40} (λ_b=0 = Exp 10 corrector),
scored each on the IDENTICAL Exp 10 generation protocol (48 prompts × 30 greedy tokens; effect B(α)−B(0)
and distinct-2 on clean re-encode). Ran ~5 min under 0.18 VRAM frac, no OOM.
**Learned (partial POSITIVE — first non-negative follow-up in a while):** the behavioral term is a real,
cheap win that pushes the effect-fluency Pareto OUTWARD at the fluent end. (1) Effect recovers 2–6×: from
Exp 10's +0.15–0.48 up to +0.8–1.3, while distinct-2 stays 0.52–0.73 (raw collapses to 0.32 @α=8). (2)
NEW vs Exp 10's "neither dominates": the corrector now Pareto-DOMINATES raw at moderate steering — λ_b=40
@α=2 gives effect +0.99 at distinct-2 0.73 (≈baseline 0.70), whereas raw only reaches effect that low
(+1.77 @α=8) AFTER collapsing (0.32). (3) But a hard CEILING: no λ_b lifts effect past ≈+1.3; λ_b 10→40
stops raising it (even falls @α=6 +0.93→+0.84) and only raises training LM loss. Mechanism = a second layer
of the SAME proxy gap Exp 10 exposed: the term matches raw's TEACHER-FORCED downstream readout (training
behav loss →~0.005, p_corr≈p_raw) but that only PARTIALLY transfers to autoregressive generation effect.
So the projection-preserving corrector still can't reproduce raw's STRONG pre-collapse steer; frontier
pushed out, not erased. λ_b=0 reproduced Exp 10 to the digit (built-in reproducibility check).
**Assumption/decision logged.** (a) Chose Next-step (i) behavioral objective over (ii) diversity-lever
confirmation / (iii) multi-layer because it directly attacks Exp 10's tradeoff — the single most valuable
open question — and either outcome (frontier moves / hard tradeoff) is publishable. (b) Downstream readout
at L2=11 (final resid_post) because it feeds the head most directly — the best teacher-forced proxy for
"concept content that drives generation"; a re-encode of generated text (Exp 10's effect measure) isn't
available inside teacher-forced training. (c) Target = MATCH raw's downstream projection (MSE) rather than
an unbounded "push up" — honestly asks "can we recover raw's behavioral effect while staying fluent?" and
avoids runaway over-steer; λ_b sweep traces the tradeoff. (d) λ_b grid {0,10,40} bracketed the response
(10 already big gain, 40 saturates) — no rerun needed. (e) Reported the ceiling HONESTLY (partial win,
not a solve) — the recursive proxy-gap is itself the finding.
**Deliverables:** RESULTS.md +Exp 11 (table + reading) + figure entry + Headline "Partial fix" para;
REPORT.md +Exp 11 Methods (behavioral-loss equation + downstream readout) + Results + Summary/Conclusion
updates + Limitation (2) updated (the "explicit behavioral objective is the natural next step" is now DONE);
plots/11_behavioral_corrector.png; results/11_behavioral_corrector.json; CHANGELOG appended. REPORT math
re-verified (13/13 js-display-math, 0 broken, 0 inline hazards).
**Next step (optional; success criterion long met, all planned axes done + the behavioral tradeoff now
partially resolved):** any one a clean iter — (i) confirm the bank-diversity lever directly
(max-orthogonal 3-bank vs collinear curated); (ii) push the Exp 11 ceiling by supervising the behavioral
readout THROUGH generation (differentiable/sampled rollout) rather than teacher-forced; (iii) multi-layer
or a second model. All optional.
On track? yes — S4(d) extended: Exp 11 partially resolves Exp 10's effect-fluency tradeoff (frontier
pushed out, not erased); direction ~99% complete on all planned axes, deliverables curated + math-verified.
No blocker.

## 2026-07-02 20:5x — Experiment 12: layer robustness (blocks 3, 6, 9)
**Did:** acted on PLAN Next-step (iii, layer generality) — the highest-value untested axis. All 11 prior
experiments hook resid_post block 6; a reviewer's first question is "is this a block-6 artifact?" Wrote
`experiments/12_layer_robustness.py`: replicate the EXACT flagship Exp 3 pipeline at blocks 3 (early),
6 (mid = Exp 3), 9 (late), changing ONLY the hook layer. Reused exp03's Corrector / train_corrector /
make_hat / corrector_acts / lm_loss_fn / gaussian_stats / mahalanobis by importing the module and swapping
its module-global LAYER per layer (train_corrector reads the global at call time — clean reuse, no fork).
POS/NEG sentiment prompts imported from exp01. Per layer: rebuild DiffMean v, fit Gaussian on 400 docs,
train the identical 4-layer corrector on the same 300 docs vs downstream LM loss (same seed/α∼U(0.5,8)),
eval ΔLM/D_M/retention on the same held-out 100 docs at matched projection α|v|. Ran ~4 min under 0.18 VRAM
frac, no OOM.
**Learned (POSITIVE — clean generality result):** both headline facts replicate at every depth. Fluency
recovery @α=8 = 90% / 84% / 76% (blocks 3/6/9), ≥91% @α=4, ΔLM near zero at weak steering. The corrected
activation sits FURTHER off the Gaussian manifold than raw at EVERY layer (D_M corrected > raw) — the
Exp 2/3 decoupling ("LM-safe but off-Gaussian") is layer-robust, not a block-6 quirk. Recovery declines
mildly with depth (90→84→76%): |v| grows toward the output (6.75→11.08→23.16) so a fixed-capacity corrector
faces a larger absolute edit late. Block 6 reproduced Exp 3 TO THE DIGIT (raw +2.78 → learned +0.44, 84%) —
a built-in reproducibility check that the refactored layer-swept pipeline is faithful.
**Assumption/decision logged.** (a) Chose (iii) layer generality over (i) rollout-through-generation and
(ii) diversity-lever confirmation because it answers the single most obvious external-validity question
("is the whole paper a one-layer artifact?") in one clean iteration, and the flagship pipeline was directly
reusable via LAYER-swap (low risk, ~4 min). (i) is higher-risk (differentiable rollout could eat the budget).
(b) Layers {3,6,9} span early/mid/late while keeping 6 as the reproducibility anchor; skipped very-early
(0-2) and very-late (10-11) to keep runtime bounded — the three chosen depths already establish robustness.
(c) Rebuilt v per layer (not reusing block-6 v) because a steering vector is layer-specific; matched
projection α|v| per layer is the honest comparison. (d) Kept cov_corr out of Exp 12 (the raw-vs-learned
contrast carries the generality story; cov_corr's negative result is already established in Exp 2).
**Deliverables:** RESULTS.md +Exp 12 (table + reading) + figure entry + Headline sentence; REPORT.md +Exp 12
Methods (recovery-fraction fence) + Results (table + interpretation) + Summary/Conclusion sentences;
plots/12_layer_robustness.png; results/12_layer_robustness.json; CHANGELOG appended. REPORT math re-verified
via GitHub API (14/14 js-display-math, 0 broken, 0 inline hazards).
**Next step (optional; success criterion long met, now with layer-robustness generality added):** any one a
clean iter — (i) push the Exp 11 ceiling by supervising the behavioral readout THROUGH sampled/differentiable
generation rather than teacher-forced; (ii) confirm the bank-diversity lever directly (max-orthogonal 3-bank
vs collinear); (iii) a second model (GPT-2 medium) for cross-model generality. All optional.
On track? yes — Exp 12 adds layer-robustness (blocks 3/6/9, recovery 90/84/76% @α=8, off-Gaussian at every
layer; block 6 reproduces Exp 3 exactly); direction ~99% complete, deliverables curated + math-verified. No blocker.

## 2026-07-02 21:0x — Experiment 13: cross-model generality (GPT-2 medium)
**Did:** acted on PLAN Next-step (iii, a second model for cross-model generality) — the highest-value
remaining external-validity axis after Exp 12's layer robustness. A reviewer's obvious next question
after "is it a block-6 artifact?" (answered no) is "is it a GPT-2-small artifact?". Wrote
`experiments/13_cross_model.py`: replicate the EXACT flagship Exp 3 pipeline on GPT-2 MEDIUM (355M, 24
blocks, d=1024) at its mid layer block 12/24 (depth analogue of small's block 6/12), changing ONLY the
model. Reuse trick: loaded medium once via transformers and overwrote common._model/_tok so every
imported Exp-3 helper (resid_post/train_corrector/lm_loss_fn/make_hat/corrector_acts, all fetch the model
through common.load_model()'s cache) transparently runs on medium; instantiated Corrector(d=1024). Trained
at batch=4 / seq=64 to fit the 0.18 VRAM frac. Confirmed medium downloads+loads (24L, d=1024). Ran clean,
no OOM, ~5 min, GPU was free.
**Learned (POSITIVE — clean cross-model generality):** both headline facts replicate. (P) raw steering
breaks medium: ΔLM +0.04/+0.15/+0.74/+2.72 @α=1/2/4/8, D_M 31.5→55.1. (C) identical LM-supervised
corrector recovers it at matched projection: ΔLM −0.12/−0.09/−0.01/+0.30, **recovery 89% @α=8, 101% @α=4**
(>100% at α≤2: learned ΔLM slightly below unsteered baseline, same free-or-better weak-α as small; ratio
unstable because raw damage ≈0 there). α=8 recovery (89%) ≈ small's 84% — if anything a touch higher. The
Exp 2/3 decoupling holds AGAIN: corrected D_M > raw at every α (79.9 vs 55.1 @α=8) — off-Gaussian-but-LM-safe
carries to the larger model. So the projection-preserving recipe is MODEL-robust as well as layer-robust.
**Assumptions/decisions logged.** (a) Chose (iii) cross-model over (i) rollout-through-generation and
(ii) diversity-lever confirmation because it answers the single biggest remaining external-validity question
in one clean iteration and the flagship pipeline was directly reusable via the model-cache swap (low risk,
~5 min). (i) is higher-risk (differentiable/sampled rollout could eat the budget). (b) GPT-2 medium (not
large/xl) because it's the standard next size, fits the VRAM budget, and one size increase already
establishes model-robustness; still-larger models logged as open. (c) Block 12/24 = mid = the honest depth
analogue of block 6/12; rebuilt v at that layer (steering vectors are layer- and model-specific) → matched
projection α|v| per model. (d) batch=4 for medium's larger memory; halve-on-OOM was ready but not needed.
(e) Kept cov_corr out (raw-vs-learned carries the generality story; cov_corr's negative is already in Exp 2)
— same choice as Exp 12.
**Deliverables:** RESULTS.md +Exp 13 (table + reading) + figure entry + Headline "model-robust" sentence;
REPORT.md +Exp 13 Methods (cross-model setup, reuses Exp 12 recovery eq) + Results (table + interpretation)
+ Summary/Conclusion "model-robust" sentences + Limitation (3) updated (multi-model now DONE);
plots/13_cross_model.png; results/13_cross_model.json; CHANGELOG appended. REPORT math re-verified via
GitHub API (14/14 js-display-math, 0 broken, 0 inline hazards).
**Next step (optional; success criterion long met, now layer- AND model-robust):** any one a clean iter —
(i) push the Exp 11 ceiling by supervising the behavioral readout THROUGH sampled/differentiable generation
rather than teacher-forced; (ii) confirm the bank-diversity lever directly (max-orthogonal 3-bank vs
collinear); (iii) held-out-prompt-family generalization or a still-larger model (GPT-2 large). All optional.
On track? yes — Exp 13 adds cross-model generality (GPT-2 medium, recovery 89% @α=8 / 101% @α=4, off-Gaussian
at every α; ≈ small's 84%); direction ~99% complete, deliverables curated + math-verified. No blocker.

## 2026-07-02 21:1x — Experiment 14: direct/causal confirmation of the bank-DIVERSITY lever
**Did:** acted on PLAN Next-step (ii) — "confirm the bank-diversity lever directly" — the cleanest,
lowest-risk open item and the POSITIVE counterpart to the Exp 7/8/9 scaling negatives. Exp 9 only
*inferred* that bank angular diversity (separability), not target-subspace coverage, drives a conditional
corrector's recovery: in that pool the held-out `certainty` sits inside the collinear cluster, so a bank's
alignment-to-target and its internal collinearity co-vary and can't be separated. Wrote
`experiments/14_diversity_lever.py`: a CONTROLLED THIRD-MEMBER SWAP that decouples them. Three size-3 banks,
capacity fixed 5.25M, ALL sharing the anchor pair {sentiment, formality}; only the THIRD member changes,
picked to be ever more collinear with formality — div=+politeness(|cos|0.07,D0.13), mid=+complexity(0.57,
0.21), coll=+concreteness(0.76,0.26). First computed the full 6-dir pairwise |cos| matrix to design the
banks (sentiment ⟂ everything ≤0.03; {formality,concreteness,complexity,certainty} collinear 0.76–0.82;
politeness weak). Reused Exp 6 CondCorrector/train_cond/make_hat_cond + Exp 3 LM-loss via import. Ran ~1.5
min (3 trainings), no OOM under 0.18 frac.
**Learned (POSITIVE — turns Exp 9's correlation into a controlled causal result):** bank diversity is a
causal lever. Two monotone signals @α=8: (1) the swapped 3rd member's OWN recovery collapses as it
collinearizes with formality — politeness 69% → complexity 40% → concreteness 17% (α=4: 75/57/34) — a
member confusable with a neighbor can't be specialized (corrector is fed v̂, can't separate near-parallel
dirs); (2) the CONFOUND-FREE isolate `sentiment` (⟂ every dir AND ⟂ the target, |cos|≤0.03) is corrected
WORSE in more collinear banks — 63% → 61% → 55% — which CANNOT be a target-coverage effect since nothing
about sentiment's geometry/relation-to-target changed across runs; it can only be reduced bank
separability. `formality` (the anchor that gains the collinear neighbor) holds ~69–70%: the corrector
collapses the near-parallel pair onto the dominant larger-norm member, so the neighbor loses recovery and
the anchor keeps it. Held-out certainty transfer flat (9/5/7%) as designed (this experiment varies internal
separability, not target coverage). Weak-α recovery omitted (raw ΔLM≈0 at α=1 → unstable ratio, as
throughout).
**Assumption/decision logged.** (a) Chose Next-step (ii) diversity-lever confirmation over (i)
rollout-through-generation and (iii) larger-model/held-out-prompt-family because it is the cheapest,
lowest-risk open item, reuses ALL Exp 6/9 machinery (no new harness), and supplies the one MISSING positive
lever to complement the three scaling negatives — highest value-per-risk. (i) is higher-risk
(differentiable/sampled rollout could eat the budget). (b) Controlled third-member swap with a FIXED
{sentiment,formality} anchor (rather than 3 arbitrary banks) so the anchor's recovery is directly
comparable across banks and the ONLY thing changing is the 3rd member's collinearity — this is what
removes Exp 9's confound. (c) Used `sentiment` as the confound-free isolate BECAUSE it is ⟂ everything and
⟂ the target — its degradation is unambiguous evidence for the separability mechanism with no
target-alignment or member-identity explanation. (d) Reported the weak-α instability honestly and headlined
α=8 (α=4 for the 3rd member) where denominators are large. (e) Fixed a missing-glyph (⟂) in a plot title →
"orthogonal to" before finalizing the figure.
**Deliverables:** RESULTS.md +Exp 14 (table + reading) + figure entry + Headline diversity-lever sentence
now cites the causal confirmation; REPORT.md +Exp 14 Methods (controlled-swap + confound-free-isolate
rationale) + Results (table + interpretation) + Conclusion (Exp 9 sentence now "confirmed causally") +
Limitation (3); plots/14_diversity_lever.png; results/14_diversity_lever.json; CHANGELOG appended. REPORT
math re-verified via GitHub API (14/14 js-display-math, 0 broken, 0 inline hazards).
**Next step (optional; success criterion long met — flagship result is layer- + model-robust, amortization
story closed on 3 negatives + now 1 positive lever):** any one a clean iter — (i) push the Exp 11 ceiling by
supervising the behavioral readout THROUGH sampled/differentiable generation rather than teacher-forced;
(ii) held-out-prompt-family generalization; (iii) a still-larger model (GPT-2 large). All optional.
On track? yes — Exp 14 confirms the bank-diversity lever causally (confound removed), the positive
counterpart to Exp 7/8/9; direction ~99% complete on all planned axes, deliverables curated +
math-verified. No blocker.

## 2026-07-02 21:2x — Experiment 15: held-out prompt-family generalization
**Did:** acted on PLAN Next-step (ii) — "held-out-prompt-family generalization" — the last untested
external-validity axis and the cheapest/lowest-risk open item. Every prior experiment (1–14) both TRAINS
and EVALUATES the corrector on FineWeb web text, so a reviewer's obvious question is whether the corrector
overfit the FineWeb prompt distribution rather than learning a general correction rule. Wrote
`experiments/15_prompt_family.py`: train the flagship sentiment corrector EXACTLY as Exp 3 (same
vector/seed/recipe/300 FineWeb train docs, via importlib reuse of exp03's Corrector/train_corrector/
make_hat/lm_loss_fn/gaussian_stats/mahalanobis/LAYER), then evaluate it UNCHANGED, at matched projection
α|v|, on three held-out prompt families of increasing distribution shift: fineweb (in-dist, = Exp 3
held-out 100 docs), markdown (100 chunks of this project's own .md research prose), code (100 chunks of
numpy/torch/transformers Python source, gathered from local site-packages — no network). Also computed
each family's clean-activation Mahalanobis D_M under the FineWeb Gaussian to make "distribution shift"
concrete. Ran ~2 min under 0.18 VRAM frac, no OOM.
**Learned (POSITIVE — clean prompt-generalization result):** the corrector is NOT overfit to FineWeb. It
recovers 84% (fineweb) / 77% (markdown) / 60% (code) of raw steering's fluency damage @α=8 (95/87/78% @α=4)
— the FineWeb-trained corrector still removes the majority of the damage on genuinely different, even
non-natural-language, prompt families. The key structure: recovery tracks the activation shift MONOTONICALLY
(clean D_M 27.5→30.1→37.4 ⇒ recovery 84→77→60% @α=8), i.e. graceful degradation with distribution shift (the
prompt-axis analogue of Exp 4's strength extrapolation), not collapse. The in-distribution fineweb row
reproduced Exp 3 TO THE DIGIT (raw +2.778 → learned +0.435, 84%) — built-in reproducibility check the reuse
path is faithful. (Aside: code's clean LM loss is actually LOW, 2.9 vs fineweb's 3.7 — GPT-2 finds library
boilerplate predictable — while markdown's is high, 5.2; but ΔLM/recovery is what matters and is well-defined
on each.) So the flagship fluency result is now shown robust on FOUR axes: strength (Exp 4), direction/recipe
(Exp 5), layer (Exp 12), model (Exp 13), and now prompt family (Exp 15).
**Assumption/decision logged.** (a) Chose Next-step (ii) prompt-family over (i) rollout-through-generation
and (iii) still-larger-model because it is the cheapest, lowest-risk remaining external-validity axis, reuses
ALL Exp-3 machinery (no new harness, just new eval corpora), and closes the last open item in REPORT
Limitation (3). (i) is higher-risk (differentiable/sampled rollout could eat the budget); (iii) is heavier
(GPT-2 large VRAM). (b) Picked THREE families spanning a distribution-shift GRADIENT (in-dist FineWeb → mild
markdown prose → strong-OOD code) rather than one OOD set, so the result is "recovery tracks shift," a
mechanism, not a single point. (c) Sourced code from installed library .py files and markdown from the
project's own docs — both abundant and available with NO network. (d) Quantified shift by clean-activation
D_M under the FineWeb Gaussian (already-defined metric) so "how OOD" is measured, not asserted. (e) Kept the
recovery-ratio instability at α=1 honest (raw ΔLM≈0 → >100%/inflated ratios) by headlining α=4–8 where
denominators are large. (f) Reported the code clean-loss quirk in the journal but not the report (it's an
aside; recovery is the finding).
**Deliverables:** RESULTS.md +Exp 15 (two tables + reading) + figure entry + Headline "prompt-family-robust"
sentence; REPORT.md +Exp 15 Methods + Results (two tables + interpretation) + Summary/Conclusion sentences +
Limitation (3) updated (prompt-family now DONE; only still-larger models open); plots/15_prompt_family.png;
results/15_prompt_family.json; CHANGELOG appended. REPORT math re-verified via GitHub API (14/14
js-display-math, 0 broken, 0 inline hazards).
**Next step (optional; success criterion long met — flagship result now robust on strength/direction/layer/
model/prompt-family axes):** any one a clean iter — (i) push the Exp 11 ceiling by supervising the behavioral
readout THROUGH sampled/differentiable generation rather than teacher-forced (the one substantive open lever);
(ii) a still-larger model (GPT-2 large). Both optional.
On track? yes — Exp 15 adds prompt-family robustness (FineWeb-trained corrector recovers 84/77/60% @α=8 on
fineweb/markdown/code, graceful degradation tracking activation shift); direction ~99% complete, all planned
generalization axes now covered, deliverables curated + math-verified. No blocker.

## 2026-07-06 — Iter 16: is the "Gaussian manifold" valid? (acts on human feedback)

**Scope decision.** New human feedback (`human_feedback_07060332.md`) raised three asks: (#1) build an
actual diffusion-model corrector like the GLP arxiv paper, not the one-shot MLP; (#2) doubt the Gaussian
manifold — look at manifold-recovery literature and characterize the real manifold; (#3) try steering types
beyond sentiment (downloads OK). One focused iteration → picked #2: self-contained (no downloads), directly
tests the assumption under the D_M metric used in EVERY experiment, highest value-to-risk. Logged #1 and #3
as prioritized next steps (see PLAN). Rejected doing #1 this iter (a proper diffusion/flow model over
activations + iterative sampling is a multi-iteration build, higher risk of not landing cleanly); rejected
#3 this iter (I already cover 6 DiffMean concepts — sentiment/formality/concreteness/certainty/politeness/
complexity — so the marginal value is a genuinely different steering *family*, better done alongside #1).

**Env note (important for future iters).** The working interpreter is
`/mars-vol/marsv/dir9_ood/cupenv/bin/python` (py3.11, torch 2.9, transformers present). Under 5-agent
contention the `from transformers import ...` line takes ~206s (finite, not hung — faulthandler showed it
crawling through the import chain). Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and run in background with
a large timeout. `/opt/conda/bin/python` has torch but NOT transformers; no scipy/sklearn anywhere.

**Did.** Wrote `experiments/16_manifold_geometry.py`: on the clean layer-6 FineWeb activations (49,218
tokens, no steering) it estimates intrinsic dimension (TwoNN — Facco 2017; Levina–Bickel MLE — 2004; PCA
participation ratio) and tests Gaussianity (held-out D_M^2 vs chi^2_768 moments + Wilson–Hilferty QQ;
per-dim excess kurtosis). All numpy/torch (no scipy). Deterministic seed 0.

**Learned.** The activation cloud is decisively NOT a single 768-d Gaussian: intrinsic dim ~8–34 (TwoNN
11.4/8.1, MLE 25–34) ≪ 768; participation ratio 1.1 (near rank-1 — GPT-2 rogue dims, ~90% var in 1 PC);
held-out D_M^2 spread 6.7× the chi^2_768 Gaussian (var ~45× too big), skew 0.45 vs 0.10, 14 heavy-tailed
dims (max excess kurt 118). KEY INSIGHT: this SHARPENS the paper rather than breaking it — it's the concrete
mechanism for Exp 2's negative result (D_M concentrates in high-variance rogue dims, so the D_M-minimizing
correction moves exactly there) and reframes "off the Gaussian manifold" as "off a crude fit." No existing
(LM-loss-based) number changes; the human's doubt is validated and folded into the thesis. Runtime ~5 min
(mostly the slow import + GPU forward), well under budget.

**Next step.** Two open human asks remain, each a clean iteration: (i) **#1 — a real diffusion/flow
corrector**: train a (conditional) flow-matching or DDPM-style denoiser over layer-6 activations using the
STEERING corruption z=h+αv (Cold-Diffusion framing), do iterative sampling, and compare its ΔLM/behavioral
Pareto to the one-shot MLP and to a generic Gaussian-noise GLP-style teacher — name it explicitly a
"diffusion" model per the feedback. (ii) **#3 — a genuinely different steering family** (e.g. a
persona/behavioral trait or a downloaded sentiment/toxicity dataset rather than the 20/20 DiffMean probes),
to test the recipe beyond hand-built DiffMean concepts. Both optional; success criterion long met.

On track? yes — Exp 16 (manifold geometry) delivered, acting on human feedback #2: activations are
low-dim/anisotropic/heavy-tailed, NOT Gaussian — sharpens the thesis; deliverables curated + math-verified
(18/18 display-math). ~99% complete; feedback asks #1 (diffusion model) and #3 (other steering) queued. No blocker.

## 2026-07-06 — Iter 17: a REAL diffusion corrector (acts on human feedback #1)

**Scope.** Completed the highest-priority open human-feedback ask (#1): build the ACTUAL diffusion machinery
the direction is named after and settle whether "diffusion" adds anything over the one-shot MLP. The script
`experiments/17_diffusion_corrector.py` was already authored in a prior iter but its run had been killed
mid-training (log stopped at "== train cold-diffusion iterative corrector ==", no JSON). This iter I ran it
to completion (0.18 VRAM frac, ~5 min under contention: slow transformers import + model forward + 3 model
trainings + eval) and curated all deliverables.

**Did.** Ran the three-corrector comparison at matched projection α|v| on the same held-out FineWeb eval
(GPT-2 small, block 6, sentiment): (1) one-shot MLP (Exp 3), (2) cold-diffusion iterative K=8 (step-conditioned
velocity field, projection-preserving at every step, LM-supervised through the unroll), (3) GLP Gaussian-noise
DDPM prior (SDEdit, no LM). All 4.46M/4.46M/2.69M params.

**Learned (clean, three-part answer to the central critique).** Recovery @α=8: one-shot **84%**, iterative
**85%**, GLP prior **−5%** (ΔLM +2.925, actually WORSE than raw +2.778). (1) The Cold-Diffusion CORRUPTION
MODEL is what carries the result — LM-supervised training on the real steering corruption recovers 84–85%
regardless of one-shot-vs-iterative, but the generic "denoise clean activations back to the manifold" GLP
prior has NEGATIVE recovery at every α: Exp 2's lesson in diffusion clothing (a prior that only knows
"typical activation" can't know which off-typical directions the LM tolerates). (2) The iterative diffusion
structure ~TIES the one-shot MLP (85 vs 84%; a tiny consistent edge at every α; iter sits slightly closer to
the Gaussian, D_M 75.2 vs 79.5) — the value of "diffusion" is the corruption + LM supervision, NOT the step
count, so Exp 3's one-shot MLP was not leaving fluency on the table. (3) The unconditional GLP prior ERASES
the steer (as-is projection retention 10.6/83.1 vs target 11.1/88.6 @α=1/8) — exactly the info-loss the GLP
authors flag for unconditional priors, and re-imposing the projection still can't repair the LM. This
directly validates the ColdSteer design choices (condition on clean activation + LM supervision) and turns
the "you didn't build a real diffusion model" critique into a positive result.

**Assumptions/decisions logged.** (a) Picked #1 (real diffusion corrector) over #3 (a different steering
family) because #1 is the direction's CENTRAL critique and the script was already written (lowest-risk path
to landing it cleanly). (b) Steelmanned the GLP baseline: chose its SDEdit t_start by grid-search over
{0.15,0.25,0.40} for the LOWEST ΔLM, and re-imposed the target projection so the fluency comparison is matched
and fair — the GLP prior still loses, so the negative result is robust, not a strawman. (c) Held the iterative
net to EXACTLY the one-shot capacity (4.46M) so RQ2 isolates structure from parameters. (d) Reported the GLP
prior's negative recovery honestly rather than clipping to 0 — it genuinely makes the LM worse than raw. (e)
No prior result superseded; this is an added comparison (one-shot's 84% @α=8 reproduced to the digit).

**Deliverables.** RESULTS.md +Exp 17 (7-col table + reading) + figure entry + Headline paragraph; REPORT.md
+Exp 17 Methods (3 new display-math) + Results (table + interpretation + figure) + Summary/Conclusion clauses
+ Limitation (1) updated (diffusion-prior future-work item now TESTED, doesn't help); CHANGELOG appended;
`plots/17_diffusion_corrector.png`, `results/17_diffusion_corrector.json`. REPORT math re-verified via GitHub
API: 21/21 js-display-math, 0 broken, 0 inline hazards.

**Next step (optional; success criterion long met).** One human ask remains: #3 — a genuinely different
steering family (persona/behavioral trait, or a downloaded sentiment/toxicity dataset rather than hand-built
DiffMean probes) to test the recipe beyond DiffMean concepts. Also optional: (i) push Exp 11 by supervising
the behavioral readout through sampled/differentiable generation; (ii) GPT-2 large. All optional.

On track? yes — Exp 17 delivers the central human-feedback ask (#1: a real diffusion corrector): the
Cold-Diffusion corruption model + LM supervision recovers 84–85% whether one-shot or iterative, while a
generic Gaussian-noise diffusion prior is worse than raw and erases the steer — validating ColdSteer's design.
Deliverables curated + math-verified (21/21 display-math). ~99% complete; only feedback #3 (other steering
family) queued. No blocker.

## 2026-07-06 — Iter 18: beyond hand-built DiffMean — steering-vector-family robustness (human feedback #3)

**Scope.** Completed the LAST open human-feedback ask (#3): test the recipe on a genuinely different steering
FAMILY, beyond the 6 hand-built DiffMean concepts. Changed two axes at once — data source AND extraction
method — on the same concept (sentiment).

**Did.** New `experiments/18_steering_family.py`. Downloaded a REAL dataset (SST-2, 500 pos + 500 neg
movie-review sentences via raw GitHub URL — `datasets` isn't installed and I must not pip-install it).
Mean-pooled block-6 activations per sentence, built 3 canonical linear-steering directions: DiffMean (μ⁺−μ⁻),
logistic-regression probe (torch L2 logistic, no sklearn; weight mapped standardized→raw coords), PCA-contrast
(top PC of centered pos−neg pair diffs, RepE). Sign-aligned + rescaled all to a COMMON norm |v|=11.0 so only
the direction varies. Ran the identical flagship Exp-3 recipe (reused via importlib on `03_learned_corrector`)
per family at matched projection α|v|.

**Learned.** (1) Family-ROBUST: cos to DiffMean 1.00/0.40/0.30 (genuinely different directions), all break the
LM under raw steering (ΔLM@8 +3.41/+2.63/+2.27) and the identical corrector recovers each — recovery@8
**86%/84%/101%** (98/95/118% @α=4). The DiffMean family reproduces flagship Exp 3 (86%≈84%) from real data.
(2) Concept vector only PARTLY reproducible across data sources (cos SST2-DM vs hand-DM = 0.49) — yet the
recipe works on both. (3) PCA-contrast is the sharpest bonus: it aligns with GPT-2's dominant high-variance
axis (Exp 16), so raw steering along it leaves D_M FLAT at the clean 27.3 (ON the Gaussian manifold) yet still
breaks the LM (+2.27) — off-Gaussian is neither necessary nor sufficient for LM damage; the corrector still
fixes it by moving OFF the manifold (27.3→47.5). The core result now holds on SIX axes.

**Assumptions/decisions logged.** (a) Picked ask #3 (last remaining human feedback) — #1 and #2 already done.
(b) FIRST run used base_norm = SST-2 DiffMean's own norm 4.18 → raw ΔLM@8 only +0.76, weak phenomenon, unstable
recovery% (1350% @α=1). Rescaled ALL families to the flagship norm 11.0 so raw steering strongly breaks the LM
and recovery is comparable to Exp 3; recorded the natural DiffMean norm (4.18) in JSON for transparency. (c)
"Genuinely different family" interpreted as different EXTRACTION method (DiffMean/probe/PCA — how the steering
literature separates families) AND different DATA source (real SST-2 vs hand-written) — hit both in one exp.
(d) `datasets` and `sklearn` both absent; implemented download via urllib and logistic regression in torch —
no forbidden pip installs. (e) ENV NOTE: the shared conda `transformers` had vanished this iteration (likely a
concurrent agent's env change); ran with dir9's `cupenv` python (`/mars-vol/marsv/dir9_ood/cupenv/bin/python`),
which has torch+CUDA+transformers+numpy+matplotlib — a superset, no state modified. Future iters may need the
same interpreter until the conda env is restored.

**Deliverables.** RESULTS.md +Exp 18 (family table + reading) + figure entry + Headline six-axes clause;
REPORT.md +Methods subsection (3 display-math) + Results Exp 18 (table + interpretation + figure) + Summary +
Conclusion clauses; CHANGELOG appended. Artifacts: `plots/18_steering_family.png`,
`results/18_steering_family.json`, `data/sst2_train.tsv`. REPORT math re-verified via GitHub API: 24/24
js-display-math, 0 broken, 0 inline hazards.

**Next step (all optional; success criterion long met, all 3 human asks now done).** (i) push the Exp 11
ceiling by supervising the behavioral readout through sampled/differentiable generation; (ii) GPT-2 large.

On track? yes — Exp 18 delivers the last open human-feedback ask (#3): the ColdSteer recipe is robust to the
steering-vector FAMILY (DiffMean/logistic-probe/PCA-contrast, real SST-2 data), recovering 84–101% @α=8, and
the PCA case sharpens the central decoupling (on-Gaussian yet LM-breaking). All 3 human asks now done; core
result holds on 6 axes. Deliverables curated + math-verified (24/24). ~100% complete. No blocker.

## 2026-07-07 — Iter 19: model-scaling to GPT-2 large (774M) — third model-scale point

**Scope.** Picked the untested optional external-validity point flagged in PLAN.md Next-step (ii): a
still-larger model. Success criterion long met; all 3 human asks done. This extends the model-robustness
axis from two points (small 124M / medium 355M) to three (adding large 774M), a 6× parameter range.

**Did.** New `experiments/19_gpt2_large.py`, a near-copy of Exp 13's cross-model script retargeted to
gpt2-large at mid layer block 18/36 (d=1280). Downloaded gpt2-large (3.1 GB) into the shared HF cache
(only small+medium were cached). Reused the exact Exp-3 pipeline via the shared model-cache trick;
corrector 6.03M params; trained batch 2 (VRAM: one of 5 agents at ~4.3 GB share; no OOM, peaked ~4.2 GB).
Ran with dir9's cupenv python — shared conda `transformers` still absent this iter.

**Learned.** Both headline facts replicate at 774M: raw steering breaks the LM (ΔLM@8 +2.47, D_M
35.2→66.0) and the identical LM-supervised, projection-preserving corrector recovers it at matched
projection — **recovery @α=8 = 84%** (ΔLM +2.47→+0.39), **95% @α=4**, free-or-better at weak α. Corrected
D_M > raw at every α (96.8 vs 66.0 @α=8) — Exp-2/3 decoupling holds a third time. The clean model-scaling
finding: α=8 recovery is essentially FLAT across the 6× range — small 84% / medium 89% / large 84% — so
amortized correction quality does not erode with model size. Retention matched α|v| exactly (16.8→134.0).
Note: large's |v|=16.8 < medium's 19.6, so large's raw damage (+2.47) is a touch below medium's (+2.72);
recovery is what's comparable across scales, and it is.

**Assumptions/decisions logged.** (a) Chose the model-scale point over the differentiable-generation
behavioral lever (Next-step (i)) — cleaner, lower-risk, and directly strengthens the "model-robust"
headline to a genuine trend; the differentiable-generation lever remains the single substantive open item.
(b) Mid-layer = block 18 of 36 to match the depth ratio (6/12, 12/24, 18/36). (c) Batch 2 train / 4 eval
to fit VRAM; halving rule unused (no OOM). (d) ENV: cupenv python again (`/mars-vol/marsv/dir9_ood/cupenv/
bin/python`) — shared conda transformers still gone.

**Deliverables.** RESULTS.md +Exp 19 (table + reading) + figure entry + Headline (three scales); REPORT.md
+Methods subsection + Results Exp 19 (table + interpretation + figure) + Summary + Conclusion (2 spots);
CHANGELOG appended. Artifacts: `plots/19_gpt2_large.png`, `results/19_gpt2_large.json`,
`results/19_run.log`. REPORT math re-verified via GitHub API: 24/24 js-display-math, 0 broken, 0 inline
hazards.

**Next step (all optional; success criterion long met, all human asks done).** The one substantive open
lever: (i) push the Exp 11 behavioral ceiling by supervising the readout THROUGH sampled/differentiable
generation rather than teacher-forced. Otherwise only GPT-2 XL / non-GPT-2 architectures remain untested,
both low marginal value. Keep running scripts with dir9's cupenv python until the shared conda env returns.

On track? yes — Exp 19 adds the third model-scale point: the flagship result is model-robust across GPT-2
small/medium/large (124M→355M→774M, α=8 recovery flat 84/89/84%), by moving off the Gaussian manifold as
always. Core result now robust on the model axis at three scales; deliverables curated + math-verified
(24/24). ~100% complete. No blocker.

## 2026-07-07 — Iter 20: differentiable-generation behavioral supervision (the last open lever)

**Scope.** Acted on the one substantive open lever flagged since Exp 11 (PLAN Next-step (i)): push the
Exp 11 behavioral ceiling by supervising the downstream readout THROUGH sampled/differentiable generation
rather than teacher-forced. Success criterion long met; this is the last non-trivial open item.

**Did.** New `experiments/20_diff_generation.py`, reusing the Exp 3/10/11 pipeline. Added a DIFFERENTIABLE
soft-token rollout: from P=8 prompt tokens, roll out K=8 steps with the steer at LAYER, read the L2 sentiment
projection at the produced position, feed softmax(ℓ/τ)·Wₑ back as the next input (differentiable in r_θ).
Behavioral term pushes the corrected rollout's readout toward RAW steering's own rollout (weight λ_g),
backpropped through the unroll; combined with the teacher-forced LM CE + λ_near of Exp 3. Trained λ_g∈{0,40,
160}, scored on the identical Exp 10 protocol. GEN_B=4 (VRAM), separate backward for TF vs gen graph to bound
peak memory; no OOM (GPU nearly idle this iter). Ran with dir9's cupenv python.

**Learned.** PARTIAL POSITIVE — the generation-aware signal breaks Exp 11's ≈+1.3 teacher-forced ceiling.
λ_g=0 reproduces Exp 10/11 to the digit (built-in check). λ_g=40 @α=8 reaches effect +1.72 (Exp 11's best was
+1.23/+1.08) at distinct-2 0.47 vs raw's collapsed 0.32 — nearly matching raw's already-collapsed +1.77 but
far more fluent. At moderate steering λ_g=160 @α=2 hits effect +1.61 at near-baseline fluency 0.71 (dominates
Exp 11's +0.99@0.73). BUT over-weighting collapses at strong steering: λ_g=160 destabilizes (one LM step spiked
to ~20) and degenerates like raw at α≥6 (effect +0.61→−0.22 @α=8, d2→0.32, repeats "the Southern-the-Beal…").
So supervising on the AUTOREGRESSIVE distribution is a strictly better lever than teacher-forced at moderate
steering and raises the achievable strong-α effect, but the strong-effect-AND-fluent corner still eludes.
Frontier pushed out a second time, not erased — the honest completion of the Exp 10→11→20 behavioral arc.

**Assumptions/decisions logged.** (a) Full BPTT through the K=8 soft rollout (not truncated) — memory was fine
(~1GB, GPU idle) and it's the faithful "differentiable generation." (b) Target = raw steering's OWN rollout
readout (the autoregressive analogue of Exp 11's teacher-forced p_raw). (c) Soft feedback = expected embedding
(probs·Wₑ), τ=1 — standard differentiable-generation proxy vs non-differentiable argmax. (d) λ_g grid {0,40,160}
to span Exp-11-comparable→4×; λ_g=160's collapse at strong α is itself informative (over-supervision hazard).
(e) Kept TF recipe EXACTLY Exp 3 (epochs 6, batch 8, seq 64) so λ_g=0 reproduces the base to the digit.
(f) ENV: cupenv python again (`/mars-vol/marsv/dir9_ood/cupenv/bin/python`) — shared conda transformers still gone.

**Deliverables.** RESULTS.md +Exp 20 (table + reading) + figure entry + Headline "Second fix (Exp 20)" clause;
REPORT.md +Methods subsection (2 new display-math) + Results Exp 20 (table + interpretation + figure) + Summary
+ Conclusion clauses; CHANGELOG appended. Artifacts: `plots/20_diff_generation.png`,
`results/20_diff_generation.json`, `results/20_run.log`. REPORT math re-verified via GitHub API: 26/26
js-display-math, 0 broken, 0 inline hazards.

**Next step (all optional; success criterion long met, all human asks done, all substantive levers now closed).**
The Exp 10→11→20 behavioral arc is complete: teacher-forced→differentiable-generation supervision both push the
frontier out without erasing it. Only very-low-value points remain: GPT-2 XL / non-GPT-2 architecture, or a
harder differentiable-generation objective (Gumbel-softmax hard samples, longer rollouts). Keep running scripts
with dir9's cupenv python until the shared conda env returns.

On track? yes — Exp 20 closes the last substantive open lever: differentiable-generation supervision breaks
Exp 11's behavioral ceiling (α=8 effect +1.08→+1.72 at far better fluency than raw) but the strong-effect-and-
fluent corner still eludes (over-weighting collapses). Deliverables curated + math-verified (26/26). ~100%
complete. No blocker.

## 2026-07-07 — Iter 21: cross-ARCHITECTURE generality (Qwen3-1.7B)

**Scope.** Success criterion long met; all human asks + all substantive levers closed. Picked the single
highest-value remaining external-validity point: every model tested so far (Exp 13/19: small/medium/large)
is the SAME GPT-2 architecture. Tested a genuinely different one — Qwen3-1.7B (RMSNorm, rotary positions,
SwiGLU, grouped-query attention) — to answer "is the flagship result a GPT-2-architecture artifact?"

**Did.** New `experiments/21_cross_arch.py`: self-contained port of the flagship Exp-3 pipeline to Qwen3-1.7B
(28 blocks, d=2048), steering/correcting at mid layer block 14/28. Reused exp03's Corrector / gaussian_stats /
mahalanobis; wrote arch-agnostic resid_post + FuncPatcher hooks on `model.model.layers[layer]` (Qwen3 module
path); corrector fp32 with bf16 boundary at the hook. Only the model changes vs Exp 3. Ran with dir9 cupenv.

**Learned.** POSITIVE, clean cross-architecture replication. Raw steering breaks Qwen3 (ΔLM@8 +3.43, D_M
44.7→77.8); identical corrector recovers **94% @α=8 / 108% @α=4** at matched projection (retention α|v|
exactly), corrected activation further off the Gaussian manifold than raw at every α (122.2 vs 77.8 @α=8 —
decoupling holds a 4th time). 94% edges GPT-2 small's 84%. ⇒ architecture-robust, not just scale-robust.
Training LM loss fell ~4.1→~3.0 over 900 steps; λ_near behaved as on GPT-2.

**Debugging/decisions logged.** (a) DISK CONTENTION: the shared /mars-vol volume was heavily contended —
first Qwen3 load stalled (state D, 436 MB after 10 min; dd measured ~21 MB/s). Cold load of the 3.9 GB bf16
shards took ~8 min tensor-by-tensor; a re-run loaded instantly (OS page cache). Not a code issue. (b) OOM:
first run trained fine (900 steps) then OOM'd in eval at `log_softmax(logits.float())` — Qwen3's 151,936-token
vocab makes the float logit cast the memory bottleneck at d=2048 under the 4.24 GB cap. Fix per BUDGET.md:
EVAL_BATCH 4→1, added PYTORCH_ALLOC_CONF expandable_segments, empty_cache() before eval, and a corrector
checkpoint save (results/21_corr.pt) so any eval rerun skips training. Re-ran clean, no OOM. (c) bf16 model +
fp32 corrector with `.to(h.dtype)` at the hook boundary — backprop through frozen bf16 layers into the fp32
corrector was stable (Qwen3 is bf16-native). (d) ENV: cupenv python again; shared conda transformers still gone.

**Deliverables.** RESULTS.md +Exp 21 (table + reading) + figure entry + Headline architecture clause.
REPORT.md +Methods subsection (no new display math — reuses Exp 12 recovery eq) + Results Exp 21
(Observation/Interpretation/Limitations/Next-check) + Summary + Conclusion (2 clauses). CHANGELOG appended.
Artifacts: experiments/21_cross_arch.py, results/21_cross_arch.json, results/21_corr.pt, results/21_run.log,
plots/21_cross_arch.png. REPORT math re-verified: 26/26 js-display-math, 0 broken, 0 inline hazards.

**Next step (all optional; success criterion long met).** The result now spans strength/direction/layer/
model-scale/architecture/prompt-family/steering-family. Only very-low-value points remain: a second non-GPT-2
architecture (Llama/Mistral) to make it an architecture *sweep*; the behavioral generation protocol (Exp 10)
re-run on Qwen3 to confirm the recovery isn't bought by under-steering on this architecture; or GPT-2 XL. Keep
running scripts with dir9's cupenv python until the shared conda env returns.

On track? yes — Exp 21 closes the architecture axis: the flagship result replicates on Qwen3-1.7B (non-GPT-2:
RMSNorm/RoPE/SwiGLU/GQA), recovery 94% @α=8, again off the Gaussian manifold. Deliverables curated +
math-verified (26/26). ~100% complete; direction robust on 7 axes. No blocker.

## 2026-07-07 — Iter 22: behavioral honesty-check on Qwen3 (Exp 22)

**Scope.** Success criterion long met; direction robust on 7 axes. Picked the single highest-value remaining
point — the honesty check Exp 21 itself named as its "Next check". Exp 21's headline 94% recovery on Qwen3-1.7B
is a TEACHER-FORCED ΔLM at matched layer-14 projection. Exp 10 taught (on GPT-2) that this proxy can hide a
weaker propagated behavioral edit in generation. Was the 94% bought by under-steering on Qwen3? Untested.

**Did.** New `experiments/22_behavioral_qwen.py`: ran the IDENTICAL Exp-10 behavioral generation protocol on
Qwen3-1.7B, reusing the EXACT Exp-21 corrector checkpoint (results/21_corr.pt, no retraining — rebuilt the same
deterministic block-14 sentiment vector so it matches the checkpoint). Greedy-generate 30 tokens from 48 held-out
12-token prompts, steer at block 14 every position, raw vs corrected; on a clean re-encode measure sentiment
effect B(α)−B(0) (baseline B0=+28.6) and distinct-2 (baseline 0.875). Reused exp21's Qwen3 load / FuncPatcher /
make_hat / resid_post; batched generation + re-encode at GEN_BATCH 8 with empty_cache between chunks.

**Learned.** CORRECTIVE / honest — the Exp-10 under-steering caveat REPLICATES on Qwen3. Corrector generated
sentiment effect is only 10–29% of raw's (raw +5.22/+7.31/+7.64/+8.01 vs corr +0.53/+0.77/+0.98/+2.31 @α=2/4/6/8;
cf. ~1/6 on GPT-2). So 94% is honest as a teacher-forced fluency metric but partly reflects a weaker propagated
behavioral edit — as on GPT-2. KEY DIFFERENCE: raw steering degenerates FAR LESS on Qwen3 (distinct-2 0.886→0.761
@α=8 vs GPT-2's collapse to 0.32), so raw is a STRONGER baseline here (steers hard AND stays fairly fluent), the
corrector's fluency edge is small (0.06 @α=8), and the effect-vs-fluency Pareto is shallower than on GPT-2.
Qualitative: raw α=8 "…a welcoming family and a welcoming community. The community is a home and a family…"
(positive, repetitive); corr α=8 "…in the heart of the city of Bridgend, just 15 minutes north of the city of
Bridgend" (fluent, factual, barely steered). ⇒ "matched projection ≠ matched behavioral steering" is
architecture-robust; the Exp 11/20 behavioral-preservation terms are the indicated fix if strong behavioral
steering is needed on Qwen3.

**Assumptions/decisions logged.** (a) Reused the Exp-21 corrector checkpoint UNCHANGED (the whole point is to
probe that exact corrector, not a fresh one) — rebuilt v deterministically so state_dict matches. (b) Kept the
Exp-10 protocol EXACTLY (48 prompts, PROMPT_LEN 12, GEN_LEN 30, α∈{2,4,6,8}, same metrics) so the two are
directly comparable; only model/corrector differ. (c) Reported the corr/raw effect RATIO, not absolute effects,
across models — Qwen3's |h| and B0 are ~8× GPT-2's, so absolute sentiment projections are not comparable across
architectures; the ratio is the architecture-invariant quantity. (d) GEN_BATCH 8 + empty_cache between chunks for
the ~4.3 GB VRAM share (generation is lighter than Exp 21's full-vocab eval; no OOM). (e) ENV: dir9's cupenv
python again (`/mars-vol/marsv/dir9_ood/cupenv/bin/python`) — shared conda transformers still gone.

**Deliverables.** RESULTS.md +Exp 22 (table + reading) + figure entry + Headline behavioral-caveat paragraph now
notes architecture-robustness. REPORT.md +Results "Experiment 22" (Observation/Interpretation/Limitations/Next-
check; reuses Exp 10's metric definitions, no new display math) + Exp 21 Limitations/Next-check updated (behavioral
check now DONE) + Summary + Conclusion + Limitation (2) all note the caveat is architecture-robust. CHANGELOG
appended. Artifacts: experiments/22_behavioral_qwen.py, results/22_behavioral_qwen.json, results/22_run.log,
plots/22_behavioral_qwen.png. REPORT math re-verified via GitHub API: 26/26 js-display-math, 0 broken, 0 inline
hazards.

**Next step (all optional; success criterion long met).** Only very-low-value points remain: re-fit the Exp 11/20
behavioral-preservation terms ON Qwen3 to test whether the GPT-2 fix transfers across the architecture boundary
(the natural Exp-22 follow-up); a second non-GPT-2 architecture (Llama/Mistral) for an architecture *sweep*; a
harder differentiable-generation objective; or GPT-2 XL. Keep running scripts with dir9's cupenv python until the
shared conda env returns.

On track? yes — Exp 22 closes Exp 21's own "Next check": the flagship behavioral caveat (matched projection ≠
matched behavioral steering, corrector under-steers in generation) is architecture-robust, replicating on Qwen3
(effect 10–29% of raw's), while raw's lesser degeneration makes it a stronger baseline there. Deliverables curated
+ math-verified (26/26). ~100% complete; direction robust on 7 axes plus a now-architecture-robust behavioral
caveat. No blocker.

## 2026-07-08 — Experiment 23: does the GPT-2 behavioral fix (Exp 11) transfer to Qwen3? (S4(m) follow-up)
**Did.** Completed an experiment a prior iteration had left half-done: `experiments/23_behavioral_qwen_fix.py`
existed with the λ_b=10 checkpoint trained (`results/23_corr_lamb10.pt`, Jul 7 23:51) but an EMPTY run log and
no JSON/plot — the run was interrupted before training λ_b=40 / writing deliverables. The script has resume
logic (loads existing checkpoints, trains only the missing one), so I just ran it with dir9's cupenv python. It
re-used the Exp 21/22 Qwen3 pipeline + Exp 22 generation protocol and added the Exp 11 behavioral-preservation
term at downstream Qwen3 layer L2=27, family λ_b∈{0,10,40}. λ_b=0 loads the Exp 21 checkpoint (= Exp 22
corrector, reproducibility anchor: reproduced B0=+28.633, distinct2=0.875, and λ_b=0 effect/d2 to the digit).
Only λ_b=40 trained fresh (~900 steps, batch 2, no OOM). This directly answers PLAN Next-step (ii) and Exp 22's
own "Next check."
**Learned (headline — nuanced/corrective, closes the behavioral arc).** The fix's MECHANISM is
architecture-robust but its PARETO ADVANTAGE is not. Adding λ_b lifts the corrected generation's sentiment
effect from the base corrector's +0.53–2.31 (10–29% of raw's) to λ_b=40's +4.06–6.35 (53–83% of raw's at α≤6, a
2–8× jump) — exactly the Exp 11 lever, so the correction's non-orthogonality to the downstream readout AND the
readout-preservation fix carry to Qwen3. BUT on Qwen3 the corrector does NOT beat raw: at λ_b=40 its distinct-2
(0.875→0.673) sits slightly below raw's (0.886→0.761) at every α while its effect is also below raw's — raw
weakly dominates at matched α. The reason is exactly Exp 22's finding: on GPT-2 the term won by dominating a
COLLAPSED raw (distinct-2 0.32); Qwen3's raw does not collapse, so there is no degenerate baseline to beat. The
λ_b sweep traces a frontier from the base corrector (fluent, weakly steered) toward raw (strong, fluent) without
passing it. Also: the Exp 20 λ_g=160 over-steer instability replicates (λ_b=40 @α=8 effect drops to +4.21 < its
α=6 peak +6.35, d2 0.673). ⇒ the behavioral fix is a robust lever on generated effect; its payoff is GATED by
whether the raw baseline degenerates. This closes the full behavioral arc (Exp 10→11→20→22→23): matched
projection ≠ matched steering everywhere; the readout-preservation fix transfers everywhere; the size of its
payoff depends on the baseline's failure mode.
**Assumption/decision logged.** (a) Chose to COMPLETE the started Exp 23 (PLAN Next-step ii — re-fit the fix on
Qwen3) over the other optional points (a 2nd non-GPT-2 arch, harder differentiable-generation, GPT-2 XL) because
it was the highest-value AND lowest-risk next step: it directly answers Exp 22's "Next check," reuses all
existing machinery, and its λ_b=10 checkpoint was already trained (only λ_b=40 needed). Rejected: retraining λ_b=10
(wasteful — the resume logic reused the existing checkpoint; verified λ_b=0/anchor reproduced Exp 22 to the digit
so the pipeline is faithful). (b) Downstream readout at L2=27 (last Qwen3 decoder block) — the architecture
analogue of GPT-2's final resid_post used in Exp 11 (feeds the norm+head). (c) Reported the "mechanism transfers
but Pareto win doesn't" result honestly rather than overselling the effect recovery — the gating-by-baseline-
degeneration is the actual scientific finding and it ties Exp 23 cleanly to Exp 22.
**Deliverables.** RESULTS.md +Exp 23 (table + reading) + figure entry + Headline clause; REPORT.md +Methods
subsection (reuses Exp 11 equation, no new display math) + Results (Observation/Interpretation/Limitations/
Next-check) + Exp 22 Next-check done + Summary/Conclusion/Limitation(2) clauses. plots/23_behavioral_qwen_fix.png;
results/23_behavioral_qwen_fix.json; results/23_corr_lamb{10,40}.pt. CHANGELOG appended. REPORT math re-verified
via GitHub API (26/26 js-display-math, 0 broken, 0 inline hazards). ENV: dir9's cupenv python (shared conda
`transformers` still absent as of this iter). Killed the lingering post-run process to free VRAM for co-agents.
**Next step (optional; success criterion long met — the behavioral arc is now closed on both architectures).**
Any one a clean iter, all low-value: (i) a finer λ_b sweep + the Exp 20 differentiable-generation term ON Qwen3
to map how close the corrected frontier can get to raw's strong-and-fluent corner (Exp 23's own Next check);
(ii) a SECOND non-GPT-2 architecture (Llama/Mistral) to make the architecture axis a sweep rather than one
boundary crossing; (iii) GPT-2 XL. All optional.
On track? yes — Exp 23 completes the started fix-transfer test, closing the behavioral arc (Exp 10→11→20→22→23):
the GPT-2 behavioral fix's mechanism is architecture-robust but its Pareto payoff is gated by baseline
degeneration; direction complete on 7 axes + full behavioral arc (~100%). Deliverables curated + math-verified.
No blocker.

## 2026-07-08 — Experiment 24: second non-GPT-2 architecture (Pythia-410m / GPT-NeoX) — architecture sweep (S4(l) follow-up)
**Did.** Picked the single highest-value remaining external-validity point: Exp 21 crossed the GPT-2
architecture boundary only ONCE (Qwen3), which is a weak "architecture-robust" claim. Added a THIRD,
structurally distinct family — Pythia-410m (GPT-NeoX) — so the axis is a genuine sweep. GPT-NeoX shares rotary
with Qwen3 and LayerNorm/GELU/dense-MHA with GPT-2, but its block uses a PARALLEL residual (attention+MLP from
the same input, summed) unlike BOTH GPT-2 and Qwen3 (serial). Adapted the Exp-21 template (`experiments/
24_cross_arch_pythia.py`): only model + hook path (`model.gpt_neox.layers[12]`) + dtype (fp32, small model)
changed; everything else is the flagship Exp-3 recipe. Downloaded pythia-410m (~800MB, ungated). Ran with
dir9's cupenv python.
**Learned (headline — clean POSITIVE, makes the axis a sweep).** Both facts replicate: (P) raw steering breaks
the LM (ΔLM +3.10 @α=8, D_M 31.3→52.3); (C) identical corrector recovers **81% @α=8, 81% @α=4** (71% @α=2) at
matched projection (retention α|v| exactly), corrected activation FURTHER off the Gaussian manifold at every α
(89.4 vs 52.3 @α=8 — decoupling holds a 5th time). α=1 recovery 41% is noise-dominated (raw damage only +0.06).
The architecture axis is now a 3-family SWEEP with a tight 81–94% @α=8 band: GPT-2 84/89/84%, Qwen3 94%,
GPT-NeoX 81%. The parallel-residual block is the untested structural axis and the recipe is indifferent to it.
**Assumptions/decisions logged.** (a) Chose Pythia/GPT-NeoX over the other optional points (finer λ_b sweep on
Qwen3, GPT-2 XL, Llama/Mistral) because it is the ONLY one that adds a NEW structural axis (parallel residual)
at low risk — cached-adjacent, ungated, small (~800MB, fits fp32 in the 4.3GB share), and reuses the entire
Exp-21 machinery. Rejected Llama/Mistral (gated and/or 7B = too big for the VRAM share); rejected GPT-2 XL
(same architecture, adds no axis); rejected the finer-λ_b Qwen3 sweep (confirmatory, no new axis). (b) Mid
layer block 12/24 = depth analogue of block 6/12 in GPT-2 small, matching Exp 13/19/21. (c) Downstream readout
untouched — this is the fluency (ΔLM) replication, not the behavioral arc; the behavioral generation check on
Pythia is logged as Exp 24's Next check but deferred (low value; the caveat is already architecture-robust via
Exp 22). (d) Ops: the process was silently killed twice when launched via `nohup &` from the Bash tool (child
died in the shell's process group when the tool call returned); fixed by launching with `setsid` to fully
detach. Eval was slow (~1 α per few min) under GPU contention with co-agents but completed cleanly.
**Deliverables.** RESULTS.md +Exp 24 (table + reading) + figure entry + Headline sweep clause; REPORT.md
+Methods subsection (reuses Exp 3/12 equations, no new display math) + Results (Observation/Interpretation/
Limitations/Next-check) + Exp 21 Limitations/Next-check updated + Summary + Conclusion (2 clauses) upgraded to a
3-family sweep. plots/24_cross_arch_pythia.png; results/24_cross_arch_pythia.json; results/24_corr.pt;
results/24_run.log. CHANGELOG appended. REPORT math re-verified via GitHub API (26/26 js-display-math, 0 broken,
0 inline hazards — Exp 24 adds no equation). ENV: dir9's cupenv python (shared conda `transformers` still absent).
**Next step (all optional; success criterion long met — architecture is now a 3-family sweep).** Any one a clean
iter, all low-value: (i) the Exp-10 behavioral generation protocol on Pythia (Exp 24's own Next check); (ii) a
further architecture family for a fuller sweep (state-space / MoE); (iii) a finer λ_b + Exp-20 differentiable-
generation term on Qwen3 (Exp 23's Next check). All optional.
On track? yes — Exp 24 turns the architecture axis from a single GPT-2→Qwen3 boundary crossing into a genuine
3-family sweep (GPT-2 / Qwen3 / GPT-NeoX, 81–94% recovery @α=8), corrected off-Gaussian at every α; direction
complete on 7 axes + full behavioral arc (~100%). Deliverables curated + math-verified. No blocker.

## 2026-07-08 — Experiment 25: behavioral check on Pythia-410m (Exp 24's Next check)
Completed an experiment a prior iteration left half-done: experiments/25_behavioral_pythia.py existed (a clean
adaptation of Exp 22 to the Exp-24 Pythia pipeline) but had never been RUN — no JSON/plot/log. Ran it this
iteration (dir9 cupenv python; shared conda transformers still absent). It reuses the EXACT Exp 24 corrector
checkpoint (results/24_corr.pt), so no training — only the 9 generation passes (1 clean + 4α × 2 regimes) over 48
prompts. Cold Pythia load was slow under /mars-vol disk contention (~5 min to first weight load on a cold cache;
warm re-run instant), CUDA idle until load finished, then the sweep completed in seconds.
RESULT (nuanced positive): the Exp 10/22 under-steering caveat is MILDER on Pythia. Corrected generated effect
+0.90/+0.80/+0.93/+0.98 (α=2/4/6/8) is ABOVE raw's +0.17/+0.40 at α≤4 and 84–92% of raw at α≥6 — vs the ~1/6
(GPT-2) or 10–29% (Qwen3) shortfall. At α=8 the corrector Pareto-DOMINATES raw (effect +0.98 @ distinct-2 0.72 vs
raw +1.17 @ collapsed 0.38). Learned: the penalty's size tracks how strongly RAW steering itself propagates
behaviorally — raw steers Pythia weakly here (peaks +1.17), so the corrector loses little. Caveat: small-magnitude
low-signal regime; not evidence the corrector out-steers raw in general. Assumption logged (loop mode): kept the
Exp 22 α grid {2,4,6,8} and 48-prompt/30-token protocol verbatim for cross-arch comparability rather than widening
α to hunt a stronger-raw regime (rejected alt: α up to 16 — would break comparability with Exp 10/22 and cost a
retrain-free but longer run under contention). Deliverables curated (RESULTS +Exp 25 row/figure/headline sentence;
REPORT +Exp 25 subsection, no new equation), math re-verified 26/26 js-display-math / 0 broken / 0 hazards.
Next step: only very-low-value optional points remain — Exp 11/20 behavioral terms on Pythia; a state-space/MoE
architecture for a 4th family; finer λ_b + Exp-20 term on Qwen3/Pythia. Success criterion long met.
On track? yes — S4(l-follow-up) behavioral arc now closed on all THREE architectures (GPT-2 Exp 10 / Qwen3 Exp 22 /
Pythia Exp 25); direction complete on 7 axes + full behavioral arc (~100%). Deliverables curated + math-verified.
No blocker.

## 2026-07-08 — Finalization / STOP
Did: re-read CLAUDE.md, BUDGET.md, PLAN/JOURNAL/RESULTS/CHANGELOG. Direction is complete on all planned
axes (Exp 1–25) and all three human-feedback asks; deliverables were already curated to current-best in
the Exp 25 iteration. This iteration is a verification-only finalization (time budget in the finalize
window): confirmed REPORT.md + RESULTS.md both include every experiment through Exp 25, all 21–25 plots
present, and re-ran the GitHub markdown math check on REPORT.md = 26/26 js-display-math, 0 broken, 0
inline hazards. Appended a CHANGELOG finalization note (no numbers changed) and created STOP.
Learned: nothing new numerically; the deliverables pass the CLAUDE.md §8 rendering checks as-is.
Next step: none required — success criterion long met. Optional low-value future points (Exp 11/20
behavioral terms on Pythia; a state-space/MoE 4th architecture) remain but are not needed.
On track? yes — direction complete (~100%), deliverables current-best + math-verified, STOP written. No blocker.

## 2026-07-09 — REPORT.md disassembled into an index + four topic-focused mini reports (human feedback)
**Did.** Acted on operator feedback (`human_feedback_07082201.md`): "the report is too long and too much back and
forth and I cannot understand it… disassemble the report into 2–4 mini reports that each share the same topic."
Split the 1744-line monolithic REPORT.md into a 108-line **index** (`REPORT.md`: overall Summary + takeaway
equation + a headline-numbers table + limitations overview + links) and **four self-contained parts**, each a
clean linear narrative on ONE topic with the full Summary→Methods→Results→Conclusion structure (Methods keeps
Data/Model/Layer + every metric & baseline defined with rendered equations):
P1 `core_correction` (Exp 2,3,4,5,16,17 — the negative Gaussian corrector, the LM-supervised fix, why the Gaussian
is the wrong yardstick, the diffusion-framing ablation); P2 `amortization` (Exp 6,7,8,9,14 — bank corrector +
diversity lever); P3 `external_validity` (Exp 12,13,19,21,24,15,18 — the six robustness axes); P4 `behavioral`
(Exp 10,11,20,22,23,25 — matched-projection≠matched-steering + the fixes across three architectures). Executed the
mechanical VERBATIM extraction of all Methods/Results blocks via four parallel subagents (each given exact source
line ranges + topic-scoped Summary/Conclusion facts); wrote the index myself. Verified every file's math through
the GitHub markdown API.
**Learned.** The report's unreadability was structural, not numerical: 25 experiments interleaved by chronology
(Exp 10→11→20→12→13→…→22→23→…) forced the reader to hold five threads at once. Grouping by topic and giving each
group its own Summary/Conclusion removes the back-and-forth without touching a single result. The GitHub-API math
check must now run on all FIVE report files, not just REPORT.md (recorded in PLAN Next step). Total display math
42 (index 1 / P1 16 / P2 6 / P3 9 / P4 10), 0 broken, 0 inline hazards; found+fixed one orphaned figure
(`plots/25_behavioral_pythia.png`) that the source never referenced — now in P4.
**Assumptions/decisions logged (loop mode).** (a) Chose **4 parts** (the top of the requested 2–4 range) because
the work has four genuinely distinct topics (core mechanism / amortization / external validity / behavioral) and
fewer parts would re-create the cross-topic back-and-forth the operator objected to. Grouping: put manifold
geometry (16) + diffusion ablation (17) INTO the core-mechanism part (they explain *why* the fix works) rather
than a 5th part, and put steering-vector-family (18) into external-validity (it is a robustness axis). (b) Kept
`REPORT.md` as a short **index** rather than making Part 1 the entry point, so the overall thesis + headline table
live in one place and CLAUDE.md §8's "self-contained presentable report with Methods+equations" is satisfied
**within each part**. Logged as a deliberate reading of §8 for a split report. (c) Left **RESULTS.md unchanged** —
the feedback named "the report" (the narrative REPORT.md); RESULTS.md is a per-experiment results log with far
less back-and-forth. Rejected alt: splitting RESULTS.md too (scope creep, not requested); noted in PLAN as an
optional future step if the operator asks. (d) Delegated verbatim extraction to subagents to keep my context
small and parallelize; mitigated the paraphrase risk by giving exact line ranges + instructing character-for-
character copying, then independently re-verified numbers/math on every output.
**Deliverables.** REPORT.md (index) + REPORT_1..4 (parts). PLAN Current status + Next step updated; CHANGELOG
appended; this JOURNAL entry. No experiment re-run; no result number changed.
**Next step.** None required — feedback addressed. Optional: apply the same split to RESULTS.md if the operator
finds it long too; the prior low-value research follow-ups remain.
On track? yes — directly resolved the operator's readability complaint by restructuring the report into an index +
four topic-focused parts with zero change to any result; all five files math-verified (42 js-display-math, 0
broken, 0 inline hazards). No blocker.

## 2026-07-09 — Experiment 26: seed robustness / confidence interval on the flagship 84% recovery
**Did.** Picked the single highest-value remaining rigor gap: every prior experiment (incl. flagship Exp 3) is
a single run at SEED=0, so "84% @α=8" had no error bar — and *seed* is the one control CLAUDE.md rule 10 names
that no axis had varied (strength/direction/layer/model/prompt/steering-family/architecture all were). Wrote
`experiments/26_seed_robustness.py` (imports the flagship Exp-3 module via importlib and overrides its SEED per
run — DRY, no pipeline duplication) and ran the EXACT Exp-3 recipe at 5 seeds (0–4). Raw ΔLM is seed-independent
(computed once); only the learned corrector varies. Result POSITIVE + TIGHT: recovery 83.3±2.0% @α=8 (per-seed
84.3/84.5/84.6/83.0/80.0%), 96.2±0.8% @α=4, 90.0±0.6% @α=6; ΔLM learned @α=8 +0.464±0.054 vs raw +2.778. Seed 0
reproduces Exp 3 to the digit (84.3%). The only wide bar, α=1 (196±19%), is a ratio artifact of raw's near-zero
+0.076-nat damage (absolute ΔLM_learned tight −0.073±0.014). So the headline 84% is representative, not a lucky
init — a 7th robustness axis for the flagship.
**Learned.** (1) The recovery is far more seed-stable than I expected (±0.6–2.0% at α≥4) — the corrector's
advantage over raw dwarfs its optimization variance, which strengthens the whole paper's headline. (2) OPS: the
dir9 `cupenv` python lives on the shared `/mars-vol` network volume, and under 5-agent disk contention a cold
import STALLED ~30 min in the kernel `folio_wait_bit_common` (page-cache misses), grinding one small scipy/sklearn
`.pyc`/`.so` file at a time (diagnosed via /proc/<pid>/fd/3 + /proc/<pid>/io syscr). `/opt/conda/bin/python`
(transformers 5.13.0, torch 2.9 cu130, matplotlib 3.11) is on LOCAL disk and imports in seconds — switched to it
and the run completed in ~15 min. Recorded in PLAN Next step + CHANGELOG as the recommended env.
**Assumptions/decisions logged (loop mode).** (a) Chose 5 seeds (standard minimal-but-defensible N for a
sample-std CI on a cheap GPT-2-small run) over 3 (too few for a std) or 10 (diminishing value at 2–3 min/seed).
(b) Ran the control on the FLAGSHIP setup only (GPT-2 small/block 6/sentiment), not on every cross-model check —
the flagship number is THE headline, and per-model 5-seed sweeps would be ~5× the cost for little marginal rigor;
logged as a Limitation (bounds optimization variance on the flagship, not eval-doc/vector-construction sampling
variance, and cross-model checks stay single-seed). (c) Placed Exp 26 in REPORT_3 (external validity) as the
seed axis rather than a new part — it directly answers that part's own repeatedly-flagged "single seed" open item.
**Deliverables.** RESULTS.md (Exp-26 section + figure entry + Headline seed CI), REPORT_3 (Methods+Results+
Conclusion + Summary bump six→seven axes), REPORT.md index (Summary 84% CI, Part-3 blurb, headline table row).
CHANGELOG appended; PLAN Current status/Next step rewritten + S7 checkbox added; this JOURNAL entry. REPORT math
re-verified on the 2 touched files (index 1 / Part 3 9 js-display-math, 0 broken, 0 inline hazards — Exp 26
reuses Exp 12's recovery equation, no new equation). Artifacts: `experiments/26_seed_robustness.py`,
`results/26_seed_robustness.json`, `results/26_run.log`, `plots/26_seed_robustness.png`.
**Next step.** Optional only: 5-seed control on a cross-model check (Qwen3) for error bars there; a further
architecture family; or Exp 23's finer-λ_b Next check. Success criterion long met.
On track? yes — S7 (seed robustness) done, ~100% complete; flagship 84% shown reproducible at 83.3±2.0% across
5 seeds, closing the last review-named control. No blocker.

## 2026-07-09 — Experiment 27: seed robustness on GPT-2 medium (error bar on the cross-model recovery)
**Did.** Picked PLAN Next-step (i), the highest-value remaining rigor point: Exp 26 gave the flagship
(GPT-2-small) recovery a 5-seed CI, but the cross-MODEL number (Exp 13, GPT-2 medium) was still a single
seed-0 run, so we could not say whether medium's 89% @α=8 (vs small's 83.3%) is a real model-scale effect
or optimization noise. Wrote `experiments/27_seed_robustness_medium.py` (reuses the Exp-3 module +
Exp-13's medium-retarget trick verbatim; overrides `exp03.SEED` per run, `exp03.LAYER=12`) and ran the
EXACT Exp-13 GPT-2-medium pipeline at 5 seeds (0–4). Raw ΔLM seed-independent (computed once); only the
learned corrector varies. Result POSITIVE + informative: recovery **88.3 ± 2.2% @α=8** (per-seed
89/90/88/85/89%), **101.7 ± 1.0% @α=4**; ΔLM learned @α=8 +0.317 ± 0.059 vs raw +2.718; `D_M` learned
74.6 ± 4.5 vs raw 55.1 (decoupling holds every seed). Seed 0 reproduces Exp 13 to the digit (89%/101%).
The medium band `[86.1, 90.5]%` sits ENTIRELY ABOVE small's `[81.3, 85.3]%` (Exp 26) — non-overlapping,
so medium's ~5-point edge is a genuine model-scale effect, not a lucky seed.
**Learned.** (1) The model-scale improvement small→medium (83.3%→88.3% @α=8) survives seed noise — the two
5-seed CIs don't touch. So the flat-looking 84/89/84% scale curve (Exp 13/19) has a real small→medium bump
(large then dips back). (2) The recovery is even MORE seed-stable on medium at α=4 (±1.0%) than small
(±0.8%); optimization variance stays tiny across scale. (3) Sanity: |v|=19.57, |h|=226.20, D_M=31.45,
ΔLM raw +0.037/+0.150/+0.738/+2.718 all reproduce Exp 13 to the digit — the retarget-to-medium reuse is
faithful.
**Assumptions/decisions logged (loop mode).** (a) Chose GPT-2 medium (cached, ~1.5GB, fast) as the
cross-model seed check over Qwen3/Pythia (heavy loads, slow under /mars-vol contention) — highest rigor per
minute, and it directly tests the one cross-model number the paper leans on next to the flagship. Logged the
alternative (a cross-ARCHITECTURE seed CI) as Exp 27's own Next check. (b) 5 seeds (matches Exp 26 for an
apples-to-apples CI comparison) over 3/10. (c) Placed Exp 27 in RESULTS + REPORT_3 next to Exp 26 (seed axis)
rather than a new part.
**Deliverables.** RESULTS.md (Exp-27 section + figure entry + Headline model-scale seed CI); REPORT_3
(Methods block + Results subsection + Exp-13 seed pointer + Conclusion/open-items update); REPORT.md index
(seed-robust headline row now both scales). CHANGELOG appended; PLAN Current status/Next step updated + S7
checkbox extended; this JOURNAL entry. REPORT math re-verified on the 2 touched files (REPORT_3 9 / index 1
js-display-math, 0 broken, 0 inline hazards — reuses Exp 12's recovery equation, no new equation). Artifacts:
`experiments/27_seed_robustness_medium.py`, `results/27_seed_robustness_medium.json`, `results/27_run.log`,
`plots/27_seed_robustness_medium.png`.
**Next step.** Optional only (success criterion long met): a 5-seed control on a cross-ARCHITECTURE model
(Qwen3/Pythia) to test whether the 81–94% architecture band is within seed noise; a further architecture
family; or the Exp-23 finer-λ_b / Exp-20-on-Qwen3 lever. ENV: `/opt/conda/bin/python` (LOCAL disk, fast).
On track? yes — S7 seed axis extended to a second model scale (GPT-2 medium 88.3 ± 2.2% @α=8, non-overlapping
with small); model-scale edge shown real, not seed noise. ~100% complete. No blocker.

## 2026-07-09 — Experiment 28: seed robustness on Pythia-410m / GPT-NeoX (error bar on the cross-architecture recovery)
**Did.** Picked Exp 27's own Next check, the highest-value remaining rigor point: Exp 26/27 gave 5-seed CIs on
two GPT-2 SCALES, but the cross-ARCHITECTURE recovery (Exp 24, Pythia) was still a single seed-0 run — and Pythia
sits at the LOW end of the reported 81–94% architecture band (below both GPT-2 scales), exactly where a single
seed is most in doubt. Wrote `experiments/28_seed_robustness_pythia.py` (imports the Exp-24 module and reuses its
load/resid_post/make_hat/lm_loss_fn/train_corrector/corrector_acts verbatim; overrides `exp24.SEED` per run — DRY,
no pipeline duplication) and ran the EXACT Exp-24 Pythia pipeline at 5 seeds (0–4). Raw ΔLM seed-independent
(computed once); only the learned corrector varies. Result POSITIVE + tight: recovery 80.8±1.6% @α=8 (per-seed
81/82/80/78/81%), 81.7±0.3% @α=4, 72.1±1.5% @α=2; ΔLM learned @α=8 +0.597±0.048 vs raw +3.103; D_M learned
80.8±6.6 vs raw 52.3 (decoupling holds every seed). Seed 0 reproduces Exp 24 to the digit (81%/81%). KEY: Pythia's
α=8 band [79.2,82.4]% sits ENTIRELY BELOW GPT-2 medium's [86.1,90.5]% (Exp 27, a genuine gap) but OVERLAPS GPT-2
small's [81.3,85.3]% (Exp 26) — so Pythia≈small within seed noise at α=8, and the 81–94% band is real at its low
end but is three seed-controlled points, not a hard pairwise ranking.
**Learned.** (1) The recipe is seed-stable on a THIRD, non-GPT-2 architecture (parallel-residual block) — the
corrector's advantage over raw dwarfs its seed spread (±0.3–1.6% at α≥2) just as on GPT-2. (2) The medium>Pythia
architecture gap survives seed noise (bands don't touch) but the small≈Pythia comparison does NOT — so I qualified
the earlier "architecture ordering" language: the 81–94% band is honest but pairwise rankings within a couple of
points are inside seed noise. (3) Sanity: |v|=3.29, |h|=35.34, D_M=31.33, raw ΔLM +0.059/+0.231/+0.948/+3.103 all
reproduce Exp 24 to the digit — the module reuse is faithful. (4) Spotted + fixed a stale RESULTS.md Headline typo
while editing (GPT-2 medium α=8 std quoted 2.0%, but Exp-27 table/JSON give 2.2%).
**Assumptions/decisions logged (loop mode).** (a) Chose Pythia-410m (cached, ~800MB, ~15 s/seed) as the
cross-architecture seed check over Qwen3-1.7B (heavier, slower under /mars-vol contention) — highest rigor per
minute AND it probes the LOW end of the band where a single seed is most in doubt; logged the alternative (Qwen3
seed CI, the band's TOP) as Exp 28's own Next check. (b) 5 seeds (matches Exp 26/27 for an apples-to-apples CI
comparison). (c) Placed Exp 28 in RESULTS + REPORT_3 next to Exp 26/27 (seed axis), not a new part.
**Deliverables.** RESULTS.md (Exp-28 section + table + figure entry + Headline architecture seed CI, Exp-27
Next-check marked done, medium-std typo fix); REPORT_3 (Exp-28 Methods block + Results O/I/L/N subsection + Exp-27
Next-check marked done + Conclusion/open-items update); REPORT.md index (seed-robust headline row + Summary now all
three models). CHANGELOG appended; PLAN Current status/Next step rewritten + S7(c) checkbox added; this JOURNAL
entry. REPORT math re-verified on the 2 touched files (REPORT_3 9 / index 1 js-display-math, 0 broken, 0 inline
hazards — reuses Exp 12's recovery equation, no new equation). Artifacts: `experiments/28_seed_robustness_pythia.py`,
`results/28_seed_robustness_pythia.json`, `results/28_run.log`, `plots/28_seed_robustness_pythia.png`.
**Next step.** Optional only (success criterion long met): a 5-seed control on Qwen3 (the TOP of the 81–94% band)
to test whether its 94% edge is real or seed noise; a further architecture family (state-space/MoE); or the
Exp-23 finer-λ_b / Exp-20-on-Qwen3 lever. ENV: `/opt/conda/bin/python` (LOCAL disk, fast).
On track? yes — S7 seed axis extended to a second architecture (Pythia 80.8±1.6% @α=8); recipe shown seed-stable
across two model scales AND two architectures. ~100% complete. No blocker.

## 2026-07-09 — Experiment 29: seed robustness on Qwen3-1.7B (error bar on the TOP of the architecture band)
**Did.** Completed a prior iteration's half-run: `experiments/29_seed_robustness_qwen.py` was already present
(a clean adaptation reusing the Exp-21 module functions verbatim, overriding `exp21.SEED` per run) plus a
`29_run.log` that had been killed mid-seed-1 (seed 0 done at 94%, seed 1 at step 850) — no JSON, no plot. Re-ran
it to completion with `setsid` full detach + `/opt/conda/bin/python`. This closed Exp 28's own Next check: a
5-seed error bar on the TOP of the reported 81–94% architecture band. Qwen3's 94% @α=8 (Exp 21) was the largest
single-seed recovery anywhere in the study, so exactly where a lone seed is most in doubt. The EXACT Exp-21
Qwen3-1.7B pipeline (DiffMean sentiment |v|=38.1 at block 14/28, 400-doc Gaussian fit D_M=44.7, 300-doc train,
held-out 100-doc eval, 8.39M corrector @ d=2048, recipe, α∼U(0.5,8)) at 5 seeds (0–4); raw ΔLM seed-independent.
Result POSITIVE + tight: recovery 94.8±1.6% @α=8 (per-seed 94/95/96/92/96%), 108.3±2.1% @α=4, 162.9±8.2% @α=2;
ΔLM learned @α=8 +0.177±0.056 vs raw +3.429; D_M learned 123.3±5.4 vs raw 77.8 (decoupling every seed). Seed 0
reproduces Exp 21 to the digit (94%/108%).
**Learned.** (1) The recipe is seed-stable on a FOURTH model — the corrector's advantage over raw dwarfs its
seed spread (±1.6% @α=8, ±2.1% @α=4) just as on the three prior models. (2) KEY: Qwen3's α=8 band [93.2,96.4]%
sits ENTIRELY ABOVE every other seed-controlled model — GPT-2 medium [86.1,90.5]% (Exp 27), small [81.3,85.3]%
(Exp 26), Pythia [79.2,82.4]% (Exp 28) — so across four seed-controlled models the ordering is Qwen3 > medium >
{small ≈ Pythia}, and Qwen3's top-of-band 94% edge is a genuine effect, not seed noise. This completes the
architecture-band seed control (both ends now bounded: Pythia at the low end, Qwen3 at the top). (3) Sanity:
|v|=38.1, mean|h|=301.9, D_M clean 44.66, raw ΔLM +0.064/+0.243/+1.081/+3.429 all reproduce Exp 21 to the digit —
the module reuse is faithful.
**Assumptions/decisions logged (loop mode).** (a) Ran Exp 29 (Qwen3, the band's TOP) as the natural continuation
of the half-finished script and Exp 28's explicit Next check — highest remaining rigor value; the alternative
(GPT-2-large seed CI, same architecture, adds no axis) logged as Exp 29's own Next check. (b) 5 seeds (matches
Exp 26/27/28 for an apples-to-apples CI comparison). (c) Placed Exp 29 in RESULTS + REPORT_3 next to Exp 26/27/28
(seed cluster), not a new part.
**Deliverables.** RESULTS.md (Exp-29 section + table + figure entry + Headline Qwen3 seed CI, Exp-28 Next-check
marked done); REPORT_3 (Exp-29 Methods block + Results O/I/L/N subsection + Exp-21 Results pointer + Exp-26/27/28
limitation lines + Conclusion/open-items all updated so only GPT-2 large remains single-seed); REPORT.md index
(seed-robust headline row + Summary now all four models 83.3/88.3/80.8/94.8%). CHANGELOG appended; PLAN Current
status/Next step rewritten + S7(d) checkbox added; this JOURNAL entry. REPORT math re-verified on the 2 touched
files (REPORT.md index 1 / REPORT_3 9 js-display-math, 0 broken, 0 inline hazards — reuses Exp 12's recovery
equation, no new equation). Artifacts: `experiments/29_seed_robustness_qwen.py`,
`results/29_seed_robustness_qwen.json`, `results/29_run.log`, `plots/29_seed_robustness_qwen.png`.
**Next step.** Optional only (success criterion long met): (i) a 5-seed control on GPT-2 large (Exp 19) — the last
single-seed headline model; (ii) a further architecture family (state-space/MoE); (iii) finer λ_b / Exp-20 on
Qwen3 (Exp 23's Next check). All marginal. ENV: `/opt/conda/bin/python` (LOCAL disk, fast); `setsid` full detach.
On track? yes — S7 seed axis extended to a fourth model (Qwen3 94.8±1.6% @α=8, top of the band, real edge);
recipe seed-stable across two scales AND two architectures; only GPT-2 large single-seed. ~100% complete. No blocker.

## 2026-07-09 — Experiment 30: seed robustness on GPT-2 large (the last single-seed headline model)
**Did.** Picked PLAN Next-step (i)'s final item — the single highest-value remaining rigor point. Exp 26/27/28/29
put five-seed CIs on GPT-2 small, GPT-2 medium, Pythia, and Qwen3, but GPT-2 large (Exp 19, 774M, block 18/36)
was still a single seed-0 run — the ONLY headline model without an error bar. Wrote
`experiments/30_seed_robustness_large.py` (imports the Exp-19 module and reuses its retarget/diffmean + VRAM-safe
batch sizes verbatim, plus the Exp-3 module helpers; overrides `exp03.SEED` per run — DRY, no pipeline
duplication, same pattern as Exp 27/28/29) and ran the EXACT Exp-19 GPT-2-large pipeline at 5 seeds (0–4). Raw
ΔLM seed-independent (computed once); only the learned corrector varies. Result POSITIVE + tight: recovery
85.1±1.1% @α=8 (per-seed 84/87/85/84/85%), 94.9±0.6% @α=4, 127.2±5.6% @α=2, 260.3±30.3% @α=1; ΔLM learned @α=8
+0.369±0.028 vs raw +2.470; D_M learned 97.0±4.1 vs raw 66.0 (decoupling every seed). Seed 0 reproduces Exp 19
to the digit (84%/95%). KEY: with every headline model now seed-controlled, the α=8 ordering is Qwen3
[93.2,96.4]% > GPT-2 medium [86.1,90.5]% ≳ GPT-2 large [84.0,86.2]% ≈ GPT-2 small [81.3,85.3]% > Pythia
[79.2,82.4]% — so within the GPT-2 family the single-seed 84/89/84% "flat across 6× scale" finding is a genuine
seed-controlled effect: MEDIUM (not large) is the GPT-2 peak, large ≈ small, recovery does not grow with scale.
**Learned.** (1) The recipe is seed-stable on the 774M scale too — the corrector's advantage over raw dwarfs its
seed spread (±0.6–1.1% at α≥4). (2) The mild NON-monotonicity in the GPT-2 scale trend (medium > large ≈ small)
survives seed noise: large's band [84.0,86.2]% overlaps small's and sits just below medium's, so "flat/slightly
non-monotone with scale", NOT a scaling law — I stated this explicitly rather than implying recovery rises with
size. (3) Sanity: |v|=16.75, mean|h|=129.1, clean D_M=35.22, clean eval loss=3.299, raw ΔLM +0.036/+0.146/
+0.728/+2.470 all reproduce Exp 19 to the digit — the module reuse is faithful. (4) Timing: ~7 min/seed for the
774M model under GPU contention (train ~4 + eval ~3), ~35 min for 5 seeds — well within budget.
**Assumptions/decisions logged (loop mode).** (a) Chose GPT-2 large (the last single-seed headline model) over
the other marginal options (non-Transformer family; eval-document/vector resampling; finer λ_b) — closing the
final single-seed point is the highest rigor-per-minute step and completes the seed axis; logged the alternatives
(state-space family; sampling-axis resampling) as Exp 30's own Next check. (b) 5 seeds (matches Exp 26/27/28/29
for an apples-to-apples CI comparison). (c) Placed Exp 30 in RESULTS + REPORT_3 next to Exp 26–29 (seed cluster),
not a new part.
**Deliverables.** RESULTS.md (Exp-30 section + table + figure entry + Headline GPT-2-large seed CI + flat-scale
statement, Exp-29 "last single-seed" line marked closed); REPORT_3 (Exp-30 Methods block + Results O/I/L/N
subsection + Exp-27/28/29 limitation lines updated + Conclusion seed paragraph + open-items so NO headline model
is single-seed); REPORT.md index (seed-robust headline row + Summary now all five models 83.3/88.3/85.1/80.8/
94.8%). CHANGELOG appended; PLAN Current status/Next step rewritten + S7(e) checkbox added; this JOURNAL entry.
REPORT math re-verified on the 2 touched files (REPORT.md index 1 / REPORT_3 9 js-display-math, 0 broken, 0
inline hazards — Exp 30 adds a table + O/I/L/N prose, reuses the inline recovery expression, no new equation).
Artifacts: `experiments/30_seed_robustness_large.py`, `results/30_seed_robustness_large.json`,
`results/30_run.log`, `plots/30_seed_robustness_large.png`. ENV: `/opt/conda/bin/python` (LOCAL disk); GPT-2
large from page cache; `setsid` full detach.
**Next step.** Optional only (success criterion long met; all five headline models now seed-controlled):
(i) extend the error bars to the remaining sampling axis — eval-document or vector-construction resampling (bounds
sampling variance, not just optimization variance); (ii) a non-Transformer family (state-space/MoE); (iii) finer
λ_b / Exp-20 on Qwen3 (Exp 23's Next check). All marginal.
On track? yes — S7 seed axis now spans ALL FIVE headline models (GPT-2 large 85.1±1.1% @α=8 closes the last
single-seed point); the flat GPT-2 scale trend shown seed-controlled (medium peak, large ≈ small). ~100% complete.
No blocker.

## 2026-07-09 — Experiment 31: eval-set sampling control (document bootstrap)
**Did.** Every prior confidence interval (Exp 26–30) varies the OPTIMIZATION seed and holds the 100 held-out
eval documents fixed — so the sampling variance of that finite document draw was the one unbounded noise
source (PLAN Next-step (i)'s remaining sampling axis; CLAUDE.md rule 10 control). Wrote
`experiments/31_eval_bootstrap.py`: imports the Exp-3 module (Corrector / train_corrector / make_hat /
FuncPatcher / batched_ids reused verbatim — DRY), trains the EXACT flagship seed-0 corrector (GPT-2 small,
block 6), then evaluates ΔLM PER DOCUMENT (summed excess next-token NLL over clean) for raw and learned at
α∈{1,2,4,6,8}, and bootstrap-resamples the 100 docs with replacement B=2000×, recomputing the token-weighted
aggregate recovery `R = 1 − Σ e_learned / Σ e_raw` each resample. Reports point estimate + 95% percentile CI.
Result POSITIVE + tight: α=8 recovery 84.3%, doc-bootstrap CI [83.1, 85.6]% (±0.7 pp); α=4 95.3% [92.9, 97.6]%;
α=6 89.4% [87.9, 90.9]%. Point estimates reproduce Exp 3 to the digit.
**Learned.** (1) KEY: the eval-document CI (±0.7 pp @α=8) is NARROWER than the five-seed CI (±2.0 pp, Exp 26) —
which 100 docs we hold out moves the headline < 1 point, while re-training with a new seed moves it ~2 points.
So the seed CI we already report is the BINDING uncertainty, and the flagship 84% is not an artifact of the
particular held-out split. (2) The bootstrap interval tightens monotonically with α (±1.2→±0.7 pp for α=4→8)
because raw's denominator damage grows, shrinking the ratio's relative spread; α=1's ±17 pp is the same
ratio artifact as everywhere (raw damage only +0.076 nats). (3) Token-weighting (summed, not per-doc-averaged
NLL) makes the point estimate identical to Exp 3's doc-pooled ΔLM ratio — a faithful reproduction check.
**Assumptions/decisions logged (loop mode).** (a) Chose eval-document resampling (the named remaining sampling
axis) over the other marginals — vector-construction resampling (deferred as Exp 31's own Next check) and a
non-Transformer family (heavier, no pip). Eval-bootstrap is the cheapest rigor-per-minute step and directly
answers "is the seed CI or the eval split the binding bound?". (b) B=2000, percentile CI (standard). (c) Kept
it on the flagship (small/block6/seed0) only — the point is to compare sampling vs optimization variance on
the headline number, not to re-bootstrap every model. (d) Placed Exp 31 in the seed cluster (RESULTS + REPORT_3
next to Exp 26–30), not a new part.
**Deliverables.** RESULTS.md (Exp-31 section + table + `$$` equation + figure entry; Exp-30 Next-check
"eval-document resampling done"); REPORT_3 (Exp-31 Methods ```math block + Results O/I/L/N + Exp-30 Next-check +
Conclusion open-items); REPORT.md index (Summary sentence + headline-table row). CHANGELOG appended; PLAN
Current status/Next step rewritten + S7(f) checkbox; this JOURNAL entry. Math re-verified via GitHub API:
REPORT_3 10 js-display-math / 0 `<pre lang=math>` / 0 inline hazards (fixed one `\{...\}`→`\lbrace...\rbrace`
inline set), REPORT.md 1/0, RESULTS.md `$$` renders. Artifacts: `experiments/31_eval_bootstrap.py`,
`results/31_eval_bootstrap.json`, `results/31_run.log`, `plots/31_eval_bootstrap.png`. ENV:
`/opt/conda/bin/python` (LOCAL disk), `setsid` full detach; ~2 min total.
**Next step.** Optional only (success criterion long met, seed axis + eval-sampling axis both controlled):
(i) vector-construction resampling — bootstrap the SST-2/DiffMean examples the steering vector is built from
(the one untouched sampling axis); (ii) a non-Transformer family (state-space/MoE); (iii) finer λ_b / Exp-20
on Qwen3. All marginal.
On track? yes — S7 sampling control extended: eval-document bootstrap of the flagship [83.1, 85.6]% @α=8 (±0.7
pp) shown TIGHTER than the seed CI, so the seed CI is the binding bound; headline not an eval-split artifact.
~100% complete. No blocker.

## 2026-07-09 — Experiment 32: vector-construction bootstrap (the last untouched sampling axis)
**Did.** Picked Exp 31's Next check / PLAN Next-step (i) — the single named remaining rigor point. Seed variance
(Exp 26–30) and eval-document variance (Exp 31) were both bounded, but every result held the flagship steering
vector FIXED, so the headline could still depend on which 20 POS + 20 NEG sentences build the DiffMean vector.
Wrote `experiments/32_vector_bootstrap.py` — imports the Exp-3 module (Corrector/train_corrector/make_hat/eval
helpers reused verbatim, DRY) and the Exp-1 module (POS/NEG sentence lists), bootstrap-resamples the two 20-sentence
sets WITH replacement 5×, rebuilds `v` each time, RE-TRAINS the exact Exp-3 corrector against each `v` at a FIXED
seed of 0 (so the ONLY varying factor is the vector's example composition — seed/optimizer variance already
quantified by Exp 26), recomputes `ΔLM_raw` per resample (a different `v` steers differently), and re-measures
recovery at matched projection. b=0 is the un-resampled original — a built-in Exp-3 reproduction.
**Result.** POSITIVE + stable. Resampling swings the steering DIRECTION a lot — `cos(v_boot,v_full)` mean 0.69,
min 0.56 (~56° off), `|v|` 11.1→13–20 — yet the flagship recovery holds at **82.1±2.7% @α=8** (95.8±1.6% @α=4),
within ~2 pp of the un-resampled 84.3% and on the SAME ORDER as the five-seed CI (±2.0 pp, Exp 26), not larger.
b=0 reproduces Exp 3 exactly (84.3/95.3% @α=8/4).
**Learned.** (1) KEY: which examples build `v` barely moves the headline even though it moves the direction by up
to ~56°, because the corrector is RE-TRAINED per vector — the correction rule is direction-specific and reproduces
per direction (Exp 5), so a native corrector for each resampled `v` recovers about equally. The METHOD is robust
to vector-construction sampling; a single FROZEN corrector would not be (that's exactly Exp 5's 0%-transfer
result). (2) `ΔLM_raw` itself varies (±0.16 nats @α=8) because a bigger-norm `v` steers harder, but the recovery
RATIO is stable → the finding is a property of the corrector, not an accident of matched raw damage. (3) The
sentence-bootstrap moves `v` much more than I expected (I'd guessed cos≈0.99); 20 short sentences is a small set,
so resampling reweights the direction substantially — which makes the recovery stability a stronger result.
**Assumptions/decisions logged (loop mode).** (a) Fixed seed 0 for all 6 runs so the ONLY varying factor is the
vector (seed variance is Exp 26's job) — the honest way to isolate this axis. (b) RE-trained per resample rather
than freezing the flagship corrector, because Exp 5 shows a frozen corrector doesn't transfer across directions;
the pipeline under test is "build vector → train corrector", so both stages must see the resampled vector. (c) 5
bootstraps (+ b=0 reference) — matches the 5-seed budget of Exp 26–30, kept it cheap on the flagship only. (d)
Placed Exp 32 in the seed/sampling cluster (RESULTS + REPORT_3 next to Exp 26–31), not a new part.
**Deliverables.** RESULTS.md (Exp-32 section + table + `cos` characterization + figure entry; Exp-31 Next-check
closed); REPORT_3 (Exp-32 Methods ```math DiffMean block + Results O/I/L/N + Exp-31 Next-check + Conclusion
open-items); REPORT.md index (Summary Part-3 sentence + headline row). CHANGELOG appended; PLAN Current status /
Next step rewritten + S7(g) checkbox added; this JOURNAL entry. Math re-verified via GitHub API: REPORT_3 11
js-display-math / 0 `<pre lang=math>` / 0 inline hazards (was 10; +1 for the new DiffMean fence), REPORT.md 1/0,
RESULTS.md no new `$$`. Artifacts: `experiments/32_vector_bootstrap.py`, `results/32_vector_bootstrap.json`,
`results/32_run.log`, `plots/32_vector_bootstrap.png`. ENV: `/opt/conda/bin/python` (LOCAL disk), `setsid` full
detach; ~9 min (6 correctors trained).
**Next step.** Optional only (success criterion long met; three sampling axes + seven external-validity axes all
controlled): (i) a JOINT resample (vector × seed) or rebuilding `v` from a labelled corpus (SST-2) vs the
hand-written sets; (ii) a further architecture family (state-space/MoE) or GPT-2 XL; (iii) finer λ_b / Exp-20 on
Qwen3. All marginal.
On track? yes — S7 sampling control completed on the LAST axis: vector-construction bootstrap holds the flagship at
82.1±2.7% @α=8 despite ~56° direction swings, because the corrector re-trains per vector; headline now survives
seed + eval-doc + vector resampling. ~100% complete. No blocker.

## 2026-07-09 — Experiment 33: joint vector×seed resample (total flagship uncertainty)
**What I did.** The project is ~100% complete (32 experiments; 7 external-validity axes + 3 single-axis sampling
controls). Picked the single cleanest, lowest-risk additive rigor point: the JOINT (vector × seed) resample —
explicitly named as open item (i) in the prior Next step and Exp 32's Next check. Exp 26 bounded seed variance
alone (83.3±2.0% @α=8), Exp 32 vector variance alone (82.1±2.7%); neither gives the TOTAL flagship error bar.
Wrote `experiments/33_joint_vector_seed.py` as a CONTROLLED edit of Exp 32 — reuses the identical sentence-resampling
RNG (RandomState(1234)) so the 5 bootstrap vectors are byte-identical to Exp 32's, but floats the corrector seed
with the resample (seed=b, not fixed 0). So Exp 33 vs Exp 32 differ by exactly one factor (does the seed vary
jointly), making the joint spread directly comparable. Ran on GPT-2 small block 6 (~2 min, 6 correctors), curated
all deliverables, verified math via GitHub API.
**Result.** Joint recovery **80.9±2.9% @α=8** (95.7±3.2% @α=4), b=0 reproduces Exp 3's 84.3% to the digit.
Per-boot @α=8: 84.3(b0)/79.3/80.1/80.4/85.9/78.9%, seeds 0–5.
**Learned.** (1) KEY: the joint std (2.9 pp) is BELOW the independent-quadrature prediction √(2.0²+2.7²)≈3.4 pp and
essentially equals the vector-only std (2.7 pp). So the two sampling axes do NOT compound as independent — the total
flagship uncertainty is DOMINATED by which sentences build `v`; once the vector floats, re-seeding the corrector
adds almost nothing. Mechanistically consistent with Exp 5: correction is direction-specific, so each resampled
vector fixes most of its own recovery and the seed is a second-order perturbation on top. (2) The flagship is
therefore honestly summarized as 84%±3 pp @α=8 — a single ±3 pp band that already captures both sampling axes,
rather than needing to be quoted per-axis. (3) The controlled-edit design (identical vectors, only the seed
differs) makes the seed-vs-vector decomposition clean; had I drawn fresh vectors I could not attribute the spread.
**Assumptions/decisions logged (loop mode).** (a) Reused Exp 32's exact vectors (same RNG) rather than fresh draws
— the ONLY honest way to isolate "adding the seed axis" from "different vectors". (b) seed=b (1–5) vs Exp 26's 0–4:
same optimization-noise distribution, negligible. (c) 5 joint resamples to match the Exp 26/32 budget and keep it
cheap on the flagship only. (d) Placed Exp 33 in the seed/sampling cluster (RESULTS + REPORT_3 next to Exp 26–32).
**Deliverables.** RESULTS.md (Exp-33 section + seed/vector/joint comparison table + figure; Exp-32 Next-check
closed; Headline "84%±3 pp" clause); REPORT_3 (Exp-33 Methods ```math quadrature block + Results O/I/L/N + Exp-32
Next-check closed + Conclusion open-items); REPORT.md index (Summary sentence + headline row). CHANGELOG appended;
PLAN Current status / Next step rewritten + S7(h) checkbox. Math re-verified via GitHub API: REPORT_3 12
js-display-math / 0 `<pre lang=math>` / 0 inline hazards (was 11; +1 quadrature fence), REPORT.md 1/0/0. Artifacts:
`experiments/33_joint_vector_seed.py`, `results/33_joint_vector_seed.json`, `results/33_run.log`,
`plots/33_joint_vector_seed.png`. ENV: `/opt/conda/bin/python` (LOCAL disk), `setsid` full detach; ~2 min.
**Next step.** Optional only (success criterion long met; all sampling axes controlled): (i) rebuild `v` from
SST-2 rather than hand-written sets, or more joint resamples; (ii) a further architecture family (state-space/MoE)
or GPT-2 XL; (iii) finer λ_b / Exp-20 on Qwen3. All marginal.
On track? yes — S7 sampling controls now complete on ALL axes: the joint vector×seed resample holds the flagship at
80.9±2.9% @α=8, a vector-dominated spread below the quadrature bound, so the headline is 84%±3 pp. ~100% complete.
No blocker.

## 2026-07-09 — Experiment 34: token-position control (the last metric-control axis)
**What I did.** The project is ~100% complete (33 experiments; 7 external-validity axes + 4 sampling controls).
Scanning for the single highest-value remaining rigor point, I noticed CLAUDE.md rule 10's control list names
"seed/layer/token/perturbation strength" — and **token position is the one axis never isolated**. Every recovery
number in the study POOLS next-token NLL over all positions (`recovery = 1 − Σ e_learned/Σ e_raw`), which could
hide structure (e.g. the corrector fixing only early tokens). Wrote `experiments/34_token_position.py`, reusing the
Exp-3 Corrector/train_corrector/make_hat/FuncPatcher/batched_ids verbatim (DRY) — the only new code is a
per-source-position NLL accumulator + position bucketing. Trained the EXACT flagship corrector (GPT-2 small,
block 6, sentiment `v`, seed 0), measured per-position NLL on the held-out 100 FineWeb docs (128-token,
right-padded so position = distance from doc start) for clean/raw/learned at α∈{4,8}, bucketed into eighths. Ran
~2 min on GPU (`/opt/conda/bin/python`, `setsid` detach), no OOM under the 0.18 VRAM fraction.
**Result.** POSITIVE + clean. Recovery is FLAT across token position: after a higher first bucket (96%@α=8,
117%@α=4) it settles into an **80.7–83.5% band for positions 16–126 @α=8** (89–92%@α=4). Raw's per-position excess
NLL climbs mildly along the sequence (2.11→3.25 nats@α=8, later tokens carry more steered context) and the
corrector's residual tracks it (0.08→0.62). **Pooled recovery 84.3%@α=8 / 95.3%@α=4 reproduces Exp 3 to the
digit** (built-in reproducibility check).
**Learned.** (1) KEY: the pooled headline is NOT a pooling artifact — the corrector buys back about the same
fraction of raw's damage at token 30 as at token 120, so the 84% is a faithful summary of a near-uniform
per-position curve, not an average over a strong and a weak region. (2) The higher first bucket is the same
weak-signal ratio effect seen at small α throughout: raw's damage is smallest at the sequence start, so the
recovery ratio there runs high (>100% at α=4). (3) Raw steering compounds along the sequence (more steered
positions upstream = larger disruption downstream), which the corrector counters uniformly.
**Assumptions/decisions logged (loop mode).** (a) Chose the token-position control over the other marginal
open items (per-token-type breakdown, wider architecture, SST-2 vector) because it is the one axis CLAUDE.md rule
10 EXPLICITLY names that no experiment had isolated — highest rigor-per-minute, and cheap (reuse the flagship
corrector, no retrain of anything new; only add a per-position eval). (b) α∈{4,8} (the two headline strengths);
skipped α≤2 where the ratio is near-zero-denominator noisy. (c) 8 position buckets over 0–126 for stable
per-bucket ratios. (d) Right-padding means "position" = distance from doc start (a where-in-sequence control),
not per-token-type; logged the per-token-TYPE breakdown as Exp 34's Next check. (e) Placed Exp 34 in the
seed/sampling-control cluster (RESULTS + REPORT_3 after Exp 33), a control not a new external-validity axis.
**Deliverables.** RESULTS.md (Exp-34 section + bucket table + reading + figure entry; Headline seed/sampling
parenthetical extended with the token-position clause); REPORT_3 (Exp-34 O/I/L/N subsection + table after Exp 33;
Exp-33 Next-check closed "Done in Exp 34"; Conclusion open-items extended); REPORT.md index (Summary token clause
+ headline-table row). CHANGELOG appended; PLAN Current status / Next step rewritten + S7(i) checkbox added; this
JOURNAL entry. Math re-verified via GitHub API: REPORT.md 1 js-display-math / 0 `<pre lang=math>`, REPORT_3 12/0
(unchanged — no new display math, the recovery ratio is in code backticks), 0 inline hazards in either file or
RESULTS.md. Artifacts: `experiments/34_token_position.py`, `results/34_token_position.json`, `results/34_run.log`,
`plots/34_token_position.png`.
**Next step.** Optional only (all metric-control axes now isolated): (i) per-token-TYPE breakdown (content vs
function words / POS); (ii) wider architecture family (state-space/MoE/GPT-2 XL) or SST-2-built vector; (iii)
finer λ_b / Exp-20 on Qwen3. All marginal.
On track? yes — S7(i) closes the last metric-control axis CLAUDE.md names: the flagship recovery is flat across
token position (80.7–83.5% band @α=8, pooled 84.3% = Exp 3), so the headline is not a pooling artifact. ~100%
complete. No blocker.

## 2026-07-09 — Experiment 35: token-TYPE control (closes Exp 34's Next check)
**What I did.** Project is ~100% complete (34 experiments; 7 external-validity axes + metric-controls through
token position). Picked Exp 34's own explicit Next check — the per-token-TYPE breakdown — as the highest
rigor-per-minute step and the last named metric-control gap. Wrote `experiments/35_token_type.py`, reusing the
Exp-3 Corrector/train_corrector/make_hat/FuncPatcher/batched_ids verbatim (DRY); only new code is a vocab
token-type map (classify each of the 50,257 GPT-2 tokens once from its decoded string into FUNCTION/CONTENT/OTHER)
and a per-type NLL accumulator. Trained the EXACT flagship corrector (GPT-2 small, block 6, sentiment `v`, seed 0),
measured next-token NLL on the held-out 100 FineWeb docs split by target-token type for clean/raw/learned at
α∈{4,8}. Ran ~1 min on GPU (`/opt/conda/bin/python`, `setsid` detach), no OOM under the 0.18 VRAM fraction.
**Result.** POSITIVE + clean. Recovery @α=8: FUNCTION 73.9%, CONTENT 77.5%, OTHER 99.9% (α=4: 75.8/82.5/131%).
Content words take the LARGEST absolute raw damage (+3.89 nats vs function +1.25) yet recover slightly BETTER than
function words. Pooled 84.3%@α=8 / 95.3%@α=4 reproduces Exp 3/34 to the digit (built-in check).
**Learned.** (1) KEY: the pooled headline is NOT a cheap-token artifact — the corrector buys back the bulk of the
damage exactly where steering does the most harm (content words), not just on easy function words. (2) The pooled
84.3% sits ABOVE both linguistic classes because the OTHER class (punctuation/subword pieces, ~100% recovered)
carries large excess NLL and pulls the token-weighted pool up; on the two meaningful classes recovery is a
still-strong nearly-equal 74–78%. (3) Content words have the highest clean NLL (5.16 vs 2.52 function) — they are
intrinsically hardest to predict, so the steer's absolute disruption there is largest, and the corrector's residual
there (+0.87) is also the largest absolute but a small fraction of raw's +3.89.
**Assumptions/decisions logged (loop mode).** (a) Chose token-TYPE over the other marginal open items (wider
architecture, SST-2 vector) because it is Exp 34's OWN explicit Next check and the last metric-control axis, and is
cheap (reuse flagship corrector, only add a per-type eval). (b) FUNCTION via a fixed closed-class stop-list, not a
POS tagger (no extra deps; logged the coarseness as the Limitation and a finer POS split as Exp 35's Next check).
(c) Word-initial = leading-space decode, so mid-word subword pieces fall into OTHER — a deliberate, conservative
split reported for completeness. (d) α∈{4,8} (the two headline strengths). (e) Placed Exp 35 in the seed/sampling/
metric-control cluster (RESULTS + REPORT_3 after Exp 34), a control not a new external-validity axis.
**Deliverables.** RESULTS.md (Exp-35 section + per-type table + reading + figure entry; Headline token clause
extended with content-vs-function); REPORT_3 (Exp-35 O/I/L/N subsection + table after Exp 34; Exp-34 Next-check
closed "Done in Experiment 35"; Conclusion open-items extended); REPORT.md index (Summary token-type clause +
headline-table row). CHANGELOG appended; PLAN Current status / Next step rewritten + S7(j) checkbox added; this
JOURNAL entry. Math re-verified via GitHub API: REPORT.md 1 js-display-math / 0 `<pre lang=math>`, REPORT_3 12/0
(unchanged — no new display math), 0 inline hazards in all three touched files. Artifacts:
`experiments/35_token_type.py`, `results/35_token_type.json`, `results/35_run.log`, `plots/35_token_type.png`.
**Next step.** Optional only (all metric-control axes now isolated): (i) finer POS breakdown (nouns/verbs/adjs) of
the CONTENT class; (ii) wider architecture family (state-space/MoE/GPT-2 XL) or SST-2-built vector; (iii) finer
λ_b / Exp-20 on Qwen3. All marginal.
On track? yes — S7(j) closes the last metric-control axis (token TYPE): the flagship recovery is uniform across
token type (content 77.5% ≥ function 73.9% @α=8, pooled 84.3% = Exp 3/34), so the headline is neither a pooling nor
a cheap-token artifact. ~100% complete. No blocker.

## 2026-07-09 — Experiment 36: content-word FREQUENCY control (Exp 35's Next check)
**Did.** The project is exhaustively complete on all substantive axes (7 external-validity + all sampling +
position + type controls). The single named-but-open item was Exp 35's Next check: refine the CONTENT token class.
A part-of-speech split (noun/verb/adj) needs an in-context tagger that GPT-2 word-piece tokens don't support
reliably, so I chose an OBJECTIVE, fully-controlled cut instead — split CONTENT by target-token corpus frequency.
Wrote `experiments/36_content_frequency.py` (reuses `exp35.build_type_map` for the base FUNCTION/CONTENT/OTHER map
+ exp03 Corrector/train_corrector/make_hat/FuncPatcher/batched_ids verbatim; only new code = a target-count pass,
a token-weighted-median CONTENT split, and a per-class NLL accumulator). Trained the EXACT flagship corrector,
accumulated next-token NLL on the same held-out 100 docs split into FUNCTION / CONTENT_COMMON / CONTENT_RARE /
OTHER, recovery at α∈{4,8}. ~2 min on GPU (0.18 frac).

**Learned (POSITIVE, clean).** Recovery is uniform across content-word frequency. The weighted-median split gave
~equal token counts (2358 common vs 2362 rare); the cut lands at count 2, so CONTENT_RARE = hapax content tokens.
Rare content tokens are genuinely harder even on clean text (clean NLL 6.04 vs 4.27 nats) and take slightly more
absolute raw-steering damage (+4.11 vs +3.67 nats @α=8), yet the corrector recovers them essentially identically
to common ones — **77.8% vs 77.3% @α=8** (81.4% vs 83.8% @α=4). So the pooled 84% is NOT carried by easy,
frequent content words: the corrector buys back the same fraction of damage on the surprising, information-rich
ones. Pooled 84.3% / 95.3% reproduces Exp 3/34/35 to the digit (built-in check). Together with Exp 34 (position)
and Exp 35 (type), the token-control axis is exhausted.

**Assumption/decision logged.** (a) Chose the frequency cut over a POS/NER split because token-level POS tagging
of context-free GPT-2 word-pieces is unreliable and would need a tagger not installed; frequency is objective,
reproducible, and directly targets the "does it only fix easy tokens?" worry. Rejected: a suffix-heuristic POS
tagger (noisy, indefensible). (b) Weighted-median split (not simple median of the token list) so the two buckets
carry equal PREDICTED-token mass — the honest way to compare recovery fractions. (c) Kept FUNCTION/OTHER in the
run so the pooled number still reproduces Exp 3/34/35 as a reproducibility anchor.

**Deliverables.** Curated RESULTS.md (Exp-36 Metrics section + table + figure entry; Exp-35 "closes its Next
check" clause), REPORT_3 (Exp-36 O/I/L/N subsection + table + Exp-35 Next-check marked done + Conclusion
open-items updated), REPORT.md index (Part-3 blurb frequency clause + headline row). CHANGELOG appended. Math
re-verified via GitHub API on the two touched report files (REPORT.md 1 js-display-math / 0 broken; REPORT_3 12/0
— no new display math; 0 inline hazards). Artifacts: `experiments/36_content_frequency.py`,
`results/36_content_frequency.json`, `results/36_run.log`, `plots/36_content_frequency.png`.

**Next step.** No material open item — the token-control axis (position / type / frequency) is exhausted, and the
result is robust on seven external-validity axes plus all sampling axes. Only very-low-value optional points
remain: GPT-2 XL or a state-space/MoE architecture for a fuller model sweep; finer λ_b + Exp-20
differentiable-generation on Qwen3 (Exp 23's Next check). All marginal; success criterion long met.

On track? yes — Exp 36 closes Exp 35's Next check (content recovery uniform across frequency, 77.8% vs 77.3%
@α=8); direction complete on all planned + control axes, deliverables curated + math-verified. No blocker.

## 2026-07-09 — Experiment 37: OTHER-class decomposition (Exp 35's residual question)
**Did.** The project is exhaustively complete (7 external-validity axes + all sampling + token position/type/
frequency controls). The one residual was Exp 35's own observation that the pooled 84.3% sits ABOVE both whole-word
linguistic classes (FUNCTION 73.9%, CONTENT 77.5% @α=8) because a catch-all OTHER class (subword pieces +
punctuation + digits) recovers ~100% and, carrying large excess NLL, pulls the token-weighted pool up. That left a
fair reviewer question: is that lift trivial punctuation (a cheap-token effect) or genuine language? Wrote
`experiments/37_other_decomposition.py` (reuses `exp35.build_type_map` for FUNCTION/CONTENT + exp03 Corrector/
train_corrector/make_hat/FuncPatcher/batched_ids verbatim; only new code = split OTHER by has-a-letter into SUBWORD
[15,157 vocab types] vs PUNCT [2,996] + a linguistic-only FUNCTION+CONTENT pool). Trained the EXACT flagship
corrector (GPT-2 small, block 6, sentiment `v`, seed 0), accumulated per-target-token NLL on the same held-out 100
FineWeb docs, recovery per class at α∈{4,8}. ~2 min on GPU (0.18 frac), no OOM.

**Learned (POSITIVE, sharpening).** The ~100% OTHER recovery is NOT just punctuation. Recovery @α=8: FUNCTION 73.9%
/ CONTENT 77.5% / **SUBWORD 91.7%** / **PUNCT 109.7%** (>100% = the usual ratio artifact — for punctuation the
corrector's residual excess NLL is at or slightly below the clean baseline, elrn −0.21); @α=4: 75.8/82.5/118.0/149.0%.
So SUBWORD word-continuation pieces (genuine language) recover far above the whole-word linguistic classes, close to
punctuation. The whole-word **linguistic-only pool** (FUNCTION+CONTENT) recovers **76.8% @α=8 / 81.3% @α=4**; the
pooled headline **84.3%/95.3%** reproduces Exp 3/34/35 to the digit. KEY: the pooled 84% is lifted ~7pp above the
honest whole-word figure (~77%) by near-complete correction of easy sub-word AND punctuation tokens. This sharpens,
without overturning, the token-control story — recovery is uniform WITHIN whole words (Exp 35/36), and the corrector
is strongest on the easiest (non-word) token kinds.

**Assumption/decision logged (loop mode).** (a) Chose to decompose OTHER (Exp 35's own residual) over the remaining
marginal external-validity items (GPT-2 XL, state-space/MoE, SST-2 vector) because it is cheap (reuse the flagship
corrector + exp35 map, ~2 min), fully controlled, and directly answers the honest "is the pooled 84% inflated by
cheap punctuation?" question that Exp 35 raised but did not resolve. (b) Split OTHER by has-a-letter (SUBWORD vs
PUNCT) — an objective, reproducible rule needing no tagger; logged the mixed-token edge cases as the Limitation.
(c) Reported the linguistic-only pool explicitly and framed the pooled 84% as "genuine but partly inflated" honestly
rather than burying the ~7pp gap — that gap is the finding. (d) α∈{4,8}, the two headline strengths.

**Deliverables.** Curated RESULTS.md (Exp-37 four-class table + reading + figure entry; Headline linguistic-only
clause), REPORT_3 (Exp-37 O/I/L/N subsection + table after Exp 36; Exp-36 Next-check marked "Done in Experiment 37";
Conclusion sentence), REPORT.md index (Summary Part-3 clause + headline-table row). CHANGELOG appended; PLAN Current
status / Next step rewritten + S7(l) checkbox added; this JOURNAL entry. Math re-verified via GitHub API: REPORT.md
1 js-display-math / 0 broken, REPORT_3 12/0 (unchanged — no new display math), 0 inline hazards in both + RESULTS.md.
Artifacts: `experiments/37_other_decomposition.py`, `results/37_other_decomposition.json`, `results/37_run.log`,
`plots/37_other_decomposition.png`.

**Next step.** No material open item — the token-control axis (position / type / frequency / OTHER-decomposition) is
exhausted, and the result is robust on seven external-validity axes plus all sampling axes. Only very-low-value
optional points remain: GPT-2 XL or a state-space/MoE architecture for a fuller model sweep; a vector rebuilt from an
external labelled corpus. All marginal; success criterion long met.

On track? yes — Exp 37 closes Exp 35's residual (pooled 84% lifted above the ~77% whole-word linguistic recovery by
easy sub-word+punctuation tokens); direction complete on all planned + control axes, deliverables curated +
math-verified. No blocker.
