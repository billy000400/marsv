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
