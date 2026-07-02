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
