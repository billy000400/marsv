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
