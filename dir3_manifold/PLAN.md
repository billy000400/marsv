# PLAN — Direction #3: Manifold Characterization of the GPT-2 Residual Stream

> The agent REWRITES "Current status" and "Next step" and ticks the stage boxes every iteration.
> Disk (this file + JOURNAL.md + RESULTS.md) is the only memory. All paths are relative to this folder.

## Success criterion (definition of "done")
Produce all three:
1. Intrinsic-dimension (ID) estimates of the GPT-2 residual stream on FineWeb, per layer, from >=3 estimators (TwoNN, MLE, PCA participation ratio) — in RESULTS.md.
2. An autoencoder bottleneck sweep (held-out reconstruction error vs k) with an identified elbow — in RESULTS.md.
3. REPORT.md stating whether the AE elbow-k agrees with the nonlinear ID estimates (and how both compare to the linear PCA value and to d_model = 768).

**A disagreement between the two estimates is a complete, valid result.** When all three exist, create an empty `STOP` file.

## Fallback (if time runs short)
Minimum acceptable: ID estimates for layer 6 from TwoNN + MLE + PCA, plus a partial AE sweep over at least k in {4,8,16,32,64,128}. Always use the final 20 min to finalize whatever exists into RESULTS.md + REPORT.md and STOP.

## Setup (fixed)
- Model: GPT-2 small (124M, d_model=768, 12 layers). Use HuggingFace `transformers` (already installed) with forward hooks on each block's output to capture the residual stream. **Do NOT `pip install transformer_lens`** — its `torchvision<0.23` pin downgrades and breaks the cluster's CUDA-13 torch.
- **Shared hardware + time limits live in `../BUDGET.md` — read it every iteration.** You share one RTX 3090, 16 GB RAM, and 4 CPU with the other agent, so stay within your half: cap VRAM (`set_per_process_memory_fraction`), memmap activation caches to disk, keep batches small, halve on OOM.
- Data: FineWeb — STREAM a sample, do not download all. ~2-5k sequences, length 128-512.

## Stages (checklist)
- [x] **S1 — collect activations.** DONE: 200k raw fp16 vectors/layer for {0,3,6,9,11}, all token positions pooled, via FineWeb REST API + GPT-2 on CPU. Files in `data/`.
- [x] **S2 — intrinsic dimension.** Linear PCA-PR done (all layers) → `results/pca_pr.json`. Nonlinear TwoNN+MLE done (pure-numpy, validated on synthetic) → `results/id_nonlinear.json` + RESULTS.md. Nonlinear ID ≈ 6–16 across layers, grows with depth, robust to standardization.
- [x] **S3 — AE bottleneck sweep (layer 6).** DONE. k∈{2..256}, STEPS=1200/BATCH=2048 (CPU-sized, identical across k) → `results/ae_results.json` + RESULTS.md. **Elbow k≈16** (kneedle on log₂-k); train/val FVU track within 0.001 (no overfit).
- [x] **S4 — report.** DONE (then revised in S5). `REPORT.md` written comparing AE elbow vs nonlinear ID vs PCA d95 vs d_model.
- [x] **S5 — address REVIEW.md (operator-required).** DONE. Toned down overclaims; fixed false claims; added the experiments the review asked for. New artifacts: synthetic validation (`results/id_validation.json`), token-position-stratified ID (`results/id_by_position.json`), longer-trained GPU AE sweep (`results/ae_results_gpu.json`), multi-seed + standardized AE + param counts (`results/ae_results_gpu_v2.json`, `results/ae_param_counts.json`). RESULTS.md + REPORT.md rewritten to the honest conclusion; `REVIEW.md`→`REVIEW.md.addressed.md`. ENV update: GPU now usable (A10 sm_86); transformers re-importable.
- [x] **S5b — Codex review (param-matched AE).** DONE (Iter 6). Param-matched AE sweep (total params fixed to 0.087% spread) → `results/ae_results_matched.json`; **knee survives matching** (within ≤0.0021 of unmatched at every k). All Codex wording points handled. RESULTS/REPORT updated; review file `→ .addressed.md`.
- [x] **S5c — Codex review 2026-06-23 (ID diagnostics + correctness).** DONE (Iter 7). New artifact `results/id_diagnostics.json` (GPU): duplicate/self-masking (#4 — 92/50k dupes, self-index masking moves TwoNN 0.00/MLE +0.17 → not an artifact) + bootstrap CIs (rec#5 — TwoNN 12.71±0.13, MLE 15.18±0.09 @n=20k; ID band widened 11–13→**11–15** as finite-sample n-dependence). Documented the **layer-11 post-final-layernorm** issue (#5: `hidden_states[11+1]` has ln_f; L6 unaffected). Wording fixes (#1 AE="raw-variance artifact"; #2 matched-AE h1 varies; #6 token-position). Filled matched train_FVU k=128/256 (#3). RESULTS/REPORT updated; review `→ .addressed.md`.
- [x] **S6 — figures + consolidation (Iter 8, 2026-06-30).** DONE. Installed matplotlib (numpy/torch pinned); `experiments/make_plots.py` renders 6 PNGs from saved JSON into `plots/` (no recompute), referenced from RESULTS.md/REPORT.md. REPORT.md rewritten to clean Summary→Methods→Results→Conclusion with `$$LaTeX$$` metric/baseline defs; "What changed after review" history moved from REPORT.md into CHANGELOG.md (rules 6–7). No numbers changed.
- [ ] *(stretch, only if reopened)* AE on outlier-dim-removed (not just z-scored) activations; bootstrap CIs for TwoNN/MLE; a second model / corpus; TDA persistent homology (`ripser`/`giotto-tda` --no-deps) on a layer-6 subsample.

## Out of scope (do NOT)
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, or flax** — they downgrade and break the cluster's CUDA-13 torch. Use only the already-installed env (torch + HuggingFace `transformers` + numpy/sklearn/matplotlib). Install missing pure-python packages with `--no-deps`.
- Don't use the manifold for any downstream task (steering / probing / editing) — utility is a separate open question.
- Don't train large models or let any single run exceed ~10 min; the AEs are small MLPs.
- Don't drift into other directions.

## On-track check (required every iteration)
End each JOURNAL.md entry with one line: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status
**COMPLETE — S1–S6 done; SIX operator reviews/feedback addressed; STOP written.** Iter 12 (2026-07-02): addressed a new one-line operator request (`human_feedback_07021056.md`): *"make a plot of accumulated PCA variance and mark the 95% and 99% points."* Recomputed the full per-layer cumulative-variance curve from the same mean-centered covariance eigen-spectrum as the existing PCA table (d95/d99 reproduce exactly), saved `results/pca_cumvar.json` (`experiments/pca_cumvar.py`), added a Fig-6 block to `experiments/make_plots.py` → `plots/pca_cumvar.png` (cumvar vs #PCs, log x, 95% ● / 99% □ marked per layer), and embedded it in RESULTS.md's linear-PCA section with a two-regime reading. Reinstalled matplotlib (env reset again) with numpy/torch pinned — verified unchanged. No result numbers moved; CHANGELOG appended, feedback renamed `→ .addressed.md`, STOP written.
**COMPLETE — S1–S6 done; FIVE operator reviews/feedback addressed; STOP written.** Iter 11 (2026-07-02): addressed a new one-off operator feedback file (`human_feedback_07010525.md`, 5 clarity questions): (1) *why/where pooled?* → added a "what pooled means / where" paragraph to REPORT Methods Data (every non-pad token = own point, `hidden_states[L+1][mask]`, not per-seq averaging) + RESULTS pointer; (2) *what is Kneedle?* → defined (Satopää 2011, max chord-distance) in Metrics; (3) *MLE?* → expanded to Maximum Likelihood Estimation in REPORT+RESULTS; (4) *why emphasize isotropic Gaussian?* → added why-clause (easiest case → necessary-not-sufficient) to validation Results; (5) *raw bend doesn't look like a bend?* → agreed & strengthened honestly in REPORT+RESULTS AE sections (only k=2→4 steep; soft Kneedle; "consistent with" not "evidence for"). No result numbers moved; render re-verified (4 js-display-math / 0 degraded / 0 hazards). Renamed feedback `→ .addressed.md`, re-wrote STOP.
**COMPLETE — S1–S6 done; FOUR operator reviews/feedback addressed; STOP written.** Iter 10 (2026-07-02): re-entry found STOP absent again (does not persist across relaunches) and no unaddressed review/feedback file; re-verified deliverable integrity (REPORT.md → 4 js-display-math / 0 degraded / 0 inline hazards; all 6 plots + all results JSON present), no deliverable change, re-wrote STOP.
**COMPLETE — S1–S6 done; FOUR operator reviews/feedback addressed; figures + consolidation finished; STOP written.** Iter 9 (2026-07-01) addressed a new one-line operator question (`human_feedback_07010347.md`): *"In TwoNN, where do the two points live? What is F()?"* — a Methods-clarity ask, no science change. REPORT.md TwoNN paragraph now states explicitly that the reference point and its two nearest neighbours live in the **ambient 768-d residual-stream space** (Euclidean, no projection) and that **$F$ is the CDF of the neighbour-distance ratios $\mu_i=r_2/r_1$** (empirical sorted-rank; Pareto $1-\mu^{-d}$). Added a display fence for $F(\mu)=1-\mu^{-d}$ (4 display eqs, all render), removed an inline eq with rule-8b-hazardous `\!`/`\,`, added a one-line pointer in RESULTS.md, renamed the feedback file `→ .addressed.md`, logged in CHANGELOG. No numbers moved.
**COMPLETE — S1–S6 done; THREE operator reviews addressed; figures + consolidation finished; STOP written.** Iter 8 (2026-06-30) closed the two remaining CLAUDE.md hygiene gaps: (1) `plots/` was EMPTY — now holds 6 figures covering every reported quantitative result (`experiments/make_plots.py`, rendered from saved JSON, no recompute), referenced from RESULTS.md/REPORT.md; (2) the consolidation pass — REPORT.md rewritten to clean Summary→Methods→Results→Conclusion with `$$LaTeX$$` metric/baseline definitions, and all "What changed after review" version-history moved from REPORT.md into the (previously empty) append-only CHANGELOG.md. No result numbers changed.
**COMPLETE — S1–S5c done; THREE operator reviews (REVIEW.md + 2× CODEX_REVIEW) addressed.** Iter 7 addressed the new `CODEX_REVIEW_20260623T001526Z.md`: ran ID diagnostics on GPU (duplicate/self-masking — ID not an artifact, moves ≤0.17; bootstrap CIs — sampling ±0.1, band is finite-sample n-dependence → ID widened to **11–15**), documented the layer-11 post-final-layernorm correctness issue (#5; layer-6 headline unaffected), and applied all wording/reporting fixes (#1,#2,#3,#6). No code-correctness change needed for layer 6.
**COMPLETE — S1–S5b done; both operator reviews (REVIEW.md + CODEX_REVIEW…) addressed; STOP written.** Iter 6 finished the one substantive outstanding Codex ask — the **parameter-matched AE sweep** — which shows the low-k bend is *not* a param-count artifact (matched curve within ≤0.0021 of unmatched at every k); the AE remains weak corroboration (no plateau; vanishes under standardization). Earlier status below stands.
**COMPLETE — S1–S5 done; all REVIEW.md points addressed; STOP written.** Success criterion met (3 estimators' per-layer ID; AE bottleneck sweep with identified knee; REPORT.md comparison) AND the operator-required review fully handled. **Honest headline: layer-6 residual stream has a low local intrinsic dimension ≈11–13 (TwoNN/MLE) — robust to subsample size, standardization, token position, and validated on synthetic data. The AE bottleneck bends in the same ~8–16 range but is weak (no plateau; vanishes under standardization), so it is corroborative, not strong proof. → "suggestive low-dim, not a demonstrated curved manifold."** ENV (updated): GPU is now a usable **A10 (sm_86)** — cu130 torch runs CUDA fine; `transformers` re-importable. The earlier CPU-only/V100-dead constraint no longer applies.

## Next step
None required — all stages (S1–S6) + ALL THREE reviews done; ID diagnostics + bootstrap CIs complete; figures rendered for every result; deliverable history consolidated to CHANGELOG.md; STOP written. If reopened (stretch, GPU now a usable RTX 3090): re-collect raw block-11 resid_post via a forward hook on `h[11]` (current L11 cache is post-ln_f); AE on activations with the massive-activation dim removed (not just z-scored); a second model / corpus; TDA persistent homology (`ripser`/`giotto-tda` --no-deps) on a layer-6 subsample.

## Operator review log
- **REVIEW.md** (Iter 5) — ADDRESSED → renamed `REVIEW.md.addressed.md`. Point-by-point in JOURNAL Iter 5.
- **CODEX_REVIEW_20260621T031919Z.md** (Iter 6) — ADDRESSED → renamed `*.addressed.md`. Point-by-point in JOURNAL Iter 6.
- **CODEX_REVIEW_20260623T001526Z.md** (Iter 7) — ADDRESSED → renamed `*.addressed.md`. Point-by-point in JOURNAL Iter 7.
- **human_feedback_07010347.md** (Iter 9) — ADDRESSED → renamed `*.addressed.md`. TwoNN space & F() clarified in REPORT Methods.
- **human_feedback_07010525.md** (Iter 11) — ADDRESSED → renamed `*.addressed.md`. 5 clarity Qs (pooling/Kneedle/MLE/isotropic-Gaussian/"raw bend"); Methods edits + honest AE strengthening; no numbers moved.
- **human_feedback_07021056.md** (Iter 12) — ADDRESSED → renamed `*.addressed.md`. Requested a cumulative-PCA-variance plot with 95%/99% marks; added `results/pca_cumvar.json` + `plots/pca_cumvar.png` (make_plots Fig 6) + RESULTS embed; no numbers moved.
  Handled: #1 AE wording ("raw-variance reconstruction artifact"), #2 matched-AE h1-varies
  caveat, #3 filled matched train_FVU k=128/256, #4 duplicate/self-masking diagnostic (new
  artifact `id_diagnostics.json`), #5 layer-11 post-final-layernorm caveat, #6 token-position
  headline wording. Recommended: bootstrap CIs (done, `id_diagnostics.json`). Remaining recs
  (raw L11 re-collection, matched-sample-size position buckets, second model) are future work.
  Handled: param-matched AE sweep (new artifact), all wording corrections (token-position
  "stable"→"estimator-dependent"; pooling scope; synthetic-validation scope; "8.3× steps /
  16.7× examples"; "validated"→synthetic-only), and this PLAN cleanup of the stale duplicated
  operator blocks the review flagged.

> Process note for future iterations: when a new `*REVIEW*`/`*feedback*` file with an
> unaddressed timestamp appears, address every point, log it in JOURNAL, update
> RESULTS/REPORT/PLAN, then append `.addressed.md` to the file. Only write STOP when no
> unaddressed review file remains.

## Consolidation pass — do this, then STOP
Per CLAUDE.md: rewrite REPORT.md and RESULTS.md to CURRENT-BEST only — remove all version history,
"changed after review" notes, and any weaker/superseded run of an experiment that a stronger run
replaces. Move everything you remove into CHANGELOG.md as dated entries (old -> new numbers). Ensure
REPORT.md has the Methods section: Data/Model/Layer, and every metric + baseline defined with $$LaTeX$$.
