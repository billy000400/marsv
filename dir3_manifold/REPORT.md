# REPORT — Manifold Characterization of the GPT-2 Residual Stream

**Direction #3.** Question: how many dimensions does the GPT-2-small residual
stream actually occupy, and does a nonlinear autoencoder bottleneck agree with
nonlinear intrinsic-dimension (ID) estimators about that number?

> **Status note.** This writeup was revised after three external reviews (`REVIEW.md`,
> `CODEX_REVIEW_20260621…`, `CODEX_REVIEW_20260623…`) flagged overclaims and gaps in earlier
> drafts. The strong original framing — "two independent methods converge → demonstrated
> low-dimensional curved manifold" — has been **toned down** to what the saved artifacts
> actually support, and the latest review added duplicate/self-masking + bootstrap-CI
> diagnostics and a layer-11 post-LayerNorm caveat. See "What changed after review" at the end.

## TL;DR (honest)
On **one pooled FineWeb activation sample** from GPT-2 small, the **layer-6** residual
stream has a **low local intrinsic dimension ≈ 11–15** (TwoNN ≈ 11.7–12.7, MLE ≈ 13.4–15.2
over n=50k→20k; bootstrap *sampling* CI only ±0.1 at fixed n, so the spread is finite-sample
n-dependence, not noise). This is the trustworthy result: it is reproducible across subsample
size, robust to per-dim standardization, robust to exact duplicates / self-masking (explicit
self-index masking moves it ≤0.17), **low across coarse token-position buckets with TwoNN
stable (~9.6–10) and MLE showing estimator-dependent variation (~8.7–13.3)**, and the
estimators are validated on *synthetic linear-Gaussian* data of known dimension. It is far
below the ambient **d_model = 768** and (at layer 6 specifically) far below the linear
**d95 = 94**.

A deep autoencoder bottleneck sweep gives only a **raw-variance reconstruction artifact
consistent with low ID — not independent corroboration**: on raw centered activations it
bends from steep to shallow near **k ≈ 8–16** (seed-stable, survives approximate
param-matching, overlapping the ID band), but it does **not** plateau and the bend
**vanishes when activations are standardized** — so it largely tracks the single
"massive-activation" dimension rather than independently measuring a manifold dimension.

**Bottom line: suggestive evidence of a low (~8–16) intrinsic dimension at layer 6, not
strong proof of a globally low-dimensional curved manifold.**

## Setup
- Model: GPT-2 small (124M, d_model=768, 12 blocks), HuggingFace `transformers`,
  `output_hidden_states` → block outputs (resid_post). Layers sampled: {0,3,6,9,11}.
- Data: FineWeb (CC-MAIN-2013-20) streamed via the HF datasets-server REST API,
  912 sequences × seq_len 256 → **200k pooled token vectors per layer**, fp16.
- Compute: activation collection + all ID estimators on **CPU, 2 threads**. The AE
  sweep ran on CPU first (the box initially exposed an sm_70 V100 incompatible with the
  cu130 torch build), then was re-run on a usable **A10 GPU** (sm_86) that later became
  available — VRAM capped at 0.45 of the card per the shared-box budget.
- ID estimators, hand-rolled in pure numpy/torch (no skdim/sklearn): linear PCA
  participation-ratio + d95/d99; nonlinear **TwoNN** (Facco) and **MLE** (Levina–Bickel,
  k=20). Validated on **synthetic isotropic Gaussians linearly embedded** in 768-d —
  saved artifact `results/id_validation.json` (true_d 5→5.2, 10→10.0, 20→17.7, 50→35.1
  for TwoNN; similar for MLE), via `experiments/validate_estimators.py`. This confirms
  accuracy on *that synthetic family only*; real residual activations are curved,
  anisotropic and clustered, so it does not by itself guarantee accuracy on the real data.

## The dimension estimates for layer 6
| Estimate | Method | Value | Notes |
|----------|--------|-------|-------|
| Ambient | — | **768** | d_model |
| Linear, 95% var | PCA d95 | **94** | flat subspace for 95% variance (layer 6) |
| Linear, 99% var | PCA d99 | **479** | flat subspace for 99% variance (layer 6) |
| **Nonlinear local ID** | TwoNN / MLE | **≈11.7–12.7 / ≈13.4–15.2** | trustworthy; robust to std, position, duplicates; bootstrap CI ±0.1 at fixed n |
| AE bottleneck bend | FVU knee (raw) | **≈8–16** | raw-variance artifact; vanishes under standardization |

PCA participation ratio is **not** usable as an ID here: from layer 3 on, a single
massive-activation dimension carries 78–94% of total variance, collapsing PR to ≈1.

## Do the AE elbow and the ID estimates agree? — partially, and weakly
At a CPU budget (1200 steps) the kneedle elbow is k≈16; trained on GPU with **8.3× more
optimizer steps and 16.7× more sampled training examples** (10000 steps × batch 4096,
3 seeds, seed-std ≤0.0018) the bend tightens to k≈8 and the FVU floor drops 0.051→0.033.
So the bend, where it exists, overlaps the nonlinear ID band (12–13).

But three checks the review demanded show the AE signal is **fragile**, not strong:
1. **No plateau.** Past k≈8 the marginal FVU gain per doubling stays ~0.006–0.0075 with
   no decay out to k=256 (e.g. 128→256 ≥ 8→16). The original "flattens after k≈16"
   claim is withdrawn — k=16 is a soft kneedle output, not a clear knee.
2. **Disappears under standardization.** On z-scored activations FVU falls almost
   linearly in log-k (var_expl 25%→72% over k=2→256) with **no knee at all**. The
   raw-data bend is therefore mostly the AE capturing the one dominant dimension first
   (k=2 already explains 90% of raw variance), not a manifold-dimension signature.
3. **Parameter count — controlled approximately (param-matched sweep).** The fixed
   architecture's param count drifts with k (1.052M→1.182M), so we ran a
   **parameter-matched** sweep that holds the total fixed (spread 1024 params = 0.087%) by
   **trading the outer hidden width h1 (576→512) against the bottleneck width k**
   (`results/ae_results_matched.json`). The matched curve is within ≤0.0021 of the unmatched
   curve at every k and shows the **same low-k bend** — so the bend is *not* a parameter-count
   artifact. **Honest caveat (Codex review 2026-06-23 #2):** because h1 also changes, k is
   **not** the "only varying information channel" — this controls the param-count confound but
   not outer-width capacity. It does **not** rescue the AE: it still does not plateau (point 1)
   and still vanishes under standardization (point 2), which is what keeps the AE evidence
   weak. (train_FVU for the high-k rows k=128/256 — previously left blank in the table — is
   now filled in, 0.0379/0.0326, confirming train/val track within ≤0.002 and the "no plateau"
   tail rests on completed runs; Codex review 2026-06-23 #3.)

The **TwoNN/MLE local ID is the robust signal**; the AE merely fails to contradict it.

## Depth trend (layers 0,3,6,9,11)
Nonlinear ID grows gently with depth — mean(TwoNN,MLE) ≈ 6 (L0) → 9 (L3) → 12 (L6)
→ 14 (L9) → ~14 (L11). Per-dim standardization leaves the estimate close (Δ<2) at
layers 0/3/6/9 but **shifts layer 11 substantially** (TwoNN 16.8→11.1, MLE 12.9→16.3),
so the "outlier dim doesn't affect local geometry" statement holds at most layers but
**not at layer 11**. The layer-6 result is robust to standardization.

**Layer-11 measurement caveat (Codex review 2026-06-23 #5).** The cache stores
`GPT2Model.hidden_states[L+1]`, whose **last** entry has the model's final LayerNorm
(`ln_f`) applied. So layers 0/3/6/9 (interior indices) are genuine raw resid_post, but
**layer 11 is the post-final-layernorm hidden state, not raw block-11 resid_post**. This is
the most likely cause of layer 11's anomalous linear spectrum (d95 collapsing to 5) and its
large standardize/estimator gaps, and is exactly why the corrected "nonlinear ≪ linear" and
"robust to standardization" claims are restricted to **layer 6** (index 7, a true interior
block output, unaffected). The headline does not depend on layer 11.

## Token-position-stratified ID (layer 6)
Re-collected 80k layer-6 vectors *with* token position and estimated ID per position
bucket: early(1–15) 9.6/8.7, mid(16–63) 9.8/12.8, late(64–127) 9.9/13.3,
tail(128–255) 10.0/13.1 (TwoNN/MLE). ID is **low in every bucket, roughly similar with
estimator-dependent variation** (TwoNN ~9.6–10 stable; MLE ~8.7–13.3). This is evidence
against **one** pooling artifact — coarse absolute-position mixing — but it does *not*
control token identity, document/topic clustering, duplicate text, or local sequence
correlations, so it is not evidence against pooling artifacts in general. The claim is
scoped to "this pooled FineWeb activation sample," not "the GPT-2 residual stream."

## Caveats
- The AE elbow is **preprocessing-sensitive and non-plateauing** — read it as a
  raw-variance reconstruction artifact *consistent with* the ID band, not independent proof.
- TwoNN/MLE underestimate at large d (synthetic d=50 → ~32–35), so a true ID modestly
  above the ~11–15 band is possible; still far below d95=94 / d_model=768 at layer 6.
- The layer-6 ID has a small **finite-sample n-dependence** (TwoNN ~11.7@50k → ~12.7@20k;
  MLE ~13.4@50k → ~15.2@20k); the *sampling* bootstrap CI at fixed n is tight (±0.1), so
  report the band ~11–15, not a single point.
- **Layer 11 is post-final-layernorm**, not raw resid_post (see Depth-trend caveat); all
  layer-11 numbers must be read accordingly. Layers 0/3/6/9 are unaffected.
- The linear-vs-nonlinear "≪" gap is **layer-6-specific**: at layers 3/11 the linear
  d95 collapses to 5–6, below the nonlinear ID, so it is not a blanket statement.
- One FineWeb slice, pooled tokens (with a position-stratified spot check). Other
  corpora, removing the massive-activation dim before the AE (vs only z-scoring),
  re-collecting raw block-11 resid_post via a forward hook, and a second model remain open
  follow-ups. The **parameter-matched AE**, **bootstrap CIs**, and the **duplicate/self-
  masking diagnostic** follow-ups are now **done** (knee survives matching; CI ±0.1;
  masking moves the ID ≤0.17).

## Artifacts
- `results/pca_pr.json` — linear PCA per layer.
- `results/id_nonlinear.json` — TwoNN+MLE, 5 layers × {10k,50k} × {centered,std}.
- `results/id_validation.json` — synthetic-Gaussian validation of TwoNN/MLE.
- `results/id_by_position.json` — layer-6 ID stratified by token position.
- `results/ae_results.json` — AE FVU vs k (layer 6, CPU budget).
- `results/ae_results_gpu.json` — AE FVU vs k (layer 6, 8.3× GPU budget).
- `results/ae_results_gpu_v2.json` — AE FVU vs k: raw seeds {0,1,2} + standardized.
- `results/ae_param_counts.json` — exact AE param count per k (drifting design).
- `results/ae_results_matched.json` — **PARAMETER-MATCHED** AE FVU vs k (full k-range,
  seed 0; total params held to 0.087% spread). Knee survives matching.
- `results/ae_matched_param_counts.json` — matched param counts per k (spread 1024).
- `results/id_diagnostics.json` — layer-6 duplicate/self-masking diagnostic + bootstrap CIs.
- `RESULTS.md` — full tables, headline, and caveats.
- `experiments/` — `collect_acts.py`, `pca_pr.py`, `id_estimate.py`,
  `validate_estimators.py`, `collect_by_position.py`, `id_diagnostics.py`, `ae_sweep.py`,
  `ae_sweep_gpu.py`, `ae_sweep_gpu_v2.py`, `ae_sweep_matched.py`.

## Conclusion
On a pooled FineWeb sample, GPT-2's **layer-6 residual stream has a low local intrinsic
dimension ≈ 11–15** (TwoNN/MLE; bootstrap sampling CI ±0.1, the band being finite-sample
n-dependence), an estimate that survives standardization, holds across token positions, is
robust to exact duplicates / self-masking, and uses validated estimators — far below the
ambient width (768) and the layer-6 linear d95 (94). A longer-trained, multi-seed,
parameter-matched autoencoder bottleneck bends in the same ~8–16 range but provides only a
raw-variance reconstruction artifact consistent with that band: it does not plateau and
disappears under standardization. So the residual stream **appears** to occupy a low-
dimensional set at layer 6, but "~12–16-dimensional curved manifold" should be read as a
suggestive pilot finding, not a demonstrated property of the full residual stream.

## What changed after review (`CODEX_REVIEW_20260623T001526Z.md`)
- **Ran ID diagnostics** (`experiments/id_diagnostics.py`, GPU): (#4) duplicate/self-masking
  — 92/50k exact dupes, explicit self-index masking moves TwoNN 0.00 / MLE +0.17, so the ID
  is not a duplicate artifact; (rec#5) bootstrap CIs — TwoNN 12.71±0.13, MLE 15.18±0.09 at
  n=20k, revealing the band is finite-sample n-dependence not noise. ID headline widened
  11–13 → **11–15**.
- **Documented the layer-11 post-final-layernorm issue** (#5): `hidden_states[11+1]` is the
  ln_f-applied final state, so all layer-11 numbers are post-LN; layer 6 (the headline) is a
  true interior block output and unaffected.
- **Wording fixes:** AE reframed "weak corroboration" → "raw-variance reconstruction artifact
  consistent with low ID" (#1); param-matched section now states h1 varies and drops "k is
  the only varying channel" (#2); headline "stable across token position" → "low across
  buckets; TwoNN stable, MLE estimator-dependent" (#6).
- **Filled the matched-AE table** train_FVU for k=128/256 from the existing JSON (#3).

## What changed after earlier review (`REVIEW.md` + `CODEX_REVIEW_20260621T031919Z.md`)
- **Ran the parameter-matched AE sweep** the Codex review required (concern #4 / step #1):
  total params held to 0.087% spread; the low-k bend survives, so it is not a param-count
  artifact (`results/ae_results_matched.json`). Also scoped "stable across position"→
  "roughly similar with estimator-dependent variation", scoped the synthetic validation to
  linear-Gaussian data only, reworded "8.3× train budget"→"8.3× steps / 16.7× examples",
  and cleaned the stale duplicated operator blocks from PLAN.md.

### Earlier changes (`REVIEW.md`)
- Retracted "AE and ID converge → strong evidence of a curved manifold"; reframed the
  AE as weak, preprocessing-sensitive corroboration (added standardized + multi-seed
  sweeps proving it).
- Fixed false claims: "standardization changes ID by <2 everywhere" (false at L11);
  "TwoNN/MLE agree within ~3 everywhere" (false at L11, gap ~5.2); "nonlinear an order
  of magnitude below d95 across layers" (false at L3/L11) — all now layer-scoped.
- Corrected the "curve flattens after k≈16" claim (it does not; no plateau).
- Saved the previously-unsubstantiated synthetic validation as an artifact.
- Added token-position-stratified ID; scoped the conclusion to the pooled sample.
- Reported the uncontrolled AE parameter count explicitly with the confound direction.
- Note: this folder is direction #3 (manifold). The review's reminder that OOD/
  adversarial-plateau work is direction #9 (`../dir9_ood`) is noted; no scope change
  here, this loop is dir #3 by construction.
