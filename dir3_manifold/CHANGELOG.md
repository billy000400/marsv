# CHANGELOG — Direction #3 (Manifold)

Append-only. History of changes to the deliverables (RESULTS.md / REPORT.md).
Current-best numbers live in those files; this file records how they got there.

---

## 2026-07-09 (Iter 18) — Learning-curve diagnostic: the lasse U-shape's rising branch is undertraining (operator `human_feedback_07082157.md` ask #3)
- **Operator ask** (`human_feedback_07082157.md`, three parts): (1) TwoNN's dependence on sample size —
  is the spread noise? (2) AE with no elbow — try last-token reconstruction; (3) for the lasse
  reproduction, plot train+val loss vs step for each `k` to check if the large-`k` AEs were undertrained,
  and infer what the plot would look like if they all converged. Asks (1) and (2) were already delivered
  in prior autoloop iterations (RESULTS.md "TwoNN/MLE depend on the number of sampled points…" →
  `results/id_vs_n.json`; "Last-token vs pooled AE…" → `results/ae_results_lasttoken.json`). This entry
  covers ask **(3)**, which was outstanding.
- **New artifacts:** `ae_study/ae_learning_curves.py` (per-step train+held-out logging of the identical
  67M `DeepAutoencoder` at `k∈{10,50,100,200,500}`, run to **8,000 steps** vs the 3,000-step
  reproduction; small resume patch so an interrupted run's completed k's are reused, not recomputed),
  `ae_study/results/qwen_lcurve_L2.json`, `ae_study/make_lcurve_plots.py` → `plots/qwen_ae_lcurve.png`
  (left: per-k learning curves; right: k-sweep at 3,000 vs 8,000 steps).
- **Result (answers ask #3, "yes, undertrained"):** at the 3,000-step lasse budget every learning curve
  is **still descending**, and larger `k` are farther from their own 8,000-step value (3,000→8,000 drop
  grows monotonically: 0.010/0.016/0.022/0.026/0.031 for k=10/50/100/200/500). Training to 8,000 steps
  **lowers the sweep and shrinks the rising branch**: held-out min stays near `k≈100` but drops
  0.464→0.442, and the rise out to `k=500` shrinks from +0.034 (3,000-step) to +0.025 (8,000-step, ≈26%
  smaller). Decisively, **at 8,000 steps the *train* rel-L2 still rises past the minimum**
  (0.406→0.418→0.435 for k=100/200/500) — a containment/dead-wiring bound says train error cannot rise
  with `k` at convergence, so the `k≥100` models remain undertrained. **Inference for "if all
  converged":** the held-out sweep would become **monotone non-increasing in `k`** (no U-shape); the
  `k≈50–100` "optimum" is a fixed-budget turning point, not a manifold dimension. Our 8,000-step run is
  one concrete step along that trajectory (shallower U, lower floor) but does not reach convergence.
- **Deliverable impact:** additive. New RESULTS.md subsection under the Qwen study; new point (b′) +
  figure in REPORT_AE.md §2; one high-level sentence + figure reference in REPORT.md's cross-model
  section. No prior numbers changed; interpretation of the reproduced elbow (optimization artifact, not
  ID) is unchanged and now empirically reinforced. Render re-verified: REPORT.md 6/6, REPORT_AE.md 6/6
  js-display-math, 0 degraded, 0 inline hazards.

## 2026-07-08 (Iter 17) — Qwen AE elbow REPRODUCED (supersedes the earlier "no elbow"); REPORT_AE.md equation-render fix (operator `human_feedback_07070249.md`)
- **Operator report:** (1) "what do you mean you cannot reproduce the AE study? His plot is at
  `ae_study/lasse.png` — try to reproduce it." (2) "in REPORT_AE.md the equations under Baselines /
  controls are not rendered correctly on GitHub."
- **(1) The colleague's elbow now REPRODUCES — verdict flipped.** The 2026-07-07 entry below concluded the
  colleague's recipe shows **NO elbow** (held-out FVU declining smoothly over `k∈{5..30}`). That was an
  artifact of a **truncated `k` range**: the held-out minimum sits at `k≈50–100`, *past* the old sweep's
  upper end. Re-sweeping the identical 67M-param `DeepAutoencoder` over the **full range to `k=500`**
  (3000 steps/k, seed 0, Qwen3-1.7B layer-2 last-token) reproduces `lasse.png`'s **U-shape**: held-out
  rel-L2 **0.576 → broad min ≈0.486 @ k≈40–100 → 0.529 @ k=500**; cosine **0.780 → peak 0.853 → 0.821**;
  FVU **0.581 → 0.410 → 0.488**. Our error floor is higher than the colleague's (0.486 vs ~0.407) only
  because we train 3000 steps vs ~50k (shifts curve height, not shape). **Superseded claim:** "colleague's
  recipe shows no elbow" → "elbow reproduces as a U-shape when the sweep extends past `k≈50`."
- **Interpretation unchanged and reinforced:** the reproduced elbow is a **turnaround (U-shape)**, and it
  is an **optimization/training-budget artifact, not a manifold dimension** — the *same* turnaround
  appears on the training set (rules out overfitting), and at convergence a wider bottleneck cannot do
  worse (dead-wire the extra latents), so the rise is under-optimization at the fixed step budget. The
  separate *sharp, plateauing* elbow (injected-massive-dim control, FVU 0.10@k=1 → flat 0.066 by k=16)
  still requires concentrated variance. So no interpretation changed; only the reproduction verdict
  flipped from "absent" to "present (U-shape, needs full range)."
- **New artifacts:** `ae_study/ae_sweep_lasse.py` (wide-`k` reproduction, logs train+val FVU/rel-L2/cos),
  `ae_study/results/qwen_sweep_L2_lasse.json`, `ae_study/make_lasse_plot.py`,
  `plots/qwen_ae_lasse_repro.png` (reproduction figure, held-out + train overlay). Embedded in REPORT.md,
  RESULTS.md, and REPORT_AE.md.
- **(2) Equation-render fix.** On the remote at commit `98958a1` (the first push after the ~20 h
  push-protection block, see Iter 16), the Baselines/controls block combined the participation-ratio and
  top-1 definitions into **one** `math` fence using `\operatorname{Var}\!\big(x^{(c)}\big)` — a
  negative-thin-space + `\big` delimiters + `\operatorname` combination that KaTeX rendered incorrectly in
  the GitHub blob view even though the `js-display-math` placement check passed. The current REPORT_AE.md
  **splits this into three separate column-0 fences using plain `\mathrm{Var}(x^{(c)})`** (no `\!`, no
  `\big`, no `\operatorname`), which render correctly. Verified now: REPORT_AE.md **6/6 display equations
  render, 0 degraded, 0 inline hazards** via the GitHub markdown API; grep for `\operatorname`/`\big`/`\!`
  in REPORT_AE.md returns none.
- **Rule-6 cleanup:** removed "an earlier version wrongly reported no elbow / correcting an earlier claim"
  self-correction wording from REPORT.md, RESULTS.md, and REPORT_AE.md (that is version history — it
  belongs here, not in the deliverables). Replaced with the current-best **methodological** statement:
  the elbow lives at `k≈50–100`, so a sweep truncated at `k≤30` misleadingly looks monotone; the range
  must extend past the minimum.
- **Deliverable impact:** no GPT-2 numbers changed; the Qwen reproduction table/figure were already on
  disk from prior (un-journaled) autoloop work — this entry documents the flipped verdict, the render fix,
  and the rule-6 rewording.

## 2026-07-07 (Iter 16) — Fixed remote-sync failure: purged a hardcoded secret blocking every push (operator request `human_feedback_07070207.md`; no result numbers changed)
- **Symptom (operator report):** REPORT.md was not updating on GitHub and REPORT_AE.md never appeared
  on the remote, even though both were committed locally and the ae_study/ tree was fully tracked.
- **Root cause:** the wrapper's automatic `git push` was being **rejected by GitHub Push Protection**
  (rule violation `GH013`), not by SSH/auth (which the wrapper self-heals). The colleague's shared
  script `ae_study/ae_share/scripts/autoencoder/run_final_eval.sh:16` hardcoded a **HuggingFace user
  access token**. Push protection refused the whole branch, so all 8 dir3 commits (2026-07-07 18:33 →
  20:11, including the REPORT.md updates and the new REPORT_AE.md) were stranded local-only for ~20 h
  while `origin/main` stayed at another direction's 00:26 commit. Because every push failed, the token
  **never reached the remote** — no public exposure.
- **Fix:** redacted the line to read the token from the `HF_TOKEN` env var (no secret in source), then
  purged the token from all 8 unpushed commits via `git filter-branch --index-filter` (blob swap;
  history rewrite is safe since none were pushed). Verified the token is absent from every rewritten
  commit and from `origin/main`, then pushed `4e40252..98958a1`. Remote now matches local HEAD:
  REPORT.md updated, REPORT_AE.md present, all 55 ae_study/ files tracked.
- **Deliverable impact:** none on numbers/figures — this restored the *availability* of the current-best
  REPORT.md/REPORT_AE.md/RESULTS.md on the remote. The only source change is the secret redaction.
- **Follow-up for the operator:** the leaked HF token was a real-looking credential in a colleague's
  shared file; even though it never hit GitHub, rotating it is the safe default.

## 2026-07-07 — Cross-model AE study (Qwen3-1.7B) + TwoNN-vs-MLE-only figure (operator request; no prior GPT-2 numbers changed)
- **Operator request** (`human_feedback_07071040.md`): (1) add a version of the nonlinear-ID plot
  showing **only TwoNN and MLE** (no linear PCA, no d_model) to judge how well the two estimators
  agree; (2) reproduce a colleague's autoencoder study (bundle `autoencoder_share.tar.gz`) in a
  dedicated workspace, report it as **REPORT_AE.md** with only high-level results in REPORT.md, and
  find whether any factor makes the AE reconstruction error show an **elbow** in FVU/reconstruction
  error — with a controlled experiment showing when it does and does not appear.
- **(1) TwoNN-vs-MLE-only figure** → `plots/id_twonn_vs_mle.png` (`experiments/make_plots.py` Fig 1b,
  linear y-axis, no PCA/d_model). Embedded in REPORT.md and RESULTS.md nonlinear-ID section. No numbers
  changed — same `id_nonlinear.json` data as the main figure.
- **(2) Qwen3-1.7B AE study** (new workspace `ae_study/`): collected Qwen3-1.7B last-token residual
  activations (layers 2 & 10, FineWeb-Edu sample-10BT, seq_len 10, 160k vectors/layer;
  `collect_qwen.py`) and swept the colleague's unmodified 67M-param `DeepAutoencoder`
  (`2048→4096→4096→2048→k`, MSE on raw acts; `ae_sweep_qwen.py`).
  - **Faithful reproduction shows NO elbow:** held-out FVU falls smoothly and never plateaus —
    L2 0.569→0.404, L10 0.629→0.434 over k=5→30; at k=30 the AE still leaves ~40% variance unexplained.
  - **Why:** these clouds are high-dimensional — PCA participation ratio **245 (L2) / 42 (L10)**,
    ≤3.4% variance in any one direction (`pca_diag.py`), vs GPT-2 L6's 90.4%/PR≈1.2.
  - **Controlled experiment (decisive):** rescaling ONE coordinate of the same isotropic Qwen L2
    activations to carry 90% of the variance (`--inject_massive 0.90`), changing nothing else, makes the
    identical AE snap to a **sharp low-k knee + flat plateau** (FVU 0.099 at k=1 → 0.066 flat by k=16),
    while the isotropic run keeps declining at k=64 (0.851→0.448). **Elbow ⟺ concentrated variance
    (anisotropy)** — not model/layer/token/dataset/AE-size.
  - New deliverable `REPORT_AE.md` (Summary→Methods→Results→Conclusion, 5 rendered math fences, all
    render / 0 degraded / 0 hazards); high-level 3-bullet result added to REPORT.md
    "Cross-model check" section; RESULTS.md gained a Qwen AE subsection with both tables + figure.
  - Figures: `plots/qwen_ae_sweep.png`, `plots/qwen_anisotropy.png`, `plots/qwen_ae_wide_controlled.png`
    (`ae_study/make_ae_plots.py`). Results JSON under `ae_study/results/qwen_sweep_*.json`,
    `qwen_pca_diag.json`.
- No GPT-2 (main-study) numbers changed; this is additive cross-model work.

## 2026-07-06 — AE elbow-k under reconstruction error & cosine similarity (operator request; no prior numbers changed)
- **Operator request** (`human_feedback_07060326.md`): *"I saw you used FVU to calculate ID from AE.
  Can you use reconstruction error and cosine similarity and find the ID again? Don't forget to
  include the definition of your cosine similarity."*
- Re-scored the **same** layer-6 GPU AE models (identical arch/optimizer/data/split/STEPS=10000/seed 0;
  `experiments/ae_sweep_metrics.py`) with two additional held-out metrics on the centered vectors:
  **per-dimension RMSE** and **mean cosine similarity** (definitions added to REPORT.md Methods and
  RESULTS.md). New artifacts `results/ae_results_metrics.json`, `results/ae_metrics_elbow.json`,
  `plots/ae_metrics_id.png`. FVU column reproduces `ae_results_gpu.json` to ≤0.0001 (models reproduced).
- **Result:** Kneedle elbow-k is **k=4 under all three metrics** (FVU, RMSE, cosine) — the AE "ID" is
  metric-robust, not an FVU artifact. But this pins the *location*, not the *strength*: cosine rises
  0.44→0.61 over the one steep step k=2→4 then climbs near-linearly to 0.86 at k=256 with no saturation
  (same no-plateau tail); absolute quality at the elbow is modest (k=4 cosine 0.61, RMSE ~89% of the
  k=256 floor), reinforcing the existing "consistent with, not proof of, low ID" conclusion.
- REPORT.md Methods: added RMSE + cosine-similarity definitions (render check 6 js-display-math /
  0 degraded / 0 inline hazards). Added AE-section point #4 + the new figure. RESULTS.md: new
  subsection with the three-metric table, elbow finding, and figure. No headline or prior numbers changed.

## 2026-07-02 — Add linear-x-axis version of the cumulative-PCA-variance plot (operator request; no numbers changed)
- **Operator request** (`human_feedback_07021113.md`): *"for the PCA variance plot, also add a non
  log scale version."*
- Added a Fig-6b block to `experiments/make_plots.py` rendering the **same** per-layer cumulative-
  variance curves (from `results/pca_cumvar.json`) on a **linear** x-axis → new figure
  `plots/pca_cumvar_linear.png` (95% ● / 99% □ crossings marked, same as the log version).
- Embedded it in RESULTS.md's linear-PCA section directly below the existing log-x figure, with a
  one-line reading (L3/6/11 near-vertical at PC 1; L0 gradual).
- **No result numbers changed** — pure visualization of the already-reported d95/d99.
- Renamed `human_feedback_07021113.md` → `human_feedback_07021113.addressed.md`.

## 2026-07-02 — Add cumulative-PCA-variance plot (operator request; no numbers changed)
- **Operator request** (`human_feedback_07021056.md`): *"make a plot of accumulated PCA variance
  and mark the 95% and 99% points."*
- New artifact `results/pca_cumvar.json` (`experiments/pca_cumvar.py`): the full cumulative-
  variance-explained curve (768 values) per layer {0,3,6,9,11}, from the **same** mean-centered
  768×768 covariance eigen-spectrum as the existing `pca_pr.json` table — recomputed only to save
  the full curve (the summary table saved only d90/d95/d99). d95/d99 reproduce the table exactly
  (L0 396/591, L3 6/318, L6 94/479, L9 329/630, L11 5/104).
- New figure `plots/pca_cumvar.png` (added a Fig-6 block to `experiments/make_plots.py`):
  cumulative variance vs #PCs (log x), one curve per layer, with the 95% crossing (●) and 99%
  crossing (□) marked on each. Referenced from RESULTS.md linear-PCA section with a paragraph
  reading the two regimes (L0 broad; L3/6/11 one-dim-dominated).
- **No result numbers changed** — pure visualization of already-reported d95/d99.
- matplotlib was absent from the env again (reset since Iter 8); reinstalled `matplotlib==3.11.0`
  with numpy/torch pinned via constraints (numpy 2.3.3 / torch 2.9.0+cu130 verified unchanged).
- Renamed `human_feedback_07021056.md` → `human_feedback_07021056.addressed.md`.

## 2026-07-02 — Address operator feedback: pooling / Kneedle / MLE / isotropic-Gaussian / "the raw bend" (no numbers changed)
- **Operator questions** (`human_feedback_07010525.md`): (1) *why pooled FineWeb activation & where is
  it pooled?*, (2) *what is Kneedle?*, (3) *what does MLE stand for?*, (4) *why emphasize isotropic
  Gaussian?*, (5) *this raw bend doesn't look like a bend?* — all Methods/interpretation clarity, **no
  result numbers changed.**
- **REPORT.md Methods → Data:** added a "what pooled means and where it happens" paragraph — every
  non-pad token position of every sequence kept as its own point (`hidden_states[L+1][attention_mask]`
  at collection), **not** per-sequence mean-pooling; why (ID/AE are set properties; maximizes kNN
  sample size to 200k).
- **REPORT.md Methods → Metrics:** defined **Kneedle** (Satopää et al. 2011 — max distance below the
  first→last chord on the normalized $\mathrm{FVU}$-vs-$\log_2 k$ curve; reports *where* a curve turns,
  does not certify a turn exists). Expanded **MLE** first use to **Maximum Likelihood Estimation**
  (Levina–Bickel) in both REPORT and RESULTS.
- **REPORT.md Results → validation:** added *why we emphasize "isotropic Gaussian"* — an isotropic
  Gaussian on a flat subspace is the easiest possible input (uniform density, no curvature/anisotropy/
  clustering), so passing it is necessary-not-sufficient; "validated on synthetic linear-Gaussian data"
  is the honest scope, not "validated" full stop.
- **REPORT.md + RESULTS.md AE sections (#5, honest strengthening):** stated explicitly that the raw
  GPU curve is close to a straight line in $\log_2 k$ — only k=2→4 is visibly steep (ΔFVU 0.0202),
  every later doubling is a flat ~0.006–0.009 — so "bend" is generous and k≈8–16 is a **soft** Kneedle
  output, which is precisely why the AE is rated *consistent with* (not evidence *for*) the ID. Agrees
  with the operator's observation; reinforces the existing honest conclusion, no numbers moved.
- Render re-verified via GitHub markdown API: **4 `js-display-math`, 0 `<pre lang="math">`, 0 inline
  backslash-punct hazards** in REPORT.md and RESULTS.md.
- Renamed `human_feedback_07010525.md` → `human_feedback_07010525.addressed.md`.

## 2026-07-01 — Address operator feedback: clarify TwoNN space & F() (no numbers changed)
- **Operator question** (`human_feedback_07010347.md`): *"In TwoNN, where do the two points live?
  What is the F()?"* — answered in REPORT.md Methods and RESULTS.md; no result changed.
- REPORT.md **TwoNN paragraph rewritten** to state explicitly: (1) each reference point and its two
  neighbours live in the **ambient 768-d residual-stream space** $\mathbb{R}^{768}$ (raw activation
  vectors), Euclidean metric, **no projection/embedding** — TwoNN reads local ID off the ambient
  cloud; (2) the two points are each vector's **1st and 2nd nearest neighbours**, with
  $\mu_i = r_2/r_1$; (3) **$F$ is the CDF of the ratios $\mu_i$**, empirically the sorted rank
  $j/(N{+}1)$, theoretically Pareto $F(\mu)=1-\mu^{-d}$. Added a dedicated `math` fence for
  $F(\mu)=1-\mu^{-d}$ (display-eq count 3→4; all 4 verified as `js-display-math`, 0 as `<pre>`).
- Removed an inline `$…$` copy of the log-log slope equation that contained `\!`/`\,` (would have
  been mangled by the rule-8b backslash-strip) — the display fence below it already shows it.
- RESULTS.md nonlinear-estimators paragraph got a one-line answer pointing to REPORT Methods.
- Renamed `human_feedback_07010347.md` → `human_feedback_07010347.addressed.md`.

## 2026-07-01 — Fix inline-math escape stripping (no science changed)
- **Rendering fix only — no numbers moved.** GitHub strips the backslash before punctuation inside
  inline `$…$`, so the Baselines set-notation `\{…\}` (PCA `d_q` bounds and the `{5,10,20,50}` /
  `{0.95,0.99}` sets) rendered without braces. Replaced `\{`/`\}` with `\lbrace`/`\rbrace`
  (backslash-letter, survives). See new `CLAUDE.md` rule **8b**.

## 2026-07-01 — Display-math STILL broke; real fix (no science changed)
- **Rendering fix only — no numbers moved.** The earlier fix (below) was wrong: an indented ```math
  fence inside a bullet still renders as a gray code box (not math) when the bullet has any inline
  `$…$`. Also the three Metrics `$$` blocks (FVU, TwoNN, MLE), though at column 0, were glued to the
  prior line with no blank line and rendered as raw text. Verified via `POST api.github.com/markdown`.
- **Fix:** Metrics `$$` blocks → column-0 ```math fences with blank lines; the two short Baselines
  equations (`PR`, `d_q`) → inline `$…$` (keeps the bullet list intact). API check: 3/3 display
  equations render, 0 code blocks, 0 raw `$$`.
- See rewritten project `CLAUDE.md` rule **8a**: never nest display math in a list item; keep it at
  column 0; verify via the markdown API before committing.

## 2026-06-30 — Figures added; report/results history migrated here (consolidation pass)
- **plots/ was empty; matplotlib was not installed.** Installed `matplotlib==3.11.0` with
  numpy/torch pinned (numpy 2.3.3 / torch 2.9.0+cu130 unchanged — verified). Added
  `experiments/make_plots.py`, which renders every reported quantitative result from the saved
  `results/*.json` (no recompute, headless Agg, savefig+close). New figures:
  - `plots/id_per_layer.png` — nonlinear TwoNN/MLE (centered & standardized, n=50k) vs linear
    PCA d95 vs d_model per layer.
  - `plots/ae_fvu_sweep.png` — AE held-out FVU vs k: CPU vs GPU-raw (seed-mean ± std) vs
    param-matched vs standardized (knee-gone) curves.
  - `plots/ae_marginal_gain.png` — ΔFVU per doubling (raw) showing no plateau.
  - `plots/id_validation.png` — estimator accuracy on synthetic Gaussians (estimated vs true d).
  - `plots/id_by_position.png` — layer-6 ID per token-position bucket.
  - `plots/id_diagnostics.png` — bootstrap 95% CIs + naive-vs-robust (self-masking) bars.
  RESULTS.md and REPORT.md now reference these figures; **no numbers changed.**
- **Consolidation per CLAUDE.md rules 6–7:** moved the "What changed after review" /
  "Status note" version-history blocks OUT of REPORT.md (a curated deliverable) into the dated
  entries below. REPORT.md/RESULTS.md now read as clean current-best documents; all change
  history lives here.

## 2026-06-23 — Codex review `CODEX_REVIEW_20260623T001526Z` addressed (Iter 7)
- **ID headline band widened 11–13 → 11–15.** New artifact `results/id_diagnostics.json`
  (GPU, `experiments/id_diagnostics.py`):
  - (#4) Duplicate / self-masking: 92/50k exact duplicate rows (0.18%); explicit self-index
    masking moves TwoNN 0.00 / MLE +0.17 → ID is **not** a duplicate/self-masking artifact.
  - (rec#5) Bootstrap CIs (B=20 disjoint draws, n=20k): TwoNN 12.71 ± 0.13, MLE 15.18 ± 0.09.
    Sampling CI ±0.1 is tight; the 11–13→11–15 spread is finite-sample n-dependence
    (n=50k → n=20k), not noise.
- **Documented layer-11 post-final-layernorm caveat (#5):** `hidden_states[11+1]` carries
  `ln_f`, so all layer-11 numbers are post-LN; layers 0/3/6/9 (and the layer-6 headline) are
  genuine interior block outputs and unaffected.
- **Wording fixes:** AE reframed "weak corroboration" → "raw-variance reconstruction artifact
  consistent with low ID" (#1); param-matched section now states outer width h1 varies and
  drops "k is the only varying channel" (#2); token-position headline "stable" → "low across
  buckets; TwoNN stable, MLE estimator-dependent" (#6); filled matched-AE train_FVU for
  k=128/256 from the existing JSON (#3).

## 2026-06-21 — Codex review `CODEX_REVIEW_20260621T031919Z` addressed (Iter 6)
- **Parameter-matched AE sweep added** (Codex concern #4 / step #1): new artifacts
  `results/ae_results_matched.json`, `results/ae_matched_param_counts.json`. Total params held
  to 0.087% spread (1024 params) by trading outer hidden width h1 (576→512) against bottleneck
  width k. The low-k bend **survives matching** (within ≤0.0021 of unmatched at every k) → the
  bend is not a parameter-count artifact.
- Scoped "stable across position" → "roughly similar with estimator-dependent variation";
  scoped the synthetic validation to linear-Gaussian data only; reworded "8.3× train budget" →
  "8.3× steps / 16.7× examples"; reported the uncontrolled AE param count with confound
  direction; cleaned stale duplicated operator blocks from PLAN.md.

## 2026-06-20/21 — Operator review `REVIEW.md` addressed (Iter 5)
- **Retracted the strong original framing** "AE and ID converge → demonstrated low-dimensional
  curved manifold." Reframed the AE as weak, preprocessing-sensitive corroboration. Added
  standardized + multi-seed GPU AE sweeps (`results/ae_results_gpu.json`,
  `results/ae_results_gpu_v2.json`) proving the knee vanishes under standardization and is
  seed-stable (std ≤0.0018). AE FVU floor dropped 0.051 (CPU 1200 steps) → 0.033 (GPU 10000
  steps) at k=256; CPU run was genuinely under-trained.
- **Fixed false "everywhere" claims**, now layer-scoped: "standardization changes ID by <2
  everywhere" (false at L11); "TwoNN/MLE agree within ~3 everywhere" (false at L11, gap ~5.2);
  "nonlinear an order of magnitude below d95 across layers" (false at L3/L11, where d95
  collapses to 5–6).
- Corrected "curve flattens after k≈16" (no plateau). Saved the synthetic-validation artifact
  (`results/id_validation.json`). Added token-position-stratified ID
  (`results/id_by_position.json`) and scoped the conclusion to "this pooled FineWeb sample."

## 2026-06-20 — Initial deliverables (S1–S4)
- S1 collect activations (200k fp16 vectors/layer, layers {0,3,6,9,11}, pooled tokens,
  FineWeb REST API + GPT-2). S2 linear PCA-PR + nonlinear TwoNN/MLE
  (`results/pca_pr.json`, `results/id_nonlinear.json`). S3 AE bottleneck sweep layer 6
  (`results/ae_results.json`). S4 first REPORT.md comparing AE elbow vs nonlinear ID vs PCA
  d95 vs d_model.
