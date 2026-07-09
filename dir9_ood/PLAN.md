# PLAN — Direction #9: Plateau-ness as an OOD / Anomaly Detector

> The agent REWRITES "Current status" and "Next step" and ticks the stage boxes every iteration.
> Disk (this file + JOURNAL.md + RESULTS.md) is the only memory. All paths are relative to this folder.

## Success criterion (definition of "done")
A fair AUROC comparison of >=2 plateau-score variants (each evaluated at the residual-layer sweep {3,6,9} and input-space) against >=3 baselines (activation L2 norm, Mahalanobis distance, Maximum Softmax Probability) on >=1 OOD task — in RESULTS.md — plus REPORT.md giving a plain verdict on whether plateau-ness beats the baselines, and whether measuring internally (residual stream) beats the simpler input-space signal.

**A null result (it does not beat them) is complete and acceptable.** When done, create an empty `STOP` file.

## Fallback (if time runs short)
Minimum acceptable: one plateau variant (perturbation-sensitivity) computed at a single point (`resid_post`@6) vs the 3 baselines on the self-contained OOD setup (held-out FineWeb vs random/shuffled tokens), AUROC in RESULTS.md. Reserve the final 20 min to finalize + STOP.

## Setup (fixed)
- Model: GPT-2 small (124M). The plateau/robustness score is computed at a **configurable measurement point** — do NOT hardcode the residual stream. Defaults: a residual-stream layer sweep (`resid_post` at layers {3,6,9}, reporting per-layer AUROC) using the last-token (or mean-over-positions) activation, AND an **input-space** variant (perturb token embeddings, measure output sensitivity — no internal hook). The residual stream is the choice faithful to the prior plateau characterization; input-space is the simpler signal it must beat to be interesting.
- In-distribution data: FineWeb text.
- **Shared hardware + time limits live in `../BUDGET.md` — read it every iteration.** You share one RTX 3090, 16 GB RAM, and 4 CPU with the other agent, so stay within your half: cap VRAM (`set_per_process_memory_fraction`), keep perturbation batches and direction counts modest, halve on OOM.

## Stages (checklist)
- [x] **S1 — plateau score (measurement-point-agnostic).** Implement `experiments/plateau_score.py` so the measurement point is a PARAMETER (a residual layer's `resid_post`, or the input embeddings) — not hardcoded. Two scalar variants: (a) **perturbation-sensitivity** — N random unit directions at the chosen point, sweep magnitudes eps, continue the forward pass from the perturbed state, measure KL of the next-token distribution vs unperturbed; summarize as radius eps* at a KL threshold or mean KL at fixed small eps (flatter = more in-distribution); (b) **gradient/Jacobian norm** of the output w.r.t. the chosen point (lower = flatter; generalizes to any point cheaply via autograd, no eps sweep). Default measurement points to evaluate: residual stream at layers {3,6,9} AND input-space (token embeddings). Keep cheap: 16-32 directions, a few eps, subsampled positions, batched. Sanity-check it separates FineWeb vs random tokens at all (quick AUROC).
- [x] **S2 — OOD task + baselines.** **If `cupbearer` is already importable** in this env (the PyTorch GitHub build, pre-installed by the user), you may use it for a proper mechanistic-anomaly benchmark and its baselines. **Otherwise use the self-contained setup** — and do NOT install cupbearer yourself (only the PyPI `0.0.1` JAX build is what `pip` would auto-pick, and it breaks the cluster torch). Self-contained: ID = held-out FineWeb; OOD (2-3 sets) = random/shuffled tokens, a different domain (e.g. code), optional char-level corruption. Baselines (implement regardless): activation L2 norm; Mahalanobis distance to a Gaussian fit on ID activations; Maximum Softmax Probability. Whatever the path, do not `pip install` any deep-learning framework — use the already-installed env.
- [x] **S3 — evaluate + report.** Write `results/auroc_table.csv` `[task, ood_set, method, measurement_point, auroc]` — one row per (OOD set × method × measurement point), covering both plateau variants at each residual layer {3,6,9} AND input-space, plus the 3 baselines. ROC and score-distribution plots under `results/plots/`. Write REPORT.md with a per-OOD-set verdict that explicitly answers: (i) does any plateau variant beat the baselines, and (ii) does the residual-stream (internal-activation) plateau signal beat the simpler input-space sensitivity — i.e. is there value in measuring internally. Create `STOP`.

## Out of scope (do NOT)
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, or flax** — they downgrade and break the cluster's CUDA-13 torch. Use only the already-installed env (torch + HuggingFace `transformers` + numpy/sklearn/matplotlib). If a needed pure-python package is missing, install it with `--no-deps`.
- Don't make the score differentiable or use it for steering/correction — separate direction.
- Don't drift into other directions.

## On-track check (required every iteration)
End each JOURNAL.md entry with one line: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status
**DONE (iter 10, 2026-07-09) — NEW operator feedback addressed: randomly-sampled residual points +
GPT-2 scaling.** Reset into a nominally-finished project but found NO `STOP` and a NEW unaddressed
feedback file `human_feedback_07082204.md`: *"try GPT-2 XL, and OOD detection with randomly sampled
points in the residual stream."* A prior partial iteration had written `experiments/rand_points.py` +
`make_randpoints_plot.py` and launched the sweep (gpt2 done, gpt2-large mid-run); this iteration let the
run finish, verified `results/auroc_randpoints.csv` (78 rows), generated both plots, and curated the
deliverables. **GPT-2 XL is not in the offline cache → used gpt2-large (774M, ~6×)** for the scale test
(assumption logged; rejected alternative = block on the missing XL weights). **Verdict UNCHANGED and
strengthened at scale:** the genuine dispersion signal `rand-points-disp` is weak/reversed (random 0.52 /
shuffled 0.27 / code 0.71 for gpt2; 0.44 / 0.26 / 0.60 for gpt2-large), losing to Mahalanobis on code
(0.913 / 0.842) and MSP on synthetic; `rand-points-ent` is a confidence baseline (near-perfect on
random/shuffled but collapses on code like MSP, 0.359 / 0.326). Added a section to RESULTS.md + REPORT.md
(Methods equation, gpt2-large Data note, Obs/Interp/Limitations/Next-check, both plots); REPORT
display-math clean (9/9). Renamed feedback `*.addressed.md`; `STOP` written. _History below:_

**DONE (iter 9, 2026-07-07) — VERIFIED FINISHED STATE + RE-CREATED MISSING STOP.** Reset into a
nominally-finished project with **no `STOP` on disk** (recurring finalization miss) but **no open
feedback** (all six review/human files are `*.addressed.md`). Verified completeness before finalizing:
all 9 figures cited by RESULTS.md/REPORT.md exist under `results/plots/`; all CSV/npz artifacts present;
REPORT.md display-math renders clean via the GitHub markdown API (8/8 `js-display-math`, 0 code
fallbacks); RESULTS.md is current-best (no version history). No science outstanding — the negative result
stands (random plateau 0.734<MSP 0.932; shuffled 0.534<MSP 0.872; code 0.649<cup-RMD@resid6 0.918) and
iter-8's eps-scan already ruled out "wrong epsilon". Created `STOP`. Deliverables unchanged → no CHANGELOG
entry. _History below:_

**DONE (iter 8, 2026-07-02) — EPSILON SCAN + STOP.** A new operator feedback file
`human_feedback_07010438.md` asked whether scanning the plateau-perturbation magnitude $\epsilon$ (fixed at
6) changes the picture. A partial prior iteration had already run `experiments/eps_scan.py` (outputs on
disk) but never finalized; iter 8 completed it. Sweep $\epsilon\in\{0.25..24\}$ × {input,resid3,6,9} on the
canonical split → `results/auroc_perturbation_eps.csv` (120 rows) + `results/plots/perturbation_eps_scan.png`
(eps=6 column reproduces `auroc_table.csv` exactly). **Findings:** residual points are eps-insensitive
(<0.05 across two decades); input-space eps=6 was a poor choice (random@input ~0.87 for eps≤2, 0.44 at
eps=6, reverses to 0.12 at eps≥8); best oracle-eps plateau-perturbation still loses everywhere (random 0.873
< MSP 0.932; shuffled 0.554 < MSP 0.872; code 0.614 < cup-RMD 0.918). **Verdict unchanged/strengthened.**
Added an "Epsilon sensitivity" section to RESULTS.md + a Results subsection/Methods note to REPORT.md (both
embed the new plot; display-math check 8/8 clean); main fixed-eps table unmodified. Feedback renamed
`*.addressed.md`; `STOP` written. _History below:_

**DONE (iter 7, 2026-07-01) — DOCUMENTATION CLARITY PASS + STOP.** A new operator feedback file
`human_feedback_07011019.md` arrived asking three *documentation* questions (no recomputation): name the
best baseline/plateau variant in the headline plot, define "canonical split", and explain why MSP detects
OOD. Addressed all three as a presentation-only pass — numbers unchanged (random plateau 0.734<MSP 0.932;
shuffled 0.534<MSP 0.872; code 0.649<cup-RMD@resid6 0.918): regenerated `summary_best_per_set.png` with
each bar annotated by its exact `method@point` (new reproducible `experiments/make_summary_plot.py`
deriving best-per-set from `auroc_table.csv`); expanded the figure caption in RESULTS.md/REPORT.md;
added a "canonical split" definition (REPORT Methods + RESULTS gloss) and a "why MSP detects OOD" note
(REPORT baselines + RESULTS gloss). `check_md.py` clean (only the known results/-prefix WARN false
positive, no ERROR). Feedback file renamed `*.addressed.md`; `STOP` re-created. _History below:_

**DONE (iter 6, 2026-06-30) — FINAL CONSOLIDATION + STOP.** No open feedback (all `*.addressed.md`);
science complete and unchanged. This iteration was CLAUDE.md deliverable hygiene only: (1) populated
the previously-empty `CHANGELOG.md` with the full dated history (iter1→iter6, old→new numbers); (2)
rewrote `REPORT.md` to the required `Summary → Methods → Results → Conclusion → Limitations` structure
with Data/Model/Layer and **`$$LaTeX$$` equations for every metric and baseline** + the AUROC
estimator; (3) stripped the inline version-history blockquotes from `RESULTS.md`/`REPORT.md` (rule 6)
into CHANGELOG; (4) added headline figure `results/plots/summary_best_per_set.png`. Numbers unchanged
(canonical split): random plateau 0.734<MSP 0.932; shuffled 0.534<MSP 0.872; code 0.649<cup-RMD@resid6
0.918. `STOP` written. _History below:_

**DONE (iter 5, 2026-06-23) — Codex review `CODEX_REVIEW_20260622T230658Z.md` addressed: CANONICAL
SPLIT.** That review confirmed the negative conclusion is faithful but flagged comparison hygiene —
chiefly (High) that the plateau table and the real-cupbearer table used *different* ID splits (first-N
vs shuffled `randperm(seed=7)`). Fixed: reran `plateau_v2.py` on the **exact** `randperm(seed=7)` split
that `extract_acts.py` used for the cupbearer acts (indices saved to `results/split/canonical_split.npz`),
so `auroc_table.csv` and `auroc_cupbearer.csv` are now strictly apples-to-apples on ONE canonical ID
split. **Verified at the value level** (new plateau idtest acts = precomputed cupbearer acts to
max|Δ|=2.5e-5; on the unified split vendored cup-RMD@resid6 code 0.918 = real 0.918, naive-maha 0.913 =
real cup-maha 0.913). Numbers shifted <0.04 (within ±0.035 noise); **verdict unchanged — plateau loses
on every OOD set** (random 0.734<MSP 0.932; shuffled 0.534<MSP 0.872; code 0.649<cup-RMD@resid6 0.918).
Also: cup-QUE scoped (vendored=transductive/superseded, real=consistent-but-not-definitive-QUE-protocol),
stale transductive caveat scoped to vendored rows, stale operator-review block removed from this file.
The cupbearer CSV was not recomputed (its acts are byte-identical/canonical — re-run would only refresh a
log). RESULTS.md + REPORT.md updated; `CODEX_REVIEW_20260622T230658Z.md` renamed `*.addressed.md`; STOP
created. Env note: base env had again lost `transformers`/`tokenizers`/`httpx` — reinstalled with
`--no-deps` (torch 2.9.0+cu130 / numpy 2.3.3 untouched, verified). _History below:_

**DONE (iter 4) — new operator follow-up addressed: REAL cupbearer run in an isolated env.** A new
`human_feedback.md` ("create a new environment to evaluate OOD with cupbearer") arrived after iter 3,
so STOP had (correctly) not persisted. iter 4 built an **isolated conda env `cupenv`** (own numpy
1.26.4 + torch 2.9.0+cu130 + torchvision cu130 + transformers 5.12.1 + datasets 5.0.0) and installed
**cupbearer editable from the GitHub clone** (not PyPI); verified GPU compat (A10/CUDA 13.2 ≥ 13.0,
CUDA works in cupenv) and that the **shared base env is untouched** (numpy 2.3.3 / torch 2.9.0+cu130).
Ran the **genuine** cupbearer detectors (`cup-maha/RMD/QUE/spectral`) on the precomputed GPT-2
activations → `results/auroc_cupbearer.csv` (48 rows), and compared to iter-2's vendored math →
`results/cup_real_vs_vendored.csv`. **Findings:** vendored cup-RMD was faithful (real@resid6 code
0.918 vs vendored 0.917); vendored cup-QUE understated the real detector (real 0.910 vs 0.572 on
code). **Verdict unchanged/strengthened:** plateau-ness loses to standard AND genuine-cupbearer
baselines on every OOD set (code: cup-RMD 0.918 / cup-maha 0.913 / cup-QUE 0.910 ≫ best plateau 0.628).
Also addressed a **second** new feedback file `CODEX_REVIEW_20260621T031213Z.md` (rebuild broken cupenv,
run the real package, compare to vendored, fix the cup-QUE per-set-covariance protocol) — all done; the
real `QuantumEntropyDetector` is fit once on ID and applied uniformly (consistent scoring). Both files
renamed `*.addressed.md`; STOP created. _History below:_

**DONE (iter 3) — STOP file now actually created.** iter 2 finished all work but the `STOP` file was
never written (same miss as iter 1). iter 3 verified the on-disk outputs match the journal
(auroc_table.csv = 87 rows, scores_full.npz, 7 plots, RESULTS.md, REPORT.md, ENV_NOTES.md, both
feedback files `*.addressed.md`), reconciled a stale "do NOT STOP" block in PLAN.md, and created STOP.
Prior (iter 2): operator review fully addressed; success criterion met with a CORRECTED negative
result. Reran on **GPU (NVIDIA A10, CUDA 13.2 — iter-1's V100/CPU-only story was
false)** at N=200/seq=64 via `experiments/plateau_v2.py` (270 s). Methods: genuine plateau metric
`plateau-jacFrob` (Jacobian-Frobenius), `plateau-perturbation`, control `selfNLL-grad` (iter-1's
mislabeled "jacobian"); baselines MSP, L2, Mahalanobis (1000-seq fit), **cupbearer `cup-RMD` + `cup-QUE`
(vendored from the GitHub repo)**; OOD {random, shuffled, code}. Outputs: results/auroc_table.csv (87
rows), scores_full.npz, plots/*.png, RESULTS.md, REPORT.md, experiments/cupbearer_helpers.py, ENV_NOTES.md.
**Verdict (corrected):** plateau-ness does NOT beat the baselines on ANY OOD set. The genuine jacFrob
is weak (≤0.725, reversed on shuffled/deep layers); iter-1's "win" was the MSP-adjacent selfNLL-grad
(confirmed). MSP wins synthetic (0.94/0.90); cupbearer cup-RMD@resid6 (0.917) & well-fit Mahalanobis
(0.911) win the code domain shift where MSP collapses (0.38). Input-space ≥ residual for jacFrob → no
value measuring internally. Clean NEGATIVE result (acceptable per plan). Feedback files renamed
*.addressed.md.

## Next step
None — project complete and finalized; deliverables are CLAUDE.md-compliant (clean current-best,
LaTeX Methods, full CHANGELOG history) and `STOP` is written. The iter-10 operator request
(`human_feedback_07082204.md`: GPT-2 XL / randomly-sampled residual points) is addressed and renamed
`*.addressed.md`; the negative result now also holds at ~6× scale (gpt2-large). If reopened by NEW
feedback: run the actual GPT-2 XL if its weights reach the offline cache, and test the collapse-on-code
pattern on a non-code real domain shift. Prior finalized state: **all** operator feedback addressed (incl.
the iter-5 Codex `CODEX_REVIEW_20260622T230658Z.md` canonical-split hygiene review, the iter-7
`human_feedback_07011019.md` documentation-clarity requests, and the iter-8
`human_feedback_07010438.md` epsilon-scan request), `STOP` written. If reopened, the
remaining ideas are all enhancements, not corrections: run cupbearer's **full task/data harness**
end-to-end in `cupenv` (not just its detectors on precomputed acts) and refresh `auroc_cupbearer.csv`
with a clean log; a multi-model sweep (Pythia/larger GPT-2); and a non-code domain shift to test whether
"plateau weak, Mahalanobis/cup-RMD/cup-QUE strong on real shift" generalises.

## Operator review — ADDRESSED (iter 2, 2026-06-21)
All points in CODEX_REVIEW.md + human_feedback.md were addressed; files renamed
`CODEX_REVIEW.addressed.md` / `human_feedback.addressed.md`. How each was handled is logged in
JOURNAL.md (iter 2) and REPORT.md ("How each operator-review point was addressed"):
- CODEX H1 false-environment → verified A10/CUDA-works, corrected ENV_NOTES, reran on GPU.
- CODEX H2 mislabeled jacobian → added genuine `plateau-jacFrob`, renamed old to `selfNLL-grad`.
- CODEX M weak benchmark → added `code` real-domain-shift OOD + cupbearer detectors.
- CODEX M "jacobian cheaper" → claim dropped. CODEX M underpowered Mahalanobis → 1000-seq fit.
- human cupbearer-from-GitHub → vendored its detector math (cup-RMD/cup-QUE); full pkg uninstallable
  (numpy<2 vs shared 2.3.3). human shared-pkgs → torch/numpy/cuda versions untouched (verified).

_(The original "address before continuing — do NOT STOP" instruction block has been fully satisfied:
both feedback files are renamed `*.addressed.md` on disk, every point logged in JOURNAL.md iter 2,
and RESULTS.md/REPORT.md/checklist updated. Block removed iter 3 to end the STOP contradiction. The
later Codex reviews (`*_20260621T031213Z` and `*_20260622T230658Z`) are likewise addressed and renamed
`*.addressed.md`; see JOURNAL iters 4–5.)_

## Consolidation pass — do this, then STOP
Per CLAUDE.md: rewrite REPORT.md and RESULTS.md to CURRENT-BEST only — remove all version history,
"changed after review" notes, and any weaker/superseded run of an experiment that a stronger run
replaces. Move everything you remove into CHANGELOG.md as dated entries (old -> new numbers). Ensure
REPORT.md has the Methods section: Data/Model/Layer, and every metric + baseline defined with $$LaTeX$$.
