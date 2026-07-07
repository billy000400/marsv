# JOURNAL — Direction 8: Plateau-ness as a test for SAE activation validity

Append-only. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-06-30 — S1/S2/S3/S4: env, SAE load, smoke, Stage A reproduction
**Did.** Found the env had NO `transformers`/`tokenizers`/`safetensors`/`huggingface_hub`
(PLAN.md assumed they were present — they were not; `matplotlib` 3.11 IS present, also
contra PLAN). Network is reachable, so installed them with `pip --no-deps` (+`regex`,
`httpx`, `hf-xet`), pinning `tokenizers==0.22.2` to satisfy transformers 5.12.1; left
torch/CUDA untouched per PLAN. Downloaded the SAE `jbloom/GPT2-Small-SAEs-Reformatted/
blocks.6.hook_resid_pre` (d_sae=24576). Wrote `experiments/smoke_plateau.py` doing the
full pipeline: capture real resid_pre@6 last-token acts (hook on block 5), build
real/recon/naive/norm_rand conditions, run the in-context `plateau_auc_low` sweep
(forward-hook overwrite of the last-token residual, KL of last-token next-token dist),
paired-bootstrap CIs, plot. Ran smoke (N=24) then full (N=200, 8 dirs).

**Learned.**
- Encoder b_dec convention: subtracting b_dec before encode gives recon err 27.7 vs 380.8
  → `apply_b_dec_to_input=True`. Recorded.
- Stage A reproduced: real 0.200, recon 0.162, naive 0.066, norm_rand 0.035. Paired gaps
  vs real all exclude 0. real ≳ recon ≫ naive > norm_rand.
- **Not a norm artifact:** norm_rand is norm-matched to real (79.5) yet flattest;
  pooled Spearman(plateau, norm)=+0.06.
- **But** Spearman(plateau, dist_to_source)=−0.82 — the gap tracks closeness to a real
  activation. This is the central open confound (decision table: "gap disappears after
  source-distance matching → plateau reflects closeness to real, not special validity").

**Assumptions logged (loop mode, could not ask).** (1) Used resid_pre@6 SAE and switched
the hook consistently (block-5 output) rather than hunting a resid_post@6 SAE — rejected
alt: jbloom OAI-128k resid-post repo (larger, 128k feats, heavier for the 5 GB budget; no
blocks.6 file surfaced in listing). (2) τ calibrated on all real acts for this smoke;
proper held-out real calibration split deferred to Stage B/D. (3) naive coefs sampled from
pooled empirical active-coefficient marginal — standard independent-composition baseline.

**Next step.** Stage B: distance-to-source matched comparison (the decisive control given
ρ=−0.82) plus L0/coef-RMS matching and a real held-out τ split; add cycle-error and
fixed-radius Jacobian covariates; regress plateau on covariates and test whether the
real-vs-naive gap survives distance matching.

On track? yes — Stage A (M1) done, ~35% of plan; blocker: distance-to-source confound must be resolved in Stage B before any independent-diagnostic claim.

## 2026-06-30 — S5: Stage B distance-to-source matched control (decisive H2 test)
**Did.** Built `experiments/stageB_distance.py`. Core idea: the Stage A confound was
Spearman(plateau, dist_to_source)=−0.82, and recon/naive/norm_rand sit at increasing
distances (25/64/113). To break the confound I built a **random-displacement reference**
family `iso_displace = x_real + δ·d` (isotropic unit d, so distance ≡ δ exactly) at
δ∈{15,30,60,120}, tracing the plateau(distance) curve for purely random off-manifold
displacement of a real activation. Then I tested whether SAE-decoded conditions (recon,
naive, sparse_match) plateau ABOVE that reference at their own (matched) distance. Also
added `sparse_match` (naive with k=source L0 and coefs rescaled to source coef-RMS) to test
the sparsity/coefficient-matching decision-table row, and switched τ to a proper **held-out
real split** (calibrate on sources <N/2, score on ≥N/2). N=200, 6 dirs, N_eval=100.

**Learned (decision-grade).**
- iso_displace reference plateau falls monotonically with distance: 0.184/0.173/0.128/0.078
  at δ=15/30/60/120. So distance-to-source ALONE reproduces most of the Stage A ordering.
- Distance-matched residual (plateau − reference @ its distance), 95% CI:
  recon −0.016 [−0.021,−0.003]; naive −0.058 [−0.065,−0.053]; sparse_match −0.063
  [−0.067,−0.049]. **No SAE-decoded condition is ABOVE the random-displacement curve.**
- recon ≈ on the curve (residual ~0): its plateau advantage is just closeness-to-real.
- naive & sparse_match sit BELOW: at equal distance from a real activation, an SAE
  independent composition is LESS flat than a random point — actively more downstream-
  sensitive. Sparsity/coef matching does NOT recover plateau (sparse_match ≈ naive).
- **H2 verdict:** the Stage A real/recon-vs-naive plateau gap does NOT survive distance
  matching as an SAE-validity signal. Decision-table row hit: "gap disappears after
  source-distance matching → plateau reflects closeness to real, not special compatibility."
  Plateau-ness = closeness-to-real proxy + direction-family effect, not an independent SAE
  diagnostic. Consistent with Direction 9's plateau-as-OOD weakness.

**Assumptions logged (loop mode).** (1) iso reference uses ONE random displacement direction
per (source,δ); the perturbation sweep still uses 6 directions — adequate for a median
reference curve, more displacement dirs deferred. (2) Reference interpolated in log-distance,
clamped to [15,120]; recon(25)/naive(64) are interior so no extrapolation. (3) Kept the
Stage-A naive construction so the comparison is apples-to-apples; the additional below-random
deficit is a real finding, not a construction change.

**Next step.** This largely settles M2 (H2) negatively for the SAE-validity claim. Highest-
value remaining: (a) direction-family robustness — does the naive below-random deficit persist
along SAE-decoder-feature perturbation directions, or is it isotropic-only (Stage B direction
control)? (b) Stage D — does plateau add ANY held-out predictive value for output-KL validity
beyond distance/norm baselines? Given the negative H2, Stage D is the gate for any residual
positive claim. If time short next iter, finalize REPORT.md with the A+B negative verdict.

On track? yes — Stage B (M2) done & decisive (~60% of plan); H2 negative: plateau is a
closeness-to-real proxy, not an independent SAE-validity diagnostic. Blocker: none; next is
direction-family robustness or Stage D predictive check.

## 2026-06-30 — S7: Stage D downstream-validity prediction gate (H4/M4) + finalize REPORT
**Did.** Built `experiments/stageD_validity.py`. Independent validity target
`output_kl = KL(p_real || p_candidate)` (in-context last-token; low=valid). Pooled 7 candidate
conditions (recon, naive, sparse_match, iso15/30/60/120) × N=200 = 1400 rows, split BY SOURCE
(train <N/2, test ≥N/2), linear least squares predicting log10 output_kl, held-out test R².
Crucially added a single fixed-radius local-sensitivity baseline `locsens = log10 mean-KL@r=0.02`
(= D6 plateau_kl) as the discriminator between "robustness" and "interpretability validity".

**Learned (decision-grade).**
- Beyond {dist,norm}: plateau ADDS predictive value — ΔR²=+0.073, partial Spearman −0.65.
  So naively H4 looks supported (and this replicates D6's "plateau predicts downstream KL
  beyond movement distance").
- Beyond {dist,norm,locsens}: plateau adds ~NOTHING — ΔR²=+0.005, partial Spearman −0.16.
  locsens alone has marginal Spearman +0.84 ≈ plateau's −0.85. So plateau's downstream-validity
  signal IS local sensitivity; the plateau-AUC shape carries no extra info beyond one
  fixed-radius KL.
- **H4 verdict:** plateau-ness measures LOCAL ROBUSTNESS, not SAE interpretability validity
  (decision-table row: "metric adds little beyond local sensitivity → robustness, not validity").
- **Project-level null COMPLETE & names the failing notion:** plateau-ness = closeness-to-real
  (Stage B) + local robustness (Stage D). Of {provenance, OOD, downstream-invalidity, mere local
  robustness}, the answer is **mere local robustness** + distance-to-real. Matches D9 (OOD weak)
  and D6 (local-sensitivity, reward-hackable). This satisfies the success criterion.

**Did (finalize).** Curated RESULTS.md (added Stage D, rewrote verdict to project-level null,
current-best only). WROTE REPORT.md (was absent): Summary→Methods→Results→Conclusion; Methods
gives Data/Model/Layer/SAE and defines every metric+baseline with $$LaTeX$$ (plateau_auc_low,
KL path, output_kl, distance, norm, iso_displace reference R(δ), locsens). Embedded
plots/plateau_stageA/B/D.png. Appended CHANGELOG. Created STOP.

**Assumptions logged (loop mode).** (1) Used isotropic perturbation directions for plateau in
Stage D (primary family); SAE-decoder-direction robustness left as a scoped open caveat — given
the local-sensitivity result it would scope, not overturn, the null. (2) locsens uses a single
radius r=0.02 (the held-out τ radius / D6 comparison point) as the canonical local-sensitivity
scalar; alternative jacFrob deferred. (3) Linear least-squares predictor (pure numpy, no sklearn)
on log10 output_kl; standardized on train only.

**Next step.** Project criterion met (null identifying "mere local robustness"); finalized and
STOPped. If reopened, highest-value adds: SAE-decoder-direction robustness for the Stage B
below-random deficit, then Stage C cycle-consistent/co-occurrence codes — neither expected to
overturn the local-sensitivity verdict.

On track? yes — Stages A(M1)+B(M2)+D(M4) done & decisive (~90% of plan; C/E intentionally
skipped as the null is complete and names its cause). Project-level null finalized; STOP created.

## 2026-07-01 (iter) — Stage B-dir curated: direction-family robustness of the null

**Context on entry.** PLAN said FINALIZED/STOP, but STOP was absent (loop wrapper removed it) and
git showed the last iteration had created+committed `experiments/stageB_directions.py` +
`stageB_dir_*` artifacts **without curating them into RESULTS.md/REPORT.md** — both still listed
"one direction family (isotropic)" and "SAE-decoder-direction robustness (not run)" as the biggest
open caveat. So the deliverables did not reflect completed work (violates CLAUDE.md §6). The
committed run was also the SMOKE config (N=24, 2 dirs, N_eval=12 — wide CIs).

**Did.** Re-ran `stageB_directions.py --full` (N=200, N_eval=100, 8 dirs, 3 perturbation-direction
families) — ~9 min, within GPU/RAM budget. Curated the result into RESULTS.md (new "Stage B-dir"
section + updated H2 verdict + Scope paragraph) and REPORT.md (new Results subsection + Methods
note + updated Conclusion scope). Appended CHANGELOG. Regenerated `plots/plateau_stageB_dir.png`.

**Learned (decision-grade).** The Stage B "naive/sparse below random-displacement" deficit is
**direction-family robust**. Distance-matched residual ρ (median [95% CI]):
- iso: recon −0.015 [−0.025,+0.003], naive −0.061 [−0.068,−0.057], sparse −0.062 [−0.069,−0.052]
- sae_single: recon −0.016 [−0.029,−0.003], naive −0.066 [−0.071,−0.058], sparse −0.062
- sae_sparse: recon −0.015 [−0.032,+0.006], naive −0.077 [−0.084,−0.065], sparse −0.071
Under every family recon sits ~on the random curve and naive/sparse sit clearly BELOW it; the
naive deficit is if anything LARGER along SAE decoder directions (sae_sparse −0.077 vs iso −0.061)
— the opposite of SAE-specific plateau validity. iso here also replicates primary Stage B
(recon −0.015 vs −0.016; naive −0.061 vs −0.058). Pooled Spearman(plateau,dist) −0.64/−0.60/−0.62.
So the project null is not an isotropic-direction artifact.

**Assumptions logged (loop mode).** (1) sae_single = single unit-normed decoder column, j drawn
from real-active features by frequency; sae_sparse = normalized signed sum of 8 active columns —
standard "feature direction" choices; alternatives (encoder-row directions, gradient directions)
not tested. (2) τ recalibrated per family on the held-out real split (matches Stage B protocol).

**Next step.** Direction-family robustness now confirmed and curated. Project null complete and
direction-robust; re-creating STOP. If reopened: Stage C (cycle/co-occurrence codes), then Stage E
(alternate layer) — expected to scope, not overturn, the local-sensitivity verdict.

On track? yes — biggest open caveat (direction-family robustness) closed & curated to current-best;
Stages A+B+B-dir+D done & decisive; project-level null direction-robust and finalized. STOP created.

## 2026-07-02 (iter) — Stage C: improved synthetic codes (H3/M3) — the last untested hypothesis

**Context on entry.** Project was FINALIZED (A+B+B-dir+D decisive null), STOP absent (wrapper
removes it each iter). ~240 min budget, so did one focused *scientific* iteration rather than
re-finalizing: closed the only remaining untested hypothesis, **H3** (Stage C, "not run" in PLAN).

**Did.** Built `experiments/stageC_synthetic.py` on the Stage B distance-matched scaffold (same
iso_displace random-displacement reference R(δ), held-out τ, N=200/N_eval=100, 6 dirs). Three new
conditions asking whether *higher-order* synthetic codes plateau ABOVE R(δ) at matched distance:
`cooc` (real support + marginal coefs — isolates support co-occurrence over naive), `cycle_consistent`
(naive filtered to encode–decode self-consistency below real-code-p75 cycle error τ_cyc=0.342), and
`cooc_full` (genuine real-derived code = recon of another real example; positive control, reuses recon
curves). First cycle attempt iterated z←encode(decode(z)) to a fixed point and DIVERGED (encode–decode
is expansive off-manifold, dist→3e5); replaced with PLAN's filter definition (measured feasibility
first: real cycle relerr median 0.239, naive 0.780, filter passes 0.56% of naive at real-p75). ~5 min.

**Learned (decision-grade). H3 NEGATIVE for constructible codes.** Distance-matched residual ρ_c
(median [95% CI], >0 = flatter than random at equal distance):
- cooc −0.044 [−0.049,−0.036], cycle_consistent −0.043 [−0.049,−0.040] — both clearly BELOW random,
  only marginally above naive (−0.054). Neither support co-occurrence nor encode–decode
  self-consistency recovers plateau.
- cooc_full +0.043 [+0.035,+0.056] — the ONLY condition ABOVE random: a genuine real-derived
  activation is flatter than a random displacement even at a large distance from the paired source.
- recon −0.012 / naive −0.054 replicate Stage B (−0.016 / −0.058), confirming the framework.
- **Reading:** the missing ingredient is real-activation MANIFOLD MEMBERSHIP, not latent-code
  marginal realism or self-consistency — matches the a-priori H3 null. cooc_full above random is the
  positive control Stage B lacked (plateau tracks genuine-activation validity, not merely distance to
  THE source), and is consistent with Stage D (real activations occupy locally-robust regions).

**Did (curate).** Inserted Stage C section into RESULTS.md + REPORT.md (before Stage D), added
condition rows to REPORT Methods table, added H3 to both verdict lists + Summary, updated Scope
(Stage C "not run"→"tested, does not overturn null"). Verified REPORT via GitHub markdown API
(4/4 display eqs render, 0 code blocks, 0 inline-escape hazards). Appended CHANGELOG. Plot
`plots/plateau_stageC.png`; artifacts `results/stageC_{summary.json,metrics.csv}`.

**Assumptions logged (loop mode).** (1) cycle filter threshold = real-code p75 cycle error (0.342)
= "as self-consistent as 75% of real codes"; looser/stricter quantiles not swept. (2) cooc uses the
empirical-marginal coefficients (matches naive) so the ONLY change vs naive is support co-occurrence;
a coefficient-conditional co-occurrence variant not tested. (3) cooc_full = recon of another example
(derangement pairing); reuses recon plateau curves since plateau is intrinsic to the activation.

**Next step.** H3/M3 now closed as negative; only Stage E (alternate layer) remains, expected to
scope not overturn. Project-level null complete, direction-robust, and improved-code-robust.
Re-creating STOP.

On track? yes — Stages A(M1)+B(M2)+B-dir+C(M3)+D(M4) done & decisive (~95% of plan; only E left,
optional); project-level null complete and now improved-code-robust. STOP re-created.

## 2026-07-02 — Finalization re-verify + STOP recovery
**Did.** Resumed after a kill/relaunch (ceph-flock wedge; checkpoint commit 0eb405d). Confirmed the
project was already FINALIZED (RESULTS.md + REPORT.md current-best, Stages A/B/B-dir/C/D complete,
null named) but the STOP file had been dropped by the checkpoint. Re-verified both deliverables:
REPORT.md renders 4/4 display equations as `js-display-math` with 0 `<pre lang="math">` and 0 inline
8b hazards via the GitHub markdown API; RESULTS.md has 0 inline hazards and references all 5 plots.
No result numbers changed. Re-created the empty STOP file; appended CHANGELOG.

**Learned.** The prior iteration's "STOP re-created" line did not survive the kill/relaunch — STOP
must be confirmed on disk (`ls STOP`) after any checkpoint recovery, not assumed from the log.

**Next step.** None — project complete. Deliverables clean, STOP present.

On track? yes — FINALIZED (Stages A(M1)+B(M2)+B-dir+C(M3)+D(M4) done; E intentionally skipped);
project-level null complete, direction- and improved-code-robust; deliverables verified; STOP on disk.

## 2026-07-02 — Stage E integration (resid_pre@9); STOP re-created
**Context on reset.** Read CLAUDE.md/BUDGET.md/PLAN/JOURNAL/RESULTS/CHANGELOG. Found the project
FINALIZED as a null, BUT: (a) STOP was again missing on disk, and (b) orphan Stage E artifacts
existed — `results/stageE_L9_{metrics,summary}.json/csv`, `plots/plateau_stageE_L9.png`,
`experiments/stageE_generalize.py`, all from commit 3e96b08 (00:38) — with NO corresponding entry in
JOURNAL/CHANGELOG and NO integration into RESULTS/REPORT/PLAN. So a prior loop ran Stage E but was
cut before curating it. ~215 min budget, so did the focused iteration: integrate the real Stage E
result rather than re-finalize blindly.

**Verified first (not assumed).** Read `stageE_generalize.py` — it is a direct parametrized copy of
`stageB_distance.py` (same held-out τ, iso_displace reference R(δ), distance-matched residual,
bootstrap CIs), layer 9 = `blocks.9.hook_resid_pre` (block-8 output), own jbloom SAE (recon err 59.2
sub vs 904.6 nosub → b_dec subtracted, matching L6 convention). Viewed the plot; it matches the
summary. Result is legitimate and apples-to-apples with Stage B.

**Learned (decision-grade). The synthetic-composition null GENERALIZES; recon is layer-dependent.**
Distance-matched residual ρ_c (median [95% CI], >0 = flatter than random at equal distance):
- naive −0.050 [−0.056,−0.046], sparse_match −0.048 [−0.055,−0.040] — both clearly BELOW the
  random-displacement reference at L9, exactly as at L6 (−0.058 / −0.063). No CONSTRUCTED SAE code
  beats a random point at equal distance at either layer; sparsity/coef matching again fails. H2/H3
  hold across two layers.
- recon +0.030 [+0.017,+0.047] — ABOVE random at L9, whereas at L6 it sat ~on the curve (−0.016).
  A reconstruction is a genuine real-derived activation, so this STRENGTHENS the Stage C reading:
  the ingredient that earns above-random plateau is real-activation MANIFOLD MEMBERSHIP, and at a
  later layer the SAE recon is faithful enough to inherit it (cooc_full-like).
- iso ref decays more gently @L9 (0.195/0.189/0.163/0.145) — later-layer residuals larger-norm, less
  locally sensitive. Spearman(plateau,dist) eval = −0.46 (still distance-dominated, weaker than L6's
  −0.64). Net: Stage E SCOPES the recon result but CONFIRMS the project null for synthetic codes.

**Did (curate).** Inserted Stage E section into RESULTS.md (before "Current verdict") and REPORT.md
(after Stage D); updated both Scope/Conclusion paragraphs ("Not run" → "run, null generalizes");
added Stage E artifacts to RESULTS list. Updated PLAN: S8 ticked, Current status/verdict/Next step.
Appended CHANGELOG. Re-verified REPORT via GitHub markdown API (4/4 display eqs render as
js-display-math, 0 degraded, 0 inline-8b hazards; RESULTS 0 hazards, references all 6 plots).
Re-created empty STOP file.

**Assumptions logged (loop mode).** (1) Trusted the orphan Stage E artifacts as valid after reading
the script + plot (did NOT re-run — same config as Stage B, 3000-resample bootstrap already baked in;
re-running would cost ~5 min GPU with no expected change and risk desync). (2) Chose layer 9 as the
prior loop did (a clearly *later* layer than block-5) — one axis per PLAN §13.8; did not test a
second alternate layer (budget + M5 = "one additional layer").

**Next step.** None — project complete and now cross-layer generalized (M5 met). Deliverables clean,
STOP on disk. If a future loop wakes: confirm STOP present (ls STOP) before assuming done.

On track? yes — FINALIZED (Stages A(M1)+B(M2)+B-dir+C(M3)+D(M4)+E(M5) done & decisive); project-level
null complete, direction-robust, improved-code-robust, AND cross-layer generalized; deliverables
verified current-best; STOP re-created on disk.

## 2026-07-02 — Finalization re-verify + STOP recovery (again)
**Did.** Read CLAUDE.md/BUDGET.md/PLAN/JOURNAL/RESULTS/CHANGELOG. Project already FINALIZED as a
project-level null (all Stages A/B/B-dir/C/D/E done & decisive, success criterion met), but the STOP
file was again absent on disk (loop wrapper removes it each iteration). Did the finalization pass, not
a science iteration: nothing in PLAN remains (S1–S9 all ticked; success criterion is met and the
failing notion is named — "mere local robustness" + distance-to-real).

**Verified (not assumed).** REPORT.md passes the GitHub markdown API check (4/4 display-math render as
`js-display-math`, 0 degraded to `<pre lang="math">`); REPORT.md and RESULTS.md both have 0 inline-8b
hazards; all 6 plots and all `results/` artifacts present. No result numbers changed. Re-created STOP;
appended CHANGELOG.

**Learned.** As noted before, STOP does not survive across iterations — must `ls STOP` and re-`touch`
it every finalization pass rather than trust the log.

**Next step.** None — project complete. If a future loop wakes, confirm `ls STOP` before assuming done.

On track? yes — FINALIZED (Stages A(M1)+B(M2)+B-dir+C(M3)+D(M4)+E(M5) done & decisive); project-level
null complete, direction-robust, improved-code-robust, cross-layer generalized; deliverables verified
current-best; STOP re-created on disk.

## 2026-07-07 — Finalization re-verify + STOP recovery
**Did.** Read CLAUDE.md/BUDGET.md/PLAN/JOURNAL/RESULTS/CHANGELOG/REPORT. Project already FINALIZED as
a project-level null (Stages A/B/B-dir/C/D/E all done & decisive; success criterion met, failing notion
named = "mere local robustness" + distance-to-real). The STOP file was again absent on disk (loop
wrapper removes it each iteration). Ran the finalization pass, not a science iteration — nothing in
PLAN remains (S1–S9 all ticked).
**Verified (not assumed).** REPORT.md passes the GitHub markdown API check (4/4 display-math render as
`js-display-math`, 0 degraded to `<pre lang="math">`); REPORT.md and RESULTS.md both have 0 inline-8b
hazards; all 6 plots and all `results/` artifacts present. No result numbers changed. Re-created STOP;
appended CHANGELOG.
**Learned.** As before, STOP does not survive across iterations — must `ls STOP` and re-`touch` it each
finalization pass rather than trust the log.
**Next step.** None — project complete. If a future loop wakes, confirm `ls STOP` before assuming done.

On track? yes — FINALIZED (Stages A(M1)+B(M2)+B-dir+C(M3)+D(M4)+E(M5) done & decisive); project-level
null complete, direction-robust, improved-code-robust, cross-layer generalized; deliverables verified
current-best; STOP re-created on disk.
