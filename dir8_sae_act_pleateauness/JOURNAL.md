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
