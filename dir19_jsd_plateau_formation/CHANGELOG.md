# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-08-05 — iteration 1: direction created; full early-training checkpoint scan

**New deliverables.** `RESULTS.md` and `REPORT.md` were placeholders (`TODO — describe this
direction`); both are now written from scratch to current-best. Six figures added to `plots/` and
embedded as rendered images in BOTH files.

**S1 — transfer and freeze.** Copied the frozen 60-pair manifest (`pair_manifest_top256.json`), the
1,000-pair manifest, the corpus/reliability manifests, and the existing step-0/1000/8000/32000/
64000/143000 assay artefacts from `dir18_continuation_jsd_plateau` unmodified; SHA-256 of all 25
inherited files recorded in `results/INHERITED_HASHES.txt`. Compatibility check: re-running the
assay at step 0 with this direction's code reproduced dir18's curves **bit-for-bit** (max |Δd| = 0
over 9,000 values; max |Δw| = 0; max |Δ output JSD| = 0), and our final-checkpoint
ρ = −0.5247 matches dir18's `summary.json` value to four decimals.

**S2/S3/S4 — new measurements.** Ran the frozen 60-pair bank at **20 released checkpoints** (steps
0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 96000,
128000, 143000; step16 subsequently excluded, see below). Added three per-checkpoint measurements
that dir18 did not have: the neighbouring-position full-output movement profile m_j, next-token loss
on a frozen 256-row held-out corpus sample, and the published learning rate.

**S5 — validation.** Ran the frozen 1,000-pair bank at step 64000 (new); combined with dir18's
existing step0 and step143000 large-bank runs to complete the prespecified 64k → final test.

**New results (no prior values existed in this direction — nothing superseded).**
- Ordering onset bracket: **after step 8, by step 32** (ρ = −0.060 [−0.384, +0.265] at step 8 →
  −0.428 [−0.753, −0.104] at step 32, simultaneous 95% band, c = 0.324). This tightens the previous
  bound of "present by step 1000" by a factor of ~30.
- Shape onset bracket: **after step 1000, by step 2000** (median w band reaches 0.805 at step 1000,
  ≤ 0.732 at step 2000; E band ≤ 0.147 at step 2000, vs straight-line references 0.8 and 0.184).
- Interval dissociation: ρ(J, Δw) = −0.466 [−0.663, −0.223] over step 8 → 32 with median Δw only
  −0.0011, versus ρ = +0.035 [−0.241, +0.307] over step 512 → 1000 with median Δw = −0.0618
  (Wilcoxon p = 1.9e−11) — the largest sharpening event is divergence-blind.
- Movement concentration: normalised entropy 1.000 → 0.630; fixed-window mass 0.200 → 0.900; total
  movement 0.0016 → 0.135 bits. Onset coincides with shape, not ordering.
- Late widening CONFIRMED on the large bank: 60-pair median Δw = +0.0121 [+0.0016, +0.0259]
  (38/60 blunter, Wilcoxon p = 0.0052); 1,000-pair median Δw = +0.0158 [+0.0081, +0.0224] under the
  endpoint-clustered dyadic bootstrap, 65.1% of pairs blunter. Prespecified verdict "the late
  widening reproduces on 1,000 pairs" selected.
- Prespecified verdict selected for the main question: **"JSD ordering appears before global plateau
  shape."**

**Data-integrity finding (new).** The artefact served as revision `step16` of
`EleutherAI/pythia-1.4b-deduped` is not a step-16 model: held-out loss 2.320 nats (neighbours 9.889
and 8.824), 9,000 d(t) values bit-identical to `step143000`, `model.safetensors` 32 bytes smaller
than all 19 other revisions queried. Excluded from every trajectory; evidence in
`results/ckpt_qc.json` and Figure 6 of both deliverables.

**Figures added and embedded:** `formation_overview.png`, `interval_sharpening.png`,
`output_movement_formation.png`, `movement_profiles.png`, `large_bank_confirmation.png`,
`checkpoint_qc.png`. All CVD-safe (green-free palette, linestyle+marker redundancy).
`experiments/check_render.py REPORT.md RESULTS.md` exits 0.

### 2026-08-05 — same iteration, addendum: ordering-onset validated on the 1,000-pair bank

Completed the last outstanding prespecified validation (PLAN "Large validation set": run the two
checkpoints that define the onset bracket). Assayed all 1,000 pairs at **step 8** and **step 32**.

- New result, Result 6 in both deliverables: endpoint-clustered ρ(J, w) = **−0.021 [−0.132, +0.104]**
  at step 8 (contains zero) and **−0.149 [−0.286, −0.011]** at step 32 (excludes zero). The step
  8 → 32 onset bracket therefore replicates on a bank 17× larger, and median w there is still 0.830 /
  0.828 against 0.831 at step 0, so the "ordering without sharpening" claim replicates too.
- Reported honestly alongside it: the step-32 effect size is much smaller on the large bank (−0.149
  vs −0.428 on the controlled bank), while the two banks nearly agree by the final checkpoint (−0.486
  vs −0.525). The large bank is unmatched on frequency/surprisal and reuses endpoint tokens, so the
  deliverables now state that the timing should be read from both banks and the magnitude from the
  controlled one.
- New figure `plots/large_bank_onset.png`, embedded as Figure 6 in RESULTS.md and REPORT.md; the
  checkpoint-QC figure renumbered 6 → 7 in both. Figure count 6 → 7 in each file.
- The three loose 1,000-pair ρ values previously reported as a sentence inside Result 5 (step 0 /
  64000 / 143000) moved into Result 6 with the two new checkpoints, so the bank now has one table
  and one figure instead of a scattered mention.

`python3 experiments/check_render.py REPORT.md RESULTS.md` exits 0 (7 embeds / 7 visible captions in
each file).

Housekeeping: `results/INHERITED_HASHES.txt` now lists only the 22 artefacts frozen from dir18 (all
re-verified byte-identical to upstream with `cmp`); the full 97-file inventory moved to
`results/ALL_ARTIFACT_HASHES.txt`. No reported number changed.

## 2026-08-06 — iteration 2: permutation inference (PLAN S5 "endpoint-label permutation")

**Feedback check.** No `human_feedback*.md` / `*REVIEW*` file without `.addressed.md` exists in this
direction. Proceeded with the plan.

**What was missing.** Every interval in the deliverables was a bootstrap. PLAN's S5 also asks for an
**endpoint-label permutation** on the 1,000-pair bank, and nothing in the direction implemented one.
This iteration adds it, plus the matching label-permutation null on the 60-pair bank. New code:
`experiments/permtest.py` (20,000 permutations, CPU only, reads saved curves) and
`experiments/plot_perm.py`; new artefact `results/permutation.json`.

**New results (nothing superseded; these are additional inference on existing statistics).**
- 60-pair cross-sectional ρ: null |ρ| reaches 0.26 pointwise and **0.353 simultaneously** over the 19
  kept checkpoints. Observed p = 0.67 / 0.67 / 0.67 / 0.60 / 0.65 at steps 0, 1, 2, 4, 8, then
  **p = 0.0007 at step 32** with family-wise **p_fw = 0.0072**. Every later checkpoint p_fw ≤ 0.013.
- 60-pair interval ρ(J, Δw): step 8 → 32 p = 0.0003, **p_fw = 0.0035** over all 18 intervals; step
  512 → 1000 p = 0.78 (p_fw = 1.00). The dissociation survives multiplicity correction.
- 1,000-pair endpoint-label (QAP) permutation over its 123 endpoint tokens: chance |ρ| reaches
  **0.09** (vs ~0.062 for 1,000 independent pairs), which prices the token reuse into the null.
  p = 0.87 (step 0), 0.64 (step 8), **0.0031 (step 32, p_fw = 0.0082)**, < 0.001 (steps 64000,
  143000). The onset bracket therefore holds under a second, assumption-free form of inference —
  ρ = −0.149 at step 32 was the weakest number in the report.
- Verification recorded in code: the 123×123 held-out JSD matrix reproduces each pair's stored
  `jsd_B` to 5.0e−7, and ρ computed from the matrix equals ρ from `jsd_B` to 6 decimals at all five
  large-bank checkpoints.

**One reading corrected (old → new).** The step 4000 → 8000 interval was described as
"ρ(J, Δw) = +0.258 [+0.002, +0.483] — large sharpening, not divergence-selective", a bootstrap
interval that just excludes zero. Its permutation p = 0.045 does **not** survive the 18-interval
correction (p_fw = 0.55), so both deliverables now describe that interval as divergence-**blind**
rather than as reversed selectivity. No other number changed.

**Deliverable changes.** REPORT.md: two new Methods paragraphs (label-permutation null; endpoint-label
permutation) with three new rendered equations; new **Result 7 — "Chance never produces this ordering,
on either bank"**; the step16 data-integrity result renumbered 7 → 8; permutation p-values added to
the Summary and to the two-onset table. RESULTS.md: permutation p / p_fw column added to the interval
table, an endpoint-label permutation p row added to the 1,000-pair table, and p_fw added to the
ordering-onset row. New figure `plots/permutation_null.png` embedded as **Figure 7** in both files
(checkpoint-QC figure renumbered 7 → 8); figure count 7 → 8 in each. CVD-safe palette, no red/green,
every series also coded by linestyle/marker.

Also fixed a rule-12 gap that predated this iteration: Figures 4–8 (REPORT.md) and Figures 1–8
(RESULTS.md) were embedded and captioned but never cited by number in the body prose. Every figure in
both files is now cited by number at least once.

`python3 experiments/check_render.py REPORT.md RESULTS.md` exits 0 (8 embeds / 8 visible captions per
file; 13 display equations in REPORT.md all render as `js-display-math`).

## 2026-08-06 — iteration 3: ranking persistence (is the step-32 ordering the FINAL ordering?)

**Feedback check.** No `human_feedback*.md` / `*REVIEW*` file without `.addressed.md` exists in this
direction. Proceeded with the plan.

**What was missing.** The deliverables showed that corpus divergence $J$ ranks the pairs by step 32,
and that the curves are still straight lines there. They never asked whether that step-32 ranking is
the ranking the trained model ends up with — a divergence-shaped ordering could in principle be
discarded and re-derived later without ever showing up in the cross-sectional $\rho_s$ trajectory.
New code: `experiments/persistence.py` (CPU only, reads `results/per_pair_trajectories.npz`; 4,000
paired bootstraps + 20,000 label permutations, the permutation null in closed form over centred rank
vectors) and `experiments/plot_persistence.py`; new artefact `results/persistence.json`.

**New results (nothing superseded; this is a new measurement).**
- Ranking persistence π(s) = Spearman(w_s, w_final) is inside the chance envelope (|π| ≤ 0.253)
  through step 64: +0.109 (step 0), +0.121 (step 8), +0.161 (step 32, p = 0.21), +0.207 (step 64,
  p = 0.11). It jumps to **+0.437 [+0.202, +0.623] at step 128** (p = 0.0007, p_fw = 0.0053 over all
  19 checkpoints), then 0.532 / 0.696 / 0.788 at steps 256 / 512 / 1000. **New onset bracket: the
  per-pair ranking becomes final after step 64, by step 128** — a third clock between the ordering
  onset (8 → 32) and the shape onset (1000 → 2000).
- Partial persistence with J removed: π⊥ = **−0.082** at step 32 (p = 0.53) — once corpus divergence
  is partialled out, the step-32 ranking and the final ranking have nothing in common. Observed
  π = 0.161 is close to the (−0.428)×(−0.525) = 0.225 the two divergence correlations alone imply.
  Pair-specific detail beyond J first clears the family-wise bar at step 256 (π⊥ = +0.380,
  p_fw = 0.0275; step 128 is +0.238, p_fw = 0.39).
- Attenuation control (rules out the boring explanation): per-pair widths agree across the three
  carrier sentences at r̄ = 0.830 at step 32 → Spearman-Brown reliability 0.936 → ceiling
  π_max = 0.935. So the low π is a real disagreement, not noise on a 0.006-wide spread. Reliability
  is ≥ 0.865 at every checkpoint, including 0.872 at step 0.

**One claim corrected (old → new).** The Summary said later training "sharpens nearly every pair
together while **largely preserving that early ranking**". That is not supported: the step-32 ranking
agrees with the final one at only π = 0.161. Both deliverables now say what is preserved is the
divergence *alignment* (ρ stays negative throughout), while the pair-specific ranking is set later, at
step 64 → 128. No numerical result changed.

**Deliverable changes.** REPORT.md: two new Methods run-in paragraphs (ranking persistence π and π⊥;
reliability of w and the ceiling it puts on π) with three new rendered equations; new **Result 8 —
"At step 32 the model holds only the divergence-aligned part of the final ranking"**; the step16
data-integrity result renumbered 8 → 9 (and its cross-reference in Data & Model); a new row in the
onsets table, which is retitled from "the two onsets" to "the onsets"; Summary and Conclusion rewritten
to carry three clocks instead of two. RESULTS.md: new persistence table (π, p/p_fw, π⊥, ceiling), new
row in the onset table, headline bullet added. New figure `plots/ranking_persistence.png` embedded as
**Figure 8** in both files (checkpoint-QC figure renumbered 8 → 9); figure count 8 → 9 in each. Panel
A is a trajectory with the null envelope and the reliability ceiling, panel B the 20×20
checkpoint-agreement matrix on `cividis`; green-free palette, every series also coded by
linestyle/marker.

`python3 experiments/check_render.py REPORT.md RESULTS.md` exits 0 (9 embeds / 9 visible captions per
file; 16 display equations in REPORT.md all render as `js-display-math`).
