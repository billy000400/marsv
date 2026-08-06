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

## 2026-08-06 — iteration 4: byte-level proof of the `step16` mislabelling, and an audit of all 21 revisions

**Feedback check.** No `human_feedback*.md` / `*REVIEW*` file without `.addressed.md` exists in this
direction. Proceeded with the plan.

**What was missing.** Result 9 claimed that Hugging Face serves the wrong artefact under revision
`step16` of `EleutherAI/pythia-1.4b-deduped`, but all three pieces of evidence were behavioural or
circumstantial (a loss outlier, curves bit-identical to `step143000`, a file 32 bytes smaller). None
of them excluded a corrupted local download, none said *what* the artefact is, and none said whether
any other revision in the trajectory is also wrong. New code: `experiments/step16_forensics.py`
(streams each revision's safetensors tensor payload from the Hugging Face CDN and hashes it, plus
per-tensor digests over a 10-tensor sample; ~11 GB streamed, nothing written to disk) and
`experiments/revision_audit.py` (header + published-digest audit of all 21 revisions, ~1 MB of
traffic). New artefacts `results/step16_forensics.json`, `results/revision_audit.json`.

**New results (nothing superseded; this strengthens Result 9's evidence, no number changed).**
- The SHA-256 of `step16`'s 2.63 GiB tensor payload **equals `step143000`'s exactly**
  (`fbd54ccec4e0f5ee…`). `step8` (`48c2b6a93871…`) and `step32` (`0459bf847197…`) differ, as they
  must. All 10 individually hashed tensors — embeddings, unembedding, final layer norm, and the
  attention output weight and bias of blocks 0, 11, 23 — match `step143000` byte for byte; 0 of 10
  match `step8` or `step32`. So the claim moves from "is not a step-16 model" to "**is the final
  model's parameters**".
- The 32-byte deficit is explained: `step16`'s header omits the `__metadata__` field
  (`{"format": "pt"}`) that all 20 other revisions carry — the signature of a file re-serialised by a
  different tool, detectable with a 34 KB range request.
- **Audit of all 21 revisions used here:** 20 share one byte-identical header layout (34,296 bytes,
  292 tensors, identical dtypes/shapes/offsets), so equal weights would force equal file digests; all
  20 published SHA-256 are distinct and none equals `step143000`'s. `step16` is the only affected
  revision, and no duplicate hides among the closely spaced late checkpoints where the loss check
  would not have flagged one.
- Stated the consequence for the headline: because no genuine step-16 weights are published, the
  ordering-onset bracket cannot be narrowed below **after step 8, by step 32** from released
  artefacts. This is a resolution limit, not a bias.

**Deliverable changes.** REPORT.md: new Methods run-in paragraph "Checkpoint provenance check" under
Data & Model (payload digest + per-tensor digests + the header-layout argument that makes the 21-way
audit exact); Result 9 rewritten around the byte-level evidence and the audit, with the resolution-limit
consequence stated; Summary's step16 paragraph upgraded from "is not a step-16 model" to "is the fully
trained final model" plus the audit; Reproducibility lists the two new scripts. RESULTS.md: the
data-integrity paragraph rewritten with the payload digest, the per-tensor counts, the packaging
explanation and the audit. `plots/checkpoint_qc.png` regenerated from 1 panel to 3 (A loss-trajectory
outlier, unchanged; B sampled-tensor byte matches against `step143000` with whole-payload digest
verdicts; C header length for all 21 revisions), and its caption rewritten in both files. Figure count
stays 9 per file. Corrected a stale count in the old caption: 20 checkpoints are kept, not 19.
Green-free CVD palette; bars carry distinct hatches, series distinct markers.

`python3 experiments/check_render.py REPORT.md RESULTS.md` exits 0 (9 embeds / 9 visible captions per
file; 16 display equations in REPORT.md all render as `js-display-math`).

### Same iteration — width-definition robustness (new Result 10 / Figure 10)

**What was missing.** Every headline number rested on one metric, $w = t(0.9) - t(0.1)$, whose levels
are a convention inherited from upstream rather than a choice the data made. If the ~60× separation
were an artefact of that convention it would be the most consequential error in the report. New code:
`experiments/threshold_robustness.py` (CPU only, reads the saved curves; re-runs both prespecified
onset rules on five width definitions with a paired trajectory bootstrap, B = 4,000) and
`experiments/plot_threshold.py`. New artefact `results/threshold_robustness.json`.

**New results (nothing superseded; this is a new robustness measurement).**
- With $w_a = t(1-a) - t(a)$ for $a \in \lbrace 0.10, 0.15, 0.20, 0.25, 0.30 \rbrace$ (straight-line
  reference $1-2a$; curve validity held at the original 0.1/0.9 rules so the same curves enter every
  trajectory), the **ordering onset bracket is step 8 → 32 for all five definitions**. $\rho_{32}$
  ranges only over $-0.428, -0.415, -0.409, -0.391, -0.385$; the interval statistic
  $\rho(J, \Delta w)$ over step 8 → 32 over $-0.466, -0.461, -0.456, -0.452, -0.452$.
- The **shape bracket moves one checkpoint earlier** (step 512 → 1000 instead of step 1000 → 2000)
  for the three wider levels $a \ge 0.20$, which weight only the steep middle of the curve. The
  separation between the two events is therefore 31× under those definitions and 62× under the two
  narrower ones — same order of events, no definition brings the brackets within a factor of 30.
- The Result 3 dissociation holds throughout: over step 512 → 1000, $\rho(J, \Delta w)$ is $+0.035$,
  $+0.142$, $+0.229$, $+0.275$, $+0.312$ — never negative, so the largest sharpening event is
  divergence-blind under every definition (mildly favouring low-divergence pairs at wider levels).
  Flagged in the report that the four alternative values were not permutation-tested.

**Deliverable changes.** REPORT.md: new Methods run-in paragraph defining $w_a$ and its straight-line
reference (one new rendered equation, display count 16 → 17), new **Result 10 — "The separation is not
an artefact of how 'width' is defined"**, a qualifying sentence added to the Summary and one under the
onsets table. RESULTS.md: new "Robustness to the width definition" paragraph. New figure
`plots/threshold_robustness.png` embedded as **Figure 10** in both files (A: five $\rho$ trajectories;
B: median $w_a$ divided by its own straight-line reference, putting all five on one scale; C: both
onset brackets per definition with the separation ratio). Figure count 9 → 10 in each file. Green-free
CVD palette, every series also coded by linestyle and marker.

`python3 experiments/check_render.py REPORT.md RESULTS.md` exits 0 (10 embeds / 10 visible captions
per file; 17 display equations in REPORT.md all render as `js-display-math`).

## 2026-08-06 — iteration 5: reference robustness of the third clock (new Result 11 / Figure 11)

**What was missing.** Result 8's "the pair ranking becomes final between step 64 and step 128" was
measured against a single reference, step 143000 — the last *released* checkpoint, not a point at
which the model demonstrably stops changing. Result 5 shows the widths still move over the last third
of training (median $\Delta w = +0.0158$ from step 64000 to the end), so the bracket could in
principle have been an artefact of scoring against that particular endpoint. New code:
`experiments/persistence_ref.py` (CPU only, reads the saved per-pair width trajectories; recomputes
$\pi$, $\pi^{\perp}$, a paired bootstrap over pairs with B = 4,000, and a 20,000-draw
one-permutation-per-trajectory label null, for five reference checkpoints) and
`experiments/plot_persistence_ref.py`. New artefact `results/persistence_ref.json`.

**New results (nothing superseded; this is a new robustness measurement).**
- Scoring against **step 8000, 32000, 64000, 128000 and 143000** returns the **identical bracket,
  after step 64 and by step 128, for all five**. At step 128, $\pi = +0.447, +0.394, +0.430, +0.410,
  +0.437$ with family-wise permutation $p = 0.0045, 0.018, 0.0059, 0.012, 0.0050$; at step 64 no
  reference is significant ($p^{\mathrm{fw}}$ between 0.47 and 0.89).
- The step-32 reading is equally stable: $\pi_{32} = +0.077, +0.163, +0.200, +0.174, +0.161$, inside
  the chance envelope for every reference ($p \ge 0.13$), with the divergence-free part
  $\pi^{\perp}_{32} = -0.147, -0.015, -0.098, -0.062, -0.082$ — at or below zero throughout.
- Mechanism for the insensitivity, now stated: the late widening changes the magnitude of $w$ without
  reshuffling which pairs are sharpest, so $\pi(w_{8000}, w_{143000}) = 0.89$ and the five
  trajectories coincide from step 128 onward.

**Deliverable changes.** REPORT.md: new Methods run-in paragraph "Persistence against other
references, $\pi_{\mathrm{ref}}$" with one new rendered equation (display count 17 → 18); new
**Result 11 — "The third clock does not depend on which checkpoint we call 'final'"** placed after
Result 10 and before the onset summary; one qualifying sentence added to the Summary's third-clock
paragraph and one to the note under the onsets table; `persistence_ref.py` and
`plot_persistence_ref.py` added to Reproducibility. RESULTS.md: new "Robustness to the reference
checkpoint" paragraph under the persistence table. New figure `plots/reference_robustness.png`
embedded as **Figure 11** in both files (A: five $\pi_{\mathrm{ref}}$ trajectories against the
permutation envelope with the bracket stripe; B: $\pi$ at step 32 and step 128 plus $\pi^{\perp}$ at
step 32, per reference, with 95% intervals). Figure count 10 → 11 in each file. Green-free CVD
palette; every series also coded by linestyle and marker; bands hatched.

`python3 experiments/check_render.py REPORT.md RESULTS.md` exits 0 (11 embeds / 11 visible captions
per file; 18 display equations in REPORT.md all render as `js-display-math`).

## 2026-08-06 — iteration 6: carrier-sentence jackknife (new Result 12 / Figure 12)

**What was missing.** Every width in this direction is a median over three fixed carrier sentences
(`"The thing was"`, `"They said it was"`, `"I thought it was"`). Averaging is what makes $w$ reliable
enough to correlate at step 32, but it is also a way for one frame's quirk to reach the deliverables
as a training-time fact: a correlation present in only one of the three would still survive the
median. Nothing in the report tested that. New code: `experiments/sentence_jackknife.py` (CPU only,
~40 s; recomputes per-pair $w$ and $E$ from each single context with no averaging, then re-runs all
three prespecified onset rules — ordering, shape, ranking — on each trajectory with its own 4,000-draw
paired bootstrap and 20,000-draw label-permutation null) and `experiments/plot_jackknife.py`. New
artefact `results/sentence_jackknife.json`.

**New results (nothing superseded; this is a new robustness measurement).**
- **Ordering bracket: step 8 → 32 in all three sentences**, identical to the primary analysis.
  $\rho_{32} = -0.363$, $-0.442$, $-0.359$ (primary median-of-three: $-0.428$), each with a
  simultaneous 95% band excluding zero, and $\rho_8$ non-significant in all three.
- **Shape bracket: step 1000 → 2000 in all three.** Median $w$ at step 32 is 0.826, 0.827, 0.828 —
  no plateau in any frame at the moment the ordering appears.
- **Ranking bracket: step 64 → 128 in two of three.** Sentences 2 and 3 give $\pi_{128} = +0.504$
  ($p^{\mathrm{fw}} = 0.0006$) and $+0.388$ ($p^{\mathrm{fw}} = 0.024$); sentence 1 closes one
  checkpoint later, at step 256 ($\pi_{128} = +0.284$, $p^{\mathrm{fw}} = 0.19$; $\pi_{256} = +0.413$,
  $p^{\mathrm{fw}} = 0.012$). Attenuation accounts for the shift: at step 128 a single context's
  ceiling is $\pi_{\max} = 0.871$ against 0.953 for the median of three, and step 64 → 256 is where
  $\pi$ climbs through the chance envelope.
- Order of the three events unchanged in every frame.

**Deliverable changes.** REPORT.md: new Methods run-in paragraph defining the single-sentence width
$w^{(c)}$ (one new rendered equation, display count 18 → 19); new **Result 12 — "No single carrier
sentence is carrying the result"** placed after Result 11 and before the onset summary; one qualifying
clause added to the Summary's robustness sentence and one to its third-clock paragraph; one sentence
added under the onsets table; `sentence_jackknife.py` and `plot_jackknife.py` added to
Reproducibility. RESULTS.md: new "Robustness to the carrier sentences" paragraph. New figure
`plots/sentence_jackknife.png` embedded as **Figure 12** in both files (A: three $\rho$ trajectories
plus the median; B: median $w$ against the straight-line reference; C: $\pi$ against the permutation
envelope; D: all three brackets for all four width definitions). Figure count 11 → 12 in each file.
Green-free CVD palette; every series also coded by linestyle and marker; bands and stripes hatched.

`python3 experiments/check_render.py REPORT.md RESULTS.md` exits 0 (12 embeds / 12 visible captions
per file; 19 display equations in REPORT.md all render as `js-display-math`).

## 2026-08-06 — iteration 7: divergence-quintile dependence of the onset (new Result 13 / Figure 13)

**What was missing.** Result 1 dates the divergence ordering by a Spearman correlation measured
across a divergence range spanning 0.137 to 0.942 bits. A rank correlation over a wide range can be
produced entirely by its extremes, in which case "corpus divergence orders the pairs at step 32" is a
two-group contrast and the bracket dates two extreme groups separating rather than a graded relation
appearing. Nothing in the report tested which pairs carry $\rho_{32}$. New code:
`experiments/quintile_loo.py` (CPU, ~14 s: recomputes the full 20-checkpoint $\rho$ trajectory,
simultaneous band, 20,000-draw label-permutation null, the prespecified ordering bracket and the
step 8 → 32 interval statistic on each leave-one-quintile-out subset and on the middle-three subset;
plus a size-matched random-drop control with 4,000 draws and per-quintile median $\Delta w$),
`experiments/quintile_large.py` (CPU, ~38 s: the same subsets on the frozen 1,000-pair bank at
step 32 and step 143000, under the endpoint-label QAP permutation restricted to each subset) and
`experiments/plot_quintile.py`. New artefacts `results/quintile_loo.json`,
`results/quintile_large.json`.

**New results (nothing superseded; this qualifies Result 1's content, not its timing).**
- **The step-32 ordering is carried by the highest-divergence quintile.** Dropping Q5 takes
  $\rho_{32}$ from $-0.428$ to $-0.191$ with a simultaneous band spanning zero
  ($p^{\mathrm{fw}} = 0.77$) and moves the ordering bracket from step 8 → 32 to step 64 → 128.
  Dropping any other quintile leaves it intact: $-0.426$ (Q1), $-0.475$ (Q2), $-0.463$ (Q3),
  $-0.473$ (Q4), all with bands excluding zero and the step 8 → 32 bracket unchanged.
- **This is not the cost of dropping pairs.** Against 4,000 size-matched random subsets, the Q1 drop
  sits at the median ($u = 0.49$) while the Q5 drop is more extreme than every draw ($u = 1.000$,
  random median $-0.425$); the both-tails subset gives $\rho_{32} = -0.134$, $u = 0.996$, bracket
  step 256 → 512.
- **In the units of the effect:** over step 8 → 32 the median $\Delta w$ is $+0.0004$, $+0.0006$,
  $-0.0013$, $-0.0013$ for Q1–Q4 (all 95% intervals covering zero) and $-0.0057$
  $[-0.0094, -0.0026]$ for Q5.
- **Power is ruled out by the large bank.** On the 1,000-pair bank at step 32, dropping Q5 takes
  $\rho$ from $-0.149$ ($p = 0.0023$) to $-0.091$ ($p = 0.081$), and its 600 middle-range pairs give
  $-0.055$ ($p = 0.35$). The same 600 pairs give $-0.300$ ($p < 0.0001$) at step 143000, so the bulk
  relation is measurable there and simply absent at step 32.

**Deliverable changes.** REPORT.md: new Methods run-in paragraph "Divergence-subset ordering,
$\rho^{(S)}$" defining the subset correlation and the size-matched random-drop control $u$ (two new
rendered equations, display count 19 → 21); new **Result 13 — "At its onset the ordering is a
top-quintile effect, not a graded one"** after Result 12 and before the onset summary; the Summary's
first numbered claim now states the top-quintile qualification with the large-bank numbers; a
qualifying sentence added under the onsets table; a new paragraph in the Conclusion narrowing the
mechanism to early separation of the most distinguishable pairs; `quintile_loo.py`,
`quintile_large.py` and `plot_quintile.py` added to Reproducibility. RESULTS.md: the first headline
bullet gains the same qualification, and a new "Which pairs carry the early ordering" paragraph.
New figure `plots/quintile_dependence.png` embedded as **Figure 13** in both files (A: $\rho_{32}$
per subset with simultaneous bands against the size-matched random envelope; B: per-quintile median
$\Delta w$ over step 8 → 32; C: the same subsets on the 1,000-pair bank at step 32 and step 143000
against the endpoint-label null). Figure count 12 → 13 in each file. Green-free CVD palette; every
series also coded by marker and linestyle; bands hatched.

`python3 experiments/check_render.py REPORT.md RESULTS.md` exits 0 (13 embeds / 13 visible captions
per file; 21 display equations in REPORT.md all render as `js-display-math`).
