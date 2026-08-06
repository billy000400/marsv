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
