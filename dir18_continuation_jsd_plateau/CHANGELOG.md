# CHANGELOG — Does continuation JSD predict plateau strength?

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-08-02 — first full result set (S1 through S5 complete)

RESULTS.md and REPORT.md went from empty templates to the complete current-best deliverable. No
result was superseded (nothing existed before), so no old -> new numbers apply.

**Added to both deliverables**

- **Corpus predictor (S1).** Byte-range sample of `EleutherAI/pile-deduped-pythia-preshuffled`:
  two distant, row-aligned 500,000-row splits (2.05B tokens total, ~0.68% of the 300B-token released
  stream). Reliability gates both pass: `Spearman(JSD_A, JSD_B) = 0.9998` (gate >= 0.90) and
  same-word split-half noise / between-word JSD = `0.072` (gate < 0.25).
- **Frozen pair bank (S2).** 75 endpoint-disjoint pairs, 15 per `JSD_A` quintile, balanced on
  log-frequency (Kruskal-Wallis p = 0.92) and model surprisal (p = 0.81). Frozen before any plateau
  curve was viewed; stored in `results/pair_manifest.json`.
- **Calibration gate (S3).** 15-pair frozen subset: valid-curve rate 1.000 (gate >= 0.80) and
  `IQR(w) = 0.115` (gate >= 0.05), so the assay has usable dynamic range.
- **Primary result (S4).** `pythia-1.4b-deduped` step143000: `Spearman(JSD_B, w) = -0.419`
  [-0.585, -0.222], p = 1.8e-4, n = 75. Step 0: `-0.155` [-0.368, +0.068], p = 0.18, with median
  `w = 0.831` and `IQR = 0.004` (essentially no plateau structure). `pythia-410m-deduped`
  step143000: `-0.320` [-0.526, -0.087], p = 5.1e-3.
- **Predictor validation.** `Spearman(JSD_B, model output JSD) = +0.729` [+0.599, +0.818],
  p = 1.2e-13 at step143000 versus `-0.144` at step 0 — this rules out the prespecified
  "global P(y|token) too coarse" verdict.
- **Sensitivity.** Partial Spearman adjusting for endpoint log-frequency, continuation entropy,
  surprisal, and block-0 cosine/distance: `-0.419 -> -0.267` (attenuated, still negative). Reported
  beside the unadjusted headline, never in place of it.
- **Controls and checks.** Block scan at blocks 0/6/12/18/23 gives monotone median `w` =
  0.549 / 0.646 / 0.726 / 0.796 / 0.805; endpoint self-test max relative error 4.7e-5; A<->B reversal
  changes `w` by at most 1.1e-5 (grid spacing 0.0204); within-pair prefix block-0 residuals differ by
  exactly 0.0; invalid-curve rate 0.000 in every bin at every checkpoint; per-context rho =
  -0.326 / -0.398 / -0.437 across the three carrier contexts.
- **Verdict recorded:** the prespecified "corpus JSD predicts model-output JSD and smaller `w`;
  step 0 does not" branch — predictive divergence is associated with learned plateau sharpening,
  with the explicit caveat that the association attenuates after geometry adjustment and is
  observational, not causal.

**Figures embedded in BOTH RESULTS.md and REPORT.md** (all new, all with visible captions, CVD-safe
palette with marker/linestyle/hatch redundancy): `plots/jsd_reliability.png` (Fig 1),
`plots/reference_curves.png` (Fig 2), `plots/jsd_vs_width.png` (Fig 3), `plots/width_by_jsd_bin.png`
(Fig 4), `plots/output_jsd_validation.png` (Fig 5), `plots/block_scan.png` (Fig 6).

**Documented plan deviation.** The plan prespecified a top-256 in-distribution endpoint filter, which
yields only 134 tokens = at most 67 endpoint-disjoint pairs, short of the 75-pair target. Per the
plan's fallback preference for an independent (non-all-pairs) design, the bank uses top-512 (258
tokens, still the top 2.8% of 18,714 eligible word tokens). Stated in REPORT.md Methods; the 12 pairs
whose endpoints are both inside the stricter top-256 give the same point estimate (rho = -0.406) but
are underpowered (p = 0.19).

---

## 2026-08-02 — formation subset added (same iteration, after the entry above)

**Added to both deliverables.** The optional formation subset from the plan's fixed setup: the SAME
frozen 75-pair bank run on `pythia-1.4b-deduped` at `step1000`, `step8000`, `step32000`, `step64000`,
giving a six-point trajectory together with the existing step0 and step143000 runs. New figure
`plots/formation.png` embedded as **Figure 6** in both RESULTS.md and REPORT.md; the block-scan figure
was renumbered **Figure 6 -> Figure 7** so numbering stays sequential in reading order.

**The finding contradicts the plan's expected pattern and is reported as such.** The plan predicted
the negative relationship would *strengthen* during training. It does not:

| step | 0 | 1000 | 8000 | 32000 | 64000 | 143000 |
|---|---|---|---|---|---|---|
| rho(JSD_B, w) | -0.155 | **-0.660** | -0.605 | -0.524 | -0.539 | -0.419 |
| rho(JSD_B, output JSD) | -0.144 | +0.779 | +0.693 | +0.726 | +0.714 | +0.729 |
| median w (IQR) | 0.831 (0.004) | 0.758 (0.087) | 0.624 (0.088) | 0.582 (0.098) | 0.541 (0.114) | 0.562 (0.111) |

The relationship is fully formed and at its strongest by step 1000, then decays, while median `w`
falls monotonically — plateaus keep sharpening but a context-free corpus statistic explains a
shrinking share of which pairs are sharp. Valid-curve rate 1.000 at all six checkpoints.

**No earlier number was superseded** — the headline step143000 and step0 figures are unchanged
(`rho = -0.419` and `-0.155`). The Summary, Headline and Conclusion sections gained an explicit
statement that the expected training trend was not observed.
