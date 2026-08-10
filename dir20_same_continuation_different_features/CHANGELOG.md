# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---
## 2026-08-10 — first results: S1+S2+S3 complete, hypothesis not supported

**RESULTS.md / REPORT.md:** both written from the placeholder templates to a full current-best
deliverable. Nothing superseded (first run), so no old -> new numbers.

New content:
- **S1 (tokenization + endpoint JSD).** All 5 prompt pairs (4 test + 1 control) validate in both
  `gpt2-medium` and `pythia-410m-deduped@step143000`: identical tokenized prefix, exactly one
  differing single final token. Endpoint JSD (nats) gpt2 / pythia — `Mary`/`her` 0.068 / 0.033;
  `four`/`4` 0.138 / 0.056; `four`/`Four` 0.377 / 0.271; `Au`/`79` 0.342 / 0.385;
  control `big`/`in` 0.659 / 0.665.
- **S2 (interpolation).** SLERP-direction + linear-norm interpolation of the block-0 `resid_post`
  at the final token, 101 alphas, patched forward. Harness identity check passes:
  |d(0)| <= 1e-4 and |d(1)-1| <= 1e-4 in all 10 cells. Final-logit w10-90 gpt2 / pythia —
  0.586 / 0.582; 0.454 / 0.758; 0.120 / 0.340; 0.358 / 0.598; control 0.516 / 0.425.
- **S3 (verdict).** Plateaus present in 9/10 cells (w_TV <= 0.27 vs linear 0.5) but endpoint
  similarity does not predict sharpness: pooled Spearman rho(JSD, w10-90) = -0.37 (p=0.29, n=10),
  rho(JSD, w_TV) = -0.15 (p=0.68), rho(JSD, PF) = +0.32 (p=0.37); sign flips between models
  (w_TV: +0.30 gpt2 vs -0.60 pythia). The control plateaus as sharply as the test pairs.

**Metric added beyond PLAN.md:** `w_TV` (alpha span carrying the middle 50% of the curve's total
variation; 0.5 for linear, ->0 for a step) and `PF` (plateau fraction). 4/10 curves are
non-monotonic, which inflates w10-90; w_TV is threshold-free and monotonicity-safe. w10-90 remains
the primary metric per PLAN.md.

**Figures added (all embedded in both deliverables):** `plots/final_logit_curves.png` (Fig 1, 5x2
raw d(alpha) curves + linear reference), `plots/jsd_vs_width.png` (Fig 2, JSD vs w10-90 and w_TV
with per-model Spearman), `plots/layerwise_widths.png` (Fig 3, w10-90 vs recording block).

**Framing (CLAUDE.md rule 9b):** the planned story ("similar continuations -> plateau") is not what
the evidence shows, so REPORT.md is built around the null and its practical consequence — a plateau
in a single-token interpolation needs a dissimilar-prompt control before it is evidence of anything,
because Figure 3 shows the sharpening is supplied by depth downstream of the patch.

## 2026-08-10 — powered test on a mined pair bank; the association reverses sign

**What changed.** Added Experiment 2: a corpus-derived bank of 200 prompt pairs per model (40
WikiText-103 validation prefixes of 10–40 tokens; final token A = model's top-1 next token, final
token B = the rank-$r$ token with $r$ log-uniform in [1,5000], 5 partners per prefix), run through
the identical block-0 SLERP sweep. New code `experiments/mine_pairs.py` and
`experiments/analyze_bank.py`; new results `results/bank_*.json|npz`, `results/bank_analysis.json`.

**Superseded result (same experiment: does endpoint similarity predict plateau sharpness?).**
- OLD (n=10 hand-picked model-pair cells): pooled Spearman rho(JSD, w10-90) = **-0.37 (p=0.29)**,
  rho(JSD, w_TV) = -0.15 (p=0.68), rho(JSD, PF) = +0.32 (p=0.37); sign flipped between models
  (w_TV: +0.30 gpt2 vs -0.60 pythia). Conclusion recorded then: "no detectable association".
- NEW (n=200 mined pairs per model): rho(JSD, w_TV) = **-0.55 [-0.66,-0.41], p=6.2e-17** (gpt2-medium)
  and -0.11 [-0.30,+0.07], p=0.12 (pythia-410m); rho(JSD, w10-90) = -0.47 (p=1.4e-12) / -0.12 (p=0.090);
  rho(JSD, PF) = +0.44 (p=5.7e-11) / +0.12 (p=0.090). All three statistics now agree on sign in both
  models. Below the ln 2 JSD ceiling (JSD<0.65): rho(w_TV) = -0.61 (n=142, p=1.5e-15) and -0.45
  (n=127, p=9.0e-8). Conclusion now: the association exists and runs **opposite** to the hypothesis —
  more divergent endpoints give sharper plateaus.
- CIs are 95% cluster bootstrap over the 40 prefixes (2000 resamples), since pairs sharing a prefix
  are not independent.

**New robustness results.** (a) JSD-ceiling check: 29% (gpt2) / 37% (pythia) of mined pairs sit at
JSD >= 0.65, which is why pythia's full-bank rho is diluted. (b) Partial Spearman controlling for the
block-0 angle Omega between the patched activations: -0.55 (gpt2, raw -0.55) and -0.16 (pythia, raw
-0.11); rho(Omega, w_TV) = 0.16 in both — the effect is not endpoint geometry.

**New prevalence result.** 82% (gpt2-medium) / 48% (pythia-410m) of arbitrary mined pairs are sharp
(w_TV < 0.25); median w_TV 0.080 / 0.266, median w10-90 0.241 / 0.593. All five hand-picked pairs lie
inside the bulk of their model's distribution. Also recorded: only 7.5% of gpt2-medium mined curves
are monotonic (98% in pythia), which further justifies w_TV.

**Figures.** Added `plots/bank_prevalence.png` (now Figure 2) and `plots/bank_regression.png` (now
Figure 3), both embedded with visible captions in RESULTS.md and REPORT.md. **Removed from both
deliverables:** `plots/jsd_vs_width.png` (the old 5-point scatter) — the same test at n=200 with
cluster-bootstrap CIs supersedes it (CLAUDE.md rule 6). The PNG stays on disk, unreferenced.
Figures renumbered in reading order: 1 raw curves, 2 prevalence, 3 regression, 4 layerwise widths.

**Framing (rule 9b), second re-frame.** OLD story: "plateaus are real but uninformative — no
detectable association (n=10)". NEW story: "plateau sharpness is *anti*-correlated with continuation
similarity — reading a plateau as evidence of a shared continuation gets the sign backwards". The
evidence that forced it is the n=200 bank, where the previously non-significant negative trend became
strongly significant and consistent across all three sharpness statistics and both models.
Unchanged and still supporting: plateaus are ubiquitous, the dissimilar control plateaus as sharply as
the test pairs, and the sharpening is supplied by depth downstream of the patch (Figure 4).
REPORT.md's title and Summary were rewritten to match; Methods gained the mining procedure, the
cluster bootstrap, the partial Spearman, the ln 2 ceiling, and the detectable-rho reference
rho_min(n) = tanh(1.96/sqrt(n-3)) (0.75 at n=10, 0.14 at n=200).
