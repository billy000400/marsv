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

## 2026-08-10 — S5: patch-depth replication turns the depth claim from descriptive to causal

**What changed.** Added Experiment 4: the entire 200-pair mined bank re-run per model with the patch
applied after **block 12** and after **block 20** instead of block 0 (11 and 3 remaining blocks below
the patch instead of 23), everything else held fixed — same prefixes, same token pairs, same 101-point
grid. Code: `experiments/run_interp.py` `sweep()` gained a `layer=` argument (default 0, so all prior
behavior is unchanged), `experiments/mine_pairs.py` takes the patch layer as `argv[1]` and writes
`results/bank_<model>_L<layer>.json|npz`; new `experiments/analyze_depth.py` writes
`results/depth_analysis.json` and `plots/depth_effect.png`.

**Nothing superseded.** Experiments 1–3 keep their numbers; this is a new experiment answering the
question REPORT.md previously listed as untested ("whether patching deeper changes the picture … we did
not run it"). The harness identity check now covers 1210 sweeps instead of 410, still at
|d(0)| <= 4e-4 and |d(1)-1| <= 4e-4 (worst case 3.5e-4).

**New results.**
- *Plateau strength vs patch depth* (median w_TV / % of pairs sharp, at 23 / 11 / 3 blocks below):
  gpt2-medium 0.080 / 82.0% -> 0.250 / 50.5% -> 0.383 / 10.0%; pythia-410m 0.266 / 47.5% ->
  0.419 / 2.5% -> 0.509 / 0.0%. Median w10-90 over the same sites: 0.241 -> 0.556 -> 0.701 and
  0.593 -> 0.749 -> 0.808. At block 20 pythia-410m is at the linear baseline (0.5 / 0.8) to within 2%
  and has zero sharp pairs.
- *Non-monotonicity is also depth-made*: share of monotonic gpt2-medium curves 7.5% -> 33% -> 72%.
- *The divergence association is separable from depth.* Spearman rho(JSD, w_TV) on the JSD<0.65 subset
  at 23 / 11 / 3 blocks below: gpt2-medium -0.61 [-0.70,-0.46] -> -0.53 [-0.64,-0.39] ->
  -0.53 [-0.66,-0.38] (all p <= 1.1e-11), i.e. flat; pythia-410m -0.45 [-0.63,-0.24] ->
  -0.44 [-0.62,-0.23] -> +0.04 [-0.11,+0.22] (p=0.62), vanishing exactly where the response became
  linear. CIs are 95% cluster bootstrap over the 40 prefixes.

**Figure.** Added `plots/depth_effect.png` as **Figure 5** (median w_TV and the ceiling-corrected rho,
both against patch site), embedded with a visible caption in RESULTS.md and REPORT.md. Figures 1–4
unchanged and still in reading order.

**Framing.** No re-frame: the headline finding is unchanged and this strengthens the mechanism section
from "sharpness accumulates with depth (read-out evidence)" to "removing the blocks below the patch
removes the plateau (intervention evidence)". REPORT.md's mechanism paragraph previously speculated
that the competition is resolved by the layers below the patch; Table 4 refines that — the *amount* of
compression scales with depth, but *which* pairs compress more is already decided in the last three
blocks of gpt2-medium. The limitation "we did not run it" was replaced by the model-depth caveat
(both models have 24 blocks) and the unattributed 82% vs 48% prevalence gap.

## 2026-08-10 — S6: third model family (OPT-350m); the inverted association replicates 3/3

**What changed.** Added `facebook/opt-350m` (331M, 24 blocks, $d_{model}$=1024) as a third model
family and re-ran every experiment on it: the 5 hand-picked pairs (S1+S2), the 200-pair mined bank at
block 0, and the same bank at blocks 12 and 20 (S5). This answers the open question the previous
REPORT.md listed as a limitation — whether the cross-model plateau-prevalence gap is a tokenizer
effect — because OPT's vocabulary is exactly GPT-2's 50257 token strings plus 8 specials and segments
our prompts identically, while pythia-410m uses the GPT-NeoX vocabulary. Code: `common.py` gained the
model entry + `m.model.decoder.layers` in `blocks()`; `run_interp.py` and `mine_pairs.py` take a model
list on argv and `run_interp.py` now merges into `results/summary.json` instead of overwriting it (so
the gpt2/pythia numbers are byte-identical to before, not re-run); `analyze*.py` iterate over three
models; `analyze_bank.py` gained `jsd_matched()` and the ceiling-subset $w_{10-90}$ correlation.

**Nothing superseded.** Every gpt2-medium and pythia-410m number in both deliverables is unchanged.
The counts that aggregate over models moved with the added cells: hand-picked cells 10 -> 15, total
sweeps 1210 -> 1815 (endpoint identity error still <= 4e-4, worst case 3.5e-4), "9 of 10 cells below
the linear baseline" -> "13 of 15", and the smallest detectable correlation for the hand-picked set
rho_min 0.75 (n=10) -> 0.51 (n=15).

**New results (opt-350m).**
- *Hand-picked pairs*: JSD / w10-90 / w_TV — `Mary`/`her` 0.038 / 0.734 / 0.356; `four`/`4`
  0.027 / 0.907 / 0.680; `four`/`Four` 0.472 / 0.530 / 0.293; `Au`/`79` 0.296 / 0.705 / 0.177;
  control `big`/`in` 0.646 / 0.143 / 0.068. The control is the sharpest of the five — the most vivid
  single instance of the inversion in the report.
- *Mined bank, block 0*: 61.0% of pairs sharp, median w_TV 0.221, median w10-90 0.511, 41.0%
  monotonic, JSD range 0.000–0.693 (35.5% at/above 0.65).
- *Association*: rho(JSD, w_TV) = -0.39 [-0.55,-0.21], p=1.3e-8; rho(JSD, w10-90) = -0.43
  [-0.56,-0.27], p=3.5e-10; rho(JSD, PF) = +0.43 [+0.28,+0.56], p=3.4e-10. Below the ln 2 ceiling
  (n=129): rho(w_TV) = -0.57 (p=1.3e-12), rho(w10-90) = -0.59 (p=3.1e-13). Partial rho controlling
  for the block-0 angle Omega: -0.44 (raw -0.39).
- *Depth*: median w_TV / % sharp at 23 / 11 / 3 blocks below the patch: 0.221 / 61.0% ->
  0.307 / 36.5% -> 0.420 / 1.0%; monotonic 41.0% -> 77.5% -> 99.5%; ceiling-corrected rho flat at
  -0.57 / -0.54 / -0.55 (all p <= 6.5e-11), matching gpt2-medium rather than pythia-410m.
- *Ceiling-subset w10-90 correlation added for the older models too* (new column in the Table 4 /
  ceiling table): gpt2-medium -0.54 (p=4.9e-12), pythia-410m -0.47 (p=2.3e-8).

**New experiment: divergence-matched cross-model comparison.** Median w_TV per model inside fixed JSD
bins, so the models are compared at equal endpoint divergence. gpt2-medium / opt-350m / pythia-410m —
bin 0.00–0.20: 0.263 / 0.496 / 0.421; 0.20–0.40: 0.103 / 0.317 / 0.276; 0.40–0.65: 0.043 / 0.147 /
0.220; 0.65–0.69: 0.047 / 0.166 / 0.274. gpt2-medium is sharpest in all four bins, so the prevalence
gap is a model property and not an artifact of bank composition; opt-350m shares gpt2-medium's
tokenizer yet plateaus 21 points less often and swaps rank with pythia-410m across the range, so the
tokenizer does not explain the gap. The previous limitation "the prevalence gap ... is not attributed
here to tokenizer or architecture" is replaced by "tokenizer ruled out; architecture, corpus and
pretraining length remain confounded".

**Figures.** Added `plots/jsd_matched.png` as **Figure 3** (median w_TV per JSD bin, three models),
embedded with a visible caption in both deliverables. Figures 1, 2, 4 (was 3), 5 (was 4) and 6 (was 5)
regenerated with a third model column/series; the old Figures 3–5 keep their content and are renumbered
in reading order. All six embedded in both files; `check_render.py` passes on both.

**Framing.** No re-frame. The headline claim is unchanged and now rests on three model families instead
of two, and the Summary/Conclusion were rewritten to say so, plus a new Conclusion corollary that the
calibration must be redone per model (Table 2).
