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

## 2026-08-10 — S7: depth-mismatched models; the plateau tracks RELATIVE depth, not block count

**What changed.** Added a fifth experiment answering the question the previous REPORT.md listed as a
limitation ("a 5-block model or a 60-block model could sit anywhere on the depth curve of Figure 6").
Experiment 4 moved the patch inside three models that all have 24 blocks, so "11 blocks below the
patch" and "just under half the stack below the patch" named the same runs. S7 separates the two
readings inside the GPT-2 family, where tokenizer, architecture and pretraining corpus are fixed and
only depth changes: `gpt2` (124M, 12 blocks, d_model 768) and `gpt2-large` (774M, 36 blocks, d_model
1280) alongside the existing `gpt2-medium` (24 blocks). Code: `common.py` gained the two model entries
and an `N_BLOCKS` map; `analyze_scaling.py` is new. `mine_pairs.py` and `run_interp.py` were not
touched — the existing patch-layer argument already covered this. 1800 new sweeps (200 mined pairs at
4 sites in gpt2-small and 5 sites in gpt2-large), ~50 min on the shared GPU; endpoint identity error
1.4e-4 (gpt2-small) and 1.4e-5 (gpt2-large), inside the report's existing <= 4e-4 bound.

**Nothing superseded.** Every gpt2-medium, pythia-410m and opt-350m number in both deliverables is
unchanged; the two new models are additions, and their banks are freshly mined (same 40 WikiText
prefixes and same rank draws, each model's own top-1 token). Counts that aggregate over runs moved:
total sweeps 1815 -> 3615. The hand-picked set stays at 15 cells in 3 models — the new models were run
on the mined bank only, and Table 1 / Figure 1 are unchanged.

**New results (Experiment 5).** Median w_TV / % sharp / median w10-90 per patch site:
- *gpt2-small (12 blocks)*: block 0 (11 below, f=1.000) 0.153 / 74.0% / 0.417; block 6 (5, f=0.455)
  0.289 / 35.5% / 0.632; block 8 (3, f=0.273) 0.363 / 12.0% / 0.703; block 10 (1, f=0.091)
  0.456 / 3.5% / 0.768.
- *gpt2-large (36 blocks)*: block 0 (35, f=1.000) 0.047 / 89.5% / 0.155; block 12 (23, f=0.657)
  0.255 / 47.0% / 0.570; block 18 (17, f=0.486) 0.342 / 22.5% / 0.673; block 24 (11, f=0.314)
  0.444 / 1.5% / 0.754; block 31 (4, f=0.114) 0.495 / 0.0% / 0.796.

**The headline of the new experiment.** The absolute reading is refuted by a direct comparison:
gpt2-large at block 12 and gpt2-medium at block 0 both have 23 blocks below the patch and give median
w_TV 0.255 vs 0.080 (factor 3.2) and 47.0% vs 82.0% sharp. Matching on the fraction f = (N-1-L)/(N-1)
instead halves the mean across-model spread of median w_TV, 0.212 -> 0.104, and is the only reading
under which the three models stay consistently ordered. Residual: at matched f the deeper model is
somewhat sharper (0.153 / 0.080 / 0.047 at f=1), a second-order effect that depth and width cannot be
told apart on inside this family.

**Association replicates 5/5 models.** Below the ln 2 ceiling at block 0, rho(JSD, w_TV) = -0.44
(gpt2-small, n=147, p=2.7e-8) and -0.64 (gpt2-large, n=137, p=6.1e-17, the strongest in the study).
Both new models follow pythia-410m rather than gpt2-medium on how rho behaves with depth: it survives
until the response goes linear and then collapses (+0.04, p=0.64 for gpt2-small with 1 block below;
-0.19 for gpt2-large with 4). This *narrows* an earlier claim: REPORT.md previously said the
divergence effect "is already fully expressed by the last three blocks"; that now holds only for the
two models that still produce a plateau there, and the floor-effect explanation given for pythia-410m
is supported rather than being special pleading.

**Figures.** Added `plots/depth_scaling.png` as **Figure 7** (median w_TV vs blocks-below and vs
fraction-below, three GPT-2 depths, with the mean across-model spread annotated in each panel),
embedded with a visible caption in both deliverables. Figures 1-6 are unchanged. Tables: the new
per-site table is **Table 6** and the matched-level spread table is **Table 7**; a draft numbering
collision with the existing Table 5 was corrected before commit, and three stale "Table 4" references
in the new section were repointed to Table 5.

**Limitations changed.** Removed "a 5-block model or a 60-block model could sit anywhere on the depth
curve of Figure 6" (now answered). Replaced with: the depth-scaling result rests on one family, where
residual width rises with depth (768/1024/1280), so the residual spread in Table 7 cannot be assigned
to depth or width separately, and relative depth has not been checked outside GPT-2 or outside the
12-36 block range.

**Framing.** No re-frame. The central claim is unchanged; the new experiment makes the depth half of
it portable to untested models and adds a third design corollary to the Conclusion (patch sites must
be quoted as a fraction of the stack, not a block number).

## 2026-08-10 — Operator feedback (human_feedback.txt): reproduction in GPT-2 Large, corrected framing, first test of the actual hypothesis

Addressed every point in `human_feedback.txt` (renamed `human_feedback.txt.addressed.md`). Both
deliverables were substantially restructured; REPORT.md was retitled.

**Point 1 — "GPT-2 Medium is not a reproduction of a GPT-2 Large result."** Accepted and acted on.
GPT-2 Large is now the primary model. Ran the full hand-picked set in `gpt2-large` and `gpt2-small` as
well as the three 24-block models (30 model-pair cells, up from 15). The point is confirmed: `big`/`in`
gives w10-90 = 0.044 in gpt2-large (near-step, PF 0.95) but 0.516 in gpt2-medium — above the plateau
criterion — and 0.691 in gpt2-small. REPORT.md now states plainly that a study swapping GPT-2 Large for
GPT-2 Medium is not testing the same phenomenon.

**Point 2 — "big/in is Matthew's positive plateau example, not a negative control; his non-plateau
comparison is big/large."** Accepted; this was a framing error throughout the previous report. Added
`The house was big / large` as a sixth pair in all five models, relabelled `big`/`in` as "M. plateau
case" and `big`/`large` as "M. smooth case" in `common.py`, Table 1 and every figure, and removed all
"dissimilar control" language. New numbers for big/large (JSD / w10-90 / w_TV): gpt2-large
0.053 / 0.592 / 0.292; gpt2-medium 0.042 / 0.719 / 0.398; gpt2-small 0.053 / 0.760 / 0.456; opt-350m
0.042 / 0.831 / 0.598; pythia-410m 0.042 / 0.802 / 0.505. It is the widest or second-widest transition
in every model, so Matthew's contrast reproduces in his model and his smooth case is smooth everywhere.

**Point 3 — "the hypothesis was changed; correlating JSD with transition width tests the wrong claim;
the missing independent variable is a measurement of circuit/feature difference."** Accepted. The
JSD-vs-width correlation is demoted to a clearly-labelled descriptive regularity with an explicit
statement that it does not bear on the hypothesis, and the report no longer claims it falsifies
anything. Added **Experiment 6**, the first direct test: output JSD is held LOW (< 0.1) instead of
varied; the independent variable is **IRD**, internal representational distance (mean over blocks of
1 - cos(h_A, h_B) at the final token); the dependent variable is **IPW**, intermediate-plateau width
(longest alpha span over which d stays within 0.10 of a level in [0.15, 0.85]; a linear response gives
0.10 by construction, so > 0.20 counts as an intermediate plateau). Result is null in both models:
gpt2-large rho(IRD, IPW) = +0.17 (p=0.31, n=38, 0.0% of pairs with an intermediate plateau);
gpt2-medium rho = -0.00 (p=0.99, n=32, 40.6%). Stated as under-powered (rho_min = 0.32 at n=38) and
proxy-based, with SAE/path-patching named as the sharper test.

**Point 4 — "'depth, not prompt content, produces the plateau' is too strong."** Accepted. The claim is
now "downstream depth is necessary, not sufficient", with big/large in gpt2-large as the direct
counterexample: 35 blocks below the patch — more than any condition in the depth table — and
w10-90 = 0.592.

**Point 5 — "the 'plateaus everywhere' claim is numerically overstated."** Accepted; the previous
"13 of 15 cells" used w_TV < 0.5, which is merely "better than linear", not the predefined criterion.
Corrected to the plan's predefined criterion throughout: **11 of 30** hand-picked cells have
w10-90 < 0.5 (14 of 30 under w_TV < 0.25), by model 5/6 gpt2-large, 3/6 gpt2-medium, 2/6 pythia-410m,
1/6 opt-350m, 0/6 gpt2-small. Table 2 now reports mined-bank prevalence under BOTH criteria: % with
w10-90 < 0.5 is 83.5 / 73.0 / 60.5 / 47.0 / 30.0 for gpt2-large / gpt2-medium / gpt2-small / opt-350m /
pythia-410m (previously only the w_TV numbers 89.5 / 82.0 / 74.0 / 61.0 / 47.5 were quoted).

**Re-framing (rule 9b).** Old story: "a plateau signals dissimilar continuations, not shared ones",
carried by the JSD-width correlation. New story: "the contrast reproduces in GPT-2 Large, relative
depth governs it, the base rate is the practical caution, and the hypothesis is still open". The
evidence that forced it is the operator's point 3 (the correlation tests a different claim) plus the
new gpt2-large runs. REPORT.md retitled from "A plateau in a single-token activation interpolation
signals *dissimilar* continuations, not shared ones" to "Activation-interpolation plateaus: what
reproduces, what depth explains, and what the hypothesis still needs".

**Superseded numbers.** Total sweeps 3615 -> 3645 (six pairs x five models replaces five x three).
Hand-picked cells 15 -> 30. Bank statistics for gpt2-small and gpt2-large added to Tables 2 and 3:
unsaturated rho(JSD, w_TV) = -0.44 (n=147) and -0.64 (n=137). No previously reported gpt2-medium,
opt-350m or pythia-410m bank number changed.

**Figures.** Added `plots/feature_plateau.png` as **Figure 8** (IRD vs IPW for low-JSD pairs in two
models, plus the two Matthew pairs in gpt2-large as reference curves). Figures 1, 2, 3, 4 and 5
regenerated over six pairs and five models; Figure 3 (bank_regression) and Figure 4 (jsd_matched)
swapped places so the association precedes the model comparison that depends on it. Figures 6 and 7
unchanged. All eight embedded with visible captions in both deliverables; `check_render.py` passes.
