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

---

## 2026-08-10 (S9 finished + S10) — feature-level hypothesis test lands in both deliverables; head ablation makes the circuit result causal

**Context.** The previous iteration ran S9 (`mine_lowjsd.py`, `circuit_features.py`) and got its
results into REPORT.md's Results section, but was cut off before finishing the curation: REPORT.md's
Conclusion still described the superseded n=38 IRD test, its sweep counts were stale, and RESULTS.md
had not been touched at all. This entry covers both the repair and a new experiment.

**Repair of the S9 curation.**
- REPORT.md Conclusion rewritten to match its own Results. Removed the superseded paragraph reporting
  the old under-powered test (rho = +0.17, p = 0.31, n = 38 in gpt2-large; rho = -0.00, p = 0.99,
  n = 32 in gpt2-medium) — that experiment is replaced, not supplemented, by the low-JSD banks
  (n = 365/399/356 from 102/119/113 prefixes). Limitations rewritten accordingly.
- Sweep totals corrected: Methods 4765 -> 4750 (30 hand-picked + 1000 block-0 bank + 1200 depth +
  1400 Experiment 5 sites + 1120 low-JSD), Results header 3645 -> 4750. Endpoint identity bound
  recomputed over every stored sweep: 4e-4 -> 3.5e-4 (worst case 3.53e-4, in the mined bank).
- RESULTS.md rewritten to current-best. Experiment 6 replaced entirely: the IRD-vs-IPW null
  (Figure 8 = feature_plateau.png) is gone, superseded by the feature-level version with SAE, head and
  neuron instruments (Figures 8-9 = sae_features.png, circuit_forest.png). `plots/feature_plateau.png`
  and `plots/jsd_vs_width.png` remain on disk but are no longer embedded in either deliverable.

**S10 (new) — `experiments/ablate_heads.py`, the intervention Experiment 6 called for.** For each
low-JSD pair, mean-ablate at the final token the k heads with the largest differential engagement
delta_h = (|c_h^A| + |c_h^B|)(1 - cos(c_h^A, c_h^B)), against a control set of the same size matched
one-for-one on total engagement but chosen to write similarly for the two prompts. Both endpoints and
all 101 alpha points are re-run under the ablation (identity check holds, |d(0)|, |d(1)-1| <= 3.5e-4).
Three pre-specified doses (3%, 6%, 10% of all heads) x 2 models x ~380 pairs = 4530 extra sweeps.

**New result (Experiment 7, Table 9, Figure 10 = `plots/ablation_causal.png`).** In gpt2-large the
association is causal and large: median w_TV 0.198 (no ablation) -> 0.358 at 3% of heads (+81%),
0.441 at 6%, 0.484 at 10% — the linear response (0.5) to within 3% — paired deltas +0.097 / +0.145 /
+0.199, 95% cluster-bootstrap CIs [+0.054,+0.146] / [+0.093,+0.201] / [+0.125,+0.268], Wilcoxon
p = 1.4e-43 / 1.8e-48 / 3.3e-47, 83-87% of pairs. The engagement-matched control does nothing
(0.198 -> 0.198 / 0.196 / 0.200). In gpt2-medium the same intervention gives +0.009 / +0.009 / +0.010
(p = 0.019 / 0.010 / 0.014), ~15x smaller, 55-56% of pairs. Manipulation check: the differential
ablation leaves 0.76/0.65/0.54 (gpt2-large) and 0.71/0.61/0.52 (gpt2-medium) of the unablated HCD,
while the control leaves 1.02-1.08. Removed-magnitude ratio differential:control is 1.01-1.02
(gpt2-large) and 1.08-1.12 (gpt2-medium).

**Story change (rule 9b).** Old: "the endpoint-plateau reading of the hypothesis holds as an
association; an intervention is the natural next step." New: "in the model where the phenomenon was
reported, a small pair-specific set of attention heads causally produces the sharp switch." REPORT.md
retitled from "... and what the hypothesis still needs" to "... and which heads cause the switch", and
the Summary gained a paragraph for the causal result. The framing offered to a user of this probe is
now that a plateau is two things at once: relative depth supplies the capacity to compress the change,
and prompt-discriminating heads decide whether that capacity is used.

**Figures.** Added `plots/ablation_causal.png` as **Figure 10** in both deliverables (2x2: median w_TV
vs dose per model, the paired difference with cluster-bootstrap CIs on a symlog axis, and the HCD
manipulation check). All ten figures embedded with visible numbered captions in both files, each now
cited by number in the prose that motivates it. `check_render.py` passes on both.

---

## 2026-08-11 (S11) — the differential heads are a shared circuit, and most of their effect comes from above the patch

**New experiments.** `experiments/localize_heads.py` (recurrence + held-out fixed-set ablation in
gpt2-large and gpt2-medium; the Experiment 7 dose sweep extended to gpt2-small),
`experiments/localize_depth.py` (the same fixed-set ablation in gpt2-large with block 0 struck from the
ranking) and `experiments/head_depth_share.py` (where gpt2-small's differential heads sit). 1111 new
held-out sweeps plus 2190 for gpt2-small's dose sweep.

**New result 1 — recurrence (Table 10).** Per-pair differential head sets overlap across prefixes at
$J = 0.090$ / $0.064$ / $0.280$ (gpt2-large / -medium / -small) against a random-set null of $0.016$;
the most-selected head enters 78.9% / 46.1% / 85.8% of pairs. GPT-2 Large's overlap is below its
magnitude-ranked set's ($0.160$) and its 22 most frequent heads carry 30.7% of all selections, so the
picture is a recurring core plus a pair-specific tail.

**New result 2 — a fixed cross-pair set beats per-pair selection (Table 11, Figure 11C).** Ranking heads
on one half of the prefixes and ablating that single 22-head set on the held-out half moves gpt2-large's
median $w_{TV}$ 0.198 -> 0.485 (paired $\Delta$ +0.189, CI [+0.140,+0.249], $p = 4\times10^{-51}$ vs the
matched control), against 0.358 for per-pair sets ($p = 1\times10^{-17}$ for the difference), i.e.
recovery 198% at 29.4% head overlap. gpt2-medium: 0.257 -> 0.254, recovery 70%, $p = 0.033$.

**New result 3 — the effect is mostly upstream of the patch (Table 12).** The most-selected heads sit in
block 0, which the patch overwrites, so they act on the interpolated endpoints rather than on the
computation below. Rebuilding the fixed set from blocks 1-35 leaves 0.198 -> 0.217 (+0.012, CI
[+0.009,+0.017], $p = 5\times10^{-24}$ vs control), 13% of the per-pair effect and 6% of the full fixed
set's.

**New result 4 — gpt2-small joins Experiment 7 (Table 9).** Paired $\Delta$ = +0.014 / +0.019 / +0.025
at the 3/6/10% doses ($p = 1.7\times10^{-4}$ / $1.6\times10^{-3}$ / $6.5\times10^{-4}$, 59-63% of 365
pairs). The intervention effect is therefore NOT ordered by model size (large +0.096 >> small +0.014 >
medium +0.009), and block-0 share does not explain it either (62.6% in gpt2-small vs 16.7% in
gpt2-large), so the cross-model gap is now explicitly described-not-attributed.

**Superseded numbers.** Sweep total 4750 -> 12581 (Experiment 7 now covers three models: 4530 -> 6720
ablated sweeps; Experiment 8 adds 1111). Experiment 7's "pair-specific set of attention heads" claim is
replaced by "a shared core, mostly upstream of the patch" — the earlier wording is removed from
RESULTS.md, REPORT.md's Summary, Results and Conclusion. The Experiment 7 paragraph "In GPT-2 Medium the
same intervention barely moves the curve" is rewritten as "In the two smaller GPT-2 models …" and no
longer implies a depth trend. Endpoint identity bound unchanged (worst case over the new runs
$2.5\times10^{-4}$, inside the reported $3.6\times10^{-4}$).

**Story change (rule 9b).** Old: "a small, pair-specific set of attention heads causally produces the
sharp switch in GPT-2 Large." New: "the set is shared across pairs and can be named once, but it works
mainly from above the patch site, so a plateau is evidence about the two interpolated endpoints as much
as about the depth that processes them." The evidence forcing it is the 94% drop when block 0 is
excluded. REPORT.md's title is unchanged; its Summary gained a paragraph and its Conclusion's final
reframing paragraph was rewritten.

**Figures.** Added `plots/localization.png` as **Figure 11** in both deliverables (A recurrence curve,
B depth profile of selected heads for three models, C held-out fixed-set ablation with the
block-0-excluded condition, D three-model dose response). Figure 10's caption no longer names series by
colour (rule 13). All eleven figures are embedded with visible numbered captions in both files and cited
by number; `experiments/check_render.py` passes on both.

---
## 2026-08-11 (S12) — the head circuit's causal effect is contingent on depth below the patch

**New experiment.** `experiments/depth_gap.py` (+ `experiments/depth_gap_table.py`): Experiment 8's
held-out fixed-set ablation repeated with the patch moved to the middle block of each stack (block 6 of
12, 12 of 24, 18 of 36; $f = 0.455 / 0.478 / 0.486$), plus the block-0 fixed-set run for gpt2-small that
S11 skipped. Three conditions per pair at the mid site (no ablation, per-pair matched control, held-out
fixed set), everything else — head selection, prefix-parity folds, 3% dose, matched control — unchanged.
3725 new sweeps; worst endpoint identity error $2.1\times10^{-4}$, inside the reported
$3.6\times10^{-4}$.

**New result 1 — the effect collapses when the patch moves (Table 13, Figure 12).** Paired
$\Delta = w_{TV}$(fixed set) $-$ $w_{TV}$(control), block 0 -> mid block: gpt2-large $+0.187$ ->
$-0.002$, gpt2-small $+0.015$ -> $+0.003$ (ns), gpt2-medium $+0.005$ -> $+0.002$ (ns). The unablated
switch is gone at the mid site (median $w_{TV}$ 0.501 / 0.448 / 0.420 against the linear response 0.5),
so there is no compression left for the heads to supply.

**New result 2 — it is not a ceiling artifact.** The headroom-normalised effect
$\hat\Delta = \Delta / (0.5 - \tilde w_{TV}(\text{control}))$ goes 61.9% -> undefined (control past the
linear response) in gpt2-large, 8.1% -> 5.0% in gpt2-small and 2.0% -> 2.0% in gpt2-medium: no cell
keeps a large normalised effect after the move.

**New result 3 — gpt2-small's block-0 fixed set (new row of Table 13).** $\Delta = +0.015$,
CI $[+0.006, +0.021]$, $p = 1.6\times10^{-3}$, $n = 365$; recurrence-ranked $k=4$ set, 100% of it in
block 0. This completes the three-model comparison at the block-0 site, which S11 ran for two models.

**Superseded numbers.** Sweep total 12581 -> 16306 (Methods and the harness-check sentence in both
files). The fixed set's block-0 membership in gpt2-large was stated as "five heads"; recounting the
stored held-out sets gives **seven of 22** in both folds (32%) — corrected in RESULTS.md's headline and
REPORT.md's Summary. Experiment 8's closing "what makes GPT-2 Large special is still open" is now
qualified: still open, but localised to the $f = 1$ patch site rather than the model.

**Story change (rule 9b).** Old: two sources of sharpness — depth below the patch, and early
prompt-discriminating heads — presented as separate contributions a curve cannot disentangle. New: they
multiply. Removing the depth also removes the heads' causal effect, so a head-ablation result of this
kind is interpretable only at a patch site where the unablated curve plateaus. Added as a Summary
paragraph, a Conclusion paragraph and the new Results subsection in REPORT.md, and as a headline
paragraph plus Experiment 9 in RESULTS.md. The claim that relative depth might explain the cross-model
gap is withdrawn with its reason stated: the three models were already matched at $f = 1$, so a second
matched-$f$ site cannot equalise them and in fact silences all three.

**Figures.** Added `plots/depth_gap.png` as **Figure 12** in both deliverables (A block-0 patch, B
mid-stack patch, both as no-ablation / matched-control / fixed-set bars per model; C the paired effect
against relative depth on a symlog axis with cluster-bootstrap intervals). Twelve figures embedded with
visible numbered captions in both files, each cited by number; `experiments/check_render.py` passes on
both.

---
## 2026-08-11 (S13) — the plateau does not depend on the interpolated token being the last token

**New experiment.** `experiments/offset_position.py` (+ `experiments/analyze_offset.py`): the model's
own greedy continuation of the A prompt, $s \in \lbrace 0,1,2,4 \rbrace$ tokens, is appended to *both*
prompts of each low-JSD pair; the block-0 SLERP patch stays at the differing position and the logits are
read $s$ tokens downstream. Evenly spaced subsamples of each bank (120 / 60 / 45 pairs in gpt2-small /
-medium / -large) at four suffix lengths: 900 new sweeps. $s = 0$ reproduces the stored `lowjsd_*` sweeps
with a maximum $w_{TV}$ difference of exactly 0 — the harness check for this experiment.

**New result 1 — the switch is invariant to the readout position (Table 14, Figure 13A/C).** Paired
median $w_{TV}$, $s{=}0 \to s{=}4$: gpt2-large $0.148 \to 0.193$ ($\Delta = +0.001$, CI
$[-0.015,+0.026]$, $p = 0.65$), gpt2-medium $0.252 \to 0.284$ ($+0.019$, $p = 0.11$), gpt2-small
$0.311 \to 0.303$ ($-0.003$, $p = 0.60$). Percent sharp $60.0 \to 53.3$ / $48.3 \to 45.0$ /
$25.8 \to 28.3$. This removes the one methodological escape route the whole report shared.

**New result 2 — endpoint divergence does not set sharpness within a pair (Figure 13B).** The shared
continuation collapses median endpoint JSD 15–16-fold ($0.0499 \to 0.0034$ gpt2-large, $0.0344 \to
0.0021$ gpt2-medium, $0.0378 \to 0.0024$ gpt2-small) with no matching change in width. Experiment 3's
across-pair regularity is therefore demoted from a candidate driver to a marker of feature disjointness
(Experiment 6), stated in RESULTS.md's Experiment 10, REPORT.md's new Results subsection and Conclusion.

**Superseded numbers.** Sweep total 16306 -> 17206 (Methods and the harness-check sentence in both
files). The report-wide endpoint identity bound $3.6\times10^{-4}$ now carries a stated exception: for
$s > 0$ the two endpoint logit vectors come within $10^{-3}$ of each other and the bound is
$2.1\times10^{-3}$. Endpoint references for this experiment are computed inside the identical batched
forward path as the swept rows, because float32 kernels vary with batch shape; with batch-1 references
the same sweeps gave errors up to $9.8\times10^{-1}$ on near-degenerate endpoints.

**Story change (rule 9b).** No reversal, one demotion and one strengthening. Endpoint divergence moves
from "a descriptive regularity that tracks sharpness" to "a marker, not the quantity that sets
sharpness", on the strength of the first within-pair manipulation in the report. The last-token design
is promoted from an unexamined assumption to a tested and discharged one.

**Figures.** Added `plots/offset_position.png` as **Figure 13** in both deliverables (A median $w_{TV}$
vs suffix length with bootstrap intervals, B endpoint JSD collapse on a log axis, C mean switch curves
at $s = 0$ and $s = 4$). Thirteen figures embedded with visible numbered captions in both files, each
cited by number; `experiments/check_render.py` passes on both.
