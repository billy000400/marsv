# JOURNAL — Does training-corpus continuation JSD predict plateau strength?

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-08-02 — iteration 1: S1 through S5 complete in one pass

**Feedback check.** Listed the direction root for `human_feedback*.md` / `*REVIEW*` without the
`.addressed.md` suffix: **none present**. Proceeded with the plan.

**What I did.**

1. **Preflight (the plan's stated make-or-break).** Downloaded only the first 4 KB of
   `document.idx` and parsed the official Megatron header: magic `MMIDIDX`, version 1, dtype code 8 =
   `uint16`, `len = 146,432,000`, every listed size 2049. Confirmed the layout arithmetic is exact —
   predicted idx size `34 + 12L + 8D = 1,757,184,042` equals the actual file size, and the 21 `.bin`
   shards total `600,078,336,000 = 146,432,000 x 2049 x 2` bytes. **Consequence: sequence `i` starts
   at byte `4098i`, so a row-aligned sample is a plain HTTP byte range** — no unsharding, no 602 GB
   download, and no need for the official reader at run time.
2. **Corpus sample.** Two distant, row-aligned 500,000-row splits: A at global row 1,000,000
   (shard 0) and B at 73,300,000 (shard 10), each wholly inside one shard. Verified both by decoding
   rows at head/middle/tail (real Pile text; all IDs < 50,277).
3. **S1 JSD table.** Counted endpoint->successor bigrams *inside* rows only, in two halves per split
   so the same-token split-half noise floor came free (no second data pass). Both gates passed by a
   wide margin: `Spearman(JSD_A,JSD_B) = 0.9998`, noise ratio `0.072`.
4. **S2 frozen bank.** 75 endpoint-disjoint pairs, 15 per `JSD_A` quintile, balanced on frequency and
   surprisal. Frozen before viewing any curve.
5. **S3 assay + calibration.** Norm-rescaled SLERP on the post-block-0 final-position residual,
   patched, read out at the logits. Calibration gate passed (100% valid curves, `IQR(w) = 0.115`).
6. **S4/S5.** Full bank on 1.4B step143000, 1.4B step0, 410M step143000; block scan; validity checks;
   all figures and statistics.

**What I learned.**

- **The result is clean and matches the prespecified positive branch.** `rho(JSD_B, w) = -0.419`
  [-0.585, -0.222] trained versus `-0.155` [-0.368, +0.068] at step 0, replicating at 410M
  (`-0.320`). Bin medians are monotone in both trained models.
- **The step-0 control is stronger than I expected.** It is not merely "no correlation": the
  untrained network has median `w = 0.831` with `IQR = 0.004`, i.e. it sits at the linear-response
  value 0.8 with essentially *no* plateau structure and *no* variance to correlate with. That makes
  the trained-model effect much harder to attribute to tokenisation or architecture geometry.
- **The predictor is validated, so a null would have been interpretable.** `rho(JSD_B, output JSD)
  = +0.729` at step143000 — a global, context-free corpus statistic really does predict how the model
  splits these words inside one specific sentence.
- **But geometry adjustment attenuates it a lot** (`-0.419 -> -0.267`). Since block-0 endpoint
  geometry plausibly lies *on the causal path* from training targets to plateau shape, the plan's own
  guidance applies: report the total association, do not claim explanation beyond learned geometry.
- **Engineering lesson that cost the most time:** the shared MooseFS volume intermittently returns
  `EDQUOT` under 4 concurrent agents. It silently killed the downloader repeatedly, tore a
  non-atomic JSON ledger, and at one point **truncated a source file I had written from a shell
  heredoc to 0 bytes**. Fixes: bulk data moved to local `/tmp`, ledger writes made atomic via
  `os.replace`, a self-restarting supervisor around the resumable downloader, and file edits done
  through the editor tool rather than shell heredocs.

**Assumptions logged (loop mode — could not ask).**

- *Top-K endpoint filter.* The plan prespecified top-256, which yields 134 tokens = at most 67
  endpoint-disjoint pairs, short of the 75-pair target. Options were (a) shrink the bank to 67,
  (b) use a dependent all-pairs design, (c) relax to top-512. **Chose (c)** — the plan explicitly
  rejects (b), and top-512 is still the top 2.8% of 18,714 eligible word tokens, so endpoints stay
  firmly in-distribution while the independent design and the 75-pair target both survive. Rejected
  (a) because it costs power for no design benefit. Recorded in the manifest, REPORT.md Methods, and
  CHANGELOG.md; the strict top-256 subset (n = 12) is reported as an underpowered sensitivity check.
- *Counting pool.* Counted a top-1024 superset (527 tokens) in the single data pass so that the
  top-256 vs top-512 decision needed no re-count. This is a compute choice, not a selection choice —
  the bank rules were fixed before the counts were inspected.
- *`d(t)` is computed in logit space* (endpoint logit vectors restricted to the 50,060 valid target
  IDs), which is how the plan's "record final-position logits ... and compute `d(t)`" reads.

**Bugs caught before they polluted results.** (i) `build_pairs.py` indexed the per-endpoint
distribution matrix by eligible-set position while the matrix rows were pool positions — would have
scrambled every pair's JSD; fixed and rerun before the bank was frozen. (ii) The 64 MB-chunk ledger
was migrated to 16 MB units rather than silently re-downloading split A.

**Next step.** Optional formation subset from the plan's fixed setup: run the same frozen bank at
`step1000`, `step8000`, `step32000`, `step64000` to test whether the negative relationship
*strengthens* during training. Everything in the plan's definition of done is already satisfied, so
this is additive; if the intermediate checkpoints cannot be downloaded, the deliverables stand as-is
and I write `STOP`.

On track? yes — S1-S5 complete (definition of done met, ~90%), no blocker; remaining work is the
optional formation-during-training subset.

---

## 2026-08-02 — iteration 1 (continued): formation subset, and the plan's expectation is refuted

**What I did.** Ran the same frozen 75-pair bank on `pythia-1.4b-deduped` at `step1000`, `step8000`,
`step32000`, `step64000` (`experiments/formation.py` fetches each checkpoint to local `/tmp`, assays
it, then deletes it, so peak disk stays at one checkpoint — the shared volume cannot hold four).
Added `plots/formation.png` as Figure 6 in both deliverables and renumbered the block scan to
Figure 7. Ran the full 75-pair bank rather than the plan's "30 frozen pairs" — same wall-clock cost
(~1 min per checkpoint), strictly more power, and it keeps every checkpoint on one identical bank.

**What I learned — this is the interesting part.** The plan's stated expectation was that the
negative relationship would *strengthen* during training. **It does the opposite.**

| step | 0 | 1000 | 8000 | 32000 | 64000 | 143000 |
|---|---|---|---|---|---|---|
| rho(JSD_B, w) | -0.155 | **-0.660** | -0.605 | -0.524 | -0.539 | -0.419 |
| rho(JSD_B, out JSD) | -0.144 | +0.779 | +0.693 | +0.726 | +0.714 | +0.729 |
| median w | 0.831 | 0.758 | 0.624 | 0.582 | 0.541 | 0.562 |

Two things move differently, and separating them is what makes this worth reporting:

- **Sharpness keeps increasing** — median `w` falls monotonically from 0.831 (no plateau, sitting at
  the linear-response value 0.8) to ~0.55, with the IQR widening from 0.004 to ~0.11. Plateau
  structure forms early and keeps deepening.
- **The corpus predictor peaks at the earliest checkpoint I ran and then fades** — `-0.660` at step
  1000 decaying to `-0.419` at step 143000. Meanwhile its correlation with the model's own *output*
  divergence is flat from step 1000 on (+0.78 -> +0.73), so the predictor is not degrading in
  general; it specifically explains a shrinking share of *which* pairs are sharp.

My reading (stated as a suggestion, not a finding, in REPORT.md): early training is dominated by
corpus-level continuation statistics, and later training adds context-sensitive structure that a
context-free `P(y|a)` cannot capture. The obvious follow-up is a context-conditioned divergence
estimate, which this direction's plan explicitly put out of scope.

**Caveat I made sure to state in the deliverable.** One trajectory, one bank, one model, and the
earliest checkpoint sampled is step 1000 — the relationship could peak earlier still, and I have no
resolution below 1000 steps to say where it forms. So this is an observation, not a training-dynamics
result.

**Deliverable state.** RESULTS.md and REPORT.md re-curated to current-best with 7 figures embedded
in both, sequentially numbered in reading order, each with a visible caption; `check_render.py`
passes (7 display equations, 0 `<pre lang="math">`, all embeds captioned). CHANGELOG.md has a second
dated entry appended; no earlier entry was rewritten and no number was superseded.

**Next step.** The plan's definition of done is fully met and the optional item is now also done. No
unaddressed `human_feedback*.md` / `*REVIEW*` files exist, so `STOP` is written. If a future
iteration finds new feedback next to the stale `STOP`, delete `STOP`, address the feedback, and only
re-write it when clean.

On track? yes — S1-S6 complete (100%), no blocker; STOP written with zero unaddressed feedback files.

---

## 2026-08-03 — iteration 2: operator feedback (all three points), primary bank rebuilt

**Feedback check.** `human_feedback.txt` present and unaddressed (the rule's glob is
`human_feedback*.md`, but this is plainly operator feedback and I treated it as such). Its point 3 is
**truncated mid-sentence** in the file — it ends at "* (\rho(JSD_{\text{corpus}},JS". I addressed the
part that is legible ("Correct the interpretation. The current evidence shows that corpus JSD predicts
learned output separation and overall transition width") and, per loop-mode rules, logged the
assumption rather than blocking: I read it as *stop calling the outcome "plateau sharpening" when `w`
measures the whole transition*, and answered it with (a) the wording change, (b) a new flatness metric
that tests the plateau claim directly, (c) two further self-corrections the same logic implies (the
step-0 floor effect, and withdrawing the "predictor decays during training" reading). Renamed the file
`human_feedback.addressed.md`. Full detail of every change is in CHANGELOG.md.

**What I did.**

1. **Strict curve validity** (`experiments/curve_metrics.py`, `rescore.py`). Span + single-crossing +
   monotonicity (backslide <= 0.02), applied per curve; pair-level rule needs >= 2 of 3 valid contexts.
   Every saved curve set re-scored into `results/qc_<tag>.json`; all raw curves exported to committed
   `results/curves_*.csv.gz` alongside the existing `.npy`.
2. **Rebuilt the bank at the prespecified top-256 filter** (60 pairs, was top-512/75) and re-ran
   everything on it: step143000, step0, 410M, the four formation checkpoints, the block scan.
3. **Reworked `analyze.py`** to treat top-256 as primary and top-512 as labelled post-hoc secondary,
   and added three figures (all raw curves; edge drift; bank comparison).
4. Re-curated RESULTS.md and REPORT.md to current-best with 10 captioned figures each; render check
   passes.

**What I learned.**

- **The lax `width()` was a real weakness but not a real bug.** The reviewer was right that the old
  first-crossing search could accept a non-monotone or multi-transition curve. It never had to: across
  1,080 curves the largest backslide is exactly **0.0000** and no curve crosses either level twice. So
  the previously reported 1.000 valid rate was correct — it just wasn't evidence until now.
- **The prespecified bank is *stronger*, not weaker.** Losing 15 pairs cost nothing: rho went
  -0.419 -> **-0.525** at 1.4B and -0.320 -> **-0.512** at 410M. My earlier reasoning for relaxing to
  top-512 ("keeps the 75-pair target and the power") was wrong on its own terms — the top-256
  endpoints are the ones the model most strongly prefers in these contexts, and the effect is cleaner
  there. Lesson: a deviation taken to protect statistical power is worth testing before adopting.
- **The "predictor decays during training" story does not survive the correct bank.** On top-512 the
  trajectory looked monotone (-0.660 -> -0.419); on top-256 it is -0.582, -0.456, -0.408, -0.628,
  -0.525 with heavily overlapping CIs. I withdrew the decay claim and kept only what is robust: the
  relationship is fully formed by step 1000 and does **not** strengthen, refuting the plan's
  expectation.
- **The curves really are plateau-shaped, but width and flatness are the same measurement.** Edge
  drift is 0.076 trained versus the 0.184 straight-line reference (and 0.213 at step 0 — the untrained
  curves are, if anything, slightly *anti*-plateau), which justifies the word "plateau" at the level of
  the curve shape. But Spearman(w, E) = +0.971, so nothing in this design can attribute the
  *correlation* to flatness rather than width. Both deliverables now say this explicitly.
- **Bin medians are not monotone at 1.4B** on the new bank (Q3 = 0.462 dips below Q4/Q5). With ~12
  pairs per bin that is expected noise, but the old caption claiming monotonicity was overstated and is
  gone.

**Assumptions logged (loop mode — could not ask).**

- Treated `human_feedback.txt` as an operator feedback file despite the `.txt` extension; renamed to
  `human_feedback.addressed.md` (contents untouched).
- Rebuilding the frozen bank after curves have been seen technically breaks "never revise this bank",
  but the *rule* used to build it is the prespecified one, so this restores preregistration rather than
  breaking it. The only post-hoc element is that I already knew the top-512 answer; that is why the
  top-512 bank stays in the report, labelled post-hoc, instead of being deleted.
- Monotonicity tolerance 0.02 and the outer-20% window for edge drift are my choices, not the plan's;
  both are stated in Methods. The validity verdict is insensitive to the tolerance (max backslide is
  0.0000, so any tolerance >= 0 gives the same answer).
- Ran the formation subset on all 60 pairs rather than the plan's "30 frozen pairs" — same wall-clock,
  strictly more power, one identical bank at every checkpoint.

**Next step.** Nothing outstanding: the plan's definition of done is met on the prespecified bank and
all feedback is addressed. If no further feedback arrives, the direction is finished. The natural
follow-up — explicitly out of scope here — is a context-conditioned divergence estimate, to test
whether it predicts width better than the global one at the late checkpoints where the global
predictor stops improving.

On track? yes — S1-S6 complete (100%) on the prespecified top-256 bank, no blocker; feedback file
addressed and renamed.

---

## 2026-08-03 (iter: operator feedback #2 — auditability, mediation, corrected wording)

**Situation on entry.** A new `human_feedback.txt` (operator review of commit c03d510, verdict *minor
revision*) sat unaddressed next to no `STOP`. The operator had re-run the primary bank and reproduced
`rho = -0.525, p = 1.68e-5, n = 60` and step 0 `-0.056`, confirming the top-256 rebuild, the
endpoint-disjointness and the new validity code. Four remaining issues, all addressed this iteration;
no GPU work was needed — everything came from the saved curves/QC JSONs, so the whole iteration was
CPU-only.

**1. Curves were never actually committed (severity: highest).** I had written three times that
`results/curves_*.npy` / `.csv.gz` are committed. They existed on disk but the repo-root `.gitignore`
excludes `*.npy` and `*.gz`, so git never saw them. Fixed by adding a direction-level `.gitignore` with
`!results/curves_*.npy` / `!results/curves_*.csv.gz` — a nested .gitignore's negation beats the parent
rule, confirmed with `git check-ignore -v` (now reports the negating line) and `git status` (all 26
curve files now listed as untracked, so the wrapper's commit will pick them up; 1.6 MB total). Chose
un-ignoring over deleting the claim because independent recomputation of the QC numbers is exactly
what makes the strict-validity result checkable.

**2. Mediation + learned sharpening (new `experiments/revisions.py`).** Two analyses the operator had
asked for and I had not done. Learned sharpening `dw = w(trained) - w(step 0)` removes each pair's own
untrained baseline: `rho(JSD_B, dw) = -0.517` [-0.694, -0.294], p = 2.3e-5 — essentially identical to
the total, so the headline is not an artefact of pairs that start sharp. All 60 `dw < 0` (median
-0.287): training narrows every pair. The adjustment ladder is the substantive correction: -0.525 ->
-0.277 (adjusting for the model's own output JSD, the obvious mediator) -> **-0.204, p = 0.119, not
significant** (mediator + the five covariates). My reproduced numbers match the operator's exactly. So
the report now says plainly: strong total association, no significant *independent* one. Note the
operator's two "controlling" numbers are the `w` ladder, not the `dw` ladder (`dw` gives -0.263 /
-0.198); I report both so neither reading is hidden.

**3. Training-dynamics wording was wrong.** "Transitions keep sharpening throughout training" is
contradicted by my own table: median `w` 0.512 at 64k -> 0.541 at final. Tested it at the pair level —
38/60 blunter, paired Wilcoxon p = 0.0052, median delta +0.012 — so the reversal is systematic, not a
median artefact. Text, panel title and caption now read "sharpens through 64k, then a modest late
reversal", and `formation.png` gained a third panel (per-pair 64k vs final scatter against y = x).

**4. The filter never enforced complete words.** `common.py` tests only the `Ġ` word-start marker,
lowercase alphabetic characters and length >= 2. Of the bank's 120 endpoints exactly one is a fragment
(`un`). Rather than silently re-freeze the bank (which would break preregistration for cosmetic
reasons), I corrected the description to "word-start tokens" everywhere including the code docstring,
and added the sensitivity check: dropping `un`/`better` gives `rho = -0.502`, p = 5.2e-5 (n = 59) —
matching the operator's number — plotted as a third series in the bank-comparison figure.

**Learned / worth remembering.** (a) A "committed" claim about artifacts is worth verifying with
`git check-ignore`, not `ls` — the files were there the whole time and still absent from the repo.
(b) An unadjusted correlation and its fully adjusted version can tell genuinely different stories
(-0.525 at p = 1.7e-5 versus -0.204 at p = 0.12); reporting only the first is what the operator
objected to, and the fix is to state which one is the claim. (c) `/tmp/dir18_data` (the corpus cache)
does not survive across sessions, so `analyze.py` now skips the reliability figure when the cache is
gone rather than crashing; the committed PNG stays current-best.

**Deliverables.** RESULTS.md and REPORT.md curated to current-best with 11 captioned figures each (new
Figure 8 = mediation; formation, bank comparison and block scan renumbered 9/10/11), `check_render.py`
passes on both. CHANGELOG entry appended. `human_feedback.txt` -> `human_feedback_2.addressed.md`.

**Next step.** Nothing outstanding — the plan's definition of done holds and zero unaddressed feedback
files remain, so `STOP` is written. If new feedback appears next to it, delete `STOP`, address it, and
re-write `STOP` only when clean. The out-of-scope follow-up is unchanged: a context-conditioned
divergence estimate, which is also the natural way to attack the mediation null (a predictor that is
not just a proxy for the model's own output separation).

On track? yes — S1-S6 complete (100%), operator feedback #2 fully addressed, no blocker.

---

## 2026-08-03 (iter: operator feedback #3 — 1,000-pair generality test + terminology overhaul)

**Situation on entry.** `human_feedback_3.txt` sat unaddressed (no `STOP` present). 26 points: mostly
definition/claim precision, plus two real experiments — a selection-split sensitivity check and a
~1,000-pair secondary analysis with endpoint-clustered inference, explicitly to be run on the final
1.4B checkpoint first.

**Blocker I hit first:** `/tmp/dir18_data` (corpus sample + bigram counts + `reliability_bank.npz`) is
wiped between sessions, and the whole 1,000-pair bank needs pairwise JSD from those counts. So the
iteration began by re-downloading both 500k-row splits by byte range (~5 min each, resumable ledger)
and re-running `count_jsd.py` (~4 min). Everything reproduced bit-for-bit: 50,060 valid target IDs,
123 eligible endpoints, Spearman(J_sel, J_hold) = 0.9998, noise ratio 0.0723. Worth remembering: the
models are cached under `/workspace/hf_home` and survive, the corpus is not.

**What I did.**

1. **`build_large_bank.py`** — 1,000 pairs from the same 123 endpoints, same frequency-ratio rule,
   endpoint-disjointness replaced by a cap of 20 uses per endpoint (actual uses 1/17/20), 200 pairs
   per selection-split quintile, seed 7, no curve consulted. Also precomputes the full 123x123
   holdout-JSD matrix, which the permutation test needs.
2. **Assay** at step143000 and step0 (3,000 curves each; ~8 min per checkpoint on the shared GPU).
   All curves valid, max backslide 0.0000.
3. **`large_analysis.py`** — endpoint-clustered inference only: dyadic (pigeonhole) endpoint
   bootstrap, endpoint-label (QAP) permutation test, 10 non-overlapping binned medians for the
   nonlinearity question, and the naive pair bootstrap shown *only* to quantify the correction.
   Trained: rho = **-0.486**, clustered CI [-0.603, -0.353], permutation p < 0.00025 (0/4,000 reached
   it; 97.5th pct of |rho_perm| was 0.116). Step 0: -0.008, CI [-0.126, +0.109], p = 0.86.
4. **`split_sensitivity.py`** — reproduces the operator's numbers exactly: -0.5260 (selection) vs
   -0.5247 (holdout), rho(J_sel, J_hold) = 0.99972.
5. **Terminology overhaul** across both deliverables *and* the figure-generating code (splits renamed
   selection/holdout, endpoints (u,v), J_hold notation on every axis and legend), plus the ~20 claim
   corrections listed in CHANGELOG.

**What I learned.**

- **Endpoint clustering costs a factor of 2.6 in the interval, not the conclusion.** The naive pair
  bootstrap gives [-0.533, -0.437] (SD 0.025) and the dyadic endpoint bootstrap [-0.603, -0.353]
  (SD 0.064) on the same 1,000 pairs. The naive p-value (2.6e-60) is meaningless; the permutation test
  still gives < 0.00025. So the operator's warning was right in principle and the effect survives it.
- **The relationship is monotone, not thresholded.** Ten binned medians fall 0.649 -> 0.499 with a
  mild flattening above ~0.75 bits. That was an open possibility the 60-pair bank could not resolve.
- **The larger bank is slightly weaker (-0.486 vs -0.525), as expected** — it is not matched
  pair-by-pair on frequency/surprisal and fills the crowded middle of the divergence range. Reported
  as such rather than as a discrepancy.
- **The 1,000-pair step-0 control tightens the untrained null** from "wide CI containing 0" to
  [-0.126, +0.109]. The restricted-range caveat still stands (untrained IQR(w) = 0.005), so I kept it.
- **`check_render.py` has grown rules 9a/9d** (a table needs a prose claim above it; a budget of 2
  "X rather than Y" constructions). Both deliverables now pass all of them.

**Assumptions logged (loop mode — could not ask).**

- Treated `human_feedback_3.txt` as an operator feedback file despite the `.txt` extension (same
  precedent as rounds 1-2); renamed to `human_feedback_3.addressed.md` with contents untouched.
- Chose 1,000 pairs = 200 per selection quintile with a 20-use cap. The operator said "approximately
  1,000" and "prevent a few endpoints from dominating"; the cap of 20 is the smallest round number
  that still admits 1,000 pairs from the 1,763 frequency-matched candidates (2,000 endpoint slots /
  123 endpoints = 16.3 average).
- Ran the optional step-0 checkpoint on the large bank (the operator called extra checkpoints
  optional) because the "requires training" claim is the one the large bank could most cheaply
  strengthen. Did not run 410M or the formation checkpoints on it — no new question they would answer.
- Kept the primary 60-pair bank as the confirmatory headline everywhere, per the operator's
  instruction, with the 1,000-pair result presented as an endpoint-dependent robustness analysis.

**Next step.** Nothing outstanding: the plan's definition of done holds, all three feedback rounds are
addressed, and zero unaddressed feedback files remain, so `STOP` is written. If new feedback appears
next to it, delete `STOP`, address it, and re-write `STOP` only when clean. The out-of-scope follow-up
is unchanged and now better motivated: a **context-conditioned** divergence estimate, which is the
natural way to attack the mediation null (a predictor that is not merely a proxy for the model's own
output separation) and to test the late checkpoints where the global statistic stops improving.

On track? yes — S1-S6 complete (100%) on the prespecified bank plus a 1,000-pair generality test;
operator feedback #3 fully addressed, no blocker.

---

## 2026-08-05 — iteration 4: operator feedback #4, the `big`/`large` vs `big`/`in` reference pairs

**Feedback check.** Listed the direction root: `human_feedback_4.txt` was present without the
`.addressed.md` suffix (the three earlier rounds are already `.addressed.md`). It asked for "a plot
that validates if the pythia models show plateau on My house is big/large and does not show plateau on
My house is big->in". That was this iteration's work; renamed to `human_feedback_4.addressed.md` when
done. Note there was no `STOP` file on re-entry (the wrapper's checkout does not carry it), so nothing
had to be deleted.

**What I did.**

1. Re-downloaded both 500,000-row corpus splits (the `/tmp` cache is gone every session) and wrote
   `experiments/reference_jsd.py`, which tracks only the three reference endpoints, so the count table
   is 3 x 50,304 instead of 1,024 x 50,304. It reproduces the pipeline exactly: 50,060 valid target
   IDs, the same number as every earlier run.
2. Wrote `experiments/reference_house.py`: the unchanged post-block-0 SLERP assay on the two pairs in
   four carriers (`My house is` + the three project carriers) at 1.4B trained / 1.4B step 0 / 410M
   trained, 24 curves. It also records **absolute output movement** `M(t) = JSD(p(t), p(0))` in bits.
3. Wrote `experiments/plot_reference_house.py` -> `plots/house_reference.png`, embedded as Figure 13
   in both deliverables with a visible caption, plus a table and prose in each.

**What I learned.**

- **The answer is the reverse of the question as phrased, and that is the interesting part.** Pythia
  plateaus on ` big`/` in` (w = 0.357, edge drift 0.043 — sharper than all 60 bank pairs) and shows no
  plateau at all on ` big`/` large` (w = 0.773 vs the linear-response 0.8; E = 0.162 vs 0.184).
- **Adding `M(t)` reconciles the two.** The trained model separates *"My house is big"* from *"My house
  is large"* by only 0.035 bits (0.008 by mid-path), so the whole path is inside ONE plateau and there
  is no boundary to cross; `d(t)` is normalised, so it divides that near-zero movement by itself and
  reports the leftover as a straight line. ` big`/` in` moves 0.935 bits, essentially all between
  t = 0.4 and 0.6. Both readings of "plateau" therefore hold at once.
- **This is a real limitation of `w` that the bank analysis hides.** A pair whose endpoints the model
  barely distinguishes has no transition to measure, and any width computed for it is describing
  noise. The bank never hits this case (its pairs are top-256 model-plausible and spread over
  0.14-0.94 bits of corpus divergence), but a reader applying the assay to their own sentences will.
  That is now stated in Methods, in the new Results subsection and in the Conclusion.
- **Consistent with, but not evidence for, the main result.** Corpus divergence orders the two pairs
  the right way (0.412 vs 0.701 bits, higher = sharper), but the observed gap is wider than the bank
  trend at those divergences (0.639 and 0.502 for neighbouring bank pairs), and ` in` is ~80x more
  frequent than ` big`, so it would fail the bank's 2x frequency-matching rule. Both caveats are in
  the deliverables; the pair is presented as an illustration, not as a data point.

**Assumptions logged (loop mode — could not ask).**

- Treated `human_feedback_4.txt` as an operator feedback file despite the `.txt` extension, as in
  rounds 1-3, and renamed it to `.addressed.md` with contents untouched.
- Ran the operator's carrier `My house is` as the headline and added the three project carriers as a
  robustness check, rather than only `My house is` (rejected: a single carrier could not show whether
  the effect is carrier-specific) and rather than replacing the project carriers (rejected: the
  operator named this sentence).
- Added `M(t)` rather than only reporting `w` and edge drift (rejected: with `M(1) = 0.035` bits for
  ` big`/` large`, `w` alone would have made the pair look like a failed plateau instead of a pair
  with nothing to transition between).
- Placed the new figure as **Figure 13** at the end of Results instead of renumbering all 12 existing
  figures to slot it next to Figure 3; the reading order stays sequential either way.
- Kept the corpus estimate defined exactly as before (full 500,000-row splits, unsmoothed JSD) rather
  than shortcutting with a partial download; the whole point is comparability with the frozen bank.

**Next step.** None outstanding: the plan's definition of done still holds, all four feedback rounds
are addressed, and zero unaddressed feedback files remain, so `STOP` is written again. If new feedback
appears next to it, delete `STOP`, address it, and re-write `STOP` only when clean. The out-of-scope
follow-up is unchanged: a **context-conditioned** divergence estimate — now with one more motivation,
since the ` big`/` in` example shows the global statistic ordering a pair correctly while badly
under-predicting how large the width gap is.

On track? yes — S1-S6 complete (100%) plus the 1,000-pair generality test and the named reference-pair
check; operator feedback #4 fully addressed, no blocker.

---

## 2026-08-05 — iteration 4b: operator feedback #5, appendix documenting the 60-pair bank

**Feedback check.** `human_feedback_5.txt` appeared in the direction root while I was finishing
feedback #4 (it was dropped at 21:58; I found it when re-listing the root before writing `STOP`).
Under CLAUDE.md rule 11 that blocks `STOP`, so I addressed it in the same session.

**What it asked.** "How did you sample the 60-pair bank? what are they? write those in the appendix of
the report."

**What I did.** Wrote `experiments/appendix_bank.py`, which regenerates the bank listing from
`results/pair_manifest_top256.json` plus the two 1.4B assay runs (so the table cannot drift from the
analysed data), and added **Appendix A** to REPORT.md: A.1 the seven-step sampling procedure with the
surviving count after each step and the balance cost written as a rendered equation, A.2 the full
60-row table (index, quintile, both endpoints, both corpus counts, `J_sel`, `J_hold`, trained and
step-0 `w`, calibration pairs asterisked). Added cross-references from Methods and from RESULTS.md.

**What I learned / noticed while writing it out.**

- The selection ceiling is worth stating explicitly and now is: 123 eligible endpoints -> at most 61
  endpoint-disjoint pairs, so n = 60 is a hard consequence of the design, not a sampling choice.
- The strata are legible from the table, which is the best defence of the predictor: Q1 is
  ` nice`/` beautiful`, ` simple`/` easy`, ` of`/` in`; Q5 is ` out`/` your`, ` un`/` better`,
  ` extremely`/` happening`. A reader can now sanity-check the divergence scale without running code.
- Both endpoints of this session's reference pair appear in the bank in *other* pairs (` in` in pair 1,
  ` big` in pair 18), which is consistent with endpoint-disjointness *within* the bank and with the
  reference pair being outside it.

**Assumptions logged (loop mode — could not ask).**

- Put the listing in REPORT.md only (the operator said "the appendix of the report") and left RESULTS.md
  with a one-line pointer, rather than duplicating a 60-row table into both deliverables.
- Reported counts summed over both corpus splits, because that is the quantity the factor-of-two
  frequency-matching rule actually used; per-split counts remain in the manifest.
- Included trained and step-0 `w` columns even though the feedback only asked "what are they", since
  the pair-level outcomes are what make the table useful for checking any figure.

**Next step.** None outstanding: plan complete, feedback rounds 1-5 all addressed, zero unaddressed
feedback files, so `STOP` is written. The out-of-scope follow-up remains a context-conditioned
divergence estimate.

On track? yes — S1-S6 complete (100%) plus the 1,000-pair test, the named reference pairs and the bank
appendix; operator feedback #5 fully addressed, no blocker.

---

## 2026-08-05 — iteration 5: operator feedback #6, correspondence-only report

**Feedback check first (CLAUDE.md Part C).** Listed the direction root: `human_feedback_6.txt` was
present without the `.addressed.md` suffix, and no `STOP` file existed. That file was this iteration's
work. `PLAN.md` had also been rewritten by the operator and now declares itself read-only.

**What it asked.** Revise `REPORT.md` and `RESULTS.md` only, so the direction answers exactly one
question — do token pairs with more different immediate-next-token distributions tend to have narrower
transitions in the trained model's output-distance score `d(t)`? Lead with the 1,000-pair result,
follow with the controlled 60-pair result, keep four checks plus the adjustment result plus the two
named examples, move construction and statistics to Methods/Appendix, delete everything about
formation, and de-jargon the main text against a supplied replacement table.

**What I did.** No experiment re-run; no number changed. Rewrote both deliverables from scratch in the
requested order (question and conclusion → 1,000 pairs → controlled 60 pairs → checks → what the score
misses → limitations → Methods/Appendix), applied every jargon replacement, cut the formation results,
the learned-`Δw` analysis, the block scan (demoted to Appendix B), the `width_by_jsd_bin` figure, and
every claim about plateaus corresponding to continuation distributions. Wrote
`experiments/plot_adjustment.py` to produce a `w`-only adjustment forest plot, since the existing
`mediation.png` had `Δw` in both of its panels. Edited only the label strings in `analyze.py`,
`large_analysis.py` and `plot_reference_house.py` and regenerated their figures; all statistics printed
identically, which is the check that I touched presentation and not analysis.

**Assumptions logged (loop mode — could not ask).**

- **Section order.** The feedback puts "Methods/Appendix" last; CLAUDE.md rule 8 mandates
  `Summary → Methods → Results → Conclusion`, and rule 12 requires every axis variable to be defined in
  Methods *before* the figure appears. I kept the mandated heading order and satisfied the feedback by
  making Methods lean and pushing pair construction, the corpus-sample check, alternative pair sets,
  self-tests and the 60-pair table into Appendices A and B. Rejected alternative: Methods physically
  last, which would have broken rule 12's define-before-you-show requirement.
- **Notation.** Replaced `Ĵ_hold`/`Ĵ_sel` with `J` and `J_sel`, because the feedback's replacement for
  "held-out JSD" is a phrase, not a symbol, and carrying a hatted two-subscript symbol through the text
  would have reintroduced the jargon it asked to remove.
- **`plots/mediation.png` and `plots/formation.png` are left on disk** but no longer appear in either
  deliverable, per the feedback's "preserve all existing plots even when they are removed".
- **`PLAN.md` not edited**, per its own read-only declaration, even though the standing loop prompt
  asks for a status update. This entry is the status record.

**What I learned / noticed while rewriting.**

- Leading with the 1,000-pair result is genuinely the better story. Its ten bin medians (0.649 → 0.499,
  one 0.002 wobble) show the *shape* of the relationship, which the 60-pair scatter cannot at ~12 pairs
  per group; the 60-pair analysis then earns its place as the controlled version rather than as the
  headline. The feedback was right that this ordering shows the relationship most clearly.
- Removing formation removed the only part of the report that had contradicted its own plan, and
  nothing in the correspondence story depended on it. The report got shorter and the claim got no
  weaker.
- The `big`/`large` example reads better as a caution about the metric than as evidence about
  plateaus: `M(1) = 0.035` bits means the two outputs never separate, so `d(t)` — which normalises by
  the total separation — has nothing to report. That framing needs no claim about what lies along the
  path, which is exactly what the feedback forbade.
- `jsd_reliability.png` cannot be regenerated: its 10,000-pair source array lived in the /tmp corpus
  cache, which does not survive across sessions. Its axis labels still say "selection-split" and
  "held-out"; the caption now maps both to the new vocabulary. Re-deriving the array would mean
  re-downloading 4.1 GB of corpus, which is not worth it for two axis labels.

**Next step.** None outstanding for this direction: `PLAN.md` stage S4 (correspondence-only report) is
what this iteration delivered, and zero unaddressed feedback files remain, so `STOP` is written. If
more feedback arrives, the wrapper's next entry should delete `STOP` first (rule 11).

On track? yes — S1–S3 complete, S4 (correspondence-only report) delivered this iteration; operator
feedback #6 fully addressed, no blocker.

---

## 2026-08-10 — feedback #7: fixing a wrong justification for the flatness caveat

**What I did.** `human_feedback_7.txt` was sitting unaddressed (a `.txt`, not `.md`, but plainly an
operator feedback file, so I treated it under Part C). It objected to one sentence in the Methods
paragraph that introduces edge drift `E`: "A narrow transition could mean flat ends with a quick move
in the middle, or just a steeper straight line" — asking how a narrow transition could be a steeper
straight line with no quick move in the middle.

The operator is right and the sentence was simply wrong. `d(t)` is a *relative* distance, so `d(0) = 0`
and `d(1) = 1` hold exactly for every curve we run. A straight line through both endpoints therefore
has slope exactly 1 and width exactly 0.8 — there is no steeper straight line to have. I had written
the standard "sharp step vs. shallow ramp" dichotomy without noticing that the endpoint pinning already
rules out one half of it.

Rather than delete the sentence, I worked out what `E` genuinely adds over `w` and measured it
(`experiments/edge_geometry.py`, CPU only, reads the committed curves): `w` is a width and says nothing
about **where** the move sits, and where it sits is exactly what decides endpoint flatness. Sweeping
the same width-`w` transition across 201 starting positions gives a *placement range* of `E`; at the
median trained width the range is 0.080 (centred) to 0.220 (parked late), a factor of 2.7 and wider
than the entire trained-to-untrained gap in `E`. So the near-perfect `w`–`E` agreement is not an
algebraic identity — it is an empirical fact about these curves: the transition midpoint is
`m = 0.505` (IQR 0.047), 96.7% of controlled pairs and 97.6% of the 1,000 within 0.1 of the middle, so
every curve sits at the flattest placement available to it.

Figure 7 went from 2 to 4 panels to show this: equal-width curves with different `E`, per-pair
placement ranges, and midpoint histograms. Added the partial correlation that makes the redundancy
concrete: `rho(J, E) = -0.520` alone, `rho(J, E | w) = -0.008` [−0.332, +0.328].

**Assumptions logged** (loop mode, no one to ask). (i) The placement family is the piecewise-linear
three-segment curve through `(0,0), (A,0.1), (A+w,0.9), (1,1)`. I first tried sliding the *measured*
curve itself, but that re-anchors `E` to the shifted endpoint values (the shifted curve no longer ends
at 1) and understated the range; the constructed family keeps `d(0) = 0`, `d(1) = 1` and is a curve the
experiment could genuinely have produced. Rejected alternative: a full min/max optimisation over all
monotone curves of a given width — more general but its extremes are degenerate limits, and the
three-segment family already makes the point with an honest, reproducible construction. (ii) I did not
rerun any GPU experiment; nothing about the main association changed.

**What I learned.** A metric caveat can be *true* for the wrong reason, and the wrong reason is worse
than no reason: the caveat "flatness and width are the same measurement" survived, but its stated
justification was geometrically impossible. The corrected version is also more informative — it names
the property that would have to change (off-centre transitions) for the two to separate.

**Next step.** None outstanding: PLAN.md S4 is delivered and feedback #7 is addressed and renamed
`human_feedback_7.addressed.md`, so `STOP` is written again. If more feedback arrives, delete `STOP`
first (rule 11).

On track? yes — S1–S3 complete, S4 delivered, feedback #7 fully addressed, no blocker.
