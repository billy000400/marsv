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
