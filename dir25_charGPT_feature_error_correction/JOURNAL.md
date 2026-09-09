# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-09-09 — iteration 1 (S0 + setup)

**Blocker found and resolved.** PLAN says to reuse dir13's trained checkpoint, but dir13's
`results/checkpoints_grok_char` is a symlink into `/tmp`, and `/tmp` has been wiped: no checkpoint
and no corpus survive. Documented assumption (no human to ask): **retrain the same model with
dir13's exact recipe** rather than substituting a different model or a different research question.
Re-downloaded tinyshakespeare (SHA-256 matches dir13's recorded training corpus) and wrote
`experiments/train_char_gpt.py`, a copy of dir13's `train_grok.py --tok char` path (12L/12H, d=240,
ctx 128, dropout 0.2, Adam lr 1e-3 cosine->1e-4, wd 0, bs 48x128, 30k steps, seeds 1337/42) that
writes into this direction. The reproduction tracks dir13's logged run step for step
(step 2500: train 1.299 / val 1.504 / acc 0.552 in both logs), so the checkpoint is equivalent.
Rejected alternatives: (a) using dir17's checkpoints — different model and task; (b) training a
smaller/shorter model — changes the object PLAN specifies.

**S0 done** (`experiments/collect_examples.py`, `results/feature_counts.json`,
`results/feature_examples.txt`). Matched examples available, matching on the final character:
speaker-label 23,010 pairs (train) / 2,742 (held-out); word-continuation 152,467 / 16,683. Both
features clear the 60-pair requirement by orders of magnitude. Split convention: directions are
built from the first 90% of the corpus (dir13's training split), every behavioural test uses the
last 10% (held-out), so no example builds and tests the same direction.

**Known confound, to state in the report:** speaker-label positives sit mid-word inside an all-caps
label, while the matched negatives are usually a capital at a line/sentence start. Tightening the
match to require an uppercase *preceding* character leaves only 2 usable held-out pairs, so the
plan's final-character matching is kept and the confound is reported rather than removed.

**Written and smoke-tested** (against the step-2500 rolling checkpoint): `feature_assay.py`,
`s1_validate.py`, `s2_compare.py`, `s3_geometry.py`. Smoke run already shows both directions moving
their target probability the expected way and above random.

Next step: run S1 on the finished checkpoint; if both behavioural checks pass, run S2 and S3.

On track? yes — S0 done, S1-S3 coded, waiting on the ~23-minute retrain.

## 2026-09-09 — iteration 2 (S1-S3 complete, deliverables written)

Retrain finished at step 30,000 (val loss 2.359, val acc 0.550) — matches dir13's recipe as intended.
Ran S1, S2, S3 on the final checkpoint.

**S1 — both directions kept.** Speaker label lifts P(uppercase or ':') from 0.0002 to 0.31 at alpha 1
(random 0.004); word continuation lifts P(letter) from 0.25 to 0.78 (random 0.26). Both clear the
behavioural check by a wide margin, so the line-ending fallback direction was not needed. Noted for the
record: the effect is one-sided — negative alpha does not suppress the target mass — and at large alpha
random directions also raise letter probability (0.60 at alpha 3), so only the small-alpha region
separates the word-continuation direction from noise. Cosine between the two raw directions is -0.60;
S2/S3 orthogonalize before mixing, as the plan requires.

**S2 — prediction not supported.** Mean JSD curve crosses 0.05 bits at alpha 0.39 (mixture), 0.38
(speaker label), 0.79 (word continuation): the mixture falls *between* the singles rather than after
both. Per anchor it is near a coin flip (7/16 and 12/19). Unpredicted and worth stating: random
directions are the most disruptive condition, crossing earliest (median alpha 0.82 vs 1.00 mixture).

**S3 — planes look alike.** Feature plane 34% of grid below 0.05 bits vs random plane 32%; at the 45
degree point mean JSD 0.12 vs 0.17. That mean difference is in the predicted direction, so before
writing anything I patched `s3_geometry.py` to save per-anchor maps and re-ran (~8 min): the feature
plane is the lower of the two in only 5 of 8 anchors, per-anchor values spanning 0.008-0.47. The
anchor spread is an order of magnitude larger than the difference in means, so the mean is not a
pattern — reported as no visible support, not as a weak positive. This is a direct measurement, not an
added statistical test (plan puts p-values/CIs out of scope).

**Selection effect recorded:** S3 drops 4 of 12 anchors where an axis never reaches 0.05 bits within
the scan, leaving the rescaling undefined. Those are the anchors where feature directions are weakest,
so the retained figure is biased *toward* the prediction. Stated in both deliverables.

**Verdict** in the plan's vocabulary: **no visible support**. The phrase "consistent with
feature-specific error correction" is deliberately not used.

**Deliverables.** `REPORT.md` rewritten to the plan's limits — 900 words exactly (wc -w) and 3 figures,
sections Motivation / Setup / Results / What this does and does not show, plus a Summary for CLAUDE.md
rule 8. Getting from the first draft (1763 words) to 900 took several passes; the binding constraint
was that captions, equations and image markdown consume ~300 of the 900. `RESULTS.md` carries the full
evidence record (S0 counts, S1/S2/S3 tables, per-anchor numbers, the confound and the selection
effect). `python3 ../check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS.

**One minor plot fix:** in fig1's example panel the alpha=+2 bars used sky blue against the alpha=-2
blue; switched to the reddish-purple hue so the three bar groups differ by hue and hatch (rule 13).

No unaddressed feedback files exist, the plan's success criterion is met, so STOP is written.

On track? yes — S1-S3 done, 100%, deliverables final, no blocker.
