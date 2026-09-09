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
