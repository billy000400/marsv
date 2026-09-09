# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT*.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and every REPORT*.md stay current-best with no history.

---

## 2026-09-09 — first complete results; REPORT.md and RESULTS.md written from stubs

Both files were TODO stubs before this entry; nothing was superseded.

- **RESULTS.md** — new. Model/data table (retrained dir13 recipe, 30,000 steps, val acc 0.550), S0
  matched-pair counts, S1 steering table, S2 mean-JSD and crossing tables, S3 plane comparison, all
  three figures embedded with captions.
- **REPORT.md** — new. Answers the plan's two questions in 900 words and 3 figures.
  Verdict: **no visible support** for the first-pass prediction.
  - Q1 both directions steer as intended: speaker label P(uppercase or ':') 0.0002 → 0.31 at alpha 1
    (random 0.004); word continuation P(letter) 0.25 → 0.78 (random 0.26).
  - Q2 not supported: mean JSD crosses 0.05 bits at alpha 0.39 (mixture) versus 0.38 / 0.79 (singles);
    per anchor 7/16 and 12/19. Random directions cross earliest (median alpha 0.82 vs 1.00).
  - S3 planes alike: 34% vs 32% of grid below 0.05 bits; 45-degree point 0.12 vs 0.17 bits, but the
    feature plane is lower in only 5 of 8 anchors.
- **Within this iteration**, the S3 numbers were recomputed after `s3_geometry.py` was patched to save
  per-anchor maps. The mean maps were unchanged; the addition was the per-anchor diagonal series
  (feature 0.008-0.47, random 0.007-0.76), which is what downgraded the 0.12-vs-0.17 mean gap from a
  weak positive to "the anchors do not support the mean". Only the later, fuller version is reported.
