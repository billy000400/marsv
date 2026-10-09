# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT*.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and every REPORT*.md stay current-best with no history.

---

## 2026-10-09 — first results

- `REPORT.md` and `RESULTS.md` written from the template: sample counts, feature sets, overlap, early-position check, suppression check.
- Superseded within this iteration (bug fix, never published): features with true score 0 were selected through float32 rounding. Tokyo set 280 -> 276; Japan set 470 -> 403; Tokyo–Japan shared 62 -> 58, Jaccard 0.090 -> 0.093; Japan–Germany Jaccard 0.126 -> 0.140; Japan–Berlin shared 14 -> 11; Japan-set features active on the Japan token 13 -> 5. Berlin and Germany sets, all strong-feature counts and the suppression results are unchanged.
- Framing: the plan's headline is the pooled overlap; the report keeps it as Result 2 and adds the score split as Result 3 because the pooled sets are dominated by weakly selective features.
