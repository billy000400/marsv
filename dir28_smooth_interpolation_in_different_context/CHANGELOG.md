# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT*.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and every REPORT*.md stay current-best with no history.

---

## 2026-09-29 — REPORT.md draft
- Replaced template REPORT.md with a draft: research question, Methods, Section 1 qualitative description.
  Sections 2–4, histograms, cross-context comparison, Summary and Conclusion are marked PENDING until the sweeps finish.

## 2026-09-29 — iteration 10

- REPORT.md: filled Sections 2–4 (counts, qualitative notes), embedded Figures 1–4 (shared-bin width histograms),
  wrote Cross-context comparison (Table 1: 229 / 262 / 318 tokens in 4 / 3 / 2 contexts; unique 120 / 405 / 690 / 143),
  Summary, Conclusion, Appendix A (complete sorted W > 0.3 lists) and Appendix B (cross-context token sets).
  Section 1 glitch-block count corrected 64 → 56 (64 was D > 4; the D ≈ 5.16 block itself is 56 tokens).
- analyze.py / report_lists.py: context-unique assignment now uses max_W instead of the 4-dp-rounded W > 0.3 test
  (two tokens with W just above 0.3 were dropped: s3 unique 689 → 690, s4 142 → 143).
- RESULTS.md: first full version (per-context summary, cross-context counts, near-threshold check, glitch block, files).
