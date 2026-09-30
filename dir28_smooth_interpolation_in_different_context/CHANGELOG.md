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

## 2026-09-30 — human_feedback_0.txt: add Direction 27 control histogram

- New `plots/section0_control_width_histogram.png`: "The house was big" `W_final` from dir27 `results/sweep.csv` (no rerun), same 100 bins (0–0.7), log y-axis, same x/y limits as Figures 1–4. The shared bins and y-limit are now computed over all five contexts; the four existing PNGs came out byte-identical, because the control's max W (0.650) and max bin count (3,653) sit inside the existing range.
- REPORT.md: Figure 5 plus a short paragraph in "Cross-context comparison"; one sentence each in Summary and Conclusion; captions of Figures 1–4 now say "Figures 1–5 use the same bins and axes". Existing numbers are unchanged.
- RESULTS.md: control row in the per-context table (816 above 0.3, median 0.112) and Figure 5.
- experiments/analyze.py: loads the dir27 control for the histograms only; token CSVs and cross-context outputs are unchanged (hashes verified).
