# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-08-10 — first and final results: invalid example (PLAN conclusion 3) + propagation measurement

**New (no prior results to supersede).** Ran S1–S4 in one iteration with GPT-2 Large (774M), single
designed codebook prompt, 101 norm-corrected SLERP interpolation points on the ` A`→` B` input
embedding at position 15.

- **RESULTS.md** written from the template: endpoint-validation table, interpolation-sweep table,
  three embedded figures.
- **REPORT.md** written from the template: Summary → Methods (Data & Model, five metrics with
  rendered `math` fences, two baselines) → Results (three subsections, Figures 1–3 embedded with
  visible captions) → Conclusion.
- **Verdict:** PLAN conclusion 3, **invalid example**. All four endpoint checks fail — top-1 after
  ` A` / ` B` is ` =` (p = 0.340 / 0.525), not ` means` (p = 6.68e-4 / 4.50e-4); top-1 after
  ` A means` / ` B means` is a quote mark, not ` cat` / ` dog`. Endpoint JSD 0.0861 nats immediate,
  0.0115 nats delayed.
- **Secondary result kept (clearly scoped):** plateau shape survives one token of propagation.
  Transition width `w` = 0.27 immediate vs 0.38 delayed vs 0.80 linear null; midpoints `t₅₀` = 0.45
  vs 0.42; endpoint separation ‖z_A−z_B‖₂ = 300.2 vs 75.4 (4.0× attenuation); both curves monotone.
  Delayed top-1 constant and top-2 margin never below 0.43, so no behavioural change.
- **Figures added:** `plots/immediate_readout.png`, `plots/delayed_distance.png`,
  `plots/delayed_tokens.png` (the three named in PLAN), CVD-safe palette, non-colour channels for
  every series.
- **Corrected before publication (never shown in a deliverable):** the first sweep reported
  `flip_t_delayed = 0.0`, an artifact of `argmax` on an all-false mask; p(` dog`) never exceeds
  p(` cat`), so the statistic is now `None`. Old → new: `0.0` → `None` (no flip).
- Not reported: JSD-from-A-endpoint across `t` (recorded in `results/delayed.json`) — `d(t)` answers
  the same question, so per rule 9 the unused metric is cut from the deliverables.
- `check_render.py` passes on both files (0 problems).
