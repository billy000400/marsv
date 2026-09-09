# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT*.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and every REPORT*.md stay current-best with no history.

---

## 2026-09-09 — first and final content for RESULTS.md and REPORT.md
- **RESULTS.md**: replaced the template with the full evidence record — S1 completion screen for all
  8 exact prompts (literal greedy 6-token continuations + tokenized lengths), the single authorized
  pair-3 wording repair, and the S2 block-0 full-sequence interpolation for pair 1 (endpoint
  reconstruction L2 0.0009 / 0.0012 against endpoint separation 445.2; per-position cosines;
  top-1 token ` Paris` at all 50 interpolation positions). Figure: `plots/p1_copy_vs_recall_dt.png`.
- **REPORT.md**: replaced the template with the concise answer to the direction's question. One
  figure (Figure 1, the pair-1 `d(t)` curve). Per-pair verdicts: pair 1 **plateau visible**
  (`d(t)` 0 -> 0.170 over t = 0..0.571, then 0.170 -> 0.878 over t = 0.571..0.673); pairs 2, 3, 4
  **not testable** (GPT-2 Large does not give the intended completion for both prompts).
- No superseded numbers: this is the first set of results for this direction.
