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

## 2026-08-12 — PLAN replaced with the capital-city example; verdict flipped to "delayed plateau"

**Why.** PLAN.md was rewritten (by the operator) around a new fixed example — prefix
`The capital of France is Paris. The capital of`, endpoints ` Japan`/` Germany`, shared successor
` is`, expected delayed tokens ` Tokyo`/` Berlin` — replacing the codebook (`A = cat`, `B = dog`)
example whose endpoints GPT-2 Large failed to reproduce. Status was reset to "fresh start", so this
iteration re-ran S1–S4 on the new example and re-curated both deliverables.

**RESULTS.md and REPORT.md — superseded numbers (old codebook example → new capital example).**
- Verdict: conclusion 3, *invalid example* → **conclusion 1, delayed plateau**.
- Endpoint validity: all four checks failed → all four pass (immediate top-1 ` is` 0.944 / 0.940;
  delayed top-1 ` Tokyo` 0.928 / ` Berlin` 0.848).
- Endpoint JSD: immediate 0.0861 nats, delayed 0.0115 nats → immediate **0.0014 bits**, delayed
  **0.9945 bits** (unit changed to bits, as the new PLAN specifies).
- Delayed transition width: `w = 0.38` → **`w = 0.28`**; midpoint 0.42 → 0.48; endpoint separation
  75.4 → 462.5.
- Behaviour: delayed top-1 never changed → delayed top-1 flips ` Tokyo`→` Berlin` at `t = 0.49`.
- Immediate readout: top-1 was ` =` throughout (planned successor ` means` at 6.7e-4) → top-1 is
  ` is` at all 101 positions, p in 0.931–0.944.

**Figures.** `plots/immediate_readout.png` removed (it belonged to the retired example) and replaced
by `plots/immediate_prediction.png`, the filename the new PLAN names. `plots/delayed_distance.png`
and `plots/delayed_tokens.png` regenerated for the new example; the distance figure now shows only
the delayed curve plus the linear reference, because the new PLAN forbids interpreting the immediate
normalized-distance curve.

**Metrics cut.** The delayed top-2 logit margin and the immediate `d(t)` curve are no longer reported
— the first is superseded by the actual top-1 flip, the second is ruled out by PLAN's "out of scope".

**Checks.** `python3 experiments/check_render.py REPORT.md RESULTS.md` passes (5 display equations,
18 inline, 3 embedded figures with visible captions in REPORT.md; 3 embedded figures in RESULTS.md).
