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

## 2026-09-09 — operator feedback: bigger model (Qwen2.5-3B) + actual-prediction column

Source: `human_feedback.txt`.

**REPORT.md** — rewritten around the new evidence.
- Model scope: GPT-2 Large only → GPT-2 Large **and Qwen2.5-3B** (3.09B params, bfloat16).
- Both screening tables gained the requested column: the model's actual top-1 next token with its
  probability (top-5 moved to RESULTS.md).
- Testable pairs: 1 of 4 → **3 of 4** on Qwen2.5-3B (pairs 3 and 4 now answer as intended; pair 3
  still needs the one authorized wording repair, which succeeds on Qwen2.5-3B and failed on GPT-2
  Large).
- Pair 2 verdict unchanged (not testable) but the cause is now documented: Qwen2.5-3B top-1 ` __`
  at 0.19 with ` seven` fourth at 0.07 (formatting), vs GPT-2 Large top-1 ` the` at 0.28 (ignorance).
- **Superseded headline.** Old: "the completion screen is the binding constraint; the one surviving
  pair plateaus." New: "a bigger model fixes most of the screen failures, and the plateau still
  appears only in pair 1." Pair 3 (repaired) → **no clear plateau**; pair 4 → **no clear plateau**.
- Pair 1 numbers restated on the new model, old → new: GPT-2 Large "17% of the gap over the first
  57%, then 71% over the next 10%" → reported per model as the share of the endpoint gap inside the
  steepest fifth of the path, GPT-2 Large 77% and Qwen2.5-3B 65%; verdict **plateau visible** on both.
- New caveat: pair 4's endpoints sit in a fill-in-the-blank state (top-1 ` ______` on both sides at
  all 50 interpolation positions).
- Figures: 1 → 2. `plots/p1_copy_vs_recall_dt.png` (GPT-2-only pair 1) replaced by
  `plots/qwen_dt_three_pairs.png` (Figure 1, three Qwen2.5-3B curves) and
  `plots/p1_gpt2_vs_qwen_dt.png` (Figure 2, pair 1 on both models).

**RESULTS.md** — rewritten to cover both models: two full screening tables with top-1 and top-5
predictions, endpoint sanity checks for both models, the `d(t)` reading table with the steepest-window
column, per-position cosines for both models, and per-pair verdicts.

**Code** — `common.py` takes `DIR26_MODEL=gpt2|qwen`; `s1_completions.py` records top-1/top-5;
`s1b_repair.py` reuses it; `s2_interp.py` runs the per-model screened pair list and writes
model-suffixed outputs; `s2_plot.py` emits the two figures. Stale unsuffixed files in `results/` and
the superseded `plots/p1_copy_vs_recall_dt.png` were removed.
