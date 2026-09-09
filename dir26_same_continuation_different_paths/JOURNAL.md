# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-09-09 — iteration 1 (S1 + S2 + S3, direction complete)

**Did.** Wrote `experiments/common.py`, `s1_completions.py`, `s1b_repair.py`, `s2_interp.py`,
`s2_plot.py`. Ran the 8 exact prompts on GPT-2 Large with greedy decoding (6 new tokens), then the
50-step full-sequence block-0 interpolation for the one pair that passed.

**Learned.**
- Only pair 1 (copy-available vs factual-recall) passes the completion screen: both sides continue
  with ` Paris`, both prompts are 12 tokens.
- Pair 2 fails hard — both `The number after six is` and `The number before eight is` continue with
  the identical string ` the number of times the player`. GPT-2 Large does not do this arithmetic.
- Pair 3 fails on the A side and is length-mismatched (5 vs 6). Applied the ONE wording repair
  PLAN.md names (`The capital city of France is`): lengths match at 6, but A then continues
  ` home to the world's largest`, so the pair is still ineligible. Stopped there — no further
  variants, per the "no large prompt search" rule.
- Pair 4 fails on both sides (` a man of the world,`) and is length-mismatched (6 vs 7). Did NOT
  spend the repair on it: the repair budget is for length mismatches on otherwise-passing pairs, and
  pair 4's failure is the answer itself.
- Pair 1's `d(t)` is a textbook plateau: 17% of the endpoint gap over the first 57% of the path,
  then 71% over the next 10%. Top-1 token is ` Paris` at every one of the 50 positions, so the
  movement is in the whole logit vector, not the prediction.

**Assumptions logged (no human to ask).**
- "50 interpolation steps" implemented as `numpy.linspace(0, 1, 50)` (50 positions inclusive of both
  endpoints). Rejected alternative: 51 points (50 intervals) — makes no difference to the visual
  verdict.
- Completion check implemented as "intended answer appears in the greedy 6-token continuation"
  rather than "is the immediate next token". This is the looser of the two readings, so it can only
  admit more pairs; under the strict reading pair 1 still passes (both sides emit ` Paris` first) and
  the three failures are unchanged.
- Endpoints `A` and `B` for `d(t)` taken from the patched run at `t = 0` and `t = 1`; verified equal
  to the clean-run logits to L2 ~1e-3 against an endpoint separation of 445.2.

**Next step.** None — the success criterion is met (all four pairs screened, one `d(t)` plot for the
one passing pair, REPORT.md carries the exact prompts, literal completions, the plot and the
interpretation). Wrote `STOP`.

On track? yes — S1/S2/S3 all complete, 100% done, no blocker.
