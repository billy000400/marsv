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

## 2026-09-09 — iteration 2 (operator feedback: bigger model + prediction column)

**Feedback.** `human_feedback.txt`: "It looks like a lot of examples failed. Can you use a bigger
model and try again? (Qwen?) Also for the failed ones, can you also add a column about what the model
actuallly predicting?"

**Did.** Parameterised `experiments/common.py` with `DIR26_MODEL=gpt2|qwen`; added a top-1/top-5
next-token record to `s1_completions.py` (reused by `s1b_repair.py`); made `s2_interp.py` run a
per-model list of screened pairs and write model-suffixed outputs; replaced `s2_plot.py` with two
figures. Reran the full pipeline on both models.

**Learned.**
- Qwen2.5-3B rescues two of the three pairs GPT-2 Large failed: pair 3 (after the one authorized
  wording repair, which now works) and pair 4 both give the intended answer, so 3 of 4 pairs reach
  interpolation instead of 1.
- Pair 2 still fails, but for a *different reason*, which is exactly what the requested prediction
  column exposes: Qwen2.5-3B reads `The number after six is` as a fill-in-the-blank quiz item and
  puts 0.19 on ` __`, with ` seven` fourth at 0.07. GPT-2 Large's failures are ignorance (top-1
  ` the` / ` a` at ~0.1). Same verdict, different cause.
- Pair 4 passes the documented screen (intended answer inside the greedy continuation) but its
  immediate next token is ` ______` on both sides, so its interpolation endpoints are a
  fill-in-the-blank state. Reported as tested with that caveat rather than silently dropped or
  silently accepted.
- **The headline changed.** More testable pairs did NOT give more plateaus: pair 1 plateaus on both
  models (steepest fifth of the path covers 65% of the endpoint gap on Qwen2.5-3B, 77% on GPT-2
  Large), while pairs 3 and 4 are gentle S-curves (44%, 37%) → **no clear plateau**. Old story:
  "the screen is the binding constraint, and the one survivor plateaus." New story: "scale fixes the
  screen, and the plateau still shows up only in the copy-vs-recall pair." Rewrote REPORT.md around
  that (rule 9b).

**Assumptions logged (no human to ask).**
- "Qwen?" has no size. Chose `Qwen/Qwen2.5-3B` — largest Qwen2.5 base model fitting the 0.225 GPU
  fraction (~7.2 GB) in bfloat16, 4x GPT-2 Large. Rejected: Qwen2.5-7B (15 GB in fp16, over budget);
  Qwen2.5-1.5B (fits comfortably but is a smaller step up). Instruct variants rejected: the prompts
  are raw completions, so a base model is the like-for-like comparison.
- bfloat16 is forced by the memory budget. Checked the cost: patched-vs-clean endpoint logits differ
  by ~3% of the endpoint separation on Qwen2.5-3B vs ~2e-6 on float32 GPT-2 Large. `d(t)` uses the
  patched endpoints, so the metric is internally consistent; the check is reported in RESULTS.md.
- "add a column about what the model actually predicting" — the tables already had the greedy
  6-token continuation, so read the ask as an explicit *prediction* column: top-1 next token with its
  probability, added to BOTH screening tables (top-5 kept in RESULTS.md). Applied to every row, not
  only failed ones, so the tables stay comparable.
- PLAN.md lists "no model sweep" as out of scope. Operator feedback overrides; added exactly the one
  requested model, no sweep.

**Next step.** None. Feedback addressed, manifest set to `review_pending`. Did not write `STOP` —
`human_feedback.txt` is still unaddressed until the wrapper's reviewer renames it (rule 11).

On track? yes — feedback iteration complete, 100% done, no blocker.
