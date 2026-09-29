# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-09-29 — iteration 1: reproduce dir27 control, launch four sweeps

- Located dir27 code (`../dir27_interp_dist_vs_plateau_width/experiments/common.py`, `s2_sweep.py`, `s3_plots.py`).
- **Assumption (logged per CLAUDE.md §1):** PLAN says "block-0 resid_post, as in Direction 27", but dir27's actual
  interpolation site is *layer 0 = the input embedding* (wte row of the last token; wpe shared), readout at
  block 18 (mid) and block 35 (final, before ln_f). PLAN's overriding instruction is to reuse dir27's implementation
  unchanged, so I follow dir27's code (embedding layer). Rejected alternative: interpolating the output of block 0,
  which would be a new interpolation method. "Final transition width" = dir27's W_final (block 35).
- Usable tokens = dir27's rule: all 50,256 token IDs except ` big` (none excluded); non-monotonic curves are kept
  but flagged with dir27's flag rule.
- `experiments/sweep.py` imports dir27 `common` and only swaps the module-level PROMPT. Control ("The house was big",
  first 256 tokens) reproduces dir27 shard 0: max |ΔW_final| = 6.9e-7, max |ΔW_mid| = 4.6e-7.
- All four prompts tokenize with ` big` (id 1263) as last token. Launched `experiments/queue.sh` (s1→s4, ~26 min each).
- Expectation to check: dir27 had 64 rare/glitch tokens at D≈5.17 with W_final≈0.33; they may populate every
  context's w>0.3 list. Will report them as a group rather than interpret.

On track? yes — Stage 0 done; sweeps running (~25% done), no blocker.

## 2026-09-29 — iteration 2: s1 complete, waiting on s2–s4

- s1 ("My house is big"): 50,256 tokens, median W_final 0.106, 692 tokens > 0.3 (7 flagged non-monotonic), 64 with D > 4.
  Top of the list: size/magnitude adjectives (HUGE, tremendous, whopping, huge, giant, colossal, vast…), mixed with
  glitch/code-identifier tokens (guiName, externalToEVAOnly, PsyNetMessage…). 64 near-zero-norm glitch tokens sit at W≈0.32.
- **Decision (report length):** PLAN's success criterion requires the *complete* w>0.3 list per prompt in REPORT.md
  (~700 tokens × 4). I read that as the explicit PLAN override WRITING.md rule 11 allows for this content, and put the lists
  in collapsed `<details>` text fences in a declared appendix; the argued body stays well under 5,000 words.
  Rejected: moving the lists to RESULTS.md only (would violate success criterion 3).

## 2026-09-29 — iteration 3: drafting while s2–s4 run

- s2 at 7/25 shards at 21:42; shard time ~193 s because dir29's sweep shares the GPU (93 s alone). Running s3/s4 in
  parallel would not raise total throughput, so the serial queue is kept. Expected finish ~00:15.
- Drafted REPORT.md: research question, Methods (prompt token IDs verified from shards/tokenizer), Section 1
  qualitative note. Section 1 reading: top-40 dominated by size/magnitude words; small-size and place/object
  adjectives fill much of the rest; code identifiers throughout; 64 glitch tokens at W≈0.32, D≈5.16.
- Methods states the actual interpolation site (layer-0 embedding, as dir27 code does) rather than PLAN's
  "block-0 resid_post" wording (see iteration-1 assumption).

## 2026-09-29 — iteration 4: waiting on s2–s4 (21:45)

- s2 at 8/25 shards; queue.sh alive; GPU 100% (shared with dir29). No queue.done yet, so no analysis this iteration.
- Draft REPORT.md passes ../check_render.py (3 display eqs, 0 problems). No other changes.

## 2026-09-29 — iteration 5: waiting on s2–s4 (21:46)

- queue.sh alive; s2 still mid-sweep; no results/queue.done. No analysis or report edits this iteration (next step unchanged).

## 2026-09-29 — iteration 6: waiting on s2–s4 (21:47)

- queue.sh alive; s2 at 8/25 shards (~193 s/shard, GPU shared with dir29). No queue.done. Waiting in-iteration for s2 to finish (~22:45) instead of spawning no-op iterations; no report edits yet.

## 2026-09-29 — iteration 7: waiting on s2–s4 (21:48)

- queue.sh alive; s2 at 9/25 shards (~193 s/shard, GPU shared). analyze.py needs all four contexts, so no report edits possible yet. Waiting in-iteration for s2 to finish.

## 2026-09-29 — iteration 8: waiting on s2–s4 (21:50)

- queue.sh alive; s2 at 9/25 shards. Waiting in-iteration (background until-loop on results/queue.done); will run analyze.py + report_lists.py and fill REPORT.md when it lands (expected ~00:30).

## 2026-09-29 — iteration 9 (21:50)

- s2 at 9/25 shards, ~193 s/shard (GPU shared with dir29). Projection: s2 ~22:45, s3 ~00:05, s4 ~01:25 at shared speed — may exceed the ~183 min budget unless dir29 frees the GPU. No parallel sweep started (GPU is at 100%, would not add throughput). Waiting in-iteration on results/queue.done; if budget runs short, finalize REPORT.md with whichever contexts have completed and mark the rest as not run.

## 2026-09-29 — iteration 10 (22:30)

- dir29 sweep finished ~22:25; our shards now ~93 s. s2 complete at 22:28 (961 tokens W_final > 0.3, median 0.135).
- s2 qualitative: size/magnitude words + comparatives dominate; code/scraped identifiers at the top; the 56-token
  D≈5.16 glitch block moved from median W 0.322 (s1) to 0.292 → all drop below 0.3 (threshold-edge shift, noted as such).
  " small" (in the s2 prefix) 0.342 → 0.211. Section 2 prose drafted in REPORT.md. Waiting for s3/s4 (~23:10 / ~23:50).

## 2026-09-29 — iteration 10, continued (23:30)

- s3 done 22:49 (1,307 > 0.3), s4 done 23:15 (736). Ran analyze.py + report_lists.py.
- Cross-context: 229 in all four; unique s1 120, s2 405, s3 690, s4 143. Unique tokens are mostly near threshold
  (median own W 0.32–0.34). Top-50 of each context: 24–36 are in the all-four set. Themed unique items noted as
  observations only (s1 shape words, s3 "dream big" verbs, s4 negative adjectives).
- Fixed rounding bug in unique assignment (see CHANGELOG).
- Word budget: main body ~2,810 words; appendices hold the complete token lists that PLAN's success criterion
  explicitly requires inside REPORT.md (treated as a declared technical appendix; placed in collapsible blocks).
- Success criterion items 1–5 met; check_render passes on REPORT.md and RESULTS.md; no feedback files → STOP.

On track? yes — all sections complete, 100%, no blocker.

## 2026-09-29 — iteration 11 (finalize only)

- Checked the success criterion against Current status: already satisfied. No feedback files present.
- check_render.py passes on REPORT.md and RESULTS.md (0 problems). Only STOP.over-budget.* existed, so wrote STOP. No experiments run.
