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
