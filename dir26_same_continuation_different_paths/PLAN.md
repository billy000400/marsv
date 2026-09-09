# PLAN â Direction: Same continuation, different computational paths

> Working folder: `same_continuation_different_paths`. Agent REWRITES "Current status"/"Next step" + ticks stages each iteration. Disk (PLAN/JOURNAL/RESULTS/CHANGELOG + ../BUDGET.md + ../CLAUDE.md) is the only memory.

## Success criterion (definition of "done")
Test the four prompt pairs below on GPT-2 Large. For every pair that completes as intended, produce one `d(t)` interpolation plot and give a simple visual verdict: **plateau visible** or **no clear plateau**. `REPORT.md` should contain the exact prompts, greedy completions, the plots, and a short interpretation. Negative results are complete results. When done, write an empty `STOP` file.

## Fallback (if time runs short)
Test all four prompt pairs for completion, then produce `d(t)` plots for the first two passing pairs, prioritizing examples 1 and 3. Report the remaining passing pairs as not yet plotted. Do not add extra analyses.

## Setup (fixed)
- **Model:** GPT-2 Large (`gpt2-large`).
- **Prompt pairs:**
  1. Copy available vs factual recall
     - A: `The capital of France is Paris. The capital of France is`
     - B: `The capital of Spain is Madrid. The capital of France is`
     - Intended completion for both: `Paris`
  2. Successor vs predecessor
     - A: `The number after six is`
     - B: `The number before eight is`
     - Intended completion for both: `seven`
  3. Different relation, same answer
     - A: `The capital of France is`
     - B: `The largest city in France is`
     - Intended completion for both: `Paris`
  4. Different entity, same relation and answer
     - A: `The author of Hamlet is`
     - B: `The author of Macbeth is`
     - Intended completion for both: `Shakespeare` (accept `William Shakespeare`)
- **Completion check:** greedily generate a short continuation from each prompt. A pair passes only if both prompts give the intended answer. Save the literal completions in `RESULTS.md`. No probability tables or statistical tests.
- **Interpolation:** use the standard activation-plateau setup with GPT-2 Large. Interpolate `resid_post` after block 0 with 50 interpolation steps and record the final-position logits. Use the same SLERP-with-norm-rescaling interpolation used in the activation-plateau reference implementation.
- Because these prompt pairs can differ at more than one token, interpolate the **full residual-stream sequence**, not only the final token. Only run the plateau experiment when the two prompts have the same GPT-2 tokenized length.
- If a passing pair has unequal tokenized lengths, make at most one minimal wording adjustment that preserves the intended contrast, then rerun the completion check. For example, for pair 3 try `The capital city of France is` vs `The largest city in France is`. Do not search over many prompt variants.
- For interpolation step `t`, let `x(t)` be the final-position logit vector and let `A` and `B` be the endpoint logit vectors. Plot

  `d(t) = ||x(t) - A|| / (||x(t) - A|| + ||x(t) - B||)`

  against interpolation position `t` from 0 to 1.
- **Plateau verdict:** inspect the curve visually. Call it a plateau only if `d(t)` stays visibly close to one endpoint for an extended part of the path and then changes much more sharply over a shorter region. Otherwise report `no clear plateau`. Do not define or fit a numerical threshold.
- If the two endpoint logit vectors are essentially indistinguishable and `d(t)` is therefore uninformative, say so and stop for that pair rather than introducing another metric.
- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md` â read both every iteration.**
- **Deliverable hygiene (see CLAUDE.md):** RESULTS.md/REPORT.md = current-best only, no history; CHANGELOG.md = the history.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, flax** â they break the CUDA build.

## Stages (checklist)
- [x] **S1 â Completion screen.** Run all 8 exact prompts on GPT-2 Large. Save the greedy continuations and mark each pair pass/fail. Check tokenized lengths for passing pairs.
- [x] **S2 â Plateau plots.** For each passing, length-matched pair, run the fixed 50-step interpolation and save one `d(t)` figure to `plots/`.
- [x] **S3 â Minimal report.** For each pair, report: intended contrast, actual completion, `d(t)` plot if applicable, and one-line verdict (`plateau visible`, `no clear plateau`, or `not testable because ...`). End with a short comparison of which examples worked best.

## Out of scope (do NOT)
- No layer sweep.
- No model sweep.
- No correlation, significance tests, confidence intervals, fitted change points, or plateau scores.
- No attention-head, MLP-neuron, probe, SAE, or causal-circuit analysis.
- No large prompt search. One minimal wording repair is allowed only when token length prevents interpolation.
- Do not claim that a visible plateau proves two different computational pathways. This experiment only asks whether these hand-designed same-continuation contrasts produce a plateau.

## On-track check (required every iteration)
End each JOURNAL.md entry with: `On track? <yes/no> â <stage, % done, blocker if any>`.

## Current status
DONE. All four prompt pairs were screened on GPT-2 Large with greedy decoding. Only pair 1
(copy-available vs factual-recall) gives the intended answer from both prompts, and its two prompts
are both 12 tokens. The one authorized wording repair was spent on pair 3 (`The capital city of
France is`): it equalised lengths at 6 tokens but the A side still does not answer Paris, so pair 3
remains ineligible. Pair 2 fails on both sides, pair 4 fails on both sides and is length-mismatched.
The 50-step full-sequence block-0 interpolation was run for pair 1: `d(t)` covers 17% of the endpoint
gap over the first 57% of the path, then 71% over the next 10% -> verdict **plateau visible**
(`plots/p1_copy_vs_recall_dt.png`). Pairs 2-4: **not testable**. REPORT.md and RESULTS.md are
written and pass `check_render.py`. `STOP` written.

## Next step
None. Direction complete.
