# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-08-10 — S1–S4 complete in one iteration; example invalid

**Did.** Checked for unaddressed `human_feedback*` / `*REVIEW*` files: none. Wrote
`experiments/run_delayed.py` (S1 validation + S2 sweep + S3 metrics; reuses the norm-corrected SLERP
from dir20) and `experiments/plot_delayed.py` (the three figures PLAN names). Ran both on GPT-2 Large
from the local HF cache (`HF_HOME=/workspace/hf_home`), 101 interpolation points, budget honoured
(`set_per_process_memory_fraction(0.225)`, `set_num_threads(2)`, chunk 16 — the job is tiny, ~1 min,
no OOM). Curated RESULTS.md and REPORT.md from their templates; `check_render.py` passes on both.

**Learned.**
1. **The example is invalid (PLAN conclusion 3).** GPT-2 Large reads `"Use the codebook A = cat and
   B = dog. Complete: Symbol A"` as a continuing list: top-1 is ` =` (p = 0.340), and ` means` sits at
   6.68e-4. After ` A means` / ` B means` it opens a quotation; ` cat` beats ` dog` in *both* branches
   (0.061/0.010 and 0.046/0.011), so there is a faint directional preference but no lookup. PLAN
   forbids editing the prompt to rescue this, so I documented and stopped.
2. **The sweep is still informative, and I kept it as a clearly-scoped secondary result** (rule 9b —
   rebuild the story around the evidence that exists). Injecting the interpolated embedding at the
   symbol and reading out after ` means` — a position that can only see the symbol through attention —
   gives `w = 0.38` against the linear null's 0.80, monotone, midpoint 0.42 vs the immediate readout's
   0.45. Plateau structure propagates and keeps its boundary; it attenuates 4.0× (‖z_A−z_B‖₂ 300.2 →
   75.4) and broadens by 0.11.
3. **The honest ceiling on that result:** the delayed top-1 never changes and the top-2 margin never
   drops below 0.43. The divergence is logit geometry, not behaviour. I say so in the Conclusion
   rather than letting `w = 0.38` imply more.
4. **Bug caught before it reached a deliverable:** `argmax` on an all-false mask returned
   `flip_t_delayed = 0.0`, which reads as "flips immediately" when the truth is "never flips". Now
   `None`. Worth remembering as a pattern — `np.argmax` on a boolean condition needs an emptiness
   guard.

**Assumptions logged (loop mode, no human to ask).**
- I reported the propagation measurement alongside the invalid-example verdict instead of shipping the
  verdict alone. *Rejected alternative:* report only "invalid example" — literal to PLAN S1's "stop",
  but it discards a valid measurement already paid for and leaves the deliverable near-empty. The
  verdict stays the headline in both files and the secondary result is scoped to this one prompt.
- `d(t)` is anchored at the sweep's own `t=0` / `t=1` logits, which equal the clean endpoint runs
  (the SLERP endpoints reproduce the original embeddings exactly).
- PLAN's "JSD from the A-endpoint" across `t` is computed and stored in `results/delayed.json` but not
  shown: `d(t)` answers the same question and rule 9 says an unused metric gets cut.

**Next step.** None — the plan is complete (S1–S4 done, verdict rendered, three figures embedded in
both deliverables, no unaddressed feedback), so I wrote `STOP` per rule 11. If a human drops feedback
later, the next iteration must delete `STOP`, address it, and only re-write `STOP` when clean. The
natural follow-up, were scope reopened: find a prompt where GPT-2 Large *demonstrably* performs the
delayed lookup — verify endpoint behaviour first, then interpolate.

On track? yes — S1–S4 100% done, verdict = conclusion 3 (invalid example), no blocker.

## 2026-08-12 — new PLAN example runs clean; verdict = delayed plateau

**Did.** Checked for unaddressed `human_feedback*` / `*REVIEW*` files: none. Found PLAN.md rewritten
around a new fixed example (capital-city lookup) with status reset to "fresh start" and the old
`STOP` removed, so I re-ran the whole plan. Retargeted `experiments/run_delayed.py` to
prefix `The capital of France is Paris. The capital of`, endpoints ` Japan`/` Germany`, successor
` is`, targets ` Tokyo`/` Berlin`; switched JSD from nats to bits (the new PLAN quotes bits); folded
the two sweeps into one forward pass per `t` reading out at both the interpolated position and the
final ` is` position, which is what PLAN's "from the same forward pass" asks for. Rewrote
`experiments/plot_delayed.py` for the three figure names PLAN specifies. Ran both on GPT-2 Large from
the local HF cache (`HF_HOME=/workspace/hf_home`), 101 points, budget honoured
(`set_per_process_memory_fraction(0.225)`, `set_num_threads(2)`, chunk 16; ~1 min, no OOM).
Re-curated RESULTS.md and REPORT.md; `check_render.py` passes on both.

**Learned.**
1. **The example reproduces the PLAN's preliminary numbers essentially exactly.** Immediate endpoint
   JSD 0.0014 bits (PLAN said ~0.0014), delayed 0.9945 bits (PLAN said ~0.99), p(` is`) 0.944/0.940
   (PLAN said ~0.94). All four endpoint top-1 checks pass, so S1 does not stop the experiment.
2. **Verdict is conclusion 1, delayed plateau, and it is clean.** Immediate top-1 is ` is` at 101/101
   positions with p in [0.931, 0.944]; delayed `d(t)` is monotone with `w = 0.28` against the linear
   null 0.80, `t₅₀ = 0.48`, flat tails (`d ≤ 0.077` for `t ≤ 0.30`, `d ≥ 0.89` for `t ≥ 0.60`).
3. **Unlike the retired codebook example, the switch is behavioural.** Delayed top-1 flips
   ` Tokyo`→` Berlin` once, at `t = 0.49`; p(` Tokyo`) 0.902 at `t = 0.45` → 0.070 at `t = 0.50`.
   That removes the "logit geometry only" caveat that limited the previous write-up.
4. **The scale-free caveat is worth checking every time.** `d(t)` would look dramatic even over a
   trivial gap, so I report the gap it normalises: ‖z_A − z_B‖₂ = 462.5, large and consistent with the
   near-1-bit endpoint JSD. The flat regions are flat relative to a real swing.

**Assumptions logged (loop mode, no human to ask).**
- Reported the immediate readout only as p(` is`) and the top-1 token, and did **not** compute or plot
  an immediate `d(t)` curve. PLAN's "out of scope" explicitly forbids interpreting it; keeping it as an
  unused number would violate rule 9's "a metric no Result uses gets cut". *Rejected alternative:*
  compute it and show it as context, as the previous iteration did.
- Deleted `plots/immediate_readout.png` (the retired example's figure) rather than leaving it beside
  the new `plots/immediate_prediction.png`. It is my own output from the superseded plan, and PLAN
  names the new filename; the old→new mapping is recorded in CHANGELOG.md.
- Dropped the delayed top-2 margin metric: with an actual top-1 flip at `t = 0.49`, the margin no
  longer answers a question the Results ask.

**Next step.** None — the plan is complete (S1–S4 done, verdict rendered, three named figures embedded
in both deliverables, render checks pass, no unaddressed feedback), so I wrote `STOP` per rule 11.
