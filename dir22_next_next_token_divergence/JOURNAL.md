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
