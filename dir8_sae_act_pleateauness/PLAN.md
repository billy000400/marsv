# PLAN — Direction: TODO — describe this direction

> Working folder: `dir8_sae_act_pleateauness`. The agent REWRITES "Current status" and "Next step" and ticks
> the stage boxes every iteration. Disk (this file + JOURNAL.md + RESULTS.md + ../BUDGET.md)
> is the only memory.

## Success criterion (definition of "done")
TODO — the concrete artifact(s) that mean this direction is finished, e.g. "Produce <X> and <Y>
in RESULTS.md, plus REPORT.md with a clear verdict." A null/negative result is still COMPLETE if
the question is answered. When done, the loop writes an empty `STOP` file.

## Fallback (if time runs short)
TODO — the minimum acceptable deliverable. The wrapper reserves the final 20 min to finalize
whatever exists into RESULTS.md + REPORT.md, then STOP.

## Setup (fixed)
- TODO — model / data / hook points. Default: GPT-2 small via HuggingFace `transformers` (already
  installed) + forward hooks; STREAM data, do not bulk-download.
- **Shared hardware + time limits live in `../BUDGET.md` — read it every iteration.** You share one
  GPU / RAM / CPU with another agent, so stay within your half: cap VRAM with
  `torch.cuda.set_per_process_memory_fraction`, memmap caches, keep batches small, halve on OOM.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, or flax** — they
  downgrade and break the cluster's CUDA build. Use the existing env; add pure-python deps with
  `--no-deps`.

## Stages (checklist — update marks each iteration)
- [ ] S1 — TODO
- [ ] S2 — TODO
- [ ] S3 — TODO

## Out of scope (do NOT)
- TODO — anything explicitly out of bounds for this direction.
- Don't drift into other directions.

## On-track check (required every iteration)
End each JOURNAL.md entry with one line: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status
(none yet — fresh start)

## Next step
TODO — the first concrete action.
