# BUDGET — shared resource & time limits (single source of truth)

> Both research loops (`dir3_manifold` and `dir9_ood`) run **concurrently on ONE machine and
> share all hardware below**. Each agent must stay within roughly **half** of every resource.
> `run.sh` reads `HOURS` and `CPU_THREADS_PER_AGENT` from this file; every agent iteration is
> told to read this file and respect the limits. **To retune, edit the values here — nothing
> else needs to change.**

## Knobs (keep the `KEY: value` format on these lines — `run.sh` greps them)
HOURS: 5
CPU_CORES_TOTAL: 8
CPU_THREADS_PER_AGENT: 4
RAM_TOTAL_GB: 32
RAM_BUDGET_GB_PER_AGENT: 14
GPU: 1x NVIDIA RTX 3090 (24 GB VRAM), shared by 2 agents
GPU_VRAM_FRACTION_PER_AGENT: 0.45

## Rules for the agent — you are ONE of TWO agents sharing this box; assume the other is busy
- **GPU (one 3090, shared).** At startup call
  `torch.cuda.set_per_process_memory_fraction(0.45)` so you physically cannot starve the other
  agent. Keep batches small, move tensors to CPU when done, and call `torch.cuda.empty_cache()`
  between stages. GPT-2 small is tiny, so VRAM is ample as long as you don't accumulate.
- **RAM (16 GB total ≈ 7 GB each).** Do NOT hold large activation matrices in RAM. Write caches
  to disk with `np.memmap` / sharded `.npy` and stream them. If a step would exceed ~7 GB, cap
  the number of cached samples or process one layer at a time.
- **CPU (4 cores ≈ 2 each).** `torch.set_num_threads(2)` and DataLoader `num_workers <= 2`.
  (`run.sh` also exports `OMP_NUM_THREADS` / `MKL_NUM_THREADS` from `CPU_THREADS_PER_AGENT`.)
- **On CUDA OOM or the box swapping:** HALVE batch/sample size and retry — never re-run the
  same size repeatedly.
- **Time.** You have `HOURS` hours of wall-clock for the WHOLE run; the wrapper enforces it and
  tells you the remaining minutes each iteration. Reserve the final 20 minutes to finalize.
