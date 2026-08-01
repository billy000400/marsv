# BUDGET — shared resource & time limits (single source of truth)

> `run.sh` reads the INPUTS below and **derives** each agent's share at launch. The GPU is NOT
> listed — `run.sh` auto-detects the current card with `nvidia-smi` every launch (the pod may get
> a different GPU each time), and `set_per_process_memory_fraction` is a fraction of whatever card
> is present, so it adapts automatically. **To retune, edit these values — nothing else changes.**

## Inputs (keep the `KEY: value` format — run.sh greps these)
MODEL: opus                 # claude model per iteration: alias (opus/sonnet/fable) or full id (claude-opus-5)
HOURS: 4
N_AGENTS: 4                 # how many loops you launch CONCURRENTLY — set to match reality
CPU_CORES_TOTAL: 8          # static
RAM_TOTAL_GB: 32            # static
VRAM_HEADROOM_FRACTION: 0.1 # leave this fraction of the card free (shared headroom)

## Derived by run.sh from the above, and told to each agent every iteration:
##   VRAM fraction / agent = (1 - VRAM_HEADROOM_FRACTION) / N_AGENTS
##   CPU threads   / agent = CPU_CORES_TOTAL / N_AGENTS   (min 1)
##   RAM budget    / agent = RAM_TOTAL_GB / N_AGENTS
## (e.g. N_AGENTS=2, 4 CPU, 16 GB  ->  vram_frac 0.45, 2 threads, 8 GB RAM each)

## Rules for the agent — you are 1 of N_AGENTS sharing this box; assume the others are busy
- **GPU.** Call `torch.cuda.set_per_process_memory_fraction(<the fraction run.sh gives you>)` at
  startup so you can't starve the others. Small batches; move tensors off-GPU when done;
  `torch.cuda.empty_cache()` between stages.
- **RAM.** Stay under your per-agent GB. Don't hold large activation matrices in RAM —
  `np.memmap` / sharded `.npy` and stream.
- **CPU.** `torch.set_num_threads(<your thread budget>)`; DataLoader `num_workers <= that`.
- **On CUDA OOM / swapping:** HALVE batch/sample size and retry — never re-run the same size.
- **Time.** You have `HOURS` hours total; the wrapper enforces it and reports remaining minutes.
  Reserve the last 20 min to finalize.
