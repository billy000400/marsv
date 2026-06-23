#!/usr/bin/env bash
# setup_tree.sh — build the autonomous-research scaffold inside your project root.
# Usage:  bash setup_tree.sh [PROJECT_ROOT]   (defaults to current dir)
# Additive: creates BUDGET.md, run.sh, dir3_manifold/ and dir9_ood/ + reference files.
# Does NOT touch your existing code/ dataset/ results/ dirs. Re-running overwrites only
# the scaffold files it writes.
set -euo pipefail
ROOT="${1:-.}"
[ -d "$ROOT" ] || { echo "project root '$ROOT' not found"; exit 1; }
cd "$ROOT"
echo "[setup] writing scaffold into $(pwd)"

mkdir -p dir3_manifold/experiments dir9_ood/experiments

cat > run.sh <<'__EOF_run_sh__'
#!/usr/bin/env bash
# Iteration-loop wrapper for one autonomous research direction.
# Usage:  ./run.sh <research-subdir> [hours]   (hours may be fractional, e.g. 4.5)
# Example: ./run.sh dir3_manifold 4.5
# Run under tmux + an outer `timeout` as a guaranteed hard kill:
#   tmux new-session -d -s dir3 'timeout 14700 ./run.sh dir3_manifold 4'   # 4h + 5min grace
set -uo pipefail

# This pod runs as root, and Claude Code blocks --dangerously-skip-permissions for uid 0
# unless a sandbox is signalled. The pod is dedicated/disposable, so we signal sandbox here.
# (The deny-list — rm -rf /, .git/.claude writes, etc. — still applies.)
export IS_SANDBOX=1

DIR="${1:?usage: run.sh <research-subdir> [hours]}"

# Frequently-changed knobs live in BUDGET.md at the project root (this dir, before we cd).
BUDGET_FILE="BUDGET.md"
BUDGET_ABS="$(pwd)/$BUDGET_FILE"
read_budget() { grep -E "^$1:" "$BUDGET_FILE" 2>/dev/null | head -1 | sed -E "s/^$1:[[:space:]]*//; s/[[:space:]].*$//"; }

# Time budget: CLI arg $2 wins, else BUDGET.md HOURS, else 4.
HOURS="${2:-$(read_budget HOURS)}"; HOURS="${HOURS:-4}"
# CPU thread cap so two concurrent agents don't oversubscribe the 4 cores.
CPU_THREADS="$(read_budget CPU_THREADS_PER_AGENT)"; CPU_THREADS="${CPU_THREADS:-2}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"

FINALIZE_MIN=20                                  # reserve this long to wrap up
BUDGET_SEC=$(awk "BEGIN{printf \"%d\", $HOURS*3600}")   # float-safe (handles 4.5)
END=$(( $(date +%s) + BUDGET_SEC ))              # hard wall-clock deadline

cd "$DIR" || { echo "[run.sh] cannot cd into $DIR"; exit 1; }
mkdir -p experiments results

echo "[run.sh] start $(date '+%F %T')  dir=$DIR  budget=${HOURS}h  cpu_threads=${CPU_THREADS}"

while [ "$(date +%s)" -lt "$END" ] && [ ! -f STOP ]; do
  REMAIN=$(( (END - $(date +%s)) / 60 ))
  echo "[run.sh] $(date '+%F %T')  ~${REMAIN} min left  -----------------------------"

  claude -p "You are mid-project and your working memory RESETS every iteration. \
FIRST read BUDGET.md (at ${BUDGET_ABS}), then PLAN.md, JOURNAL.md, and RESULTS.md in full. \
BUDGET.md has the SHARED GPU/RAM/CPU limits you MUST stay within — you share one machine with \
another agent, so cap VRAM via torch.cuda.set_per_process_memory_fraction, memmap large caches, \
set torch threads to the CPU budget, and HALVE batch size on any OOM. \
You have ${REMAIN} minutes of wall-clock left. \
If that number is <= ${FINALIZE_MIN}: do ONLY finalization — write the final summary and headline \
into RESULTS.md, write/update REPORT.md from whatever exists, then create an empty STOP file and stop. \
Otherwise do ONE focused iteration: advance the plan by the smallest useful step, write/modify code \
under experiments/, RUN it, record any metrics in RESULTS.md, then append to JOURNAL.md (what you did, \
what you learned, the revised next step) and update PLAN.md's 'Current status', 'Next step', and stage \
checkboxes. End the JOURNAL entry with the required one-line 'On track?' check. \
Persist ALL state to disk before you finish; assume nothing carries over." \
    --append-system-prompt "Durable state lives only on disk (PLAN.md, JOURNAL.md, RESULTS.md, experiments/, results/). You share one GPU/CPU/RAM with another agent — respect the limits in BUDGET.md every iteration. Persist every iteration. Prefer small verifiable steps over big leaps. If an experiment breaks, debug minimally or record the failure and fall back per PLAN.md rather than rabbit-holing." \
    --dangerously-skip-permissions \
    2>&1 | tee -a session.log

  sleep 2
done

if [ -f STOP ]; then
  echo "[run.sh] STOP file present — goal reached. Ended $(date '+%F %T')."
else
  echo "[run.sh] deadline hit — ended $(date '+%F %T')."
fi
__EOF_run_sh__

cat > BUDGET.md <<'__EOF_budget__'
# BUDGET — shared resource & time limits (single source of truth)

> Both research loops (`dir3_manifold` and `dir9_ood`) run **concurrently on ONE machine and
> share all hardware below**. Each agent must stay within roughly **half** of every resource.
> `run.sh` reads `HOURS` and `CPU_THREADS_PER_AGENT` from this file; every agent iteration is
> told to read this file and respect the limits. **To retune, edit the values here — nothing
> else needs to change.**

## Knobs (keep the `KEY: value` format on these lines — `run.sh` greps them)
HOURS: 4
CPU_CORES_TOTAL: 4
CPU_THREADS_PER_AGENT: 2
RAM_TOTAL_GB: 16
RAM_BUDGET_GB_PER_AGENT: 7
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
__EOF_budget__

cat > README_LAUNCH.md <<'__EOF_readme__'
# Launch guide — two autonomous research loops

## Layout (drop this inside your existing `<PROJECT_ROOT>` from item #3)
```
<PROJECT_ROOT>/
  run.sh                       # generic iteration-loop wrapper
  .claude/settings.json        # OPTIONAL: the acceptEdits scoped variant (see below)
  dir3_manifold/               # Direction #3 — manifold characterization
    PLAN.md  JOURNAL.md  RESULTS.md  experiments/
  dir9_ood/                    # Direction #9 — plateaus as OOD detector
    PLAN.md  JOURNAL.md  RESULTS.md  experiments/
```
Each loop runs with its working dir = the direction folder, so everything (data, results, STOP) stays self-contained per direction and the two never collide.

## Pre-flight (do these ONCE before launching, or the agent burns iterations on setup)
```bash
chmod +x run.sh
# DO NOT bulk-install torch/transformer_lens/cupbearer — they downgrade a custom CUDA build.
# Install ONLY the light, torch-independent deps (transformers is usually already present):
pip install --no-cache-dir transformers datasets scikit-learn matplotlib
# scikit-dimension is optional (pulls numba); skip if it fights — the agent can hand-roll TwoNN:
pip install --no-cache-dir scikit-dimension || echo "skip scikit-dimension; agent will hand-roll TwoNN"
# confirm torch is intact, deps import, FineWeb is reachable, and the key is set:
python -c "import torch, transformers, datasets, sklearn; print('ok', torch.__version__, torch.cuda.is_available())"
echo "$ANTHROPIC_API_KEY" | head -c 6   # should print sk-ant
```
> None of `transformers / datasets / scikit-learn / matplotlib / scikit-dimension` depend on torch or torchvision, so they cannot disturb your CUDA build. The plans use HuggingFace `transformers` + forward hooks (not transformer_lens) and a self-contained OOD setup (not cupbearer) specifically to avoid the dependency conflicts that break a custom torch.

### Recommended: isolate in a venv so the cluster base env is never touched
```bash
python -m venv --system-site-packages ~/marsv/.venv   # inherits the cluster's torch
source ~/marsv/.venv/bin/activate                      # activate in the SAME shell you launch from
pip install --no-cache-dir transformers datasets scikit-learn matplotlib
# then launch the tmux sessions from this activated shell so claude/python use the venv
```

### Optional: cupbearer for dir9 (install from GitHub, NOT PyPI)
PyPI `cupbearer==0.0.1` is the old JAX build and will wreck a custom torch. The GitHub
build is PyTorch-based with lower-bound-only torch pins, so it won't downgrade torch —
but it pulls `numpy<2` + lightning/torchattacks/torchmetrics. Install it yourself, in the
venv, dry-run first; the agent is told to USE it if importable but never to install it.
```bash
source ~/marsv/.venv/bin/activate
pip install --dry-run "git+https://github.com/ejnnr/cupbearer.git"   # read what changes
# only if torch/torchvision are NOT in the change list:
pip install "git+https://github.com/ejnnr/cupbearer.git"
```

## Launch (each loop in its own tmux session, outer `timeout` as a hard kill)
```bash
tmux new-session -d -s dir3 'timeout 14700 ./run.sh dir3_manifold 4'   # 4h budget, +5min hard kill
tmux new-session -d -s dir9 'timeout 14700 ./run.sh dir9_ood 4'

# check in:    tmux attach -t dir3     (detach again: Ctrl-b then d)
# tail a log:  tail -f dir3_manifold/session.log
# stop early:  touch dir3_manifold/STOP   (the loop exits after the current iteration)
```
GPT-2 small is tiny, so both loops sharing one GPU is usually fine; if you see contention during activation collection, stagger their starts by a few minutes.

## Retuning resources/time — edit BUDGET.md only
`BUDGET.md` (project root) is the single source of truth for the knobs that change often:
GPU/RAM/CPU and the `HOURS` time budget. `run.sh` reads `HOURS` and `CPU_THREADS_PER_AGENT`
from it (a CLI `[hours]` arg still overrides), and every agent iteration is told to read it and
stay within its half of the shared 3090 / 16 GB RAM / 4 CPU. Change a value there and relaunch —
no need to touch run.sh or the plans.

## IMPORTANT — the permission flag (this is the one fix to the original snippet)
For a fully unattended loop you want NO approval prompts. The canonical flag is
`--dangerously-skip-permissions` (equivalent to `--permission-mode bypassPermissions`); that is what
`run.sh` uses. Two things to know:
- **Verify on first run.** On some builds, starting interactively in bypass mode shows a one-time
  confirmation dialog. In `-p` (print/headless) mode with output piped to `tee` there's no TTY, so it
  normally proceeds — but watch the very first iteration in `tmux attach`. If it stalls waiting on a
  prompt, that's the gotcha; re-verify the flag name with `claude --help` on your installed version.
- **Do NOT swap in `--permission-mode dontAsk` as a drop-in.** `dontAsk` does not prompt, but it
  *denies* any tool not explicitly allow-listed — so a free-running experiment loop would have Bash
  denied. It only works paired with an allow-list (like the acceptEdits variant below).

### Safer alternative: acceptEdits + scoped allow-list
If you'd rather not grant full system access:
1. Copy `claude_settings_acceptEdits.json` to `<PROJECT_ROOT>/.claude/settings.json`.
2. In `run.sh`, replace `--dangerously-skip-permissions` with `--permission-mode acceptEdits`.
This auto-approves file edits and lets through exactly the commands in the allow-list (python, pip,
pytest, git). You MUST enumerate every command the agent needs, or it stalls with nobody watching —
check `session.log` if an iteration goes quiet.

## Safety note
`bypassPermissions` / `--dangerously-skip-permissions` grants the agent unrestricted shell on the pod.
That's acceptable precisely because it's a remote, dedicated compute pod with nothing else you care
about on it — never run this on a machine with credentials or work you can't lose. Keep your real
secrets off the pod.

## Two cheap reliability mitigations (already baked in)
- Each PLAN.md opens with a crisp success criterion **and** a fallback, so a short or broken run still
  produces a finalized RESULTS.md + REPORT.md.
- Every iteration must end its JOURNAL.md entry with an `On track? yes/no — ...` line, so drift is
  visible at a glance when you skim the log.
```
```
__EOF_readme__

cat > claude_settings_acceptEdits.json <<'__EOF_settings__'
{
  "_comment": "SAFER VARIANT. Place at <PROJECT_ROOT>/.claude/settings.json and change run.sh's flag from --dangerously-skip-permissions to --permission-mode acceptEdits. acceptEdits auto-approves file edits + filesystem bash (mkdir/touch/rm/mv/cp/sed) inside the working dir, but PROMPTS for other Bash (python, pip, pytest, git) unless allow-listed here. VERIFY the rule syntax against your build with `claude --help` / current docs before relying on it unattended.",
  "defaultMode": "acceptEdits",
  "permissions": {
    "allow": [
      "Bash(python:*)",
      "Bash(python3:*)",
      "Bash(pip install:*)",
      "Bash(pip3 install:*)",
      "Bash(python -m pip:*)",
      "Bash(pytest:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git status:*)",
      "Bash(timeout:*)",
      "Bash(nohup:*)",
      "Edit(./**)",
      "Write(./**)",
      "Read(./**)"
    ],
    "deny": [
      "Bash(rm -rf /:*)",
      "Bash(sudo:*)",
      "Read(./.env)",
      "Read(./**/*secret*)",
      "Read(./**/*credential*)"
    ]
  }
}
__EOF_settings__

cat > dir3_manifold/PLAN.md <<'__EOF_plan3__'
# PLAN — Direction #3: Manifold Characterization of the GPT-2 Residual Stream

> The agent REWRITES "Current status" and "Next step" and ticks the stage boxes every iteration.
> Disk (this file + JOURNAL.md + RESULTS.md) is the only memory. All paths are relative to this folder.

## Success criterion (definition of "done")
Produce all three:
1. Intrinsic-dimension (ID) estimates of the GPT-2 residual stream on FineWeb, per layer, from >=3 estimators (TwoNN, MLE, PCA participation ratio) — in RESULTS.md.
2. An autoencoder bottleneck sweep (held-out reconstruction error vs k) with an identified elbow — in RESULTS.md.
3. REPORT.md stating whether the AE elbow-k agrees with the nonlinear ID estimates (and how both compare to the linear PCA value and to d_model = 768).

**A disagreement between the two estimates is a complete, valid result.** When all three exist, create an empty `STOP` file.

## Fallback (if time runs short)
Minimum acceptable: ID estimates for layer 6 from TwoNN + MLE + PCA, plus a partial AE sweep over at least k in {4,8,16,32,64,128}. Always use the final 20 min to finalize whatever exists into RESULTS.md + REPORT.md and STOP.

## Setup (fixed)
- Model: GPT-2 small (124M, d_model=768, 12 layers). Use HuggingFace `transformers` (already installed) with forward hooks on each block's output to capture the residual stream. **Do NOT `pip install transformer_lens`** — its `torchvision<0.23` pin downgrades and breaks the cluster's CUDA-13 torch.
- **Shared hardware + time limits live in `../BUDGET.md` — read it every iteration.** You share one RTX 3090, 16 GB RAM, and 4 CPU with the other agent, so stay within your half: cap VRAM (`set_per_process_memory_fraction`), memmap activation caches to disk, keep batches small, halve on OOM.
- Data: FineWeb — STREAM a sample, do not download all. ~2-5k sequences, length 128-512.

## Stages (checklist)
- [ ] **S1 — collect activations.** Hook `resid_post` at layers {0,3,6,9,11}; gather ~200k vectors/layer into `data/acts_layer{L}.npy` (+ `data/collection_meta.json`). Record the mean-centering choice; pool all token positions; do NOT unit-normalize.
- [ ] **S2 — intrinsic dimension.** Per layer: TwoNN, MLE (prefer `scikit-dimension`), PCA participation ratio (linear contrast). Check stability across subsample sizes (10k/50k/200k). Write the table to RESULTS.md.
- [ ] **S3 — AE bottleneck sweep (layer 6).** Fixed deep MLP `768->512->256->k->256->512->768` (GELU); vary ONLY k in {2,4,8,16,24,32,48,64,128,256}; identical optimizer/LR/batch/steps/split across all k. Metric: held-out fraction-of-variance-unexplained. Append one RESULTS.md row per k immediately. Identify the elbow.
- [ ] **S4 — report.** Write REPORT.md (elbow-k vs ID vs PCA vs 768) and embed/reference the two plots. Create `STOP`.
- [ ] *(stretch, only if S1-S4 done)* TDA persistent homology (`ripser`/`giotto-tda`) on a layer-6 subsample.

## Out of scope (do NOT)
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, or flax** — they downgrade and break the cluster's CUDA-13 torch. Use only the already-installed env (torch + HuggingFace `transformers` + numpy/sklearn/matplotlib). Install missing pure-python packages with `--no-deps`.
- Don't use the manifold for any downstream task (steering / probing / editing) — utility is a separate open question.
- Don't train large models or let any single run exceed ~10 min; the AEs are small MLPs.
- Don't drift into other directions.

## On-track check (required every iteration)
End each JOURNAL.md entry with one line: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status
(none yet — fresh start)

## Next step
Begin S1: write `experiments/collect_acts.py` and run it to cache activations.
__EOF_plan3__

cat > dir3_manifold/JOURNAL.md <<'__EOF_journal3__'
# JOURNAL — Direction #3 (Manifold)

Append-only. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---
__EOF_journal3__

cat > dir3_manifold/RESULTS.md <<'__EOF_results3__'
# RESULTS — Direction #3 (Manifold)

## Intrinsic dimension estimates
| layer | estimator | n_points | id_mean | id_std |
|-------|-----------|----------|---------|--------|

## Autoencoder bottleneck sweep (layer 6)
| k | val_frac_var_unexplained | train_loss | val_loss |
|---|--------------------------|------------|----------|

## Headline
_(filled at finalize: AE elbow-k vs nonlinear ID vs PCA participation ratio vs d_model=768)_
__EOF_results3__

cat > dir9_ood/PLAN.md <<'__EOF_plan9__'
# PLAN — Direction #9: Plateau-ness as an OOD / Anomaly Detector

> The agent REWRITES "Current status" and "Next step" and ticks the stage boxes every iteration.
> Disk (this file + JOURNAL.md + RESULTS.md) is the only memory. All paths are relative to this folder.

## Success criterion (definition of "done")
A fair AUROC comparison of >=2 plateau-score variants (each evaluated at the residual-layer sweep {3,6,9} and input-space) against >=3 baselines (activation L2 norm, Mahalanobis distance, Maximum Softmax Probability) on >=1 OOD task — in RESULTS.md — plus REPORT.md giving a plain verdict on whether plateau-ness beats the baselines, and whether measuring internally (residual stream) beats the simpler input-space signal.

**A null result (it does not beat them) is complete and acceptable.** When done, create an empty `STOP` file.

## Fallback (if time runs short)
Minimum acceptable: one plateau variant (perturbation-sensitivity) computed at a single point (`resid_post`@6) vs the 3 baselines on the self-contained OOD setup (held-out FineWeb vs random/shuffled tokens), AUROC in RESULTS.md. Reserve the final 20 min to finalize + STOP.

## Setup (fixed)
- Model: GPT-2 small (124M). The plateau/robustness score is computed at a **configurable measurement point** — do NOT hardcode the residual stream. Defaults: a residual-stream layer sweep (`resid_post` at layers {3,6,9}, reporting per-layer AUROC) using the last-token (or mean-over-positions) activation, AND an **input-space** variant (perturb token embeddings, measure output sensitivity — no internal hook). The residual stream is the choice faithful to the prior plateau characterization; input-space is the simpler signal it must beat to be interesting.
- In-distribution data: FineWeb text.
- **Shared hardware + time limits live in `../BUDGET.md` — read it every iteration.** You share one RTX 3090, 16 GB RAM, and 4 CPU with the other agent, so stay within your half: cap VRAM (`set_per_process_memory_fraction`), keep perturbation batches and direction counts modest, halve on OOM.

## Stages (checklist)
- [ ] **S1 — plateau score (measurement-point-agnostic).** Implement `experiments/plateau_score.py` so the measurement point is a PARAMETER (a residual layer's `resid_post`, or the input embeddings) — not hardcoded. Two scalar variants: (a) **perturbation-sensitivity** — N random unit directions at the chosen point, sweep magnitudes eps, continue the forward pass from the perturbed state, measure KL of the next-token distribution vs unperturbed; summarize as radius eps* at a KL threshold or mean KL at fixed small eps (flatter = more in-distribution); (b) **gradient/Jacobian norm** of the output w.r.t. the chosen point (lower = flatter; generalizes to any point cheaply via autograd, no eps sweep). Default measurement points to evaluate: residual stream at layers {3,6,9} AND input-space (token embeddings). Keep cheap: 16-32 directions, a few eps, subsampled positions, batched. Sanity-check it separates FineWeb vs random tokens at all (quick AUROC).
- [ ] **S2 — OOD task + baselines.** **If `cupbearer` is already importable** in this env (the PyTorch GitHub build, pre-installed by the user), you may use it for a proper mechanistic-anomaly benchmark and its baselines. **Otherwise use the self-contained setup** — and do NOT install cupbearer yourself (only the PyPI `0.0.1` JAX build is what `pip` would auto-pick, and it breaks the cluster torch). Self-contained: ID = held-out FineWeb; OOD (2-3 sets) = random/shuffled tokens, a different domain (e.g. code), optional char-level corruption. Baselines (implement regardless): activation L2 norm; Mahalanobis distance to a Gaussian fit on ID activations; Maximum Softmax Probability. Whatever the path, do not `pip install` any deep-learning framework — use the already-installed env.
- [ ] **S3 — evaluate + report.** Write `results/auroc_table.csv` `[task, ood_set, method, measurement_point, auroc]` — one row per (OOD set × method × measurement point), covering both plateau variants at each residual layer {3,6,9} AND input-space, plus the 3 baselines. ROC and score-distribution plots under `results/plots/`. Write REPORT.md with a per-OOD-set verdict that explicitly answers: (i) does any plateau variant beat the baselines, and (ii) does the residual-stream (internal-activation) plateau signal beat the simpler input-space sensitivity — i.e. is there value in measuring internally. Create `STOP`.

## Out of scope (do NOT)
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, or flax** — they downgrade and break the cluster's CUDA-13 torch. Use only the already-installed env (torch + HuggingFace `transformers` + numpy/sklearn/matplotlib). If a needed pure-python package is missing, install it with `--no-deps`.
- Don't make the score differentiable or use it for steering/correction — separate direction.
- Don't drift into other directions.

## On-track check (required every iteration)
End each JOURNAL.md entry with one line: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status
(none yet — fresh start)

## Next step
Begin S1: implement `experiments/plateau_score.py` and run the FineWeb-vs-random sanity AUROC.
__EOF_plan9__

cat > dir9_ood/JOURNAL.md <<'__EOF_journal9__'
# JOURNAL — Direction #9 (OOD)

Append-only. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---
__EOF_journal9__

cat > dir9_ood/RESULTS.md <<'__EOF_results9__'
# RESULTS — Direction #9 (OOD)

## AUROC table
| task | ood_set | method | measurement_point | auroc |
|------|---------|--------|-------------------|-------|

_methods: plateau-perturbation, plateau-jacobian, baseline-L2norm, baseline-mahalanobis, baseline-MSP_
_measurement_point: resid3 / resid6 / resid9 / input-space (n/a for baselines as appropriate)_

## Headline
_(filled at finalize: does plateau-ness beat the baselines? which variant, on which OOD sets, and does the residual-stream signal beat input-space?)_
__EOF_results9__

touch dir3_manifold/experiments/.gitkeep dir9_ood/experiments/.gitkeep
chmod +x run.sh
echo "[setup] done. Tree:"
if command -v tree >/dev/null 2>&1; then tree -L 2 -a -I '.git'; else find . -maxdepth 2 | sort; fi