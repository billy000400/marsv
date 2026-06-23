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
