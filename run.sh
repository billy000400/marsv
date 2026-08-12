#!/usr/bin/env bash
# Iteration-loop wrapper for one autonomous research direction.
# Usage:  ./run.sh <research-subdir> [hours]
# Example: ./run.sh dir3_manifold 4   (or launch via ./launch.sh dir3_manifold)
set -uo pipefail

export IS_SANDBOX=1              # allow --dangerously-skip-permissions as root on this disposable pod
export MPLBACKEND=Agg            # headless matplotlib: figures save to file, never try to display
export GIT_TERMINAL_PROMPT=0     # never block on a git credential prompt (HTTPS remote, no stored creds)

# Don't depend on the launching shell's PATH: the claude CLI location varies per environment
# (this box installs it via nvm at ~/.nvm/versions/node/*/bin; other pods used ~/.local/bin),
# and tmux/cron/non-login shells may not have it on PATH. Prepend every plausible location,
# then fail LOUDLY if claude is still missing — otherwise every iteration silently prints
# 'claude: command not found' and the loop spins for hours doing nothing.
export PATH="${HOME:-/home/user}/.local/bin:$PATH"
for _d in "${HOME:-/home/user}"/.nvm/versions/node/*/bin; do
  [ -d "$_d" ] && PATH="$_d:$PATH"
done
export PATH
command -v claude >/dev/null 2>&1 || {
  echo "[run.sh] FATAL: 'claude' not found on PATH (looked in ~/.local/bin and ~/.nvm/versions/node/*/bin)." >&2
  echo "[run.sh] Install/symlink claude or fix PATH, then relaunch. Aborting instead of spin-failing." >&2
  exit 127
}

DIR="${1:?usage: run.sh <research-subdir> [hours]}"

BUDGET_FILE="BUDGET.md"
BUDGET_ABS="$(pwd)/$BUDGET_FILE"
RULES_ABS="$(pwd)/CLAUDE.md"     # operator rules (read before write, curation, report structure)
WRITING_ABS="$(pwd)/WRITING.md" # appended with Claude's supported system-prompt-file flag
WORKFLOW_ABS="$(pwd)/workflow.py"
CHECK_MD_ABS="$(pwd)/check_md.py"
CHECK_RENDER_ABS="$(pwd)/check_render.py"
[ -f "$WRITING_ABS" ] && [ -f "$WORKFLOW_ABS" ] && [ -f "$CHECK_MD_ABS" ] && [ -f "$CHECK_RENDER_ABS" ] || {
  echo "[run.sh] missing WRITING.md, workflow.py, check_md.py, or check_render.py" >&2
  exit 1
}
read_budget() { grep -E "^$1:" "$BUDGET_FILE" 2>/dev/null | head -1 | sed -E "s/^$1:[[:space:]]*//; s/[[:space:]].*$//"; }

MODEL="$(read_budget MODEL)";                    MODEL="${MODEL:-opus}"
HOURS="${2:-$(read_budget HOURS)}";              HOURS="${HOURS:-4}"
N_AGENTS="$(read_budget N_AGENTS)";              N_AGENTS="${N_AGENTS:-1}"
CPU_CORES="$(read_budget CPU_CORES_TOTAL)";      CPU_CORES="${CPU_CORES:-4}"
RAM_TOTAL="$(read_budget RAM_TOTAL_GB)";         RAM_TOTAL="${RAM_TOTAL:-16}"
HEADROOM="$(read_budget VRAM_HEADROOM_FRACTION)";HEADROOM="${HEADROOM:-0.1}"

CPU_THREADS=$(awk "BEGIN{n=$CPU_CORES/$N_AGENTS; if(n<1)n=1; printf \"%d\", n}")
RAM_PER_AGENT=$(awk "BEGIN{printf \"%.1f\", $RAM_TOTAL/$N_AGENTS}")
VRAM_FRACTION=$(awk "BEGIN{f=(1-$HEADROOM)/$N_AGENTS; if(f<=0)f=0.1; printf \"%.3f\", f}")
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"

GPU_NAME="unknown"; GPU_VRAM_GB="?"; VRAM_PER_AGENT="?"
if command -v nvidia-smi >/dev/null 2>&1; then
  _g=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
  if [ -n "${_g:-}" ]; then
    GPU_NAME=$(printf '%s' "$_g" | cut -d, -f1 | sed 's/^ *//;s/ *$//')
    _mib=$(printf '%s' "$_g" | cut -d, -f2 | grep -oE '[0-9]+' | head -1)
    GPU_VRAM_GB=$(awk "BEGIN{printf \"%.1f\", ${_mib:-0}/1024}")
    VRAM_PER_AGENT=$(awk "BEGIN{printf \"%.1f\", $GPU_VRAM_GB*$VRAM_FRACTION}")
  fi
fi

FINALIZE_MIN=20
BUDGET_SEC=$(awk "BEGIN{printf \"%d\", $HOURS*3600}")
END=$(( $(date +%s) + BUDGET_SEC ))

python3 "$WORKFLOW_ABS" prepare "$DIR" >/dev/null || exit 1
cd "$DIR" || { echo "[run.sh] cannot cd into $DIR"; exit 1; }
mkdir -p experiments results plots

echo "[run.sh] start $(date '+%F %T')  dir=$DIR  budget=${HOURS}h  agents=${N_AGENTS}  model=${MODEL}"
echo "[run.sh] gpu='${GPU_NAME}' ${GPU_VRAM_GB}GB  | per-agent: vram_frac=${VRAM_FRACTION} (~${VRAM_PER_AGENT}GB)  ram=${RAM_PER_AGENT}GB  threads=${CPU_THREADS}"

# --- git checkpoint --------------------------------------------------------
# Commit THIS direction's work and push, after every iteration. Because up to
# N_AGENTS loops share one repo, all git ops are serialized through a single
# repo-wide flock and each commit is scoped to the current direction only
# (never sibling dirs or shared root files). STOP/session.log are gitignored,
# so the working tree is clean after the commit — the rebase-pull below never
# needs to stash, so it can NEVER lose uncommitted work (a blocked pull just
# skips this push and retries next iteration). If the push fails on SSH/auth
# (e.g. the ephemeral pod restarted and lost its git-ssh config), it self-heals
# by running /mars-vol/setup_github_ssh.sh once, then retries.
SSH_SETUP="/mars-vol/setup_github_ssh.sh"
git_sync() {
  local phase="${1:-iter}" root branch lock
  root="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "[git] $DIR: not a git repo — skip"; return 0; }
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"; branch="${branch:-main}"
  # Lock MUST live on a LOCAL fs (/tmp is overlay), NOT on the CephFS repo: CephFS
  # distributed flock can wedge in the kernel (ceph_lock_wait_for_completion) and stall
  # every loop's commit indefinitely. All loops share this pod, so /tmp is shared among them.
  lock="/tmp/marsv-git.lock"

  # -w 30: never block forever. If the lock can't be acquired in 30s, flock exits non-zero,
  # git_sync returns non-zero, the caller's `|| true` swallows it, and we retry next iteration.
  flock -w 30 "$lock" bash -c '
    phase="$1"; branch="$2"; DIR="$3"; SSH_SETUP="$4"
    # stage ONLY this direction (cwd = the dir); never sibling dirs / shared root files.
    git add -A -- . >/dev/null 2>&1 || true

    if git diff --cached --quiet; then
      echo "[git] $DIR: nothing to commit ($phase)"
    elif git commit -q -m "[$DIR] autoloop $phase $(date "+%F %T")"; then
      echo "[git] $DIR: committed ($phase)"
    else
      echo "[git] $DIR: commit failed — skip push"; exit 0
    fi

    push() {
      # fold in siblings other loops pushed. NO --autostash: never stash uncommitted
      # work (that once lost edits) — if a rebase cannot proceed, abort and just retry
      # the push; a blocked/behind push is harmless and re-attempted next iteration.
      git pull --rebase origin "$branch" >/dev/null 2>&1 || git rebase --abort >/dev/null 2>&1 || true
      git push origin "HEAD:$branch" 2>&1
    }
    fail_re="permission denied|host key verification|could not read from remote|authenticity of host|publickey|connection timed out|authentication failed|could not read username"
    out="$(push)"; echo "$out"
    if printf "%s" "$out" | grep -qiE "$fail_re"; then
      if [ -x "$SSH_SETUP" ]; then
        echo "[git] $DIR: push failed on ssh/auth -> running $SSH_SETUP"
        bash "$SSH_SETUP" || true
        out="$(push)"; echo "$out"
      fi
      if printf "%s" "$out" | grep -qiE "$fail_re"; then
        echo "[git] $DIR: push failing (no push credential in this env) — committed LOCALLY, will retry next iteration."
      fi
    fi
  ' _ "$phase" "$branch" "$DIR" "$SSH_SETUP"
}

# Crash recovery: a completed feedback-only relaunch must stop before any worker call.
python3 "$WORKFLOW_ABS" feedback-only-check . >/dev/null || exit 1

while [ "$(date +%s)" -lt "$END" ] && [ ! -f STOP ]; do
  if [ "$(python3 "$WORKFLOW_ABS" feedback-only-check .)" = "RESTORED" ]; then
    echo "[gate] all feedback was already addressed; restored STOP before worker invocation."
    break
  fi
  REMAIN=$(( (END - $(date +%s)) / 60 ))
  echo "[run.sh] $(date '+%F %T')  ~${REMAIN} min left  -----------------------------"

  python3 "$WORKFLOW_ABS" prepare . >/dev/null || break
  ACTIVE_MANIFEST="$(python3 "$WORKFLOW_ABS" active .)"
  if [ -n "$ACTIVE_MANIFEST" ]; then
    python3 "$WORKFLOW_ABS" guard-source "$ACTIVE_MANIFEST" || break
    python3 "$WORKFLOW_ABS" seal "$ACTIVE_MANIFEST" >/dev/null || break
  fi

  COMPACT_CONTEXT="$(python3 "$WORKFLOW_ABS" context .)" || break
  WORKER_RULES="$(python3 "$WORKFLOW_ABS" worker-rules)" || break
  WORK_PROMPT="${WORKER_RULES}

Shared-resource limits: 1 of ${N_AGENTS} agents on GPU '${GPU_NAME}' (${GPU_VRAM_GB} GB); use torch.cuda.set_per_process_memory_fraction(${VRAM_FRACTION}) (~${VRAM_PER_AGENT} GB), keep RAM under ~${RAM_PER_AGENT} GB, torch.set_num_threads(${CPU_THREADS}), and halve batch size on OOM. ${REMAIN} minutes remain. If <= ${FINALIZE_MIN} and there is no active feedback, finalize only. Persist all state before finishing.

${COMPACT_CONTEXT}"

  claude -p "$WORK_PROMPT" \
    --append-system-prompt-file "$WRITING_ABS" \
    --model "$MODEL" \
    --dangerously-skip-permissions \
    2>&1 | tee -a session.log

  if [ -z "$ACTIVE_MANIFEST" ] && [ -f STOP ]; then
    if ! python3 "$WORKFLOW_ABS" check-budgets . >/dev/null; then
      PREMATURE_STOP=".tasks/STOP.over-budget.$(date +%s)"
      mkdir -p .tasks
      mv STOP "$PREMATURE_STOP"
      echo "[gate] moved STOP to $PREMATURE_STOP; report budget must pass before completion."
    fi
  fi

  if [ -n "$ACTIVE_MANIFEST" ]; then
    # A worker cannot bypass the gate by creating STOP or renaming feedback itself.
    if [ -f STOP ]; then
      PREMATURE_STOP=".tasks/STOP.premature.$(date +%s)"
      mv STOP "$PREMATURE_STOP"
      echo "[gate] moved premature STOP to $PREMATURE_STOP; feedback is not approved."
    fi
    python3 "$WORKFLOW_ABS" guard-source "$ACTIVE_MANIFEST" || break
    python3 "$WORKFLOW_ABS" validate "$ACTIVE_MANIFEST" >/dev/null || {
      echo "[gate] invalid task manifest; feedback remains unaddressed."
      break
    }
    TASK_STATE="$(python3 "$WORKFLOW_ABS" state "$ACTIVE_MANIFEST")"
    if [ "$TASK_STATE" = "blocked" ]; then
      echo "[gate] task is blocked on a material ambiguity; feedback remains unaddressed."
      break
    fi
    if [ "$TASK_STATE" = "review_pending" ]; then
      FORMAT_PASSED=yes
      mapfile -t FORMAT_FILES < <(python3 "$WORKFLOW_ABS" format-files "$ACTIVE_MANIFEST")
      if [ "${#FORMAT_FILES[@]}" -eq 0 ] || ! python3 "$CHECK_MD_ABS" "${FORMAT_FILES[@]}"; then
        FORMAT_PASSED=no
      fi
      REPORT_FILES=(REPORT*.md)
      if [ -e "${REPORT_FILES[0]}" ]; then
        RENDER_FILES=("${REPORT_FILES[@]}")
        [ -f RESULTS.md ] && RENDER_FILES+=(RESULTS.md)
        python3 "$CHECK_RENDER_ABS" "${RENDER_FILES[@]}" || FORMAT_PASSED=no
      fi

      REVIEW_DIR="$(mktemp -d /tmp/marsv-content-review.XXXXXX)" || break
      if python3 "$WORKFLOW_ABS" make-review-bundle "$ACTIVE_MANIFEST" "$REVIEW_DIR"; then
        REVIEW_SYSTEM="$(python3 "$WORKFLOW_ABS" review-rules)"
        REVIEW_PROMPT="$(<"$REVIEW_DIR/review_prompt.txt")"
        REVIEW_SCHEMA='{"type":"object","properties":{"pass":{"type":"boolean"},"inspected_outputs":{"type":"array","items":{"type":"string"}},"failures":{"type":"array","items":{"type":"string"}},"summary":{"type":"string"}},"required":["pass","inspected_outputs","failures","summary"],"additionalProperties":false}'
        if (cd "$REVIEW_DIR" && claude -p "$REVIEW_PROMPT" \
              --safe-mode --tools Read --system-prompt "$REVIEW_SYSTEM" \
              --json-schema "$REVIEW_SCHEMA" --output-format json \
              --model "$MODEL" --dangerously-skip-permissions) >"$REVIEW_DIR/review.json"; then
          if python3 "$WORKFLOW_ABS" record-review "$ACTIVE_MANIFEST" "$REVIEW_DIR/review.json" \
              --format-passed "$FORMAT_PASSED"; then
            if [ "$(python3 "$WORKFLOW_ABS" feedback-only-check .)" = "RESTORED" ]; then
              echo "[gate] final feedback approved; restored STOP and ended feedback-only relaunch."
            fi
          fi
        else
          echo "[gate] reviewer failed to run; feedback remains unaddressed."
        fi
      fi
      rm -rf "$REVIEW_DIR"
    fi
  fi

  git_sync iter || true          # checkpoint this iteration's investigation to GitHub
  sleep 2
done

git_sync final || true           # push the finished/deadline state (STOP + REPORT already written)

if [ -f STOP ]; then
  echo "[run.sh] STOP file present — goal reached. Ended $(date '+%F %T')."
else
  echo "[run.sh] deadline hit — ended $(date '+%F %T')."
fi
