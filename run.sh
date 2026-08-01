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

while [ "$(date +%s)" -lt "$END" ] && [ ! -f STOP ]; do
  REMAIN=$(( (END - $(date +%s)) / 60 ))
  echo "[run.sh] $(date '+%F %T')  ~${REMAIN} min left  -----------------------------"

  claude -p "You are mid-project and your working memory RESETS every iteration. \
FIRST read CLAUDE.md (operator rules, at ${RULES_ABS}) and BUDGET.md (at ${BUDGET_ABS}), then \
PLAN.md, JOURNAL.md, RESULTS.md, and CHANGELOG.md in full. OBEY every rule in CLAUDE.md. \
FEEDBACK FIRST (CLAUDE.md Part C): before advancing the plan, list this direction for files matching \
'human_feedback*.md' or '*REVIEW*' that do NOT end in '.addressed.md'. If any exist, addressing them IS \
this iteration: read each in full, do every ask (run the experiment / add the plot / add the metric / \
answer the question in RESULTS.md+REPORT.md), then 'mv' the file to end in '.addressed.md' (never delete/edit \
it), and log it in CHANGELOG.md + JOURNAL.md. \
KEY RULES: RESULTS.md and REPORT.md are FINAL, presentable deliverables — read them, then overwrite \
to current-best ONLY (no version history, no 'changed after review', no weaker/superseded variant of \
an experiment when a stronger one exists). Put ALL change history in CHANGELOG.md (append-only). \
SHARED-RESOURCE LIMITS (this run): you are 1 of ${N_AGENTS} agents on GPU '${GPU_NAME}' (${GPU_VRAM_GB} GB); \
call torch.cuda.set_per_process_memory_fraction(${VRAM_FRACTION}) (~${VRAM_PER_AGENT} GB), keep RAM under \
~${RAM_PER_AGENT} GB (memmap caches), torch.set_num_threads(${CPU_THREADS}), HALVE batch on OOM. \
You have ${REMAIN} minutes of wall-clock left. \
If that number is <= ${FINALIZE_MIN}: do ONLY finalization — refresh RESULTS.md (current-best only) and \
write a clean presentable REPORT.md per CLAUDE.md (Summary -> Methods -> Results -> Conclusion; the \
Methods section MUST give Data/Model/Layer, and DEFINE every metric and baseline with rendered \$\$LaTeX\$\$ \
equations; embed plots/ figures; current-best numbers only). Append a final CHANGELOG.md entry, then \
create an empty STOP file and stop — BUT ONLY IF no unaddressed 'human_feedback*.md'/'*REVIEW*' file remains \
(never write STOP while feedback is unaddressed; a STOP'd direction stops looping and ignores it). \
Otherwise do ONE focused iteration: advance the plan by the smallest useful step, write/modify code under \
experiments/, RUN it, then CURATE RESULTS.md to current-best (read it, overwrite clean — no history) and \
save a PNG for every quantitative result into plots/ (plt.savefig + plt.close, NEVER plt.show; headless Agg) \
and EMBED each as a rendered Markdown image '![caption](plots/foo.png)' in BOTH RESULTS.md AND REPORT.md \
(a bare '(plots/foo.png)' path in prose does NOT render — it must be an '![...](...)' image; embed REPORT.md \
figures every iteration, do not defer them to finalization). APPEND to CHANGELOG.md what changed in the deliverables this iteration \
(old -> new numbers if a result was superseded). Then append to JOURNAL.md (what you did, learned, next step) \
and update PLAN.md 'Current status'/'Next step'/checkboxes. End the JOURNAL entry with the 'On track?' line. \
Persist ALL state to disk before you finish; assume nothing carries over." \
    --append-system-prompt "Obey CLAUDE.md every iteration. File roles are STRICT: RESULTS.md and REPORT.md are curated, presentable, current-best — read then overwrite clean, NEVER keep history or superseded/weaker results in them. CHANGELOG.md and JOURNAL.md are append-only history. REPORT.md must have a Methods section defining every metric and baseline with \$\$LaTeX\$\$ equations plus the data/model/layer used. Visualize every reported result: PNGs in plots/ (savefig+close, never show) EMBEDDED as rendered '![caption](plots/x.png)' images in BOTH RESULTS.md and REPORT.md every iteration — a bare path in prose does not render. OPERATOR FEEDBACK (CLAUDE.md Part C): each iteration, before other work, address any 'human_feedback*.md'/'*REVIEW*' file lacking a '.addressed.md' suffix, then rename it to '.addressed.md'; NEVER write STOP while any such file is unaddressed. Read before write. Persist every iteration. Prefer small verifiable steps; on a broken experiment debug minimally or fall back per PLAN.md." \
    --model "$MODEL" \
    --dangerously-skip-permissions \
    2>&1 | tee -a session.log

  git_sync iter || true          # checkpoint this iteration's investigation to GitHub
  sleep 2
done

git_sync final || true           # push the finished/deadline state (STOP + REPORT already written)

if [ -f STOP ]; then
  echo "[run.sh] STOP file present — goal reached. Ended $(date '+%F %T')."
else
  echo "[run.sh] deadline hit — ended $(date '+%F %T')."
fi
