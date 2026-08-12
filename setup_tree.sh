#!/usr/bin/env bash
# setup_tree.sh — validate the git-tracked canonical workflow files.
# The old embedded templates duplicated run.sh and could resurrect stale prompts.
# Restore missing files from git instead of synthesizing a second source of truth.
# Usage: bash setup_tree.sh [PROJECT_ROOT]
set -euo pipefail
ROOT="${1:-.}"
[ -d "$ROOT" ] || { echo "[setup] project root '$ROOT' not found"; exit 1; }
cd "$ROOT"

required=(run.sh new_direction.sh launch.sh relaunch.sh workflow.py WRITING.md CLAUDE.md BUDGET.md check_md.py)
missing=0
for file in "${required[@]}"; do
  if [ ! -f "$file" ]; then
    echo "[setup] MISSING: $file — restore the tracked canonical file from git."
    missing=1
  fi
done
[ "$missing" -eq 0 ] || exit 1

chmod +x run.sh new_direction.sh launch.sh relaunch.sh workflow.py
bash -n run.sh new_direction.sh launch.sh relaunch.sh
python3 -m py_compile workflow.py check_md.py
echo "[setup] canonical workflow files are present and syntactically valid."
echo "[setup] add a direction with: ./new_direction.sh <dir_name> \"Title\""
