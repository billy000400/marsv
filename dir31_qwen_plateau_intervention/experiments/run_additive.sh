#!/bin/bash
# Stage S4 (operator addendum, human_feedback_additive_berlin.md): additive Berlin edit at the country token.
# Rerun from this directory: bash run_additive.sh   (about 5 minutes; streams 21 transcoder layers through /tmp)
# Needs dir31's results/probe.npz, results/slices/, results/s2_slices.csv and dir30's results/ (feature_sets.json,
# samples.jsonl, acts/).
# frozen_edit.json is never rewritten: on a rerun the freeze step only checks that the selection still matches it.
set -e
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf_home/} MPLBACKEND=Agg
mkdir -p ../results/additive ../plots/additive
python3 additive_s1_candidates.py > ../results/additive/s1_candidates.log 2>&1  # S4.1 candidates + reference amplitudes
python3 additive_s2_screen.py > ../results/additive/s2_screen.log 2>&1          # S4.2 tuning-prompt screen, all checks
python3 additive_s3_freeze.py > ../results/additive/s3_freeze.log 2>&1          # tuning-only selection -> preregistered_gate.json, frozen_edit.json
python3 additive_s4_tuning_example.py > ../results/additive/s4_tuning_example.log 2>&1  # tuning prompt: completion, probabilities
python3 additive_s5_reserved_eval.py > ../results/additive/s5_reserved_eval.log 2>&1    # S4.3 frozen edit on the reserved prompts -> pilot_verdict.json
python3 additive_plots.py                                                       # figures
# S4.4 (41-point path) is allowed only if pilot_verdict.json has pilot_passes == true. On 2026-10-11 the pilot did
# not pass, so the path was not run and no code exists for it.
python3 -c "import json; v = json.load(open('../results/additive/pilot_verdict.json')); print(v['criteria_met_on_both_prompts'], 'pilot_passes =', v['pilot_passes'], '|', v['plateau_test'])"
