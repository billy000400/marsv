#!/bin/bash
# Rerun the whole study from this directory: bash run_all.sh   (about 10 minutes; needs the dir30 samples.jsonl)
set -e
cd "$(dirname "$0")"
export MPLBACKEND=Agg
python3 s1_plateau.py   # baseline answers, 41-point path, country readout
python3 s2_slices.py    # compact transcoder slices for all 28 layers (downloads each dictionary once)
python3 s2_scan.py      # gradient ranking + intervention scan on the tuning pair
python3 s2_select.py    # apply the pre-declared selection rule, freeze the edit
python3 s3_path.py      # reserved-prompt evaluation, context prompts, path under each edit
python3 s4_plots.py     # figures
