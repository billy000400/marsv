#!/bin/bash
# Rerun everything from the pinned sources. Run from this directory: bash run_all.sh
set -e
python3 download.py          # Qwen model + transcoder weights (layers missing from the cache are streamed later)
python3 s0_validate.py 10    # one-layer hook / transcoder check -> results/s0_validation_layer10.json
python3 s1_build_filter.py   # groups, backgrounds, correctness filter -> samples.jsonl, sample_counts.csv, run_config.json
python3 s2_extract.py        # sparse activations for all 28 layers + early-position baseline
python3 s3_analyze.py        # rankings, sets, overlap, early-position table, figures
python3 s5_suppress.py       # small suppression check
