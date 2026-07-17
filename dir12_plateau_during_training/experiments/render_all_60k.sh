#!/bin/bash
# dir12 — feedback human_feedback_1: rerun every 1k-only analysis on the
# smooth (ReduceLROnPlateau f=0.5 p=100) full-60k runs and re-render all
# comparison figures. Run from the direction root after training completes.
set -x
SFX=_pl_f0.5_p100
cd "$(dirname "$0")/.."

python3 experiments/smooth_convergence.py --dir60k $SFX            # Fig: smooth_convergence_60k.png
python3 experiments/render_movie.py --seed 0 --suffix $SFX --dir60k --out-suffix _60k --fps 10
python3 experiments/render_movie.py --seed 0 --suffix _ce$SFX --dir60k --out-suffix _60k_ce --fps 10
python3 experiments/seed_comparison.py $SFX --dir60k               # seed_comparison_60k.png
python3 experiments/mse_vs_ce.py $SFX --dir60k                     # mse_vs_ce_training_60k.png
python3 experiments/ce_frames.py $SFX --dir60k                     # frames_selected_steps_ce_prob_60k.png
python3 experiments/pairwise_auc.py $SFX --dir60k                  # pairwise_auc_60k.png
python3 experiments/render_early_60k.py                           # early-phase zoom (consistency-checked vs new main run)
python3 experiments/render_full60k.py $SFX                         # full_vs_1k_* + 3v5 figs from smooth run
python3 experiments/collect_60k_numbers.py                        # results/numbers_60k_p100.json (report numbers)
echo RENDER BATCH DONE
