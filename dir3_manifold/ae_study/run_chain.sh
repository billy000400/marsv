#!/bin/bash
cd /mars-vol/marsv/dir3_manifold/ae_study
# wait for the running L10 sweep to finish
while pgrep -f "ae_sweep_qwen.py --layer 10" >/dev/null 2>&1; do sleep 15; done
echo "=== L10 done, starting inject_massive controlled experiment ===" >> chain.log
python -u ae_sweep_qwen.py --layer 2 --ks 5 10 15 20 25 30 --n_steps 4000 \
    --inject_massive 0.90 --tag _inject > sweep_L2_inject.log 2>&1
echo "=== inject_massive done ===" >> chain.log
