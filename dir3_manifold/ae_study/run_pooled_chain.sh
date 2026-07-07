#!/bin/bash
# Complete the controlled experiment: after the inject_massive sweep finishes,
# collect Qwen L2 all-token-pooled activations and run the same AE k-sweep.
cd /mars-vol/marsv/dir3_manifold/ae_study
echo "=== waiting for inject sweep to finish ===" >> chain.log
while pgrep -f "ae_sweep_qwen.py --layer 2 .* --inject_massive" >/dev/null 2>&1; do sleep 20; done
echo "=== inject done ($(date)); collecting pooled L2 ===" >> chain.log
python -u collect_qwen_pooled.py > collect_pooled.log 2>&1
echo "=== pooled collected ($(date)); sweeping pooled L2 ===" >> chain.log
python -u ae_sweep_qwen.py --layer 2 --ks 5 10 15 20 25 30 --n_steps 4000 \
    --acts cache/acts_qwen_L2_pooled.npy --tag _pooled > sweep_L2_pooled.log 2>&1
echo "=== pooled sweep done ($(date)) ===" >> chain.log
