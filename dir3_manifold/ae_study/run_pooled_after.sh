#!/bin/bash
# Wait for the wide-injection sweep (PID passed as $1) to finish, then run the
# factor-3 pooled control: collect all-token pooled Qwen L2 activations and run
# the same wide-k sweep on them. Serialized so only one GPU job runs at a time.
set -e
cd /mars-vol/marsv/dir3_manifold/ae_study
WPID=$1
echo "=== waiting for wide_inject PID $WPID ($(date)) ==="
while kill -0 "$WPID" 2>/dev/null; do sleep 10; done
echo "=== wide_inject done; collecting pooled activations ($(date)) ==="
python3 collect_qwen_pooled.py
echo "=== pooled wide sweep ($(date)) ==="
python3 ae_sweep_qwen.py --layer 2 --ks 1 2 4 8 16 32 64 --n_steps 2000 \
    --acts cache/acts_qwen_L2_pooled.npy --tag _pooled_wide
echo "=== pooled chain done ($(date)) ==="
