#!/bin/bash
# Decisive controlled experiment: sweep a WIDE k range (1..64) so the elbow (or
# its absence) is visible within the plotted window, for two conditions on the
# SAME Qwen L2 last-token activations:
#   (A) baseline (isotropic, top1~0.012)   -> expect NO steep-then-plateau knee
#   (B) + injected massive dim (top1=0.90) -> expect a sharp knee at k~1-2
# Then a factor-3 control: all-token pooled L2 (does pooling change anisotropy?).
cd /mars-vol/marsv/dir3_manifold/ae_study
WKS="1 2 4 8 16 32 64"
echo "=== wide chain: waiting for inject [5..30] sweep to finish ($(date)) ===" >> chain.log
while pgrep -f "ae_sweep_qwen.py --layer 2 .* --inject_massive 0.90 --tag _inject" >/dev/null 2>&1; do sleep 20; done

echo "=== wide baseline L2 ($(date)) ===" >> chain.log
python -u ae_sweep_qwen.py --layer 2 --ks $WKS --n_steps 2000 --tag _wide \
    > sweep_L2_wide.log 2>&1

echo "=== wide inject L2 ($(date)) ===" >> chain.log
python -u ae_sweep_qwen.py --layer 2 --ks $WKS --n_steps 2000 --inject_massive 0.90 \
    --tag _wide_inject > sweep_L2_wide_inject.log 2>&1

echo "=== pooled collect ($(date)) ===" >> chain.log
python -u collect_qwen_pooled.py > collect_pooled.log 2>&1

echo "=== pooled wide sweep L2 ($(date)) ===" >> chain.log
python -u ae_sweep_qwen.py --layer 2 --ks $WKS --n_steps 2000 \
    --acts cache/acts_qwen_L2_pooled.npy --tag _pooled_wide > sweep_L2_pooled_wide.log 2>&1

echo "=== wide chain DONE ($(date)) ===" >> chain.log
