# Block-0 resid_post rerun queue: validate (S2) then sweep (S3) each model, one at a time.
# 8B weights are staged in /tmp (not in the shared HF cache), one model at a time.
cd /workspacen/marsv_agent_haoyang/dir27_interp_dist_vs_plateau_width/experiments
export DIR27_SITE=block0_resid_post
G=../results/block0_resid_post/gen
while pgrep -f "python -u s2_sweep.py" > /dev/null; do sleep 10; done
for k in gpt2-xl pythia-1.4b; do
  HF_HUB_OFFLINE=1 python -u gen_sweep.py $k > $G/sweep_$k.log 2>&1
done
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/tmp/hf_hub python -u b0_validate.py qwen3-8b-base > $G/validate_qwen3-8b-base.log 2>&1
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/tmp/hf_hub python -u gen_sweep.py qwen3-8b-base 128 > $G/sweep_qwen3-8b-base.log 2>&1
rm -rf /tmp/hf_hub/models--Qwen--Qwen3-8B-Base
HF_HUB_DISABLE_XET=1 HF_HUB_CACHE=/tmp/hf_hub python /tmp/dir27_tools/dl.py unsloth/Meta-Llama-3.1-8B > $G/dl_llama.log 2>&1
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/tmp/hf_hub python -u b0_validate.py llama-3.1-8b > $G/validate_llama-3.1-8b.log 2>&1
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/tmp/hf_hub python -u gen_sweep.py llama-3.1-8b 128 > $G/sweep_llama-3.1-8b.log 2>&1
rm -rf /tmp/hf_hub/models--unsloth--Meta-Llama-3.1-8B
echo done > $G/queue.done
