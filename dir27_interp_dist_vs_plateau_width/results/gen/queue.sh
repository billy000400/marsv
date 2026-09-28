# Feedback-3 sweep queue for the 8B models. Weights are staged in /tmp (shared volume is at quota),
# one model at a time. (GPT2-XL and Pythia 1.4B were run first from the shared HF cache.)
cd /workspacen/marsv_agent_haoyang/dir27_interp_dist_vs_plateau_width/experiments
G=../results/gen
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/tmp/hf_hub python -u gen_sweep.py qwen3-8b-base 128 > $G/sweep_qwen3-8b-base.log 2>&1
rm -rf /tmp/hf_hub/models--Qwen--Qwen3-8B-Base
HF_HUB_DISABLE_XET=1 HF_HUB_CACHE=/tmp/hf_hub python /tmp/dl.py unsloth/Meta-Llama-3.1-8B > $G/dl_llama.log 2>&1
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/tmp/hf_hub python -u gen_sweep.py llama-3.1-8b 128 > $G/sweep_llama-3.1-8b.log 2>&1
rm -rf /tmp/hf_hub/models--unsloth--Meta-Llama-3.1-8B
