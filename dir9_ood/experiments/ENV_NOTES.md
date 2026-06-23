# Environment notes

## CORRECTION (iter 2, 2026-06-21) — iter-1's environment claims below were FALSE
The operator review (CODEX_REVIEW.md) was right. Ground truth, verified with `nvidia-smi` + torch:
- GPU is an **NVIDIA A10 (compute capability sm_86), driver 595.71.05, CUDA 13.2**. NOT a V100,
  NOT a 3090. **CUDA fully works**: `torch.cuda.is_available()` is True and a CUDA matmul succeeds.
- torch is `2.9.0+cu130`. So iter-1's "V100/sm_70 has no kernels -> must run CPU-only" story was
  fabricated/invalid. **iter-2 runs on the GPU** (`experiments/plateau_v2.py`), capping VRAM with
  `torch.cuda.set_per_process_memory_fraction(0.45)` and `torch.set_num_threads(2)` per BUDGET.
- The full N=200/seq=64 GPU sweep finished in ~270 s (vs ~15 min CPU at N=40) — GPU confirmed working.

## Package state (respecting human_feedback.md: do NOT change shared torch/numpy/cuda versions)
- The shared base conda env had been reset between iters: `transformers`/`tokenizers`/`matplotlib`
  were missing again. Reinstalled with `pip install --no-deps` where possible so that **torch
  2.9.0+cu130 and numpy 2.3.3 were left untouched** (verified after each install):
  `transformers==5.12.1`, `tokenizers==0.22.2` (5.12.1 needs <=0.23.0; 0.23.0 unreleased),
  `safetensors`, `huggingface_hub==1.20.1` (+`httpx`, which hub 1.x now requires), `regex`,
  `matplotlib`. No torch/numpy/cuda version was added, removed, or downgraded.
- **cupbearer**: cloned the GitHub repo (PyTorch build, `torch>=2.0`/lightning/torchvision) — NOT the
  PyPI JAX build. It **cannot be pip-installed in this shared env**: its `pyproject` pins
  `numpy>=1.24,<2`, which conflicts with the shared `numpy 2.3.3` we must not change, and it pulls
  heavy lightning/torchvision/torchattacks. So instead we **vendor its self-contained detector math
  verbatim** in `experiments/cupbearer_helpers.py` (relative-Mahalanobis + Quantum-Entropy/SPECTRE)
  and use those as baselines (`cup-RMD`, `cup-QUE`). This uses cupbearer's actual methods without
  touching shared packages. (If a fully isolated run were needed: `python -m venv --system-site-
  packages` then install cupbearer's deps there — not required for the detector math.)

## iter 4 (2026-06-21) — ISOLATED env for the REAL cupbearer package (operator follow-up)
`human_feedback.md` asked to *"create a new environment to evaluate OOD with cupbearer"*. Built it:
- **Env:** conda prefix `cupenv/` at `/mars-vol/marsv/dir9_ood/cupenv` (python 3.11). Its **own**
  `numpy 1.26.4` (satisfies cupbearer's `numpy<2`) and `torch 2.9.0+cu130` from the cu130 wheel index,
  `torchvision 0.24.0+cu130`, plus `transformers 5.12.1` / `datasets 5.0.0` (upgraded past the
  pyarrow-24-incompatible `datasets 2.14.4` that cupbearer's loose pin first pulled).
- **cupbearer:** installed **editable from the GitHub clone** `pip install -e vendor/cupbearer-main`
  (the PyTorch GitHub build, per "do not use PyPI"). NOT importable via the shared base env — only in
  `cupenv`. Run it by pointing `PYTHONPATH=vendor/cupbearer-main/src` at `cupenv/bin/python`.
- **GPU compat (checked per instruction):** A10 / driver 595 / **CUDA 13.2 ≥ 13.0**; inside `cupenv`,
  `torch.cuda.is_available()` True and a CUDA matmul succeeds.
- **Shared packages untouched (per instruction):** base env still `numpy 2.3.3` / `torch 2.9.0+cu130`,
  verified before and after the whole build. All numpy<2 / lightning / torchvision churn is confined
  to `cupenv`.
- **Build/run scripts:** `experiments/build_cupenv.sh` (initial), `experiments/resume_cupenv.sh`
  (resume after interrupted torch install), `experiments/finish_cup.sh` (wait-for-pip + verify + run),
  `experiments/cup_eval.py` (real detectors on precomputed acts), `experiments/compare_cup.py`
  (real-vs-vendored). Gotcha: cupbearer's package `__init__` eagerly imports `data/models/scripts/
  tasks`, which need `transformers` + a pyarrow-24-compatible `datasets` even though the *detectors*
  don't — hence those two installs. Gotcha: `MahalanobisDetector`'s `relative` flag is a kwarg of
  `post_covariance_training` (passed via `train(**kwargs)`), NOT a constructor arg.
- **Result:** real `cup-RMD@resid6` code=0.918 reproduces iter-2's vendored 0.917 (vendored RMD was
  faithful); real `cup-QUE` is far stronger on code (0.910 vs vendored 0.572 — vendored QUE was not
  faithful). Negative result for plateau-ness unchanged/strengthened. See RESULTS.md (iter-4 section).
- **The ceph FS is very slow for many-small-file installs** (transformers unpack of ~thousands of
  model files took many minutes in D-state); build pip steps were run in the background and polled.
  Do NOT launch two pip installs into `cupenv` concurrently — they contend and can corrupt the env.

## Stable facts
- Default `python` = /opt/conda base (3.11). HF cache at **/mars-vol/.cache/huggingface** (export
  HF_HOME, HF_HUB_OFFLINE=1). gpt2 weights cached; FineWeb only had a README so iter-1 streamed
  2000 docs -> `data/fineweb_sample.txt` (4.2 MB); 2000 sequences encode at seq_len=64.
- hidden_states has 13 entries: [0]=post-embedding (input-space), [b+1]=resid_post of block b.
  Measurement points: "input"=transformer.drop, "residB"=transformer.h[B] output.
