# Deep autoencoders on LLM activations

Self-contained slice of a larger research repo: trains vanilla deep autoencoders
(MLP encoder/decoder, k-dim bottleneck, MSE loss, no sparsity) on residual-stream
activations, plus a TopK-SAE baseline for comparison.

## Layout
- `src/autoencoders.py` — model definitions: `DeepAutoencoder`, `FlexAutoencoder`
  (configurable depth/activation/norm/residual), `Autoencoder`, `TopKSAE`, loaders.
- `src/model.py` — `load_model` (HuggingFace causal LMs).
- `src/data.py` — `load_fineweb_fixed_length` (pulls FineWeb via `datasets`).
- `scripts/autoencoder/`
  - `train_autoencoder.py` — main entrypoint (deep AE, MSE).
  - `hillclimb_autoencoder.py` — architecture/config hill-climb search.
  - `sweep_autoencoder_bottleneck.py` — bottleneck-k sweep.
  - `cache_activations.py`, `create_subsets.py` — data prep.
  - `eval_ce_loss.py`, `quick_overfit_check.py` — evaluation.
  - `plot_*.py` — figures.
  - `run_*.sh` — pipeline orchestration.

## Setup
```bash
pip install -r requirements.txt
```

## Run (from this directory — imports assume the bundle root is the cwd)
```bash
python -u scripts/autoencoder/train_autoencoder.py \
    --model Qwen/Qwen3-1.7B --layer 2 --k 10 \
    --output_dir results/qwen3-1.7b_layer2_k10
```
Activations are cached under `--output_dir`; reuse with
`--reuse_cache --cache_path <dir>/cached_activations.pt`.

Note: the `plot_*.py` and `quick_overfit_check.py` scripts have hardcoded
`results/...` paths pointing at outputs from the original runs — adjust them to
wherever your own runs land.
