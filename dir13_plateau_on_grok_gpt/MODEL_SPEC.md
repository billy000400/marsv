# MODEL_SPEC.md — the 12-layer character-level Shakespeare GPT

Every field is tagged **[confirmed from source]** (paper text / official repo) or **[reconstruction
choice]** (chosen by us because the source does not specify it). This direction analyzes only a
*late-training* checkpoint of this model.

## Source-audit findings (S1, bounded to ~30 min)

- Paper: Humayun, Balestriero, Baraniuk, *Deep Networks Always Grok and Here is Why*, ICML 2024
  (arXiv:2402.15555). Figure 9 shows a **12-layer, 12-head causal GPT trained for next-character
  prediction on Shakespeare text with GeLU MLPs**. **[confirmed from source]**
- Official repository `AhmedImtiazPrio/grok-adversarial` (branch `main`, git tree audited via the
  GitHub API on 2026-07-15) contains training code **only** for `train_mlp_mnist.py` and
  `train_resnet18_cifar10.py`; `models.py`/`configs.py` define an MLP and a ResNet18 and **no GPT /
  transformer / Shakespeare code or checkpoint**. **[confirmed absent]**
- Conclusion: the paper's exact GPT training code, config, and checkpoint are **not publicly
  released**. We therefore test a **minimal faithful reconstruction** and, per PLAN success-criterion
  (3), keep all conclusions explicitly about that reconstruction, not the paper's exact checkpoint.

## Architecture

| Field | Value | Tag |
|---|---|---|
| Model family | decoder-only causal GPT (nanoGPT-style) | [confirmed from source] |
| Number of transformer blocks | **12** | [confirmed from source] |
| Attention heads per block | **12** | [confirmed from source] |
| MLP nonlinearity | **GeLU** | [confirmed from source] |
| Task | next-character prediction, Shakespeare text | [confirmed from source] |
| Embedding width `d_model` | **240** (head dim 20) | [reconstruction choice] |
| MLP hidden width | `4 * d_model = 960` | [reconstruction choice] |
| Context length | **128** tokens | [reconstruction choice] |
| Positional embedding | learned absolute | [reconstruction choice] |
| LayerNorm | pre-norm (LN before attn/MLP), final LN before head | [reconstruction choice] |
| Weight tying (embed ↔ head) | yes | [reconstruction choice] |
| Dropout | 0.2 | [reconstruction choice] |
| Vocabulary | character-level, size = #unique chars in corpus (65) | [reconstruction choice] |

Rationale for reconstruction choices: `d_model` must be divisible by 12 heads; 240 keeps the 12×12
model small enough for the shared 7.2 GB VRAM budget while retaining capacity to learn Shakespeare.
Context 128 (vs a common 256) halves attention cost under the time budget. All other choices follow
the widely used nanoGPT char-level recipe, the closest standard reference for this exact task.

## Data

| Field | Value | Tag |
|---|---|---|
| Corpus | Tiny Shakespeare (`input.txt`, 1,115,394 chars) | [reconstruction choice] |
| Source | karpathy/char-rnn tinyshakespeare mirror | [reconstruction choice] |
| Train/val split | first 90% / last 10% (contiguous) | [reconstruction choice] |
| Tokenization | per-character integer ids | [reconstruction choice] |

The paper says "Shakespeare text"; the canonical char-level Shakespeare corpus is Tiny Shakespeare,
so we use it and mark it a reconstruction choice.

## Optimizer / schedule

| Field | Value | Tag |
|---|---|---|
| Optimizer | AdamW (betas 0.9/0.99, wd 0.1) | [reconstruction choice] |
| Peak LR | 1e-3 | [reconstruction choice] |
| Schedule | linear warmup (100 steps) → cosine decay | [reconstruction choice] |
| Batch size | 48 sequences × 128 tokens | [reconstruction choice] |
| Precision | fp32 (numerically stable for the assay) | [reconstruction choice] |
| Training steps | fixed wall-clock budget under shared GPU; log-spaced checkpoints saved | [reconstruction choice] |

## Provenance recorded at train time (see `results/train_meta.json`)

Corpus SHA-256, vocab, model seed, data seed, package versions, device, and the git commit are
written by the training script so the checkpoint is reproducible.
