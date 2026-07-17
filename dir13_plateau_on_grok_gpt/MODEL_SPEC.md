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

---

# Addendum (2026-07-17, reopened plan): fresh Grokking-horizon runs & Figure-9 measurements

## Fresh runs (`train_grok.py`; configs/grok_char.yaml, configs/grok_bpe.yaml)

| Field | Value | Tag |
|---|---|---|
| Optimizer | **Adam, zero weight decay** (betas 0.9/0.99) | [confirmed from source] (repo `configs.py`: `optimizer='adam'`, `weight_decay=0.`, `lr=1e-3` for the released trainings; PLAN source-lock) |
| Peak LR | 1e-3 | [confirmed from source] (repo default) |
| Schedule | warmup 100 → cosine 1e-3→1e-4 designed for the full horizon | [reconstruction choice] (repo vision config uses `lr_schedule_flag=False`; cosine kept for LM stability) |
| Horizon | **30,000 steps** (PLAN asks ~1e5; budget-limited per ../BUDGET.md) | [reconstruction choice — DEVIATION, recorded prominently] |
| BPE tokenizer | GPT-2 byte-level BPE, vocab 50257 | per PLAN Exp 2 |
| BPE batch | 48×128 via 4×12 gradient accumulation (OOM workaround; token exposure matched to char run) | [reconstruction choice] |
| Dropout | 0.2 (kept identical to pilot run) | [reconstruction choice] |

## Figure-9 measurements (`fig9.py`), source-locked to repo `local_complexity.py` / `attacks.py` / `configs.py`

| Field | Value | Tag |
|---|---|---|
| LC sample count | 1024 train / 1024 test / 1024 random points | [confirmed from source] (`approx_n=1024`; PLAN) |
| LC neighborhood | P=25 vertices: 12 ± antipodal pairs on the radius-r L2 sphere + centroid | [confirmed from source] (`get_hull_around_samples` ± pairs; `inc_centroid`; P=25 per PLAN) |
| LC radius | r = 0.005 | [confirmed from source] (`r_frame=0.005`) |
| LC statistic | per layer: #units whose activation **sign** is non-constant across the P vertices (`get_intersection_from_activation_batched`); we sum over the 12 GeLU pre-activation layers | [confirmed from source]; GPT adaptation below |
| CIs | 99% (2.576×SEM over the 1024 points) | [confirmed from source] (PLAN) |
| PGD | untargeted L∞, eps=0.03, 10 iterations, alpha=eps/4, random start, iterates clamped to the clean batch min/max | eps [confirmed] (Fig. 9); iters=10 [confirmed] (`atk_itrs=10`); alpha ratio & clamping [reconstruction choice] (their `alpha=2/255` for `eps=8/255`; `dmin/dmax` = data range) |

GPT adaptation choices (the repo has no GPT code — all **[reconstruction choice]**, applied
identically to char and BPE models without post-hoc tuning):
- Perturbations (LC hull and PGD) act on the **summed token+position embedding** of the full
  128-token window ("token-embedding space" per Figure 9's caption).
- LC signs are taken at the 12 MLP **pre-GeLU** activations (the model's only elementwise
  nonlinearities), flattened over positions × hidden units, exactly as the repo flattens
  activations before sign comparison.
- "Random points" = uniform random token sequences (the vision code uses random inputs).
- Accuracy (clean and adversarial) = mean next-token accuracy over all 128 positions of 1024
  held-out windows.
