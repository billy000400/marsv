"""Fresh Grokking-horizon training runs (PLAN Exp 1.3 / Exp 2.2).

Trains the 12L/12H GeLU GPT from scratch on Tiny Shakespeare with either the char tokenizer
(--tok char, paper-faithful control) or the GPT-2 byte-level BPE tokenizer (--tok bpe, bridge to
Matthew's exact examples). Paper defaults per PLAN source-lock: Adam, ZERO weight decay.
Horizon is budget-limited (see MODEL_SPEC.md): schedule is designed for the full --steps horizon.
Saves log-spaced + linear checkpoints and full provenance.
"""
import os, sys, json, time, hashlib, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")


def get_batch(data, block, bs, device, g):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=g)
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, block, bs, device, g, iters=20):
    model.eval()
    losses = []
    correct = tot = 0
    for _ in range(iters):
        x, y = get_batch(data, block, bs, device, g)
        logits, loss = model(x, y)
        losses.append(loss.item())
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        tot += y.numel()
    model.train()
    return float(np.mean(losses)), correct / tot


def tokenize_corpus(tok, text):
    if tok == "char":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        data = np.array([stoi[c] for c in text], dtype=np.uint16)
        return data, len(chars), {"stoi": stoi, "itos": {i: c for c, i in stoi.items()}}
    # bpe: GPT-2 byte-level BPE via transformers (cached to /tmp so reruns are cheap)
    cache = "/tmp/tinyshakespeare_gpt2bpe.npy"
    from transformers import GPT2TokenizerFast
    tkz = GPT2TokenizerFast.from_pretrained("gpt2")
    if os.path.exists(cache):
        data = np.load(cache)
    else:
        ids = tkz(text)["input_ids"]
        data = np.array(ids, dtype=np.uint16)
        np.save(cache, data)
    return data, tkz.vocab_size, {"tokenizer": "gpt2", "vocab_size": tkz.vocab_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", choices=["char", "bpe"], required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--max_minutes", type=float, default=80.0)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--accum", type=int, default=1, help="micro-batch splits (keeps tokens/step matched)")
    ap.add_argument("--vram_frac", type=float, default=0.08)
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.vram_frac)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_seed, data_seed = 1337, 42
    torch.manual_seed(model_seed)

    tag = f"grok_{args.tok}"
    ckpt_dir = os.path.join(RES, f"checkpoints_{tag}")
    os.makedirs(ckpt_dir, exist_ok=True)

    raw = open("/tmp/tinyshakespeare.txt", "rb").read()
    corpus_sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    data, vocab_size, tok_meta = tokenize_corpus(args.tok, text)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = GPTConfig(vocab_size=vocab_size, block_size=128, n_layer=12, n_head=12, n_embd=240, dropout=0.2)
    model = GPT(cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.bs * cfg.block_size
    print(f"[{tag}] params={nparams/1e6:.2f}M vocab={vocab_size} corpus_tokens={len(data)} "
          f"train_tokens={n} epoch_every={n/tokens_per_step:.0f} steps", flush=True)

    # PLAN source-lock: Adam, zero weight decay (paper defaults). Schedule designed for the
    # full horizon: 100-step warmup then cosine 1e-3 -> 1e-4 over --steps (reconstruction choice).
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0)
    warmup, total = 100, args.steps

    def lr_at(step):
        if step < warmup:
            return 1e-3 * step / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + np.cos(np.pi * min(1.0, prog)))

    gtr = torch.Generator().manual_seed(data_seed)
    gev = torch.Generator().manual_seed(data_seed + 1)

    # checkpoints: log-spaced (grokking transitions are log-time) + every 2500 linear
    logsp = np.unique(np.round(np.logspace(0, np.log10(total), 24)).astype(int))
    ckpt_steps = sorted(set([0] + list(logsp) + list(range(2500, total + 1, 2500)) + [total]))
    hist = {"step": [], "train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    step = 0
    model.train()
    while step <= total:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_data, cfg.block_size, args.bs, device, gtr)
        opt.zero_grad(set_to_none=True)
        mb = args.bs // args.accum
        loss_val = 0.0
        for k in range(args.accum):
            logits, loss = model(x[k * mb:(k + 1) * mb], y[k * mb:(k + 1) * mb])
            (loss / args.accum).backward()
            loss_val += loss.item() / args.accum
        loss = torch.tensor(loss_val)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 250 == 0 or step == total:
            vl, va = eval_loss(model, val_data, cfg.block_size, args.bs // args.accum, device, gev,
                               iters=20 * args.accum)
            hist["step"].append(step); hist["train_loss"].append(loss.item())
            hist["val_loss"].append(vl); hist["val_acc"].append(va)
            print(f"[{tag}] step {step:6d} lr {lr_at(step):.2e} train {loss.item():.3f} val {vl:.3f} "
                  f"acc {va:.3f} [{(time.time()-t0)/60:.1f}m]", flush=True)

        if step in ckpt_steps:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step, "tok": args.tok},
                       os.path.join(ckpt_dir, f"ckpt_{step:06d}.pt"))

        if (time.time() - t0) / 60 > args.max_minutes:
            print(f"[{tag}] time budget hit at step {step}, stopping", flush=True)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step, "tok": args.tok},
                       os.path.join(ckpt_dir, f"ckpt_{step:06d}.pt"))
            break
        step += 1

    final_step = hist["step"][-1]
    json.dump(hist, open(os.path.join(RES, f"train_hist_{tag}.json"), "w"), indent=2)
    meta = {"corpus_sha256": corpus_sha, "n_chars": len(text), "tokenizer": args.tok,
            "vocab_size": vocab_size, "corpus_tokens": int(len(data)), "train_tokens": int(n),
            "raw_chars_per_token": len(text) / len(data),
            "corpus_epochs": final_step * tokens_per_step / n,
            "model_seed": model_seed, "data_seed": data_seed, "params_millions": nparams / 1e6,
            "cfg": cfg.__dict__, "optimizer": "Adam(lr 1e-3 cosine->1e-4, betas 0.9/0.99, wd 0.0)",
            "planned_steps": total, "final_step": final_step,
            "final_val_loss": hist["val_loss"][-1], "final_val_acc": hist["val_acc"][-1],
            "ckpt_steps": [s for s in ckpt_steps if s <= final_step], "torch": torch.__version__}
    json.dump(meta, open(os.path.join(RES, f"train_meta_{tag}.json"), "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["step"], hist["train_loss"], label="train", alpha=0.7)
    ax[0].plot(hist["step"], hist["val_loss"], label="val", lw=2)
    ax[0].set_xlabel("step"); ax[0].set_ylabel("cross-entropy loss (nats)")
    ax[0].set_xscale("symlog", linthresh=100)
    ax[0].set_title(f"{tag}: loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(hist["step"], hist["val_acc"], color="C2", lw=2)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("val next-token accuracy")
    ax[1].set_xscale("symlog", linthresh=100)
    ax[1].set_title(f"final val acc = {hist['val_acc'][-1]:.3f}"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, f"training_curves_{tag}.png"), dpi=120); plt.close(fig)
    print(f"[{tag}] DONE final_step={final_step} val_acc={hist['val_acc'][-1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
