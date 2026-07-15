"""Train the reconstruction 12-layer/12-head char GPT on Tiny Shakespeare within a shared-GPU budget.

Saves log-spaced checkpoints, a training-curves plot, and provenance meta. See ../MODEL_SPEC.md.
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
CKPT = os.path.join(RES, "checkpoints")
os.makedirs(CKPT, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--max_minutes", type=float, default=26.0)
    ap.add_argument("--bs", type=int, default=48)
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_seed, data_seed = 1337, 42
    torch.manual_seed(model_seed)

    text = open("/tmp/tinyshakespeare.txt", "rb").read()
    corpus_sha = hashlib.sha256(text).hexdigest()
    text = text.decode("utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = GPTConfig(vocab_size=len(chars), block_size=128, n_layer=12, n_head=12, n_embd=240, dropout=0.2)
    model = GPT(cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params={nparams/1e6:.2f}M vocab={len(chars)} device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1)
    warmup, total = 100, args.steps

    def lr_at(step):
        if step < warmup:
            return 1e-3 * step / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + np.cos(np.pi * min(1.0, prog)))

    gtr = torch.Generator().manual_seed(data_seed)
    gev = torch.Generator().manual_seed(data_seed + 1)

    # log-spaced checkpoint steps
    ckpt_steps = sorted(set([0] + [int(x) for x in np.unique(np.round(np.logspace(0, np.log10(total), 8)).astype(int))] + [total]))
    hist = {"step": [], "train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    step = 0
    model.train()
    while step <= total:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_data, cfg.block_size, args.bs, device, gtr)
        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0 or step == total:
            vl, va = eval_loss(model, val_data, cfg.block_size, args.bs, device, gev)
            hist["step"].append(step); hist["train_loss"].append(loss.item())
            hist["val_loss"].append(vl); hist["val_acc"].append(va)
            print(f"step {step:5d} lr {lr_at(step):.2e} train {loss.item():.3f} val {vl:.3f} acc {va:.3f} "
                  f"[{(time.time()-t0)/60:.1f}m]", flush=True)

        if step in ckpt_steps:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step,
                        "stoi": stoi, "itos": itos}, os.path.join(CKPT, f"ckpt_{step:05d}.pt"))

        if (time.time() - t0) / 60 > args.max_minutes:
            print("time budget hit, stopping", flush=True)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step,
                        "stoi": stoi, "itos": itos}, os.path.join(CKPT, f"ckpt_{step:05d}.pt"))
            break
        step += 1

    final_step = hist["step"][-1]
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": final_step,
                "stoi": stoi, "itos": itos}, os.path.join(CKPT, "ckpt_final.pt"))

    json.dump(hist, open(os.path.join(RES, "train_hist.json"), "w"), indent=2)
    meta = {"corpus_sha256": corpus_sha, "n_chars": len(text), "vocab_size": len(chars),
            "vocab": "".join(chars), "model_seed": model_seed, "data_seed": data_seed,
            "params_millions": nparams / 1e6, "cfg": cfg.__dict__, "device": device,
            "torch": torch.__version__, "final_step": final_step,
            "final_val_loss": hist["val_loss"][-1], "final_val_acc": hist["val_acc"][-1],
            "ckpt_steps": [s for s in ckpt_steps if s <= final_step]}
    json.dump(meta, open(os.path.join(RES, "train_meta.json"), "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["step"], hist["train_loss"], label="train", alpha=0.7)
    ax[0].plot(hist["step"], hist["val_loss"], label="val", lw=2)
    ax[0].set_xlabel("step"); ax[0].set_ylabel("cross-entropy loss (nats)")
    ax[0].set_title("Training curves — 12L/12H char GPT"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(hist["step"], hist["val_acc"], color="C2", lw=2)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("val next-char accuracy")
    ax[1].set_title(f"Final val acc = {hist['val_acc'][-1]:.3f}"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "training_curves.png"), dpi=120); plt.close(fig)
    print("DONE", json.dumps(meta)[:200], flush=True)


if __name__ == "__main__":
    main()
