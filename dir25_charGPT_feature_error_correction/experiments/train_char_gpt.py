"""Reproduce dir13's step-30000 character-level Shakespeare GPT.

dir13's checkpoints lived under /tmp (symlinked from its results/) and have been wiped, so this
direction retrains the same model with dir13's exact recipe, config and seeds (see
../dir13_plateau_on_grok_gpt/experiments/train_grok.py and configs/grok_char.yaml).
Only the final checkpoint is kept (plus a rolling one for crash safety).
"""
import os, sys, json, time, hashlib
import numpy as np
import torch

D13 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "dir13_plateau_on_grok_gpt", "experiments")
sys.path.insert(0, D13)
from model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CKPT_DIR = os.path.join(RES, "checkpoints_char")
CORPUS = "/tmp/tinyshakespeare.txt"
CORPUS_SHA = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
STEPS, BS = 30000, 48
MAX_MINUTES = 105.0


def get_batch(data, block, bs, device, g):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=g)
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, block, bs, device, g, iters=20):
    model.eval()
    losses, correct, tot = [], 0, 0
    for _ in range(iters):
        x, y = get_batch(data, block, bs, device, g)
        logits, loss = model(x, y)
        losses.append(loss.item())
        correct += (logits.argmax(-1) == y).sum().item()
        tot += y.numel()
    model.train()
    return float(np.mean(losses)), correct / tot


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    os.makedirs(CKPT_DIR, exist_ok=True)

    raw = open(CORPUS, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == CORPUS_SHA, f"corpus SHA {sha} != dir13 training corpus"
    text = raw.decode("utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = GPTConfig(vocab_size=len(chars), block_size=128, n_layer=12, n_head=12, n_embd=240, dropout=0.2)
    model = GPT(cfg).to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M vocab={len(chars)}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0)
    warmup = 100

    def lr_at(step):
        if step < warmup:
            return 1e-3 * step / warmup
        prog = (step - warmup) / max(1, STEPS - warmup)
        return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + np.cos(np.pi * min(1.0, prog)))

    gtr = torch.Generator().manual_seed(42)
    gev = torch.Generator().manual_seed(43)
    hist = {"step": [], "train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    step = 0
    model.train()
    while step <= STEPS:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_data, cfg.block_size, BS, device, gtr)
        opt.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 250 == 0 or step == STEPS:
            vl, va = eval_loss(model, val_data, cfg.block_size, BS, device, gev)
            hist["step"].append(step); hist["train_loss"].append(loss.item())
            hist["val_loss"].append(vl); hist["val_acc"].append(va)
            print(f"step {step:6d} lr {lr_at(step):.2e} train {loss.item():.3f} val {vl:.3f} "
                  f"acc {va:.3f} [{(time.time()-t0)/60:.1f}m]", flush=True)
        if step % 2500 == 0 and step > 0:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step,
                        "stoi": stoi}, os.path.join(CKPT_DIR, "ckpt_rolling.pt"))
        if (time.time() - t0) / 60 > MAX_MINUTES:
            print(f"time budget hit at step {step}", flush=True)
            break
        step += 1

    final_step = hist["step"][-1]
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": final_step, "stoi": stoi},
               os.path.join(CKPT_DIR, "ckpt_final.pt"))
    json.dump(hist, open(os.path.join(RES, "train_hist_char.json"), "w"), indent=2)
    json.dump({"corpus_sha256": sha, "vocab_size": len(chars), "final_step": final_step,
               "final_val_loss": hist["val_loss"][-1], "final_val_acc": hist["val_acc"][-1],
               "cfg": cfg.__dict__, "model_seed": 1337, "data_seed": 42,
               "recipe": "dir13 train_grok.py --tok char (Adam lr 1e-3 cosine->1e-4, wd 0, bs 48x128)",
               "torch": torch.__version__},
              open(os.path.join(RES, "train_meta_char.json"), "w"), indent=2)
    print(f"DONE final_step={final_step} val_acc={hist['val_acc'][-1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
