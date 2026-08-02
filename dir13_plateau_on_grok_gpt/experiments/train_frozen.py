"""Freeze a block group at its step-0 weights and train the rest -- PLAN 5.5's falsifiable prediction.

The per-block MLP scan (mlp_block_scan.py) showed the plateau sharpness is *built* by blocks 1-4:
deleting their MLP branches returns the median transition width to the untrained value while the
readout still makes sharp next-character decisions. The hypothesis in RESULTS/REPORT therefore ends
on a prediction that no post-hoc ablation can test, because ablation removes a *trained* component:

    freeze blocks 1-4 at their step-0 (initialization) weights, train the rest of the network to
    matched validation accuracy, and the interpolation paths should stay straight (median width
    near the untrained 0.80) even though the model predicts next characters as well as the
    reference run.

This script runs that. Everything except the frozen parameters is identical to the reference fresh
character run (`train_grok.py --tok char`): same corpus + SHA, same tokenizer, same 90/10 split,
same model seed 1337 (so the frozen blocks hold *exactly* the reference run's initialization), same
data seed, same Adam(1e-3 cosine->1e-4, betas 0.9/0.99, wd 0) over the same 30000-step schedule,
same batch size, same checkpoint grid.

Two conditions, run identically:
  --freeze 1,2,3,4    the causal group (the prediction's test)
  --freeze 8,9,10,11  specificity control: the same *number* of blocks, at the depth the MLP-gain
                      probe showed contributes almost nothing to sharpness. If freezing any four
                      blocks straightened the paths, this control would straighten them too.

Checkpoints go to --ckpt_root (default /tmp, off the quota-limited shared volume; checkpoints are
gitignored scratch in this project). History/meta -> results/train_{hist,meta}_<tag>.json.
The first checkpoint whose val accuracy reaches the reference run's final val accuracy is saved
separately as ckpt_matched.pt: that is the "matched validation accuracy" the prediction asks for.
"""
import os, sys, json, time, hashlib, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig
from train_grok import get_batch, eval_loss, tokenize_corpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

REF_FINAL_VAL_ACC = 0.5501790364583333  # results/train_meta_grok_char.json, the reference run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", default="", help="comma-separated block indices frozen at init ('' = none)")
    ap.add_argument("--tag", required=True, help="run tag, e.g. frozen_early")
    ap.add_argument("--n_embd", type=int, default=240, help="model width (240 = reference run)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--max_minutes", type=float, default=95.0)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--vram_frac", type=float, default=0.11)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ckpt_root", default="/tmp/dir13_frozen")
    ap.add_argument("--seed", type=int, default=1337, help="model init seed (1337 = reference run)")
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.vram_frac)
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frozen = [int(x) for x in args.freeze.split(",") if x.strip()]
    model_seed, data_seed = args.seed, 42
    torch.manual_seed(model_seed)

    tag = args.tag
    ckpt_dir = os.path.join(args.ckpt_root, f"checkpoints_{tag}")
    os.makedirs(ckpt_dir, exist_ok=True)

    raw = open("/tmp/tinyshakespeare.txt", "rb").read()
    corpus_sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    data, vocab_size, _ = tokenize_corpus("char", text)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = GPTConfig(vocab_size=vocab_size, block_size=128, n_layer=12, n_head=12, n_embd=args.n_embd, dropout=0.2)
    model = GPT(cfg).to(device)

    n_frozen = 0
    for l in frozen:
        for p in model.blocks[l].parameters():
            p.requires_grad = False
            n_frozen += p.numel()
    trainable = [p for p in model.parameters() if p.requires_grad]
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] params={nparams/1e6:.2f}M frozen={n_frozen/1e6:.2f}M "
          f"({100*n_frozen/nparams:.1f}%) blocks={frozen} vocab={vocab_size}", flush=True)

    opt = torch.optim.Adam(trainable, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0)
    warmup, total = 100, args.steps

    def lr_at(step):
        if step < warmup:
            return 1e-3 * step / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + np.cos(np.pi * min(1.0, prog)))

    gtr = torch.Generator().manual_seed(data_seed)
    gev = torch.Generator().manual_seed(data_seed + 1)

    logsp = np.unique(np.round(np.logspace(0, np.log10(total), 24)).astype(int))
    ckpt_steps = sorted(set([0] + list(logsp) + list(range(2500, total + 1, 2500)) + [total]))
    hist = {"step": [], "train_loss": [], "val_loss": [], "val_acc": []}
    matched = None
    t0 = time.time()
    step = 0
    model.train()
    while step <= total:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_data, cfg.block_size, args.bs, device, gtr)
        opt.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % 250 == 0 or step == total:
            vl, va = eval_loss(model, val_data, cfg.block_size, args.bs, device, gev, iters=20)
            hist["step"].append(step); hist["train_loss"].append(loss.item())
            hist["val_loss"].append(vl); hist["val_acc"].append(va)
            print(f"[{tag}] step {step:6d} lr {lr_at(step):.2e} train {loss.item():.3f} val {vl:.3f} "
                  f"acc {va:.3f} [{(time.time()-t0)/60:.1f}m]", flush=True)
            json.dump(hist, open(os.path.join(ckpt_dir, "train_hist.json"), "w"))
            if matched is None and va >= REF_FINAL_VAL_ACC:
                matched = {"step": step, "val_acc": va, "val_loss": vl}
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step,
                            "tok": "char", "frozen_blocks": frozen, "val_acc": va},
                           os.path.join(ckpt_dir, "ckpt_matched.pt"))
                print(f"[{tag}] MATCHED reference val acc {REF_FINAL_VAL_ACC:.4f} at step {step} "
                      f"(acc {va:.4f})", flush=True)

        if step in ckpt_steps:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step,
                        "tok": "char", "frozen_blocks": frozen},
                       os.path.join(ckpt_dir, f"ckpt_{step:06d}.pt"))

        out_of_time = (time.time() - t0) / 60 > args.max_minutes
        if out_of_time or step == total:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step,
                        "tok": "char", "frozen_blocks": frozen, "val_acc": hist["val_acc"][-1]},
                       os.path.join(ckpt_dir, "ckpt_last.pt"))
            if out_of_time:
                print(f"[{tag}] time budget hit at step {step}, stopping", flush=True)
                break
        step += 1

    final_step = hist["step"][-1]
    meta = {"corpus_sha256": corpus_sha, "tokenizer": "char", "vocab_size": vocab_size,
            "frozen_blocks": frozen, "frozen_params": int(n_frozen), "params": int(nparams),
            "model_seed": model_seed, "data_seed": data_seed, "cfg": cfg.__dict__,
            "optimizer": "Adam(lr 1e-3 cosine->1e-4, betas 0.9/0.99, wd 0.0)",
            "planned_steps": total, "final_step": int(final_step), "batch_size": args.bs,
            "ckpt_dir": ckpt_dir,
            "final_val_loss": hist["val_loss"][-1], "final_val_acc": hist["val_acc"][-1],
            "ref_final_val_acc": REF_FINAL_VAL_ACC, "matched": matched,
            "best_val_acc": float(max(hist["val_acc"])),
            "minutes": round((time.time() - t0) / 60, 2), "torch": torch.__version__}
    json.dump(hist, open(os.path.join(ckpt_dir, "train_hist.json"), "w"), indent=2)
    json.dump(meta, open(os.path.join(ckpt_dir, "train_meta.json"), "w"), indent=2)
    for name, obj in (("hist", hist), ("meta", meta)):   # best-effort copy onto the shared volume
        try:
            json.dump(obj, open(os.path.join(RES, f"train_{name}_{tag}.json"), "w"), indent=2)
        except OSError as e:
            print(f"[{tag}] could not write train_{name}_{tag}.json to results/: {e}", flush=True)
    print(f"[{tag}] DONE step={final_step} val_acc={hist['val_acc'][-1]:.4f} "
          f"best={max(hist['val_acc']):.4f} matched={matched}", flush=True)


if __name__ == "__main__":
    main()
