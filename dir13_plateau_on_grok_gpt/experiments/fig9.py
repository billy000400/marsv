"""Figure 9 measurement pipeline (PLAN Exp 1/2): local complexity + PGD adversarial accuracy
across checkpoints, source-locked to github.com/AhmedImtiazPrio/grok-adversarial.

Per checkpoint:
  - clean next-token accuracy on held-out (val) windows;
  - eps=0.03 l_inf-PGD adversarial accuracy in token-embedding space (10 iters, alpha=eps/4,
    random start, clamped to the clean batch's embedding min/max — mirrors their dmin/dmax);
  - local complexity on 1024 train / test / random points: P=25 hull vertices (12 +/- antipodal
    pairs on the L2 sphere of radius r=0.005 in flattened embedding space, plus the centroid),
    counting per layer the GeLU pre-activation units whose sign is non-constant across the hull
    (their get_intersection_from_activation_batched), summed over the 12 MLP layers.

Reconstruction choices (recorded in MODEL_SPEC.md): perturbations act on wte+pos embeddings;
random points are uniform random token sequences; accuracy is the mean over all positions.
Resumable: skips checkpoints already in the output JSON.
"""
import os, sys, json, glob, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def embed(model, idx):
    pos = torch.arange(idx.size(1), device=idx.device)
    return model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]


def logits_from_emb(model, e):
    x = e
    for blk in model.blocks:
        x = blk(x)
    return model.head(model.ln_f(x))


@torch.no_grad()
def signs_from_emb(model, e):
    """Pre-GeLU activation signs per MLP layer: list of 12 bool tensors [B,T,4C]."""
    x = e
    signs = []
    for blk in model.blocks:
        x = x + blk.attn(blk.ln_1(x))
        h = blk.mlp.c_fc(blk.ln_2(x))
        signs.append(h > 0)
        x = x + blk.mlp.c_proj(F.gelu(h))
    return signs


@torch.no_grad()
def local_complexity(model, idx_batch, r, P, gen):
    """Sum over 12 layers of #units with non-constant sign over the P-vertex hull. [B]"""
    B, T = idx_batch.shape
    e = embed(model, idx_batch)                       # [B,T,C]
    C = e.size(-1)
    npair = (P - 1) // 2
    d = torch.randn(npair, B, T * C, device=e.device, generator=gen)
    d = d / d.norm(dim=-1, keepdim=True) * r          # radius-r sphere, flattened emb space
    d = d.reshape(npair, B, T, C)
    hull = torch.cat([e[None] + d, e[None] - d, e[None]], dim=0)   # [P,B,T,C]
    hull = hull.transpose(0, 1).reshape(B * P, T, C)
    signs = signs_from_emb(model, hull)
    n_inter = torch.zeros(B, device=e.device)
    for s in signs:
        s = s.reshape(B, P, -1)
        n_inter += (s[:, 1:, :] != s[:, :1, :]).any(dim=1).sum(dim=-1).float()
    return n_inter.cpu().numpy()


def pgd_adv_accuracy(model, x, y, eps, alpha, steps, gen):
    """Untargeted l_inf PGD in embedding space (faithful port of their PGD class)."""
    e = embed(model, x).detach()
    dmin, dmax = e.min(), e.max()
    adv = e + (torch.rand(e.shape, device=e.device, generator=gen) * 2 * eps - eps)
    adv = torch.clamp(adv, min=dmin, max=dmax).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        logits = logits_from_emb(model, adv)
        cost = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        grad = torch.autograd.grad(cost, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        delta = torch.clamp(adv - e, min=-eps, max=eps)
        adv = torch.clamp(e + delta, min=dmin, max=dmax).detach()
    with torch.no_grad():
        pred = logits_from_emb(model, adv).argmax(-1)
    return (pred == y).float().mean().item()


def windows(data, T, N, g):
    ix = torch.randint(len(data) - T - 1, (N,), generator=g)
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
    return x, y


def ci99(a):
    return 2.576 * float(np.std(a)) / np.sqrt(len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--tok", choices=["char", "bpe"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="all", help="comma list of ckpt steps to evaluate, or 'all'")
    ap.add_argument("--n_points", type=int, default=1024)   # paper: 1024
    ap.add_argument("--P", type=int, default=25)            # paper: 25
    ap.add_argument("--r", type=float, default=0.005)       # paper: 0.005
    ap.add_argument("--eps", type=float, default=0.03)      # paper: 0.03
    ap.add_argument("--pgd_steps", type=int, default=10)    # their atk_itrs
    ap.add_argument("--lc_bs", type=int, default=8)         # 8 anchors x 25 hull = 200 seqs/fwd
    ap.add_argument("--pgd_bs", type=int, default=64)
    ap.add_argument("--vram_frac", type=float, default=0.05)
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.vram_frac)
    torch.set_num_threads(1)
    device = "cuda"
    T = 128

    text = open("/tmp/tinyshakespeare.txt").read()
    if args.tok == "char":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        data = np.array([stoi[c] for c in text], dtype=np.uint16)
        vocab = len(chars)
    else:
        data = np.load("/tmp/tinyshakespeare_gpt2bpe.npy")
        vocab = 50257
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # frozen evaluation points (same across all checkpoints)
    g = torch.Generator().manual_seed(20260717)
    xtr, ytr = windows(train_data, T, args.n_points, g)
    xte, yte = windows(val_data, T, args.n_points, g)
    xrnd = torch.randint(vocab, (args.n_points, T), generator=g)
    xa, ya = windows(val_data, T, args.n_points, g)   # attack set (held-out)

    done = {}
    if os.path.exists(args.out):
        done = {rec["step"]: rec for rec in json.load(open(args.out))}

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_*.pt")))
    ckpts = [c for c in ckpts if "final" not in c]
    if args.steps != "all":
        want = {int(s) for s in args.steps.split(",")}
        ckpts = [c for c in ckpts if int(os.path.basename(c)[5:-3]) in want]
    print(f"{len(ckpts)} checkpoints, {len(done)} already done", flush=True)

    records = list(done.values())
    for cp in ckpts:
        ck = torch.load(cp, map_location=device, weights_only=False)
        step = ck["step"]
        if step in done:
            continue
        t0 = time.time()
        cfg = GPTConfig(**ck["cfg"])
        model = GPT(cfg).to(device)
        model.load_state_dict(ck["model"])
        model.eval()

        # clean accuracy (val)
        with torch.no_grad():
            correct = tot = 0
            for i in range(0, args.n_points, args.pgd_bs):
                xb, yb = xte[i:i + args.pgd_bs].to(device), yte[i:i + args.pgd_bs].to(device)
                pred = logits_from_emb(model, embed(model, xb)).argmax(-1)
                correct += (pred == yb).sum().item(); tot += yb.numel()
            clean_acc = correct / tot

        # PGD adversarial accuracy
        gp = torch.Generator(device=device).manual_seed(step + 1)
        accs = []
        for i in range(0, args.n_points, args.pgd_bs):
            xb, yb = xa[i:i + args.pgd_bs].to(device), ya[i:i + args.pgd_bs].to(device)
            accs.append(pgd_adv_accuracy(model, xb, yb, args.eps, args.eps / 4, args.pgd_steps, gp))
        adv_acc = float(np.mean(accs))

        # local complexity on train / test / random
        lc = {}
        gl = torch.Generator(device=device).manual_seed(step + 2)
        for name, xs in [("train", xtr), ("test", xte), ("random", xrnd)]:
            vals = []
            for i in range(0, args.n_points, args.lc_bs):
                vals.append(local_complexity(model, xs[i:i + args.lc_bs].to(device), args.r, args.P, gl))
            v = np.concatenate(vals)
            lc[name] = {"mean": float(v.mean()), "ci99": ci99(v)}

        rec = {"step": step, "clean_acc": clean_acc, "adv_acc": adv_acc, "lc": lc}
        records.append(rec)
        records.sort(key=lambda r: r["step"])
        json.dump(records, open(args.out, "w"), indent=1)
        print(f"step {step:6d} clean {clean_acc:.3f} adv {adv_acc:.3f} "
              f"LC tr {lc['train']['mean']:.0f} te {lc['test']['mean']:.0f} rnd {lc['random']['mean']:.0f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
