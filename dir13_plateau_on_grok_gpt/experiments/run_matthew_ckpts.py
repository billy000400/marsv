"""PLAN Exp 3 — Matthew's exact assay across training checkpoints.

BPE primary: shared_context "The house was", pairs (" big"," in") and (" big"," large"),
each completion a single GPT-2 BPE token (gate: tokenization_check.txt).
Char control: same context + final chars ('b','i') and ('b','l') — labeled tokenizer controls.

Per checkpoint and pair: sweep EVERY interpolation layer (resid_post blocks 0..11), 50 evenly
spaced t including endpoints, slerp_rescale (spherical direction + linear norm), patch only the
final position, and record Matthew's downstream hooks (attn_out, resid_mid, mlp_post, mlp_out,
resid_post per downstream block) plus final logits; d(t) = ||x(t)-x_A|| / (||x(t)-x_A||+||x(t)-x_B||).

Saves raw d(t) for every (step, pair, interp_layer, hook) to a compressed npz + a tidy summary
JSON with transition widths. Resumable per checkpoint.
"""
import os, sys, json, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig
from matthew_assay import slerp_norm, rel_dist, transition_width, self_test

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

CONTEXT = "The house was"
BPE_PAIRS = [("big", "in"), ("big", "large")]
CHAR_PAIRS = [("b", "i"), ("b", "l")]


def make_sequences(tok):
    """Return list of (pair_name, seq_A, seq_B) as int64 numpy arrays."""
    out = []
    if tok == "bpe":
        from transformers import GPT2TokenizerFast
        tkz = GPT2TokenizerFast.from_pretrained("gpt2")
        ctx = tkz(CONTEXT)["input_ids"]
        for a, b in BPE_PAIRS:
            ia = tkz(CONTEXT + " " + a)["input_ids"]
            ib = tkz(CONTEXT + " " + b)["input_ids"]
            assert ia[:len(ctx)] == ctx and ib[:len(ctx)] == ctx
            assert len(ia) == len(ctx) + 1 and len(ib) == len(ctx) + 1, "completion not single token"
            out.append((f"{a}/{b}", np.array(ia, dtype=np.int64), np.array(ib, dtype=np.int64)))
    else:
        text = open("/tmp/tinyshakespeare.txt").read()
        stoi = {c: i for i, c in enumerate(sorted(set(text)))}
        base = CONTEXT + " "
        for a, b in CHAR_PAIRS:
            sa = np.array([stoi[c] for c in base + a], dtype=np.int64)
            sb = np.array([stoi[c] for c in base + b], dtype=np.int64)
            out.append((f"{a}/{b}", sa, sb))
    return out


@torch.no_grad()
def forward_hooks(model, x, start_block):
    """Run blocks[start_block:] on residual x [K,T,C]; return dict of final-position activations
    at every Matthew hook downstream, plus final logits. Keys like 'b{l}.attn_out'."""
    rec = {}
    cur = x
    for l in range(start_block, len(model.blocks)):
        blk = model.blocks[l]
        attn_out = blk.attn(blk.ln_1(cur))
        resid_mid = cur + attn_out
        h = blk.mlp.c_fc(blk.ln_2(resid_mid))
        mlp_post = F.gelu(h)
        mlp_out = blk.mlp.c_proj(mlp_post)
        cur = resid_mid + mlp_out
        rec[f"b{l}.attn_out"] = attn_out[:, -1, :]
        rec[f"b{l}.resid_mid"] = resid_mid[:, -1, :]
        rec[f"b{l}.mlp_post"] = mlp_post[:, -1, :]
        rec[f"b{l}.mlp_out"] = mlp_out[:, -1, :]
        rec[f"b{l}.resid_post"] = cur[:, -1, :]
    rec["logits"] = model.head(model.ln_f(cur))[:, -1, :]
    return rec


@torch.no_grad()
def run_pair_ckpt(model, seq_A, seq_B, ts, device):
    """Full interpolation-layer sweep for one pair. Returns
    {interp_block: {hook: d(t) np.array}}, plus check diagnostics."""
    assert (seq_A[:-1] == seq_B[:-1]).all() and seq_A[-1] != seq_B[-1]
    xb = torch.from_numpy(np.stack([seq_A, seq_B])).to(device)
    res, logits = model.residuals_and_logits(xb)
    prefix_err = max(float((r[0, :-1] - r[1, :-1]).abs().max()) for r in res)
    tst = torch.tensor(ts, dtype=res[0].dtype, device=device)

    out, checks = {}, {"prefix_err": prefix_err}
    for L in range(len(model.blocks)):
        hA = res[L][0, -1].clone()
        hB = res[L][1, -1].clone()
        H = slerp_norm(hA, hB, tst)                        # [K,C]
        base = res[L][0]                                    # prefix rows identical (checked)
        x = base[None].repeat(len(ts), 1, 1)
        x[:, -1, :] = H
        rec = forward_hooks(model, x, L + 1)
        # endpoint references from the SAME hook walk on the unpatched pair
        ref = forward_hooks(model, torch.stack([res[L][0], res[L][1]]), L + 1)
        d = {}
        for k, v in rec.items():
            if not k.startswith(f"b") or int(k.split(".")[0][1:]) > L:
                d[k] = rel_dist(v, ref[k][0], ref[k][1])
        out[L] = d
        if L == 0:
            checks["t0_logit_err"] = float((rec["logits"][0] - logits[0, -1]).abs().max())
            checks["t1_logit_err"] = float((rec["logits"][-1] - logits[1, -1]).abs().max())
    return out, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", choices=["char", "bpe"], required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--steps", default="all", help="comma list of ckpt steps, or 'all'")
    ap.add_argument("--n_t", type=int, default=50)          # Matthew: exactly 50
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--vram_frac", type=float, default=0.05)
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.vram_frac)
    torch.set_num_threads(1)
    device = "cuda"
    self_test()

    seqs = make_sequences(args.tok)
    ts = np.linspace(0.0, 1.0, args.n_t)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_*.pt")))
    ckpts = [c for c in ckpts if "final" not in c]
    if args.steps != "all":
        want = {int(s) for s in args.steps.split(",")}
        ckpts = [c for c in ckpts if int(os.path.basename(c)[5:-3]) in want]
    print(f"{len(ckpts)} checkpoints x {len(seqs)} pairs", flush=True)

    npz_path = os.path.join(RES, f"matthew_{args.out_tag}_raw.npz")
    sum_path = os.path.join(RES, f"matthew_{args.out_tag}_summary.json")
    raw = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}
    summary = json.load(open(sum_path)) if os.path.exists(sum_path) else []
    done_steps = {s["step"] for s in summary}

    for cp in ckpts:
        ck = torch.load(cp, map_location=device, weights_only=False)
        step = ck["step"]
        if step in done_steps:
            continue
        model = GPT(GPTConfig(**ck["cfg"])).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        rec_steps = {"step": step, "pairs": {}}
        for name, sa, sb in seqs:
            out, checks = run_pair_ckpt(model, sa, sb, ts, device)
            pair_sum = {"checks": checks, "interp": {}}
            for L, hooks in out.items():
                for hk, d in hooks.items():
                    raw[f"s{step}|{name}|L{L}|{hk}"] = d
                w, t_lo, t_hi, dev = transition_width(ts, hooks["logits"])
                pair_sum["interp"][L] = {"w_logits": None if np.isnan(w) else round(float(w), 4),
                                         "iso_dev": round(float(dev), 4)}
            rec_steps["pairs"][name] = pair_sum
            print(f"step {step:6d} {name}: w_logits by interp L: "
                  + " ".join(f"{L}:{v['w_logits']}" for L, v in pair_sum['interp'].items()),
                  flush=True)
        summary.append(rec_steps)
        summary.sort(key=lambda r: r["step"])
        raw["ts"] = ts
        np.savez_compressed(npz_path, **raw)
        json.dump(summary, open(sum_path, "w"), indent=1)
        del model
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
