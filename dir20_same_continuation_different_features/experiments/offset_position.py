"""S13: does the plateau survive when the interpolated token is NOT the last token?

Every sweep in this report so far interpolates the FINAL token of the prompt and reads the very
next logits, so the interpolated vector reaches the readout through the residual stream of its own
position. Here the same low-JSD pairs get a shared continuation appended after the differing token:

    A = prefix + [a] + suffix        B = prefix + [b] + suffix

with `suffix` = the model's own greedy continuation of `prefix + [a]` (so both sequences carry the
SAME continuation, and the difference is a feature of an earlier position). The block-0 SLERP patch
is applied at the differing position, and d(alpha) is still read at the final logits -- which now
sit s tokens downstream and can only see the patched position through attention.

s = 0 reproduces the existing lowjsd sweep exactly and serves as a harness check.

Writes results/offset_<model>.json (per-pair per-s metrics) and results/offset_<model>.npz (curves).
"""
import json
import os
import sys

import numpy as np
import torch

from analyze import plateau_fraction, tv_width
from common import RESULTS, blocks, load
from mine_lowjsd import SEED, get_prefixes
from run_interp import jsd, rel_dist, slerp_lerp_norm, width_10_90

N_PAIRS = int(os.environ.get("N_PAIRS", 120))   # evenly spaced over each model's low-JSD bank
SUFFIX_LENS = [0, 1, 2, 4]
N_ALPHA = 101
CHUNK = 32


def clean_at(m, ids, pos):
    """Block-0 resid_post at `pos` and the final-position logits, for one prompt."""
    state = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        state["h"] = h[0, pos, :].detach().float().clone()

    hd = blocks(m)[0].register_forward_hook(hook)
    with torch.no_grad():
        lg = m(torch.tensor([ids], device=next(m.parameters()).device),
               use_cache=False).logits[0, -1, :].float().clone()
    hd.remove()
    return state["h"], lg


def sweep_at(m, ids, vecs, pos):
    """Patch block-0 resid_post at `pos` with each row of `vecs`; return final-position logits.

    Every chunk is padded to exactly CHUNK rows. With a shared suffix the two endpoint logit
    vectors can sit within 1e-3 of each other, and float32 matmul kernels differ with batch shape,
    so a batch-1 reference run would put a numerical offset into d(alpha) comparable to the signal.
    A constant batch shape makes every row -- including the endpoint references -- kernel-identical.
    """
    dev = next(m.parameters()).device
    state = {}

    def patch(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, pos, :] = state["v"].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    hd = blocks(m)[0].register_forward_hook(patch)
    inp = torch.tensor([ids], device=dev).repeat(CHUNK, 1)
    rows = []
    for s in range(0, vecs.shape[0], CHUNK):
        v = vecs[s:s + CHUNK]
        n = v.shape[0]
        if n < CHUNK:
            v = torch.cat([v, v[-1:].repeat(CHUNK - n, 1)])
        state["v"] = v
        with torch.no_grad():
            rows.append(m(inp, use_cache=False).logits[:, -1, :].float().cpu()[:n])
    hd.remove()
    return torch.cat(rows)


def greedy(m, ids, n):
    """The model's own greedy continuation of `ids`, n tokens."""
    dev = next(m.parameters()).device
    cur = list(ids)
    out = []
    for _ in range(n):
        with torch.no_grad():
            lg = m(torch.tensor([cur], device=dev), use_cache=False).logits[0, -1, :]
        t = int(lg.argmax())
        out.append(t)
        cur.append(t)
    return out


def main(model_keys):
    alphas = np.linspace(0, 1, N_ALPHA)
    smax = max(SUFFIX_LENS)
    for mkey in model_keys:
        tok, m = load(mkey)
        prefixes = get_prefixes(tok, np.random.default_rng(SEED))
        bank = json.load(open(os.path.join(RESULTS, f"lowjsd_{mkey}.json")))
        take = np.unique(np.linspace(0, len(bank) - 1, N_PAIRS).round().astype(int))
        rows, curves = [], {}

        for n_done, bi in enumerate(take):
            r0 = bank[int(bi)]
            pre = prefixes[r0["prefix_idx"]]
            ta, tb = r0["id_a"], r0["id_b"]
            pos = len(pre)
            suf = greedy(m, pre + [ta], smax)

            for s in SUFFIX_LENS:
                ida = pre + [ta] + suf[:s]
                idb = pre + [tb] + suf[:s]
                ha, lga = clean_at(m, ida, pos)
                hb, lgb = clean_at(m, idb, pos)
                vecs, omega, cos = slerp_lerp_norm(ha, hb, alphas)
                # rows 0,1 are the endpoint references, run through the identical batched path
                out = sweep_at(m, ida, torch.cat([ha.unsqueeze(0), hb.unsqueeze(0), vecs]), pos)
                xa, xb, lgs = out[:1], out[1:2], out[2:]
                sep = float((xa - xb).norm())
                d = rel_dist(lgs, xa, xb)
                w = width_10_90(alphas, d)
                key = f"{r0['key']}_s{s}"
                curves[key] = d.astype(np.float32)
                rows.append(dict(
                    key=key, pair=r0["key"], s=s, jsd_final=jsd(torch.softmax(lga, -1),
                                                                torch.softmax(lgb, -1)),
                    jsd_s0=r0["jsd"], w=None if w is None else float(w), sep=sep,
                    wtv=tv_width(alphas, d), pf=plateau_fraction(d),
                    mono=bool(np.all(np.diff(d) >= -1e-6)), cos_h0=cos, omega=omega,
                    endpoint_err=[float(abs(d[0])), float(abs(d[-1] - 1))]))
            if n_done % 20 == 0:
                print(f"  {mkey} pair {n_done}/{len(take)}", flush=True)

        np.savez(os.path.join(RESULTS, f"offset_{mkey}.npz"), alphas=alphas, **curves)
        with open(os.path.join(RESULTS, f"offset_{mkey}.json"), "w") as f:
            json.dump(rows, f, indent=1)

        for s in SUFFIX_LENS:
            g = [r for r in rows if r["s"] == s]
            wtv = np.array([r["wtv"] for r in g])
            sharp = np.mean([r["w"] is not None and r["w"] < 0.5 for r in g])
            jf = np.median([r["jsd_final"] for r in g])
            print(f"{mkey} s={s}: n={len(g)} median w_TV={np.median(wtv):.3f} "
                  f"sharp={100*sharp:.1f}% median JSD_final={jf:.4f}", flush=True)
        print(f"{mkey}: max endpoint err {max(max(r['endpoint_err']) for r in rows):.2e}",
              flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ("gpt2-small", "gpt2-medium", "gpt2-large"))
