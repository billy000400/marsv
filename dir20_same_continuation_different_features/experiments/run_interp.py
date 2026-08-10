"""S1+S2: validate tokenization, endpoint JSD, block-0 last-token interpolation sweep.

Writes results/<model>.npz (per-pair d-curves) and results/summary.json.
"""
import json
import os

import numpy as np
import torch

from common import PAIRS, PLOTS, RESULTS, blocks, load, tokenize_pair

N_ALPHA = 101
CHUNK = 32
torch.manual_seed(0)


def clean_run(m, ids):
    """Return (resid_post at every block, last position) and final logits for one prompt."""
    rec = {}
    hs = []

    def mk(i):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            rec[i] = h[0, -1, :].detach().float().clone()
        return hook

    for i, b in enumerate(blocks(m)):
        hs.append(b.register_forward_hook(mk(i)))
    with torch.no_grad():
        logits = m(torch.tensor([ids], device=next(m.parameters()).device),
                   use_cache=False).logits[0, -1, :].float().clone()
    for h in hs:
        h.remove()
    return rec, logits


def slerp_lerp_norm(ha, hb, alphas):
    """SLERP the direction, linearly interpolate the L2 norm (Matthew's interpolation)."""
    na, nb = ha.norm(), hb.norm()
    u, v = ha / na, hb / nb
    cos = torch.clamp((u * v).sum(), -1.0, 1.0)
    om = torch.arccos(cos)
    a = torch.tensor(alphas, device=ha.device, dtype=ha.dtype).unsqueeze(1)
    if om.abs() < 1e-6:
        d = (1 - a) * u + a * v
        d = d / d.norm(dim=1, keepdim=True)
    else:
        d = (torch.sin((1 - a) * om) * u + torch.sin(a * om) * v) / torch.sin(om)
    return ((1 - a) * na + a * nb) * d, float(om), float(cos)


def sweep(m, ids_a, vecs, layer=0):
    """Patch block-`layer` last-token resid_post with each row of `vecs`; record downstream + logits."""
    dev = next(m.parameters()).device
    nb = len(blocks(m))
    outs = {i: [] for i in range(layer + 1, nb)}
    logit_rows = []
    state = {}

    def patch(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, -1, :] = state["v"].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def mk(i):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            state["rec"][i] = h[:, -1, :].detach().float().clone()
        return hook

    bl = blocks(m)
    hs = [bl[layer].register_forward_hook(patch)]
    for i in range(layer + 1, nb):
        hs.append(bl[i].register_forward_hook(mk(i)))

    for s in range(0, vecs.shape[0], CHUNK):
        chunk = vecs[s:s + CHUNK]
        state["v"] = chunk
        state["rec"] = {}
        inp = torch.tensor([ids_a], device=dev).repeat(chunk.shape[0], 1)
        with torch.no_grad():
            lg = m(inp, use_cache=False).logits[:, -1, :].float().cpu()
        logit_rows.append(lg)
        for i in range(layer + 1, nb):
            outs[i].append(state["rec"][i].cpu())
    for h in hs:
        h.remove()
    return {i: torch.cat(v) for i, v in outs.items()}, torch.cat(logit_rows)


def rel_dist(x, xa, xb):
    """d = |x-xA| / (|x-xA| + |x-xB|), rows of x."""
    da = (x - xa).norm(dim=1)
    db = (x - xb).norm(dim=1)
    return (da / (da + db)).numpy()


def width_10_90(alphas, d):
    """First upward crossing of 0.1 and 0.9, linearly interpolated; None if never crossed."""
    def cross(t):
        for i in range(len(d) - 1):
            if d[i] <= t <= d[i + 1] and d[i + 1] > d[i]:
                f = (t - d[i]) / (d[i + 1] - d[i])
                return alphas[i] + f * (alphas[i + 1] - alphas[i])
        return None
    lo, hi = cross(0.1), cross(0.9)
    return None if (lo is None or hi is None) else hi - lo


def jsd(pa, pb):
    """Jensen-Shannon divergence in nats."""
    m = 0.5 * (pa + pb)
    kl = lambda p, q: float((p * (torch.log(p + 1e-45) - torch.log(q + 1e-45))).sum())
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)
    alphas = np.linspace(0, 1, N_ALPHA)
    summary = {}

    for mk_ in ["gpt2-medium", "pythia-410m"]:
        tok, m = load(mk_)
        curves, meta = {}, {}
        for key, label, prefix, a_str, b_str, is_ctrl in PAIRS:
            ok, ida, idb = tokenize_pair(tok, prefix, a_str, b_str)
            entry = dict(label=label, control=is_ctrl, valid=bool(ok),
                         tok_a=tok.convert_ids_to_tokens(ida[-1]),
                         tok_b=tok.convert_ids_to_tokens(idb[-1]),
                         n_prefix=len(ida) - 1)
            if not ok:
                meta[key] = entry
                print(mk_, key, "INVALID tokenization")
                continue

            reca, lga = clean_run(m, ida)
            recb, lgb = clean_run(m, idb)
            pa, pb = torch.softmax(lga, -1), torch.softmax(lgb, -1)
            entry["jsd"] = jsd(pa, pb)
            entry["top_a"] = [tok.convert_ids_to_tokens(int(i)) for i in pa.topk(5).indices]
            entry["top_b"] = [tok.convert_ids_to_tokens(int(i)) for i in pb.topk(5).indices]
            entry["ptop_a"] = [round(float(v), 4) for v in pa.topk(5).values]
            entry["ptop_b"] = [round(float(v), 4) for v in pb.topk(5).values]

            ha, hb = reca[0], recb[0]
            vecs, omega, cos = slerp_lerp_norm(ha, hb, alphas)
            entry["omega_rad"] = omega
            entry["cos_h0"] = cos
            entry["norm_a"] = float(ha.norm())
            entry["norm_b"] = float(hb.norm())

            downs, lg = sweep(m, ida, vecs)
            d_log = rel_dist(lg, lga.cpu().unsqueeze(0), lgb.cpu().unsqueeze(0))
            curves[f"{key}__logits"] = d_log
            entry["w10_90_logits"] = width_10_90(alphas, d_log)
            entry["endpoint_err"] = [float(abs(d_log[0])), float(abs(d_log[-1] - 1))]
            entry["monotonic"] = bool(np.all(np.diff(d_log) >= -1e-6))

            lw, lay = [], []
            for i in sorted(downs):
                di = rel_dist(downs[i], reca[i].cpu().unsqueeze(0), recb[i].cpu().unsqueeze(0))
                curves[f"{key}__L{i}"] = di
                lay.append(i)
                lw.append(width_10_90(alphas, di))
            entry["layers"] = lay
            entry["w10_90_layers"] = lw
            meta[key] = entry
            print(f"{mk_:12s} {key:14s} JSD={entry['jsd']:.4f}  w10-90={entry['w10_90_logits']}")

        np.savez(os.path.join(RESULTS, f"{mk_}.npz"), alphas=alphas, **curves)
        summary[mk_] = meta
        del m
        torch.cuda.empty_cache()

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", os.path.join(RESULTS, "summary.json"))


if __name__ == "__main__":
    main()
