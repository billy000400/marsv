"""S2: full-sequence resid_post interpolation after block 0, 50 steps, final-position logits."""
import json
import os

import numpy as np
import torch

from common import DEV, PAIRS, RESULTS, blocks, load

N_T = 50
CHUNK = 25
LAYER = 0


def resid_post0(m, ids):
    """resid_post after block 0 for the whole sequence, plus the clean final-position logits."""
    store = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        store["h"] = h[0].detach().float().clone()

    hd = blocks(m)[LAYER].register_forward_hook(hook)
    with torch.no_grad():
        lg = m(torch.tensor([ids], device=DEV), use_cache=False).logits[0, -1, :].float().clone()
    hd.remove()
    return store["h"], lg


def slerp_norm(ha, hb, ts):
    """Per-position SLERP of the direction with linear interpolation of the L2 norm.

    ha, hb: (T, D). Returns (n_t, T, D).
    """
    na, nb = ha.norm(dim=-1, keepdim=True), hb.norm(dim=-1, keepdim=True)
    u, v = ha / na, hb / nb
    cos = torch.clamp((u * v).sum(-1, keepdim=True), -1.0, 1.0)
    om = torch.arccos(cos)
    t = torch.tensor(ts, device=ha.device, dtype=ha.dtype).view(-1, 1, 1)
    sin = torch.sin(om)
    d = torch.where(sin.abs() < 1e-6,
                    (1 - t) * u + t * v,
                    (torch.sin((1 - t) * om) * u + torch.sin(t * om) * v) / sin.clamp(min=1e-12))
    d = d / d.norm(dim=-1, keepdim=True)
    return ((1 - t) * na + t * nb) * d, cos.squeeze(-1).cpu().numpy()


def sweep(m, ids, vecs):
    """Patch the FULL block-0 output sequence with each interpolated state; collect final logits."""
    state = {}

    def patch(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = state["v"].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    hd = blocks(m)[LAYER].register_forward_hook(patch)
    rows = []
    for s in range(0, vecs.shape[0], CHUNK):
        state["v"] = vecs[s:s + CHUNK]
        inp = torch.tensor([ids], device=DEV).repeat(state["v"].shape[0], 1)
        with torch.no_grad():
            rows.append(m(inp, use_cache=False).logits[:, -1, :].float().cpu())
    hd.remove()
    return torch.cat(rows)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    tok, m = load()
    ts = np.linspace(0.0, 1.0, N_T)
    summary = []
    for key, label, pa, pb, intended in PAIRS:
        ida, idb = tok(pa)["input_ids"], tok(pb)["input_ids"]
        if key != "p1_copy_vs_recall":
            continue  # only the pair that passed the S1 completion screen
        ha, lga = resid_post0(m, ida)
        hb, lgb = resid_post0(m, idb)
        vecs, cos = slerp_norm(ha, hb, ts)
        X = sweep(m, ida, vecs)
        A, B = X[0], X[-1]
        da = (X - A).norm(dim=1)
        db = (X - B).norm(dim=1)
        d = (da / (da + db)).numpy()
        top = [tok.decode(int(i)) for i in X.argmax(dim=1)]
        summary.append(dict(
            key=key, label=label, prompt_a=pa, prompt_b=pb, n_tokens=len(ida),
            endpoint_logit_l2=float((lga - lgb).norm()),
            recon_err_a=float((A - lga.cpu()).norm()), recon_err_b=float((B - lgb.cpu()).norm()),
            top_a=tok.decode(int(lga.argmax())), top_b=tok.decode(int(lgb.argmax())),
            top_tokens_unique=sorted(set(top)),
            n_steps_top_is_intended=int(sum(t.strip() == intended for t in top)),
            per_pos_cos=[round(float(c), 4) for c in cos]))
        np.savez(os.path.join(RESULTS, f"{key}_interp.npz"), t=ts, d=d,
                 logits_A=lga.cpu().numpy(), logits_B=lgb.cpu().numpy())
        print(key, "d(t) =", np.round(d, 3))
    with open(os.path.join(RESULTS, "s2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
