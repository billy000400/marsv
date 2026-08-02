"""S4 — Matthew-style two-endpoint interpolation assay.

For a minimal pair (prefix+A, prefix+B): take final-position resid_post activations h_A, h_B at
interpolation block L, spherically interpolate (slerp directions, linearly interpolate norms),
patch h(t) into the final position, run the remaining blocks, and record downstream activations
and final logits. Score with Matthew's relative distance

    d(t) = ||x(t)-x_A|| / (||x(t)-x_A|| + ||x(t)-x_B||).

Also provides the optional scalar summary: transition width w_{10->90} = t(d=0.9) - t(d=0.1),
computed on an isotonic (PAVA) copy of the curve; raw curves are always what gets plotted.
"""
import numpy as np
import torch

EPS_THETA = 1e-4  # near-collinear fallback threshold (radians), documented in REPORT.md


def slerp_norm(hA, hB, ts):
    """Norm-interpolating slerp. hA,hB: [C] tensors; ts: [K] tensor in [0,1]. Returns [K,C].
    Slerps the unit directions and linearly interpolates the norms (Matthew's scheme).
    Near-collinear fallback (theta < EPS_THETA): linear interpolation of unit directions,
    renormalized — in that regime it agrees with slerp to O(theta^2)."""
    nA, nB = torch.linalg.norm(hA), torch.linalg.norm(hB)
    uA, uB = hA / nA, hB / nB
    cos = torch.clamp(torch.dot(uA, uB), -1.0, 1.0)
    theta = torch.arccos(cos)
    ts = ts[:, None]
    if theta < EPS_THETA:
        u = (1 - ts) * uA[None] + ts * uB[None]
        u = u / torch.linalg.norm(u, dim=-1, keepdim=True)
    else:
        s = torch.sin(theta)
        u = (torch.sin((1 - ts) * theta) / s) * uA[None] + (torch.sin(ts * theta) / s) * uB[None]
    norm = (1 - ts) * nA + ts * nB
    return norm * u


def rel_dist(X, xA, xB):
    """Matthew's relative distance. X: [K,C]; xA,xB: [C]. Returns [K] numpy array."""
    dA = torch.linalg.norm(X - xA[None], dim=-1)
    dB = torch.linalg.norm(X - xB[None], dim=-1)
    return (dA / (dA + dB)).cpu().numpy()


@torch.no_grad()
def run_pair(model, seq_A, seq_B, interp_block, ts, device, batch_k=101, return_logits=False):
    """Run the full assay for one pair.

    seq_A/seq_B: [T] int64 numpy arrays identical except the last element.
    interp_block: interpolate resid_post of this block (patched activation feeds block+1).
    ts: [K] numpy array of interpolation steps in [0,1].

    Returns dict with d(t) in final-logit space, d(t) per downstream resid_post block,
    and check diagnostics (endpoint reproduction errors, prefix activation match).
    """
    assert (seq_A[:-1] == seq_B[:-1]).all() and seq_A[-1] != seq_B[-1]
    xb = torch.from_numpy(np.stack([seq_A, seq_B])).to(device)
    res, logits = model.residuals_and_logits(xb)  # res[l]: [2,T,C]
    T = xb.shape[1]
    # prefix invariance: earlier-position activations of A and B must match at every block
    prefix_err = max(float((res[l][0, :-1] - res[l][1, :-1]).abs().max()) for l in range(len(res)))
    hA = res[interp_block][0, -1].clone()
    hB = res[interp_block][1, -1].clone()
    xA_logit, xB_logit = logits[0, -1].clone(), logits[1, -1].clone()

    tst = torch.tensor(ts, dtype=hA.dtype, device=device)
    H = slerp_norm(hA, hB, tst)  # [K,C]

    n_layer = len(res)
    rec_blocks = list(range(interp_block + 1, n_layer))  # downstream resid_post recording points
    d_layer = {l: np.empty(len(ts)) for l in rec_blocks}
    d_logit = np.empty(len(ts))
    argmax = np.empty(len(ts), dtype=np.int64)
    all_logits = np.empty((len(ts), xA_logit.shape[0]), dtype=np.float32) if return_logits else None
    ep_err = {}

    base = res[interp_block][0]  # [T,C]; prefix rows identical for A and B (checked above)
    for i0 in range(0, len(ts), batch_k):
        i1 = min(i0 + batch_k, len(ts))
        x = base[None].repeat(i1 - i0, 1, 1)
        x[:, -1, :] = H[i0:i1]
        cur = x
        rec = {}
        for l in range(interp_block + 1, n_layer):
            cur = model.blocks[l](cur)
            rec[l] = cur[:, -1, :]
        z = model.ln_f(cur)
        lg = model.head(z)[:, -1, :]
        for l in rec_blocks:
            d_layer[l][i0:i1] = rel_dist(rec[l], res[l][0, -1], res[l][1, -1])
        d_logit[i0:i1] = rel_dist(lg, xA_logit, xB_logit)
        argmax[i0:i1] = lg.argmax(dim=-1).cpu().numpy()
        if return_logits:
            all_logits[i0:i1] = lg.float().cpu().numpy()
        if i0 == 0:
            ep_err["t0_logit"] = float((lg[0] - xA_logit).abs().max())
            ep_err["t0_patch"] = float((H[0] - hA).abs().max())
        if i1 == len(ts):
            ep_err["t1_logit"] = float((lg[-1] - xB_logit).abs().max())
            ep_err["t1_patch"] = float((H[-1] - hB).abs().max())

    out = {"d_logit": d_logit, "d_layer": d_layer, "argmax": argmax, "prefix_err": prefix_err,
           "endpoint_err": ep_err, "d0": float(d_logit[0]), "d1": float(d_logit[-1])}
    if return_logits:
        out["logits"] = all_logits
    return out


def pava_isotonic(y):
    """Pool-adjacent-violators: least-squares nondecreasing fit of y. Returns array like y."""
    y = np.asarray(y, dtype=float)
    vals, wts = [], []
    for v in y:
        vals.append(v); wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            m = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals = vals[:-2] + [m]; wts = wts[:-2] + [w]
    out = np.concatenate([np.full(int(w), v) for v, w in zip(vals, wts)])
    return out


def iso_crossing(ts, iso, level):
    """Linearly interpolated first upward crossing of `level` by the isotonic curve `iso`."""
    idx = np.argmax(iso >= level)
    if iso[idx] < level:
        return np.nan
    if idx == 0:
        return float(ts[0])
    t0, t1, y0, y1 = ts[idx - 1], ts[idx], iso[idx - 1], iso[idx]
    return float(t0 + (level - y0) / (y1 - y0) * (t1 - t0)) if y1 > y0 else float(t1)


def transition_width(ts, d, lo=0.1, hi=0.9):
    """w_{10->90} on the isotonic copy: first t with iso-d >= hi minus first t with iso-d >= lo.
    Returns (width, t_lo, t_hi, max_iso_dev). max_iso_dev = max |raw - isotonic|: large values
    flag non-monotone curves, which are reported separately and excluded from the width stat."""
    iso = pava_isotonic(d)
    max_dev = float(np.abs(d - iso).max())
    t_lo, t_hi = iso_crossing(ts, iso, lo), iso_crossing(ts, iso, hi)
    w = t_hi - t_lo if np.isfinite(t_lo) and np.isfinite(t_hi) else np.nan
    return w, t_lo, t_hi, max_dev


def is_plateau(ts, d, w_max=0.25, margin=0.10, dev_max=0.10):
    """Frozen candidate-plateau rule from PLAN.md: narrow transition (w<=0.25) AND >=10% of the
    path near each endpoint outside the transition, on a (near-)monotone curve."""
    w, t_lo, t_hi, max_dev = transition_width(ts, d)
    if not np.isfinite(w) or max_dev > dev_max:
        return False, w, t_lo, t_hi, max_dev
    ok = (w <= w_max) and (t_lo >= margin) and (t_hi <= 1.0 - margin)
    return bool(ok), w, t_lo, t_hi, max_dev


def self_test():
    """Synthetic-path checks required by PLAN.md."""
    ts = np.linspace(0, 1, 101)
    step = 1 / (1 + np.exp(-(ts - 0.5) / 0.02))          # sharp sigmoid ~ plateau-boundary-plateau
    line = ts.copy()
    ok_s, w_s, *_ = is_plateau(ts, step)
    ok_l, w_l, *_ = is_plateau(ts, line)
    assert ok_s and w_s < 0.15, f"step not detected: w={w_s}"
    assert (not ok_l) and w_l > 0.25, f"line misdetected: w={w_l}"
    # rel_dist endpoint identities
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    d = rel_dist(X, X[0], X[1])
    assert d[0] == 0.0 and d[1] == 1.0
    # slerp endpoints + norm interpolation
    hA = torch.tensor([2.0, 0.0]); hB = torch.tensor([0.0, 4.0])
    H = slerp_norm(hA, hB, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(H[0], hA, atol=1e-6) and torch.allclose(H[2], hB, atol=1e-6)
    assert abs(float(torch.linalg.norm(H[1])) - 3.0) < 1e-5  # linear norm interp
    # near-collinear fallback
    H2 = slerp_norm(hA, hA * 1.5 + 1e-9, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.isfinite(H2).all()
    print("self_test OK: step w=%.3f detected, line w=%.3f rejected" % (w_s, w_l))


if __name__ == "__main__":
    self_test()
