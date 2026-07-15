"""Plateau assay for the char GPT (see ../PLAN.md).

Intervene on the FINAL-position residual stream after block l; measure downstream displacement of the
pre-head residual and of the next-char distribution. Because attention is causal, modifying only the
final position at block l changes only the final position downstream, so a full forward from block
l+1 with the clean residual and a replaced final position is exact.

Metrics:
  d_hidden(alpha) = || z(h + alpha u) - z(h) ||_2 / sqrt(d_model)
  PI = ∫_0^1 [ x - y(x) ] dx,  x = rho/rho_max, y = d(rho)/d(rho_max)   (positive => delayed response)
  sharpness = max finite-diff slope / mean slope of the normalized curve.
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["stoi"], ck["itos"]


def natural_scale(final_acts):
    """Median L2 distance between random pairs of final-position activations [N, C]."""
    N = final_acts.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    d = torch.norm(final_acts - final_acts[perm], dim=1)
    return float(d.median())


@torch.no_grad()
def cache_clean(model, contexts, device, mb=64):
    """Return per-block clean residuals list (len n_layer) each [N,T,C], and baseline logits [N,V]
    (final position). contexts: LongTensor [N,T]."""
    n_layer = model.cfg.n_layer
    res = [[] for _ in range(n_layer)]
    base_logits = []
    for i in range(0, contexts.size(0), mb):
        xb = contexts[i:i + mb].to(device)
        r, logits = model.residuals_and_logits(xb)
        for l in range(n_layer):
            res[l].append(r[l].cpu())
        base_logits.append(logits[:, -1, :].cpu())
    res = [torch.cat(v, 0) for v in res]
    return res, torch.cat(base_logits, 0)


@torch.no_grad()
def response_curves(model, clean_res_l, block_l, basepoints, directions, rhos, s_l, device, mb=48):
    """For each basepoint (final-position vector) and direction, sweep rho and measure downstream
    response. Returns dict of arrays shaped [n_bp, n_dir, n_rho].

    clean_res_l: clean post-block-l residual [n_bp, T, C] (surrounding positions held fixed).
    basepoints:  final-position vectors to place at position -1 [n_bp, C] (natural h or control).
    directions:  unit directions [n_dir, C].
    """
    n_bp, T, C = clean_res_l.shape
    n_dir = directions.shape[0]
    n_rho = len(rhos)
    start = block_l + 1
    d_hidden = np.zeros((n_bp, n_dir, n_rho), np.float32)
    jsd = np.zeros((n_bp, n_dir, n_rho), np.float32)
    flip = np.zeros((n_bp, n_dir, n_rho), np.float32)
    directions = directions.to(device)

    # baseline z(h) per basepoint (rho=0): build residual with basepoint at final pos
    def run(res_batch):
        z, logits = model.forward_from_block(res_batch, start)
        return z[:, -1, :], logits[:, -1, :]

    # precompute baseline (rho=0) using each basepoint
    base_z = torch.zeros(n_bp, C)
    base_p = torch.zeros(n_bp, model.cfg.vocab_size)
    for i in range(0, n_bp, mb):
        rb = clean_res_l[i:i + mb].clone().to(device)
        rb[:, -1, :] = basepoints[i:i + mb].to(device)
        z0, lg0 = run(rb)
        base_z[i:i + mb] = z0.cpu()
        base_p[i:i + mb] = F.softmax(lg0, -1).cpu()
    base_top1 = base_p.argmax(-1)

    for di in range(n_dir):
        u = directions[di]
        for ri, rho in enumerate(rhos):
            alpha = rho * s_l
            for i in range(0, n_bp, mb):
                rb = clean_res_l[i:i + mb].clone().to(device)
                bp = basepoints[i:i + mb].to(device)
                rb[:, -1, :] = bp + alpha * u
                z, lg = run(rb)
                p = F.softmax(lg, -1)
                z0 = base_z[i:i + mb].to(device)
                d_hidden[i:i + mb, di, ri] = (torch.norm(z - z0, dim=1) / math.sqrt(C)).cpu().numpy()
                p0 = base_p[i:i + mb].to(device)
                m = 0.5 * (p + p0)
                js = 0.5 * (p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1) \
                    + 0.5 * (p0 * (p0.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
                jsd[i:i + mb, di, ri] = js.clamp_min(0).cpu().numpy()
                flip[i:i + mb, di, ri] = (p.argmax(-1).cpu() != base_top1[i:i + mb]).float().numpy()
    return {"d_hidden": d_hidden, "jsd": jsd, "flip": flip}


def plateau_index(curve, rhos):
    """PI = ∫_0^1 [x - y(x)] dx for x=rho/rho_max, y=d/d_max. curve: 1D over rhos (rho[0]=0)."""
    rhos = np.asarray(rhos, float)
    x = rhos / rhos[-1]
    dmax = curve[-1]
    if dmax <= 1e-9:
        return 0.0
    y = curve / dmax
    return float(np.trapz(x - y, x))


def boundary_sharpness(curve, rhos):
    x = np.asarray(rhos, float) / rhos[-1]
    y = curve / (curve[-1] + 1e-12)
    dy = np.diff(y) / np.diff(x)
    mean_slope = (y[-1] - y[0]) / (x[-1] - x[0] + 1e-12)
    return float(np.max(dy) / (mean_slope + 1e-12))


# ---------------------------------------------------------------------------
def _unit_test():
    """The assay must detect a synthetic delayed-then-steep curve and score a linear one near zero."""
    rhos = np.linspace(0, 1, 41)
    linear = rhos.copy()
    # delayed: flat until 0.7 then steep
    delayed = np.where(rhos < 0.7, 0.05 * rhos / 0.7, 0.05 + 0.95 * (rhos - 0.7) / 0.3)
    pi_lin = plateau_index(linear, rhos)
    pi_del = plateau_index(delayed, rhos)
    s_lin = boundary_sharpness(linear, rhos)
    s_del = boundary_sharpness(delayed, rhos)
    assert abs(pi_lin) < 0.02, f"linear PI should be ~0, got {pi_lin}"
    assert pi_del > 0.2, f"delayed PI should be large, got {pi_del}"
    assert s_del > 2 * s_lin, f"delayed sharpness {s_del} should exceed linear {s_lin}"
    print(f"UNIT TEST PASS: PI_linear={pi_lin:.3f} PI_delayed={pi_del:.3f} "
          f"sharp_lin={s_lin:.2f} sharp_del={s_del:.2f}")


if __name__ == "__main__":
    _unit_test()
