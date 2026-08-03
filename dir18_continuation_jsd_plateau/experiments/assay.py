"""S3: Matthew-style block-0-to-logit activation-plateau assay on Pythia.

For an endpoint pair (a, b) in a carrier context we take the final-position residual stream after
block `L`, norm-rescaled-SLERP between the two endpoint states at 50 points, patch only that one
position, run the remaining blocks, and read the final-position logits. The outcome is the relative
distance in LOGIT space,

    d(t) = ||z(t) - z_A|| / (||z(t) - z_A|| + ||z(t) - z_B||),

and the transition width w = t(d=0.9) - t(d=0.1). Small w = sharp transition = plateau.
"""
import numpy as np
import torch

import curve_metrics

N_T = 50
GRID = np.linspace(0.0, 1.0, N_T)


def slerp_bank(xa, xb, grid=GRID):
    """Norm-rescaled SLERP. Returns (len(grid), d) and a flag for the near-collinear fallback."""
    na, nb = xa.norm(), xb.norm()
    ua, ub = xa / na, xb / nb
    cos = torch.clamp((ua * ub).sum(), -1.0, 1.0)
    om = torch.arccos(cos)
    t = torch.tensor(grid, device=xa.device, dtype=xa.dtype).unsqueeze(1)
    if torch.sin(om) < 1e-6:  # near-collinear: SLERP is numerically unstable, LERP then renormalise
        u = (1 - t) * ua + t * ub
        u = u / u.norm(dim=1, keepdim=True)
        fallback = True
    else:
        u = (torch.sin((1 - t) * om) * ua + torch.sin(t * om) * ub) / torch.sin(om)
        fallback = False
    r = (1 - t) * na + t * nb
    return r * u, fallback, float(cos)


class Patcher:
    """Replaces the final-position hidden state emitted by block `layer` with a bank of states."""

    def __init__(self, model, layer):
        self.h = model.gpt_neox.layers[layer].register_forward_hook(self._hook)
        self.bank = None
        self.captured = None

    def _hook(self, mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        self.captured = hs[:, -1, :].detach().clone()
        if self.bank is None:
            return out
        hs = hs.clone()
        hs[:, -1, :] = self.bank
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def close(self):
        self.h.remove()


@torch.inference_mode()
def endpoint_states(model, ids_a, ids_b, layer):
    """Final-position post-block-`layer` residual for both endpoint prompts, plus prefix check."""
    p = Patcher(model, layer)
    p.bank = None
    logits, states = [], []
    for ids in (ids_a, ids_b):
        logits.append(model(ids).logits[0, -1].float())
        states.append(p.captured[0].clone())
    p.close()
    return states[0], states[1], logits[0], logits[1]


@torch.inference_mode()
def run_pair(model, ids_a, ids_b, layer=0, valid=None, grid=GRID):
    """Return dict with the raw d(t) curve, width w, and the endpoint self-tests."""
    xa, xb, za, zb = endpoint_states(model, ids_a, ids_b, layer)
    bank, fallback, cos = slerp_bank(xa, xb, grid)

    p = Patcher(model, layer)
    p.bank = bank
    batch = ids_a.repeat(len(grid), 1)
    z = model(batch).logits[:, -1, :].float()
    p.close()

    if valid is not None:
        z, za_v, zb_v = z[:, valid], za[valid], zb[valid]
    else:
        za_v, zb_v = za, zb
    da = (z - za_v).norm(dim=1)
    db = (z - zb_v).norm(dim=1)
    d = (da / (da + db)).cpu().numpy()

    # endpoint self-test: patching t=0 / t=1 must reproduce the unpatched endpoint logits
    err0 = float((z[0] - za_v).abs().max())
    err1 = float((z[-1] - zb_v).abs().max())
    scale = float(za_v.abs().max())
    return dict(d=d, w=width(d, grid), cos=cos, fallback=fallback,
                err0=err0 / scale, err1=err1 / scale,
                out_jsd=float(logit_jsd(za_v, zb_v)),
                dist=float((xa - xb).norm()),
                cos_block0=float(torch.nn.functional.cosine_similarity(xa, xb, dim=0)))


def width(d, grid=GRID):
    """w = t(d=0.9) - t(d=0.1); NaN unless the curve passes the strict validity criteria."""
    return curve_metrics.metrics(d, grid)["w"]


def logit_jsd(za, zb):
    """Base-2 JSD between the two endpoint next-token distributions the model actually predicts."""
    pa = torch.softmax(za, -1)
    pb = torch.softmax(zb, -1)
    m = 0.5 * (pa + pb)
    kl = lambda p: (p * (torch.log2(p.clamp_min(1e-30)) - torch.log2(m.clamp_min(1e-30)))).sum()
    return 0.5 * (kl(pa) + kl(pb))
