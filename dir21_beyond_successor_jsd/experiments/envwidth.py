"""A width statistic that is defined for EVERY interpolation curve, not only monotone ones.

dir18's width w is only defined when the curve d(t) rises monotonically and crosses 0.1 and 0.9
exactly once each. Every Pythia curve measured in this direction satisfies that. GPT-2's do not: they
wander (backslide up to 0.29), so 89% of them are rejected and the cross-model comparison the plan
asks for cannot be made with the strict statistic.

The envelope width replaces the curve by its running maximum e(t) = max_{s<=t} d(s), which is
non-decreasing, starts at d(0)=0 and ends at d(1)=1 by construction, so it crosses 0.1 and 0.9 exactly
once and

    w_env = t(e = 0.9) - t(e = 0.1)

exists for every curve. On monotone curves e = d and w_env = w exactly, so the two statistics are
comparable and the substitution is validated inside Pythia (where both are defined) before it is used
on GPT-2.
"""
import numpy as np
import torch

from anchor_width import GRID, BATCH, slerp_bank
from basin_probe import jsd_bits

import sys
sys.path.append("/workspace/marsv_agent_haoyang/dir18_continuation_jsd_plateau/experiments")
import curve_metrics


def _cross(e, grid, level):
    """Position where the non-decreasing curve e first reaches `level` (linear interpolation)."""
    k = int(np.argmax(e >= level))
    if k == 0:
        return float(grid[0])
    lo, hi = e[k - 1], e[k]
    return float(grid[k - 1] + (level - lo) / (hi - lo) * (grid[k] - grid[k - 1]))


def edge_drift(d, grid=GRID):
    """How much d(t) moves in the outer 10% of the path at each end: d(0.1) + (1 - d(0.9)).

    d(0) = 0 and d(1) = 1 by construction, so this is the total rise over the first and last tenth
    of the path. A flat plateau leaves the endpoints alone (E ~ 0); a straight line in output space
    gives d(t) = t and E = 0.2 exactly.
    """
    d = np.asarray(d, float)
    return float(np.interp(0.1, grid, d) + 1.0 - np.interp(0.9, grid, d))


def env_metrics(d, grid=GRID):
    """Envelope width, the strict width, and how far the raw curve backslides."""
    d = np.asarray(d, float)
    e = np.maximum.accumulate(d)
    m = curve_metrics.metrics(d, grid)
    return dict(w_env=_cross(e, grid, 0.9) - _cross(e, grid, 0.1), w=m["w"],
                valid=bool(m["valid"]), backslide=m["backslide"], edge=edge_drift(d, grid),
                span=bool(m["span"]), mono=bool(m["mono"]), single=bool(m["single"]))


@torch.inference_mode()
def run_pair(model, patcher, ids_a, xa, za, xb, zb):
    """Both width statistics for one (token, anchor) pair in one frame."""
    bank, _ = slerp_bank(xa, xb)
    z = []
    for s in range(0, len(GRID), BATCH):
        chunk = bank[s:s + BATCH]
        patcher.bank = chunk
        z.append(model(ids_a.repeat(len(chunk), 1)).logits[:, -1, :].float())
    patcher.bank = None
    z = torch.cat(z)
    da = (z - za).norm(dim=1)
    db = (z - zb).norm(dim=1)
    out = env_metrics((da / (da + db)).cpu().numpy())
    out["out_jsd"] = float(jsd_bits(za.log_softmax(-1).unsqueeze(0), zb.log_softmax(-1).unsqueeze(0))[0])
    return out


def token_widths(model, patcher, tok, ids_by_str, anchors, frames, log_every=40):
    """Per-token lists of the 18 (frame, anchor) measurements of each statistic."""
    keys = ("w_env", "w", "backslide", "edge", "out_jsd", "valid", "mono", "single")
    raw = {s: {k: [] for k in keys} for s in ids_by_str}
    for fi, frame in enumerate(frames):
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        anc = []
        for a in anchors:
            ids = torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1)
            patcher.bank = None
            z = model(ids).logits[0, -1].float()
            anc.append((patcher.captured[0].clone().float(), z))
        for k, (s, i) in enumerate(ids_by_str.items()):
            ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
            patcher.bank = None
            z = model(ids).logits[0, -1].float()
            x = patcher.captured[0].clone().float()
            for xb, zb in anc:
                m = run_pair(model, patcher, ids, x, z, xb, zb)
                for key in keys:
                    raw[s][key].append(float(m[key]))
            if log_every and k % log_every == 0:
                print(f"  frame {fi} token {k}/{len(ids_by_str)} {s!r} "
                      f"w_env={np.median(raw[s]['w_env']):.3f} "
                      f"valid={np.mean(raw[s]['valid']):.2f}", flush=True)
    return raw
