"""GPT-2 Large activation-interpolation assay (Matthew Shinkle & StefanHex reference method).

For a minimal pair (prefix + endpoint A, prefix + endpoint B):
  1. read the final-position `resid_post` activation of block L for both prompts,
  2. interpolate with `slerp_rescale` (spherical interpolation of the direction, linear
     interpolation of the L2 norm),
  3. patch the interpolated vector into the final sequence position at block L's `resid_post`,
  4. run the remaining blocks and record downstream sites + final logits,
  5. score every recording site with Matthew's relative endpoint distance
         d(t) = ||x(t)-x_A|| / (||x(t)-x_A|| + ||x(t)-x_B||).

A plateau is a curve that stays near one endpoint (d~0), changes sharply, and stays near the
other (d~1). The raw d(t) curves are the primary evidence; width / location / presence are
frozen descriptive summaries of them.
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

SITES = ("attn_out", "resid_mid", "mlp_post", "mlp_out", "resid_post")
EPS_THETA = 1e-4   # near-collinear fallback threshold (radians)


# ----------------------------------------------------------------------------- interpolation
def slerp_rescale(hA, hB, ts):
    """Matthew's interpolation. hA,hB: [C]; ts: [K] in [0,1]. Returns [K,C].

    Slerp of the unit directions, linear interpolation of the norms. Near-collinear fallback
    (theta < EPS_THETA): renormalized linear interpolation, which agrees with slerp to O(theta^2).
    """
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
    return ((1 - ts) * nA + ts * nB) * u


def rel_dist(X, xA, xB):
    """Relative endpoint distance. X: [K,C]; xA,xB: [C]. Returns [K] numpy array."""
    dA = torch.linalg.norm(X - xA[None], dim=-1)
    dB = torch.linalg.norm(X - xB[None], dim=-1)
    return (dA / (dA + dB)).float().cpu().numpy()


# ------------------------------------------------------------------------------------- model
class Assay:
    """GPT-2 Large with recording hooks on every site and a final-position patch hook."""

    def __init__(self, model_name="gpt2-large", device="cuda", mem_frac=0.225, threads=2,
                 revision="main"):
        torch.set_num_threads(threads)
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(mem_frac)
        self.model_name, self.revision, self.device = model_name, revision, device
        self.tok = GPT2TokenizerFast.from_pretrained(model_name, revision=revision)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, revision=revision)
        self.model = self.model.to(device=device, dtype=torch.float32).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_layer = self.model.config.n_layer
        self._rec = {}
        self._patch = None          # (layer, H[K,C]) or None
        self._install_hooks()

    # -- hooks ---------------------------------------------------------------------------
    @staticmethod
    def _h(out):
        return out[0] if isinstance(out, tuple) else out

    def _install_hooks(self):
        m = self.model.transformer
        for l, blk in enumerate(m.h):
            blk.attn.register_forward_hook(self._mk_rec(f"L{l}.attn_out"))
            blk.ln_2.register_forward_pre_hook(self._mk_rec_pre(f"L{l}.resid_mid"))
            blk.mlp.act.register_forward_hook(self._mk_rec(f"L{l}.mlp_post"))
            blk.mlp.register_forward_hook(self._mk_rec(f"L{l}.mlp_out"))
            blk.register_forward_hook(self._mk_block(l))

    def _mk_rec(self, key):
        def fn(mod, inp, out):
            self._rec[key] = self._h(out)[:, -1, :].detach()
        return fn

    def _mk_rec_pre(self, key):
        def fn(mod, inp):
            self._rec[key] = inp[0][:, -1, :].detach()
        return fn

    def _mk_block(self, l):
        """resid_post recorder; also applies the patch when this is the interpolation layer."""
        def fn(mod, inp, out):
            h = self._h(out)
            if self._patch is not None and self._patch[0] == l:
                h = h.clone()
                h[:, -1, :] = self._patch[1]
                self._rec[f"L{l}.resid_post"] = h[:, -1, :].detach()
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self._rec[f"L{l}.resid_post"] = h[:, -1, :].detach()
            return None
        return fn

    # -- forward -------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, ids, patch=None):
        """ids: [B,T] int64 tensor. patch: (layer, H[B,C]) or None.
        Returns (rec dict site->[B,C] final-position activations, logits [B,V])."""
        self._rec, self._patch = {}, patch
        out = self.model(input_ids=ids.to(self.device))
        self._patch = None
        rec = self._rec
        self._rec = {}
        return rec, out.logits[:, -1, :].detach()

    # -- the assay -----------------------------------------------------------------------
    @torch.no_grad()
    def run_pair(self, ids_A, ids_B, interp_layer, ts, batch_k=50):
        """Full assay for one endpoint pair at one interpolation layer.

        ids_A/ids_B: [T] int64 numpy arrays, identical except the last element.
        Returns dict: d[site name] -> d(t) array (downstream sites only), plus check diagnostics.
        """
        assert ids_A.shape == ids_B.shape and (ids_A[:-1] == ids_B[:-1]).all() \
            and ids_A[-1] != ids_B[-1], "prompts must share the prefix and differ in the last token"
        xb = torch.from_numpy(np.stack([ids_A, ids_B])).long()
        rec0, lg0 = self.forward(xb)
        hA = rec0[f"L{interp_layer}.resid_post"][0].clone()
        hB = rec0[f"L{interp_layer}.resid_post"][1].clone()
        refA = {k: v[0].clone() for k, v in rec0.items()}
        refB = {k: v[1].clone() for k, v in rec0.items()}
        lgA, lgB = lg0[0].clone(), lg0[1].clone()

        tst = torch.tensor(ts, dtype=torch.float32, device=self.device)
        H = slerp_rescale(hA, hB, tst)                      # [K,C]

        keys = [f"L{l}.{s}" for l in range(interp_layer + 1, self.n_layer) for s in SITES]
        d = {k: np.empty(len(ts)) for k in keys}
        d["logits"] = np.empty(len(ts))
        ep = {}
        base = torch.from_numpy(ids_A).long()[None]         # prefix rows are identical for A/B
        for i0 in range(0, len(ts), batch_k):
            i1 = min(i0 + batch_k, len(ts))
            ids = base.repeat(i1 - i0, 1)
            rec, lg = self.forward(ids, patch=(interp_layer, H[i0:i1]))
            for k in keys:
                d[k][i0:i1] = rel_dist(rec[k], refA[k], refB[k])
            d["logits"][i0:i1] = rel_dist(lg, lgA, lgB)
            if i0 == 0:
                ep["t0_logit_maxabs"] = float((lg[0] - lgA).abs().max())
                ep["t0_patch_maxabs"] = float((H[0] - hA).abs().max())
                ep["t0_resid_maxabs"] = max(
                    float((rec[k] [0] - refA[k]).abs().max()) for k in keys) if keys else 0.0
            if i1 == len(ts):
                ep["t1_logit_maxabs"] = float((lg[-1] - lgB).abs().max())
                ep["t1_patch_maxabs"] = float((H[-1] - hB).abs().max())
                ep["t1_resid_maxabs"] = max(
                    float((rec[k][-1] - refB[k]).abs().max()) for k in keys) if keys else 0.0
        # scale-free versions of the endpoint-fidelity checks
        ep["t0_logit_rel"] = ep["t0_logit_maxabs"] / float(lgA.abs().max())
        ep["t1_logit_rel"] = ep["t1_logit_maxabs"] / float(lgB.abs().max())
        return {"d": d, "endpoint_err": ep,
                "norms": (float(torch.linalg.norm(hA)), float(torch.linalg.norm(hB))),
                "cos_AB": float(torch.dot(hA / torch.linalg.norm(hA), hB / torch.linalg.norm(hB)))}


# -------------------------------------------------------------------------- curve summaries
def pava_isotonic(y):
    """Pool-adjacent-violators: least-squares nondecreasing fit of y."""
    y = np.asarray(y, dtype=float)
    vals, wts = [], []
    for v in y:
        vals.append(float(v)); wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            m = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals = vals[:-2] + [m]; wts = wts[:-2] + [w]
    return np.concatenate([np.full(int(w), v) for v, w in zip(vals, wts)])


def _cross(ts, iso, level):
    """First upward crossing of `level`, linearly interpolated on the grid; nan if none."""
    idx = int(np.argmax(iso >= level))
    if iso[idx] < level:
        return np.nan
    if idx == 0:
        return float(ts[0])
    t0, t1, y0, y1 = ts[idx - 1], ts[idx], iso[idx - 1], iso[idx]
    return float(t0 + (level - y0) / (y1 - y0) * (t1 - t0)) if y1 > y0 else float(t1)


def summarize(ts, d, near=0.10, min_run_frac=0.20, dev_max=0.10):
    """Frozen descriptive summary of one d(t) curve (thresholds fixed before any curve was seen).

    width    = t(d=0.9) - t(d=0.1) on the isotonic copy; nan if either crossing is missing.
    location = t(d=0.5); nan if missing.
    plateau  = a run of >= min_run_frac of the grid with d <= near at the low end AND a run with
               d >= 1-near at the high end, a defined width, and a (near-)monotone curve
               (max |raw - isotonic| <= dev_max). The straight line d(t)=t spends exactly 10% of
               the grid within 0.1 of each endpoint, so min_run_frac = 0.20 rejects it by design.
    """
    d = np.asarray(d, dtype=float)
    iso = pava_isotonic(d)
    max_dev = float(np.abs(d - iso).max())
    t_lo, t_mid, t_hi = _cross(ts, iso, near), _cross(ts, iso, 0.5), _cross(ts, iso, 1 - near)
    width = t_hi - t_lo if np.isfinite(t_lo) and np.isfinite(t_hi) else np.nan
    n_min = max(1, int(round(min_run_frac * len(ts))))
    lo_run = _max_run(d <= near)
    hi_run = _max_run(d >= 1 - near)
    plateau = bool(np.isfinite(width) and max_dev <= dev_max
                   and lo_run >= n_min and hi_run >= n_min)
    return {"width": width, "location": t_mid, "t_lo": t_lo, "t_hi": t_hi,
            "max_dev": max_dev, "lo_run": int(lo_run), "hi_run": int(hi_run),
            "plateau": plateau}


def _max_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# ------------------------------------------------------------------------------- self-tests
def self_test():
    ts = np.linspace(0, 1, 50)
    step = 1 / (1 + np.exp(-(ts - 0.5) / 0.02))
    line = ts.copy()
    s_step, s_line = summarize(ts, step), summarize(ts, line)
    assert s_step["plateau"] and s_step["width"] < 0.15, s_step
    assert not s_line["plateau"] and s_line["width"] > 0.7, s_line
    assert abs(s_step["location"] - 0.5) < 0.03, s_step

    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    dd = rel_dist(X, X[0], X[1])
    assert dd[0] == 0.0 and dd[1] == 1.0

    hA, hB = torch.tensor([2.0, 0.0]), torch.tensor([0.0, 4.0])
    H = slerp_rescale(hA, hB, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(H[0], hA, atol=1e-6) and torch.allclose(H[2], hB, atol=1e-6)
    assert abs(float(torch.linalg.norm(H[1])) - 3.0) < 1e-5
    assert torch.isfinite(slerp_rescale(hA, hA * 1.5 + 1e-9, torch.tensor([0.0, .5, 1.0]))).all()
    print("assay.self_test OK  (step width=%.3f, line width=%.3f)"
          % (s_step["width"], s_line["width"]))


if __name__ == "__main__":
    self_test()
