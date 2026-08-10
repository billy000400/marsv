"""Partner-free basin radius: how far can a token's block-0 state move before its output changes?

For each endpoint token and sentence frame we step away from the token's post-block-0 residual state
along directions that have nothing to do with any pair partner (random directions, and directions
toward anchor tokens used in no pair), and record the angle at which the output distribution has moved
a fixed number of bits. Writes results/basin.json.
"""
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from common import D18, RESULTS

MODEL = "EleutherAI/pythia-1.4b-deduped"
REVISION = "step143000"
FRAMES = ["The thing was", "They said it was", "I thought it was"]
THETA = np.linspace(0.0, 1.0, 26)          # radians of great-circle travel from the endpoint state
TAUS = [0.05, 0.1, 0.2]                    # bits of output movement that define "left the basin"
N_RANDOM = 6
N_ANCHOR = 6
BATCH = 64

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


class Patcher:
    """Replace the final-position hidden state emitted by block `layer` (same hook as dir18).

    `layer="emb"` moves the site to the input embedding, before any block runs.
    """

    def __init__(self, model, layer=0):
        site = model.gpt_neox.embed_in if layer == "emb" else model.gpt_neox.layers[layer]
        self.h = site.register_forward_hook(self._hook)
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


def jsd_bits(logp, logq):
    """Base-2 JSD between two batches of log-probability rows."""
    p, q = logp.exp(), logq.exp()
    m = (0.5 * (p + q)).clamp_min(1e-30).log()
    kl = lambda lx, x: (x * (lx - m)).sum(-1)
    return (0.5 * kl(logp, p) + 0.5 * kl(logq, q)) / np.log(2)


@torch.inference_mode()
def endpoint(model, patcher, ids):
    """Post-block-0 final-position state and the unpatched final-position logits."""
    patcher.bank = None
    z = model(ids).logits[0, -1].float()
    return patcher.captured[0].clone().float(), z


@torch.inference_mode()
def sweep(model, patcher, ids, x, logp0, dirs):
    """Output movement M(theta) in bits along each direction; shape (n_dirs, len(THETA))."""
    n = x.norm()
    u = x / n
    th = torch.tensor(THETA, device=x.device, dtype=x.dtype).unsqueeze(1)
    out = []
    for d in dirs:
        d = d - (d @ u) * u
        d = d / d.norm()
        bank = n * (torch.cos(th) * u + torch.sin(th) * d)
        m = []
        for s in range(0, len(THETA), BATCH):
            chunk = bank[s:s + BATCH]
            patcher.bank = chunk
            z = model(ids.repeat(len(chunk), 1)).logits[:, -1, :].float()
            m.append(jsd_bits(z.log_softmax(-1), logp0.expand(len(chunk), -1)).cpu())
        patcher.bank = None
        out.append(torch.cat(m).numpy())
    return np.stack(out)


def radius(m, tau):
    """First angle at which output movement crosses tau bits (linear interpolation)."""
    hit = np.flatnonzero(m >= tau)
    if len(hit) == 0:
        return float("nan")
    k = hit[0]
    if k == 0:
        return 0.0
    lo, hi = m[k - 1], m[k]
    return float(THETA[k - 1] + (tau - lo) / (hi - lo) * (THETA[k] - THETA[k - 1]))


def main():
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    endpoints = sorted(ids_by_str.items())                    # 123 (string, id)
    used = set(ids_by_str.values())
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    anchors = pool[:: max(1, len(pool) // N_ANCHOR)][:N_ANCHOR]

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    patcher = Patcher(model)
    g = torch.Generator(device="cpu").manual_seed(0)
    rand_dirs = torch.randn(N_RANDOM, model.config.hidden_size, generator=g).cuda()

    rows = {s: dict(token_id=i, radius_random={f"{t}": [] for t in TAUS},
                    radius_anchor={f"{t}": [] for t in TAUS},
                    logit_norm=[], out_entropy=[], state_norm=[])
            for s, i in endpoints}
    for fi, frame in enumerate(FRAMES):
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        anchor_x = []
        for a in anchors:
            ids = torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1)
            anchor_x.append(endpoint(model, patcher, ids)[0])
        for k, (s, i) in enumerate(endpoints):
            ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
            x, z = endpoint(model, patcher, ids)
            logp0 = z.log_softmax(-1)
            dirs = list(rand_dirs) + [ax - x for ax in anchor_x]
            m = sweep(model, patcher, ids, x, logp0, dirs)
            r = rows[s]
            for tau in TAUS:
                rr = [radius(m[j], tau) for j in range(N_RANDOM)]
                ra = [radius(m[j], tau) for j in range(N_RANDOM, len(dirs))]
                r["radius_random"][f"{tau}"].append(float(np.nanmedian(rr)))
                r["radius_anchor"][f"{tau}"].append(float(np.nanmedian(ra)))
            r["logit_norm"].append(float((z - z.mean()).norm()))
            r["out_entropy"].append(float(-(logp0.exp() * logp0).sum() / np.log(2)))
            r["state_norm"].append(float(x.norm()))
            if k % 40 == 0:
                print(f"frame {fi} token {k}/{len(endpoints)} {s!r} "
                      f"r_rand(0.1)={r['radius_random']['0.1'][-1]:.3f}", flush=True)

    json.dump(dict(model=MODEL, revision=REVISION, frames=FRAMES, theta=list(THETA),
                   taus=TAUS, n_random=N_RANDOM,
                   anchors=[tok.convert_ids_to_tokens(a) for a in anchors],
                   tokens=rows),
              open(os.path.join(RESULTS, "basin.json"), "w"), indent=1)
    print("wrote results/basin.json")


if __name__ == "__main__":
    main()
