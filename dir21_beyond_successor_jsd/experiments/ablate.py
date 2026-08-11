"""Which early components carry the per-token width trait?

Embedding edits are exhausted (they destroy the trait rather than steer it), so this moves the
intervention into the computation. For each of blocks 0-5 we mean-ablate ONE component at a time at
the final token position -- each of the 16 attention heads, or the whole MLP -- re-measure the
per-token anchor width w_hat_u for 12 tokens against the 6 anchor tokens, and score the component by
how much of the across-token SPREAD in w_hat_u it destroys and how far it moves the model's output.

Mean ablation replaces the component's final-position output by its mean over the 18 endpoint prompts
(12 tokens + 6 anchors) measured with nothing ablated. Writes results/ablate.json (partial-safe).
"""
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import run_pair
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint, jsd_bits
from common import D18, RESULTS

N_ANCHOR = 6
BLOCKS = range(6)
FRAME = FRAMES[0]
DEADLINE = float(os.environ.get("ABLATE_DEADLINE_S", 900))

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


class Ablator:
    """Mean-ablate one component (an attention head, or the whole MLP) at the last position."""

    def __init__(self, model, n_heads=16):
        self.model = model
        self.hd = model.config.hidden_size // n_heads
        self.spec = None          # (block, "mlp"|head index)
        self.mean = None          # replacement vector
        self.rec = {}             # capture buffer when self.spec is None and self.capture is True
        self.capture = False
        self.h = []
        for L, layer in enumerate(model.gpt_neox.layers):
            self.h.append(layer.mlp.register_forward_hook(self._mlp_hook(L)))
            self.h.append(layer.attention.dense.register_forward_pre_hook(self._attn_hook(L)))

    def _mlp_hook(self, L):
        def hook(mod, inp, out):
            if self.capture:
                self.rec[(L, "mlp")] = out[:, -1, :].detach().float().mean(0)
            if self.spec == (L, "mlp"):
                out = out.clone()
                out[:, -1, :] = self.mean.to(out.dtype)
            return out
        return hook

    def _attn_hook(self, L):
        def hook(mod, inp):
            x = inp[0]
            if self.capture:
                self.rec[(L, "attn")] = x[:, -1, :].detach().float().mean(0)
            if self.spec is not None and self.spec[0] == L and self.spec[1] != "mlp":
                h = self.spec[1]
                x = x.clone()
                x[:, -1, h * self.hd:(h + 1) * self.hd] = self.mean.to(x.dtype)
            return (x,) + tuple(inp[1:])
        return hook


def measure(model, patcher, ids_by_tok, anchor_ids, pre):
    """w_hat_u for every token against the 6 anchors, plus each endpoint's final-position logits."""
    anc = [endpoint(model, patcher, torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1))
           for a in anchor_ids]
    w, logits = {}, {}
    for s, i in ids_by_tok.items():
        ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
        x, z = endpoint(model, patcher, ids)
        logits[s] = z
        w[s] = float(np.nanmedian([run_pair(model, patcher, ids, x, z, xb, zb)[0]
                                   for xb, zb in anc]))
    return w, logits


def main():
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    used = set(ids_by_str.values())
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    anchors = pool[:: max(1, len(pool) // N_ANCHOR)][:N_ANCHOR]

    toks12 = list(json.load(open(f"{RESULTS}/mode_split.json"))["tokens"].keys())
    ids_by_tok = {s: ids_by_str[s] for s in toks12}

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    pre = tok(FRAME, return_tensors="pt").input_ids.cuda()
    patcher = Patcher(model)
    abl = Ablator(model)

    # replacement means over the 18 endpoint prompts, nothing ablated
    abl.capture = True
    ids = torch.cat([pre.repeat(len(ids_by_tok) + len(anchors), 1),
                     torch.tensor(list(ids_by_tok.values()) + anchors,
                                  device=pre.device).unsqueeze(1)], 1)
    with torch.inference_mode():
        patcher.bank = None
        model(ids)
    abl.capture = False
    means = {k: v.clone() for k, v in abl.rec.items()}

    t0 = time.time()
    base_w, base_z = measure(model, patcher, ids_by_tok, anchors, pre)
    base = np.array([base_w[s] for s in toks12])
    print(f"baseline mean {base.mean():.3f} sd {base.std(ddof=1):.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    comps = [(L, c) for L in BLOCKS for c in ["mlp"] + list(range(16))]
    rows = []
    for n, (L, c) in enumerate(comps):
        if time.time() - t0 > DEADLINE:
            print(f"deadline: stopping after {n}/{len(comps)} components", flush=True)
            break
        abl.spec = (L, c)
        abl.mean = (means[(L, "mlp")] if c == "mlp"
                    else means[(L, "attn")][c * abl.hd:(c + 1) * abl.hd])
        w, z = measure(model, patcher, ids_by_tok, anchors, pre)
        abl.spec = None
        v = np.array([w[s] for s in toks12])
        ok = np.isfinite(v) & np.isfinite(base)
        bits = float(np.mean([jsd_bits(base_z[s].log_softmax(-1).unsqueeze(0),
                                       z[s].log_softmax(-1).unsqueeze(0))[0].item()
                              for s in toks12]))
        rows.append(dict(block=L, comp=("mlp" if c == "mlp" else f"head{c}"),
                         w=[float(x) for x in v], mean=float(np.nanmean(v)),
                         sd=float(np.nanstd(v, ddof=1)),
                         r=float(np.corrcoef(base[ok], v[ok])[0, 1]) if ok.sum() > 3 else float("nan"),
                         bits=bits, n_valid=int(ok.sum())))
        print(f"[{n + 1}/{len(comps)}] block {L} {rows[-1]['comp']}: mean {rows[-1]['mean']:.3f} "
              f"sd {rows[-1]['sd']:.3f} r {rows[-1]['r']:+.2f} bits {bits:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        json.dump(dict(model=MODEL, revision=REVISION, frame=FRAME,
                       anchors=[tok.convert_ids_to_tokens(a) for a in anchors],
                       tokens=toks12, base_w=[float(x) for x in base], rows=rows),
                  open(f"{RESULTS}/ablate.json", "w"), indent=1)
    print("wrote results/ablate.json")


if __name__ == "__main__":
    main()
