"""Does the per-token width trait survive a different tokenizer, corpus and architecture family?

Every measurement in this direction so far comes from one model family (Pythia, trained on the Pile,
parallel attention+MLP residual). The claim the deliverables make -- that a token's transition width is
a property of the TOKEN and the level is a property of the network -- has been tested across context,
model size and training time, but never outside that family. GPT-2 small is the cheapest real test: a
different training corpus (WebText), a different BPE vocabulary, and a serial rather than parallel
residual block. All 123 endpoint token strings and all 6 anchor strings are single tokens in GPT-2's
vocabulary too, so the SAME strings, anchors and frames can be measured with no substitution.

A. anchor widths w_hat_u at block 0 for the 123 endpoint tokens (3 frames x 6 anchors);
B. an embedding probe refitted inside GPT-2 (its own W_E -> its own widths), held out;
C. mean-ablation of the MLP and of the attention block in blocks 0-5, to ask whether the block-0 MLP
   is again the single early component whose removal erases the ordering.

Writes results/gpt2.json stage by stage.
"""
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPT2LMHeadModel

import second_model
from basin_probe import FRAMES, endpoint, jsd_bits
from common import RESULTS
from mlp_read import probe
from second_model import endpoint_set, widths

MODEL = "gpt2"
BLOCKS = range(6)
N_TRAIN = 80
DEADLINE = float(os.environ.get("GPT2_DEADLINE_S", 1800))

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


class Patcher:
    """Replace the final-position hidden state emitted by block `layer` (GPT-2 block layout)."""

    def __init__(self, model, layer=0):
        self.h = model.transformer.h[layer].register_forward_hook(self._hook)
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


class Ablator:
    """Mean-ablate one whole component (the MLP, or the attention block) at the last position."""

    def __init__(self, model):
        self.spec = None
        self.mean = None
        self.rec = {}
        self.capture = False
        for L, block in enumerate(model.transformer.h):
            block.mlp.register_forward_hook(self._out_hook(L, "mlp"))
            block.attn.register_forward_hook(self._out_hook(L, "attn"))

    def _out_hook(self, L, comp):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if self.capture:
                self.rec[(L, comp)] = hs[:, -1, :].detach().float().mean(0)
            if self.spec == (L, comp):
                hs = hs.clone()
                hs[:, -1, :] = self.mean.to(hs.dtype)
                return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
            return out
        return hook


def main():
    out_path = os.path.join(RESULTS, "gpt2.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res.update(model=MODEL, frames=FRAMES)
    save = lambda: json.dump(res, open(out_path, "w"), indent=1)

    ref_ids, ref_anchor_ids = endpoint_set()                 # Pythia ids, used only for the STRINGS
    ref_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped", revision="step143000")
    anchor_strs = [ref_tok.decode([a]) for a in ref_anchor_ids]

    tok = AutoTokenizer.from_pretrained(MODEL)
    second_model.TOK = tok                                   # widths() reads the tokenizer globally
    single = lambda s: tok(s).input_ids
    ids_by_str = {s: single(s)[0] for s in ref_ids if len(single(s)) == 1}
    anchors = [single(a)[0] for a in anchor_strs]
    assert len(anchors) == len(anchor_strs) and all(len(single(a)) == 1 for a in anchor_strs)
    res.update(n_shared=len(ids_by_str), n_reference=len(ref_ids), anchors=anchor_strs,
               anchor_ids=anchors)
    print(f"shared single tokens: {len(ids_by_str)}/{len(ref_ids)}; anchors {anchor_strs}", flush=True)

    model = GPT2LMHeadModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval().cuda()
    cfg = model.config
    res["config"] = dict(n_layer=cfg.n_layer, d_model=cfg.n_embd, n_head=cfg.n_head,
                         vocab=cfg.vocab_size)
    patcher = Patcher(model)
    t0 = time.time()

    # ---- A. anchor widths at block 0, same protocol as every Pythia run ---------------------------
    if "w" not in res:
        print(f"[A] anchor widths, {len(ids_by_str)} tokens x {len(anchors)} anchors x "
              f"{len(FRAMES)} frames", flush=True)
        w, raw, oj, valid = widths(model, patcher, ids_by_str, anchors, FRAMES)
        res.update(w=w, w_raw={s: [float(x) for x in v] for s, v in raw.items()},
                   out_jsd=oj, valid_frac=valid)
        print(f"[A] done in {time.time() - t0:.0f}s: valid {valid:.3f}, median w "
              f"{np.median(list(w.values())):.3f}, sd {np.std(list(w.values())):.3f}", flush=True)
        save()
    w = res["w"]
    names = [s for s in ids_by_str if np.isfinite(w[s])]

    # ---- B. embedding probe refitted inside GPT-2 -------------------------------------------------
    E = model.transformer.wte.weight.detach().float().cpu().numpy()
    F = np.array([E[ids_by_str[s]] for s in names])
    y = np.array([w[s] for s in names])
    n_train = min(N_TRAIN, len(y) - 20)
    rho, r2, lam = probe(F, y, np.random.default_rng(1), n_train)
    rng = np.random.default_rng(0)
    rho_n, r2_n, _ = probe(F, rng.permutation(y), np.random.default_rng(1), n_train)
    res["probe"] = dict(n=len(y), n_train=n_train, rho_mean=float(rho.mean()),
                        rho_sd=float(rho.std()), r2_mean=float(r2.mean()), r2_sd=float(r2.std()),
                        median_alpha=lam, null_rho_mean=float(rho_n.mean()),
                        null_r2_mean=float(r2_n.mean()),
                        norm_rho=[float(x) for x in spearmanr(np.linalg.norm(F, axis=1), y)[:2]])
    print(f"[B] probe rho {rho.mean():+.3f} +- {rho.std():.3f} (null {rho_n.mean():+.3f}), "
          f"R2 {r2.mean():+.3f}", flush=True)
    save()

    # ---- C. which early component carries it in this architecture? -------------------------------
    toks12 = [s for s in json.load(open(f"{RESULTS}/mode_split.json"))["tokens"] if s in ids_by_str]
    small = {s: ids_by_str[s] for s in toks12}
    abl = Ablator(model)
    pre = tok(FRAMES[0], return_tensors="pt").input_ids.cuda()
    abl.capture = True
    with torch.inference_mode():
        patcher.bank = None
        model(torch.cat([pre.repeat(len(small) + len(anchors), 1),
                         torch.tensor(list(small.values()) + anchors,
                                      device=pre.device).unsqueeze(1)], 1))
    abl.capture = False
    means = {k: v.clone() for k, v in abl.rec.items()}

    base_w, _, _, _ = widths(model, patcher, small, anchors, FRAMES[:1], log_every=0)
    base = np.array([base_w[s] for s in toks12])
    base_z = {s: endpoint(model, patcher,
                          torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1))[1]
              for s, i in small.items()}
    print(f"[C] baseline (frame 0, {len(toks12)} tokens): mean {base.mean():.3f} "
          f"sd {base.std(ddof=1):.3f}", flush=True)
    res["ablate_base"] = dict(tokens=toks12, w=[float(x) for x in base])

    rows = res.get("ablate", [])
    done = {(r["block"], r["comp"]) for r in rows}
    for L in BLOCKS:
        for c in ("mlp", "attn"):
            if (L, c) in done or time.time() - t0 > DEADLINE:
                continue
            abl.spec, abl.mean = (L, c), means[(L, c)]
            wa, _, _, _ = widths(model, patcher, small, anchors, FRAMES[:1], log_every=0)
            za = {s: endpoint(model, patcher,
                              torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1))[1]
                  for s, i in small.items()}
            abl.spec = None
            v = np.array([wa[s] for s in toks12])
            ok = np.isfinite(v) & np.isfinite(base)
            bits = float(np.mean([jsd_bits(base_z[s].log_softmax(-1).unsqueeze(0),
                                           za[s].log_softmax(-1).unsqueeze(0))[0].item()
                                  for s in toks12]))
            rows.append(dict(block=L, comp=c, w=[float(x) for x in v],
                             mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)),
                             rho=float(spearmanr(base[ok], v[ok])[0]) if ok.sum() > 3 else float("nan"),
                             bits=bits, n_valid=int(ok.sum())))
            res["ablate"] = rows
            save()
            print(f"[C] block {L} {c}: mean {rows[-1]['mean']:.3f} sd {rows[-1]['sd']:.3f} "
                  f"rho {rows[-1]['rho']:+.2f} bits {bits:.3f} ({time.time() - t0:.0f}s)", flush=True)

    save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
