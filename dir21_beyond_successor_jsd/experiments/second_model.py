"""Is the per-token width trait a property of the TOKEN, or a calibration of this one network?

Everything in this direction was measured on Pythia 1.4B. The practical deliverable -- a free
static-embedding lookup that predicts a token's transition width -- silently assumes the trait belongs
to the token's representation rather than to the particular network that was probed. This runs the
cheap end of the pipeline on a second (and third) Pythia size, with the SAME token ids, anchors and
frames, so the three numbers that matter can be compared directly:

A. anchor widths w_hat_u at block 0 for the 123 endpoint tokens (3 frames x 6 anchors);
B. the embedding probe's held-out rho, refitted from THIS model's W_E to THIS model's widths;
C. cross-model rank agreement of the measured widths on the shared 123 tokens;
D. mean-ablation of the MLP and of the whole attention block in blocks 0-5, to ask whether the block-0
   MLP is again the single early component whose removal erases the across-token ordering.

Writes results/second_<tag>.json (stage-by-stage, so a partial run is still usable).
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import N_ANCHOR, run_pair
from basin_probe import REVISION, FRAMES, Patcher, endpoint, jsd_bits
from common import D18, RESULTS
from mlp_read import probe

MODELS = {"160m": "EleutherAI/pythia-160m-deduped",
          "410m": "EleutherAI/pythia-410m-deduped",
          "1b": "EleutherAI/pythia-1b-deduped"}
BLOCKS = range(6)
N_TRAIN = 80
DEADLINE = float(os.environ.get("SECOND_DEADLINE_S", 2400))

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


class Ablator:
    """Mean-ablate one whole component (the MLP, or the attention block) at the last position."""

    def __init__(self, model):
        self.spec = None            # (block, "mlp"|"attn")
        self.mean = None
        self.rec = {}
        self.capture = False
        for L, layer in enumerate(model.gpt_neox.layers):
            layer.mlp.register_forward_hook(self._mlp_hook(L))
            layer.attention.dense.register_forward_pre_hook(self._attn_hook(L))

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
            if self.spec == (L, "attn"):
                x = x.clone()
                x[:, -1, :] = self.mean.to(x.dtype)
            return (x,) + tuple(inp[1:])
        return hook


def endpoint_set():
    """The 123 endpoint tokens (string -> id) and the 6 anchor ids, exactly as in the 1.4B pipeline."""
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    used = set(ids_by_str.values())
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    anchors = pool[:: max(1, len(pool) // N_ANCHOR)][:N_ANCHOR]
    return dict(sorted(ids_by_str.items())), anchors


def widths(model, patcher, ids_by_str, anchors, frames, log_every=40):
    """Median anchor width per token, plus the raw per-(frame, anchor) values and validity rate."""
    raw = {s: [] for s in ids_by_str}
    ojs = {s: [] for s in ids_by_str}
    for fi, frame in enumerate(frames):
        pre = TOK(frame, return_tensors="pt").input_ids.cuda()
        anc = [endpoint(model, patcher,
                        torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1))
               for a in anchors]
        for k, (s, i) in enumerate(ids_by_str.items()):
            ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
            x, z = endpoint(model, patcher, ids)
            for xb, zb in anc:
                w, oj, _ = run_pair(model, patcher, ids, x, z, xb, zb)
                raw[s].append(w)
                ojs[s].append(oj)
            if log_every and k % log_every == 0:
                print(f"  frame {fi} token {k}/{len(ids_by_str)} {s!r} "
                      f"w={np.nanmedian(raw[s]):.3f}", flush=True)
    med = {s: float(np.nanmedian(v)) for s, v in raw.items()}
    valid = float(np.mean(np.isfinite(np.array([raw[s] for s in ids_by_str]))))
    return med, raw, {s: float(np.median(v)) for s, v in ojs.items()}, valid


def main():
    tag = sys.argv[1]
    name = MODELS[tag]
    out_path = os.path.join(RESULTS, f"second_{tag}.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res.update(model=name, revision=REVISION, frames=FRAMES)
    save = lambda: json.dump(res, open(out_path, "w"), indent=1)

    ids_by_str, anchors = endpoint_set()
    global TOK
    TOK = AutoTokenizer.from_pretrained(name, revision=REVISION)
    ref_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped", revision=REVISION)
    # the comparison is only meaningful if the two models read the same token ids as the same strings
    ids = list(ids_by_str.values()) + anchors
    assert TOK.convert_ids_to_tokens(ids) == ref_tok.convert_ids_to_tokens(ids)

    model = GPTNeoXForCausalLM.from_pretrained(name, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    cfg = model.config
    res["config"] = dict(n_layer=cfg.num_hidden_layers, d_model=cfg.hidden_size,
                         n_head=cfg.num_attention_heads)
    patcher = Patcher(model)
    t0 = time.time()

    # ---- A. anchor widths at block 0, same protocol as the 1.4B run -------------------------------
    if "w" not in res:
        print(f"[A] anchor widths, {len(ids_by_str)} tokens x {len(anchors)} anchors x "
              f"{len(FRAMES)} frames", flush=True)
        w, raw, oj, valid = widths(model, patcher, ids_by_str, anchors, FRAMES)
        res.update(w=w, w_raw={s: [float(x) for x in v] for s, v in raw.items()},
                   out_jsd=oj, valid_frac=valid,
                   anchors=TOK.convert_ids_to_tokens(anchors), anchor_ids=anchors)
        print(f"[A] done in {time.time() - t0:.0f}s: valid {valid:.3f}, median w "
              f"{np.median(list(w.values())):.3f}, sd {np.std(list(w.values())):.3f}", flush=True)
        save()
    w = res["w"]
    names = [s for s in ids_by_str if np.isfinite(w[s])]

    # ---- C. cross-model rank agreement on the shared tokens ---------------------------------------
    ref = {s: float(np.nanmedian(v["w"]))
           for s, v in json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}
    shared = [s for s in names if np.isfinite(ref.get(s, np.nan))]
    a = np.array([w[s] for s in shared])
    b = np.array([ref[s] for s in shared])
    res["cross_model"] = dict(n=len(shared), rho=[float(x) for x in spearmanr(a, b)[:2]],
                              r=float(np.corrcoef(a, b)[0, 1]),
                              median_this=float(np.median(a)), sd_this=float(a.std(ddof=1)),
                              median_ref=float(np.median(b)), sd_ref=float(b.std(ddof=1)))
    print(f"[C] cross-model rho vs 1.4B = {res['cross_model']['rho'][0]:+.3f} "
          f"(p {res['cross_model']['rho'][1]:.2g}, n {len(shared)})", flush=True)
    save()

    # ---- B. embedding probe, refitted inside this model -------------------------------------------
    E = model.gpt_neox.embed_in.weight.detach().float().cpu().numpy()
    F = np.array([E[ids_by_str[s]] for s in names])
    y = np.array([w[s] for s in names])
    n_train = min(N_TRAIN, len(y) - 20)
    rho, r2, lam = probe(F, y, np.random.default_rng(1), n_train)
    rng = np.random.default_rng(0)
    rho_n, r2_n, _ = probe(F, rng.permutation(y), np.random.default_rng(1), n_train)
    res["probe"] = dict(n=len(y), n_train=n_train, rho_mean=float(rho.mean()),
                        rho_sd=float(rho.std()), r2_mean=float(r2.mean()),
                        r2_sd=float(r2.std()), median_alpha=lam,
                        null_rho_mean=float(rho_n.mean()), null_r2_mean=float(r2_n.mean()),
                        norm_rho=[float(x) for x in spearmanr(np.linalg.norm(F, axis=1), y)[:2]])
    print(f"[B] probe rho {rho.mean():+.3f} +- {rho.std():.3f} (null {rho_n.mean():+.3f}), "
          f"R2 {r2.mean():+.3f}", flush=True)
    save()

    # ---- D. which early component carries it here? ------------------------------------------------
    toks12 = list(json.load(open(f"{RESULTS}/mode_split.json"))["tokens"].keys())
    small = {s: ids_by_str[s] for s in toks12}
    abl = Ablator(model)
    pre = TOK(FRAMES[0], return_tensors="pt").input_ids.cuda()
    abl.capture = True
    with torch.inference_mode():
        patcher.bank = None
        model(torch.cat([pre.repeat(len(small) + len(anchors), 1),
                         torch.tensor(list(small.values()) + anchors,
                                      device=pre.device).unsqueeze(1)], 1))
    abl.capture = False
    means = {k: v.clone() for k, v in abl.rec.items()}

    base_w, base_raw, _, _ = widths(model, patcher, small, anchors, FRAMES[:1], log_every=0)
    base = np.array([base_w[s] for s in toks12])
    base_z = {}
    for s, i in small.items():
        base_z[s] = endpoint(model, patcher,
                             torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1))[1]
    print(f"[D] baseline (frame 0, 12 tokens): mean {base.mean():.3f} sd {base.std(ddof=1):.3f}",
          flush=True)
    res["ablate_base"] = dict(tokens=toks12, w=[float(x) for x in base])

    rows = res.get("ablate", [])
    done = {(r["block"], r["comp"]) for r in rows}
    for L in BLOCKS:
        for c in ("mlp", "attn"):
            if (L, c) in done or L >= cfg.num_hidden_layers:
                continue
            if time.time() - t0 > DEADLINE:
                print("[D] deadline reached", flush=True)
                break
            abl.spec, abl.mean = (L, c), means[(L, c)]
            wa, _, _, _ = widths(model, patcher, small, anchors, FRAMES[:1], log_every=0)
            za = {}
            for s, i in small.items():
                za[s] = endpoint(model, patcher,
                                 torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1))[1]
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
            print(f"[D] block {L} {c}: mean {rows[-1]['mean']:.3f} sd {rows[-1]['sd']:.3f} "
                  f"rho {rows[-1]['rho']:+.2f} bits {bits:.3f} ({time.time() - t0:.0f}s)", flush=True)

    save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
