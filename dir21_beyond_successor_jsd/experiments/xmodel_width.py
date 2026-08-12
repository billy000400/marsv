"""Anchor widths for the same 123 token strings in models with DIFFERENT tokenizers and corpora.

Usage: python3 xmodel_width.py {gpt2|1.4b|410m}

Everything measured in this direction so far comes from Pythia (the Pile, parallel attention+MLP
residual). GPT-2 small is the cheapest test outside that family: a different training corpus
(WebText), a different BPE vocabulary and a serial residual block -- and, conveniently, all 123
endpoint token strings and all 6 anchor strings are single tokens in its vocabulary too, so the SAME
strings, anchors and frames transfer with no substitution.

Records BOTH width statistics per measurement (dir18's strict w and the envelope width of
envwidth.py) so the substitution can be validated inside Pythia, where both are defined. The Pythia
runs here re-measure what results/anchor_width.json already holds, because that file stores only the
strict w and the cross-model comparison needs the envelope statistic on the same curves.

Also refits the embedding probe inside each model (its own W_E -> its own envelope widths), and for
GPT-2 mean-ablates the MLP and attention block of blocks 0-5 to ask whether the block-0 MLP is again
the single early component whose removal erases the ordering.

Writes results/xwidth_<tag>.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPT2LMHeadModel, GPTNeoXForCausalLM

from basin_probe import REVISION, FRAMES, Patcher as NeoXPatcher, jsd_bits
from common import RESULTS
from envwidth import token_widths
from gpt2_model import Ablator as GPT2Ablator, Patcher as GPT2Patcher
from mlp_read import probe
from second_model import Ablator as NeoXAblator, endpoint_set

MODELS = {"gpt2": "gpt2", "410m": "EleutherAI/pythia-410m-deduped",
          "1.4b": "EleutherAI/pythia-1.4b-deduped"}
N_TRAIN = 80

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def load(tag):
    """Model, tokenizer, block-0 patcher, embedding matrix, ablator factory."""
    name = MODELS[tag]
    if tag == "gpt2":
        tok = AutoTokenizer.from_pretrained(name)
        model = GPT2LMHeadModel.from_pretrained(name, torch_dtype=torch.float32).eval().cuda()
        return (model, tok, GPT2Patcher(model),
                model.transformer.wte.weight.detach().float().cpu().numpy(), GPT2Ablator)
    tok = AutoTokenizer.from_pretrained(name, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(name, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    return (model, tok, NeoXPatcher(model),
            model.gpt_neox.embed_in.weight.detach().float().cpu().numpy(), NeoXAblator)


def main():
    tag = sys.argv[1]
    ablate = len(sys.argv) > 2 and sys.argv[2] == "ablate"
    out_path = os.path.join(RESULTS, f"xwidth_{tag}.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res.update(model=MODELS[tag], tag=tag, frames=FRAMES)
    save = lambda: json.dump(res, open(out_path, "w"), indent=1)

    ref_ids, ref_anchor_ids = endpoint_set()
    ref_tok = AutoTokenizer.from_pretrained(MODELS["1.4b"], revision=REVISION)
    anchor_strs = [ref_tok.decode([a]) for a in ref_anchor_ids]

    model, tok, patcher, E, Ablator = load(tag)
    one = lambda s: tok(s).input_ids
    ids_by_str = {s: one(s)[0] for s in ref_ids if len(one(s)) == 1}
    anchors = [one(a)[0] for a in anchor_strs]
    assert all(len(one(a)) == 1 for a in anchor_strs)
    res.update(n_shared=len(ids_by_str), n_reference=len(ref_ids), anchors=anchor_strs)
    print(f"[{tag}] {len(ids_by_str)}/{len(ref_ids)} token strings are single tokens here", flush=True)
    t0 = time.time()

    if "raw" not in res:
        raw = token_widths(model, patcher, tok, ids_by_str, anchors, FRAMES)
        res["raw"] = raw
        wv = np.array([np.median(raw[s]["w_env"]) for s in ids_by_str])
        res["valid_frac"] = float(np.mean([np.mean(raw[s]["valid"]) for s in ids_by_str]))
        print(f"[A] {time.time() - t0:.0f}s: median w_env {np.median(wv):.3f}, sd {wv.std(ddof=1):.3f},"
              f" strict-valid fraction {res['valid_frac']:.3f}", flush=True)
        save()
    raw = res["raw"]

    # embedding probe refitted inside this model, against its own envelope widths
    names = list(ids_by_str)
    F = np.array([E[ids_by_str[s]] for s in names])
    y = np.array([np.median(raw[s]["w_env"]) for s in names])
    n_train = min(N_TRAIN, len(y) - 20)
    rho, r2, lam = probe(F, y, np.random.default_rng(1), n_train)
    rho_n, _, _ = probe(F, np.random.default_rng(0).permutation(y), np.random.default_rng(1), n_train)
    res["probe"] = dict(n=len(y), n_train=n_train, rho_mean=float(rho.mean()),
                        rho_sd=float(rho.std()), r2_mean=float(r2.mean()), r2_sd=float(r2.std()),
                        median_alpha=lam, null_rho_mean=float(rho_n.mean()),
                        norm_rho=[float(x) for x in spearmanr(np.linalg.norm(F, axis=1), y)[:2]])
    print(f"[B] probe rho {rho.mean():+.3f} +- {rho.std():.3f} (shuffled {rho_n.mean():+.3f}), "
          f"R2 {r2.mean():+.3f}", flush=True)
    save()

    if not ablate:
        print(f"wrote {out_path}")
        return

    # which early component carries the ordering in THIS architecture?
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

    def measure():
        r = token_widths(model, patcher, tok, small, anchors, FRAMES[:1], log_every=0)
        w = np.array([np.median(r[s]["w_env"]) for s in toks12])
        z = {}
        with torch.inference_mode():
            for s, i in small.items():
                patcher.bank = None
                z[s] = model(torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
                             ).logits[0, -1].float()
        return w, z

    base, base_z = measure()
    res["ablate_base"] = dict(tokens=toks12, w=[float(x) for x in base])
    print(f"[C] baseline: mean {base.mean():.3f} sd {base.std(ddof=1):.3f}", flush=True)

    rows = []
    for L in range(6):
        for c in ("mlp", "attn"):
            abl.spec, abl.mean = (L, c), means[(L, c)]
            v, za = measure()
            abl.spec = None
            bits = float(np.mean([jsd_bits(base_z[s].log_softmax(-1).unsqueeze(0),
                                           za[s].log_softmax(-1).unsqueeze(0))[0].item()
                                  for s in toks12]))
            rows.append(dict(block=L, comp=c, w=[float(x) for x in v], mean=float(v.mean()),
                             sd=float(v.std(ddof=1)), rho=float(spearmanr(base, v).statistic),
                             bits=bits))
            res["ablate"] = rows
            save()
            print(f"[C] block {L} {c}: mean {rows[-1]['mean']:.3f} sd {rows[-1]['sd']:.3f} "
                  f"rho {rows[-1]['rho']:+.2f} bits {bits:.3f}", flush=True)

    save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
