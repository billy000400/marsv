"""Does the second model's block-0 MLP localisation survive a movement-matched control?

The 410M ablation sweep has the same confound the 1.4B one had: the block-0 MLP is the only early
component whose removal the model feels (0.44 bits against <= 0.02 for every other), and any large
disturbance flattens the token ordering. `dose2.py` settled that on 1.4B by softening the ablation and
matching each dose to a random perturbation of the SAME residual stream that moves EACH PROMPT's
output by the same number of bits. This reruns exactly that protocol on the second model, reusing
dose2's dose, per-prompt binary search and measurement code unchanged.

Writes results/second_ctrl_<tag>.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from basin_probe import REVISION, FRAMES, Patcher, endpoint, jsd_bits
from common import D18, RESULTS
from dose2 import Dose, bits_per_prompt, measure, search_scales
from second_model import MODELS, endpoint_set

FRAME = FRAMES[0]
ALPHAS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def main():
    tag = sys.argv[1]
    name = MODELS[tag]
    ids_by_str, anchors = endpoint_set()
    toks12 = list(json.load(open(f"{RESULTS}/mode_split.json"))["tokens"].keys())
    ids_by_tok = {s: ids_by_str[s] for s in toks12}

    tok = AutoTokenizer.from_pretrained(name, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(name, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    pre = tok(FRAME, return_tensors="pt").input_ids.cuda()
    patcher = Patcher(model)
    dose = Dose(model)

    keys = list(toks12) + [("anchor", a) for a in anchors]
    ids_all = torch.cat([pre.repeat(len(keys), 1),
                         torch.tensor(list(ids_by_tok.values()) + anchors,
                                      device=pre.device).unsqueeze(1)], 1)
    dose.capture = True
    with torch.inference_mode():
        patcher.bank = None
        model(ids_all)
    dose.capture = False
    dose.mean = dose.rec.clone()
    with torch.inference_mode():
        patcher.bank = None
        base_lp = model(ids_all).logits[:, -1, :].float().log_softmax(-1)

    t0 = time.time()
    base_w = measure(model, patcher, dose, ids_by_tok, anchors, pre, None)
    base = np.array([base_w[s] for s in toks12])
    print(f"baseline mean {base.mean():.3f} sd {base.std(ddof=1):.3f}", flush=True)

    def score(tag_, w, bits_tok):
        v = np.array([w[s] for s in toks12])
        ok = np.isfinite(v) & np.isfinite(base)
        rho = float(spearmanr(base[ok], v[ok]).statistic) if ok.sum() > 3 else float("nan")
        row = dict(arm=tag_, rho=rho, mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)),
                   w=[float(x) for x in v], bits=float(np.mean(bits_tok)),
                   bits_tok=[float(b) for b in bits_tok], n_valid=int(ok.sum()))
        print(f"  {tag_}: bits {row['bits']:.4f} rho {rho:+.2f} mean {row['mean']:.3f} "
              f"sd {row['sd']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        return row

    rows = []
    for a in ALPHAS:
        print(f"alpha {a}", flush=True)
        dose.mode, dose.alpha = "mlp", a
        b_mlp = bits_per_prompt(model, patcher, ids_all, base_lp)
        dose.mode = None
        w_mlp = measure(model, patcher, dose, ids_by_tok, anchors, pre, "mlp", alpha=a)
        r = score(f"mlp a={a}", w_mlp, b_mlp[:len(toks12)].tolist())
        r.update(alpha=a, seed=None)
        rows.append(r)
        for seed in SEEDS:
            g = torch.Generator(device="cuda").manual_seed(seed)
            rr = torch.randn(model.config.hidden_size, generator=g, device="cuda")
            dose.r = rr / rr.norm()
            c, ach = search_scales(model, patcher, dose, ids_all, base_lp, b_mlp)
            cmap = {k: float(c[j]) for j, k in enumerate(keys)}
            w_c = measure(model, patcher, dose, ids_by_tok, anchors, pre, "ctrl", cmap=cmap)
            r = score(f"ctrl seed={seed} a={a}", w_c, ach[:len(toks12)].tolist())
            r.update(alpha=a, seed=seed)
            rows.append(r)
        json.dump(dict(model=name, revision=REVISION, frame=FRAME, tokens=toks12,
                       base_w=[float(x) for x in base], alphas=ALPHAS, seeds=SEEDS, rows=rows),
                  open(f"{RESULTS}/second_ctrl_{tag}.json", "w"), indent=1)

    # paired per-token test: does the dose move a token's width further than ITS OWN matched control?
    tests = []
    for a in ALPHAS:
        m = [r for r in rows if r["seed"] is None and r["alpha"] == a][0]
        cs = [r for r in rows if r["seed"] is not None and r["alpha"] == a]
        dm = np.abs(np.array(m["w"]) - base)
        dc = np.mean([np.abs(np.array(r["w"]) - base) for r in cs], axis=0)
        # level-free version: remove each arm's mean width shift before taking the distance
        em = np.abs((np.array(m["w"]) - base) - (np.array(m["w"]) - base).mean())
        ec = np.mean([np.abs((np.array(r["w"]) - base) - (np.array(r["w"]) - base).mean())
                      for r in cs], axis=0)
        tests.append(dict(alpha=a, bits=m["bits"], mlp_move=float(dm.mean()),
                          ctrl_move=float(dc.mean()),
                          p=float(wilcoxon(dm, dc).pvalue),
                          mlp_move_centred=float(em.mean()), ctrl_move_centred=float(ec.mean()),
                          p_centred=float(wilcoxon(em, ec).pvalue)))
        print(f"alpha {a}: |dw| mlp {dm.mean():.3f} vs ctrl {dc.mean():.3f} "
              f"(p {tests[-1]['p']:.4f}); centred {em.mean():.3f} vs {ec.mean():.3f} "
              f"(p {tests[-1]['p_centred']:.4f})", flush=True)

    json.dump(dict(model=name, revision=REVISION, frame=FRAME, tokens=toks12,
                   base_w=[float(x) for x in base], alphas=ALPHAS, seeds=SEEDS,
                   rows=rows, paired=tests),
              open(f"{RESULTS}/second_ctrl_{tag}.json", "w"), indent=1)
    print(f"wrote results/second_ctrl_{tag}.json")


if __name__ == "__main__":
    main()
