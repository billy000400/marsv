"""Is GPT-2's failure to reproduce the width ordering an artifact of WHERE the path is drawn?

The cross-model run interpolates at the output of block 0, the site every Pythia measurement uses.
GPT-2 small has 12 blocks against Pythia-1.4B's 24, so block 0 sits at a different relative depth, and
GPT-2's curves there are non-monotone. Before concluding that the trait is absent, repeat the
measurement at several sites and ask whether any of them gives curves as well-behaved as Pythia's,
a per-token measurement as reliable, or a ranking that agrees with Pythia's.

Writes results/gpt2_sites.json.
"""
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPT2LMHeadModel

from basin_probe import REVISION, FRAMES
from common import RESULTS
from envwidth import token_widths
from gpt2_model import Patcher
from second_model import endpoint_set
from xmodel_analysis import reliability

SITES = [0, 1, 2, 4, 6, 8]

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def main():
    ref_ids, ref_anchor_ids = endpoint_set()
    ref_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped", revision=REVISION)
    anchor_strs = [ref_tok.decode([a]) for a in ref_anchor_ids]

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32).eval().cuda()
    one = lambda s: tok(s).input_ids
    ids_by_str = {s: one(s)[0] for s in ref_ids if len(one(s)) == 1}
    anchors = [one(a)[0] for a in anchor_strs]

    ref = json.load(open(f"{RESULTS}/xwidth_1.4b.json"))["raw"]
    names = sorted(ids_by_str)
    w14 = np.array([np.median(np.array(ref[s]["w_env"], float)) for s in names])

    rows = []
    for L in SITES:
        p = Patcher(model, layer=L)
        raw = token_widths(model, patcher=p, tok=tok, ids_by_str=ids_by_str, anchors=anchors,
                           frames=FRAMES, log_every=0)
        p.h.remove()
        w = np.array([np.median(np.array(raw[s]["w_env"], float)) for s in names])
        rel = reliability(raw, names, "w_env")
        r = spearmanr(w, w14)
        rows.append(dict(layer=L, median_w=float(np.median(w)), sd_w=float(w.std(ddof=1)),
                         valid=float(np.mean([np.mean(raw[s]["valid"]) for s in names])),
                         backslide=float(np.median([np.median(raw[s]["backslide"]) for s in names])),
                         half_rho=rel[0], reliability=rel[1],
                         rho_pythia=float(r.statistic), p_pythia=float(r.pvalue)))
        print(f"block {L}: median w_env {rows[-1]['median_w']:.3f} sd {rows[-1]['sd_w']:.3f} | "
              f"strict-valid {rows[-1]['valid']:.3f} backslide {rows[-1]['backslide']:.3f} | "
              f"reliability {rows[-1]['reliability']:+.3f} | vs Pythia-1.4B "
              f"{rows[-1]['rho_pythia']:+.3f} (p {rows[-1]['p_pythia']:.1e})", flush=True)

    json.dump(dict(model="gpt2", sites=SITES, n_tokens=len(names), frames=FRAMES, rows=rows),
              open(f"{RESULTS}/gpt2_sites.json", "w"), indent=1)
    print("wrote results/gpt2_sites.json")


if __name__ == "__main__":
    main()
