"""Does GPT-2 have a plateau to measure at all, or plateaus in a different token order?

The cross-model run found no agreement between GPT-2's and Pythia's per-token widths. Two very
different things can produce that. Either GPT-2's interpolation curves are not plateau-shaped at the
measured site -- if d(t) rises roughly linearly there is no plateau, and a "width" measured on it is
close to 1 by construction and carries no information -- or the curves ARE plateau-shaped and GPT-2
simply orders tokens differently.

Edge drift separates the two with no new machinery: E = d(0.1) + (1 - d(0.9)), the total movement of
the curve over the outer tenth of the path at each end. A plateau leaves the endpoints alone (E ~ 0);
a straight line in output space has d(t) = t and E = 0.2 exactly.

Measures E on every curve (123 tokens x 6 anchors x 3 frames) for GPT-2 at blocks 0, 4 and 8 and for
Pythia-160M / 410M / 1.4B at block 0. Pythia-160M is the prediction test: it is the one Pythia that
does NOT carry the width ordering, so if the failure is about missing plateau structure it should look
like GPT-2 here, and if it is about a different ordering it should look like the larger Pythias.

Writes results/edgedrift.json.
"""
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPT2LMHeadModel, GPTNeoXForCausalLM

from basin_probe import REVISION, FRAMES, Patcher as NeoXPatcher
from common import RESULTS
from envwidth import token_widths
from gpt2_model import Patcher as GPT2Patcher
from second_model import endpoint_set

CONFIGS = [("gpt2", "gpt2", 0), ("gpt2", "gpt2", 4), ("gpt2", "gpt2", 8),
           ("160m", "EleutherAI/pythia-160m-deduped", 0),
           ("410m", "EleutherAI/pythia-410m-deduped", 0),
           ("1.4b", "EleutherAI/pythia-1.4b-deduped", 0)]
LINE_E = 0.2  # d(t) = t, the straight-line reference

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def load(name, layer):
    if name == "gpt2":
        tok = AutoTokenizer.from_pretrained(name)
        model = GPT2LMHeadModel.from_pretrained(name, torch_dtype=torch.float32).eval().cuda()
        return model, tok, GPT2Patcher(model, layer=layer)
    tok = AutoTokenizer.from_pretrained(name, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(name, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    return model, tok, NeoXPatcher(model, layer=layer)


def main():
    out_path = os.path.join(RESULTS, "edgedrift.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {"rows": {}}
    save = lambda: json.dump(res, open(out_path, "w"), indent=1)

    ref_ids, ref_anchor_ids = endpoint_set()
    ref_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped", revision=REVISION)
    anchor_strs = [ref_tok.decode([a]) for a in ref_anchor_ids]
    res.update(frames=FRAMES, anchors=anchor_strs, line_reference=LINE_E)

    for tag, name, layer in CONFIGS:
        key = f"{tag}_block{layer}"
        if key in res["rows"]:
            continue
        model, tok, patcher = load(name, layer)
        one = lambda s: tok(s).input_ids
        ids_by_str = {s: one(s)[0] for s in ref_ids if len(one(s)) == 1}
        anchors = [one(a)[0] for a in anchor_strs]
        raw = token_widths(model, patcher, tok, ids_by_str, anchors, FRAMES, log_every=0)
        names = sorted(ids_by_str)
        E = np.array([raw[s]["edge"] for s in names])          # tokens x 18 curves
        W = np.array([raw[s]["w_env"] for s in names])
        res["rows"][key] = dict(
            model=name, tag=tag, layer=layer, n_tokens=len(names), tokens=names,
            edge_median=float(np.median(E)), edge_q=[float(q) for q in np.percentile(E, [10, 90])],
            edge_frac_above_half_line=float(np.mean(E > 0.5 * LINE_E)),
            edge_by_token=[float(x) for x in np.median(E, axis=1)],
            w_by_token=[float(x) for x in np.median(W, axis=1)],
            w_median=float(np.median(W)),
            rho_edge_w=[float(x) for x in spearmanr(np.median(E, axis=1), np.median(W, axis=1))[:2]],
            edge_curves=[float(x) for x in E.ravel()],
            w_curves=[float(x) for x in W.ravel()])
        r = res["rows"][key]
        print(f"[{key}] median E {r['edge_median']:.4f} (10-90% {r['edge_q'][0]:.4f}-"
              f"{r['edge_q'][1]:.4f}), {r['edge_frac_above_half_line']:.3f} of curves above "
              f"{0.5 * LINE_E:.2f}, median w_env {r['w_median']:.3f}, "
              f"rho(E, w) {r['rho_edge_w'][0]:+.3f}", flush=True)
        save()
        patcher.h.remove()
        del model
        torch.cuda.empty_cache()

    save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
