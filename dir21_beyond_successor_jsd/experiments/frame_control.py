"""Is w_hat_u a property of the token, or of the token in this one slot?

Every measurement in this direction uses three short declarative frames ending in `was`, with the
token in final position. Here the anchor-width measurement is repeated for the same 123 tokens and the
same six anchors in four structurally different contexts, and each context's token ranking is compared
with the original one — and with the agreement among the three original frames, which is the ceiling
this comparison should be read against.

Writes results/frames.json.
"""
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import N_ANCHOR, run_pair
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint
from common import D18, RESULTS

NEW_FRAMES = {
    "mid-sentence": "She kept walking because everything felt",
    "question": "Is it really",
    "list": "The report mentions the following:",
    "code": "def solve(x):\n    return",
}

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def main():
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    endpoints = sorted(ids_by_str.items())
    used = set(ids_by_str.values())
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    anchors = pool[:: max(1, len(pool) // N_ANCHOR)][:N_ANCHOR]
    assert len(anchors) == N_ANCHOR

    # original measurement, and its three frames separately (w is stored frame-major, 6 anchors each)
    orig = json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"]
    names = [s for s, _ in endpoints]
    w_orig = np.array([np.nanmedian(orig[s]["w"]) for s in names])
    w_orig_by_frame = np.array([[np.nanmedian(orig[s]["w"][6 * f:6 * f + 6]) for s in names]
                                for f in range(len(FRAMES))])

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    patcher = Patcher(model)

    res = {"anchors": [tok.convert_ids_to_tokens(a) for a in anchors],
           "orig_frames": FRAMES, "new_frames": NEW_FRAMES, "tokens": names,
           "w_orig": [float(v) for v in w_orig], "by_frame": {}}
    for tag, frame in NEW_FRAMES.items():
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        anc = [endpoint(model, patcher,
                        torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1))
               for a in anchors]
        w = []
        for k, (s, i) in enumerate(endpoints):
            ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
            x, z = endpoint(model, patcher, ids)
            w.append([run_pair(model, patcher, ids, x, z, xb, zb)[0] for xb, zb in anc])
            if k % 60 == 0:
                print(f"{tag}: token {k}/{len(endpoints)}", flush=True)
        w = np.array(w, dtype=float)
        med = np.nanmedian(w, axis=1)
        r = spearmanr(med, w_orig)
        res["by_frame"][tag] = dict(
            frame=frame, valid=float(np.mean(~np.isnan(w))),
            w={s: float(v) for s, v in zip(names, med)},
            median=float(np.nanmedian(med)),
            iqr=float(np.subtract(*np.nanpercentile(med, [75, 25]))),
            rho_vs_orig=[float(x) for x in r[:2]])
        print(f"{tag}: valid {res['by_frame'][tag]['valid']:.3f}, "
              f"median w {res['by_frame'][tag]['median']:.3f}, "
              f"rho vs original {r[0]:+.3f} (p={r[1]:.1e})", flush=True)
    patcher.close()

    # reference ceiling: how well do the three ORIGINAL frames agree with each other?
    pairs = [(0, 1), (0, 2), (1, 2)]
    within = [float(spearmanr(w_orig_by_frame[a], w_orig_by_frame[b])[0]) for a, b in pairs]
    tags = list(NEW_FRAMES)
    across_new = [float(spearmanr(np.array([res["by_frame"][a]["w"][s] for s in names]),
                                  np.array([res["by_frame"][b]["w"][s] for s in names]))[0])
                  for i, a in enumerate(tags) for b in tags[i + 1:]]
    res["within_orig_rho"] = within
    res["within_orig_mean"] = float(np.mean(within))
    res["across_new_rho_mean"] = float(np.mean(across_new))
    res["orig_median"] = float(np.nanmedian(w_orig))
    json.dump(res, open(os.path.join(RESULTS, "frames.json"), "w"), indent=1)
    print(f"agreement among the three original frames: {['%.3f' % v for v in within]} "
          f"(mean {np.mean(within):+.3f})")
    print(f"mean agreement among the four new contexts: {np.mean(across_new):+.3f}")
    print("wrote results/frames.json")


if __name__ == "__main__":
    main()
