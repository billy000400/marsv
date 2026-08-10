"""Does a token's fitted width contribution transfer to partners it was never paired with?

For each of the 123 endpoint tokens we run dir18's interpolation protocol against 6 ANCHOR tokens
that appear in none of the 1,000 pairs, and summarise the token by its median width against those
anchors. That gives a per-token number measured entirely outside the pair bank, which can be tested
against the token effects a_u fitted inside it. Writes results/anchor_width.json.
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

sys.path.append("/workspace/marsv_agent_haoyang/dir18_continuation_jsd_plateau/experiments")
import curve_metrics                                            # dir18's strict validity criteria

from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint, jsd_bits
from common import D18, RESULTS

GRID = np.linspace(0.0, 1.0, 50)
N_ANCHOR = 6
BATCH = 50

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def slerp_bank(xa, xb):
    """Norm-rescaled SLERP, identical to dir18's."""
    na, nb = xa.norm(), xb.norm()
    ua, ub = xa / na, xb / nb
    cos = torch.clamp((ua * ub).sum(), -1.0, 1.0)
    om = torch.arccos(cos)
    t = torch.tensor(GRID, device=xa.device, dtype=xa.dtype).unsqueeze(1)
    u = (torch.sin((1 - t) * om) * ua + torch.sin(t * om) * ub) / torch.sin(om)
    return ((1 - t) * na + t * nb) * u, float(cos)


@torch.inference_mode()
def run_pair(model, patcher, ids_a, xa, za, xb, zb):
    """d(t) curve and width for one (token, anchor) pair in one frame."""
    bank, cos = slerp_bank(xa, xb)
    z = []
    for s in range(0, len(GRID), BATCH):
        chunk = bank[s:s + BATCH]
        patcher.bank = chunk
        z.append(model(ids_a.repeat(len(chunk), 1)).logits[:, -1, :].float())
    patcher.bank = None
    z = torch.cat(z)
    da = (z - za).norm(dim=1)
    db = (z - zb).norm(dim=1)
    d = (da / (da + db)).cpu().numpy()
    m = curve_metrics.metrics(d, GRID)
    out_jsd = float(jsd_bits(za.log_softmax(-1).unsqueeze(0), zb.log_softmax(-1).unsqueeze(0))[0])
    return m["w"], out_jsd, cos


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

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    patcher = Patcher(model)

    rows = {s: dict(token_id=i, w=[], out_jsd=[], cos=[]) for s, i in endpoints}
    for fi, frame in enumerate(FRAMES):
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        anc = []
        for a in anchors:
            ids = torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1)
            anc.append(endpoint(model, patcher, ids))
        for k, (s, i) in enumerate(endpoints):
            ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
            x, z = endpoint(model, patcher, ids)
            for xb, zb in anc:
                w, oj, cos = run_pair(model, patcher, ids, x, z, xb, zb)
                rows[s]["w"].append(w)
                rows[s]["out_jsd"].append(oj)
                rows[s]["cos"].append(cos)
            if k % 40 == 0:
                print(f"frame {fi} token {k}/{len(endpoints)} {s!r} "
                      f"median w vs anchors = {np.nanmedian(rows[s]['w']):.3f}", flush=True)

    json.dump(dict(model=MODEL, revision=REVISION, frames=FRAMES, grid=list(GRID),
                   anchors=[tok.convert_ids_to_tokens(a) for a in anchors],
                   anchor_ids=anchors, tokens=rows),
              open(os.path.join(RESULTS, "anchor_width.json"), "w"), indent=1)
    print("wrote results/anchor_width.json")


if __name__ == "__main__":
    main()
