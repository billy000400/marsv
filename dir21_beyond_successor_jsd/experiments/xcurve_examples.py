"""Example d(t) curves for the same (token, anchor, frame) in Pythia-1.4B and GPT-2.

The cross-model comparison rests on a claim about curve SHAPE -- that Pythia's interpolation curves
rise monotonically and GPT-2's wander -- which a scatter of summary statistics cannot show. This saves
the raw curves for four tokens against one anchor in both models. Writes results/xcurves.json.
"""
import json

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPTNeoXForCausalLM

from anchor_width import GRID, BATCH, slerp_bank
from basin_probe import REVISION, FRAMES, Patcher as NeoXPatcher
from common import RESULTS
from envwidth import env_metrics
from gpt2_model import Patcher as GPT2Patcher
from second_model import endpoint_set

TOKENS = [" wrong", " fun", " over", " about"]
ANCHOR = " close"

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


@torch.inference_mode()
def curves(model, tok, patcher):
    pre = tok(FRAMES[0], return_tensors="pt").input_ids.cuda()
    def state(s):
        ids = torch.cat([pre, torch.tensor([[tok(s).input_ids[0]]], device=pre.device)], 1)
        patcher.bank = None
        z = model(ids).logits[0, -1].float()
        return ids, patcher.captured[0].clone().float(), z
    _, xb, zb = state(ANCHOR)
    out = {}
    for s in TOKENS:
        ids, x, z = state(s)
        bank, _ = slerp_bank(x, xb)
        zs = []
        for st in range(0, len(GRID), BATCH):
            patcher.bank = bank[st:st + BATCH]
            zs.append(model(ids.repeat(len(patcher.bank), 1)).logits[:, -1, :].float())
        patcher.bank = None
        zz = torch.cat(zs)
        da = (zz - z).norm(dim=1)
        db = (zz - zb).norm(dim=1)
        d = (da / (da + db)).cpu().numpy()
        m = env_metrics(d)
        out[s] = dict(d=[float(v) for v in d], **{k: (float(v) if isinstance(v, float) else v)
                                                  for k, v in m.items()})
        print(f"  {s!r}: w={m['w']:.3f} w_env={m['w_env']:.3f} backslide={m['backslide']:.3f}",
              flush=True)
    return out


def main():
    res = dict(grid=[float(g) for g in GRID], tokens=TOKENS, anchor=ANCHOR, frame=FRAMES[0])

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped", revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-deduped", revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    print("pythia-1.4b:", flush=True)
    res["pythia"] = curves(model, tok, NeoXPatcher(model))
    del model
    torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32).eval().cuda()
    print("gpt2:", flush=True)
    res["gpt2"] = curves(model, tok, GPT2Patcher(model))

    json.dump(res, open(f"{RESULTS}/xcurves.json", "w"), indent=1)
    print("wrote results/xcurves.json")


if __name__ == "__main__":
    main()
