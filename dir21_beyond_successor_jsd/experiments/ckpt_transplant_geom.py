"""Do the two checkpoints' block-0 MLP outputs live in the same coordinate system?

The transplant writes step143000's m_u into the step128 model. If that fails, there are two very
different reasons, and only a direct measurement separates them: the vector may not carry the trait,
or the two networks may simply not read the same coordinates, in which case the write is a large
foreign vector and the experiment could not have succeeded whatever m_u encodes.

Three measurements on the same 123 tokens and 3 frames:
  cosine        cos(m_u^final, m_u^early) per token, raw and after subtracting the across-token mean
                (the mean is the part shared by every token; what identifies a token is the deviation);
  norm          rank agreement of ||m_u|| across the two checkpoints;
  geometry      rank agreement of the 7,503 pairwise cosines between tokens -- if the two checkpoints
                arrange the SAME tokens the same way relative to each other, this is high even when the
                absolute coordinates have rotated.

Writes results/ckpt_transplant_geom.json.
"""
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from basin_probe import FRAMES, Patcher
from ckpt_transplant import EARLY, FINAL, MODEL, capture_m
from common import RESULTS
from mlp_read import MLPOut
from second_model import endpoint_set

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def main():
    ids_by_str, _ = endpoint_set()
    names = list(ids_by_str)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=FINAL)

    M = {}
    for rev in (FINAL, EARLY):
        model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=rev,
                                                   torch_dtype=torch.float32).eval().cuda()
        patcher, mlp = Patcher(model), MLPOut(model)
        m = capture_m(model, mlp, patcher, tok, ids_by_str)
        M[rev] = np.stack([np.stack([m[f][s].numpy() for s in names]) for f in range(len(FRAMES))])
        patcher.close()
        mlp.h.remove()
        del model
        torch.cuda.empty_cache()
        print(f"[{rev}] m shape {M[rev].shape}", flush=True)

    A, B = M[FINAL].mean(0), M[EARLY].mean(0)          # (123, d), averaged over the 3 frames
    unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    cos = float(np.mean(np.sum(unit(A) * unit(B), 1)))
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    cos_c = float(np.mean(np.sum(unit(Ac) * unit(Bc), 1)))
    iu = np.triu_indices(len(names), 1)
    G = lambda X: (unit(X) @ unit(X).T)[iu]
    out = dict(model=MODEL, early=EARLY, final=FINAL, n=len(names), tokens=names,
               norm_final=[float(x) for x in np.linalg.norm(A, axis=1)],
               norm_early=[float(x) for x in np.linalg.norm(B, axis=1)],
               mean_cosine=cos, mean_cosine_centred=cos_c,
               norm_rho=float(spearmanr(np.linalg.norm(A, axis=1),
                                        np.linalg.norm(B, axis=1)).statistic),
               geometry_rho=float(spearmanr(G(A), G(B)).statistic),
               geometry_rho_centred=float(spearmanr(G(Ac), G(Bc)).statistic),
               mean_norm={FINAL: float(np.linalg.norm(A, axis=1).mean()),
                          EARLY: float(np.linalg.norm(B, axis=1).mean())})
    json.dump(out, open(f"{RESULTS}/ckpt_transplant_geom.json", "w"), indent=1)
    for k, v in out.items():
        if isinstance(v, float):
            print(f"{k:22s} {v:+.4f}")
    print("wrote results/ckpt_transplant_geom.json")


if __name__ == "__main__":
    main()
