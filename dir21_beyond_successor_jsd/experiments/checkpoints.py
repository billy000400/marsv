"""When during training does the per-token width trait appear, and does it track a corpus statistic?

The direction's practical deliverable is a free static-embedding lookup that ranks tokens by transition
width. Iteration 11 showed the ranking is shared by Pythia-410M/1B/1.4B but ABSENT in 160M, so it is
learned rather than architectural. This asks WHEN it is learned, in one model (Pythia-410M-deduped)
across training checkpoints, with the same 123 tokens, 6 anchors, 3 frames and block-0 site.

Per checkpoint we record:
  A. anchor widths w_hat_u for the 123 endpoint tokens (median over 3 frames x 6 anchors);
  B. the split-half (Spearman-Brown) reliability of that measurement, so a weak agreement can be told
     apart from a noisy one;
  C. the embedding probe refitted INSIDE this checkpoint (does W_E already encode width here?);
  D. rank agreement with two corpus statistics that need no model at all -- the token's unigram count
     and its successor entropy, both from dir18's manifest.

Agreement with the final checkpoint is computed in checkpoints_analysis.py. Results are written
incrementally to results/checkpoints.json so a partial sweep is still usable.
"""
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

import second_model
from basin_probe import FRAMES, Patcher
from common import RESULTS
from mlp_read import probe
from second_model import endpoint_set, widths

MODEL = "EleutherAI/pythia-410m-deduped"
# the four steps the deliverables named, first, then a finer grid while the budget lasts
REVISIONS = ["step1000", "step8000", "step32000", "step143000",
             "step0", "step2000", "step64000", "step512", "step16000", "step4000",
             # the ordering was already in place at step512, so refine the first 512 steps
             "step128", "step32", "step256", "step8", "step64", "step16", "step2"]
N_TRAIN = 80
DEADLINE = float(os.environ.get("CKPT_DEADLINE_S", 3000))

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def main():
    out_path = os.path.join(RESULTS, "checkpoints.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res.update(model=MODEL, frames=FRAMES)
    res.setdefault("ckpt", {})
    save = lambda: json.dump(res, open(out_path, "w"), indent=1)

    ids_by_str, anchors = endpoint_set()
    res["anchor_ids"] = anchors
    t0 = time.time()

    for rev in REVISIONS:
        if rev in res["ckpt"]:
            continue
        if time.time() - t0 > DEADLINE:
            print("deadline reached", flush=True)
            break
        print(f"=== {rev} ===", flush=True)
        second_model.TOK = AutoTokenizer.from_pretrained(MODEL, revision=rev)
        model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=rev,
                                                   torch_dtype=torch.float32).eval().cuda()
        patcher = Patcher(model)

        w, raw, oj, valid = widths(model, patcher, ids_by_str, anchors, FRAMES, log_every=60)
        names = [s for s in ids_by_str if np.isfinite(w[s])]
        y = np.array([w[s] for s in names])

        E = model.gpt_neox.embed_in.weight.detach().float().cpu().numpy()
        F = np.array([E[ids_by_str[s]] for s in names])
        n_train = min(N_TRAIN, len(y) - 20)
        rho, r2, lam = probe(F, y, np.random.default_rng(1), n_train)
        rho_n, _, _ = probe(F, np.random.default_rng(0).permutation(y),
                            np.random.default_rng(1), n_train)

        res["ckpt"][rev] = dict(
            w=w, w_raw={s: [float(x) for x in v] for s, v in raw.items()},
            out_jsd=oj, valid_frac=valid,
            median_w=float(np.median(y)), sd_w=float(y.std(ddof=1)), n=len(y),
            probe=dict(rho_mean=float(rho.mean()), rho_sd=float(rho.std()),
                       r2_mean=float(r2.mean()), null_rho_mean=float(rho_n.mean()),
                       n_train=n_train, median_alpha=lam))
        save()
        print(f"[{rev}] valid {valid:.3f} median w {np.median(y):.3f} sd {y.std(ddof=1):.3f} "
              f"probe rho {rho.mean():+.3f} +- {rho.std():.3f} (null {rho_n.mean():+.3f}) "
              f"[{time.time() - t0:.0f}s]", flush=True)

        patcher.close()
        del model, patcher
        torch.cuda.empty_cache()

    save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
