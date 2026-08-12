"""Is the block-0 MLP output enough to INSTALL the width ordering in a model that lacks most of it?

Everything causal in this direction so far has been measured inside one fully-trained network: damaging
the block-0 MLP destroys the across-token width ordering, and swapping m_u between tokens moves the
recipient's width toward the donor's. That shows the vector matters where the ordering already exists.
The checkpoint sweep gives a second kind of test. At step128 of Pythia-410M the final ordering is only
half present (rho vs final 0.443, 0.50 of the reliability ceiling) while at step143000 it is complete.
So we can ask a sufficiency question that no single network can answer: if the ONLY thing we change in
the step128 model is the block-0 MLP's final-position output m_u -- copied token-by-token from the
final checkpoint -- does the step128 model's width ranking move to the final checkpoint's?

Six conditions per direction, each a full anchor-width sweep (123 tokens x 3 frames x 6 anchors):
  base           nothing written -- reproduces the stored sweep;
  self           the recipient's OWN m_u written back -- must reproduce base exactly (hook sanity);
  donor          the other checkpoint's m_u, as measured;
  donor_scaled   the same vectors times one global factor kappa matching the median m-norm, in case
                 the two checkpoints simply operate at different scales;
  shuffle        the donor vectors under a fixed derangement -- same vectors, wrong token identity.
                 This is the control that separates "the ordering was carried in" from "any large
                 perturbation of m_u changes widths";
  shuffle_scaled the same, scaled.

Both directions are run: early <- final (does the ordering appear?) and final <- early (does it go
away?). Writes results/ckpt_transplant.json incrementally, so a partial run is still usable.
"""
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import run_pair
from basin_probe import FRAMES, Patcher, endpoint, jsd_bits
from common import RESULTS
from mlp_read import MLPOut
from second_model import endpoint_set

MODEL = "EleutherAI/pythia-410m-deduped"
EARLY = "step128"
FINAL = "step143000"
CONDS = ["base", "self", "donor", "donor_scaled", "shuffle", "shuffle_scaled"]
DEADLINE = float(os.environ.get("XPLANT_DEADLINE_S", 4200))

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def load(rev):
    return GPTNeoXForCausalLM.from_pretrained(MODEL, revision=rev,
                                              torch_dtype=torch.float32).eval().cuda()


@torch.inference_mode()
def capture_m(model, mlp, patcher, tok, ids_by_str):
    """Block-0 MLP output at the final position, per frame and token: m[frame][token] on the CPU."""
    out = []
    for frame in FRAMES:
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        d = {}
        for s, i in ids_by_str.items():
            mlp.capture = True
            patcher.bank = None
            model(torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1))
            mlp.capture = False
            d[s] = mlp.rec[0].clone().cpu()
        out.append(d)
    return out


def sweep(model, patcher, mlp, tok, ids_by_str, anchors, write=None, z_ref=None):
    """Anchor widths with an optional per-frame, per-token vector written into the block-0 MLP.

    Only the endpoint state is computed with the write active: run_pair overwrites the whole
    post-block-0 state with the interpolation bank, so the write cannot leak into the path itself.
    """
    raw = {s: [] for s in ids_by_str}
    bits, z0 = {}, {}
    for fi, frame in enumerate(FRAMES):
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        anc = [endpoint(model, patcher,
                        torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1))
               for a in anchors]
        for k, (s, i) in enumerate(ids_by_str.items()):
            ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
            mlp.write = None if write is None else write[fi][s].cuda()
            x, z = endpoint(model, patcher, ids)
            mlp.write = None
            if fi == 0:
                lz = z.log_softmax(-1)
                z0[s] = lz.cpu()
                if z_ref is not None:
                    bits[s] = float(jsd_bits(z_ref[s].cuda().unsqueeze(0), lz.unsqueeze(0))[0])
            for xb, zb in anc:
                raw[s].append(run_pair(model, patcher, ids, x, z, xb, zb)[0])
            if k % 60 == 0:
                print(f"    frame {fi} token {k}/{len(ids_by_str)} {s!r} "
                      f"w={np.nanmedian(raw[s]):.3f}", flush=True)
    w = {s: float(np.nanmedian(v)) for s, v in raw.items()}
    valid = float(np.mean(np.isfinite(np.array([raw[s] for s in ids_by_str]))))
    return dict(w=w, w_raw={s: [float(x) for x in v] for s, v in raw.items()}, valid_frac=valid,
                median_bits=float(np.median(list(bits.values()))) if bits else 0.0), z0


def norms(m):
    return float(np.median([float(v.norm()) for d in m for v in d.values()]))


def main():
    out_path = os.path.join(RESULTS, "ckpt_transplant.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res.update(model=MODEL, early=EARLY, final=FINAL, frames=FRAMES, conditions=CONDS)
    res.setdefault("runs", {})
    save = lambda: json.dump(res, open(out_path, "w"), indent=1)

    ids_by_str, anchors = endpoint_set()
    res["anchor_ids"] = anchors
    tok = AutoTokenizer.from_pretrained(MODEL, revision=FINAL)
    tok_e = AutoTokenizer.from_pretrained(MODEL, revision=EARLY)
    ids = list(ids_by_str.values()) + anchors
    assert tok.convert_ids_to_tokens(ids) == tok_e.convert_ids_to_tokens(ids)

    names = list(ids_by_str)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(names))
    while (perm == np.arange(len(names))).any():              # a derangement: no token keeps its own
        perm = rng.permutation(len(names))
    res["shuffle_map"] = {names[i]: names[j] for i, j in enumerate(perm)}
    shuf = lambda m: [{names[i]: d[names[j]] for i, j in enumerate(perm)} for d in m]

    # ---- capture m_u in both checkpoints -----------------------------------------------------
    t0 = time.time()
    M = {}
    for rev in (FINAL, EARLY):
        model = load(rev)
        patcher, mlp = Patcher(model), MLPOut(model)
        M[rev] = capture_m(model, mlp, patcher, tok, ids_by_str)
        patcher.close()
        mlp.h.remove()
        del model
        torch.cuda.empty_cache()
        print(f"[m] {rev}: median ||m_u|| = {norms(M[rev]):.3f} ({time.time() - t0:.0f}s)", flush=True)
    res["m_norm"] = {r: norms(M[r]) for r in M}

    # ---- both transplant directions ----------------------------------------------------------
    for recip, don in ((EARLY, FINAL), (FINAL, EARLY)):
        tag = f"{recip}<-{don}"
        res["runs"].setdefault(tag, {})
        kappa = res["m_norm"][recip] / res["m_norm"][don]
        res["runs"][tag]["kappa"] = kappa
        writes = {"base": None, "self": M[recip], "donor": M[don],
                  "donor_scaled": [{s: v * kappa for s, v in d.items()} for d in M[don]],
                  "shuffle": shuf(M[don]),
                  "shuffle_scaled": shuf([{s: v * kappa for s, v in d.items()} for d in M[don]])}
        if all(c in res["runs"][tag] for c in CONDS):
            continue
        print(f"=== {tag} (kappa {kappa:.3f}) ===", flush=True)
        model = load(recip)
        patcher, mlp = Patcher(model), MLPOut(model)
        z_ref = None
        for c in CONDS:
            if c in res["runs"][tag]:
                continue
            if time.time() - t0 > DEADLINE:
                print("deadline reached", flush=True)
                break
            print(f"  -- {c}", flush=True)
            r, z0 = sweep(model, patcher, mlp, tok, ids_by_str, anchors, writes[c], z_ref)
            if c == "base":
                z_ref = z0
            res["runs"][tag][c] = r
            save()
            y = np.array([r["w"][s] for s in names])
            print(f"  [{c}] median w {np.median(y):.3f} sd {y.std(ddof=1):.3f} "
                  f"valid {r['valid_frac']:.3f} bits {r['median_bits']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        patcher.close()
        mlp.h.remove()
        del model
        torch.cuda.empty_cache()

    save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
