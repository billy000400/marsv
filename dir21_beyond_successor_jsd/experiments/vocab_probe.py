"""Does the embedding lookup hold outside dir18's curated pool?

Every token measured so far is a common single-token alphabetic word. Here the probe (fitted on the
123 bank tokens) is applied to all 50,304 embedding rows, and ~32 tokens spanning its predicted range
are selected from FOUR classes the pool excludes or under-samples — ordinary words outside the pool,
subword fragments, punctuation/numerals, and capitalised names — and their anchor widths are measured
at block 0 with the usual six anchors and three frames.

Writes results/vocab.json.
"""
import json
import os
import re

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import N_ANCHOR, run_pair
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint
from common import D18, RESULTS
from embed_forward import fit_probe

PER_CLASS = 8

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def classify(s):
    """Which of the four token classes a decoded token string belongs to (None = skip)."""
    if not s or s.strip() == "" or len(s) > 15:
        return None
    body = s[1:] if s.startswith(" ") else s
    if not body:
        return None
    if re.fullmatch(r"[A-Za-z]+", body):
        if body[0].isupper():
            return "capitalised" if s.startswith(" ") else None
        return "word" if s.startswith(" ") else "fragment"
    if re.fullmatch(r"[^A-Za-z\s]+", body):
        return "symbol"
    return None


def pick(pred, strs, used_ids):
    """PER_CLASS tokens per class, evenly spaced over that class's predicted-width quantiles."""
    out = {}
    for cls in ("word", "fragment", "symbol", "capitalised"):
        ids = [i for i, s in enumerate(strs)
               if i not in used_ids and classify(s) == cls]
        ids.sort(key=lambda i: pred[i])
        if len(ids) < PER_CLASS:
            out[cls] = ids
            continue
        idx = np.linspace(0, len(ids) - 1, PER_CLASS).round().astype(int)
        out[cls] = [ids[j] for j in idx]
    return out


def main():
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    used = set(ids_by_str.values())
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    anchors = pool[:: max(1, len(pool) // N_ANCHOR)][:N_ANCHOR]
    assert len(anchors) == N_ANCHOR

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    E = model.gpt_neox.embed_in.weight.detach().float().cpu().numpy()

    w0 = {s: float(np.nanmedian(v["w"]))
          for s, v in json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}
    bank = sorted(w0)
    predict, lam, bank_r2 = fit_probe(np.array([E[ids_by_str[s]] for s in bank]),
                                      np.array([w0[s] for s in bank]), np.random.default_rng(3))
    pred_all = predict(E)
    print(f"probe on {len(bank)} bank tokens (alpha={lam:g}, bank CV R2={bank_r2:+.3f}); "
          f"vocab predictions span {pred_all.min():.3f}-{pred_all.max():.3f}", flush=True)

    strs = [tok.convert_tokens_to_string([t]) if t is not None else ""   # ids past the vocab
            for t in tok.convert_ids_to_tokens(list(range(len(E))))]
    exclude = used | set(anchors) | set(cand["pool"])
    chosen = pick(pred_all, strs, exclude)
    sel = [(cls, i, strs[i]) for cls, ids in chosen.items() for i in ids]
    print(f"selected {len(sel)} tokens: "
          + ", ".join(f"{cls}:{s!r}" for cls, _, s in sel), flush=True)

    patcher = Patcher(model)
    rows = {s: dict(token_id=i, cls=cls, pred=float(pred_all[i]), w=[]) for cls, i, s in sel}
    for fi, frame in enumerate(FRAMES):
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        anc = [endpoint(model, patcher,
                        torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1))
               for a in anchors]
        for cls, i, s in sel:
            ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
            x, z = endpoint(model, patcher, ids)
            for xb, zb in anc:
                rows[s]["w"].append(run_pair(model, patcher, ids, x, z, xb, zb)[0])
        print(f"frame {fi} done", flush=True)
    patcher.close()

    for s in rows:
        rows[s]["w_med"] = float(np.nanmedian(rows[s]["w"]))
        rows[s]["valid"] = float(np.mean(~np.isnan(np.array(rows[s]["w"], dtype=float))))

    names = list(rows)
    p = np.array([rows[s]["pred"] for s in names])
    m = np.array([rows[s]["w_med"] for s in names])
    ok = ~np.isnan(m)
    res = dict(anchors=[tok.convert_ids_to_tokens(a) for a in anchors], alpha=lam,
               bank_cv_r2=float(bank_r2), n_selected=len(names),
               valid_frac=float(np.mean([rows[s]["valid"] for s in names])),
               rho_all=[float(x) for x in spearmanr(p[ok], m[ok])[:2]],
               mae=float(np.abs(p[ok] - m[ok]).mean()),
               bank_span=[float(min(w0.values())), float(max(w0.values()))],
               vocab_span=[float(np.nanmin(m)), float(np.nanmax(m))],
               tokens=rows)
    for cls in ("word", "fragment", "symbol", "capitalised"):
        k = [j for j, s in enumerate(names) if rows[s]["cls"] == cls and ok[j]]
        res[f"rho_{cls}"] = [float(x) for x in spearmanr(p[k], m[k])[:2]] if len(k) > 3 else None
        res[f"median_w_{cls}"] = float(np.median(m[k])) if k else None
    json.dump(res, open(os.path.join(RESULTS, "vocab.json"), "w"), indent=1)
    print(f"valid curves {res['valid_frac']:.3f}; rho(predicted, measured) over {int(ok.sum())} "
          f"tokens = {res['rho_all'][0]:+.3f} (p={res['rho_all'][1]:.1e}), MAE {res['mae']:.3f}")
    for cls in ("word", "fragment", "symbol", "capitalised"):
        print(f"  {cls:12s} rho {res[f'rho_{cls}']}, median measured w {res[f'median_w_{cls}']}")


if __name__ == "__main__":
    main()
