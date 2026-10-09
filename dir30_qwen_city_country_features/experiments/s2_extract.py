"""S2/S4 extraction: cache MLP inputs at (a) the last prompt token of every model-correct row and
(b) the country token of the held-out capital prefixes, then encode them one transcoder layer at a time.
Writes results/acts/layer_{L}.npz (sparse activations), results/early_baseline.json and results/layer_stats.csv."""
import csv
import json
import os

import numpy as np
import torch

from common import DEV, N_LAYERS, OUT, MLPTap, encode, load_model, load_transcoder, setup_torch

EARLY = [("Japan", "Tokyo"), ("Germany", "Berlin")]


def stack(tap, pos):
    return (torch.stack([tap.x[i][0, pos] for i in range(N_LAYERS)]),
            torch.stack([tap.y[i][0, pos] for i in range(N_LAYERS)]))


def main():
    setup_torch()
    os.makedirs(os.path.join(OUT, "acts"), exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(OUT, "samples.jsonl"))]
    keep = [i for i, r in enumerate(rows) if r["correct"]]
    tok, model = load_model()
    tap = MLPTap(model)

    X, Y = [], []
    for i in keep:
        enc = tok(rows[i]["prompt"], return_tensors="pt").to(DEV)
        model(**enc)
        assert enc.input_ids.shape[1] - 1 == rows[i]["last_token_index"]
        x, y = stack(tap, -1)
        X.append(x)
        Y.append(y)

    # Early position: last sub-token of the country name, read in the prefix-only run and in the "... is" run.
    base, XE, early_rows = [], [], []
    for country, city in EARLY:
        prefix = f"The capital of {country}"
        enc = tok(prefix, return_tensors="pt", return_offsets_mapping=True)
        start = prefix.index(country)
        span = [j for j, (a, b) in enumerate(enc.offset_mapping[0].tolist()) if b > start and a < start + len(country)]
        pos = span[-1]
        ids = enc.input_ids.to(DEV)
        logp = model(ids).logits[0, -1].float().log_softmax(-1)
        x_prefix, _ = stack(tap, pos)
        ids_full = tok(prefix + " is", return_tensors="pt").input_ids.to(DEV)
        assert ids_full[0, :ids.shape[1]].tolist() == ids[0].tolist() and ids_full.shape[1] == ids.shape[1] + 1
        logp_full = model(ids_full).logits[0, -1].float().log_softmax(-1)
        x_full, _ = stack(tap, pos)
        city_ids = tok(" " + city).input_ids
        gen = model.generate(ids_full, do_sample=False, max_new_tokens=8, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.eos_token_id)
        base.append(dict(
            prefix=prefix, country=country, city=city, tokens=[tok.decode([t]) for t in ids[0].tolist()],
            country_span=span, country_position=pos, country_token=tok.decode(ids[0, pos:pos + 1]),
            prefix_top1=tok.decode(logp.argmax().item()), prefix_top1_prob=logp.max().exp().item(),
            bridge_token=tok.decode(ids_full[0, -1:]), bridge_prob=logp[ids_full[0, -1]].exp().item(),
            bridge_rank=int((logp > logp[ids_full[0, -1]]).sum().item()) + 1,
            city_token_ids=city_ids, after_bridge_top1=tok.decode(logp_full.argmax().item()),
            city_prob_after_bridge=logp_full[city_ids[0]].exp().item(),
            continuation_after_bridge=tok.decode(gen[0, ids_full.shape[1]:]),
            max_abs_diff_prefix_vs_full=(x_prefix - x_full).abs().max().item(),
            max_abs_mlp_input=x_prefix.abs().max().item()))
        XE += [x_prefix, x_full]
        early_rows += [f"{country}|prefix", f"{country}|full"]
    json.dump(base, open(os.path.join(OUT, "early_baseline.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(base, indent=1, ensure_ascii=False))
    tap.remove()
    del model
    torch.cuda.empty_cache()

    X, Y, XE = torch.stack(X), torch.stack(Y), torch.stack(XE)  # [rows, layers, d_model]
    stats = []
    for L in range(N_LAYERS):
        W_enc, b_enc, W_dec, b_dec = load_transcoder(L)
        a = encode(X[:, L], W_enc, b_enc)
        ae = encode(XE[:, L], W_enc, b_enc)
        rel = ((a @ W_dec + b_dec - Y[:, L]).norm(dim=-1) / Y[:, L].norm(dim=-1))
        stats.append(dict(layer=L, n_rows=len(a), mean_active_features=a.gt(0).sum(-1).float().mean().item(),
                          rel_l2_error=rel.mean().item(), finite=bool(torch.isfinite(a).all())))
        print(stats[-1], flush=True)
        r, f = a.nonzero(as_tuple=True)
        re_, fe = ae.nonzero(as_tuple=True)
        np.savez(os.path.join(OUT, "acts", f"layer_{L}.npz"),
                 sample_index=np.array(keep)[r.cpu().numpy()], feature=f.cpu().numpy(), value=a[r, f].cpu().numpy(),
                 early_row=re_.cpu().numpy(), early_feature=fe.cpu().numpy(), early_value=ae[re_, fe].cpu().numpy(),
                 early_row_names=np.array(early_rows))
        del W_enc, b_enc, W_dec, b_dec, a, ae
        torch.cuda.empty_cache()
    with open(os.path.join(OUT, "layer_stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)


if __name__ == "__main__":
    main()
