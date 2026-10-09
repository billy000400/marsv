"""S2a: one pass over the 28 transcoder layers. For the tuning and original capital pairs, encode the native MLP
input at the country and bridge (` is`) positions, and keep only the rows of features active at either endpoint,
plus 64 random decoder rows per layer for the norm-matched controls. Each 1.3 GB dictionary is loaded, sliced and
released. Writes results/slices/layer_<L>.pt and results/s2_slices.csv."""
import csv
import os

import torch

from common import COUNTRIES, DEV, N_LAYERS, OUT, SEED, SLICES, TEMPLATES, d30, ids_of, setup

N_RANDOM = 64


def main():
    tok, model = setup()
    d30.SCRATCH = "/tmp/hf_dir31"
    os.makedirs(SLICES, exist_ok=True)
    tap = d30.MLPTap(model)
    sites = {}  # (template, country, site) -> (x [L, d], y [L, d])
    for name in ["tune", "original"]:
        prefix, bridge = TEMPLATES[name]
        for c in COUNTRIES:
            pos = tok(prefix.format(c), return_tensors="pt").input_ids.shape[1] - 1
            model(ids_of(tok, prefix.format(c) + " is"))
            for site, p in [("country", pos), ("is", pos + 1)]:
                sites[(name, c, site)] = (torch.stack([tap.x[L][0, p] for L in range(N_LAYERS)]),
                                          torch.stack([tap.y[L][0, p] for L in range(N_LAYERS)]))
    tap.remove()
    keys = list(sites)
    rows = []
    for L in range(N_LAYERS):
        W_enc, b_enc, W_dec, b_dec = d30.load_transcoder(L)
        acts = torch.stack([d30.encode(sites[k][0][L], W_enc, b_enc) for k in keys])  # [n_sites, F]
        union = (acts > 0).any(0).nonzero().flatten()
        g = torch.Generator().manual_seed(SEED * 1000 + L)
        rand = torch.randperm(W_dec.shape[0], generator=g)[:N_RANDOM].to(DEV)
        for i, k in enumerate(keys):
            recon = acts[i] @ W_dec + b_dec
            rows.append(dict(layer=L, template=k[0], country=k[1], site=k[2], n_active=int((acts[i] > 0).sum()),
                             mlp_out_l2=sites[k][1][L].norm().item(),
                             rel_recon_error=((recon - sites[k][1][L]).norm() / sites[k][1][L].norm()).item()))
        torch.save(dict(layer=L, keys=keys, ids=union.cpu(), acts=acts[:, union].cpu(), W_enc=W_enc[union].cpu(),
                        b_enc=b_enc[union].cpu(), W_dec=W_dec[union].cpu(), rand_ids=rand.cpu(), rand_dec=W_dec[rand].cpu()),
                   os.path.join(SLICES, f"layer_{L}.pt"))
        print(L, "union", len(union), {f"{k[0]}|{k[1]}|{k[2]}": r["n_active"] for k, r in zip(keys, rows[-len(keys):])}, flush=True)
        del W_enc, b_enc, W_dec, b_dec, acts
        torch.cuda.empty_cache()
    with open(os.path.join(OUT, "s2_slices.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
