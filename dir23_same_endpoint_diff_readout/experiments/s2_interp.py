"""S2: one Japan->Germany embedding interpolation, run through all five readout suffixes.

d(t) is computed from the full 50257-dim logit vector in memory; the saved artefacts are the
derived curves (the raw logit matrices are ~50 MB and are not kept).
Writes results/interp.npz (d-curves, probability tracks, top-1 ids), results/interp.csv and
results/transitions.json.
"""
import csv
import json
import os

import numpy as np
import torch

from common import (ALPHAS, ENDPOINT_A, ENDPOINT_B, MODEL, PREFIX, READOUTS,
                    RESULTS, jsd_bits, load, rel_dist, slerp_lerp_norm,
                    transition_stats)

CHUNK = 16


def main():
    tok, m, dev = load()
    wte = m.transformer.wte.weight
    prefix_ids = tok.encode(PREFIX)
    pos = len(prefix_ids)            # index of the interpolated country token
    id_a, id_b = tok.encode(ENDPOINT_A)[0], tok.encode(ENDPOINT_B)[0]
    nl_id = tok.encode("\n")[0]

    ea, eb = wte[id_a].detach().float(), wte[id_b].detach().float()
    vecs, omega, cos = slerp_lerp_norm(ea, eb, ALPHAS)      # (101, d), built once
    print(f"endpoint norms |A|={float(ea.norm()):.3f} |B|={float(eb.norm()):.3f} "
          f"cos={cos:.4f} omega={omega:.4f} rad")

    def sweep(full_ids):
        """Run all 101 interpolated embeddings through one token sequence; last-position logits."""
        base = wte[torch.tensor(full_ids, device=dev)].detach().float()   # (T, d)
        rows = []
        for s in range(0, len(ALPHAS), CHUNK):
            v = vecs[s:s + CHUNK]
            emb = base.unsqueeze(0).repeat(v.shape[0], 1, 1)
            emb[:, pos, :] = v
            with torch.no_grad():
                lg = m(inputs_embeds=emb, use_cache=False).logits[:, -1, :].float().cpu()
            rows.append(lg)
        return torch.cat(rows).numpy()

    store = {"alphas": ALPHAS}
    trans = {}

    # --- immediate position: prefix + country token, no readout suffix -------
    lg_imm = sweep(prefix_ids + [id_a])
    p_imm = torch.softmax(torch.tensor(lg_imm), dim=-1).numpy()
    store["immediate_p_newline"] = p_imm[:, nl_id]
    store["immediate_top1"] = np.argmax(p_imm, axis=1)
    d_imm = rel_dist(lg_imm.astype(np.float64), lg_imm[0].astype(np.float64), lg_imm[-1].astype(np.float64))
    store["immediate_d"] = d_imm
    trans["immediate"] = dict(transition_stats(ALPHAS, d_imm),
                              jsd_bits=jsd_bits(p_imm[0], p_imm[-1]))
    print(f"immediate: p(nl) {p_imm[0, nl_id]:.4f} -> {p_imm[-1, nl_id]:.4f}, "
          f"top1 always newline: {bool((store['immediate_top1'] == nl_id).all())}")

    # --- five downstream readouts, same 101 embeddings ----------------------
    for name, suf, ans_a, ans_b in READOUTS:
        ids = prefix_ids + [id_a] + tok.encode(suf)
        lg = sweep(ids)
        p = torch.softmax(torch.tensor(lg), dim=-1).numpy()
        z = lg.astype(np.float64)
        d = rel_dist(z, z[0], z[-1])
        ia, ib = tok.encode(ans_a)[0], tok.encode(ans_b)[0]
        store[f"d_{name}"] = d
        store[f"p_A_{name}"] = p[:, ia]
        store[f"p_B_{name}"] = p[:, ib]
        store[f"top1_{name}"] = np.argmax(p, axis=1)
        st = transition_stats(ALPHAS, d)
        st["jsd_bits"] = jsd_bits(p[0], p[-1])
        st["answer_A"], st["answer_B"] = ans_a, ans_b
        st["top1_switch_t"] = None
        t1 = np.argmax(p, axis=1)
        sw = np.where(t1[:-1] != t1[1:])[0]
        st["top1_switch_t"] = [float(ALPHAS[i + 1]) for i in sw]
        st["n_distinct_top1"] = int(len(np.unique(t1)))
        trans[name] = st
        print(f"{name:10s} t10={st['t10']} t50={st['t50']} t90={st['t90']} w={st['w']} "
              f"mono={st['monotonic']} maxback={st['max_backstep']:.4f} "
              f"crossings(10/50/90)={st['n_cross_t10']}/{st['n_cross_t50']}/{st['n_cross_t90']}")

    prim = [trans[n]["t50"] for n in ("Capital", "Continent", "Currency", "Language")]
    trans["delta_t50_primary"] = float(max(prim) - min(prim))
    trans["omega_rad"], trans["cos_endpoints"] = omega, cos
    print("delta_t50 (primary four) =", trans["delta_t50_primary"])

    np.savez_compressed(os.path.join(RESULTS, "interp.npz"), **store)
    with open(os.path.join(RESULTS, "transitions.json"), "w") as f:
        json.dump(trans, f, indent=2)

    names = [n for n, _, _, _ in READOUTS]
    with open(os.path.join(RESULTS, "interp.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = (["t", "immediate_p_newline"]
                + [f"d_{n}" for n in names]
                + [f"p_A_{n}" for n in names] + [f"p_B_{n}" for n in names])
        w.writerow(cols)
        for i, t in enumerate(ALPHAS):
            w.writerow([f"{t:.2f}", store["immediate_p_newline"][i]]
                       + [store[f"d_{n}"][i] for n in names]
                       + [store[f"p_A_{n}"][i] for n in names]
                       + [store[f"p_B_{n}"][i] for n in names])
    print("wrote results/interp.npz, interp.csv, transitions.json")


if __name__ == "__main__":
    main()
