"""Do the token-path third regions behave like ONE state across their whole run?

The S4 analogue for the same-context token screen.  For every A|C|B candidate of the
token-embedding screen we greedily decode 20 tokens from five points of the path -- the centre of
the A run, the first / middle / last grid point of the C run, and the centre of the B run -- plus
the two unpatched endpoint sequences as controls.  The path's context is shared, so all seven
continuations are conditioned on exactly the same 31 tokens.

Reported per candidate: the common greedy prefix length across the three C-run points (0-20; high
means the C region decodes the same way throughout its run rather than at one grid point) and the
common prefix between the C-run centre and each endpoint continuation (how different the third
region's text is from either endpoint's).

usage: python token_continuations.py
"""
import json
import os
import pickle

import numpy as np
import torch
from transformers import GPT2TokenizerFast

from common import ALPHAS, MODEL, RESULTS, REVISION, Runner, slerp_rescale
from plot_token import D_HI, D_LO, RHO_FLAT, shelf

N_NEW = 20


@torch.no_grad()
def greedy(run, ids, embeds, n_new=N_NEW):
    """ids [1,T]; embeds [B,C] replacing the final position -> [B, n_new] greedy continuations."""
    run._rec_layers, run._rec, run._patch = set(), {}, None
    m = run.model
    e = m.transformer.wte(ids.to(run.device)).repeat(len(embeds), 1, 1)
    e = torch.cat([e[:, :-1, :], embeds[:, None, :].to(e.dtype)], dim=1)
    out = m.transformer(inputs_embeds=e, use_cache=True)
    past, h = out.past_key_values, out.last_hidden_state[:, -1:, :]
    toks = []
    for _ in range(n_new):
        nxt = m.lm_head(h)[:, -1, :].float().argmax(-1, keepdim=True)
        toks.append(nxt)
        out = m.transformer(input_ids=nxt, past_key_values=past, use_cache=True)
        past, h = out.past_key_values, out.last_hidden_state
    return torch.cat(toks, dim=1).cpu().numpy()


def prefix(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def main():
    tok = GPT2TokenizerFast.from_pretrained(MODEL, revision=REVISION)
    W = np.load(os.path.join(RESULTS, "ctx.npz"))["windows"]
    z = np.load(os.path.join(RESULTS, "token_interp_curves.npz"))
    rows = pickle.load(open(os.path.join(RESULTS, "token_interp_rows.pkl"), "rb"))["rows"]["token_embed"]
    D = z["token_embed_d"]
    cand = [n for n, r in enumerate(rows) if r["is_candidate"]]

    run = Runner()
    dev = run.device
    wte = run.model.transformer.wte.weight
    out = []
    for n in cand:
        r = rows[n]
        rho, dbar = shelf(D[n], r["k_in"], r["k_out"])
        kA = r["k_in"] // 2                                  # centre of the A run
        kB = (r["k_out"] + 1 + len(ALPHAS) - 1) // 2         # centre of the B run
        ks = [kA, r["k_in"], (r["k_in"] + r["k_out"]) // 2, r["k_out"], kB]
        ts = torch.tensor([ALPHAS[k] for k in ks], dtype=torch.float32, device=dev)
        H = slerp_rescale(wte[r["t_A"]].float(), wte[r["t_B"]].float(), ts)
        ids = torch.from_numpy(W[r["i"]][None]).long()
        g = greedy(run, ids, H)
        ends = greedy(run, ids, torch.stack([wte[r["t_A"]].float(), wte[r["t_B"]].float()]))
        c_first, c_mid, c_last = g[1], g[2], g[3]
        out.append({
            "path": int(n), "C": tok.decode([r["C"]]), "A": tok.decode([r["A"]]),
            "B": tok.decode([r["B"]]), "t_A": tok.decode([r["t_A"]]),
            "t_B": tok.decode([r["t_B"]]), "rho": float(rho), "d_mean_C": float(dbar),
            "run_len": int(r["run_len"]),
            "subplateau": bool(rho < RHO_FLAT and D_LO < dbar < D_HI),
            "prefix_C_run": int(min(prefix(c_first, c_mid), prefix(c_mid, c_last))),
            "prefix_C_vs_A_end": int(prefix(c_mid, ends[0])),
            "prefix_C_vs_B_end": int(prefix(c_mid, ends[1])),
            "prefix_A_region_vs_A_end": int(prefix(g[0], ends[0])),
            "prefix_B_region_vs_B_end": int(prefix(g[4], ends[1])),
            "text_C_mid": tok.decode(c_mid), "text_A_end": tok.decode(ends[0]),
            "text_B_end": tok.decode(ends[1]),
            "context": tok.decode(W[r["i"], :-1]),
        })
        if len(out) % 20 == 0:
            print(f"  {len(out)}/{len(cand)}", flush=True)

    sub = [o for o in out if o["subplateau"]]
    rest = [o for o in out if not o["subplateau"]]
    med = lambda v: float(np.median(v)) if v else float("nan")
    summ = {
        "n_candidates": len(out), "n_subplateau": len(sub),
        "median_prefix_C_run_all": med([o["prefix_C_run"] for o in out]),
        "median_prefix_C_run_subplateau": med([o["prefix_C_run"] for o in sub]),
        "median_prefix_C_run_other": med([o["prefix_C_run"] for o in rest]),
        "frac_prefix_ge_5_all": float(np.mean([o["prefix_C_run"] >= 5 for o in out])),
        "frac_prefix_ge_5_subplateau": float(np.mean([o["prefix_C_run"] >= 5 for o in sub])),
        "frac_prefix_ge_10_subplateau": float(np.mean([o["prefix_C_run"] >= 10 for o in sub])),
        "frac_prefix_full_subplateau": float(np.mean([o["prefix_C_run"] >= N_NEW for o in sub])),
        "median_prefix_C_vs_A_end": med([o["prefix_C_vs_A_end"] for o in out]),
        "median_prefix_C_vs_B_end": med([o["prefix_C_vs_B_end"] for o in out]),
        "median_prefix_A_region_vs_A_end": med([o["prefix_A_region_vs_A_end"] for o in out]),
        "median_prefix_B_region_vs_B_end": med([o["prefix_B_region_vs_B_end"] for o in out]),
        "n_new": N_NEW,
    }
    with open(os.path.join(RESULTS, "token_continuations.json"), "w") as f:
        json.dump({"summary": summ, "candidates": out}, f, indent=1)
    print(json.dumps(summ, indent=1))
    print("saved results/token_continuations.json")


if __name__ == "__main__":
    main()
