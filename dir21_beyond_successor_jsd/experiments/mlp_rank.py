"""How many directions of the block-0 MLP output does the width trait need?

The full transplant of m_u transports a token's width almost completely (slope +0.913 on the donor).
This asks how compressible that vector is: transplant only the projection of the donor-recipient
difference onto the top k principal components of m across the 123 endpoint tokens,

    m_write = m_r + P_k (m_d - m_r),

and sweep k. Slope reaching +0.913 at small k => the trait is a low-dimensional feature that could be
monitored or edited as a unit. Slope tracking the variance retained => it is spread over the whole
output. A random k-dimensional subspace at the same k is the control that separates "the top PCs" from
"any k directions". Writes results/mlp_rank.json (partial-safe).
"""
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import N_ANCHOR, run_pair
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint, jsd_bits
from common import D18, RESULTS
from mlp_read import MLPOut, states

KS = [1, 2, 4, 8, 16, 32, 64, 122]
KS_RANDOM = [1, 4, 16, 64]
KS_TAIL = [16, 32, 58]              # the LAST k of the 122 components: the low-variance tail
SEED = 0

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


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
    toks12 = list(json.load(open(f"{RESULTS}/mode_split.json"))["tokens"].keys())

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    patcher = Patcher(model)
    mlp = MLPOut(model)
    pre = tok(FRAMES[0], return_tensors="pt").input_ids.cuda()

    M = []
    for s, i in endpoints:
        m, _, _ = states(model, mlp, patcher, pre, i)
        M.append(m.cpu().numpy())
    M = np.stack(M)                                        # (123, 2048)
    Mc = M - M.mean(0)
    _, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    var = np.cumsum(S ** 2) / (S ** 2).sum()

    m_by_tok, z_by_tok = {}, {}
    for s in toks12:
        m, _, z = states(model, mlp, patcher, pre, ids_by_str[s])
        m_by_tok[s], z_by_tok[s] = m, z.log_softmax(-1)
    anc = [endpoint(model, patcher, torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1))
           for a in anchors]

    @torch.inference_mode()
    def width_with(recipient, vec):
        mlp.write = vec
        ids = torch.cat([pre, torch.tensor([[ids_by_str[recipient]]], device=pre.device)], 1)
        x, z = endpoint(model, patcher, ids)
        mlp.write = None
        bits = float(jsd_bits(z_by_tok[recipient].unsqueeze(0), z.log_softmax(-1).unsqueeze(0))[0])
        w = float(np.nanmedian([run_pair(model, patcher, ids, x, z, xb, zb)[0] for xb, zb in anc]))
        return w, bits

    base = np.array([width_with(s, m_by_tok[s])[0] for s in toks12])   # self-transplant = baseline
    print(f"baseline mean {base.mean():.3f} sd {base.std(ddof=1):.3f}", flush=True)

    def sweep(basis, tag, k):
        """Transplant only the component of (m_d - m_r) inside `basis` (rows orthonormal)."""
        Bt = torch.tensor(basis, device="cuda", dtype=torch.float32)
        W = np.full((len(toks12), len(toks12)), np.nan)
        bits = []
        for i, r in enumerate(toks12):
            for j, d in enumerate(toks12):
                if i == j:
                    W[i, j] = base[i]
                    continue
                diff = m_by_tok[d] - m_by_tok[r]
                v = m_by_tok[r] + Bt.T @ (Bt @ diff)
                W[i, j], b = width_with(r, v)
                bits.append(b)
        rho = [float(spearmanr(np.delete(base, i), np.delete(W[i], i)).statistic)
               for i in range(len(toks12))]
        slope = [float(np.polyfit(np.delete(base, i), np.delete(W[i], i), 1)[0])
                 for i in range(len(toks12))]
        row = dict(basis=tag, k=k, rho=float(np.mean(rho)), slope=float(np.mean(slope)),
                   rho_all=rho, slope_all=slope, w=W.tolist(),
                   median_bits=float(np.median(bits)),
                   var_retained=float(var[k - 1]) if tag == "pca" else None)
        print(f"{tag} k={k:3d}: slope {row['slope']:+.3f} rho {row['rho']:+.3f} "
              f"(var retained {row['var_retained'] if row['var_retained'] is None else round(row['var_retained'], 3)}, "
              f"{row['median_bits']:.3f} bits)", flush=True)
        return row

    rows = []
    g = np.random.default_rng(SEED)
    out = dict(model=MODEL, revision=REVISION, frame=FRAMES[0], tokens=toks12,
               base_w=[float(x) for x in base],
               anchors=[tok.convert_ids_to_tokens(a) for a in anchors],
               spectrum=[float(x) for x in var[:130]], rows=rows)
    for k in KS:
        rows.append(sweep(Vt[:k], "pca", k))
        if k in KS_RANDOM:
            Q, _ = np.linalg.qr(g.standard_normal((M.shape[1], k)))
            rows.append(sweep(Q.T, "random", k))
        json.dump(out, open(f"{RESULTS}/mlp_rank.json", "w"), indent=1)
    for k in KS_TAIL:
        row = sweep(Vt[122 - k:122], "tail", k)
        row["var_retained"] = float(1 - var[121 - k])
        rows.append(row)
        json.dump(out, open(f"{RESULTS}/mlp_rank.json", "w"), indent=1)
    print("wrote results/mlp_rank.json")


if __name__ == "__main__":
    main()
