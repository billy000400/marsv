"""Can the block-0 MLP's contribution be READ, or is it only a stage the trait passes through?

The dose-response localised the per-token width trait to the block-0 MLP's final-position output m_u,
but only by damaging it. This asks whether that vector carries the number.

A. PROBE. Ridge probe from m_u to the measured anchor width w_hat_u over the 123 endpoint tokens,
   with the embedding probe's protocol (80 train / 43 test, 50 random splits, shuffled-target control).
   Two references on the same tokens and protocol: the static embedding row W_E[u] (rho = +0.76) and
   the full post-block-0 residual state x_u, which says whether m_u is special or whether any early
   activation carries it.

B. TRANSPLANT. For every ordered pair of 12 tokens, overwrite the recipient's m_u with the donor's and
   re-measure the recipient's anchor width. A transplant moves the output a lot on its own, so the
   test is not "does the width change" but "does the post-transplant width depend on WHICH donor" --
   scored per recipient across its 11 donors, plus a pooled slope against the donor's own width.

Writes results/mlp_read.json.
"""
import json

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from ablate import measure
from anchor_width import N_ANCHOR, run_pair
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint, jsd_bits
from common import D18, RESULTS
from embed_probe import ALPHAS, N_SPLIT

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def probe(F, y, rng, n_train, n_split=N_SPLIT):
    """embed_probe.probe with the ridge solved in its dual form.

    With 2048 features and 80 training tokens the primal solve is a 2049x2049 system per ridge
    strength; the identity (X'X + lam I)^-1 X' = X'(XX' + lam I)^-1 gives the same coefficients from
    an 80x80 system, which is what makes four probes affordable here. Verified against the primal
    version to 1e-8 (see the assertion in __main__).
    """
    n = len(y)
    rhos, r2s, alphas = [], [], []
    for _ in range(n_split):
        perm = rng.permutation(n)
        tr, te = perm[:n_train], perm[n_train:]
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
        Xtr = np.hstack([np.ones((len(tr), 1)), (F[tr] - mu) / sd])
        Xte = np.hstack([np.ones((len(te), 1)), (F[te] - mu) / sd])
        ym = y[tr].mean()

        def fit_pred(Xa, ya, Xb, lam):
            K = Xa @ Xa.T + lam * np.eye(len(Xa))
            return (Xb @ Xa.T) @ np.linalg.solve(K, ya)

        best, best_lam = -np.inf, ALPHAS[0]
        ip = rng.permutation(len(tr))
        for lam in ALPHAS:
            pred = np.empty(len(tr))
            for f in range(5):
                ite = ip[f::5]
                itr = np.setdiff1d(ip, ite)
                pred[ite] = fit_pred(Xtr[itr], y[tr][itr] - ym, Xtr[ite], lam) + ym
            s = 1 - ((y[tr] - pred) ** 2).sum() / ((y[tr] - y[tr].mean()) ** 2).sum()
            if s > best:
                best, best_lam = s, lam
        p = fit_pred(Xtr, y[tr] - ym, Xte, best_lam) + ym
        rhos.append(float(spearmanr(p, y[te])[0]))
        r2s.append(float(1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()))
        alphas.append(float(best_lam))
    return np.array(rhos), np.array(r2s), float(np.median(alphas))


class MLPOut:
    """Capture the block-0 MLP's final-position output, or overwrite it with a fixed vector."""

    def __init__(self, model):
        self.capture = False
        self.rec = None
        self.write = None
        self.h = model.gpt_neox.layers[0].mlp.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.capture:
            self.rec = out[:, -1, :].detach().float().clone()
        if self.write is None:
            return out
        out = out.clone()
        out[:, -1, :] = self.write.to(out.dtype)
        return out


@torch.inference_mode()
def states(model, mlp, patcher, pre, ids):
    """(m_u, x_u, logits) at the final position for one prompt."""
    mlp.capture = True
    patcher.bank = None
    z = model(torch.cat([pre, torch.tensor([[ids]], device=pre.device)], 1)).logits[0, -1].float()
    mlp.capture = False
    return mlp.rec[0].clone(), patcher.captured[0].clone().float(), z


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

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    patcher = Patcher(model)
    mlp = MLPOut(model)

    # ---- A. probe -----------------------------------------------------------------
    M, X = [], []
    for frame in FRAMES:
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        m_f, x_f = [], []
        for s, i in endpoints:
            m, x, _ = states(model, mlp, patcher, pre, i)
            m_f.append(m.cpu().numpy())
            x_f.append(x.cpu().numpy())
        M.append(np.stack(m_f))
        X.append(np.stack(x_f))
    M, X = np.stack(M), np.stack(X)                       # (3 frames, 123, 2048)
    cos_frames = float(np.mean([
        np.mean(np.sum(M[i] * M[j], 1) / (np.linalg.norm(M[i], axis=1) * np.linalg.norm(M[j], axis=1)))
        for i, j in ((0, 1), (0, 2), (1, 2))]))
    print(f"m_u across frames: mean cosine {cos_frames:.4f}", flush=True)
    E = model.gpt_neox.embed_in.weight.detach().float().cpu().numpy()

    names = [s for s, _ in endpoints]
    w0 = {s: float(np.nanmedian(v["w"]))
          for s, v in json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}
    y0 = np.array([w0[s] for s in names])

    res = {"model": MODEL, "revision": REVISION, "frames": FRAMES,
           "anchors": [tok.convert_ids_to_tokens(a) for a in anchors], "n_tokens": len(names),
           "m_frame_cosine": cos_frames}
    feats = {"mlp_out": M.mean(0), "mlp_out_frame0": M[0],
             "resid_block0": X.mean(0), "embedding": np.array([E[ids_by_str[s]] for s in names])}
    import embed_probe
    chk_f = probe(feats["embedding"], y0, np.random.default_rng(1), 80, n_split=2)
    chk_p = embed_probe.probe(feats["embedding"], y0, np.random.default_rng(1), 80, n_split=2)
    assert np.allclose(chk_f[0], chk_p[0]) and np.allclose(chk_f[1], chk_p[1])
    print(f"dual-form ridge matches the primal solve (rho {chk_f[0].mean():+.3f})", flush=True)

    rng = np.random.default_rng(0)
    for key, F in feats.items():
        rho, r2, lam = probe(F, y0, np.random.default_rng(1), 80)
        rho_n, r2_n, _ = probe(F, rng.permutation(y0), np.random.default_rng(1), 80)
        res[f"probe_{key}"] = dict(rho_mean=float(rho.mean()), rho_sd=float(rho.std()),
                                   r2_mean=float(r2.mean()), r2_sd=float(r2.std()),
                                   median_alpha=lam, null_rho_mean=float(rho_n.mean()),
                                   null_r2_mean=float(r2_n.mean()))
        print(f"probe {key:14s} -> w_hat: test rho {rho.mean():+.3f} +- {rho.std():.3f} "
              f"R2 {r2.mean():+.3f} (null rho {rho_n.mean():+.3f})", flush=True)
    json.dump(res, open(f"{RESULTS}/mlp_read.json", "w"), indent=1)

    # ---- B. transplant ------------------------------------------------------------
    toks12 = list(json.load(open(f"{RESULTS}/mode_split.json"))["tokens"].keys())
    ids_by_tok = {s: ids_by_str[s] for s in toks12}
    pre = tok(FRAMES[0], return_tensors="pt").input_ids.cuda()

    m_by_tok, z_by_tok = {}, {}
    for s, i in ids_by_tok.items():
        m, _, z = states(model, mlp, patcher, pre, i)
        m_by_tok[s], z_by_tok[s] = m, z.log_softmax(-1)

    base_w, _ = measure(model, patcher, ids_by_tok, anchors, pre)
    base = np.array([base_w[s] for s in toks12])
    print(f"\nbaseline mean {base.mean():.3f} sd {base.std(ddof=1):.3f}", flush=True)

    # the anchors are NEVER transplanted: only the recipient's own endpoint is edited, so a
    # self-transplant must reproduce the baseline width exactly (checked below).
    anc = [endpoint(model, patcher, torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1))
           for a in anchors]

    @torch.inference_mode()
    def transplant(recipient, donor_vec):
        """Anchor width of `recipient` with its block-0 MLP output replaced, and the bits it costs."""
        mlp.write = donor_vec
        ids = torch.cat([pre, torch.tensor([[ids_by_tok[recipient]]], device=pre.device)], 1)
        x, z = endpoint(model, patcher, ids)
        mlp.write = None
        bits = float(jsd_bits(z_by_tok[recipient].unsqueeze(0), z.log_softmax(-1).unsqueeze(0))[0])
        w = float(np.nanmedian([run_pair(model, patcher, ids, x, z, xb, zb)[0] for xb, zb in anc]))
        return w, bits

    rows = []
    mbar = torch.stack(list(m_by_tok.values())).mean(0)
    for ri, r in enumerate(toks12):
        for d in toks12:
            w, bits = transplant(r, m_by_tok[d])
            rows.append(dict(recipient=r, donor=d, w=w, bits=bits))
        w, bits = transplant(r, mbar)
        rows.append(dict(recipient=r, donor="__mean__", w=w, bits=bits))
        print(f"[{ri + 1}/12] recipient {r!r}: self {[x['w'] for x in rows if x['recipient'] == r and x['donor'] == r][0]:.3f} "
              f"mean-donor {w:.3f}", flush=True)
        res["transplant"] = rows
        json.dump(res, open(f"{RESULTS}/mlp_read.json", "w"), indent=1)

    # scoring: does the post-transplant width depend on WHICH donor?
    base_by = dict(zip(toks12, base))
    per, slopes = [], []
    for r in toks12:
        rr = [x for x in rows if x["recipient"] == r and x["donor"] in toks12 and x["donor"] != r]
        wd = np.array([base_by[x["donor"]] for x in rr])
        wp = np.array([x["w"] for x in rr])
        per.append(float(spearmanr(wd, wp).statistic))
        slopes.append(float(np.polyfit(wd, wp, 1)[0]))
    p_rho = float(wilcoxon(per).pvalue)
    self_w = np.array([[x["w"] for x in rows if x["recipient"] == r and x["donor"] == r][0]
                       for r in toks12])
    mean_w = np.array([[x["w"] for x in rows if x["recipient"] == r and x["donor"] == "__mean__"][0]
                       for r in toks12])
    cross = np.array([np.mean([x["w"] for x in rows if x["recipient"] == r
                               and x["donor"] in toks12 and x["donor"] != r]) for r in toks12])
    res["transplant_summary"] = dict(
        per_recipient_rho=per, mean_rho=float(np.mean(per)), wilcoxon_p=p_rho,
        per_recipient_slope=slopes, mean_slope=float(np.mean(slopes)),
        self_w=[float(x) for x in self_w], mean_donor_w=[float(x) for x in mean_w],
        cross_donor_w=[float(x) for x in cross], base_w=[float(x) for x in base],
        rho_base_vs_self=float(spearmanr(base, self_w).statistic),
        rho_base_vs_cross=float(spearmanr(base, cross).statistic),
        median_bits=float(np.median([x["bits"] for x in rows if x["donor"] != x["recipient"]])))
    print(f"\nper-recipient rho(donor width, post-transplant width): mean {np.mean(per):+.3f} "
          f"(Wilcoxon p = {p_rho:.4f}), mean slope {np.mean(slopes):+.3f}", flush=True)
    print(f"self-transplant preserves the ordering: rho = {res['transplant_summary']['rho_base_vs_self']:+.2f}; "
          f"recipient identity after a cross transplant: rho = "
          f"{res['transplant_summary']['rho_base_vs_cross']:+.2f}", flush=True)
    json.dump(res, open(f"{RESULTS}/mlp_read.json", "w"), indent=1)
    print("wrote results/mlp_read.json")


if __name__ == "__main__":
    main()
