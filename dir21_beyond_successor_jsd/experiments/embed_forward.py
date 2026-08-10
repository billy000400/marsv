"""Zero-forward-pass screen: predict unseen pairs' widths from static embeddings alone.

The forward screen (forward_screen.py) measured anchor widths for 40 tokens outside the bank and
predicted their 780 pairs. This asks whether the measurement step can be dropped entirely: fit the
embedding probe on the 123 bank tokens, look up w_hat for the 40 new tokens from their embedding rows,
and score the same 718 gated pairs. Nothing here touches the new tokens' measurements.

Writes results/embed_forward.json.
"""
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from basin_probe import MODEL, REVISION
from common import D18, RESULTS
from embed_probe import ALPHAS, N_TRAIN, probe, ridge_fit

torch.set_num_threads(2)


def fit_probe(F, y, rng):
    """Ridge probe on all of F, ridge strength chosen by 5-fold CV. Returns predict()."""
    mu, sd = F.mean(0), F.std(0) + 1e-8
    X = np.hstack([np.ones((len(y), 1)), (F - mu) / sd])
    ym = y.mean()
    perm = rng.permutation(len(y))
    best, best_lam = -np.inf, ALPHAS[0]
    for lam in ALPHAS:
        pred = np.empty(len(y))
        for f in range(5):
            te = perm[f::5]
            tr = np.setdiff1d(perm, te)
            pred[te] = X[te] @ ridge_fit(X[tr], y[tr] - ym, lam) + ym
        s = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if s > best:
            best, best_lam = s, lam
    b = ridge_fit(X, y - ym, best_lam)
    return (lambda G: np.hstack([np.ones((len(G), 1)), (G - mu) / sd]) @ b + ym), float(best_lam), best


def main():
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]

    emb = json.load(open(f"{RESULTS}/embed.json"))
    fwd = json.load(open(f"{RESULTS}/forward.json"))
    w0 = {s: float(np.nanmedian(v["w"]))
          for s, v in json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval()
    E = model.gpt_neox.embed_in.weight.detach().float().numpy()
    del model

    bank = sorted(w0)
    Fb = np.array([E[ids_by_str[s]] for s in bank])
    yb = np.array([w0[s] for s in bank])
    predict, lam, in_bank_r2 = fit_probe(Fb, yb, np.random.default_rng(3))
    print(f"probe fitted on {len(bank)} bank tokens, alpha={lam:g}, bank CV R2={in_bank_r2:+.3f}")

    # baseline for the probe: embedding norm alone, same 80/43 splits as embed_probe.py
    rho_n, r2_n, _ = probe(np.linalg.norm(Fb, axis=1)[:, None], yb,
                           np.random.default_rng(1), N_TRAIN)
    print(f"norm-only probe -> w_block0: test rho {rho_n.mean():+.3f} +- {rho_n.std():.3f}, "
          f"test R2 {r2_n.mean():+.3f}")

    new = fwd["new_tokens"]
    new_ids = [tok(s, add_special_tokens=False).input_ids for s in new]
    assert all(len(i) == 1 for i in new_ids), "a new token is not a single BPE token"
    Fn = np.array([E[i[0]] for i in new_ids])
    w_look = dict(zip(new, predict(Fn)))

    # how well the lookup reproduces the measured anchor width of tokens it never saw
    meas = np.array([fwd["anchor_width"][s] for s in new])
    rho_tok = spearmanr(np.array([w_look[s] for s in new]), meas)

    # pair-level mapping, fitted on bank pairs with OUT-OF-FOLD probe features, then frozen
    from common import load
    t, _, _ = load()
    m = t["out_jsd_min"] >= 0.2
    p_hat = emb["probe_pred"]
    su = (np.array([p_hat[x] for x in t["a_str"]])[m]
          + np.array([p_hat[x] for x in t["b_str"]])[m])
    y = t["w"][m]
    beta1, beta0 = np.polyfit(su, y, 1)
    print(f"bank mapping from probe lookups: w = {beta0:+.4f} + {beta1:.4f} * (w_u + w_v)")

    rows = [r for r in fwd["rows"] if r["n_valid"] == 3 and r["out_jsd_min"] >= 0.2]
    obs = np.array([r["w"] for r in rows])
    pred = np.array([beta0 + beta1 * (w_look[r["a"]] + w_look[r["b"]]) for r in rows])
    r2 = float(1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())
    rho = spearmanr(pred, obs)
    mae = float(np.abs(obs - pred).mean())
    q = np.quantile(pred, [1 / 3, 2 / 3])
    terc = [float(np.median(obs[np.digitize(pred, q) == k])) for k in range(3)]

    res = dict(n_bank=len(bank), alpha=lam, bank_cv_r2=float(in_bank_r2),
               probe_norm_only=dict(rho_mean=float(rho_n.mean()), rho_sd=float(rho_n.std()),
                                    r2_mean=float(r2_n.mean()), r2_sd=float(r2_n.std())),
               n_new_tokens=len(new), n_scored=len(rows),
               rho_lookup_vs_measured_anchor_width=[float(x) for x in rho_tok[:2]],
               beta0=float(beta0), beta1=float(beta1),
               r2_forward=r2, rho_forward=[float(x) for x in rho[:2]], mean_abs_err=mae,
               terciles=terc,
               measured_screen=dict(r2=fwd["r2_forward"], rho=fwd["rho_forward"][0],
                                    mae=fwd["mean_abs_err"]),
               w_lookup={s: float(v) for s, v in w_look.items()})
    json.dump(res, open(os.path.join(RESULTS, "embed_forward.json"), "w"), indent=1)
    print(f"lookup vs measured anchor width on 40 unseen tokens: rho {rho_tok[0]:+.3f} "
          f"(p={rho_tok[1]:.1e})")
    print(f"pairs scored {len(rows)}: R2 {r2:+.3f}, rho {rho[0]:+.3f}, MAE {mae:.3f}, "
          f"terciles {['%.3f' % v for v in terc]}")
    print(f"(measured screen on the same pairs: R2 {fwd['r2_forward']:+.3f}, "
          f"rho {fwd['rho_forward'][0]:+.3f}, MAE {fwd['mean_abs_err']:.3f})")


if __name__ == "__main__":
    main()
