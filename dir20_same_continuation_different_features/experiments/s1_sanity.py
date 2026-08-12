"""S1 (fresh confirmatory plan): validate the block-0 interpolation harness in GPT-2 Large.

Runs Matthew's two contrasts -- `The house was big`/`in` (plateau case) and `big`/`large`
(smooth case) -- through the frozen protocol of PLAN.md: block-0 resid_post rescaled SLERP at
the final token, 101 alphas, final-token logit readout, relative distance d(alpha), width w_TV.

Gate: endpoint reconstruction error < 1e-4 and w_TV(big,in) < w_TV(big,large).

Writes results/s1_sanity.json and plots/matthew_sanity.png.
"""
import json
import os

import numpy as np
import torch

from common import PLOTS, RESULTS, load, tokenize_pair
from run_interp import clean_run, jsd, rel_dist, slerp_lerp_norm, sweep, width_10_90

N_ALPHA = 101
CASES = [
    ("big_in", "big / in (plateau case)", "The house was", " big", " in"),
    ("big_large", "big / large (smooth case)", "The house was", " big", " large"),
]


def w_tv(alphas, d):
    """Total-variation width: alpha(c=0.75) - alpha(c=0.25) of the normalized cumulative variation."""
    tv = np.abs(np.diff(d))
    c = np.concatenate([[0.0], np.cumsum(tv) / tv.sum()])
    return float(np.interp(0.75, c, alphas) - np.interp(0.25, c, alphas))


def nonmono(d):
    """Fraction of the total variation of d that runs backwards (0 = monotone)."""
    dd = np.diff(d)
    return float(np.abs(dd[dd < 0]).sum() / np.abs(dd).sum())


def main():
    alphas = np.linspace(0, 1, N_ALPHA)
    tok, m = load("gpt2-large")
    out, curves = {}, {}

    for key, label, prefix, a_str, b_str in CASES:
        ok, ida, idb = tokenize_pair(tok, prefix, a_str, b_str)
        assert ok, f"{key}: tokenization is not a single differing final token"

        reca, lga = clean_run(m, ida)
        recb, lgb = clean_run(m, idb)
        pa, pb = torch.softmax(lga, -1), torch.softmax(lgb, -1)

        ha, hb = reca[0], recb[0]
        vecs, omega, cos = slerp_lerp_norm(ha, hb, alphas)
        _, lg = sweep(m, ida, vecs)

        # Endpoint reconstruction: alpha=0 must reproduce prompt A's logits, alpha=1 prompt B's.
        err_a = float((lg[0] - lga.cpu()).norm() / lga.cpu().norm())
        err_b = float((lg[-1] - lgb.cpu()).norm() / lgb.cpu().norm())

        d = rel_dist(lg, lga.cpu().unsqueeze(0), lgb.cpu().unsqueeze(0))
        curves[key] = d
        out[key] = dict(
            label=label, prefix=prefix, tok_a=a_str, tok_b=b_str,
            jsd=jsd(pa, pb), w_tv=w_tv(alphas, d), w10_90=width_10_90(alphas, d),
            nonmono=nonmono(d), endpoint_rel_err=[err_a, err_b],
            omega_rad=omega, cos_h0=cos,
            norm_a=float(ha.norm()), norm_b=float(hb.norm()),
            top_a=[tok.convert_ids_to_tokens(int(i)) for i in pa.topk(5).indices],
            top_b=[tok.convert_ids_to_tokens(int(i)) for i in pb.topk(5).indices],
        )
        print(f"{key:10s} JSD={out[key]['jsd']:.4f} w_TV={out[key]['w_tv']:.4f} "
              f"w10-90={out[key]['w10_90']} err={err_a:.2e}/{err_b:.2e}")

    gate = dict(
        endpoint_err_ok=bool(max(max(v["endpoint_rel_err"]) for v in out.values()) < 1e-4),
        ordering_ok=bool(out["big_in"]["w_tv"] < out["big_large"]["w_tv"]),
    )
    gate["passed"] = bool(gate["endpoint_err_ok"] and gate["ordering_ok"])
    out["_gate"] = gate
    print("GATE", gate)

    with open(os.path.join(RESULTS, "s1_sanity.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.join(RESULTS, "s1_sanity.npz"), alphas=alphas, **curves)

    plot(alphas, curves, out)
    return gate


CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]


def plot(alphas, curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    style = dict(big_in=("-", "o", CVD[0]), big_large=("--", "s", CVD[1]))
    for key, (ls, mk, c) in style.items():
        e = out[key]
        ax[0].plot(alphas, curves[key], ls, color=c, marker=mk, markevery=10, ms=4,
                   label=f"{e['tok_a'].strip()} / {e['tok_b'].strip()}  "
                         f"(JSD={e['jsd']:.3f}, $w_{{TV}}$={e['w_tv']:.3f})")
    ax[0].plot(alphas, alphas, ":", color="0.45", lw=1.4, label="linear reference")
    ax[0].set_xlabel(r"interpolation position $\alpha$")
    ax[0].set_ylabel(r"relative distance $d(\alpha)$")
    ax[0].set_title("Block-0 final-token interpolation, GPT-2 Large")
    ax[0].legend(fontsize=8, loc="upper left")

    for key, (ls, mk, c) in style.items():
        d = curves[key]
        tv = np.abs(np.diff(d))
        cc = np.concatenate([[0.0], np.cumsum(tv) / tv.sum()])
        ax[1].plot(alphas, cc, ls, color=c, marker=mk, markevery=10, ms=4,
                   label=f"{out[key]['tok_a'].strip()} / {out[key]['tok_b'].strip()}")
    for t in (0.25, 0.75):
        ax[1].axhline(t, color="0.45", ls=":", lw=1.2)
    ax[1].plot(alphas, alphas, ":", color="0.45", lw=1.4, label="linear reference")
    ax[1].set_xlabel(r"interpolation position $\alpha$")
    ax[1].set_ylabel(r"cumulative variation $c(\alpha)$")
    ax[1].set_title(r"$w_{TV}=\alpha(c{=}0.75)-\alpha(c{=}0.25)$")
    ax[1].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "matthew_sanity.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
