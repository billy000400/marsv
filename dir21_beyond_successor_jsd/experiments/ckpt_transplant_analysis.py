"""Score the cross-checkpoint transplant: did writing m_u in move the width ordering?

Each condition is summarised by two rank agreements on the same 123 tokens -- with the FINAL
checkpoint's own measured widths (the ordering we are asking about) and with the RECIPIENT's own
unmodified widths (the ordering the recipient already had). A transplant that installs the trait
raises the first and lowers the second; a perturbation that merely damages the measurement lowers
both, which is what the shuffled-donor control is there to show.

Uncertainty on a difference between two agreements measured on the same tokens comes from a paired
bootstrap over tokens (2,000 resamples), which needs no distributional assumption.

Writes results/ckpt_transplant_summary.json.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from checkpoints_analysis import partial_rho, reliability
from common import RESULTS

N_BOOT = 2000
CONDS = ["base", "self", "donor", "donor_scaled", "shuffle", "shuffle_scaled"]


def boot_rho(a, c, rng, n=N_BOOT):
    """Bootstrap CI for rho(a, c) itself, resampling tokens."""
    r = [spearmanr(a[i], c[i]).statistic
         for i in (rng.integers(0, len(c), len(c)) for _ in range(n))]
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def boot_diff(a, b, c, rng, n=N_BOOT):
    """Bootstrap CI for rho(a, c) - rho(b, c), resampling the shared tokens."""
    d = []
    for _ in range(n):
        i = rng.integers(0, len(c), len(c))
        if len(np.unique(c[i])) < 5:
            continue
        d.append(spearmanr(a[i], c[i]).statistic - spearmanr(b[i], c[i]).statistic)
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    d = json.load(open(f"{RESULTS}/ckpt_transplant.json"))
    early, final = d["early"], d["final"]
    runs = d["runs"]
    names = sorted(runs[f"{early}<-{final}"]["base"]["w"])
    get = lambda tag, c: np.array([runs[tag][c]["w"][s] for s in names])

    rng = np.random.default_rng(0)
    g = json.load(open(f"{RESULTS}/ckpt_transplant_geom.json"))
    norm = {r: np.array([dict(zip(g["tokens"], g[f"norm_{k}"]))[s] for s in names])
            for r, k in ((final, "final"), (early, "early"))}
    out = dict(model=d["model"], early=early, final=final, tokens=names,
               m_norm=d["m_norm"], geometry=g, rows=[])
    for recip, don in ((early, final), (final, early)):
        tag = f"{recip}<-{don}"
        if "base" not in runs.get(tag, {}):
            continue
        base = get(tag, "base")                       # the ordering the recipient already had
        target = get(f"{don}<-{recip}", "base")       # the donor checkpoint's own ordering
        rel_t = reliability(runs[f"{don}<-{recip}"]["base"]["w_raw"], names)[1]
        for c in CONDS:
            if c not in runs[tag]:
                continue
            v = get(tag, c)
            # the length of the vector actually written into this recipient, per token
            dn = norm[don] if c in ("donor", "donor_scaled") else None
            rel = reliability(runs[tag][c]["w_raw"], names)[1]
            rt, rb = spearmanr(v, target), spearmanr(v, base)
            ceil = float(np.sqrt(max(rel, 0) * max(rel_t, 0)))
            row = dict(direction=tag, cond=c, kappa=runs[tag]["kappa"],
                       median_w=float(np.median(v)), sd_w=float(v.std(ddof=1)),
                       valid=runs[tag][c]["valid_frac"], bits=runs[tag][c]["median_bits"],
                       mean_abs_dw=float(np.mean(np.abs(v - base))),
                       reliability=rel, ceiling=ceil,
                       rho_donor=float(rt.statistic), p_donor=float(rt.pvalue),
                       rho_donor_ci=boot_rho(v, target, rng),
                       disattenuated=float(rt.statistic / ceil) if ceil > 0 else float("nan"),
                       rho_recipient_base=float(rb.statistic))
            # agreement with the donor's ordering after removing the one the recipient already had
            row["partial_donor"], row["partial_donor_p"] = partial_rho(v, target, [base])
            # and after also removing the donor vector's LENGTH, so a pure magnitude effect --
            # bigger vector in, wider measured curve out -- cannot produce the agreement
            if dn is not None:
                row["rho_donor_norm"] = float(spearmanr(v, dn).statistic)
                row["partial_donor_no_norm"] = partial_rho(v, target, [base, dn])[0]
            if c not in ("base", "self"):
                ctrl = "shuffle" if c == "donor" else "shuffle_scaled"
                if ctrl in runs[tag] and c != ctrl:
                    row["vs_shuffle"] = boot_diff(v, get(tag, ctrl), target, rng)
                row["vs_base"] = boot_diff(v, base, target, rng)
            out["rows"].append(row)
            print(f"{tag:>26} {c:>14} med {row['median_w']:.3f} sd {row['sd_w']:.3f} "
                  f"rel {rel:.3f} | vs donor order {row['rho_donor']:+.3f} "
                  f"(partial {row['partial_donor']:+.3f}) | vs own base "
                  f"{row['rho_recipient_base']:+.3f} | |dw| {row['mean_abs_dw']:.4f} "
                  f"bits {row['bits']:.4f}", flush=True)
    json.dump(out, open(f"{RESULTS}/ckpt_transplant_summary.json", "w"), indent=1)
    print("\nwrote results/ckpt_transplant_summary.json")
    for r in out["rows"]:
        if "vs_shuffle" in r:
            m, lo, hi = r["vs_shuffle"]
            print(f"{r['direction']} {r['cond']}: agreement with the donor's ordering minus its "
                  f"shuffled control {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
