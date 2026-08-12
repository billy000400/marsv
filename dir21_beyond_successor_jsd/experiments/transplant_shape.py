"""Does the m_u transplant move the recipient's curve SHAPE, or only its width?

Pattern 23 wrote a donor token's block-0 MLP output m_u over a recipient's and found the recipient's
anchor width follows the donor's almost completely (per-recipient rho +0.968, slope +0.913). It scored
width alone. Patterns 41/42 then found that what the static embedding one block earlier actually ranks
is curve SHAPE -- edge drift E = d(0.1) + (1 - d(0.9)) -- and that width with shape removed is at
chance. So the transplant needs the same second statistic.

This repeats pattern 23 unchanged -- same 12 tokens, same 6 anchors, same frame, same hook, same write
-- and records E on every curve alongside w. Both statistics come from the SAME forward passes, so the
comparison is matched by construction and costs no extra compute.

Scoring, per recipient, across its 11 cross donors:
  A. marginal   -- Spearman rho and OLS slope of the post-transplant value on the donor's own baseline
                   value, separately for w and for E. Slope ~1 = the donor's value transports whole.
  B. partial    -- E and w are correlated across tokens, so a marginal slope on one can be inherited
                   from the other. Rank-partial correlations rho(donor E, post E | donor w) and
                   rho(donor w, post w | donor E) say which donor property actually drives which
                   recipient property.
  C. reliability-- split-half (3 anchors vs 3) of each baseline statistic over the 12 tokens, with a
                   token-bootstrap interval, because a noisy regressor attenuates a slope and could
                   fake a difference between the two.

Writes results/transplant_shape.json.
"""
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import N_ANCHOR
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint, jsd_bits
from common import D18, RESULTS
from envwidth import run_pair
from mlp_read import MLPOut, states

SEED = 0
N_BOOT = 2000

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def partial_rank(x, y, z):
    """Spearman correlation of x and y with z's ranking regressed out of both."""
    rx, ry, rz = (np.argsort(np.argsort(v)).astype(float) for v in (x, y, z))
    Z = np.vstack([np.ones_like(rz), rz]).T
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(spearmanr(ex, ey).statistic)


def split_half(per_curve, rng):
    """Reliability of a per-token median over 6 anchors: rho(median of 3, median of 3), bootstrapped.

    per_curve is (n_tokens, 6). The 3-vs-3 split is over anchors and is the same for every token, so
    the estimate is a correlation between two half-measurements of the same 12 tokens. The bootstrap
    resamples TOKENS, which is the axis the correlation is taken over.
    """
    a = np.median(per_curve[:, :3], axis=1)
    b = np.median(per_curve[:, 3:], axis=1)
    r = float(spearmanr(a, b).statistic)
    n = len(a)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(a[idx])) > 2:
            boot.append(spearmanr(a[idx], b[idx]).statistic)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(r=r, ci=[float(lo), float(hi)], n_boot=len(boot))


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

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    patcher = Patcher(model)
    mlp = MLPOut(model)

    toks12 = list(json.load(open(f"{RESULTS}/mode_split.json"))["tokens"].keys())
    ids_by_tok = {s: ids_by_str[s] for s in toks12}
    pre = tok(FRAMES[0], return_tensors="pt").input_ids.cuda()

    m_by_tok, z_by_tok = {}, {}
    for s, i in ids_by_tok.items():
        m, _, z = states(model, mlp, patcher, pre, i)
        m_by_tok[s], z_by_tok[s] = m, z.log_softmax(-1)

    anc = [endpoint(model, patcher, torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1))
           for a in anchors]

    @torch.inference_mode()
    def curves(recipient, donor_vec):
        """Per-anchor w and E for one recipient, with its block-0 MLP output optionally replaced."""
        mlp.write = donor_vec
        ids = torch.cat([pre, torch.tensor([[ids_by_tok[recipient]]], device=pre.device)], 1)
        x, z = endpoint(model, patcher, ids)
        mlp.write = None
        bits = float(jsd_bits(z_by_tok[recipient].unsqueeze(0), z.log_softmax(-1).unsqueeze(0))[0])
        ms = [run_pair(model, patcher, ids, x, z, xb, zb) for xb, zb in anc]
        return ([m["w"] for m in ms], [m["edge"] for m in ms], bits)

    out_path = os.path.join(RESULTS, "transplant_shape.json")
    res = {"model": MODEL, "revision": REVISION, "frame": FRAMES[0], "tokens": toks12,
           "anchors": [tok.convert_ids_to_tokens(a) for a in anchors]}

    # ---- baselines: no write ------------------------------------------------------
    base = {}
    for s in toks12:
        w, e, _ = curves(s, None)
        base[s] = dict(w=w, edge=e)
        print(f"base {s!r}: w {np.nanmedian(w):.3f}  E {np.median(e):.4f}", flush=True)
    res["base"] = base
    json.dump(res, open(out_path, "w"), indent=1)

    # sanity: the widths must reproduce pattern 23's stored baseline exactly (same curves, same code
    # path through curve_metrics), which is what makes the two runs comparable.
    old = json.load(open(f"{RESULTS}/mlp_read.json"))["transplant_summary"]["base_w"]
    new = [float(np.nanmedian(base[s]["w"])) for s in toks12]
    res["base_w_matches_pattern23"] = bool(np.allclose(old, new, atol=1e-9))
    print(f"baseline widths reproduce pattern 23: {res['base_w_matches_pattern23']} "
          f"(max |diff| {np.abs(np.array(old) - np.array(new)).max():.2e})", flush=True)

    # ---- transplant ---------------------------------------------------------------
    rows = []
    mbar = torch.stack(list(m_by_tok.values())).mean(0)
    for ri, r in enumerate(toks12):
        for d in toks12:
            w, e, bits = curves(r, m_by_tok[d])
            rows.append(dict(recipient=r, donor=d, w=w, edge=e, bits=bits))
        w, e, bits = curves(r, mbar)
        rows.append(dict(recipient=r, donor="__mean__", w=w, edge=e, bits=bits))
        got = [x for x in rows if x["recipient"] == r and x["donor"] == r][0]
        print(f"[{ri + 1}/{len(toks12)}] recipient {r!r}: self w {np.nanmedian(got['w']):.3f} "
              f"E {np.median(got['edge']):.4f} | mean-donor w {np.nanmedian(w):.3f} "
              f"E {np.median(e):.4f}", flush=True)
        res["transplant"] = rows
        json.dump(res, open(out_path, "w"), indent=1)

    # ---- scoring ------------------------------------------------------------------
    med = lambda row, k: float(np.nanmedian(row[k]))
    bw = np.array([float(np.nanmedian(base[s]["w"])) for s in toks12])
    be = np.array([float(np.median(base[s]["edge"])) for s in toks12])
    res["rho_base_w_vs_base_E"] = float(spearmanr(bw, be).statistic)
    print(f"\nacross the 12 tokens, baseline w and E rank at {res['rho_base_w_vs_base_E']:+.3f}",
          flush=True)

    bw_by, be_by = dict(zip(toks12, bw)), dict(zip(toks12, be))
    score = {}
    for key, base_by in (("w", bw_by), ("edge", be_by)):
        per_rho, per_slope, per_partial = [], [], []
        for r in toks12:
            rr = [x for x in rows if x["recipient"] == r and x["donor"] in toks12
                  and x["donor"] != r]
            dv = np.array([base_by[x["donor"]] for x in rr])              # donor's own baseline
            pv = np.array([med(x, key) for x in rr])                      # recipient's post value
            other = np.array([(be_by if key == "w" else bw_by)[x["donor"]] for x in rr])
            per_rho.append(float(spearmanr(dv, pv).statistic))
            per_slope.append(float(np.polyfit(dv, pv, 1)[0]))
            per_partial.append(partial_rank(dv, pv, other))
        # control: donor fixed, recipient varying. If the part of the state the transplant leaves
        # alone carried the property, this would be the large number instead.
        per_recip = []
        for d in toks12:
            rr = [x for x in rows if x["donor"] == d and x["recipient"] != d]
            per_recip.append(float(spearmanr([base_by[x["recipient"]] for x in rr],
                                             [med(x, key) for x in rr]).statistic))
        self_v = np.array([med([x for x in rows if x["recipient"] == r and x["donor"] == r][0], key)
                           for r in toks12])
        cross_v = np.array([np.mean([med(x, key) for x in rows if x["recipient"] == r
                                     and x["donor"] in toks12 and x["donor"] != r])
                            for r in toks12])
        score[key] = dict(
            per_recipient_rho=per_rho, mean_rho=float(np.mean(per_rho)),
            wilcoxon_p=float(wilcoxon(per_rho).pvalue),
            per_recipient_slope=per_slope, mean_slope=float(np.mean(per_slope)),
            per_recipient_partial=per_partial, mean_partial=float(np.mean(per_partial)),
            wilcoxon_partial_p=float(wilcoxon(per_partial).pvalue),
            recip_rho=per_recip, mean_recip_rho=float(np.mean(per_recip)),
            wilcoxon_recip_p=float(wilcoxon(per_recip).pvalue),
            base=[float(x) for x in (bw if key == "w" else be)],
            self_v=[float(x) for x in self_v], cross_v=[float(x) for x in cross_v],
            rho_base_vs_self=float(spearmanr(bw if key == "w" else be, self_v).statistic),
            rho_base_vs_cross=float(spearmanr(bw if key == "w" else be, cross_v).statistic))
        s = score[key]
        print(f"[{key}] per-recipient rho {s['mean_rho']:+.3f} (p {s['wilcoxon_p']:.4f}), "
              f"slope {s['mean_slope']:+.3f}, partial rho {s['mean_partial']:+.3f} "
              f"(p {s['wilcoxon_partial_p']:.4f}), self-check rho {s['rho_base_vs_self']:+.2f}, "
              f"recipient-dependence rho {s['mean_recip_rho']:+.3f} (p {s['wilcoxon_recip_p']:.3f})",
              flush=True)
    res["score"] = score

    # paired over the same 12 recipients: is one statistic transported more completely?
    res["paired"] = {
        stat: dict(w=float(np.mean(score["w"][f"per_recipient_{stat}"])),
                   edge=float(np.mean(score["edge"][f"per_recipient_{stat}"])),
                   n_w_greater=int(np.sum(np.array(score["w"][f"per_recipient_{stat}"])
                                          > np.array(score["edge"][f"per_recipient_{stat}"]))),
                   wilcoxon_p=float(wilcoxon(score["w"][f"per_recipient_{stat}"],
                                             score["edge"][f"per_recipient_{stat}"]).pvalue))
        for stat in ("rho", "slope", "partial")}
    for stat, v in res["paired"].items():
        print(f"paired {stat:7s}: w {v['w']:+.3f} vs E {v['edge']:+.3f} "
              f"({v['n_w_greater']}/12 recipients favour w, p {v['wilcoxon_p']:.4f})", flush=True)

    rng = np.random.default_rng(SEED)
    res["reliability"] = {
        "w": split_half(np.array([base[s]["w"] for s in toks12]), rng),
        "edge": split_half(np.array([base[s]["edge"] for s in toks12]), rng)}
    for k, v in res["reliability"].items():
        print(f"baseline {k:4s} split-half reliability R = {v['r']:.3f} "
              f"[{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]", flush=True)

    res["median_bits"] = float(np.median([x["bits"] for x in rows if x["donor"] != x["recipient"]]))
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
