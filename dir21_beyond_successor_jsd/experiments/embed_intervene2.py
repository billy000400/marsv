"""Was the intervention null a step-size null? Calibrate the edit on the MODEL, not on the probe.

The first intervention sized each embedding edit so the probe's OWN prediction moved by a fixed number
of width units. Those steps turned out to shift the token's next-token distribution by only 0.0001
bits, so the model never noticed them. Here the step is grown along the probe direction until the
token's output distribution moves a fixed number of bits (0.05 / 0.1 / 0.2), in both signs, with a
random unit direction calibrated to the SAME output shift as the control. Writes results/intervene2.json.
"""
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import N_ANCHOR
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint, jsd_bits
from common import D18, RESULTS
from embed_intervene import measure, probe_direction

BITS = [0.05, 0.1, 0.2]        # output movement each calibrated step must produce
N_TOKENS = 12
LADDER = 0.05 * 1.6 ** np.arange(15)      # geometric ladder of step norms for the calibration scan

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


@torch.inference_mode()
def out_logp(model, tok, token_id):
    """The token's next-token log-probabilities in the three frames (no patching, no interpolation)."""
    out = []
    for frame in FRAMES:
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        ids = torch.cat([pre, torch.tensor([[token_id]], device=pre.device)], 1)
        out.append(model(ids).logits[0, -1].float().log_softmax(-1))
    return torch.stack(out)


def calibrate(model, tok, W, i, base_row, base_logp, direction):
    """Step norms along `direction` that move the token's output by each of BITS, from a log-log scan."""
    cs, bits = [], []
    for c in LADDER:
        W.data[i] = base_row + torch.tensor(c * direction, dtype=W.dtype, device=W.device)
        b = float(jsd_bits(base_logp, out_logp(model, tok, i)).mean())
        cs.append(float(c))
        bits.append(max(b, 1e-9))
        if b > 1.4 * BITS[-1]:
            break
    W.data[i] = base_row
    lc, lb = np.log(cs), np.log(bits)
    keep = np.argsort(lb)                       # JSD grows monotonically with step norm in practice
    return [float(np.exp(np.interp(np.log(t), lb[keep], lc[keep]))) for t in BITS], bits[-1]


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
    W = model.gpt_neox.embed_in.weight
    E = W.detach().float().cpu().numpy()

    w0 = {s: float(np.nanmedian(v["w"]))
          for s, v in json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}
    bank = sorted(w0)
    g, lam = probe_direction(np.array([E[ids_by_str[s]] for s in bank]),
                             np.array([w0[s] for s in bank]), np.random.default_rng(3))
    gn = float(np.linalg.norm(g))
    u = g / gn                                  # unit step of norm 1 moves the probe's prediction by gn
    row_norm = float(np.median(np.linalg.norm(E, axis=1)))
    print(f"probe direction: alpha={lam:g}, ||g||={gn:.4g}; a unit step changes the probe's "
          f"prediction by {gn:.3g} width units (median embedding row norm {row_norm:.4g})", flush=True)

    order = sorted(bank, key=lambda s: w0[s])
    sel = [order[j] for j in np.linspace(0, len(order) - 1, N_TOKENS).round().astype(int)]
    rng = np.random.default_rng(11)

    patcher = Patcher(model)
    rows = {}
    for k, s in enumerate(sel):
        i = ids_by_str[s]
        base_row = W.data[i].clone()
        base_logp = out_logp(model, tok, i)
        base_w, _ = measure(model, patcher, tok, anchors, i)
        r = dict(token_id=i, base_w=base_w, w0_ref=w0[s],
                 row_norm=float(np.linalg.norm(E[i])), probe=[], random=[])
        rdir = rng.normal(size=E.shape[1])
        rdir = rdir / np.linalg.norm(rdir)
        for tag, base_dir in (("probe", u), ("random", rdir)):
            for sign in (+1, -1):
                d = sign * base_dir
                cs, bmax = calibrate(model, tok, W, i, base_row, base_logp, d)
                for target, c in zip(BITS, cs):
                    W.data[i] = base_row + torch.tensor(c * d, dtype=W.dtype, device=W.device)
                    w_new, logp_new = measure(model, patcher, tok, anchors, i)
                    got = float(jsd_bits(base_logp, logp_new).mean())
                    W.data[i] = base_row
                    r[tag].append(dict(bits_target=target, bits_got=got, sign=sign, step_norm=c,
                                       dw=w_new - base_w, w=w_new,
                                       pred_dw=float(sign * c * gn) if tag == "probe" else 0.0))
        rows[s] = r
        json.dump(dict(partial=True, tokens=rows), open(f"{RESULTS}/intervene2.json", "w"), indent=1)
        pr = " ".join(f"{d['sign'] * d['bits_target']:+.2f}b->{d['dw']:+.3f}" for d in r["probe"])
        print(f"[{k + 1}/{N_TOKENS}] {s!r} base w {base_w:.3f} | probe {pr}", flush=True)
    patcher.close()

    def summarise(tag):
        recs = [d for s in rows for d in rows[s][tag]]
        out = dict(mean_abs_dw=float(np.mean([abs(d["dw"]) for d in recs])),
                   step_norm=float(np.median([d["step_norm"] for d in recs])),
                   bits_got=float(np.median([d["bits_got"] for d in recs])), by_bits={})
        for b in BITS:
            sub = [d for d in recs if d["bits_target"] == b]
            out["by_bits"][str(b)] = dict(
                mean_abs_dw=float(np.mean([abs(d["dw"]) for d in sub])),
                mean_dw_plus=float(np.mean([d["dw"] for d in sub if d["sign"] > 0])),
                mean_dw_minus=float(np.mean([d["dw"] for d in sub if d["sign"] < 0])),
                step_norm=float(np.median([d["step_norm"] for d in sub])),
                bits_got=float(np.median([d["bits_got"] for d in sub])))
        if tag == "probe":
            pred = np.array([d["pred_dw"] for d in recs])
            got = np.array([d["dw"] for d in recs])
            out["slope_vs_pred"] = float(np.polyfit(pred, got, 1)[0])
            out["rho_vs_pred"] = [float(x) for x in spearmanr(pred, got)[:2]]
            out["frac_sign_agree"] = float(np.mean(np.sign(got) == np.sign(pred)))
        else:
            sgn = np.array([d["sign"] for d in recs])
            got = np.array([d["dw"] for d in recs])
            out["frac_sign_agree"] = float(np.mean(np.sign(got) == sgn))
        return out

    res = dict(n_tokens=len(rows), bits=BITS, alpha=lam, grad_norm=gn, median_row_norm=row_norm,
               tokens=rows, probe=summarise("probe"), random=summarise("random"))
    # does the probe direction move width MORE than a random direction of the same behavioural size?
    pa = np.array([abs(d["dw"]) for s in rows for d in rows[s]["probe"]])
    ra = np.array([abs(d["dw"]) for s in rows for d in rows[s]["random"]])
    res["ratio_probe_over_random"] = float(pa.mean() / ra.mean())
    res["frac_probe_larger"] = float(np.mean(pa > ra))
    json.dump(res, open(os.path.join(RESULTS, "intervene2.json"), "w"), indent=1)

    for tag in ("probe", "random"):
        o = res[tag]
        print(f"{tag:7s}: median step norm {o['step_norm']:.3f} "
              f"(row norm {row_norm:.3f}), median output shift {o['bits_got']:.3f} bits, "
              f"mean |dw| {o['mean_abs_dw']:.4f}, sign agreement {o['frac_sign_agree']:.2f}")
        for b in BITS:
            d = o["by_bits"][str(b)]
            print(f"    {b:.2f} bits: |dw| {d['mean_abs_dw']:.4f}, "
                  f"dw(+) {d['mean_dw_plus']:+.4f}, dw(-) {d['mean_dw_minus']:+.4f}, "
                  f"step {d['step_norm']:.3f}")
    print(f"probe slope vs its own prediction {res['probe']['slope_vs_pred']:+.4f}, "
          f"rho {res['probe']['rho_vs_pred'][0]:+.3f} (p={res['probe']['rho_vs_pred'][1]:.1e})")
    print(f"|dw| probe / random = {res['ratio_probe_over_random']:.2f}, "
          f"probe larger in {res['frac_probe_larger']:.2f} of matched pairs")


if __name__ == "__main__":
    main()
