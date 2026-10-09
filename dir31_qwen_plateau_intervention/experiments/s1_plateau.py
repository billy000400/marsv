"""S1: verify tokenization and baseline answers, reproduce the Japan->Germany plateau (layer-0 block output,
country position, 41 points), and fit the frozen country readout on non-capital prompts.
Writes results/s1_endpoints.json, results/s1_path.npz, results/probe.npz, results/s1_layers.csv."""
import csv
import json
import os

import numpy as np
import torch

from common import (CITY, COUNTRIES, DEV, N_LAYERS, N_T, OUT, PROBE_FAMILIES, TEMPLATES, ids_of, jsd_bits, run, setup,
                    slerp_lerp_norm, to_np)


def path_metrics(resid, ref0, ref1):
    """resid [T, L, d]; fixed references ref0/ref1 [L, d]. Returns d(t) [L, T], adjacent-step L2 [L, T-1],
    absolute distances to each reference [L, T]."""
    da = (resid - ref0[None]).norm(dim=-1).T
    db = (resid - ref1[None]).norm(dim=-1).T
    step = (resid[1:] - resid[:-1]).norm(dim=-1).T
    return da / (da + db), step, da, db


def crossing(ts, y, level):
    """First linearly interpolated t at which y crosses `level` (nan if never)."""
    s = np.sign(y - level)
    for k in range(len(ts) - 1):
        if s[k] != s[k + 1]:
            return float(ts[k] + (level - y[k]) / (y[k + 1] - y[k]) * (ts[k + 1] - ts[k]))
    return float("nan")


def main():
    tok, model = setup()
    tid = {w: tok(" " + w).input_ids for w in ["Japan", "Germany", "Tokyo", "Berlin", "is"]}
    assert all(len(v) == 1 for v in tid.values()), tid
    tid = {k: v[0] for k, v in tid.items()}

    # ---- baseline answers on every capital template ----
    endpoints = []
    for name, (prefix, bridge) in TEMPLATES.items():
        for c in COUNTRIES:
            p = prefix.format(c)
            pos = tok(p, return_tensors="pt").input_ids.shape[1] - 1
            ids = ids_of(tok, p + bridge)
            assert ids[0, pos].item() == tid[c] and ids[0, pos + 1].item() == tid["is"]
            r = run(model, ids, pos)
            lp_pos, lp = r["logits_pos"][0].log_softmax(-1), r["logits_last"][0].log_softmax(-1)
            gen = model.generate(ids, do_sample=False, max_new_tokens=8, temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.eos_token_id)
            endpoints.append(dict(
                template=name, prompt=p + bridge, country=c, country_position=pos,
                tokens=[tok.decode([i]) for i in ids[0].tolist()],
                p_is_after_country=lp_pos[tid["is"]].exp().item(), top1_after_country=tok.decode(lp_pos.argmax().item()),
                p_tokyo=lp[tid["Tokyo"]].exp().item(), p_berlin=lp[tid["Berlin"]].exp().item(),
                margin_tokyo_minus_berlin=(r["logits_last"][0, tid["Tokyo"]] - r["logits_last"][0, tid["Berlin"]]).item(),
                top1_after_bridge=tok.decode(lp.argmax().item()), completion=tok.decode(gen[0, ids.shape[1]:])))
            assert endpoints[-1]["top1_after_bridge"].strip() == CITY[c], endpoints[-1]
    for e in endpoints:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k != "tokens"})

    # ---- interpolation path on the original pair ----
    prefix, bridge = TEMPLATES["original"]
    pos = 3
    idsJ, idsG = ids_of(tok, prefix.format("Japan") + bridge), ids_of(tok, prefix.format("Germany") + bridge)
    rJ, rG = run(model, idsJ, pos), run(model, idsG, pos)
    hA, hB = rJ["resid"][0, 0], rG["resid"][0, 0]
    ts = np.linspace(0, 1, N_T)
    H, omega = slerp_lerp_norm(hA, hB, ts)
    r = run(model, idsJ.expand(N_T, -1), pos, patch0=H)
    resid = r["resid"]
    d, step, da, db = path_metrics(resid, rJ["resid"][0], rG["resid"][0])
    p_pos, p_last = r["logits_pos"].softmax(-1), r["logits_last"].softmax(-1)
    margin = r["logits_last"][:, tid["Tokyo"]] - r["logits_last"][:, tid["Berlin"]]
    # the bridge token's own layer-0 output comes from the Japan-token base; quantify the t=1 mismatch
    is0 = [run(model, x, pos, read_pos=pos + 1)["resid"][0, 0] for x in (idsJ, idsG)]
    mismatch = dict(
        cos_layer0_output_endpoints=float(torch.cos(torch.tensor(omega))),
        is_position_layer0_rel_diff=((is0[0] - is0[1]).norm() / is0[1].norm()).item(),
        t0_max_abs_logit_diff_vs_native_japan=(r["logits_last"][0] - rJ["logits_last"][0]).abs().max().item(),
        t1_max_abs_logit_diff_vs_native_germany=(r["logits_last"][-1] - rG["logits_last"][0]).abs().max().item(),
        t1_jsd_bits_vs_native_germany=jsd_bits(p_last[-1], rG["logits_last"][0].softmax(-1)).item(),
        t1_p_berlin_path_vs_native=[p_last[-1, tid["Berlin"]].item(), rG["logits_last"][0].softmax(-1)[tid["Berlin"]].item()],
        t1_country_resid_max_rel_diff=((resid[-1] - rG["resid"][0]).norm(dim=-1) / rG["resid"][0].norm(dim=-1)).max().item())
    print(mismatch)

    # ---- frozen country readout: mean-difference direction on explicit-country, non-capital prefixes ----
    feats, fam_of, lab = [], [], []
    for fam, temps in PROBE_FAMILIES.items():
        for t in temps:
            for c in COUNTRIES:
                ids = ids_of(tok, t.format(c))
                assert ids[0, -1].item() == tid[c] and ids.shape[1] > 1
                feats.append(run(model, ids, ids.shape[1] - 1)["resid"][0])
                fam_of.append(fam)
                lab.append(1.0 if c == "Japan" else -1.0)
    X, y = torch.stack(feats), torch.tensor(lab, device=DEV)  # [N, L, d]
    fam_of = np.array(fam_of)

    def fit(mask):
        m = torch.tensor(mask, device=DEV)
        muJ, muG = X[m & (y > 0)].mean(0), X[m & (y < 0)].mean(0)
        w = muJ - muG
        return w, (muJ + muG) / 2, (w * w).sum(-1) / 2

    def score(h, w, mid, scale):  # h [..., L, d] -> [..., L]; +1 = mean Japan training state, -1 = mean Germany
        return ((h - mid) * w).sum(-1) / scale

    lofo = {}
    for fam in PROBE_FAMILIES:
        s = score(X[torch.tensor(fam_of == fam, device=DEV)], *fit(fam_of != fam))
        yy = y[torch.tensor(fam_of == fam, device=DEV)]
        lofo[fam] = dict(acc=to_np(((s > 0) == (yy[:, None] > 0)).float().mean(0)),
                         mean_signed=to_np((s * yy[:, None]).mean(0)), min_signed=to_np((s * yy[:, None]).min(0).values))
    w, mid, scale = fit(np.ones(len(fam_of), bool))
    cap = {}
    for name in ["tune", "original"]:
        for c in COUNTRIES:
            ids = ids_of(tok, TEMPLATES[name][0].format(c))
            cap[f"{name}|{c}"] = to_np(score(run(model, ids, ids.shape[1] - 1)["resid"][0], w, mid, scale))
    probe_path = to_np(score(resid, w, mid, scale)).T  # [L, T]
    np.savez(os.path.join(OUT, "probe.npz"), w=to_np(w), mid=to_np(mid), scale=to_np(scale),
             lofo_acc=np.stack([lofo[f]["acc"] for f in PROBE_FAMILIES]),
             lofo_mean_signed=np.stack([lofo[f]["mean_signed"] for f in PROBE_FAMILIES]),
             lofo_min_signed=np.stack([lofo[f]["min_signed"] for f in PROBE_FAMILIES]),
             families=np.array(list(PROBE_FAMILIES)), capital_keys=np.array(list(cap)), capital_scores=np.stack(list(cap.values())))

    # ---- per-layer plateau summary ----
    d_np, step_np = to_np(d), to_np(step)
    rows = []
    for L in range(N_LAYERS):
        t10, t50, t90 = (crossing(ts, d_np[L], q) for q in (0.1, 0.5, 0.9))
        rows.append(dict(
            layer=L, t_d10=t10, t_d50=t50, t_d90=t90, width_10_90=t90 - t10,
            peak_step_share=float(step_np[L].max() / step_np[L].sum()), t_peak_step=float(ts[step_np[L].argmax()] + 0.0125),
            path_length=float(step_np[L].sum()), endpoint_distance=float(to_np(da)[L, -1]),
            probe_lofo_acc_min=float(min(lofo[f]["acc"][L] for f in PROBE_FAMILIES)),
            probe_lofo_min_signed=float(min(lofo[f]["min_signed"][L] for f in PROBE_FAMILIES)),
            probe_t0=float(probe_path[L, 0]), probe_t1=float(probe_path[L, -1]), probe_t_zero=crossing(ts, probe_path[L], 0.0),
            probe_capital_original_japan=float(cap["original|Japan"][L]), probe_capital_original_germany=float(cap["original|Germany"][L])))
    with open(os.path.join(OUT, "s1_layers.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for row in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    m_np = to_np(margin)
    summary = dict(token_ids=tid, endpoints=endpoints, mismatch=mismatch, slerp_angle_rad=omega,
                   layer0_norms=[hA.norm().item(), hB.norm().item()],
                   answer_margin_t_zero=crossing(ts, m_np, 0.0),
                   p_tokyo_t_half_of_start=crossing(ts, to_np(p_last[:, tid["Tokyo"]]), 0.5 * p_last[0, tid["Tokyo"]].item()),
                   p_is_min_max=[p_pos[:, tid["is"]].min().item(), p_pos[:, tid["is"]].max().item()],
                   next_token_jsd_bits_endpoints=jsd_bits(p_pos[0], p_pos[-1]).item(),
                   after_bridge_jsd_bits_endpoints=jsd_bits(p_last[0], p_last[-1]).item(),
                   probe_n_train=len(lab), probe_families={k: v for k, v in PROBE_FAMILIES.items()})
    json.dump(summary, open(os.path.join(OUT, "s1_endpoints.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "s1_path.npz"), ts=ts, resid=to_np(resid), d=d_np, step=step_np, da=to_np(da), db=to_np(db),
             ref0=to_np(rJ["resid"][0]), ref1=to_np(rG["resid"][0]), h0=to_np(H),
             p_is=to_np(p_pos[:, tid["is"]]), top1_pos=to_np(p_pos.argmax(-1)), top1_last=to_np(p_last.argmax(-1)),
             jsd_pos_to_t0=to_np(jsd_bits(p_pos, p_pos[:1])), jsd_last_to_t0=to_np(jsd_bits(p_last, p_last[:1])),
             p_tokyo=to_np(p_last[:, tid["Tokyo"]]), p_berlin=to_np(p_last[:, tid["Berlin"]]), margin=m_np, probe=probe_path)
    print({k: v for k, v in summary.items() if k not in ("endpoints", "probe_families")})
    print("top1 after bridge along path:", [tok.decode([i]) for i in p_last.argmax(-1).tolist()])
    print("top1 after country along path:", sorted({tok.decode([i]) for i in p_pos.argmax(-1).tolist()}))


if __name__ == "__main__":
    main()
