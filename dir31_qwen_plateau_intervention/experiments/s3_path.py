"""S2d + S3 with the frozen edit (results/frozen_edit.json); nothing is reselected here.
(a) Reserved capital prompts: margin, probabilities, greedy completion, P(' is'), country readout; 3 random controls.
(b) Country-context prompts (currency / language): does the edit change those answers?
(c) The 41-point Japan->Germany path under baseline, Tokyo-directed, Berlin-directed and control edits.
Writes results/s2_frozen_eval.csv, results/s2_context.csv, results/s3_path.npz, results/s3_summary.csv."""
import csv
import json
import os

import numpy as np
import torch

from common import (CITY, COUNTRIES, DEV, N_LAYERS, N_T, OUT, PROBE_FAMILIES, TEMPLATES, Edit, d30, edited, ids_of, jsd_bits, run,
                    setup, slerp_lerp_norm, to_np)
from s1_plateau import crossing, path_metrics
from s2_scan import DIRECTIONS, load_slices

N_CTRL = 3
BETAS = [0.5, 1.0, 2.0]


def main():
    tok, model = setup()
    cfg = json.load(open(os.path.join(OUT, "frozen_edit.json")))
    assert cfg["gate"] == "passed", cfg
    slices = load_slices()
    tid = {w: tok(" " + w).input_ids[0] for w in ["Tokyo", "Berlin", "is"]}
    pr = np.load(os.path.join(OUT, "probe.npz"))
    w, mid, scale = (torch.tensor(pr[k], device=DEV) for k in ("w", "mid", "scale"))
    probe = lambda h: ((h - mid) * w).sum(-1) / scale  # [..., L, d] -> [..., L]
    last_edit = max(cfg["layers"])

    def edits_for(dname, pos, ctrl_seed=None, beta=None):
        out = []
        for L, e in cfg["directions"][dname]["layers"].items():
            sl, idx = slices[int(L)], e["slice_idx"]
            ctrl = None
            if ctrl_seed is not None:
                g = torch.Generator().manual_seed(1000 * ctrl_seed + int(L))
                ctrl = sl["rand_dec"][torch.randperm(sl["rand_dec"].shape[0], generator=g)[:len(idx)].to(DEV)]
            out.append(Edit(int(L), pos, sl["W_enc"][idx], sl["b_enc"][idx], sl["W_dec"][idx],
                            torch.tensor(e["a_donor"], device=DEV), cfg["beta"] if beta is None else beta, ctrl))
        return out

    arms = [("none", None)] + [("edit", None)] + [(f"control{s}", s) for s in range(1, N_CTRL + 1)]

    def gen(ids, edits):
        with edited(model, edits):
            g = model.generate(ids, do_sample=False, max_new_tokens=8, temperature=None, top_p=None, top_k=None,
                               pad_token_id=tok.eos_token_id, use_cache=False)
        return tok.decode(g[0, ids.shape[1]:])

    # ---- frozen features: are they active on the country word in NON-capital prefixes too? ----
    feat_rows = []
    for dname in DIRECTIONS:
        for L, e in cfg["directions"][dname]["layers"].items():
            sl, idx = slices[int(L)], e["slice_idx"]
            acts = {c: [] for c in COUNTRIES}
            for temps in PROBE_FAMILIES.values():
                for t in temps:
                    for c in COUNTRIES:
                        ids = ids_of(tok, t.format(c))
                        cache = {}
                        h = model.model.layers[int(L)].mlp.register_forward_hook(lambda m, a, o: cache.update(x=a[0][0, -1]))
                        model(ids)
                        h.remove()
                        acts[c].append(torch.relu(cache["x"] @ sl["W_enc"][idx].T + sl["b_enc"][idx]))
            aJ, aG = torch.stack(acts["Japan"]), torch.stack(acts["Germany"])
            for j, f in enumerate(e["features"]):
                feat_rows.append(dict(direction=dname, layer=int(L), rank=j + 1, feature=f, a_donor=e["a_donor"][j],
                                      a_recipient_tune=e["a_recipient_tune"][j],
                                      noncapital_active_japan=int((aJ[:, j] > 0).sum()), noncapital_active_germany=int((aG[:, j] > 0).sum()),
                                      noncapital_mean_japan=aJ[:, j].mean().item(), noncapital_mean_germany=aG[:, j].mean().item(),
                                      n_noncapital_prefixes=len(aJ)))
    with open(os.path.join(OUT, "frozen_features.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(feat_rows[0]))
        wr.writeheader()
        wr.writerows(feat_rows)
    for r in feat_rows:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})

    # ---- (a) capital prompts: tuning pair for reference, reserved pairs for evaluation ----
    rows = []
    for name, (prefix, bridge) in TEMPLATES.items():
        for dname, (rec, don, sign) in DIRECTIONS.items():
            pos = tok(prefix.format(rec), return_tensors="pt").input_ids.shape[1] - 1
            ids = ids_of(tok, prefix.format(rec) + bridge)
            for beta, (arm, seed) in [(0.0, arms[0])] + [(b, a) for b in BETAS for a in arms[1:]]:
                ed = [] if arm == "none" else edits_for(dname, pos, seed, beta)
                r = run(model, ids, pos, edits=ed)
                lp, lpp = r["logits_last"][0].log_softmax(-1), r["logits_pos"][0].log_softmax(-1)
                margin = (r["logits_last"][0, tid["Tokyo"]] - r["logits_last"][0, tid["Berlin"]]).item()
                s = probe(r["resid"][0])
                rows.append(dict(
                    template=name, role="tuning" if name == "tune" else "reserved", prompt=prefix.format(rec) + bridge,
                    direction=dname, arm=arm, beta=beta, frozen_strength=arm == "none" or beta == cfg["beta"],
                    margin=margin, margin_toward_target=sign * margin, reversed=sign * margin > 0,
                    p_tokyo=lp[tid["Tokyo"]].exp().item(), p_berlin=lp[tid["Berlin"]].exp().item(),
                    top1=tok.decode(lp.argmax().item()), top1_is_target=tok.decode(lp.argmax().item()).strip() == CITY[don],
                    completion=gen(ids, ed), p_is_after_country=lpp[tid["is"]].exp().item(),
                    readout_layer_after_edit=s[last_edit].item(), readout_layer26=s[26].item(),
                    readout_toward_donor_layer26=-sign * s[26].item(),
                    edit_l2="+".join(f"{e.norm.item():.2f}" for e in ed)))
    base_again = run(model, ids, pos)["logits_last"][0, tid["Tokyo"]] - run(model, ids, pos)["logits_last"][0, tid["Berlin"]]
    assert abs(base_again.item() - next(r["margin"] for r in reversed(rows) if r["arm"] == "none")) < 1e-5  # hooks removed
    assert all(abs(r["margin"] - next(q["margin"] for q in rows if q["arm"] == "none" and q["prompt"] == r["prompt"])) < 1e-5
               for r in rows if r["beta"] == 0)
    with open(os.path.join(OUT, "s2_frozen_eval.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        if r["frozen_strength"] or r["arm"] == "edit":
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("role", "margin_toward_target", "frozen_strength")})

    # ---- (b) collateral effects on dir30's correct country-context prompts ----
    samples = [json.loads(l) for l in open(os.path.join(os.path.dirname(d30.OUT), "results", "samples.jsonl"))]
    ctx = []
    for smp in samples:
        if smp["role"] != "context_control" or not smp["correct"]:
            continue
        country = smp["subject"]
        dname = "to_Berlin" if country == "Japan" else "to_Tokyo"  # push the country token toward the OTHER country's capital
        enc = tok(smp["prompt"], return_offsets_mapping=True)
        start = smp["prompt"].index(country)
        pos = [j for j, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < start + len(country)][-1]
        ids = ids_of(tok, smp["prompt"])
        ans = tok(" " + (smp["answer"].lower() if smp["answer"] == "Yen" else smp["answer"])).input_ids[0]
        for arm, seed in arms:
            ed = [] if arm == "none" else edits_for(dname, pos, seed)
            r = run(model, ids, pos, edits=ed)
            lp = r["logits_last"][0].log_softmax(-1)
            ctx.append(dict(prompt=smp["prompt"], relation=smp["relation"], country=country, direction=dname, arm=arm,
                            country_token=tok.decode(ids[0, pos].item()), country_position=pos, country_is_first_token=pos == 0,
                            answer=tok.decode(ans),
                            p_answer=lp[ans].exp().item(), top1=tok.decode(lp.argmax().item()),
                            readout_layer26=probe(r["resid"][0])[26].item(), completion=gen(ids, ed),
                            edit_l2="+".join(f"{e.norm.item():.2f}" for e in ed)))
    with open(os.path.join(OUT, "s2_context.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(ctx[0]))
        wr.writeheader()
        wr.writerows(ctx)
    for r in ctx:
        if r["arm"] in ("none", "edit"):
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("relation", "country_position")})

    # ---- (c) the 41-point path under each condition ----
    prefix, bridge = TEMPLATES["original"]
    pos = 3
    idsJ, idsG = ids_of(tok, prefix.format("Japan") + bridge), ids_of(tok, prefix.format("Germany") + bridge)
    hA, hB = run(model, idsJ, pos)["resid"][0, 0], run(model, idsG, pos)["resid"][0, 0]
    ts = np.linspace(0, 1, N_T)
    H, _ = slerp_lerp_norm(hA, hB, ts)
    batch = idsJ.expand(N_T, -1)
    conds = [("baseline", None, None)] + [(d, d, None) for d in DIRECTIONS] + \
            [(f"{d}_control{s}", d, s) for d in DIRECTIONS for s in range(1, N_CTRL + 1)]
    store, summary, ref = {"ts": ts}, [], {}
    for cname, dname, seed in conds:
        ed = [] if dname is None else edits_for(dname, pos, seed)
        r = run(model, batch, pos, patch0=H, edits=ed)
        r_is = run(model, batch, pos, patch0=H, edits=ed, read_pos=pos + 1)["resid"]
        if cname == "baseline":  # fixed, unedited endpoint references
            ref = dict(c0=r["resid"][0], c1=r["resid"][-1], i0=r_is[0], i1=r_is[-1], p0=r["logits_last"][0].softmax(-1),
                       p1=r["logits_last"][-1].softmax(-1))
        d, step, da, db = path_metrics(r["resid"], ref["c0"], ref["c1"])
        d_is, step_is, da_is, db_is = path_metrics(r_is, ref["i0"], ref["i1"])
        p_last, p_pos = r["logits_last"].softmax(-1), r["logits_pos"].softmax(-1)
        margin = to_np(r["logits_last"][:, tid["Tokyo"]] - r["logits_last"][:, tid["Berlin"]])
        pb = to_np(probe(r["resid"])).T
        norms = np.stack([to_np(e.norm) for e in ed]) if ed else np.zeros((1, N_T))
        for k, v in dict(d=d, step=step, da=da, db=db, d_is=d_is, step_is=step_is, da_is=da_is, db_is=db_is, probe=pb,
                         margin=margin, p_tokyo=p_last[:, tid["Tokyo"]], p_berlin=p_last[:, tid["Berlin"]],
                         p_is=p_pos[:, tid["is"]], top1_last=p_last.argmax(-1), edit_l2=norms,
                         jsd_last_to_ref0=jsd_bits(p_last, ref["p0"][None]), jsd_last_to_ref1=jsd_bits(p_last, ref["p1"][None])).items():
            store[f"{cname}/{k}"] = to_np(v)
        d_np, step_np, dis_np, stepis_np = to_np(d), to_np(step), to_np(d_is), to_np(step_is)
        for L in range(N_LAYERS):
            summary.append(dict(
                condition=cname, layer=L, downstream_of_edit=L >= min(cfg["layers"]),
                answer_t_margin_zero=crossing(ts, margin, 0.0), margin_t0=float(margin[0]), margin_t1=float(margin[-1]),
                country_t_d50=crossing(ts, d_np[L], 0.5), country_t_peak_step=float(ts[step_np[L].argmax()] + 0.0125),
                country_peak_over_mean_step=float(step_np[L].max() / step_np[L].mean()),
                country_d_t0=float(d_np[L, 0]), country_d_t1=float(d_np[L, -1]),
                is_t_d50=crossing(ts, dis_np[L], 0.5), is_t_peak_step=float(ts[stepis_np[L].argmax()] + 0.0125),
                is_peak_over_mean_step=float(stepis_np[L].max() / max(stepis_np[L].mean(), 1e-12)),
                is_d_t0=float(dis_np[L, 0]), is_d_t1=float(dis_np[L, -1]),
                readout_t_zero=crossing(ts, pb[L], 0.0), readout_t0=float(pb[L, 0]), readout_t1=float(pb[L, -1])))
        print(cname, "answer boundary t =", round(summary[-1]["answer_t_margin_zero"], 3), "margin t0/t1 =",
              round(float(margin[0]), 2), round(float(margin[-1]), 2), "| top1:", " ".join(
                  tok.decode([i]).strip()[:3] for i in p_last.argmax(-1).tolist()[::2]))
    np.savez(os.path.join(OUT, "s3_path.npz"), **store)
    with open(os.path.join(OUT, "s3_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)


if __name__ == "__main__":
    main()
