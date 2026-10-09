"""S2b: rank features by gradient attribution on the TUNING pair and scan real native-model interventions.
For recipient r and donor d (the other country), with m = logit(Tokyo) - logit(Berlin) after the bridge and
g = dm/d(MLP output) at one layer and position in the recipient run:
    est_i = sign * (a_d,i - a_r,i) * (g . decoder_i),   sign = +1 toward Tokyo, -1 toward Berlin.
Top 1 / 5 / 10 features with est_i > 0 are patched toward the donor with beta in {0, 0.5, 1, 2}.
Sites: the bridge position (positive control, layers 0-27) and the country position (layers 0-26), then
two-adjacent-layer groups at the country position. Writes results/s2_ranking.csv and results/s2_scan.csv."""
import csv
import os

import torch

from common import CITY, COUNTRIES, DEV, N_LAYERS, OUT, SLICES, TEMPLATES, Edit, ids_of, run, setup

KS = [1, 5, 10]
BETAS = [0.0, 0.5, 1.0, 2.0]
DIRECTIONS = {"to_Berlin": ("Japan", "Germany", -1.0), "to_Tokyo": ("Germany", "Japan", 1.0)}  # recipient, donor, sign


def load_slices():
    return [{k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in torch.load(os.path.join(SLICES, f"layer_{L}.pt")).items()}
            for L in range(N_LAYERS)]


def mlp_out_grads(model, ids, tok_t, tok_b):
    """Gradient of logit(Tokyo) - logit(Berlin) at the last position w.r.t. every layer's MLP output. [L, T, d]"""
    outs, handles = {}, []
    for i, block in enumerate(model.model.layers):
        def fn(module, args, output, i=i):
            output.retain_grad()
            outs[i] = output
        handles.append(block.mlp.register_forward_hook(fn))
    with torch.enable_grad():
        emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
        logits = model(inputs_embeds=emb).logits[0, -1]
        (logits[tok_t] - logits[tok_b]).backward()
    for h in handles:
        h.remove()
    return torch.stack([outs[i].grad[0] for i in range(N_LAYERS)])


def make_edit(sl, template, site, pos, donor, feat_idx, beta, ctrl_seed=None):
    """feat_idx indexes into the layer slice. Donor activations are the stored native values of the donor run."""
    a_donor = sl["acts"][sl["keys"].index((template, donor, site))][feat_idx]
    ctrl = None
    if ctrl_seed is not None:
        g = torch.Generator().manual_seed(ctrl_seed)
        ctrl = sl["rand_dec"][torch.randperm(sl["rand_dec"].shape[0], generator=g)[:len(feat_idx)].to(DEV)]
    return Edit(sl["layer"], pos, sl["W_enc"][feat_idx], sl["b_enc"][feat_idx], sl["W_dec"][feat_idx], a_donor, beta, ctrl)


def readout(r, tid, tok):
    lp, lpp = r["logits_last"][0].log_softmax(-1), r["logits_pos"][0].log_softmax(-1)
    return dict(margin=(r["logits_last"][0, tid["Tokyo"]] - r["logits_last"][0, tid["Berlin"]]).item(),
                p_tokyo=lp[tid["Tokyo"]].exp().item(), p_berlin=lp[tid["Berlin"]].exp().item(),
                top1=tok.decode(lp.argmax().item()), p_is=lpp[tid["is"]].exp().item())


def main():
    tok, model = setup()
    for p in model.parameters():
        p.requires_grad_(False)
    tid = {w: tok(" " + w).input_ids[0] for w in ["Tokyo", "Berlin", "is"]}
    slices = load_slices()
    prefix, bridge = TEMPLATES["tune"]
    pos = tok(prefix.format("Japan"), return_tensors="pt").input_ids.shape[1] - 1
    site_pos = {"country": pos, "is": pos + 1}
    ids = {c: ids_of(tok, prefix.format(c) + bridge) for c in COUNTRIES}
    grads = {c: mlp_out_grads(model, ids[c], tid["Tokyo"], tid["Berlin"]) for c in COUNTRIES}
    base = {c: readout(run(model, ids[c], pos), tid, tok) for c in COUNTRIES}
    print("baseline", base)

    ranking, order = [], {}
    for dname, (rec, don, sign) in DIRECTIONS.items():
        for site, p in site_pos.items():
            for L in range(N_LAYERS):
                sl = slices[L]
                a_r = sl["acts"][sl["keys"].index(("tune", rec, site))]
                a_d = sl["acts"][sl["keys"].index(("tune", don, site))]
                est = sign * (a_d - a_r) * (sl["W_dec"] @ grads[rec][L, p])
                idx = [i for i in est.argsort(descending=True).tolist() if est[i] > 0]
                order[(dname, site, L)] = idx
                for rank, i in enumerate(idx[:10]):
                    ranking.append(dict(direction=dname, site=site, layer=L, rank=rank + 1, feature=sl["ids"][i].item(),
                                        a_recipient=a_r[i].item(), a_donor=a_d[i].item(), est_margin_shift=est[i].item()))
    with open(os.path.join(OUT, "s2_ranking.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ranking[0]))
        w.writeheader()
        w.writerows(ranking)

    rows = []

    def trial(dname, site, layers, k, beta):
        rec, don, sign = DIRECTIONS[dname]
        edits, k_used, est = [], [], 0.0
        for L in layers:
            idx = order[(dname, site, L)][:k]
            if idx:
                edits.append(make_edit(slices[L], "tune", site, site_pos[site], don, idx, beta))
            k_used.append(len(idx))
        out = readout(run(model, ids[rec], pos, edits=edits), tid, tok)
        rows.append(dict(direction=dname, site=site, layers="+".join(map(str, layers)), k=k, k_used="+".join(map(str, k_used)),
                         beta=beta, **out, base_margin=base[rec]["margin"],
                         signed_shift=sign * (out["margin"] - base[rec]["margin"]), reversed=sign * out["margin"] > 0,
                         top1_is_target=out["top1"].strip() == CITY[don],
                         edit_l2="+".join(f"{e.norm.item():.2f}" for e in edits)))

    for dname in DIRECTIONS:
        for site in ["is", "country"]:
            for L in range(N_LAYERS if site == "is" else N_LAYERS - 1):
                for k in KS:
                    for beta in BETAS:
                        trial(dname, site, [L], k, beta)
        for L in range(N_LAYERS - 2):  # two-adjacent-layer groups, country position, layers <= 26
            for k in KS:
                for beta in BETAS:
                    trial(dname, "country", [L, L + 1], k, beta)
    # beta = 0 must reproduce the baseline exactly
    assert all(abs(r["margin"] - r["base_margin"]) < 1e-4 for r in rows if r["beta"] == 0)
    with open(os.path.join(OUT, "s2_scan.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for dname in DIRECTIONS:
        for site in ["is", "country"]:
            print(f"\n== {dname} | site={site} | signed margin shift at beta=1 (k=1,5,10) and beta=2 (k=10); * = margin reversed")
            for r in rows:
                if r["direction"] == dname and r["site"] == site and r["k"] == 10 and r["beta"] == 1.0:
                    same = [q for q in rows if (q["direction"], q["site"], q["layers"]) == (dname, site, r["layers"])]
                    get = lambda k, b: next(q for q in same if q["k"] == k and q["beta"] == b)
                    cells = [get(1, 1.0), get(5, 1.0), get(10, 1.0), get(10, 2.0)]
                    print(f"  L{r['layers']:>5} k_used={r['k_used']:>5} " + " ".join(
                        f"{c['signed_shift']:7.2f}{'*' if c['reversed'] else ' '}" for c in cells)
                        + f"  top1(k10,b1)={r['top1']!r} p_is={r['p_is']:.2f} l2={r['edit_l2']}")


if __name__ == "__main__":
    main()
