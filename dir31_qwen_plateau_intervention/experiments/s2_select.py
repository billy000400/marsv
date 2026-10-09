"""S2c: apply the selection rule declared in JOURNAL.md to results/s2_scan.csv and freeze the edit.
Rule: country-position edits, layers <= 26; success = margin reversed in BOTH directions at the same layer(s), k and
beta; prefer beta <= 1 (natural donor replacement), then fewest features in total, then smaller beta, then the larger
worst-direction margin shift. Writes results/frozen_edit.json (sites, features, donor activations, strength)."""
import csv
import json
import os

import torch

from common import OUT, SLICES

DONOR = {"to_Berlin": "Germany", "to_Tokyo": "Japan"}


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, "s2_scan.csv"))) if r["site"] == "country" and float(r["beta"]) > 0]
    by = {}
    for r in rows:
        by.setdefault((r["layers"], int(r["k"]), float(r["beta"])), {})[r["direction"]] = r
    ok = []
    for (layers, k, beta), d in by.items():
        if len(d) == 2 and all(x["reversed"] == "True" for x in d.values()):
            n_feat = max(sum(map(int, x["k_used"].split("+"))) for x in d.values())
            worst = min(float(x["signed_shift"]) for x in d.values())
            ok.append((beta > 1.0, n_feat, beta, -worst, layers, k))
    print(f"{len(ok)} country-position settings reverse the margin in both directions")
    for o in sorted(ok)[:12]:
        print("  extrapolation=%s n_features=%d beta=%.1f worst_shift=%.2f layers=%s k=%d" % (o[0], o[1], o[2], -o[3], o[4], o[5]))
    if not ok:
        json.dump(dict(gate="failed", reason="no country-position edit reverses the margin in both directions"),
                  open(os.path.join(OUT, "frozen_edit.json"), "w"), indent=1)
        return
    extrap, n_feat, beta, neg_worst, layers, k = sorted(ok)[0]
    ranking = list(csv.DictReader(open(os.path.join(OUT, "s2_ranking.csv"))))
    cfg = dict(gate="passed", site="country", layers=[int(x) for x in layers.split("+")], k=k, beta=beta,
               extrapolation=extrap, worst_direction_margin_shift=-neg_worst, donor_template="tune", directions={})
    for dname, donor in DONOR.items():
        per_layer = {}
        for L in cfg["layers"]:
            sl = torch.load(os.path.join(SLICES, f"layer_{L}.pt"))
            feats = [int(r["feature"]) for r in ranking
                     if r["direction"] == dname and r["site"] == "country" and int(r["layer"]) == L and int(r["rank"]) <= k]
            idx = [sl["ids"].tolist().index(f) for f in feats]
            a_don = sl["acts"][sl["keys"].index(("tune", donor, "country"))][idx].tolist()
            a_rec = sl["acts"][sl["keys"].index(("tune", "Japan" if donor == "Germany" else "Germany", "country"))][idx].tolist()
            per_layer[str(L)] = dict(features=feats, slice_idx=idx, a_donor=a_don, a_recipient_tune=a_rec)
        cfg["directions"][dname] = dict(donor=donor, layers=per_layer)
    json.dump(cfg, open(os.path.join(OUT, "frozen_edit.json"), "w"), indent=1)
    print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
