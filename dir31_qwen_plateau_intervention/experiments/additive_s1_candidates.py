"""S4.1: candidates and reference amplitudes. No intervention is run here.
Candidates = dir30's Berlin features outside its Germany set (top50), layers 1-26.
Reference amplitude = median positive activation at the last prompt token (dir30's extraction position) of dir30's
7 Berlin-answer discovery prompts, from the native MLP input of that layer. Nothing is read at a country token of a
capital prompt except the tuning prompt's own Japan token (the edit site), for the native MLP-output norm and the
candidates' native activation there (diagnostics, not amplitudes).
Writes results/additive/candidate_counts.json, candidates.csv, candidates.npz, discovery_prompts.csv, site_norms.csv."""
import collections
import csv
import json
import os

import numpy as np
import torch

import common as C
from additive_common import D30_OUT, GERMANY_CUTOFF, LAYERS, OUT, write_csv


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = json.load(open(os.path.join(D30_OUT, "feature_sets.json")))["sets"]
    berlin = {k: {tuple(x) for x in v["Berlin"]} for k, v in fs.items()}
    germany = {k: {tuple(x) for x in v["Germany"]} for k, v in fs.items()}
    assert berlin["top10"] == berlin["top20"] == berlin["top50"]
    in_range = lambda s: sorted(x for x in s if x[0] in LAYERS)
    cands = in_range(berlin["top50"] - germany[GERMANY_CUTOFF])
    counts = dict(
        berlin_set_size=len(berlin["top50"]), germany_set_size={k: len(v) for k, v in germany.items()},
        germany_cutoff_used=GERMANY_CUTOFF, layers=[LAYERS[0], LAYERS[-1]],
        candidates={k: len(in_range(berlin["top50"] - g)) for k, g in germany.items()},
        berlin_in_range_also_in_germany=[list(x) for x in in_range(berlin["top50"] & germany[GERMANY_CUTOFF])],
        berlin_only_outside_layer_range=[list(x) for x in sorted(berlin["top50"] - germany[GERMANY_CUTOFF]) if x[0] not in LAYERS],
        admitted_only_under_top10=[list(x) for x in in_range(germany["top50"] - germany["top10"]) if x in berlin["top50"]],
        candidates_per_layer={L: sum(x[0] == L for x in cands) for L in LAYERS})
    json.dump(counts, open(os.path.join(OUT, "candidate_counts.json"), "w"), indent=1)
    print(json.dumps(counts))

    rows30 = [json.loads(l) for l in open(os.path.join(D30_OUT, "samples.jsonl"))]
    disc = [i for i, r in enumerate(rows30) if r["group"] == "Berlin" and r["role"] == "discovery" and r["correct"]]
    facts = sorted({rows30[i]["fact_id"] for i in disc})
    assert len(disc) == 7 and len(facts) == 2, (len(disc), facts)
    rank30 = {(int(r["layer"]), int(r["feature"])): r for r in csv.DictReader(open(os.path.join(D30_OUT, "feature_rankings.csv")))
              if r["group"] == "Berlin"}

    tok, model = C.setup()
    C.d30.SCRATCH = "/tmp/hf_dir31_additive"
    tap = C.d30.MLPTap(model)
    X, drows = [], []
    for i in disc:  # native MLP input at the last prompt token, every layer
        ids = C.ids_of(tok, rows30[i]["prompt"])
        assert ids.shape[1] - 1 == rows30[i]["last_token_index"]
        model(ids)
        X.append(torch.stack([tap.x[L][0, -1] for L in range(C.N_LAYERS)]))
        drows.append(dict(dir30_row=i, fact_id=rows30[i]["fact_id"], relation=rows30[i]["relation"], prompt=rows30[i]["prompt"],
                          extraction_token_index=ids.shape[1] - 1, extraction_token=tok.decode(ids[0, -1].item()),
                          dir30_completion=rows30[i]["completion"]))
    X = torch.stack(X)  # [7, L, d]
    write_csv("discovery_prompts.csv", drows)
    prefix, bridge = C.TEMPLATES["tune"]
    ids = C.ids_of(tok, prefix.format("Japan") + bridge)
    pos = ids.shape[1] - 2
    assert tok.decode(ids[0, pos].item()) == " Japan"
    model(ids)
    x_site = torch.stack([tap.x[L][0, pos] for L in range(C.N_LAYERS)])
    site = [dict(layer=L, template="tune", prompt=prefix.format("Japan") + bridge, country_position=pos,
                 native_mlp_out_l2=tap.y[L][0, pos].norm().item()) for L in range(C.N_LAYERS)]
    write_csv("site_norms.csv", site)
    tap.remove()
    del model
    torch.cuda.empty_cache()

    fact_of = np.array([facts.index(rows30[i]["fact_id"]) for i in disc])
    out, dec = [], []
    for L in sorted({x[0] for x in cands}):
        feats = [f for (l, f) in cands if l == L]
        W_enc, b_enc, W_dec, b_dec = C.d30.load_transcoder(L)
        a = C.d30.encode(X[:, L], W_enc, b_enc)[:, feats].cpu().numpy()  # [7, k]
        a_site = C.d30.encode(x_site[L], W_enc, b_enc)[feats].cpu().numpy()
        z = np.load(os.path.join(D30_OUT, "acts", f"layer_{L}.npz"))  # dir30's stored activations, for the cross-check
        stored = {(int(s), int(f)): float(v) for s, f, v in zip(z["sample_index"], z["feature"], z["value"])}
        for j, f in enumerate(feats):
            on = a[:, j] > 0
            diff = max(abs(a[i, j] - stored.get((disc[i], f), 0.0)) for i in range(len(disc)))
            r = rank30[(L, f)]
            out.append(dict(
                layer=L, feature=f, amplitude=float(np.median(a[on, j])), n_positive_prompts=int(on.sum()), n_prompts=len(disc),
                n_positive_facts=len(set(fact_of[on])), n_facts=len(facts), positive_min=float(a[on, j].min()),
                positive_max=float(a[on, j].max()), extraction_position="last prompt token",
                positive_prompt_tokens=";".join(f"row{disc[i]}:{drows[i]['extraction_token_index']}:{drows[i]['extraction_token'].strip()}"
                                                for i in np.where(on)[0]),
                max_abs_diff_vs_dir30_acts=float(diff), decoder_l2=W_dec[f].norm().item(),
                native_mlp_out_l2_tune_site=site[L]["native_mlp_out_l2"], native_act_at_japan_token_tune=float(a_site[j]),
                dir30_score=float(r["score"]), dir30_rank_in_layer=int(r["rank_in_layer"]), dir30_rate_Berlin=float(r["rate_Berlin"]),
                dir30_rate_Germany=float(r["rate_Germany"]), dir30_rate_Japan=float(r["rate_Japan"]),
                dir30_rate_Tokyo=float(r["rate_Tokyo"]), dir30_background_rate=float(r["background_rate"]),
                in_germany_top10=(L, f) in germany["top10"], in_germany_top20=(L, f) in germany["top20"]))
        dec.append(W_dec[feats].cpu().numpy())
        print(L, [(o["feature"], round(o["amplitude"], 3), f"{o['n_positive_prompts']}/7") for o in out[-len(feats):]], flush=True)
        del W_enc, b_enc, W_dec, b_dec
        torch.cuda.empty_cache()
    assert max(o["max_abs_diff_vs_dir30_acts"] for o in out) < 1e-3, "re-encoded activations differ from dir30's"
    assert all(o["n_positive_facts"] == 2 for o in out)
    write_csv("candidates.csv", out)
    np.savez(os.path.join(OUT, "candidates.npz"), layer=np.array([o["layer"] for o in out]), feature=np.array([o["feature"] for o in out]),
             amplitude=np.array([o["amplitude"] for o in out], np.float32), W_dec=np.concatenate(dec).astype(np.float32))
    print("candidates", len(out), "per layer", dict(collections.Counter(o["layer"] for o in out)),
          "| max |diff| vs dir30 acts", max(o["max_abs_diff_vs_dir30_acts"] for o in out))


if __name__ == "__main__":
    main()
