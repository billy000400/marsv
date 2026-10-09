"""S5: small suppression check at the country token of "The capital of <country> is".
Predeclared rule: candidates = features in the city set but not the country set that are active at the country
token (set-exclusive candidates); pick the single layer (0..26) with the highest total discovery score; use up to
five candidates there. The same rule applied to shared candidates is run as a labelled extra.
Writes results/suppression.csv, results/suppression_config.json and plots/suppression.png."""
import collections
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import DEV, OUT, PLOTS, encode, load_model, load_transcoder, setup_torch

ALPHAS = [0.0, 0.5, 1.0]
PAIRS = [("Japan", "Tokyo"), ("Germany", "Berlin")]
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]


def pick(city, country, status):
    """Layer and up to five features chosen from the frozen early-position table, never from intervention outcomes."""
    by_layer = collections.defaultdict(list)
    for e in csv.DictReader(open(os.path.join(OUT, "early_target_features.csv"))):
        if (e["country_position"] == country and e["feature_set"] == city and e["active_before_is"] == "True"
                and e["status"].startswith(status) and int(e["layer"]) <= 26):
            by_layer[int(e["layer"])].append((float(e["discovery_score"]), int(e["feature"])))
    if not by_layer:
        return None, []
    layer = max(sorted(by_layer), key=lambda L: sum(s for s, _ in by_layer[L]))
    return layer, [f for _, f in sorted(by_layer[layer], reverse=True)[:5]]


def country_pos(tok, text, country):
    enc = tok(text, return_offsets_mapping=True)
    start = text.index(country)
    return [j for j, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < start + len(country)][-1]


def main():
    setup_torch()
    tok, model = load_model()
    sets = json.load(open(os.path.join(OUT, "feature_sets.json")))["sets"]["top20"]
    samples = [json.loads(l) for l in open(os.path.join(OUT, "samples.jsonl"))]
    state = {}

    def hook(module, args, output):
        if state.get("alpha"):
            x = args[0][0, state["pos"]]
            a = torch.relu(x @ state["W_enc"].T + state["b_enc"])
            output = output.clone()
            output[0, state["pos"]] -= state["alpha"] * (a[:, None] * state["W_dec"]).sum(0)
            state["delta_norm"] = (state["alpha"] * (a[:, None] * state["W_dec"]).sum(0)).norm().item()
            return output

    out, cfg = [], []
    for country, city in PAIRS:
        for kind, status in [("set-exclusive (predeclared)", "unique"), ("shared (extra)", "shared")]:
            layer, feats = pick(city, country, status)
            if layer is None:
                cfg.append(dict(country=country, city=city, kind=kind, skipped="no early-active candidates"))
                continue
            W_enc, b_enc, W_dec, b_dec = load_transcoder(layer)
            handle = model.model.layers[layer].mlp.register_forward_hook(hook)
            prefix = f"The capital of {country}"
            pos = country_pos(tok, prefix, country)
            # natural activations at the country token, to choose the norm-matched control
            cache = {}
            h2 = model.model.layers[layer].mlp.register_forward_hook(lambda m, a, o: cache.update(x=a[0][0, pos]))
            model(tok(prefix, return_tensors="pt").input_ids.to(DEV))
            h2.remove()
            acts = encode(cache["x"], W_enc, b_enc)
            norms = acts * W_dec.norm(dim=-1)
            target_norm = (acts[feats, None] * W_dec[feats]).sum(0).norm().item()
            excluded = {f for g in (city, country) for L, f in sets[g] if L == layer}
            pool = [f for f in acts.nonzero().flatten().tolist() if f not in excluded]
            pool.sort(key=lambda f: abs(norms[f].item() - target_norm / len(feats)))
            control = pool[:len(feats)]
            control_norm = (acts[control, None] * W_dec[control]).sum(0).norm().item()
            cfg.append(dict(country=country, city=city, kind=kind, layer=layer, features=feats, edited_position=pos,
                            natural_activations=acts[feats].tolist(), perturbation_l2_at_alpha1=target_norm,
                            control_features=control, control_natural_activations=acts[control].tolist(),
                            control_perturbation_l2_at_alpha1=control_norm,
                            control_pool_size=len(pool), mlp_output_l2=None))

            prompts = [dict(role="capital", prompt=prefix, bridge=" is", answer=city)]
            if kind.startswith("set-exclusive"):
                prompts += [dict(role="context_control", prompt=r["prompt"], bridge="", answer=r["answer"])
                            for r in samples if r["group"] == country + "-context" and r["correct"]]
            for arm, fs in [("candidate", feats), ("norm-matched control", control)]:
                state.update(W_enc=W_enc[fs], b_enc=b_enc[fs], W_dec=W_dec[fs])
                for p in prompts:
                    state["pos"] = country_pos(tok, p["prompt"], country)
                    ids_prefix = tok(p["prompt"], return_tensors="pt").input_ids.to(DEV)
                    ids = tok(p["prompt"] + p["bridge"], return_tensors="pt").input_ids.to(DEV)
                    ans = tok(" " + p["answer"].lower() if p["answer"] == "Yen" else " " + p["answer"]).input_ids
                    for alpha in ALPHAS + [0.0]:  # trailing 0 re-checks that the original output is restored
                        state["alpha"] = alpha
                        state["delta_norm"] = 0.0
                        lp = model(ids).logits[0, -1].float().log_softmax(-1)
                        gen = model.generate(ids, do_sample=False, max_new_tokens=8, temperature=None, top_p=None,
                                             top_k=None, pad_token_id=tok.eos_token_id, use_cache=False)
                        row = dict(country=country, city=city, kind=kind, layer=layer, arm=arm, role=p["role"],
                                   prompt=p["prompt"] + p["bridge"], edited_position=state["pos"], alpha=alpha,
                                   answer=p["answer"], answer_first_token=tok.decode(ans[:1]),
                                   answer_prob=lp[ans[0]].exp().item(), top1=tok.decode(lp.argmax().item()),
                                   continuation=tok.decode(gen[0, ids.shape[1]:]),
                                   perturbation_l2=state["delta_norm"], bridge_prob=None)
                        if p["bridge"]:
                            lpp = model(ids_prefix).logits[0, -1].float().log_softmax(-1)
                            row["bridge_prob"] = lpp[ids[0, -1]].exp().item()
                        out.append(row)
            state["alpha"] = 0.0
            handle.remove()
            del W_enc, b_enc, W_dec, b_dec
            torch.cuda.empty_cache()

    # alpha = 0 must reproduce the baseline, before and after the edits
    for key in {(r["kind"], r["country"], r["arm"], r["prompt"]) for r in out}:
        z = [r["answer_prob"] for r in out if (r["kind"], r["country"], r["arm"], r["prompt"]) == key and r["alpha"] == 0]
        assert len(z) == 2 and abs(z[0] - z[1]) < 1e-6, key
    out = [r for i, r in enumerate(out) if not (r["alpha"] == 0 and i > 0 and out[i - 1]["alpha"] == 1.0)]
    with open(os.path.join(OUT, "suppression.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    json.dump(dict(alphas=ALPHAS, rule=__doc__, runs=cfg), open(os.path.join(OUT, "suppression_config.json"), "w"), indent=1)
    for c in cfg:
        print(c)
    for r in out:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("city", "answer_first_token")})

    plt.rcParams.update({"font.size": 10})
    runs = [c for c in cfg if "layer" in c]
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.6), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], runs):
        for i, arm in enumerate(["candidate", "norm-matched control"]):
            rr = [r for r in out if r["kind"] == c["kind"] and r["country"] == c["country"] and r["arm"] == arm and r["role"] == "capital"]
            ax.plot([r["alpha"] for r in rr], [r["answer_prob"] for r in rr], color=CVD[i], linestyle=["-", "--"][i],
                    marker=["o", "s"][i], label=arm)
        ax.set_title(f"{c['city']}: {c['kind']}\nlayer {c['layer']}, {len(c['features'])} feature(s)", fontsize=9)
        ax.set_xlabel("suppression strength alpha")
        ax.set_xticks(ALPHAS)
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel('P(city | "The capital of <country> is")')
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "suppression.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
