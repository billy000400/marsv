"""S1 - build the two feature directions on the training split and check, on held-out text,
whether steering along them changes the next-character distribution in the expected way.

Writes results/s1.json and plots/fig1_steering.png.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_assay import (load_model, contexts, resid_and_probs, probs_from_resid, perturb,
                           char_mask, matched_pairs, feature_positions, BLOCK)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")
N_BUILD, N_TEST, N_RAND = 30, 30, 10
ALPHAS = np.linspace(-3.0, 3.0, 25)
SEED = 0
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

FEATS = [("speaker_label", "speaker_pos", "speaker_neg", "uppercase letter or ':'",
          lambda c: (c.isalpha() and c.isupper()) or c == ":"),
         ("word_continuation", "wordcont_pos", "wordcont_neg", "any letter",
          lambda c: c.isalpha())]


def build_direction(model, text, pairs, stoi, device):
    hp, _ = resid_and_probs(model, contexts(text, [p for p, _ in pairs], stoi), device)
    hn, _ = resid_and_probs(model, contexts(text, [n for _, n in pairs], stoi), device)
    d = (hp - hn).mean(0)
    return d / d.norm()


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    model, stoi, step = load_model(device)
    itos = {i: c for c, i in stoi.items()}
    tr_text, tr = feature_positions("train")
    ho_text, ho = feature_positions("heldout")

    out = {"ckpt_step": int(step), "block": BLOCK, "alphas": ALPHAS.tolist(),
           "n_build": N_BUILD, "n_test": N_TEST, "features": {}}
    dirs, curves, examples = {}, {}, {}

    g = torch.Generator().manual_seed(SEED)
    for name, pk, nk, label, pred in FEATS:
        pairs_tr = matched_pairs(tr_text, tr[pk], tr[nk], N_BUILD, rng)
        v = build_direction(model, tr_text, pairs_tr, stoi, device)
        dirs[name] = v
        pairs_ho = matched_pairs(ho_text, ho[pk], ho[nk], N_TEST, rng)
        ids = contexts(ho_text, [n for _, n in pairs_ho], stoi)   # steer the feature-off contexts
        h0, p0 = resid_and_probs(model, ids, device)
        m = torch.tensor(char_mask(stoi, pred), dtype=torch.float64)

        S = np.zeros((len(ALPHAS), len(ids)))
        for a, alpha in enumerate(ALPHAS):
            P = probs_from_resid(model, ids, perturb(h0, v, float(alpha)), device)
            S[a] = (P.double() @ m).numpy()
        R = np.zeros((N_RAND, len(ALPHAS)))
        for r in range(N_RAND):
            u = torch.randn(v.shape[0], generator=g)
            u = u / u.norm()
            for a, alpha in enumerate(ALPHAS):
                P = probs_from_resid(model, ids, perturb(h0, u, float(alpha)), device)
                R[r, a] = float((P.double() @ m).mean())

        # one example context's next-character distribution at three steering sizes
        ex_alphas = [-2.0, 0.0, 2.0]
        ex = {}
        for alpha in ex_alphas:
            P = probs_from_resid(model, ids[:1], perturb(h0[:1], v, alpha), device)
            ex[str(alpha)] = P[0].double().numpy()
        examples[name] = {"context": ho_text[pairs_ho[0][1] - 39:pairs_ho[0][1] + 1],
                          "dists": ex, "alphas": ex_alphas}
        curves[name] = {"S": S, "R": R, "clean": float((p0.double() @ m).mean()), "label": label}
        out["features"][name] = {
            "metric": f"probability mass on {label}",
            "clean_mean": float((p0.double() @ m).mean()),
            "steer_mean": S.mean(1).tolist(),
            "steer_per_context": S.tolist(),
            "random_mean": R.mean(0).tolist(),
            "build_pairs_final_chars": sorted({tr_text[p] for p, _ in pairs_tr}),
            "test_pairs_final_chars": sorted({ho_text[p] for p, _ in pairs_ho})}
        print(f"{name}: clean {out['features'][name]['clean_mean']:.3f} "
              f"alpha-3 {S.mean(1)[0]:.3f} alpha+3 {S.mean(1)[-1]:.3f} "
              f"random+3 {R.mean(0)[-1]:.3f}", flush=True)

    out["cosine_between_directions"] = float(torch.dot(dirs["speaker_label"], dirs["word_continuation"]))
    torch.save({k: v for k, v in dirs.items()}, os.path.join(RES, "directions.pt"))
    json.dump(out, open(os.path.join(RES, "s1.json"), "w"), indent=2)

    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.4))
    for j, (name, pk, nk, label, pred) in enumerate(FEATS):
        c = curves[name]
        a0 = ax[0, j]
        for i in range(c["S"].shape[1]):
            a0.plot(ALPHAS, c["S"][:, i], color=CVD[0], alpha=0.16, lw=0.8)
        a0.plot(ALPHAS, c["S"].mean(1), color=CVD[0], lw=2.4, ls="-", marker="o", ms=3,
                label="feature direction (mean)")
        a0.plot(ALPHAS, c["R"].mean(0), color=CVD[1], lw=2.0, ls="--", marker="s", ms=3,
                label="random directions (mean)")
        a0.axhline(c["clean"], color="0.35", ls=":", lw=1.2, label="unperturbed")
        a0.set_xlabel(r"steering size $\alpha$")
        a0.set_ylabel(f"P({label})")
        a0.set_title(name.replace("_", " "))
        a0.grid(alpha=0.3); a0.legend(fontsize=7)
        a1 = ax[1, j]
        e = examples[name]
        base = e["dists"]["0.0"]
        top = np.argsort(-base)[:10]
        order = sorted(top, key=lambda i: -base[i])
        w = 0.27
        for k, alpha in enumerate(e["alphas"]):
            d = e["dists"][str(alpha)]
            a1.bar(np.arange(len(order)) + (k - 1) * w, d[order], width=w,
                   color=CVD[k if k < 2 else 3], hatch=["//", "", ".."][k],
                   edgecolor="white", label=fr"$\alpha={alpha:+.0f}$")
        a1.set_xticks(np.arange(len(order)))
        a1.set_xticklabels([{" ": "sp", "\n": "\\n"}.get(itos[i], itos[i]) for i in order])
        a1.set_xlabel("next character (top 10 unperturbed)")
        a1.set_ylabel("probability")
        a1.set_title("example context: " + repr(e["context"][-24:]), fontsize=7)
        a1.grid(alpha=0.3, axis="y"); a1.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig1_steering.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/fig1_steering.png", flush=True)


if __name__ == "__main__":
    main()
