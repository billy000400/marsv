"""S5+S6 — Run the Matthew-style assay on all frozen pairs and save tidy results + plots.

Primary: interpolate final-position resid_post after block 0 (11 blocks downstream), record
final-logit d(t) on a 101-step grid. Layerwise: same interpolation, record resid_post d(t) at
every later block. Depth comparison: move the interpolation block across {0,2,4,6,8,10},
recording final logits. All implementation checks from PLAN.md run first and abort on failure.
"""
import os, sys, json, hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig
from matthew_assay import run_pair, slerp_norm, rel_dist, transition_width, is_plateau, self_test
from cvd_style import CVD, REF_DIAG, REF_RULE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")

N_T = 101
INTERP_BLOCKS_CMP = [0, 2, 4, 6, 8, 10]
TOL_ENDPOINT = 1e-3   # max abs logit error allowed at t=0 / t=1 vs direct forwards
TOL_PREFIX = 1e-4     # max abs activation mismatch allowed at prefix positions


def load_model(device):
    ckpt = torch.load(os.path.join(RES, "checkpoints", "ckpt_final.pt"),
                      map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def seqs_for(pair, val):
    s, L = pair["val_start"], pair["prefix_len"]
    prefix = val[s:s + L]
    seq_A = np.concatenate([prefix, [pair["char_A_id"]]]).astype(np.int64)
    seq_B = np.concatenate([prefix, [pair["char_B_id"]]]).astype(np.int64)
    return seq_A, seq_B


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self_test()

    text = open("/tmp/tinyshakespeare.txt", "rb").read()
    meta = json.load(open(os.path.join(RES, "train_meta.json")))
    assert hashlib.sha256(text).hexdigest() == meta["corpus_sha256"]
    text = text.decode("utf-8")
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    val = data[int(0.9 * len(data)):]

    pairs_doc = json.load(open(os.path.join(RES, "prompt_pairs.json")))
    pairs = pairs_doc["pairs"]
    model = load_model(device)
    ts = np.linspace(0.0, 1.0, N_T)

    # ---- implementation checks on the first two pairs (PLAN 'Required implementation checks') --
    for pair in pairs[:2]:
        seq_A, seq_B = seqs_for(pair, val)
        r = run_pair(model, seq_A, seq_B, 0, ts, device)
        assert r["prefix_err"] < TOL_PREFIX, f"prefix mismatch {r['prefix_err']}"
        assert r["endpoint_err"]["t0_logit"] < TOL_ENDPOINT, r["endpoint_err"]
        assert r["endpoint_err"]["t1_logit"] < TOL_ENDPOINT, r["endpoint_err"]
        assert r["d0"] < 1e-4 and r["d1"] > 1 - 1e-4, (r["d0"], r["d1"])
        # batched-vs-single-example reference: run each t alone, compare d(t)
        d_ref = np.array([run_pair(model, seq_A, seq_B, 0, np.array([t, 1.0]), device)["d_logit"][0]
                          for t in ts[::20]])
        assert np.abs(d_ref - r["d_logit"][::20]).max() < 1e-5, "batched != single-example"
    print("implementation checks passed on pairs 0-1", flush=True)

    # ---- primary + layerwise runs (interp block 0) --------------------------------------------
    rows = []  # tidy: pair_id, interp_block, record_point, t, d
    summary = []
    curves_logit = {}
    curves_layer = {}
    for pair in pairs:
        seq_A, seq_B = seqs_for(pair, val)
        r = run_pair(model, seq_A, seq_B, 0, ts, device)
        assert r["prefix_err"] < TOL_PREFIX and r["endpoint_err"]["t0_logit"] < TOL_ENDPOINT \
            and r["endpoint_err"]["t1_logit"] < TOL_ENDPOINT, f"pair {pair['pair_id']} checks failed"
        pid = pair["pair_id"]
        curves_logit[pid] = r["d_logit"]
        curves_layer[pid] = r["d_layer"]
        for i, t in enumerate(ts):
            rows.append((pid, 0, "logits", t, float(r["d_logit"][i])))
            for l, dl in r["d_layer"].items():
                rows.append((pid, 0, f"resid_post_{l}", t, float(dl[i])))
        ok, w, t_lo, t_hi, max_dev = is_plateau(ts, r["d_logit"])
        summary.append({"pair_id": pid, "plateau": ok, "w_10_90": None if np.isnan(w) else round(float(w), 4),
                        "t_lo": None if np.isnan(t_lo) else round(float(t_lo), 4),
                        "t_hi": None if np.isnan(t_hi) else round(float(t_hi), 4),
                        "max_iso_dev": round(float(max_dev), 4),
                        "monotone": bool(max_dev <= 0.10),
                        "endpoint_logit_dist": pair["endpoint_logit_dist"]})
    n_plateau = sum(s["plateau"] for s in summary)
    n_nonmono = sum(not s["monotone"] for s in summary)
    ws = [s["w_10_90"] for s in summary if s["w_10_90"] is not None and s["monotone"]]
    print(f"primary: {n_plateau}/{len(pairs)} plateau, {n_nonmono} non-monotone, "
          f"median w={np.median(ws):.3f} range [{min(ws):.3f},{max(ws):.3f}]", flush=True)

    # ---- depth comparison (S6): vary interpolation block, record final logits -----------------
    depth_curves = {}  # (pid, block) -> d(t)
    for pair in pairs:
        seq_A, seq_B = seqs_for(pair, val)
        for blk in INTERP_BLOCKS_CMP:
            if blk == 0:
                depth_curves[(pair["pair_id"], 0)] = curves_logit[pair["pair_id"]]
                continue
            r = run_pair(model, seq_A, seq_B, blk, ts, device)
            depth_curves[(pair["pair_id"], blk)] = r["d_logit"]
            for i, t in enumerate(ts):
                rows.append((pair["pair_id"], blk, "logits", t, float(r["d_logit"][i])))
    depth_w = {blk: [] for blk in INTERP_BLOCKS_CMP}
    for (pid, blk), d in depth_curves.items():
        w, _, _, dev = transition_width(ts, d)
        if np.isfinite(w) and dev <= 0.10:
            depth_w[blk].append(w)

    # ---- save tidy results ---------------------------------------------------------------------
    with open(os.path.join(RES, "matthew_tidy.csv"), "w") as f:
        f.write("pair_id,interp_block,record_point,t,d\n")
        for row in rows:
            f.write("%d,%d,%s,%.4f,%.6f\n" % row)
    json.dump({"n_pairs": len(pairs), "n_t": N_T, "grid": "linspace(0,1,101)",
               "n_plateau": n_plateau, "n_nonmonotone": n_nonmono,
               "plateau_rule": "w_10_90<=0.25 & t_lo>=0.10 & t_hi<=0.90 & max_iso_dev<=0.10",
               "median_w_monotone": float(np.median(ws)),
               "w_range_monotone": [float(min(ws)), float(max(ws))],
               "depth_median_w": {str(b): float(np.median(v)) for b, v in depth_w.items() if v},
               "depth_n_monotone": {str(b): len(v) for b, v in depth_w.items()},
               "per_pair": summary}, open(os.path.join(RES, "matthew_summary.json"), "w"), indent=1)

    # ---- plots ----------------------------------------------------------------------------------
    # 1. individual final-logit curves, all pairs
    fig, axes = plt.subplots(5, 8, figsize=(20, 12), sharex=True, sharey=True)
    for ax, pair in zip(axes.flat, pairs):
        pid = pair["pair_id"]
        s = summary[pid]
        ax.plot(ts, ts, **REF_DIAG)
        ax.plot(ts, curves_logit[pid], color=CVD[0], lw=1.8)
        lab = f"#{pid} '{pair['char_A']}'->'{pair['char_B']}'".replace("\n", "\\n")
        ax.set_title(f"{lab}  w={s['w_10_90']}", fontsize=8)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("d(t)")
    fig.suptitle("Matthew relative distance d(t) in final-logit space — every frozen pair, interpolation after block 0 "
                 "(gray dashed = diagonal d=t non-plateau reference)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(PLOTS, "pair_curves_logits.png"), dpi=110)
    plt.close(fig)

    # 2. layerwise emergence for 4 representative fixed pairs (frozen choice: first 4 pair IDs)
    rep = [0, 1, 2, 3]
    fig, axes = plt.subplots(1, len(rep), figsize=(4.5 * len(rep), 4), sharey=True)
    cmap = plt.get_cmap("cividis")
    for ax, pid in zip(axes, rep):
        for l in sorted(curves_layer[pid]):
            ax.plot(ts, curves_layer[pid][l], color=cmap(l / 11), lw=1.2)
        ax.plot(ts, curves_logit[pid], color="k", lw=3, label="final logits (thick black)")
        ax.plot(ts, ts, **REF_DIAG)
        ax.set_title(f"pair #{pid}", fontsize=10)
        ax.set_xlabel("t"); ax.grid(alpha=0.2)
    axes[0].set_ylabel("d(t)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 11))
    fig.colorbar(sm, ax=axes, label="recording block (resid_post)", shrink=0.9)
    axes[0].legend(fontsize=8)
    fig.suptitle("Layerwise emergence: d(t) at successive downstream blocks (interpolation fixed after block 0)")
    fig.savefig(os.path.join(PLOTS, "layerwise_emergence.png"), dpi=110)
    plt.close(fig)

    # 3. interpolation-layer comparison: median + IQR of final-logit d(t) per interp block,
    #    plus per-block median transition width
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, blk in enumerate(INTERP_BLOCKS_CMP):
        D = np.stack([depth_curves[(p["pair_id"], blk)] for p in pairs])
        med = np.median(D, axis=0)
        ax[0].plot(ts, med, color=cmap(j / (len(INTERP_BLOCKS_CMP) - 1)), lw=1.8, label=f"block {blk}")
    ax[0].plot(ts, ts, label="d=t (dashed)", **REF_DIAG)
    ax[0].set_xlabel("t"); ax[0].set_ylabel("median d(t) over 40 pairs")
    ax[0].set_title("Final-logit d(t) vs interpolation block"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2)
    blks = [b for b in INTERP_BLOCKS_CMP if depth_w[b]]
    ax[1].errorbar([b for b in blks], [np.median(depth_w[b]) for b in blks],
                   yerr=[[np.median(depth_w[b]) - np.percentile(depth_w[b], 25) for b in blks],
                         [np.percentile(depth_w[b], 75) - np.median(depth_w[b]) for b in blks]],
                   fmt="o-", color=CVD[0], capsize=3, label="median (solid, circles)")
    ax[1].axhline(0.25, label="plateau bar w=0.25 (dotted)", **REF_RULE)
    ax[1].axhline(0.8, label="linear reference w=0.8 (dashed)", **REF_DIAG)
    ax[1].set_xlabel("interpolation block"); ax[1].set_ylabel("median w_10-90 (IQR bars)")
    ax[1].set_title("Transition width vs interpolation depth"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "interpolation_layer_comparison.png"), dpi=110)
    plt.close(fig)

    print("DONE. summary:", json.dumps({k: v for k, v in json.load(
        open(os.path.join(RES, "matthew_summary.json"))).items() if k != "per_pair"}), flush=True)


if __name__ == "__main__":
    main()
