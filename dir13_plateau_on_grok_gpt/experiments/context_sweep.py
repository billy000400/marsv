"""Context control for the comma sweep: does the plateau result depend on the one shared context?

Every plateau number so far comes from the single context "The house was ". This script repeats
the operator-requested comma-to-every-other-character sweep in 8 further contexts drawn from
held-out validation text, chosen to span the model's own probability for a comma in that slot
(from "a comma is essentially impossible here" to "a comma is the likely continuation"). That
second axis also tests the standing caveat that the comma endpoint is an implausible input.

Everything else is identical to comma_sweep.py: endpoint A = context + ',', endpoint B = context
+ one of the other 64 characters, 50 evenly spaced interpolation values, `slerp_rescale`, patch of
the final position of the block-0 residual stream, d(t) in final-logit space, width w_10->90.

Writes results/context_sweep_summary.json and results/context_sweep_raw.npz.
"""
import os, sys, json, hashlib
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig
from matthew_assay import run_pair, transition_width, is_plateau, self_test

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CKPT = os.path.join(RES, "checkpoints_grok_char", "ckpt_030000.pt")
REF_CONTEXT = "The house was "     # the context every previous plateau number used
CTX_LEN = 64                       # characters of held-out text per new context
N_CANDIDATES = 256
N_CONTEXTS = 8                     # new contexts, spanning p(comma) from lowest to highest
N_T = 50
SEED = 20260725


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    self_test()

    text = open("/tmp/tinyshakespeare.txt", "rb").read()
    # train_meta_grok_char.json is truncated (a known post-run save crash); the pilot metadata
    # records the same corpus SHA and is intact.
    meta = json.load(open(os.path.join(RES, "train_meta.json")))
    assert hashlib.sha256(text).hexdigest() == meta["corpus_sha256"], "corpus mismatch"
    text = text.decode("utf-8")
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    val = text[int(0.9 * len(text)):]

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    targets = [c for c in sorted(stoi, key=lambda c: stoi[c]) if c != ","]
    ts = np.linspace(0.0, 1.0, N_T)

    # ---- candidate contexts from held-out text, ranked by the model's p(comma) in that slot ----
    rng = np.random.default_rng(SEED)
    starts = rng.choice(len(val) - CTX_LEN - 1, size=N_CANDIDATES, replace=False)
    cands = sorted({val[s:s + CTX_LEN] for s in starts})
    with torch.no_grad():
        p_comma = []
        for ctx in cands:
            x = torch.tensor([[stoi[c] for c in ctx]], device=device)
            p = torch.softmax(model(x)[0][0, -1], dim=-1)
            p_comma.append(float(p[stoi[","]]))
    order = np.argsort(p_comma)
    picks = np.linspace(0, len(cands) - 1, N_CONTEXTS).round().astype(int)   # spread over the range
    contexts = [(REF_CONTEXT, None)] + [(cands[order[i]], float(p_comma[order[i]])) for i in picks]

    raw, per_context = {"ts": ts}, []
    for ci, (ctx, _) in enumerate(contexts):
        seq_A = np.array([stoi[c] for c in ctx + ","], dtype=np.int64)
        with torch.no_grad():
            x = torch.tensor([[stoi[c] for c in ctx]], device=device)
            p_next = torch.softmax(model(x)[0][0, -1], dim=-1).cpu().numpy()
        ws, ps, plateau, checks = [], [], 0, []
        for c in targets:
            seq_B = np.array([stoi[x] for x in ctx + c], dtype=np.int64)
            r = run_pair(model, seq_A, seq_B, 0, ts, device)
            ok, w, t_lo, t_hi, dev = is_plateau(ts, r["d_logit"])
            ws.append(float(w)); ps.append(float(p_next[stoi[c]])); plateau += bool(ok)
            checks.append((r["prefix_err"], r["endpoint_err"]["t0_logit"], r["d0"], r["d1"]))
            raw[f"c{ci}|{stoi[c]}"] = r["d_logit"]
        ws, ps = np.array(ws), np.array(ps)
        rho = spearmanr(ws, ps)
        chk = np.array(checks)
        per_context.append({
            "context": ctx, "is_reference": ci == 0,
            "p_comma": float(p_next[stoi[","]]),
            "median_w": round(float(np.median(ws)), 4),
            "q25_w": round(float(np.percentile(ws, 25)), 4),
            "q75_w": round(float(np.percentile(ws, 75)), 4),
            "min_w": round(float(ws.min()), 4), "max_w": round(float(ws.max()), 4),
            "n_strict_plateau": int(plateau), "n_near_linear": int((ws >= 0.70).sum()),
            "n_le_035": int((ws <= 0.35).sum()), "n_pairs": len(ws),
            "spearman_rho_w_vs_pnext": round(float(rho.statistic), 4),
            "spearman_p": float(rho.pvalue),
            "max_prefix_err": float(chk[:, 0].max()), "max_endpoint_err": float(chk[:, 1].max()),
            "max_d0": float(chk[:, 2].max()), "min_d1": float(chk[:, 3].min()),
            "widths": [round(float(v), 4) for v in ws],
            "p_next": [float(v) for v in ps],
        })
        print(f"ctx {ci}: p(comma)={per_context[-1]['p_comma']:.2e} median w={np.median(ws):.3f} "
              f"strict {plateau}/{len(ws)} near-linear {(ws >= 0.70).sum()} rho={rho.statistic:+.2f}",
              flush=True)

    # pooled across every context
    all_w = np.concatenate([np.array(d["widths"]) for d in per_context])
    all_p = np.concatenate([np.array(d["p_next"]) for d in per_context])
    pooled = spearmanr(all_w, all_p)
    med = np.array([d["median_w"] for d in per_context])
    pc = np.array([d["p_comma"] for d in per_context])
    ctx_rho = spearmanr(med, pc)
    summary = {
        "n_contexts": len(per_context), "n_pairs_per_context": len(targets), "n_t": N_T,
        "ctx_len": CTX_LEN, "seed": SEED, "step": 30000, "interp_block": 0,
        "pooled_median_w": round(float(np.median(all_w)), 4),
        "pooled_n_pairs": int(all_w.size),
        "pooled_n_strict": int(sum(d["n_strict_plateau"] for d in per_context)),
        "pooled_n_near_linear": int((all_w >= 0.70).sum()),
        "pooled_rho_w_vs_pnext": round(float(pooled.statistic), 4),
        "pooled_rho_p": float(pooled.pvalue),
        "rho_medianw_vs_pcomma": round(float(ctx_rho.statistic), 4),
        "rho_medianw_vs_pcomma_p": float(ctx_rho.pvalue),
        "per_context": per_context,
    }
    print(f"POOLED: {all_w.size} pairs, median w={summary['pooled_median_w']}, "
          f"strict {summary['pooled_n_strict']}, near-linear {summary['pooled_n_near_linear']}, "
          f"rho={summary['pooled_rho_w_vs_pnext']}; median-w vs p(comma) rho="
          f"{summary['rho_medianw_vs_pcomma']} (p={summary['rho_medianw_vs_pcomma_p']:.3f})", flush=True)
    np.savez_compressed(os.path.join(RES, "context_sweep_raw.npz"), **raw)
    json.dump(summary, open(os.path.join(RES, "context_sweep_summary.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
