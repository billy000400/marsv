"""Direction 8 — Stage C: improved synthetic latent constructions (H3).

Stage B established: naive/sparse SAE compositions plateau BELOW a random-displacement
reference at matched distance; recon sits ~ON it. H3 (decision table) asks whether
*higher-order* code structure -- co-occurrence-aware supports or encode-decode
cycle-consistent codes -- recovers plateau ABOVE the random-displacement reference at
matched distance. If yes: higher-order latent structure is a real ingredient of plateau
validity. If they stay ON/BELOW the curve like recon/naive: the project null is complete
(plateau = closeness-to-real + local robustness, not recoverable by better codes).

New conditions vs Stage B (same iso_displace reference, same held-out tau, same dirs):
  * cooc         -- co-occurrence support: take a REAL example j's active feature SET
                    (support), assign coefficients from the empirical active-coefficient
                    marginal (exactly as `naive`). Isolates *support co-occurrence* as the
                    only change over naive (which draws indices independently by frequency).
  * cooc_full    -- support AND coefficients from real example j (real code of j, decoded),
                    paired to source i for distance. Maximally-realistic latent structure
                    (a genuine on-manifold point) at a large distance from source i. Reuses
                    the `recon` plateau curves (plateau is intrinsic to the activation), so
                    no extra forward passes.
  * cycle_consistent -- iterate z <- encode(decode(z)) to an encode-decode fixed point
                    starting from the naive code, then decode. Tests whether self-consistency
                    under the SAE map (||encode(decode(z))-z|| small) recovers plateau.

Reuses the Stage B in-context forward-hook plateau method verbatim.
"""
import os, json, time, csv, argparse
os.environ.setdefault("HF_HOME", "/mars-vol/.cache/huggingface")
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

torch.set_num_threads(2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.225)

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); PLOTS = os.path.join(ROOT, "plots")
os.makedirs(RES, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
TXT = os.path.join(ROOT, "..", "dir3_manifold", "data", "fineweb_texts.json")

SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"; SAE_PRE = "blocks.6.hook_resid_pre"
HOOK_BLOCK = 5
SEQ = 64; BS = 32
ISO_DELTAS = [15.0, 30.0, 60.0, 120.0]


def load_sae():
    cfgp = hf_hub_download(SAE_REPO, f"{SAE_PRE}/cfg.json")
    wp = hf_hub_download(SAE_REPO, f"{SAE_PRE}/sae_weights.safetensors")
    cfg = json.load(open(cfgp)); w = load_file(wp)
    sae = {k: w[k].float().to(DEVICE) for k in ["W_enc", "W_dec", "b_enc", "b_dec"]}
    sae["cfg"] = cfg
    return sae


def sae_encode(x, sae, sub_bdec):
    pre = (x - sae["b_dec"]) if sub_bdec else x
    return torch.relu(pre @ sae["W_enc"] + sae["b_enc"])


def sae_decode(z, sae):
    return z @ sae["W_dec"] + sae["b_dec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    N = 200 if args.full else 24
    N_DIRS = 6 if args.full else 2
    R_GRID = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
    rng = np.random.default_rng(0)
    t0 = time.time()

    sae = load_sae()
    m = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    texts = [t for t in json.load(open(TXT)) if len(t.strip()) > 200][: N + 50]

    enc = tok(texts, return_tensors="pt", truncation=True, max_length=SEQ, padding="max_length")
    ids = enc["input_ids"]; mask = enc["attention_mask"]
    last = mask.sum(1) - 1
    keep = (mask.sum(1) >= 16)
    ids, mask, last = ids[keep][:N], mask[keep][:N], last[keep][:N]
    ids, mask, last = ids.to(DEVICE), mask.to(DEVICE), last.to(DEVICE)
    N = ids.shape[0]
    block = m.transformer.h[HOOK_BLOCK]

    def hooked_last_logits(rep, ids_b, mask_b, last_b):
        h = None
        if rep is not None:
            def hook(mod, inp, out):
                hh = (out[0] if isinstance(out, tuple) else out).clone()
                hh[torch.arange(hh.shape[0], device=hh.device), last_b] = rep.to(hh.dtype)
                return (hh,) + tuple(out[1:]) if isinstance(out, tuple) else hh
            h = block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                logits = m(input_ids=ids_b, attention_mask=mask_b).logits
        finally:
            if h is not None:
                h.remove()
        return logits[torch.arange(logits.shape[0], device=logits.device), last_b]

    # ---- capture real resid_pre@6 last-token activations ----
    cap = {}
    def cap_hook(mod, inp, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out)
    real = np.empty((N, 768), np.float32)
    for i in range(0, N, BS):
        sl = slice(i, i + BS)
        hh = block.register_forward_hook(cap_hook)
        with torch.no_grad():
            m(input_ids=ids[sl], attention_mask=mask[sl])
        hh.remove()
        h = cap["h"]; lb = last[sl]
        real[sl] = h[torch.arange(h.shape[0], device=h.device), lb].float().cpu().numpy()
    print(f"[{time.time()-t0:.0f}s] captured {N} real acts", flush=True)

    Xr = torch.from_numpy(real).to(DEVICE)
    SUB = True  # established in Stage A (recon err 27.7 vs 380.8)

    # ---- real codes & marginals ----
    z_real = sae_encode(Xr, sae, SUB)
    recon = sae_decode(z_real, sae)
    z_np = z_real.cpu().numpy()
    l0_real = (z_np > 0).sum(1)
    active_coefs = z_np[z_np > 0]
    feat_freq = (z_np > 0).mean(0); active_feats = np.where(feat_freq > 0)[0]
    p_feat = feat_freq[active_feats] / feat_freq[active_feats].sum()
    d_sae = sae["W_dec"].shape[0]

    # naive: independent composition (Stage A/B) -- reference synthetic
    z_naive = np.zeros((N, d_sae), np.float32)
    for i in range(N):
        k = int(max(1, l0_real[rng.integers(N)]))
        idx = rng.choice(active_feats, size=min(k, len(active_feats)), replace=False, p=p_feat)
        z_naive[i, idx] = rng.choice(active_coefs, size=len(idx))
    naive = sae_decode(torch.from_numpy(z_naive).to(DEVICE), sae).cpu().numpy().astype(np.float32)
    l0_naive = (z_naive > 0).sum(1)

    # cooc: real example j's SUPPORT + empirical-marginal coefs (isolates co-occurrence)
    z_cooc = np.zeros((N, d_sae), np.float32)
    for i in range(N):
        j = int(rng.integers(N))
        supp = np.where(z_np[j] > 0)[0]
        if len(supp) == 0:
            supp = rng.choice(active_feats, size=1)
        z_cooc[i, supp] = rng.choice(active_coefs, size=len(supp))
    cooc = sae_decode(torch.from_numpy(z_cooc).to(DEVICE), sae).cpu().numpy().astype(np.float32)
    l0_cooc = (z_cooc > 0).sum(1)

    # cycle_consistent: FILTER naive candidates to encode-decode self-consistency
    # (PLAN def: keep candidates with ||encode(decode(z))-z||/||z|| below a validation quantile).
    # Threshold = 75th percentile of REAL-code cycle error (as self-consistent as 75% of real codes).
    def cyc_relerr(z_t):
        x = sae_decode(z_t, sae); z2 = sae_encode(x, sae, SUB)
        return ((z2 - z_t).norm(dim=1) / z_t.norm(dim=1).clamp_min(1e-8)).cpu().numpy()
    real_cyc = cyc_relerr(z_real)
    cyc_thresh = float(np.percentile(real_cyc, 75))
    CHUNK = 4000; CAP = 160000
    sel_codes = []; n_gen = 0; n_pass = 0
    while len(sel_codes) < N and n_gen < CAP:
        zp = np.zeros((CHUNK, d_sae), np.float32)
        for i in range(CHUNK):
            k = int(max(1, l0_real[rng.integers(N)]))
            idx = rng.choice(active_feats, size=min(k, len(active_feats)), replace=False, p=p_feat)
            zp[i, idx] = rng.choice(active_coefs, size=len(idx))
        pc = cyc_relerr(torch.from_numpy(zp).to(DEVICE))
        n_gen += CHUNK; n_pass += int((pc <= cyc_thresh).sum())
        for i in np.where(pc <= cyc_thresh)[0]:
            sel_codes.append(zp[i])
            if len(sel_codes) >= N:
                break
    cyc_pass_rate = n_pass / max(n_gen, 1)
    z_cyc = np.stack(sel_codes[:N]).astype(np.float32)
    if z_cyc.shape[0] < N:  # rare: pad by tiling so the sweep sees N rows (note in summary)
        reps = int(np.ceil(N / z_cyc.shape[0]))
        z_cyc = np.tile(z_cyc, (reps, 1))[:N]
    cycle = sae_decode(torch.from_numpy(z_cyc).to(DEVICE), sae).cpu().numpy().astype(np.float32)
    l0_cyc = (z_cyc > 0).sum(1)
    cyc_sel_relerr = cyc_relerr(torch.from_numpy(z_cyc).to(DEVICE))
    print(f"[{time.time()-t0:.0f}s] cycle filter: thresh={cyc_thresh:.3f} pass_rate={cyc_pass_rate*100:.2f}% "
          f"generated={n_gen} selected={len(sel_codes[:N])} sel_relerr_med={np.median(cyc_sel_relerr):.3f}", flush=True)

    # cooc_full: real code of example j decoded == recon[j]; plateau intrinsic -> reuse recon curves.
    perm = rng.permutation(N)
    while np.any(perm == np.arange(N)):  # derangement so j != i
        perm = rng.permutation(N)

    conditions = {"real": real,
                  "recon": recon.cpu().numpy().astype(np.float32),
                  "naive": naive, "cooc": cooc, "cycle_consistent": cycle}
    l0_by = {"real": l0_real, "recon": l0_real, "naive": l0_naive,
             "cooc": l0_cooc, "cycle_consistent": l0_cyc}

    # iso_displace reference family: x_real + delta * d_unit (distance == delta exactly)
    iso = {}
    for delta in ISO_DELTAS:
        d = rng.standard_normal((N, 768)).astype(np.float32)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        iso[f"iso{int(delta)}"] = (real + delta * d).astype(np.float32)
    conditions.update(iso)
    for k in iso:
        l0_by[k] = np.zeros(N)

    # perturbation directions (shared across conditions for pairing)
    dirs = rng.standard_normal((N_DIRS, N, 768)).astype(np.float32)
    dirs = dirs / np.linalg.norm(dirs, axis=2, keepdims=True)

    def plateau_kl_curves(X):
        Xt = torch.from_numpy(X).to(DEVICE).float()
        xn = Xt.norm(dim=1, keepdim=True)
        out = np.zeros((N, N_DIRS, len(R_GRID)), np.float32)
        base_lp = torch.empty(N, 50257, device=DEVICE)
        for i in range(0, N, BS):
            sl = slice(i, i + BS)
            base_lp[sl] = torch.log_softmax(hooked_last_logits(Xt[sl], ids[sl], mask[sl], last[sl]), -1)
        base_p = base_lp.exp()
        for di in range(N_DIRS):
            d = torch.from_numpy(dirs[di]).to(DEVICE)
            for ri, r in enumerate(R_GRID):
                if r == 0.0:
                    continue
                pert = Xt + r * xn * d
                for i in range(0, N, BS):
                    sl = slice(i, i + BS)
                    lp = torch.log_softmax(hooked_last_logits(pert[sl], ids[sl], mask[sl], last[sl]), -1)
                    kl = (base_p[sl] * (base_lp[sl] - lp)).sum(-1)
                    out[sl, di, ri] = kl.cpu().numpy()
        del base_lp, base_p
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return out

    curves = {}
    for c, X in conditions.items():
        curves[c] = plateau_kl_curves(X)
        print(f"[{time.time()-t0:.0f}s] curves {c} done", flush=True)

    # ---- held-out tau: calibrate on real, sources < N//2; score on sources >= N//2 ----
    split = N // 2
    calib = np.arange(N) < split
    evalm = ~calib
    r02 = R_GRID.index(0.02)
    tau = float(np.median(curves["real"][calib, :, r02].mean(1)))
    print(f"[{time.time()-t0:.0f}s] tau(held-out real calib)={tau:.5f}", flush=True)

    rmax = R_GRID[-1]
    def auc_low(kl_curve):
        mono = np.maximum.accumulate(kl_curve, axis=2)
        integrand = np.clip(1 - mono / max(tau, 1e-8), 0, 1)
        return (np.trapezoid(integrand, R_GRID, axis=2) / rmax).mean(1)

    aucs = {c: auc_low(curves[c]) for c in conditions}
    dist = {c: np.linalg.norm(conditions[c] - real, axis=1) for c in conditions}

    # cooc_full: reuse recon plateau/activation but re-pair to a different source (derangement)
    aucs["cooc_full"] = aucs["recon"][perm]
    dist["cooc_full"] = np.linalg.norm(conditions["recon"][perm] - real, axis=1)
    l0_by["cooc_full"] = l0_real[perm]

    # ---- iso_displace plateau(distance) reference (EVAL half) ----
    iso_x = np.array(ISO_DELTAS)
    iso_y = np.array([np.median(aucs[f"iso{int(dd)}"][evalm]) for dd in ISO_DELTAS])
    def ref_at(distvals):
        return np.interp(np.log(np.clip(distvals, iso_x[0], iso_x[-1])),
                         np.log(iso_x), iso_y)

    # ---- distance-matched residual test (EVAL half) ----
    def boot_median(v, nb=3000):
        idx = rng.integers(0, len(v), size=(nb, len(v)))
        b = np.median(v[idx], axis=1)
        return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    TEST_CONDS = ["recon", "naive", "cooc", "cycle_consistent", "cooc_full"]
    matched = {}
    for c in TEST_CONDS:
        resid = aucs[c][evalm] - ref_at(dist[c][evalm])
        med, lo, hi = boot_median(resid)
        matched[c] = {"median_dist": round(float(np.median(dist[c][evalm])), 2),
                      "plateau_median": round(float(np.median(aucs[c][evalm])), 5),
                      "ref_at_matched_dist": round(float(np.median(ref_at(dist[c][evalm]))), 5),
                      "residual_median": round(med, 5), "residual_ci95": [round(lo, 5), round(hi, 5)],
                      "verdict": ("ABOVE random-displacement (recovers plateau)" if lo > 0
                                  else "BELOW" if hi < 0 else "indistinguishable from random displacement")}

    # cycle-error covariate: median rel cycle err of selected cycle codes vs naive vs real
    naive_relerr = cyc_relerr(torch.from_numpy(z_naive).to(DEVICE))

    # pooled Spearman plateau vs distance (eval half, SAE-decoded + iso conds)
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        dd = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        return float((rx * ry).sum() / dd) if dd > 0 else 0.0
    pool_c = TEST_CONDS + [f"iso{int(dd)}" for dd in ISO_DELTAS]
    pa = np.concatenate([aucs[c][evalm] for c in pool_c])
    pdd = np.concatenate([dist[c][evalm] for c in pool_c])
    sp_pd = round(spearman(pa, pdd), 3)

    summary = {"stage": "C", "layer": "resid_pre@6", "N": N, "n_eval": int(evalm.sum()),
               "n_dirs": N_DIRS, "r_grid": R_GRID, "tau_heldout": tau, "sub_bdec": SUB,
               "iso_deltas": ISO_DELTAS,
               "cycle_filter_thresh_real_p75": round(cyc_thresh, 4),
               "cycle_filter_pass_rate": round(cyc_pass_rate, 5),
               "cycle_generated": int(n_gen), "cycle_selected": int(min(len(sel_codes), N)),
               "cycle_sel_relerr_median": round(float(np.median(cyc_sel_relerr)), 4),
               "real_relerr_median": round(float(np.median(real_cyc)), 4),
               "naive_relerr_median": round(float(np.median(naive_relerr)), 4),
               "iso_reference_plateau": {f"d={int(dd)}": round(float(y), 5) for dd, y in zip(ISO_DELTAS, iso_y)},
               "groups_eval": {c: {"plateau_median": round(float(np.median(aucs[c][evalm])), 5),
                                   "dist_median": round(float(np.median(dist[c][evalm])), 2),
                                   "l0_median": float(np.median(l0_by[c]))} for c in list(conditions) + ["cooc_full"]},
               "distance_matched_residual": matched,
               "spearman_plateau_vs_dist_pooled_eval": sp_pd}
    with open(os.path.join(RES, "stageC_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(RES, "stageC_metrics.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["condition", "src", "plateau_auc_low", "dist_to_source", "l0", "ref_at_dist", "residual"])
        for c in TEST_CONDS:
            ref = ref_at(dist[c])
            for i in np.where(evalm)[0]:
                wr.writerow([c, int(i), round(float(aucs[c][i]), 5), round(float(dist[c][i]), 3),
                             int(l0_by[c][i]), round(float(ref[i]), 5),
                             round(float(aucs[c][i] - ref[i]), 5)])

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(iso_x, iso_y, "-o", color="0.4", lw=2, label="iso_displace (random) reference")
    colmap = {"recon": "tab:green", "naive": "tab:red", "cooc": "tab:purple",
              "cycle_consistent": "tab:blue", "cooc_full": "tab:brown"}
    for c in TEST_CONDS:
        ax[0].scatter(dist[c][evalm], aucs[c][evalm], s=8, alpha=0.25, color=colmap[c])
        mx, my = np.median(dist[c][evalm]), np.median(aucs[c][evalm])
        ax[0].scatter([mx], [my], s=130, color=colmap[c], edgecolor="k", zorder=5,
                      marker="D", label=f"{c} (median)")
    ax[0].axhline(np.median(aucs["real"][evalm]), ls=":", c="b", lw=1,
                  label=f"real (dist=0): {np.median(aucs['real'][evalm]):.3f}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("distance to source real activation")
    ax[0].set_ylabel("plateau_auc_low (higher=flatter)")
    ax[0].set_title(f"Stage C: plateau vs distance (N_eval={int(evalm.sum())})")
    ax[0].legend(fontsize=7)
    cs = TEST_CONDS
    meds = [matched[c]["residual_median"] for c in cs]
    los = [matched[c]["residual_median"] - matched[c]["residual_ci95"][0] for c in cs]
    his = [matched[c]["residual_ci95"][1] - matched[c]["residual_median"] for c in cs]
    ax[1].bar(range(len(cs)), meds, yerr=[los, his], capsize=5,
              color=[colmap[c] for c in cs], edgecolor="k")
    ax[1].set_xticks(range(len(cs))); ax[1].set_xticklabels(cs, rotation=20, ha="right", fontsize=8)
    ax[1].axhline(0, c="k", lw=1)
    ax[1].set_ylabel("plateau residual vs random displacement\n(at matched distance, 95% CI)")
    ax[1].set_title("Distance-matched residual\n(>0 = recovers plateau above random)")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "plateau_stageC.png"), dpi=110); plt.close()

    print(f"[{time.time()-t0:.0f}s] DONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
