"""Direction 8 — Stage B: distance-to-source matched control (the decisive ρ=-0.82 test).

Stage A found real >= recon >> naive > norm_rand in plateau_auc_low, NOT a norm artifact,
but pooled Spearman(plateau, distance-to-source)=-0.82. Stage B asks the decisive question
from the decision table: does the real/recon-vs-naive plateau gap survive matching on
distance-to-source?

Method. Build a RANDOM-DISPLACEMENT reference family `iso{delta}`: x_real + delta*d_unit
(d_unit isotropic), at delta in {15,30,60,120} -- a real activation pushed off-manifold by
a *random* direction to a controlled distance (distance == delta exactly). This traces the
plateau(distance) curve for pure random displacement of a real activation. Overlay the SAE
conditions (recon ~25, naive ~64) and a sparsity/coef-RMS-matched naive at their natural
distances.

Decisive logic:
  * If recon/naive lie ON the iso_displace plateau(distance) curve -> plateau is purely a
    closeness-to-real proxy; SAE structure adds nothing beyond distance. (decision table)
  * If recon/naive lie ABOVE iso_displace at matched distance -> SAE-decoded points are
    FLATTER than random displacement at equal distance -> plateau encodes more than distance.

tau is calibrated on a HELD-OUT real split (sources < N//2); all conditions are scored on
the EVAL half (sources >= N//2) so tau never sees the scored activations.

Reuses the in-context forward-hook plateau method from smoke_plateau.py.
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

    # ---- conditions ----
    z_real = sae_encode(Xr, sae, SUB)
    recon = sae_decode(z_real, sae)
    z_np = z_real.cpu().numpy()
    l0_real = (z_np > 0).sum(1)
    active_coefs = z_np[z_np > 0]
    feat_freq = (z_np > 0).mean(0); active_feats = np.where(feat_freq > 0)[0]
    p_feat = feat_freq[active_feats] / feat_freq[active_feats].sum()
    coef_rms_real = np.sqrt(np.array([(z_np[i, z_np[i] > 0] ** 2).mean() if l0_real[i] > 0 else 0.0
                                      for i in range(N)]))

    # naive: independent composition (k~empirical L0, idx~freq, coef~empirical) -- as Stage A
    z_naive = np.zeros((N, sae["W_dec"].shape[0]), np.float32)
    for i in range(N):
        k = int(max(1, l0_real[rng.integers(N)]))
        idx = rng.choice(active_feats, size=min(k, len(active_feats)), replace=False, p=p_feat)
        z_naive[i, idx] = rng.choice(active_coefs, size=len(idx))
    naive = sae_decode(torch.from_numpy(z_naive).to(DEVICE), sae).cpu().numpy().astype(np.float32)
    l0_naive = (z_naive > 0).sum(1)

    # sparse_match: k = source's OWN L0, coefs sampled then rescaled to source coef-RMS
    z_sm = np.zeros((N, sae["W_dec"].shape[0]), np.float32)
    for i in range(N):
        k = int(max(1, l0_real[i]))
        idx = rng.choice(active_feats, size=min(k, len(active_feats)), replace=False, p=p_feat)
        c = rng.choice(active_coefs, size=len(idx)).astype(np.float32)
        crms = np.sqrt((c ** 2).mean())
        if crms > 0 and coef_rms_real[i] > 0:
            c = c * (coef_rms_real[i] / crms)
        z_sm[i, idx] = c
    sparse_match = sae_decode(torch.from_numpy(z_sm).to(DEVICE), sae).cpu().numpy().astype(np.float32)
    l0_sm = (z_sm > 0).sum(1)

    conditions = {"real": real,
                  "recon": recon.cpu().numpy().astype(np.float32),
                  "naive": naive, "sparse_match": sparse_match}
    l0_by = {"real": l0_real, "recon": l0_real, "naive": l0_naive, "sparse_match": l0_sm}

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

    # ---- iso_displace plateau(distance) reference (EVAL half) ----
    iso_x = np.array(ISO_DELTAS)
    iso_y = np.array([np.median(aucs[f"iso{int(dd)}"][evalm]) for dd in ISO_DELTAS])
    # interpolate reference plateau at an arbitrary distance, in log-distance, clamped
    def ref_at(distvals):
        return np.interp(np.log(np.clip(distvals, iso_x[0], iso_x[-1])),
                         np.log(iso_x), iso_y)

    # ---- distance-matched residual test (EVAL half) ----
    def boot_median(v, nb=3000):
        idx = rng.integers(0, len(v), size=(nb, len(v)))
        b = np.median(v[idx], axis=1)
        return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    matched = {}
    for c in ["recon", "naive", "sparse_match"]:
        resid = aucs[c][evalm] - ref_at(dist[c][evalm])  # plateau above random-displacement at matched dist
        med, lo, hi = boot_median(resid)
        matched[c] = {"median_dist": round(float(np.median(dist[c][evalm])), 2),
                      "plateau_median": round(float(np.median(aucs[c][evalm])), 5),
                      "ref_at_matched_dist": round(float(np.median(ref_at(dist[c][evalm]))), 5),
                      "residual_median": round(med, 5), "residual_ci95": [round(lo, 5), round(hi, 5)],
                      "verdict": ("ABOVE random-displacement (survives dist-match)" if lo > 0
                                  else "BELOW" if hi < 0 else "indistinguishable from random displacement")}

    # raw paired gap vs real (eval half), for reference (Stage A style, now held-out tau)
    def boot_diff(a, b, nb=3000):
        idx = rng.integers(0, len(a), size=(nb, len(a)))
        bb = np.median(a[idx] - b[idx], axis=1)
        return float(np.median(a - b)), float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))
    raw_gap = {}
    for c in ["recon", "naive", "sparse_match"]:
        med, lo, hi = boot_diff(aucs["real"][evalm], aucs[c][evalm])
        raw_gap[c] = {"median_gap": round(med, 5), "ci95": [round(lo, 5), round(hi, 5)]}

    # pooled Spearman plateau vs distance (eval half, SAE-decoded conds only, excl real dist=0)
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        dd = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        return float((rx * ry).sum() / dd) if dd > 0 else 0.0
    pool_c = ["recon", "naive", "sparse_match"] + [f"iso{int(dd)}" for dd in ISO_DELTAS]
    pa = np.concatenate([aucs[c][evalm] for c in pool_c])
    pd = np.concatenate([dist[c][evalm] for c in pool_c])
    sp_pd = round(spearman(pa, pd), 3)

    summary = {"stage": "B", "layer": "resid_pre@6", "N": N, "n_eval": int(evalm.sum()),
               "n_dirs": N_DIRS, "r_grid": R_GRID, "tau_heldout": tau, "sub_bdec": SUB,
               "iso_deltas": ISO_DELTAS,
               "iso_reference_plateau": {f"d={int(dd)}": round(float(y), 5) for dd, y in zip(ISO_DELTAS, iso_y)},
               "groups_eval": {c: {"plateau_median": round(float(np.median(aucs[c][evalm])), 5),
                                   "dist_median": round(float(np.median(dist[c][evalm])), 2),
                                   "l0_median": float(np.median(l0_by[c]))} for c in conditions},
               "raw_gap_vs_real_eval": raw_gap,
               "distance_matched_residual": matched,
               "spearman_plateau_vs_dist_pooled_eval": sp_pd}
    with open(os.path.join(RES, "stageB_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # per-candidate csv (eval half)
    with open(os.path.join(RES, "stageB_metrics.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["condition", "src", "plateau_auc_low", "dist_to_source", "l0", "ref_at_dist", "residual"])
        for c in conditions:
            ref = ref_at(dist[c]) if c in matched else np.full(N, np.nan)
            for i in np.where(evalm)[0]:
                wr.writerow([c, int(i), round(float(aucs[c][i]), 5), round(float(dist[c][i]), 3),
                             int(l0_by[c][i]),
                             round(float(ref[i]), 5) if c in matched else "",
                             round(float(aucs[c][i] - ref[i]), 5) if c in matched else ""])

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # (left) plateau vs distance: iso reference curve + SAE conditions overlaid
    ax[0].plot(iso_x, iso_y, "-o", color="0.4", lw=2, label="iso_displace (random) reference")
    colmap = {"recon": "tab:green", "naive": "tab:red", "sparse_match": "tab:orange"}
    for c in ["recon", "naive", "sparse_match"]:
        ax[0].scatter(dist[c][evalm], aucs[c][evalm], s=10, alpha=0.35, color=colmap[c])
        mx, my = np.median(dist[c][evalm]), np.median(aucs[c][evalm])
        ax[0].scatter([mx], [my], s=140, color=colmap[c], edgecolor="k", zorder=5,
                      marker="D", label=f"{c} (median)")
    ax[0].axhline(np.median(aucs["real"][evalm]), ls=":", c="b", lw=1,
                  label=f"real (dist=0): {np.median(aucs['real'][evalm]):.3f}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("distance to source real activation")
    ax[0].set_ylabel("plateau_auc_low (higher=flatter)")
    ax[0].set_title(f"Plateau vs distance-to-source (N_eval={int(evalm.sum())})")
    ax[0].legend(fontsize=8)
    # (right) distance-matched residual: plateau ABOVE random-displacement at matched dist
    cs = ["recon", "naive", "sparse_match"]
    meds = [matched[c]["residual_median"] for c in cs]
    los = [matched[c]["residual_median"] - matched[c]["residual_ci95"][0] for c in cs]
    his = [matched[c]["residual_ci95"][1] - matched[c]["residual_median"] for c in cs]
    ax[1].bar(cs, meds, yerr=[los, his], capsize=6,
              color=[colmap[c] for c in cs], edgecolor="k")
    ax[1].axhline(0, c="k", lw=1)
    ax[1].set_ylabel("plateau residual vs random displacement\n(at matched distance, 95% CI)")
    ax[1].set_title("Distance-matched residual\n(>0 = flatter than random at equal distance)")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "plateau_stageB.png"), dpi=110); plt.close()

    print(f"[{time.time()-t0:.0f}s] DONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
