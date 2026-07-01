"""Direction 8 — end-to-end plateau pipeline (smoke + Stage-A reproduction).

Pipeline (one script, --full to scale up):
  1. Load GPT-2 small + SAE (jbloom reformatted, blocks.6.hook_resid_pre, d_sae=24576).
  2. Capture prompt-aligned REAL resid_pre@6 last-token activations (hook on block 5 output)
     via full in-context forwards over FineWeb prompts.
  3. Build candidate conditions paired to source prompts:
       real      : the real activation itself
       recon     : SAE encode->decode of the real activation
       naive     : independent latent composition (k~empirical L0, idx~freq, coef~empirical) decoded
       norm_rand : random direction rescaled to the source norm
  4. Plateau sweep (PRIMARY metric, genuinely in-context): inject candidate at the last token
     of its source prompt, perturb x(r)=x + r*||x||*d (isotropic unit d), measure
     KL(p_x || p_x(r)) where p_x is the candidate's OWN unperturbed output. tau calibrated on
     REAL train split = median real KL at r=0.02. plateau_auc_low = mean_d integral_r
     clip(1 - KL_monotone/tau,0,1) / max(r).  Higher = wider low-response plateau.

Reuses the corrected in-context forward-hook method from
  ../dir6_jacobian_real_act_detector/experiments/context_validation_v2.py (block output == resid_post@L
  == resid_pre@(L+1); overwrite ONLY the last real-token residual during a full forward).
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
HOOK_BLOCK = 5  # output of block 5 == resid_pre@6 == SAE hook point
SEQ = 64; BS = 32


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
    N_DIRS = 8 if args.full else 2
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
    keep = (mask.sum(1) >= 16)  # need real context
    ids, mask, last = ids[keep][:N], mask[keep][:N], last[keep][:N]
    ids, mask, last = ids.to(DEVICE), mask.to(DEVICE), last.to(DEVICE)
    N = ids.shape[0]
    block = m.transformer.h[HOOK_BLOCK]

    # in-context last-position logits with last-token resid_pre@6 optionally overwritten by rep
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
    print(f"[{time.time()-t0:.0f}s] captured {N} real resid_pre@6 acts", flush=True)

    Xr = torch.from_numpy(real).to(DEVICE)
    # choose b_dec convention by reconstruction error
    errs = {}
    for sub in (True, False):
        z = sae_encode(Xr, sae, sub); xh = sae_decode(z, sae)
        errs[sub] = float((xh - Xr).norm(dim=1).mean())
    SUB = min(errs, key=errs.get)
    print(f"[{time.time()-t0:.0f}s] recon-err sub_bdec True={errs[True]:.3f} False={errs[False]:.3f} -> use {SUB}", flush=True)

    # ---- conditions ----
    z_real = sae_encode(Xr, sae, SUB)
    recon = sae_decode(z_real, sae)
    recon_err = (recon - Xr).norm(dim=1).cpu().numpy()
    z_np = z_real.cpu().numpy()
    l0_real = (z_np > 0).sum(1)
    active_coefs = z_np[z_np > 0]
    feat_freq = (z_np > 0).mean(0); active_feats = np.where(feat_freq > 0)[0]
    p_feat = feat_freq[active_feats] / feat_freq[active_feats].sum()

    # naive independent composition
    naive = np.zeros((N, 768), np.float32)
    z_naive = np.zeros((N, sae["W_dec"].shape[0]), np.float32)
    for i in range(N):
        k = int(max(1, l0_real[rng.integers(N)]))
        idx = rng.choice(active_feats, size=min(k, len(active_feats)), replace=False, p=p_feat)
        z_naive[i, idx] = rng.choice(active_coefs, size=len(idx))
    naive_t = sae_decode(torch.from_numpy(z_naive).to(DEVICE), sae)
    naive = naive_t.cpu().numpy()
    l0_naive = (z_naive > 0).sum(1)

    # norm-matched random
    g = rng.standard_normal((N, 768)).astype(np.float32)
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    norm_rand = (g * np.linalg.norm(real, axis=1, keepdims=True)).astype(np.float32)

    conditions = {
        "real": real,
        "recon": recon.cpu().numpy().astype(np.float32),
        "naive": naive.astype(np.float32),
        "norm_rand": norm_rand,
    }
    l0_by = {"real": l0_real, "recon": l0_real, "naive": l0_naive, "norm_rand": np.zeros(N)}

    # fixed isotropic unit directions per prompt (shared across conditions for pairing)
    dirs = rng.standard_normal((N_DIRS, N, 768)).astype(np.float32)
    dirs = dirs / np.linalg.norm(dirs, axis=2, keepdims=True)

    def plateau_kl_curves(X):
        """X [N,768] -> KL[N, N_DIRS, len(R_GRID)] (KL vs candidate's own r=0 output)."""
        Xt = torch.from_numpy(X).to(DEVICE).float()
        xn = Xt.norm(dim=1, keepdim=True)
        out = np.zeros((N, N_DIRS, len(R_GRID)), np.float32)
        # baseline p_x (r=0)
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
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
        return out

    curves = {c: plateau_kl_curves(X) for c, X in conditions.items()}
    print(f"[{time.time()-t0:.0f}s] plateau curves done", flush=True)

    # sanity: real-vs-real r=0 KL is identically 0 by construction; check direction monotone-ish
    # tau from real (all real used as calibration here for smoke); document split in journal
    r02 = R_GRID.index(0.02)
    real_kl_r02 = curves["real"][:, :, r02].mean(1)  # mean over dirs
    tau = float(np.median(real_kl_r02))
    print(f"[{time.time()-t0:.0f}s] tau=median real KL@r=0.02 = {tau:.4f}", flush=True)

    rmax = R_GRID[-1]
    def auc_low(kl_curve):
        # kl_curve [N,N_DIRS,len(R)] -> per-activation plateau_auc_low
        mono = np.maximum.accumulate(kl_curve, axis=2)
        integrand = np.clip(1 - mono / max(tau, 1e-8), 0, 1)
        # trapezoid over r, normalized by rmax, then mean over dirs
        area = np.trapezoid(integrand, R_GRID, axis=2) / rmax
        return area.mean(1)

    rows = []
    for c in conditions:
        a = auc_low(curves[c])
        norms = np.linalg.norm(conditions[c], axis=1)
        dist = np.linalg.norm(conditions[c] - real, axis=1)
        re = recon_err if c == "recon" else (np.zeros(N) if c == "real" else dist)
        for i in range(N):
            rows.append({"condition": c, "src": i, "plateau_auc_low": round(float(a[i]), 5),
                         "norm": round(float(norms[i]), 3), "dist_to_source": round(float(dist[i]), 3),
                         "recon_err": round(float(re[i]), 3), "l0": int(l0_by[c][i]),
                         "kl_r02": round(float(curves[c][i, :, r02].mean()), 5)})

    with open(os.path.join(RES, "plateau_metrics.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    # group summary with paired-bootstrap CI of (real - cond) plateau gap
    aucs = {c: auc_low(curves[c]) for c in conditions}
    def boot_diff(a, b, nb=2000):
        idx = rng.integers(0, N, size=(nb, N))
        d = np.median(a[idx] - b[idx], axis=1)  # bootstrap the MEDIAN paired difference
        return float(np.median(a - b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    summary = {"layer": "resid_pre@6", "N": N, "n_dirs": N_DIRS, "r_grid": R_GRID,
               "tau": tau, "sub_bdec": SUB, "recon_err_mean": round(errs[SUB], 3),
               "groups": {}, "real_minus": {}}
    for c in conditions:
        summary["groups"][c] = {"plateau_auc_low_median": round(float(np.median(aucs[c])), 5),
                                "norm_median": round(float(np.median(np.linalg.norm(conditions[c], axis=1))), 3),
                                "l0_median": float(np.median(l0_by[c]))}
    for c in ["recon", "naive", "norm_rand"]:
        med, lo, hi = boot_diff(aucs["real"], aucs[c])
        summary["real_minus"][c] = {"median_gap": round(med, 5), "ci95": [round(lo, 5), round(hi, 5)]}
    # Stage-B preview: does a simple covariate explain plateau? pool all candidates.
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        return float((rx * ry).sum() / d) if d > 0 else 0.0
    all_auc = np.concatenate([aucs[c] for c in conditions])
    all_norm = np.concatenate([np.linalg.norm(conditions[c], axis=1) for c in conditions])
    all_dist = np.concatenate([np.linalg.norm(conditions[c] - real, axis=1) for c in conditions])
    summary["covariate_spearman_pooled"] = {
        "plateau_vs_norm": round(spearman(all_auc, all_norm), 3),
        "plateau_vs_dist_to_source": round(spearman(all_auc, all_dist), 3)}
    with open(os.path.join(RES, "plateau_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # mean plateau KL curve per condition
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c in conditions:
        mc = curves[c].mean(axis=(0, 1))  # mean over acts & dirs
        ax[0].plot(R_GRID, mc, marker="o", label=c)
    ax[0].axhline(tau, ls="--", c="k", lw=0.8, label=f"tau={tau:.3f}")
    ax[0].set_xlabel("perturbation radius r"); ax[0].set_ylabel("mean downstream KL")
    ax[0].set_title(f"Plateau curves (resid_pre@6, N={N})"); ax[0].legend(); ax[0].set_yscale("log")
    # plateau_auc_low distribution
    data = [aucs[c] for c in conditions]
    ax[1].boxplot(data, tick_labels=list(conditions.keys()))
    ax[1].set_ylabel("plateau_auc_low (higher=flatter)")
    ax[1].set_title("Plateau-AUC-low by condition")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "plateau_stageA.png"), dpi=110); plt.close()

    print(f"[{time.time()-t0:.0f}s] DONE. summary:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
