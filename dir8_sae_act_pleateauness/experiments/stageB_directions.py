"""Direction 8 — Stage B direction-family robustness (Section 7 / experiment-queue #5).

Stage B (isotropic perturbation directions) found: at matched distance-to-source, NO
SAE-decoded condition plateaus ABOVE a random-displacement reference; naive/sparse_match
sit BELOW it. Section 7 warns that an isotropic-only gap is "a generic local sensitivity
artifact, not evidence of SAE downstream validity", and that if SAE decoder-feature
directions REVERSE the conclusion the metric must be reported as direction-dependent.

This script recomputes the exact Stage B distance-matched residual under THREE perturbation
direction families, apples-to-apples within one run:
  * iso        — isotropic Gaussian unit directions (primary family; reproduces Stage B).
  * sae_single — each direction is a single SAE decoder column W_dec[j] (unit-normed),
                 j sampled from real-code-active features by frequency; shared across sources.
  * sae_sparse — each direction is a normalized sum of 8 random active decoder columns.

For every family we trace the iso_displace(delta) random-displacement plateau(distance)
reference and test whether recon/naive/sparse_match plateau ABOVE it at matched distance.

Decisive logic:
  * If the naive below-random deficit persists (residual<0) under SAE-decoder directions too
    -> the null is NOT an isotropic artifact; plateau is direction-robustly a
       closeness-to-real + local-robustness proxy. Scopes, does not overturn, the project null.
  * If SAE-decoder directions push naive/recon ABOVE the reference (residual>0)
    -> the metric is direction-dependent; report SAE-decoder-specific plateau validity.

Reuses the in-context forward-hook plateau method and conditions from stageB_distance.py.
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
FAMILIES = ["iso", "sae_single", "sae_sparse"]
SPARSE_K = 8


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
    SUB = True

    # ---- conditions (identical construction to stageB_distance.py) ----
    z_real = sae_encode(Xr, sae, SUB)
    recon = sae_decode(z_real, sae)
    z_np = z_real.cpu().numpy()
    l0_real = (z_np > 0).sum(1)
    active_coefs = z_np[z_np > 0]
    feat_freq = (z_np > 0).mean(0); active_feats = np.where(feat_freq > 0)[0]
    p_feat = feat_freq[active_feats] / feat_freq[active_feats].sum()
    coef_rms_real = np.sqrt(np.array([(z_np[i, z_np[i] > 0] ** 2).mean() if l0_real[i] > 0 else 0.0
                                      for i in range(N)]))

    z_naive = np.zeros((N, sae["W_dec"].shape[0]), np.float32)
    for i in range(N):
        k = int(max(1, l0_real[rng.integers(N)]))
        idx = rng.choice(active_feats, size=min(k, len(active_feats)), replace=False, p=p_feat)
        z_naive[i, idx] = rng.choice(active_coefs, size=len(idx))
    naive = sae_decode(torch.from_numpy(z_naive).to(DEVICE), sae).cpu().numpy().astype(np.float32)
    l0_naive = (z_naive > 0).sum(1)

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

    # iso_displace reference family (distance == delta), directions independent of pert family
    iso = {}
    for delta in ISO_DELTAS:
        d = rng.standard_normal((N, 768)).astype(np.float32)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        iso[f"iso{int(delta)}"] = (real + delta * d).astype(np.float32)
    conditions.update(iso)
    for k in iso:
        l0_by[k] = np.zeros(N)

    # ---- perturbation-direction families, each shape (N_DIRS, N, 768), unit-norm per row ----
    W_dec = sae["W_dec"].cpu().numpy()  # (d_sae, 768)
    def build_dirs(fam):
        if fam == "iso":
            d = rng.standard_normal((N_DIRS, N, 768)).astype(np.float32)
        elif fam == "sae_single":
            js = rng.choice(active_feats, size=N_DIRS, replace=False, p=p_feat)
            cols = W_dec[js]                                 # (N_DIRS, 768)
            d = np.repeat(cols[:, None, :], N, axis=1).astype(np.float32)
        elif fam == "sae_sparse":
            cols = np.empty((N_DIRS, 768), np.float32)
            for di in range(N_DIRS):
                js = rng.choice(active_feats, size=SPARSE_K, replace=False, p=p_feat)
                signs = rng.choice([-1.0, 1.0], size=SPARSE_K).astype(np.float32)
                cols[di] = (signs[:, None] * W_dec[js]).sum(0)
            d = np.repeat(cols[:, None, :], N, axis=1).astype(np.float32)
        d = d / np.linalg.norm(d, axis=2, keepdims=True)
        return d

    def plateau_kl_curves(X, dirs):
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

    dist = {c: np.linalg.norm(conditions[c] - real, axis=1) for c in conditions}
    split = N // 2
    evalm = np.arange(N) >= split
    calib = ~evalm
    r02 = R_GRID.index(0.02)
    rmax = R_GRID[-1]
    iso_x = np.array(ISO_DELTAS)

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        dd = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        return float((rx * ry).sum() / dd) if dd > 0 else 0.0

    def boot_median(v, nb=3000):
        idx = rng.integers(0, len(v), size=(nb, len(v)))
        b = np.median(v[idx], axis=1)
        return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    # ---- run every family ----
    per_family = {}
    for fam in FAMILIES:
        dirs = build_dirs(fam)
        curves = {c: plateau_kl_curves(conditions[c], dirs) for c in conditions}
        print(f"[{time.time()-t0:.0f}s] family {fam}: curves done", flush=True)
        # tau: held-out real calib split, family-specific
        tau = float(np.median(curves["real"][calib, :, r02].mean(1)))

        def auc_low(kl_curve):
            mono = np.maximum.accumulate(kl_curve, axis=2)
            integrand = np.clip(1 - mono / max(tau, 1e-8), 0, 1)
            return (np.trapezoid(integrand, R_GRID, axis=2) / rmax).mean(1)

        aucs = {c: auc_low(curves[c]) for c in conditions}
        iso_y = np.array([np.median(aucs[f"iso{int(dd)}"][evalm]) for dd in ISO_DELTAS])
        def ref_at(distvals):
            return np.interp(np.log(np.clip(distvals, iso_x[0], iso_x[-1])), np.log(iso_x), iso_y)

        matched = {}
        for c in ["recon", "naive", "sparse_match"]:
            resid = aucs[c][evalm] - ref_at(dist[c][evalm])
            med, lo, hi = boot_median(resid)
            matched[c] = {"median_dist": round(float(np.median(dist[c][evalm])), 2),
                          "plateau_median": round(float(np.median(aucs[c][evalm])), 5),
                          "ref_at_matched_dist": round(float(np.median(ref_at(dist[c][evalm]))), 5),
                          "residual_median": round(med, 5), "residual_ci95": [round(lo, 5), round(hi, 5)],
                          "verdict": ("ABOVE random-displacement" if lo > 0
                                      else "BELOW random-displacement" if hi < 0
                                      else "indistinguishable from random displacement")}
        pool_c = ["recon", "naive", "sparse_match"] + [f"iso{int(dd)}" for dd in ISO_DELTAS]
        pa = np.concatenate([aucs[c][evalm] for c in pool_c])
        pdd = np.concatenate([dist[c][evalm] for c in pool_c])
        per_family[fam] = {
            "tau_heldout": tau,
            "iso_reference_plateau": {f"d={int(dd)}": round(float(y), 5) for dd, y in zip(ISO_DELTAS, iso_y)},
            "plateau_median": {c: round(float(np.median(aucs[c][evalm])), 5) for c in conditions},
            "distance_matched_residual": matched,
            "spearman_plateau_vs_dist_pooled_eval": round(spearman(pa, pdd), 3),
        }
        print(f"[{time.time()-t0:.0f}s] {fam} residuals: "
              + ", ".join(f"{c} {per_family[fam]['distance_matched_residual'][c]['residual_median']:+.3f}"
                          for c in ["recon", "naive", "sparse_match"]), flush=True)

    summary = {"stage": "B_direction", "layer": "resid_pre@6", "N": N, "n_eval": int(evalm.sum()),
               "n_dirs": N_DIRS, "sparse_k": SPARSE_K, "r_grid": R_GRID, "sub_bdec": SUB,
               "iso_deltas": ISO_DELTAS, "families": FAMILIES, "per_family": per_family}
    with open(os.path.join(RES, "stageB_dir_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(RES, "stageB_dir_metrics.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["family", "condition", "median_dist", "plateau_median",
                     "ref_at_matched_dist", "residual_median", "ci_lo", "ci_hi", "verdict"])
        for fam in FAMILIES:
            for c in ["recon", "naive", "sparse_match"]:
                d = per_family[fam]["distance_matched_residual"][c]
                wr.writerow([fam, c, d["median_dist"], d["plateau_median"], d["ref_at_matched_dist"],
                             d["residual_median"], d["residual_ci95"][0], d["residual_ci95"][1], d["verdict"]])

    # ---- plot: distance-matched residual by direction family ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    cs = ["recon", "naive", "sparse_match"]
    colmap = {"iso": "0.45", "sae_single": "tab:purple", "sae_sparse": "tab:cyan"}
    labelmap = {"iso": "isotropic (primary)", "sae_single": "SAE decoder (single col)",
                "sae_sparse": f"SAE decoder (sum of {SPARSE_K})"}
    x = np.arange(len(cs)); w = 0.26
    for fi, fam in enumerate(FAMILIES):
        meds = [per_family[fam]["distance_matched_residual"][c]["residual_median"] for c in cs]
        los = [meds[k] - per_family[fam]["distance_matched_residual"][cs[k]]["residual_ci95"][0] for k in range(len(cs))]
        his = [per_family[fam]["distance_matched_residual"][cs[k]]["residual_ci95"][1] - meds[k] for k in range(len(cs))]
        ax.bar(x + (fi - 1) * w, meds, w, yerr=[los, his], capsize=4,
               color=colmap[fam], edgecolor="k", label=labelmap[fam])
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(cs)
    ax.set_ylabel("plateau residual vs random displacement\n(at matched distance, 95% CI)")
    ax.set_title("Distance-matched plateau residual by perturbation-direction family\n"
                 "(>0 = flatter than random at equal distance; N_eval="
                 f"{int(evalm.sum())}, {N_DIRS} dirs)")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "plateau_stageB_dir.png"), dpi=110); plt.close()

    print(f"[{time.time()-t0:.0f}s] DONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
