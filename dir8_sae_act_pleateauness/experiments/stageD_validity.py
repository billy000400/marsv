"""Direction 8 — Stage D: does plateau predict downstream validity BEYOND baselines? (H4/M4)

Stages A/B settled H2 negatively: the plateau gap is a closeness-to-real proxy, not an
independent SAE-validity signal. Stage D is the project-level gate from the decision table:
"Plateau predicts downstream validity beyond all baselines" vs "Plateau detects
provenance/robustness but not downstream validity".

Independent downstream-validity target (PLAN sec 8, target 1, "output KL validity"):
For a candidate activation x_c paired to a source prompt, overwrite the last-token resid in
full context and measure
    output_kl(x_c) = KL(p_real || p_candidate)
where p_real is the next-token distribution when the REAL source activation sits in context
and p_candidate when x_c sits in context. LOW output_kl == downstream-valid (the candidate
yields nearly the same model behavior as the real activation at that prompt).

Question (H4): across a pool of candidate activations (recon, naive, sparse_match, and the
iso_displace random-displacement family at distances 15/30/60/120), does the candidate's
plateau_auc_low improve HELD-OUT prediction of log output_kl beyond distance-to-source + norm?

Design:
  * Pool every candidate (condition x source) as one row.
  * Split by SOURCE prompt (first half train, second half test) so no source leaks across.
  * Standardize features on train; fit plain linear least squares to predict log10 output_kl.
  * Compare TEST R^2 and Spearman for: baselines {dist,norm}; plateau-only; combined.
  * Partial test: residualize log_kl and plateau on {dist,norm} (linear, train-fit), then
    Spearman of residuals on TEST -> plateau's unique value beyond baselines.

Reuses the in-context forward-hook plateau method + condition builders from stageB_distance.py.
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
CAND_CONDS = ["recon", "naive", "sparse_match", "iso15", "iso30", "iso60", "iso120"]


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
    SUB = True

    # ---- candidate conditions (same constructions as Stage B) ----
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

    conditions = {"real": real,
                  "recon": recon.cpu().numpy().astype(np.float32),
                  "naive": naive, "sparse_match": sparse_match}
    for delta in ISO_DELTAS:
        d = rng.standard_normal((N, 768)).astype(np.float32)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        conditions[f"iso{int(delta)}"] = (real + delta * d).astype(np.float32)

    # perturbation directions (shared across conditions for pairing)
    dirs = rng.standard_normal((N_DIRS, N, 768)).astype(np.float32)
    dirs = dirs / np.linalg.norm(dirs, axis=2, keepdims=True)

    # ---- base next-token log-probs per condition (r=0) AND plateau curves ----
    def base_and_curves(X):
        Xt = torch.from_numpy(X).to(DEVICE).float()
        xn = Xt.norm(dim=1, keepdim=True)
        base_lp = torch.empty(N, 50257, device=DEVICE)
        for i in range(0, N, BS):
            sl = slice(i, i + BS)
            base_lp[sl] = torch.log_softmax(hooked_last_logits(Xt[sl], ids[sl], mask[sl], last[sl]), -1)
        out = np.zeros((N, N_DIRS, len(R_GRID)), np.float32)
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
        lp_cpu = base_lp.cpu().numpy()
        del base_lp, base_p
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return lp_cpu, out

    base_lp = {}; curves = {}
    for c, X in conditions.items():
        base_lp[c], curves[c] = base_and_curves(X)
        print(f"[{time.time()-t0:.0f}s] {c} base+curves done", flush=True)

    # ---- held-out tau (real, sources < N//2) ----
    split = N // 2
    calib = np.arange(N) < split
    train = calib.copy()            # train sources for the predictive model
    test = ~calib                   # held-out test sources
    r02 = R_GRID.index(0.02)
    tau = float(np.median(curves["real"][calib, :, r02].mean(1)))
    rmax = R_GRID[-1]

    def auc_low(kl_curve):
        mono = np.maximum.accumulate(kl_curve, axis=2)
        integrand = np.clip(1 - mono / max(tau, 1e-8), 0, 1)
        return (np.trapezoid(integrand, R_GRID, axis=2) / rmax).mean(1)

    plateau = {c: auc_low(curves[c]) for c in conditions}
    # local-sensitivity baseline: single fixed-radius mean KL at r=0.02 (Direction-6 plateau_kl),
    # log10 so it is on a comparable scale. Tests whether plateau_auc adds beyond local sensitivity.
    locsens = {c: np.log10(curves[c][:, :, r02].mean(1) + 1e-8) for c in conditions}

    # ---- independent validity target: output_kl = KL(p_real_source || p_candidate) ----
    real_lp = base_lp["real"]                 # (N, V) log p when REAL activation is in context
    real_p = np.exp(real_lp)
    output_kl = {}
    for c in CAND_CONDS:
        output_kl[c] = (real_p * (real_lp - base_lp[c])).sum(1)   # (N,)
    dist = {c: np.linalg.norm(conditions[c] - real, axis=1) for c in CAND_CONDS}
    norm = {c: np.linalg.norm(conditions[c], axis=1) for c in CAND_CONDS}
    LS = np.concatenate([locsens[c] for c in CAND_CONDS])

    # ---- build pooled rows: (condition, source) ----
    rows_c, rows_src = [], []
    for c in CAND_CONDS:
        rows_c += [c] * N; rows_src += list(range(N))
    rows_c = np.array(rows_c); rows_src = np.array(rows_src)
    P = np.concatenate([plateau[c] for c in CAND_CONDS])
    D = np.concatenate([dist[c] for c in CAND_CONDS])
    Nm = np.concatenate([norm[c] for c in CAND_CONDS])
    Y = np.log10(np.concatenate([output_kl[c] for c in CAND_CONDS]) + 1e-8)
    in_train = np.concatenate([train for _ in CAND_CONDS])
    in_test = ~in_train

    # ---- linear least squares with intercept; standardize on train ----
    def fit_predict(feat_cols):
        Xtr = np.column_stack([feat_cols[k][in_train] for k in feat_cols]) if feat_cols else np.zeros((in_train.sum(), 0))
        Xte = np.column_stack([feat_cols[k][in_test] for k in feat_cols]) if feat_cols else np.zeros((in_test.sum(), 0))
        mu = Xtr.mean(0) if Xtr.shape[1] else np.zeros(0)
        sd = Xtr.std(0) + 1e-8 if Xtr.shape[1] else np.ones(0)
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        Atr = np.column_stack([np.ones(Xtr.shape[0]), Xtr])
        Ate = np.column_stack([np.ones(Xte.shape[0]), Xte])
        beta, *_ = np.linalg.lstsq(Atr, Y[in_train], rcond=None)
        pred = Ate @ beta
        ytrue = Y[in_test]
        ss_res = ((ytrue - pred) ** 2).sum()
        ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
        return pred, 1 - ss_res / ss_tot

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        dd = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        return float((rx * ry).sum() / dd) if dd > 0 else 0.0

    models = {
        "baseline(dist,norm)": {"dist": D, "norm": Nm},
        "plateau_only": {"plateau": P},
        "combined(dist,norm,plateau)": {"dist": D, "norm": Nm, "plateau": P},
        "baseline+locsens(dist,norm,locsens)": {"dist": D, "norm": Nm, "locsens": LS},
        "all(dist,norm,locsens,plateau)": {"dist": D, "norm": Nm, "locsens": LS, "plateau": P},
    }
    results = {}
    ytest = Y[in_test]
    for name, fc in models.items():
        pred, r2 = fit_predict(fc)
        results[name] = {"test_R2": round(float(r2), 4),
                         "test_spearman_pred_vs_true": round(spearman(pred, ytest), 4)}

    # ---- partial test: plateau's unique value beyond a baseline set, on held-out ----
    # residualize a target on [1, *cols] using TRAIN-fit coefficients, return test residual
    def resid_on(target, cols):
        Atr = np.column_stack([np.ones(in_train.sum())] + [c[in_train] for c in cols])
        Ate = np.column_stack([np.ones(in_test.sum())] + [c[in_test] for c in cols])
        beta, *_ = np.linalg.lstsq(Atr, target[in_train], rcond=None)
        return target[in_test] - Ate @ beta
    # plateau beyond {dist,norm}
    partial = round(spearman(resid_on(P, [D, Nm]), resid_on(Y, [D, Nm])), 4)
    # plateau beyond {dist,norm,locsens} -- the discriminating control
    partial_beyond_ls = round(spearman(resid_on(P, [D, Nm, LS]), resid_on(Y, [D, Nm, LS])), 4)

    # marginal correlations (test) for context
    marg = {"spearman_plateau_vs_logkl": round(spearman(P[in_test], ytest), 4),
            "spearman_dist_vs_logkl": round(spearman(D[in_test], ytest), 4),
            "spearman_norm_vs_logkl": round(spearman(Nm[in_test], ytest), 4),
            "spearman_locsens_vs_logkl": round(spearman(LS[in_test], ytest), 4)}

    dR2 = round(results["combined(dist,norm,plateau)"]["test_R2"]
                - results["baseline(dist,norm)"]["test_R2"], 4)
    dR2_beyond_ls = round(results["all(dist,norm,locsens,plateau)"]["test_R2"]
                          - results["baseline+locsens(dist,norm,locsens)"]["test_R2"], 4)

    summary = {
        "stage": "D", "layer": "resid_pre@6", "N": N, "n_dirs": N_DIRS,
        "n_train_rows": int(in_train.sum()), "n_test_rows": int(in_test.sum()),
        "candidate_conditions": CAND_CONDS, "tau_heldout": tau,
        "target": "output_kl = KL(p_real || p_candidate), in-context last-token; lower=valid",
        "split": "by source prompt: train sources <N//2, test sources >=N//2",
        "models_heldout": results,
        "delta_R2_plateau_beyond_dist_norm": dR2,
        "delta_R2_plateau_beyond_dist_norm_locsens": dR2_beyond_ls,
        "partial_spearman_plateau_beyond_dist_norm": partial,
        "partial_spearman_plateau_beyond_dist_norm_locsens": partial_beyond_ls,
        "marginal_spearman_test": marg,
        "output_kl_median_by_condition": {c: round(float(np.median(output_kl[c])), 5) for c in CAND_CONDS},
        "verdict_vs_dist_norm": ("plateau ADDS held-out validity prediction beyond dist+norm"
                                 if dR2 > 0.02 and abs(partial) > 0.1 else
                                 "plateau adds NO held-out validity prediction beyond dist+norm"),
        "verdict_vs_local_sensitivity": ("plateau adds beyond a single-radius local-sensitivity baseline"
                                         if dR2_beyond_ls > 0.02 and abs(partial_beyond_ls) > 0.1 else
                                         "plateau adds ~nothing beyond local sensitivity -> robustness, NOT interpretability validity"),
    }
    with open(os.path.join(RES, "stageD_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(RES, "stageD_metrics.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["condition", "src", "split", "plateau_auc_low", "dist_to_source", "norm", "output_kl", "log10_output_kl"])
        for j in range(len(rows_c)):
            s = int(rows_src[j])
            wr.writerow([rows_c[j], s, "train" if in_train[j] else "test",
                         round(float(P[j]), 5), round(float(D[j]), 3), round(float(Nm[j]), 3),
                         round(float(10 ** Y[j] - 1e-8), 6), round(float(Y[j]), 4)])

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # (left) plateau vs output_kl validity target, colored by distance (test rows)
    sc = ax[0].scatter(P[in_test], ytest, c=np.log10(D[in_test] + 1), s=14, cmap="viridis", alpha=0.7)
    ax[0].set_xlabel("plateau_auc_low (higher = flatter)")
    ax[0].set_ylabel("log10 output_kl  (lower = downstream-valid)")
    ax[0].set_title(f"Validity target vs plateau (held-out, n={int(in_test.sum())})\n"
                    f"marginal ρ={marg['spearman_plateau_vs_logkl']:+.2f}, "
                    f"partial ρ|dist,norm={partial:+.2f}")
    cb = plt.colorbar(sc, ax=ax[0]); cb.set_label("log10(distance-to-source +1)")
    # (right) held-out R^2 bars
    order = ["baseline(dist,norm)", "plateau_only", "combined(dist,norm,plateau)",
             "baseline+locsens(dist,norm,locsens)", "all(dist,norm,locsens,plateau)"]
    labels = ["dist,norm", "plateau\nonly", "+plateau", "dist,norm,\nlocsens", "+plateau"]
    r2s = [results[n]["test_R2"] for n in order]
    cols = ["0.6", "tab:purple", "tab:green", "0.45", "tab:olive"]
    ax[1].bar(range(len(order)), r2s, color=cols, edgecolor="k")
    ax[1].set_xticks(range(len(order)))
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("held-out test $R^2$ for log10 output_kl")
    ax[1].set_title(f"ΔR² plateau beyond dist,norm = {dR2:+.3f}\n"
                    f"ΔR² plateau beyond dist,norm,locsens = {dR2_beyond_ls:+.3f}")
    for i, v in enumerate(r2s):
        ax[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "plateau_stageD.png"), dpi=110); plt.close()

    print(f"[{time.time()-t0:.0f}s] DONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
