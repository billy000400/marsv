"""Pilot (freeze rho_max) + confirmatory plateau test on the trained char GPT.

Outputs a tidy result table, a frozen-calibration file, summary stats (Delta PI, bootstrap CI,
Cliff's delta), and the four required figures.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import assay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
CKPT = os.path.join(RES, "checkpoints", "ckpt_final.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PILOT_BLOCKS = list(range(11))          # sweep every block 0..10 (signal found -> full sweep)
PANEL_BLOCKS = [0, 2, 4, 6, 8, 10]      # blocks shown in the multi-panel curve figures
N_CTX = 48          # confirmatory held-out contexts
N_PILOT = 16
N_DIR = 8
N_RHO = 41
BLOCK = 128


def build_contexts(model, stoi, itos, device, seed=7):
    text = open("/tmp/tinyshakespeare.txt").read()
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data))
    val = data[n:]
    rng = np.random.default_rng(seed)
    # sample windows; keep those where model predicts the true next char correctly
    ctx, conf, want = [], [], N_CTX + N_PILOT
    tries = 0
    while len(ctx) < want * 4 and tries < 4000:
        tries += 1
        i = rng.integers(0, len(val) - BLOCK - 1)
        window = val[i:i + BLOCK]
        target = val[i + BLOCK]  # true next char after the window
        x = torch.from_numpy(window).long()[None].to(device)
        with torch.no_grad():
            logits, _ = model(x)
        p = torch.softmax(logits[0, -1], -1)
        if int(p.argmax()) == int(target):
            ctx.append(window)
            conf.append(float(p.max()))
    ctx = np.stack(ctx); conf = np.array(conf)
    order = np.argsort(-conf)  # high confidence first
    ctx, conf = ctx[order], conf[order]
    # pre-register: pilot = a slice; confirmatory = top-confidence quartile emphasis but use a fixed
    # held-out set disjoint from pilot
    pilot = ctx[:N_PILOT]
    rest = ctx[N_PILOT:]
    conf_rest = conf[N_PILOT:]
    # high-confidence quartile threshold recorded for the report
    hi_thr = float(np.quantile(conf_rest, 0.75)) if len(conf_rest) else 0.0
    confirm = rest[:N_CTX]
    return (torch.from_numpy(pilot).long(), torch.from_numpy(confirm).long(),
            conf_rest[:N_CTX], hi_thr)


def directions(C, seed=11):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(N_DIR, C, generator=g)
    return u / u.norm(dim=1, keepdim=True)


def matched_control(final_acts, seed=23):
    """Per-layer diagonal Gaussian sample rescaled to match each natural activation's norm."""
    mu = final_acts.mean(0)
    sd = final_acts.std(0)
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(final_acts.shape, generator=g) * sd + mu
    scale = final_acts.norm(dim=1, keepdim=True) / (z.norm(dim=1, keepdim=True) + 1e-9)
    return z * scale


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((ai > b).sum() for ai in a)
    lt = sum((ai < b).sum() for ai in a)
    return (gt - lt) / (len(a) * len(b))


def hier_bootstrap_delta(pi_nat, pi_ctrl, n=2000, seed=5):
    """pi_* shaped [n_ctx, n_dir]. Resample contexts then directions; return median-diff samples."""
    rng = np.random.default_rng(seed)
    nc, nd = pi_nat.shape
    out = []
    for _ in range(n):
        ci = rng.integers(0, nc, nc)
        di = rng.integers(0, nd, nd)
        a = pi_nat[ci][:, di].ravel()
        b = pi_ctrl[ci][:, di].ravel()
        out.append(np.median(a) - np.median(b))
    return np.array(out)


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    model, cfg, stoi, itos = assay.load_model(CKPT, DEVICE)
    C = cfg.n_embd

    pilot_ctx, conf_ctx, conf_vals, hi_thr = build_contexts(model, stoi, itos, DEVICE)
    print(f"contexts: pilot={pilot_ctx.shape[0]} confirm={conf_ctx.shape[0]} hi_thr={hi_thr:.3f}", flush=True)
    dirs = directions(C)

    # implementation check: alpha=0 partial-forward reproduces the full unmodified forward pass
    with torch.no_grad():
        xb = conf_ctx[:8].to(DEVICE)
        full_logits, _ = model(xb)
        r, _ = model.residuals_and_logits(xb)
        for l in [0, 5, 10]:
            _, lg = model.forward_from_block(r[l].clone(), l + 1)
            err = (lg[:, -1, :] - full_logits[:, -1, :]).abs().max().item()
            assert err < 1e-3, f"alpha=0 reconstruction mismatch at block {l}: {err}"
        print("CHECK alpha=0 reconstruction OK (max logit err < 1e-3)", flush=True)

    # cache clean residuals
    pilot_res, _ = assay.cache_clean(model, pilot_ctx, DEVICE)
    conf_res, conf_base_logits = assay.cache_clean(model, conf_ctx, DEVICE)

    # ---- PILOT: freeze rho_max per block (smallest rho with mean top-1 flip >= 0.8, else cap) ----
    pilot_grid = np.linspace(0, 12, 49)
    frozen = {}
    for l in PILOT_BLOCKS:
        s_l = assay.natural_scale(pilot_res[l][:, -1, :])
        bp = pilot_res[l][:, -1, :]
        out = assay.response_curves(model, pilot_res[l], l, bp, dirs[:4], pilot_grid, s_l, DEVICE)
        flip_mean = out["flip"].mean(axis=(0, 1))  # over bp,dir -> [n_rho]
        idx = np.where(flip_mean >= 0.8)[0]
        rho_max = float(pilot_grid[idx[0]]) if len(idx) else 12.0
        frozen[l] = {"s_l": s_l, "rho_max": rho_max, "flip_at_max": float(flip_mean[idx[0]] if len(idx) else flip_mean[-1])}
        print(f"pilot block {l}: s_l={s_l:.3f} rho_max={rho_max:.2f} flip={frozen[l]['flip_at_max']:.2f}", flush=True)
    json.dump(frozen, open(os.path.join(RES, "frozen_calibration.json"), "w"), indent=2)

    # ---- CONFIRMATORY ----
    rows = []          # tidy table
    summary = {}
    per_block_curves = {}  # for plotting
    for l in PILOT_BLOCKS:
        s_l = frozen[l]["s_l"]
        rho_max = frozen[l]["rho_max"]
        rhos = np.linspace(0, rho_max, N_RHO)
        nat_bp = conf_res[l][:, -1, :]
        ctrl_bp = matched_control(nat_bp)
        nat = assay.response_curves(model, conf_res[l], l, nat_bp, dirs, rhos, s_l, DEVICE)
        ctrl = assay.response_curves(model, conf_res[l], l, ctrl_bp, dirs, rhos, s_l, DEVICE)

        def pis(out, metric):
            arr = out[metric]  # [nc,nd,nr]
            pi = np.zeros(arr.shape[:2]); sh = np.zeros(arr.shape[:2])
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    pi[i, j] = assay.plateau_index(arr[i, j], rhos)
                    sh[i, j] = assay.boundary_sharpness(arr[i, j], rhos)
            return pi, sh

        pi_nat_h, sh_nat = pis(nat, "d_hidden")
        pi_ctrl_h, sh_ctrl = pis(ctrl, "d_hidden")
        pi_nat_j, _ = pis(nat, "jsd")
        pi_ctrl_j, _ = pis(ctrl, "jsd")

        boot = hier_bootstrap_delta(pi_nat_h, pi_ctrl_h)
        dPI = float(np.median(pi_nat_h) - np.median(pi_ctrl_h))
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        cd = cliffs_delta(pi_nat_h.ravel(), pi_ctrl_h.ravel())
        dPI_j = float(np.median(pi_nat_j) - np.median(pi_ctrl_j))
        summary[l] = {"s_l": s_l, "rho_max": rho_max,
                      "dPI_hidden": dPI, "dPI_hidden_ci": ci, "cliffs_delta": float(cd),
                      "dPI_jsd": dPI_j,
                      "median_PI_nat": float(np.median(pi_nat_h)), "median_PI_ctrl": float(np.median(pi_ctrl_h)),
                      "flip_frac_at_max_nat": float(nat["flip"][:, :, -1].mean()),
                      "mean_sharp_nat": float(sh_nat.mean()), "mean_sharp_ctrl": float(sh_ctrl.mean())}
        per_block_curves[l] = {
            "rhos": rhos.tolist(),
            "nat_mean": nat["d_hidden"].mean(axis=(0, 1)).tolist(),
            "nat_lo": np.percentile(nat["d_hidden"].reshape(-1, N_RHO), 2.5, axis=0).tolist(),
            "nat_hi": np.percentile(nat["d_hidden"].reshape(-1, N_RHO), 97.5, axis=0).tolist(),
            "ctrl_mean": ctrl["d_hidden"].mean(axis=(0, 1)).tolist(),
            "ctrl_lo": np.percentile(ctrl["d_hidden"].reshape(-1, N_RHO), 2.5, axis=0).tolist(),
            "ctrl_hi": np.percentile(ctrl["d_hidden"].reshape(-1, N_RHO), 97.5, axis=0).tolist(),
            "nat_indiv": nat["d_hidden"][:6, 0, :].tolist(),   # 6 individual rays (dir 0)
            "ctrl_indiv": ctrl["d_hidden"][:6, 0, :].tolist(),
            "pi_nat": pi_nat_h.tolist(), "pi_ctrl": pi_ctrl_h.tolist(),
        }
        # tidy rows (subsample: every 4th rho to keep file modest)
        for bt, out in [("natural", nat), ("control", ctrl)]:
            for i in range(out["d_hidden"].shape[0]):
                for j in range(N_DIR):
                    for ri in range(0, N_RHO, 4):
                        rows.append((i, j, l, bt, float(rhos[ri]),
                                     float(out["d_hidden"][i, j, ri]),
                                     float(out["jsd"][i, j, ri]),
                                     float(out["flip"][i, j, ri])))
        print(f"CONFIRM block {l}: dPI={dPI:+.4f} CI={ci} cliff={cd:+.3f} "
              f"flip@max={summary[l]['flip_frac_at_max_nat']:.2f}", flush=True)

    json.dump(summary, open(os.path.join(RES, "confirm_summary.json"), "w"), indent=2)
    json.dump(per_block_curves, open(os.path.join(RES, "confirm_curves.json"), "w"))
    import csv
    with open(os.path.join(RES, "tidy_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["context", "direction", "block", "basepoint", "rho", "d_hidden", "jsd", "flip"])
        w.writerows(rows)
    print(f"wrote {len(rows)} tidy rows", flush=True)
    make_plots(per_block_curves, summary)


def make_plots(curves, summary):
    all_blocks = PILOT_BLOCKS
    blocks = PANEL_BLOCKS
    # 1) response_by_layer
    fig, axes = plt.subplots(1, len(blocks), figsize=(3.2 * len(blocks), 3.4), sharey=False)
    for ax, l in zip(axes, blocks):
        c = curves[l]; r = np.array(c["rhos"])
        ax.plot(r, c["nat_mean"], color="C0", lw=2, label="natural")
        ax.fill_between(r, c["nat_lo"], c["nat_hi"], color="C0", alpha=0.2)
        ax.plot(r, c["ctrl_mean"], color="C3", lw=2, label="matched control")
        ax.fill_between(r, c["ctrl_lo"], c["ctrl_hi"], color="C3", alpha=0.2)
        ax.set_title(f"block {l}"); ax.set_xlabel(r"$\rho$"); ax.grid(alpha=0.3)
        if l == blocks[0]:
            ax.set_ylabel(r"$d_{hidden}$"); ax.legend(fontsize=8)
    fig.suptitle("Downstream response vs perturbation radius (95% band over rays)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "response_by_layer.png"), dpi=120); plt.close(fig)

    # 2) plateau_score_by_layer (ALL blocks 0..10)
    dPI = [summary[l]["dPI_hidden"] for l in all_blocks]
    lo = [summary[l]["dPI_hidden_ci"][0] for l in all_blocks]
    hi = [summary[l]["dPI_hidden_ci"][1] for l in all_blocks]
    pin = [summary[l]["median_PI_nat"] for l in all_blocks]
    pic = [summary[l]["median_PI_ctrl"] for l in all_blocks]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(all_blocks))
    blocks_lbl = all_blocks
    ax[0].bar(x - 0.2, pin, 0.4, label="natural", color="C0")
    ax[0].bar(x + 0.2, pic, 0.4, label="control", color="C3")
    ax[0].set_xticks(x); ax[0].set_xticklabels(blocks_lbl); ax[0].set_xlabel("intervention block")
    ax[0].set_ylabel("median PI"); ax[0].axhline(0, color="k", lw=0.8); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Plateau index (PI>0 = plateau; both < 0 = saturating)")
    yerr = [np.array(dPI) - np.array(lo), np.array(hi) - np.array(dPI)]
    ax[1].errorbar(x, dPI, yerr=yerr, fmt="o", capsize=4, color="C4")
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xticks(x); ax[1].set_xticklabels(blocks_lbl)
    ax[1].set_xlabel("intervention block"); ax[1].set_ylabel(r"$\Delta$PI (nat - ctrl)")
    ax[1].set_title("Control-calibrated plateau effect (95% CI)"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "plateau_score_by_layer.png"), dpi=120); plt.close(fig)

    # 3) individual_curves
    fig, axes = plt.subplots(1, len(blocks), figsize=(3.2 * len(blocks), 3.4))
    for ax, l in zip(axes, blocks):
        c = curves[l]; r = np.array(c["rhos"])
        for ray in c["nat_indiv"]:
            ax.plot(r, ray, color="C0", alpha=0.6, lw=1)
        for ray in c["ctrl_indiv"]:
            ax.plot(r, ray, color="C3", alpha=0.6, lw=1)
        # straight-line reference (linear response to same endpoint)
        ax.plot(r, np.array(c["nat_indiv"]).mean(0)[-1] * r / r[-1], "k--", lw=1, alpha=0.7)
        ax.set_title(f"block {l}"); ax.set_xlabel(r"$\rho$"); ax.grid(alpha=0.3)
        if l == blocks[0]:
            ax.set_ylabel(r"$d_{hidden}$ (individual rays)")
    fig.suptitle("Individual rays (blue=natural, red=control, dashed=linear ref)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "individual_curves.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
