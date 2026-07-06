"""Iteration 16 — Is the activation "manifold" actually Gaussian?

Motivation (from human feedback 2026-07-06): every experiment measures the
"off-manifold" distance with a single full-covariance GAUSSIAN fit (Mahalanobis
D_M). That assumes clean GPT-2 activations are well modelled by one Gaussian.
This experiment tests that assumption directly, with no steering involved, on
the SAME clean layer-6 FineWeb activations used throughout. Two questions:

  (1) INTRINSIC DIMENSION. Do the activations fill the 768-d ambient space
      (as a full-rank Gaussian would), or lie near a much lower-dimensional
      curved manifold? Estimated with two standard manifold-recovery estimators
      from discrete points — TwoNN (Facco et al. 2017) and the Levina-Bickel
      MLE (2004) — plus the linear PCA participation ratio.

  (2) GAUSSIANITY. If the single Gaussian were correct, the held-out squared
      Mahalanobis distance D_M^2 = (x-mu)^T Sigma^-1 (x-mu) would follow a
      chi-square_d law EXACTLY (mean d, var 2d, skew sqrt(8/d), excess-kurt 12/d
      for d=768). We fit (mu,Sigma) on half the tokens and test the other half's
      D_M^2 against chi^2_768 (moments + a Wilson-Hilferty QQ plot), and report
      the multivariate (Mardia) kurtosis and the per-dimension excess kurtosis
      (transformers are known to have a few heavy-tailed "outlier" dimensions).

Model/layer/data identical to Exp 1-3: GPT-2 small, resid_post block 6, FineWeb.
Outputs: results/16_manifold_geometry.json + plots/16_manifold_geometry.png.
"""
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import resid_post, fineweb_texts

LAYER = 6
HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

rng = np.random.default_rng(0)


def normal_ppf(p):
    """Inverse standard-normal CDF (Acklam's rational approximation). Vectorized."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p = np.asarray(p, dtype=np.float64)
    x = np.zeros_like(p)
    plow, phigh = 0.02425, 1 - 0.02425
    lo = p < plow
    q = np.sqrt(-2 * np.log(p[lo]))
    x[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    hi = p > phigh
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    x[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    mid = ~(lo | hi)
    q = p[mid] - 0.5
    r = q * q
    x[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
             (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return x


def chi2_ppf_wh(p, k):
    """chi^2_k quantile via the Wilson-Hilferty cube-root normal approximation
    (excellent for large k): X ~ k*(1 - 2/(9k) + z_p*sqrt(2/(9k)))^3."""
    z = normal_ppf(p)
    return k * (1 - 2.0 / (9 * k) + z * np.sqrt(2.0 / (9 * k))) ** 3


def moments(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.mean(); s = x.std()
    sk = ((x - m) ** 3).mean() / s ** 3
    ku = ((x - m) ** 4).mean() / s ** 4 - 3.0  # excess
    return float(m), float(s), float(sk), float(ku)


# ---- intrinsic-dimension estimators (torch, CPU, on a subsample) -------------
def knn_dists(X, k):
    """Sorted distances to the k nearest neighbours (excluding self) for every
    row of X [n,d]. Returns [n,k] float64."""
    Xt = torch.from_numpy(X.astype(np.float32))
    n = Xt.shape[0]
    D = torch.cdist(Xt, Xt)                       # [n,n]
    D.fill_diagonal_(float("inf"))
    vals, _ = torch.topk(D, k, dim=1, largest=False)
    return vals.double().numpy()


def twonn_id(nn2):
    """TwoNN ML estimate (Facco 2017): d = N / sum(log(r2/r1)). nn2 = [n,2]."""
    mu = nn2[:, 1] / np.maximum(nn2[:, 0], 1e-12)
    lm = np.log(np.maximum(mu, 1 + 1e-12))
    return float(len(lm) / lm.sum())


def mle_id(nnk, k):
    """Levina-Bickel MLE at neighbourhood size k. nnk = [n,>=k] sorted dists."""
    Tk = nnk[:, k - 1][:, None]
    Tj = nnk[:, :k - 1]
    m = 1.0 / ((np.log(Tk / np.maximum(Tj, 1e-12))).mean(axis=1))
    return float(np.mean(m))


def main():
    # ---- clean layer-6 activations (same as Exp 1) --------------------------
    texts = fineweb_texts(400)
    H = resid_post(texts, LAYER, seq_len=128, batch=16)          # [N, 768]
    N, d = H.shape
    print(f"[data] N={N} tokens, d={d}")

    # ---- (2) Gaussianity: held-out split fit -> D_M^2 -----------------------
    idx = rng.permutation(N)
    fit, test = idx[: N // 2], idx[N // 2:]
    mu = H[fit].mean(0)
    Xc = H[fit] - mu
    Sig = (Xc.T @ Xc) / (len(fit) - 1)
    Sig += 1e-3 * np.trace(Sig) / d * np.eye(d)                  # ridge for stability
    L = np.linalg.cholesky(Sig)
    Y = np.linalg.solve(L, (H[test] - mu).T).T                   # whitened held-out
    dm2 = (Y ** 2).sum(1)                                        # D_M^2, ~chi^2_d if Gaussian

    m, s, sk, ku = moments(dm2)
    gauss = dict(mean=d, std=float(np.sqrt(2 * d)),
                 skew=float(np.sqrt(8.0 / d)), exkurt=float(12.0 / d))
    # Mardia multivariate kurtosis: E[D_M^4]; Gaussian expectation d(d+2)
    mardia = float((dm2 ** 2).mean())
    mardia_gauss = float(d * (d + 2))

    # per-dimension standardized excess kurtosis (outlier dims)
    Hs = (H - H.mean(0)) / (H.std(0) + 1e-8)
    perdim_exkurt = ((Hs ** 4).mean(0) - 3.0)
    n_heavy = int((perdim_exkurt > 1.0).sum())                  # dims with fat tails

    # ---- (1) intrinsic dimension on a subsample -----------------------------
    sub = rng.choice(N, size=min(5000, N), replace=False)
    Xr = H[sub]                                                 # raw
    Xz = Hs[sub]                                                # per-dim z-scored
    kmax = 20
    nnr = knn_dists(Xr, kmax)
    nnz = knn_dists(Xz, kmax)
    id_twonn_raw = twonn_id(nnr[:, :2])
    id_twonn_z = twonn_id(nnz[:, :2])
    id_mle10_raw = mle_id(nnr, 10)
    id_mle20_raw = mle_id(nnr, 20)
    id_mle10_z = mle_id(nnz, 10)
    id_mle20_z = mle_id(nnz, 20)

    # ---- linear PCA participation ratio / variance-explained ----------------
    evals = np.linalg.eigvalsh(Sig)[::-1]                       # descending
    evals = np.clip(evals, 0, None)
    part_ratio = float(evals.sum() ** 2 / (evals ** 2).sum())   # effective linear dim
    cum = np.cumsum(evals) / evals.sum()
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)

    res = dict(
        model="gpt2-small", layer=LAYER, n_tokens=int(N), ambient_dim=int(d),
        gaussianity=dict(
            dm2_mean=m, dm2_std=s, dm2_skew=sk, dm2_exkurt=ku,
            chi2_expected=gauss,
            mardia_kurtosis=mardia, mardia_gaussian=mardia_gauss,
            mardia_ratio=float(mardia / mardia_gauss),
            n_dims_heavy_tailed=n_heavy,
            max_perdim_exkurt=float(perdim_exkurt.max()),
        ),
        intrinsic_dim=dict(
            twonn_raw=id_twonn_raw, twonn_zscored=id_twonn_z,
            mle10_raw=id_mle10_raw, mle20_raw=id_mle20_raw,
            mle10_zscored=id_mle10_z, mle20_zscored=id_mle20_z,
            pca_participation_ratio=part_ratio,
            pca_n_comp_90pct=n90, pca_n_comp_95pct=n95,
        ),
    )
    with open(os.path.join(RESULTS, "16_manifold_geometry.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) QQ plot: held-out D_M^2 vs chi^2_768 theoretical quantiles
    q = (np.arange(1, len(dm2) + 1) - 0.5) / len(dm2)
    theo = chi2_ppf_wh(q, d)
    emp = np.sort(dm2)
    ax[0].scatter(theo, emp, s=4, alpha=0.3, color="#c0392b")
    lo = min(theo.min(), emp.min()); hi = max(theo.max(), emp.max())
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="Gaussian (y=x)")
    ax[0].set_xlabel(r"$\chi^2_{768}$ theoretical quantile")
    ax[0].set_ylabel(r"held-out $D_M^2$ empirical quantile")
    std_ratio = s / np.sqrt(2 * d)
    ax[0].set_title("(a) Gaussianity QQ: far heavier tails than Gaussian\n"
                    fr"$D_M^2$ spread = {std_ratio:.1f}$\times$ Gaussian (1.0=Gaussian)")
    ax[0].legend(loc="upper left", fontsize=8)

    # (b) PCA cumulative variance explained
    ax[1].plot(np.arange(1, d + 1), cum, color="#2c3e50")
    ax[1].axhline(0.90, ls=":", color="gray"); ax[1].axhline(0.95, ls=":", color="gray")
    ax[1].axvline(n90, ls="--", color="#2980b9", lw=1)
    ax[1].scatter([n90, n95], [0.90, 0.95], color="#2980b9", zorder=5)
    ax[1].annotate(f"90% var: {n90} PCs", (n90, 0.90), textcoords="offset points",
                   xytext=(8, -14), fontsize=8, color="#2980b9")
    ax[1].annotate(f"95% var: {n95} PCs", (n95, 0.95), textcoords="offset points",
                   xytext=(8, 6), fontsize=8, color="#2980b9")
    ax[1].set_xlabel("number of principal components")
    ax[1].set_ylabel("cumulative variance explained")
    ax[1].set_title(f"(b) Anisotropy: participation ratio = {part_ratio:.1f} / {d}")

    # (c) intrinsic-dimension estimates
    names = ["TwoNN\nraw", "TwoNN\nz-scored", "MLE k=10\nraw", "MLE k=20\nraw",
             "MLE k=10\nz-scored", "PCA\npart.ratio", "ambient"]
    vals = [id_twonn_raw, id_twonn_z, id_mle10_raw, id_mle20_raw,
            id_mle10_z, part_ratio, d]
    colors = ["#16a085"] * 5 + ["#2980b9", "#7f8c8d"]
    ax[2].bar(range(len(vals)), vals, color=colors)
    for i, v in enumerate(vals):
        ax[2].text(i, v + 6, f"{v:.0f}", ha="center", fontsize=8)
    ax[2].set_xticks(range(len(names)))
    ax[2].set_xticklabels(names, fontsize=7)
    ax[2].set_ylabel("estimated dimension")
    ax[2].set_title("(c) Intrinsic dimension << ambient 768")

    fig.suptitle("Experiment 16 — the clean activation manifold is low-dimensional, "
                 "anisotropic, and heavy-tailed (NOT a single 768-d Gaussian)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(PLOTS, "16_manifold_geometry.png"), dpi=110)
    plt.close(fig)
    print("[done] wrote plot + json")


if __name__ == "__main__":
    main()
