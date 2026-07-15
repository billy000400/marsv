"""S2 — reproduce the paper-consistent weekday setup.

Collect layer-28 last-token residual activations for the 49 weekday-addition
prompts, build the 8-bin behavior distributions (7 weekdays + `other`), fit
PCA-64, seven ground-truth centroids, and a periodic cubic spline. Save arrays
+ validation metrics + a diagnostic PCA plot.
"""
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C


@torch.no_grad()
def collect():
    model, tok = C.load_model()
    rows = C.build_prompts()
    assert len(rows) == 49
    # validate 7 prompts per ground-truth weekday
    from collections import Counter
    cnt = Counter(r["gt"] for r in rows)
    assert all(cnt[w] == 7 for w in C.WEEKDAYS), cnt

    wids = C.weekday_token_ids(tok)
    for w in C.WEEKDAYS:
        assert len(wids[w]) > 0, f"no token ids for {w}"

    acts = np.zeros((49, model.config.hidden_size), dtype=np.float32)
    dist8 = np.zeros((49, 8), dtype=np.float64)  # 7 weekdays + other
    pred_correct = np.zeros(49, dtype=bool)

    for i, r in enumerate(rows):
        enc = tok(r["prompt"], return_tensors="pt")
        ids = enc.input_ids.to(model.device)
        out = model(ids, output_hidden_states=True)
        h = out.hidden_states[C.LAYER][0, -1]           # [hidden] resid @ layer 28, last tok
        acts[i] = h.float().cpu().numpy()
        logits = out.logits[0, -1].float()              # next-token logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        wm = np.array([probs[wids[w]].sum() for w in C.WEEKDAYS])
        dist8[i, :7] = wm
        dist8[i, 7] = max(0.0, 1.0 - wm.sum())
        pred_correct[i] = int(np.argmax(wm)) == r["gt_idx"]

    return rows, wids, acts, dist8, pred_correct


def fit_manifold(acts, rows, n_pca=64):
    from sklearn.decomposition import PCA
    # Only 49 activations exist, so the activation subspace has rank <= 48.
    # PLAN/paper request "PCA-64"; we retain all non-degenerate components.
    # The first-32 optimization subspace and PCA-32 recovery metric are
    # unaffected (both <= 48). Documented deviation.
    n_pca = min(n_pca, acts.shape[0] - 1)
    pca = PCA(n_components=n_pca, svd_solver="full")
    Z = pca.fit_transform(acts)                         # [49, 64]
    gt = np.array([r["gt_idx"] for r in rows])
    centroids = np.stack([Z[gt == d].mean(0) for d in range(7)])  # [7, 64]
    return pca, Z, centroids


def periodic_spline(centroids, n_samples=400):
    """Periodic cubic spline through the 7 weekday centroids (Appendix A.3).
    Parameterize at uniform knots t=0..6 with period 7; sample densely."""
    from scipy.interpolate import CubicSpline
    t = np.arange(7)
    cs = CubicSpline(np.append(t, 7), np.vstack([centroids, centroids[0]]),
                     bc_type="periodic", axis=0)
    ts = np.linspace(0, 7, n_samples, endpoint=False)
    return cs, ts, cs(ts)


def main():
    raw_path = os.path.join(C.RESULTS, "raw_collect.npz")
    rows = C.build_prompts()
    if os.path.exists(raw_path):
        d = np.load(raw_path, allow_pickle=True)
        acts, dist8, correct = d["acts"], d["dist8"], d["correct"]
        wids = d["wids"].item()
    else:
        rows, wids, acts, dist8, correct = collect()
        np.savez(raw_path, acts=acts, dist8=dist8, correct=correct,
                 wids=np.array(wids, dtype=object))
    pca, Z, centroids = fit_manifold(acts, rows)
    cs, ts, spline_pts = periodic_spline(centroids)

    acc = float(correct.mean())
    weekday_mass = float(dist8[:, :7].sum(1).mean())
    other_mass = float(dist8[:, 7].mean())
    cumvar = float(pca.explained_variance_ratio_[:64].sum())

    np.savez(os.path.join(C.RESULTS, "weekday_setup.npz"),
             acts=acts, Z=Z, centroids=centroids, dist8=dist8,
             gt_idx=np.array([r["gt_idx"] for r in rows]),
             entity_idx=np.array([r["entity_idx"] for r in rows]),
             k=np.array([r["k"] for r in rows]),
             pca_mean=pca.mean_, pca_components=pca.components_,
             explained_variance=pca.explained_variance_,
             spline_ts=ts, spline_pts=spline_pts,
             correct=correct)
    meta = dict(
        model=C.MODEL_ID, layer=C.LAYER, n_prompts=49,
        task_accuracy=acc, mean_weekday_mass=weekday_mass,
        mean_other_mass=other_mass, pca64_cumvar=cumvar,
        weekday_token_ids={w: wids[w] for w in C.WEEKDAYS},
        prompts=[r["prompt"] for r in rows],
        gt=[r["gt"] for r in rows],
    )
    with open(os.path.join(C.RESULTS, "weekday_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # diagnostic PCA plot (PC1-PC2)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    gt = np.array([r["gt_idx"] for r in rows])
    cmap = plt.get_cmap("hsv")
    for d in range(7):
        m = gt == d
        ax.scatter(Z[m, 0], Z[m, 1], color=cmap(d / 7), s=28,
                   label=C.WEEKDAYS[d], alpha=0.75, edgecolor="k", linewidth=0.3)
        ax.scatter(centroids[d, 0], centroids[d, 1], color=cmap(d / 7),
                   s=180, marker="*", edgecolor="k", linewidth=0.8, zorder=5)
    ax.plot(spline_pts[:, 0], spline_pts[:, 1], "k-", lw=1.3,
            label="periodic spline", zorder=4)
    # close the loop visually
    ax.plot([spline_pts[-1, 0], spline_pts[0, 0]],
            [spline_pts[-1, 1], spline_pts[0, 1]], "k-", lw=1.3)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Weekday activations @ layer {C.LAYER} (PCA)\n"
                 f"acc={acc:.2f}  weekday mass={weekday_mass:.2f}  "
                 f"other={other_mass:.2f}")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(C.PLOTS, "s2_pca_weekday_manifold.png"), dpi=130)
    plt.close(fig)

    print(json.dumps(dict(task_accuracy=acc, mean_weekday_mass=weekday_mass,
                          mean_other_mass=other_mass, pca64_cumvar=cumvar,
                          n_weekday_variant_tokens={w: len(wids[w]) for w in C.WEEKDAYS}),
                     indent=2))


if __name__ == "__main__":
    main()
