"""Compute the AE 'ID' (kneedle elbow-k) under three reconstruction metrics
(FVU, per-dim RMSE, mean cosine similarity) and render one figure.
Unified kneedle: normalize x=log2(k) and y to [0,1], take max |curve - chord|,
which handles both decreasing (FVU/RMSE) and increasing (cosine) curves."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS, exist_ok=True)

rows = sorted(json.load(open(os.path.join(RES, "ae_results_metrics.json"))),
              key=lambda r: r["k"])
ks = np.array([r["k"] for r in rows])
fvu = np.array([r["val_fvu"] for r in rows])
rmse = np.array([r["val_rmse"] for r in rows])
cos = np.array([r["val_cos"] for r in rows])


def kneedle(ks, y):
    x = np.log2(ks.astype(float))
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (y - y.min()) / (y.max() - y.min())
    chord = yn[0] + (yn[-1] - yn[0]) * xn      # straight line first->last
    dist = np.abs(yn - chord)
    return int(ks[int(np.argmax(dist))]), dist


k_fvu, _ = kneedle(ks, fvu)
k_rmse, _ = kneedle(ks, rmse)
k_cos, _ = kneedle(ks, cos)
print(f"kneedle elbow-k: FVU={k_fvu}  RMSE={k_rmse}  cosine={k_cos}")

summary = {"elbow_k": {"fvu": k_fvu, "rmse": k_rmse, "cosine": k_cos},
           "rows": rows}
json.dump(summary, open(os.path.join(RES, "ae_metrics_elbow.json"), "w"), indent=2)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, y, name, ke, better in [
        (ax[0], fvu, "held-out FVU", k_fvu, "lower better"),
        (ax[1], rmse, "held-out RMSE (per-dim)", k_rmse, "lower better"),
        (ax[2], cos, "held-out mean cosine sim", k_cos, "higher better")]:
    a.plot(ks, y, "o-", color="#2b6cb0")
    a.axvline(ke, color="#c53030", ls="--", label=f"elbow k={ke}")
    a.set_xscale("log", base=2)
    a.set_xticks(ks)
    a.set_xticklabels(ks, rotation=45, fontsize=8)
    a.set_xlabel("bottleneck k")
    a.set_ylabel(name)
    a.set_title(f"{name}\n({better})", fontsize=10)
    a.legend()
    a.grid(alpha=0.3)
fig.suptitle("AE layer-6 bottleneck sweep: elbow-k under three reconstruction "
             "metrics (same trained models)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(PLOTS, "ae_metrics_id.png"), dpi=110)
plt.close(fig)
print("wrote plots/ae_metrics_id.png")
