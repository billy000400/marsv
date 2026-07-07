"""Render the lasse.png reproduction figure from qwen_sweep_L2_lasse.json.
Two panels matching the colleague's lasse.png (cosine similarity + relative L2
error vs bottleneck dimension), but we ALSO overlay the TRAIN curve so a reader
can tell whether the post-minimum rise is overfitting (train keeps improving) or
fixed-budget undertraining (train worsens too). Headless Agg, savefig+close."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
PLOTS = os.path.join(os.path.dirname(HERE), "plots")
os.makedirs(PLOTS, exist_ok=True)

d = json.load(open(os.path.join(R, "qwen_sweep_L2_lasse.json")))
rows = d["rows"]
ks = [r["k"] for r in rows]
v_cos = [r["val_cos"] for r in rows]; t_cos = [r["train_cos"] for r in rows]
v_rel = [r["val_rel_l2"] for r in rows]; t_rel = [r["train_rel_l2"] for r in rows]

# locate the held-out minimum / maximum (the "elbow" the colleague's plot shows)
k_best_rel = ks[min(range(len(ks)), key=lambda i: v_rel[i])]
k_best_cos = ks[max(range(len(ks)), key=lambda i: v_cos[i])]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

ax[0].plot(ks, v_cos, "o-", color="C0", label="held-out (val)")
ax[0].plot(ks, t_cos, "s--", color="C0", alpha=0.45, label="train")
ax[0].axvline(k_best_cos, color="grey", ls=":", lw=1)
ax[0].set_title(f"Reconstruction quality vs bottleneck\n(held-out cosine peaks at k={k_best_cos})")
ax[0].set_xlabel("Bottleneck dimension  k"); ax[0].set_ylabel("Cosine similarity")
ax[0].legend(frameon=False)

ax[1].plot(ks, v_rel, "o-", color="C3", label="held-out (val)")
ax[1].plot(ks, t_rel, "s--", color="C3", alpha=0.45, label="train")
ax[1].axvline(k_best_rel, color="grey", ls=":", lw=1)
ax[1].set_title(f"Reconstruction error vs bottleneck\n(held-out rel-L2 minimises at k={k_best_rel})")
ax[1].set_xlabel("Bottleneck dimension  k"); ax[1].set_ylabel("Relative L2 error")
ax[1].legend(frameon=False)

fig.suptitle(f"Qwen3-1.7B layer-2 last-token — colleague's DeepAutoencoder (67M), "
             f"{d['n_steps']} steps/k  (reproduction of lasse.png)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "qwen_ae_lasse_repro.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote qwen_ae_lasse_repro.png")
print("val rel-L2 min @ k =", k_best_rel, " val cos max @ k =", k_best_cos)
for r in rows:
    print(f"  k={r['k']:4d}  val rel-L2={r['val_rel_l2']:.4f} cos={r['val_cos']:.4f} | "
          f"train rel-L2={r['train_rel_l2']:.4f} cos={r['train_cos']:.4f}")
