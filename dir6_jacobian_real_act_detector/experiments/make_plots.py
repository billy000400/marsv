"""Render all RESULTS.md/REPORT.md figures from the cached result CSVs.

Environment has no matplotlib (only numpy + PIL), so figures are drawn with PIL.
Every quantitative result table in RESULTS.md gets one PNG under ../plots/.
Run: python experiments/make_plots.py  (CPU only, no GPU/model needed).
"""
import csv, os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "plots")
os.makedirs(OUT, exist_ok=True)

PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
           (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127)]
F = lambda s: ImageFont.load_default(s)


def _read(path):
    with open(os.path.join(RES, path)) as f:
        return list(csv.DictReader(f))


def _text(d, xy, s, font, fill=(0, 0, 0), anchor="la"):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def grouped_bars(fname, title, groups, series, data, ylim, ref=None,
                 ylabel="AUROC", W=1180, H=620):
    """data[series_name][group_index] -> value. ref = (y, label) dashed line."""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    L, R, T, B = 80, 240, 56, 110
    pw, ph = W - L - R, H - T - B
    y0, y1 = ylim
    def Y(v): return T + ph * (1 - (v - y0) / (y1 - y0))
    _text(d, (W / 2, 20), title, F(22), anchor="ma")
    # y axis grid + labels
    nt = 6
    for i in range(nt + 1):
        v = y0 + (y1 - y0) * i / nt
        yy = Y(v)
        d.line([(L, yy), (L + pw, yy)], fill=(230, 230, 230))
        _text(d, (L - 8, yy), f"{v:.2f}", F(15), anchor="rm")
    d.line([(L, T), (L, T + ph)], fill=(0, 0, 0))
    d.line([(L, Y(y0)), (L + pw, Y(y0))], fill=(0, 0, 0))
    _text(d, (L, T - 14), ylabel, F(15), (90, 90, 90), anchor="lb")
    if ref is not None:
        ry = Y(ref[0])
        for x in range(L, L + pw, 12):
            d.line([(x, ry), (x + 6, ry)], fill=(120, 120, 120), width=2)
        _text(d, (L + pw - 4, ry - 4), ref[1], F(14), (90, 90, 90), anchor="rs")
    # zero line if range crosses 0
    if y0 < 0 < y1:
        d.line([(L, Y(0)), (L + pw, Y(0))], fill=(0, 0, 0))
    ng, ns = len(groups), len(series)
    gw = pw / ng
    bw = gw * 0.8 / ns
    for gi, g in enumerate(groups):
        gx = L + gi * gw + gw * 0.1
        for si, sname in enumerate(series):
            v = data[sname][gi]
            x = gx + si * bw
            yv = Y(v)
            yb = Y(max(y0, min(0, y1))) if y0 < 0 < y1 else Y(y0)
            top, bot = min(yv, yb), max(yv, yb)
            d.rectangle([x, top, x + bw * 0.92, bot], fill=PALETTE[si % len(PALETTE)])
        _text(d, (gx + gw * 0.4, T + ph + 10), g, F(15), anchor="ma")
    # legend
    lx, ly = L + pw + 24, T + 6
    for si, sname in enumerate(series):
        d.rectangle([lx, ly, lx + 18, ly + 18], fill=PALETTE[si % len(PALETTE)])
        _text(d, (lx + 26, ly + 9), sname, F(15), anchor="lm")
        ly += 26
    img.save(os.path.join(OUT, fname))
    print("wrote", fname)


def heatmap(fname, title, rows, cols, M, vmin=0.0, vmax=1.0, W=1080, H=560):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    L, R, T, B = 150, 30, 56, 120
    pw, ph = W - L - R, H - T - B
    _text(d, (W / 2, 20), title, F(22), anchor="ma")
    nr, nc = len(rows), len(cols)
    cw, ch = pw / nc, ph / nr
    def color(v):
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
        # blue(low)->white(mid)->red(high)
        if t < 0.5:
            s = t / 0.5
            return (int(40 + 215 * s), int(80 + 175 * s), 200)
        s = (t - 0.5) / 0.5
        return (255, int(255 - 200 * s), int(255 - 235 * s))
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            v = M[ri][ci]
            x, y = L + ci * cw, T + ri * ch
            d.rectangle([x, y, x + cw, y + ch], fill=color(v), outline=(255, 255, 255))
            fill = (0, 0, 0) if 0.25 < (v - vmin) / (vmax - vmin) < 0.78 else (255, 255, 255)
            _text(d, (x + cw / 2, y + ch / 2), f"{v:.2f}", F(15), fill, anchor="mm")
        _text(d, (L - 8, T + ri * ch + ch / 2), r, F(15), anchor="rm")
    for ci, c in enumerate(cols):
        _text(d, (L + ci * cw + cw / 2, T + ph + 8), c, F(14), anchor="ma")
    img.save(os.path.join(OUT, fname))
    print("wrote", fname)


# ---- Fig 1: Phase 2 baseline AUROC @ L6 ----
rows = _read("baseline_metrics.csv")
fams = ["iso_gauss", "cov_gauss", "shuffle_coord", "norm_pert"]
bases = ["norm", "mean_l2", "mahalanobis", "pca_recon", "coord_quantile", "knn_distance"]
lut = {(r["family"], r["baseline"]): float(r["auroc"]) for r in rows}
data = {b: [lut[(f, b)] for f in fams] for b in bases}
grouped_bars("fig1_baselines_L6.png",
             "Phase 2 - baseline AUROC by negative family (GPT-2 small, resid_post L6)",
             fams, bases, data, (0.4, 1.0), ref=(0.5, "chance 0.5"))

# ---- Fig 2: Phase 4 LOFO detectors vs kNN ----
rows = _read("detector_metrics.csv")
g = [r["held_out_family"] for r in rows if r["held_out_family"] not in ("HELD_IN_all4",)]
lr = {r["held_out_family"]: float(r["logreg_auroc"]) for r in rows}
ml = {r["held_out_family"]: float(r["mlp_auroc"]) for r in rows}
grouped_bars("fig2_detectors_lofo.png",
             "Phase 4 - learned detectors, leave-one-family-out (vs unsupervised kNN 0.913)",
             g, ["logreg", "mlp"], {"logreg": [lr[k] for k in g], "mlp": [ml[k] for k in g]},
             (0.4, 1.0), ref=(0.913, "kNN baseline 0.913"))

# ---- Fig 3: Phase 2b cross-layer heatmap (L6, all 7 families) ----
rows = _read("baseline_layers_metrics.csv")
fams7 = ["iso_gauss", "cov_gauss", "shuffle_coord", "norm_pert", "interp", "tangent_pert", "orth_pert"]
lut6 = {(r["family"], r["baseline"]): float(r["auroc"]) for r in rows if r["layer"] == "6"}
M = [[lut6[(f, b)] for f in fams7] for b in bases]
heatmap("fig3_baselines_L6_heatmap.png",
        "Phase 2b - baseline AUROC heatmap @ L6 (interp defeats every statistic)",
        bases, fams7, M, vmin=0.0, vmax=1.0)

# ---- Fig 4: interp AUROC across layers (best statistical scores ~chance) ----
def col(layer, fam, b):
    return next(float(r["auroc"]) for r in rows
               if r["layer"] == layer and r["family"] == fam and r["baseline"] == b)
layers = ["3", "6", "9"]
sc = ["knn_distance", "mahalanobis", "mean_l2"]
data = {b: [col(l, "interp", b) for l in layers] for b in sc}
grouped_bars("fig4_interp_across_layers.png",
             "Phase 2b - interp (one-sided 'too-far') AUROC across layers ~ chance",
             ["L3", "L6", "L9"], sc, data, (0.0, 1.0), ref=(0.5, "chance 0.5"))

# ---- Fig 5: functional probe ----
rows = _read("functional_metrics.csv")
ffams = ["cov_gauss", "norm_pert", "interp", "tangent_pert"]
feats = ["entropy", "msp", "plateau_kl", "logit_max"]
lut = {(r["family"], r["feature"]): float(r["auroc_oriented"]) for r in rows}
data = {ft: [lut[(f, ft)] for f in ffams] for ft in feats}
grouped_bars("fig5_functional_probe.png",
             "Phase 3 - functional probe AUROC (oriented) by family",
             ffams, feats, data, (0.4, 1.0), ref=(0.5, "chance 0.5"))

# ---- Fig 6: combined score (single scores two-sided + COMBINED lofo) ----
rows = _read("combined_metrics.csv")
cfams = ["cov_gauss", "norm_pert", "interp", "tangent_pert"]
csc = ["mahalanobis", "knn", "entropy", "plateau_kl", "COMBINED_lofo"]
lut = {(r["family"], r["score"]): float(r["auroc"]) for r in rows}
data = {s: [lut[(f, s)] for f in cfams] for s in csc}
grouped_bars("fig6_combined_score.png",
             "Phase 3 capstone - combined detector (LOFO) vs single scores",
             cfams, csc, data, (0.4, 1.0), ref=(0.5, "chance 0.5"))

# ---- Fig 7: Phase 5 prediction partial rho ----
rows = _read("functional_prediction_v2.csv")
psc = ["plateau_kl", "entropy", "maha_twosided", "knn_distance"]
def pr(sweep, s):
    return next(float(r["partial_spearman_ctrl_dist"]) for r in rows
               if r["sweep"] == sweep and r["score"] == s)
data = {s: [pr("noise", s), pr("interp", s)] for s in psc}
grouped_bars("fig7_prediction_partial_rho.png",
             "Phase 5 - partial Spearman(score, in-context KL | dist_to_orig)",
             ["noise sweep", "interp sweep"], psc, data, (-0.4, 0.7),
             ylabel="partial rho")

# ---- Fig 8: Phase 6 causal repair external KL ----
rows = _read("repair_metrics_v2.csv")
order = ["corrupted(start)", "maha_descent", "func_descent", "shrink_mean(matched)",
         "random_move(maha-matched)", "random_move(func-matched)",
         "shrink_clean(oracle,matched)"]
lut = {r["method"]: float(r["ext_KL_clean"]) for r in rows}
short = ["corrupted", "maha\ndescent", "func\ndescent", "shrink\nmean",
         "rand\n(maha)", "rand\n(func)", "oracle\nclean"]
data = {"ext KL(clean||x)": [lut[m] for m in order]}
grouped_bars("fig8_causal_repair_KL.png",
             "Phase 6 - causal repair: external downstream KL (lower=better)",
             short, ["ext KL(clean||x)"], data, (0.0, 9.5),
             ylabel="KL(clean||x)", W=1180)

print("done")
