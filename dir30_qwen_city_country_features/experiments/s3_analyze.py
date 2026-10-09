"""S2 selection, S3 overlap and S4 early-position readout from the cached sparse activations (no model runs).
Writes feature_rankings.csv, feature_sets.json, feature_overlap.csv, shared_unique_ids.json,
early_target_features.csv, candidates.json and the figures in plots/."""
import collections
import csv
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import N_LAYERS, OUT, PLOTS

GROUPS = ["Tokyo", "Japan", "Berlin", "Germany"]
HEADLINE = [("Tokyo", "Japan"), ("Berlin", "Germany")]
CUTOFFS = [10, 20, 50]
MAIN = 20
MIN_FACTS = 2
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
HATCH = ["", "//", "..", "\\\\", "xx"]

rows = [json.loads(l) for l in open(os.path.join(OUT, "samples.jsonl"))]
cell = lambda r: (r["relation"], r["template"])
bg_cells = collections.defaultdict(list)
for i, r in enumerate(rows):
    if r["correct"] and r["group"] == "background":
        bg_cells[cell(r)].append(i)


def fact_rows(group, role, supported_only):
    """fact_id -> row indices, optionally restricted to cells that have a correct background row."""
    out = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["correct"] and r["group"] == group and r["role"] == role and (not supported_only or cell(r) in bg_cells):
            out[r["fact_id"]].append(i)
    return out


def fact_mean(M, facts):
    """Average rows within each fact, then across facts. M is [n_rows_total, n_features]."""
    if not facts:
        return np.full(M.shape[1], np.nan)
    return np.mean([M[idx].mean(0) for idx in facts.values()], 0)


disc = {g: fact_rows(g, "discovery", True) for g in GROUPS}
disc_all = {g: fact_rows(g, "discovery", False) for g in GROUPS}
held = {g: fact_rows(g, "heldout", False) for g in GROUPS}
support = {}
weights = {}
for g in GROUPS:
    w = collections.Counter()
    for idx in disc[g].values():
        for i in idx:
            w[cell(rows[i])] += 1 / len(idx) / len(disc[g])
    weights[g] = w
    support[g] = dict(discovery_facts=len(disc_all[g]), supported_facts=len(disc[g]),
                      discovery_prompts=sum(map(len, disc_all[g].values())),
                      supported_prompts=sum(map(len, disc[g].values())),
                      cells={" | ".join(c): dict(weight=round(v, 4), background_rows=len(bg_cells[c])) for c, v in w.items()},
                      sufficient=len(disc[g]) >= MIN_FACTS)

ranking, early, layer_l0 = [], {}, []
for L in range(N_LAYERS):
    z = np.load(os.path.join(OUT, "acts", f"layer_{L}.npz"))
    feats = np.unique(z["feature"])
    col = {f: j for j, f in enumerate(feats)}
    V = np.zeros((len(rows), len(feats)), np.float64)
    V[z["sample_index"], [col[f] for f in z["feature"]]] = z["value"]
    A = (V > 0).astype(np.float64)
    names = list(z["early_row_names"])
    early[L] = {(names[r], int(f)): float(v) for r, f, v in zip(z["early_row"], z["early_feature"], z["early_value"])}
    layer_l0.append({n: int((z["early_row"] == k).sum()) for k, n in enumerate(names)})
    rate = {g: fact_mean(A, disc[g]) for g in GROUPS}
    act = {g: fact_mean(V, disc[g]) for g in GROUPS}
    hrate = {g: fact_mean(A, held[g]) for g in GROUPS}
    for g in GROUPS:
        if not support[g]["sufficient"]:
            continue
        bg = sum(w * A[bg_cells[c]].mean(0) for c, w in weights[g].items())
        n_facts = np.sum([A[idx].max(0) for idx in disc[g].values()], 0)
        score = rate[g] - bg
        ok = np.where((score > 1e-6) & (n_facts >= MIN_FACTS))[0]  # 1e-6 guards against rounding noise at score 0
        ok = sorted(ok, key=lambda j: (-round(float(score[j]), 6), -round(float(rate[g][j]), 6), feats[j]))
        for rank, j in enumerate(ok, 1):
            row = dict(group=g, layer=L, feature=int(feats[j]), rank_in_layer=rank, score=float(score[j]),
                       positive_rate=float(rate[g][j]), background_rate=float(bg[j]), n_active_facts=int(n_facts[j]),
                       heldout_last_token_rate=float(hrate[g][j]))
            for h in GROUPS:
                row[f"rate_{h}"] = float(rate[h][j])
                row[f"mean_act_{h}"] = float(act[h][j])
            row["background_rate_unweighted"] = float(A[sum(bg_cells.values(), [])].mean(0)[j])
            row["active_prompts"] = [(rows[i]["group"], rows[i]["prompt"], round(float(V[i, j]), 3))
                                     for i in np.where(V[:, j] > 0)[0] if rows[i]["group"] != "background"]
            ranking.append(row)

sets = {k: {g: sorted((r["layer"], r["feature"]) for r in ranking if r["group"] == g and r["rank_in_layer"] <= k)
            for g in GROUPS} for k in CUTOFFS}
with open(os.path.join(OUT, "feature_rankings.csv"), "w", newline="") as fh:
    cols = [c for c in ranking[0] if c != "active_prompts"]
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(ranking)
json.dump(dict(definition="feature id = [layer, feature_index]; top-k per layer among qualifying features",
               min_active_facts=MIN_FACTS, support=support,
               sets={f"top{k}": {g: [list(x) for x in s] if support[g]["sufficient"] else "insufficient data"
                                 for g, s in v.items()} for k, v in sets.items()}),
          open(os.path.join(OUT, "feature_sets.json"), "w"), indent=1)

# ---- S3 overlap ----
ratio = lambda a, b: a / b if b else "N/A"
overlap = []
for k in CUTOFFS:
    for scope in ["all_layers_pooled"] + list(range(N_LAYERS)):
        for a, b in itertools.combinations(GROUPS, 2):
            if not (support[a]["sufficient"] and support[b]["sufficient"]):
                overlap.append(dict(cutoff=k, scope=scope, group_a=a, group_b=b, note="insufficient data"))
                continue
            sa = {x for x in sets[k][a] if scope == "all_layers_pooled" or x[0] == scope}
            sb = {x for x in sets[k][b] if scope == "all_layers_pooled" or x[0] == scope}
            sh = sa & sb
            overlap.append(dict(cutoff=k, scope=scope, group_a=a, group_b=b, n_a=len(sa), n_b=len(sb), shared=len(sh),
                                unique_a=len(sa - sb), unique_b=len(sb - sa), shared_over_a=ratio(len(sh), len(sa)),
                                shared_over_b=ratio(len(sh), len(sb)), jaccard=ratio(len(sh), len(sa | sb)), note=""))
with open(os.path.join(OUT, "feature_overlap.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["cutoff", "scope", "group_a", "group_b", "n_a", "n_b", "shared", "unique_a",
                                       "unique_b", "shared_over_a", "shared_over_b", "jaccard", "note"])
    w.writeheader()
    w.writerows(overlap)
by_key = {(r["group"], r["layer"], r["feature"]): r for r in ranking}
S = {g: set(sets[MAIN][g]) for g in GROUPS}
rates_of = lambda g, x: {h: round(by_key[(g, *x)][f"rate_{h}"], 3) for h in GROUPS}
json.dump({f"{a}_vs_{b}": dict(shared=[dict(id=list(x), rates=rates_of(a, x)) for x in sorted(S[a] & S[b])],
                               unique_a=[list(x) for x in sorted(S[a] - S[b])],
                               unique_b=[list(x) for x in sorted(S[b] - S[a])])
           for a, b in itertools.combinations(GROUPS, 2)},
          open(os.path.join(OUT, "shared_unique_ids.json"), "w"), indent=1)

# ---- overlap split by discovery score (added after seeing that most selected features have low scores) ----
STRONG = 0.5
score_of = {(r["group"], r["layer"], r["feature"]): r["score"] for r in ranking if r["rank_in_layer"] <= MAIN}
strata = []
for thr in [0.25, 0.5, 0.75]:
    for a in GROUPS:
        for b in GROUPS:
            if a == b:
                continue
            hi = [x for x in S[a] if score_of[(a, *x)] >= thr]
            lo = [x for x in S[a] if score_of[(a, *x)] < thr]
            strata.append(dict(threshold=thr, set_a=a, set_b=b, n_strong=len(hi), strong_also_in_b=sum(x in S[b] for x in hi),
                               n_weak=len(lo), weak_also_in_b=sum(x in S[b] for x in lo)))
with open(os.path.join(OUT, "overlap_by_score.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(strata[0]))
    w.writeheader()
    w.writerows(strata)

# ---- S4 early position ----
partner = {"Tokyo": "Japan", "Japan": "Tokyo", "Berlin": "Germany", "Germany": "Berlin"}
erows = []
for country in ["Japan", "Germany"]:
    for g in GROUPS:
        for (L, f) in sorted(S[g]):
            vp, vf = early[L].get((f"{country}|prefix", f), 0.0), early[L].get((f"{country}|full", f), 0.0)
            erows.append(dict(country_position=country, feature_set=g, layer=L, feature=f,
                              active_before_is=vp > 0, value_prefix_run=vp, value_full_run=vf,
                              status=f"shared with {partner[g]}" if (L, f) in S[partner[g]] else f"unique to {g}",
                              discovery_score=by_key[(g, L, f)]["score"]))
with open(os.path.join(OUT, "early_target_features.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(erows[0]))
    w.writeheader()
    w.writerows(erows)
early_summary = []
for country in ["Japan", "Germany"]:
    for g in GROUPS:
        sel = [e for e in erows if e["country_position"] == country and e["feature_set"] == g]
        act_ = [e for e in sel if e["active_before_is"]]
        early_summary.append(dict(country_position=country, feature_set=g, set_size=len(sel), active=len(act_),
                                  active_shared=sum(e["status"].startswith("shared") for e in act_),
                                  active_unique=sum(e["status"].startswith("unique") for e in act_),
                                  layers_active=sorted({e["layer"] for e in act_}),
                                  max_abs_diff_prefix_vs_full=max(abs(e["value_prefix_run"] - e["value_full_run"]) for e in sel) if sel else None))
total_active_early = {n: sum(l[n] for l in layer_l0) for n in layer_l0[0]}

# ---- inspection panel: up to five highest-ranked candidates per entity ----
cands = {}
for g in GROUPS:
    rr = sorted((r for r in ranking if r["group"] == g and r["rank_in_layer"] <= MAIN),
                key=lambda r: (-round(r["score"], 6), -round(r["positive_rate"], 6), -r[f"mean_act_{g}"]))[:5]
    for r in rr:
        c = "Japan" if g in ("Tokyo", "Japan") else "Germany"
        r["early_value_own_country"] = early[r["layer"]].get((f"{c}|prefix", r["feature"]), 0.0)
        r["early_value_Japan"] = early[r["layer"]].get(("Japan|prefix", r["feature"]), 0.0)
        r["early_value_Germany"] = early[r["layer"]].get(("Germany|prefix", r["feature"]), 0.0)
    cands[g] = rr
json.dump(cands, open(os.path.join(OUT, "candidates.json"), "w"), indent=1, ensure_ascii=False)

summary = dict(support=support, set_sizes={f"top{k}": {g: len(sets[k][g]) for g in GROUPS} for k in CUTOFFS},
               qualifying={g: sum(r["group"] == g for r in ranking) for g in GROUPS},
               pooled_overlap=[o for o in overlap if o["scope"] == "all_layers_pooled"],
               early_summary=early_summary, total_active_features_at_country_position=total_active_early,
               heldout_last_token_mean_rate={g: float(np.nanmean([by_key[(g, *x)]["heldout_last_token_rate"] for x in S[g]]))
                                             if S[g] and held[g] else None for g in GROUPS})
summary["median_score"] = {g: float(np.median([score_of[(g, *x)] for x in S[g]])) for g in GROUPS}
summary["overlap_by_score_0.5"] = [s for s in strata if s["threshold"] == STRONG]
summary["strong_heldout_last_token_rate"] = {
    g: float(np.mean([by_key[(g, *x)]["heldout_last_token_rate"] for x in S[g] if score_of[(g, *x)] >= STRONG]))
    for g in ("Tokyo", "Berlin")}
cfg_path = os.path.join(OUT, "run_config.json")
cfg = json.load(open(cfg_path))
cfg["feature_selection"] = dict(
    layers=list(range(N_LAYERS)), extraction_position="last prompt token (index = n_tokens - 1)", active="activation > 0",
    score="positive_rate - background_rate", min_active_facts=MIN_FACTS, main_cutoff_per_layer=MAIN, cutoffs=CUTOFFS,
    tie_break="score, then positive_rate, then feature index", strong_score_threshold_post_hoc=STRONG,
    early_position="last sub-token of the country name in 'The capital of <country>'")
json.dump(cfg, open(cfg_path, "w"), indent=1)
json.dump(summary, open(os.path.join(OUT, "analysis_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))

# ---- figures ----
plt.rcParams.update({"font.size": 10, "axes.prop_cycle": plt.cycler(color=CVD)})
counts = {(r["group"], r["relation"]): r for r in csv.DictReader(open(os.path.join(OUT, "sample_counts.csv")))}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
stages = [("raw", "raw source"), ("model_correct", "Qwen-correct"), ("discovery", "used for discovery"), ("heldout", "held-out capital")]
for ax, unit in zip(axes, ["facts", "prompts"]):
    for s, (key, lab) in enumerate(stages):
        vals = [int(counts[(g, "ALL")][f"{key}_{unit}"]) for g in GROUPS]
        bars = ax.bar(np.arange(4) + (s - 1.5) * 0.2, vals, 0.2, label=lab, color=CVD[s], hatch=HATCH[s], edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(np.arange(4), GROUPS)
    ax.set_xlabel("answer group (expected answer of the prompts)")
    ax.set_ylabel(f"number of distinct {unit}")
    ax.set_title(f"Distinct {unit} per answer group")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "sample_counts.png"), dpi=150)
plt.close(fig)

pooled = {(o["group_a"], o["group_b"]): o for o in overlap if o["cutoff"] == MAIN and o["scope"] == "all_layers_pooled"}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1.1]))
J = np.full((4, 4), np.nan)
ax = axes[0]
for i, a in enumerate(GROUPS):
    for j, b in enumerate(GROUPS):
        o = pooled.get((a, b)) or pooled.get((b, a))
        if i == j:
            J[i, j] = 1.0
            ax.text(j, i, f"{len(S[a])}\nfeatures", ha="center", va="center", color="black", fontsize=7.5)
        elif o and o.get("note") == "":
            J[i, j] = o["jaccard"] if o["jaccard"] != "N/A" else np.nan
            ax.text(j, i, f"J={J[i, j]:.3f}\n{o['shared']} shared", ha="center", va="center", fontsize=7.5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
        else:
            ax.text(j, i, "N/A", ha="center", va="center")
off = J.copy()
np.fill_diagonal(off, np.nan)
im = ax.imshow(off, cmap="cividis", vmin=0, vmax=max(0.05, np.nanmax(off)))
ax.set_xticks(range(4), GROUPS)
ax.set_yticks(range(4), GROUPS)
ax.set_title(f"Jaccard overlap of top-{MAIN}-per-layer sets\n(all 28 layers pooled, exact layer+index IDs)")
fig.colorbar(im, ax=ax, fraction=0.046, label="Jaccard overlap")
ax = axes[1]
for i, (a, b) in enumerate(HEADLINE):
    o = pooled[(a, b)]
    left = 0
    for v, lab, c, h in [(o["unique_a"], f"only in city set", CVD[0], ""), (o["shared"], "shared", CVD[4], "xx"),
                         (o["unique_b"], f"only in country set", CVD[1], "//")]:
        ax.barh(i, v, left=left, color=c, hatch=h, edgecolor="black", linewidth=0.5, label=lab if i == 0 else None)
        if v:
            ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
        left += v
ax.set_yticks(range(2), [f"{a} (city) vs\n{b} (country)" for a, b in HEADLINE])
ax.invert_yaxis()
ax.set_xlabel("number of selected feature IDs (union of the two sets)")
ax.set_title("Headline pairs: exact counts")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "feature_overlap.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.6))
for i, (a, b) in enumerate(HEADLINE):
    per = {o["scope"]: o for o in overlap if o["cutoff"] == MAIN and o["scope"] != "all_layers_pooled" and (o["group_a"], o["group_b"]) == (a, b)}
    ax.plot(range(N_LAYERS), [per[L]["shared"] for L in range(N_LAYERS)], color=CVD[i], linestyle=["-", "--"][i],
            marker=["o", "s"][i], label=f"{a} and {b}: shared")
ax.set_xlabel("layer (0 = first)")
ax.set_ylabel(f"shared feature IDs (of at most {MAIN} per set)")
ax.set_ylim(-0.5, MAIN + 0.5)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "overlap_by_layer.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 3.8))
other = {"Tokyo": ["Japan", "Berlin", "Germany"], "Japan": ["Tokyo", "Germany", "Berlin"],
         "Berlin": ["Germany", "Tokyo", "Japan"], "Germany": ["Berlin", "Japan", "Tokyo"]}
labels = ["same-country partner set", "same-type set of the other country", "other-type set of the other country"]
for k in range(3):
    xs, ys, txt = [], [], []
    for i, a in enumerate(GROUPS):
        s_ = next(s for s in strata if s["threshold"] == STRONG and s["set_a"] == a and s["set_b"] == other[a][k])
        xs.append(i + (k - 1) * 0.27)
        ys.append(s_["strong_also_in_b"] / s_["n_strong"] if s_["n_strong"] else 0)
        txt.append(f"{s_['strong_also_in_b']}/{s_['n_strong']}\n{other[a][k]}")
    bars = ax.bar(xs, ys, 0.27, color=CVD[k], hatch=HATCH[k], edgecolor="black", linewidth=0.5, label=labels[k])
    for x, y, t in zip(xs, ys, txt):
        ax.text(x, y + 0.02, t, ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(range(4), [f"{g} set" for g in GROUPS])
ax.set_ylabel(f"fraction of strong features (score >= {STRONG})\nalso selected for the other entity")
ax.set_ylim(0, 1.1)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "overlap_by_score.png"), dpi=150)
plt.close(fig)

flat = [(g, r) for g in GROUPS for r in cands[g]]
if flat:
    R = np.array([[r[f"rate_{h}"] for h in GROUPS] + [r["background_rate"]] for _, r in flat])
    E = np.array([[r["early_value_Japan"], r["early_value_Germany"]] for _, r in flat])
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.36 * len(flat) + 1.6), gridspec_kw=dict(width_ratios=[5, 2]), sharey=True)
    im = axes[0].imshow(R, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(R):
        axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v < 0.6 else "black")
    axes[0].set_xticks(range(5), GROUPS + ["matched\nbackground"])
    axes[0].set_yticks(range(len(flat)), [f"{g} set: L{r['layer']} #{r['feature']}" for g, r in flat])
    axes[0].set_title("Fraction of facts where the feature is active\n(last prompt token, answer group on x)")
    fig.colorbar(im, ax=axes[0], fraction=0.03, label="active rate")
    im = axes[1].imshow(E, cmap="cividis", vmin=0, vmax=max(1e-6, E.max()), aspect="auto")
    for (i, j), v in np.ndenumerate(E):
        axes[1].text(j, i, f"{v:.2f}" if v > 0 else "0", ha="center", va="center", fontsize=8,
                     color="white" if v < 0.6 * max(1e-6, E.max()) else "black")
    axes[1].set_xticks(range(2), ['"Japan"\ntoken', '"Germany"\ntoken'])
    axes[1].set_title('Activation before "is" in\n"The capital of <country>"')
    fig.colorbar(im, ax=axes[1], fraction=0.08, label="activation value")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "candidate_activations.png"), dpi=150)
    plt.close(fig)
