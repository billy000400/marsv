"""Block-0 rerun: write the table sections of RESULTS.md (R1 checks, R3 five-group tables, R5 other models)
to results/block0_resid_post/tables.md from the validation, meta and summary JSON files."""
import json
import os

from common import RESULTS
from gen_common import NAMES

KEYS = ["gpt2-large", "gpt2-xl", "pythia-1.4b", "llama-3.1-8b", "qwen3-8b-base"]
NAMES = dict(NAMES, **{"gpt2-large": "GPT-2 Large"})
S = json.load(open(os.path.join(RESULTS, "summary_all.json")))
out = []
f3 = lambda x: f"{x:.3f}"

out.append("<!-- R1 -->\n| Model | Hook (block-0 output) | Tokens | Injected vs r(t), max abs | Prefix vs unpatched A, max abs | "
           "Readout at t=0 vs unpatched A (mid / final) | Readout at t=1 vs unpatched B (mid / final) |\n|---|---|---|---|---|---|---|")
for k in KEYS:
    p = os.path.join(RESULTS, "validate", f"{k}.json")
    if not os.path.exists(p):
        continue
    v = json.load(open(p))
    a, b = v["readout_t0_vs_unpatched_A_max_rel_l2"], v["readout_t1_vs_unpatched_B_max_rel_l2"]
    out.append(f"| {NAMES[k]} | `{v['hook_module']}` | {v['n_tokens']} | {v['injected_vs_r_t_max_abs']:.0e} | "
               f"{v['prefix_vs_unpatched_A_max_abs']:.1e} | {a[0]:.1e} / {a[1]:.1e} | {b[0]:.1e} / {b[1]:.1e} |")

out.append("\n<!-- R1b -->\n| Model | Tokens per forward | Readout difference, max relative (mid / final) | "
           "Width difference, median (mid / final) | Width difference, max (mid / final) |\n|---|---|---|---|---|")
for k in KEYS[1:]:
    p = os.path.join(RESULTS, "gen", k, "meta.json")
    if not os.path.exists(p):
        continue
    m = json.load(open(p))
    r, d, x = m["cache_check_max_rel_diff"], m["cache_check_abs_dW_median"], m["cache_check_abs_dW_max"]
    out.append(f"| {NAMES[k]} | {m['tok_per_batch']} x 101 | {r[0]:.1e} / {r[1]:.1e} | {d[0]:.1e} / {d[1]:.1e} | "
               f"{x[0]:.1e} / {x[1]:.1e} |")

out.append("\n<!-- R2 -->\n| Model | Site | Tokens | D: min / 5% / median / 95% / max | W_mid: 5% / 25% / median / 75% / 95% | "
           "W_final: 5% / 25% / median / 75% / 95% | Final narrower | Flagged mid / final / either |\n|---|---|---|---|---|---|---|---|")
for k in KEYS:
    for site, lab in (("block0_resid_post", "block 0"), ("embed", "embedding")):
        s = S[k][site]
        if not s:
            continue
        out.append(f"| {NAMES[k]} | {lab} | {s['n']:,} | " + " / ".join(f"{x:.2f}" for x in s["D_pct_0_5_50_95_100"]) + " | "
                   + " / ".join(map(f3, s["W_mid_pct_5_25_50_75_95"])) + " | " + " / ".join(map(f3, s["W_final_pct_5_25_50_75_95"]))
                   + f" | {s['frac_final_narrower']:.1%} | {s['flagged_mid']:,} / {s['flagged_final']:,} / "
                   f"{s['flagged_either']:,} ({s['flagged_either'] / s['n']:.1%}) |")

out.append("\n<!-- R3 -->\n| Model | Site | Fifth | D range | median W_mid (25th–75th) | median W_final (25th–75th) |\n|---|---|---|---|---|---|")
for k in KEYS:
    for site, lab in (("block0_resid_post", "block 0"), ("embed", "embedding")):
        s = S[k][site]
        if not s:
            continue
        for i, b in enumerate(s["D_quintiles"]):
            out.append(f"| {NAMES[k]} | {lab} | {i + 1} | {b['D_range'][0]:.2f}–{b['D_range'][1]:.2f} | "
                       f"{b['W_mid'][1]:.3f} ({b['W_mid'][0]:.3f}–{b['W_mid'][2]:.3f}) | "
                       f"{b['W_final'][1]:.3f} ({b['W_final'][0]:.3f}–{b['W_final'][2]:.3f}) |")

for k in KEYS[1:]:
    p = os.path.join(RESULTS, "gen", f"{k}_summary.json")
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    out.append(f"\n<!-- R5 {k} -->\nhist peak {s['hist_peak_W'][0]:.3f}-{s['hist_peak_W'][1]:.3f} (n={s['hist_peak_n']}); "
               f"W_final p90 {s['W_final_p90']:.3f}, p99 {s['W_final_p99']:.3f}; W_final(' large') {s['W_final_large']:.3f}; "
               f"zoom D {s['D_p5_p95'][0]:.1f}-{s['D_p5_p95'][1]:.1f}, W_mid {s['zoom_W_range_mid'][0]:.3f}-{s['zoom_W_range_mid'][1]:.3f}, "
               f"W_final {s['zoom_W_range_final'][0]:.3f}-{s['zoom_W_range_final'][1]:.3f}\n")
    out.append("| W_final band | tokens | lowest-D examples |\n|---|---|---|")
    for b in s["bands"]:
        ex = ", ".join("`" + e.replace("`", "'").replace("|", "¦").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + "`"
                       for e in b["lowest_D_examples"][:6])
        out.append(f"| {b['band'][0]:.2f}–{min(b['band'][1], 1):.2f} | {b['n']:,} | {ex} |")
    out.append("\nwidest 12: " + ", ".join(f"`{w}` ({x:.2f})" for w, x in s["widest"][:12]))
open(os.path.join(RESULTS, "tables.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
