"""D6 Phase 6b CIs — paired bootstrap on the manifold-repair KL deltas.

The claim-4 upgrade rests on point estimates over N=300 prompts: the fractional kNN manifold step
(t=0.25) has lower in-context KL(clean||x) than (a) the corrupted start and (b) a matched-size RANDOM
move. Here we test whether those gaps are statistically real with a PAIRED bootstrap over the 300
prompts (each method's KL is measured on the SAME prompt, so we resample prompt indices and recompute
the mean delta). We also test the FULL-projection overshoot (t=1.00 vs its matched random) — the
verdict there is that the manifold step must stay in a trust region, i.e. random should BEAT the full
step. Reads per-prompt KL from manifold_repair_perprompt_kl.npz (written by manifold_repair.py).

Delta convention: delta = KL(reference) - KL(treatment). delta>0 => treatment is BETTER (lower KL).
"sig" one-sided => the 95% CI excludes 0 in the reported direction.
"""
import os, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
B = 5000
rng = np.random.default_rng(0)

# (label, reference_method, treatment_method, expected_sign_note)
COMPARISONS = [
    ("t=0.10 vs random-matched", "random_move(t=0.25-matched)", "knn_project(t=0.10)", "scaled"),
    ("t=0.25 vs random-matched", "random_move(t=0.25-matched)", "knn_project(t=0.25)", "direction test"),
    ("t=0.50 vs random-matched", "random_move(t=0.50-matched)", "knn_project(t=0.50)", "direction test"),
    ("t=1.00 vs random-matched", "random_move(knn1-matched)",  "knn_project(t=1.00)", "overshoot: random should win"),
    ("t=0.25 vs corrupted-start", "corrupted(start)",           "knn_project(t=0.25)", "repair beats doing nothing"),
]
# t=0.10 reference: its own matched random isn't stored; use the t=0.25-matched random as an upper bound
# on random KL at a SMALLER move (random KL grows with move), so this is conservative for the kNN step.


def main():
    d = np.load(os.path.join(RES, "manifold_repair_perprompt_kl.npz"))
    n = len(d["corrupted(start)"])
    idx = rng.integers(0, n, size=(B, n))  # shared resample indices per bootstrap replicate
    rows = []
    for label, ref, trt, note in COMPARISONS:
        delta = d[ref] - d[trt]                       # per-prompt, delta>0 => treatment better
        point = float(delta.mean())
        boot = delta[idx].mean(1)                     # [B] paired bootstrap of the mean delta
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        p_pos = float((boot > 0).mean())
        rows.append({
            "comparison": label,
            "ref_KL": round(float(d[ref].mean()), 4),
            "trt_KL": round(float(d[trt].mean()), 4),
            "delta_KL": round(point, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "frac_boot_delta_gt0": round(p_pos, 4),
            "sig_treatment_better": bool(lo > 0),
            "note": note,
        })
    with open(os.path.join(RES, "manifold_repair_ci.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    with open(os.path.join(RES, "manifold_repair_ci_summary.json"), "w") as f:
        json.dump({"n_prompts": int(n), "B": B, "method": "paired bootstrap of mean KL delta; "
                   "delta = KL(ref) - KL(trt); >0 => treatment lower KL", "rows": rows}, f, indent=2)

    print(f"paired bootstrap B={B}, N={n} prompts. delta = KL(ref) - KL(trt); >0 => treatment BETTER")
    print(f"{'comparison':30s}{'refKL':>8}{'trtKL':>8}{'dKL':>9}{'95% CI':>20}{'sig':>6}")
    for r in rows:
        ci = "[%+.3f,%+.3f]" % (r["ci_lo"], r["ci_hi"])
        sig = "YES" if r["sig_treatment_better"] else "no"
        print(f"{r['comparison']:30s}{r['ref_KL']:>8}{r['trt_KL']:>8}{r['delta_KL']:>9}{ci:>20}{sig:>6}")


if __name__ == "__main__":
    main()
