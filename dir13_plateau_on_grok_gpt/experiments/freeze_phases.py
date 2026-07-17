"""Freeze the 6 Matthew-assay checkpoint phases from Figure-9 curves (PLAN Exp 3), BEFORE any
plateau curve is inspected. Rule (preregistered here):
  P1 earliest usable checkpoint (step 0);
  P2 before the first test-LC peak (closest ckpt to peak_step/4, must be < peak);
  P3 near the test-LC peak (argmax of test LC);
  P4 start of the second LC descent (first ckpt after the peak where test LC has fallen by
     >= 25% of (peak - final) — i.e. the descent has clearly begun);
  P5 near adversarial-robustness onset (first ckpt with adv_acc >= 50% of final adv_acc,
     restricted to ckpts after P3; falls back to midpoint of P4..P6 if adv never rises);
  P6 final checkpoint.
Deduplicates while preserving order; if the curves are degenerate (no interior peak), falls back
to log-spaced picks and says so. Usage: freeze_phases.py <fig9.json> <out.json>
"""
import sys, json
import numpy as np

recs = json.load(open(sys.argv[1]))
steps = np.array([r["step"] for r in recs])
lc_te = np.array([r["lc"]["test"]["mean"] for r in recs])
adv = np.array([r["adv_acc"] for r in recs])

note = ""
ipk = int(np.argmax(lc_te))
if ipk == 0 or ipk == len(steps) - 1:
    note = "degenerate LC curve (peak at boundary); log-spaced fallback"
    idx = sorted(set(np.linspace(0, len(steps) - 1, 6).round().astype(int)))
else:
    p1 = 0
    p3 = ipk
    tgt = steps[ipk] / 4
    pre = [i for i in range(ipk) if steps[i] <= tgt]
    p2 = pre[-1] if pre else max(0, ipk - 2)
    peak, fin = lc_te[ipk], lc_te[-1]
    thresh = peak - 0.25 * (peak - fin)
    post = [i for i in range(ipk + 1, len(steps)) if lc_te[i] <= thresh]
    p4 = post[0] if post else min(ipk + 1, len(steps) - 1)
    p6 = len(steps) - 1
    fin_adv = adv[-1]
    cand = [i for i in range(p3, len(steps)) if adv[i] >= 0.5 * fin_adv and fin_adv > 0.02]
    p5 = cand[0] if cand else (p4 + p6) // 2
    idx = [p1, p2, p3, p4, p5, p6]
    seen, dedup = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i); dedup.append(i)
    idx = sorted(dedup)

sel = [int(steps[i]) for i in idx]
out = {"phases": sel, "note": note,
       "labels": ["P1 init", "P2 pre-LC-peak", "P3 LC peak", "P4 2nd descent start",
                  "P5 robustness onset", "P6 final"][:len(sel)]}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out))
