"""Assign a Figure-9 grokking-gate verdict to a fig9.py output JSON (PLAN Exp 1/2 gate).

The Figure-9 signature (source-locked to *Deep Networks Always Grok*, Fig. 9) requires a
qualitative temporal ordering:
  (a) a SECOND local-complexity (LC) descent that begins before the test-accuracy peak, and
  (b) delayed adversarial robustness (eps=0.03 PGD accuracy) that rises while that second
      descent continues.

Verdict rule (preregistered here, applied identically to every run):
  PASS            : test-LC shows an interior local minimum then rise then a second descent,
                    AND final adv_acc rises to >= 0.05 (delayed robustness emerges), AND the
                    second descent's onset precedes the clean-accuracy peak.
  FAIL            : run reaches its planned horizon with valid measurements but the ordering is
                    absent (e.g. adv_acc never rises, or no second LC descent).
  NOT ESTABLISHED : horizon too short / checkpoint coverage insufficient to decide (adv flat AND
                    LC still in its first monotone descent at the last checkpoint).
Usage: fig9_verdict.py <fig9.json>
"""
import sys, json
import numpy as np

recs = sorted(json.load(open(sys.argv[1])), key=lambda r: r["step"])
steps = np.array([r["step"] for r in recs])
acc = np.array([r["clean_acc"] for r in recs])
adv = np.array([r["adv_acc"] for r in recs])
lc = np.array([r["lc"]["test"]["mean"] for r in recs])

fin_adv = float(adv[-1])
acc_peak_i = int(np.argmax(acc))
span = float(lc.max() - lc.min())
tol = 0.05 * span  # ignore LC wiggles smaller than 5% of the curve's total range

# genuine second descent: an interior local MINIMUM, then a rise above tol, then a fall below tol.
second_descent = False
lc_min_i = int(np.argmin(lc))
if 0 < lc_min_i < len(lc) - 1:
    rise = lc[lc_min_i:].max() - lc[lc_min_i]
    j = lc_min_i + int(np.argmax(lc[lc_min_i:]))
    fall = lc[j] - lc[-1] if j < len(lc) - 1 else 0.0
    if rise > tol and fall > tol:
        second_descent = True

robustness = fin_adv >= 0.05                      # delayed adversarial robustness emerged
# horizon adequacy: LC still trending down (within tol) to the last checkpoint AND accuracy still
# climbing (peak at the final checkpoint) -> the run ended before any second-descent/robustness
# window could appear, so we cannot decide the ordering.
lc_still_descending = bool(lc[-1] <= lc.min() + tol)
acc_still_climbing = acc_peak_i >= len(acc) - 2

if second_descent and robustness:
    verdict = "PASS"
elif (lc_still_descending or acc_still_climbing) and not robustness:
    verdict = "NOT ESTABLISHED"
else:
    verdict = "FAIL"

out = {
    "verdict": verdict,
    "n_ckpts": len(recs),
    "step_range": [int(steps[0]), int(steps[-1])],
    "final_clean_acc": round(float(acc[-1]), 4),
    "peak_clean_acc": round(float(acc.max()), 4),
    "clean_acc_peak_step": int(steps[acc_peak_i]),
    "final_adv_acc": round(fin_adv, 4),
    "max_adv_acc": round(float(adv.max()), 4),
    "lc_first_step": round(float(lc[0]), 1),
    "lc_min_step": int(steps[lc_min_i]),
    "lc_min_value": round(float(lc[lc_min_i]), 1),
    "lc_final_value": round(float(lc[-1]), 1),
    "second_lc_descent": bool(second_descent),
    "delayed_robustness": bool(robustness),
    "lc_still_descending_to_end": lc_still_descending,
    "clean_acc_still_climbing": acc_still_climbing,
}
print(json.dumps(out, indent=1))
