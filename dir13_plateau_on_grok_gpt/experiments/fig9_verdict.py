"""Assign a Figure-9 grokking-gate verdict to a fig9.py output JSON (PLAN Exp 1/2 gate).

The Figure-9 signature (source-locked to *Deep Networks Always Grok*, Fig. 9) requires a
qualitative temporal ordering:
  (a) a SECOND local-complexity (LC) descent -- LC falls, turns back UP, then falls again --
      whose onset precedes the clean test-accuracy peak, and
  (b) delayed adversarial robustness (eps=0.03 PGD accuracy) that rises during or after that
      second descent.

Detection (rewritten 2026-08-02 after operator feedback #4; the previous version used
`np.argmin(lc)`, i.e. the GLOBAL minimum, which for these runs is the LAST checkpoint -- the end
of the second descent -- so no rise could ever be found after it and every run was scored as
having no second descent).  The corrected rule finds the structure in order:

  1. FIRST significant local minimum  : the earliest interior index i with lc[i] < lc[i-1] whose
     following rise exceeds `tol` (5% of the curve's total range).  Smaller dips are wiggles.
  2. The SUBSEQUENT local maximum j   : the highest point of that rise, searched from i up to the
     first checkpoint that falls back below lc[i].  steps[j] is the second descent's ONSET.
  3. A SUSTAINED second descent       : after j the curve falls by more than `tol`, ends within
     `tol` of its post-j minimum (it does not rebound), and finishes BELOW the first local
     minimum lc[i] (a genuine second descent, not an undo of the rise).

Preregistered ordering checks (PLAN "Grokking-paper signature"; both were missing before):
  - `onset_before_acc_peak`      : steps[j] < step of the maximum clean test accuracy.
  - `robustness_during_or_after` : eps=0.03 PGD accuracy rises by >= 0.05 from its value at the
    onset j to its maximum at or after j, and reaches >= 0.05 in absolute terms.  We also report
    `adv_onset_step`, the first checkpoint from which adv accuracy stays >= 0.05 for the rest of
    the run (sustained onset, so a one-checkpoint transient does not count), and whether it lies
    at or after the second-descent onset.

Verdict:
  PASS            : sustained second descent AND onset_before_acc_peak AND robustness_during_or_after.
  FAIL            : the run reached its planned horizon with valid measurements but the ordering
                    is absent.
  NOT ESTABLISHED : coverage too short to decide -- no second descent, LC still in its first
                    monotone descent at the last checkpoint, and robustness never emerged.

Usage: fig9_verdict.py <fig9.json>
"""
import sys, json
import numpy as np

TOL_FRAC = 0.05   # LC wiggles smaller than this fraction of the curve's range are ignored
ADV_FLOOR = 0.05  # eps=0.03 PGD accuracy counted as "robust" above this


def find_second_descent(lc, tol):
    """Return (i, j, diag) for the first significant local min i, its following local max j.

    j is the second descent's onset. (None, None, diag) if no sustained second descent exists.
    """
    n = len(lc)
    diag = {"n_candidate_minima": 0, "largest_rejected_rise": 0.0}
    for i in range(1, n - 1):
        if lc[i] >= lc[i - 1]:
            continue                      # not descending into i -> not a local minimum
        diag["n_candidate_minima"] += 1
        # the rise that follows i, up to the first checkpoint that falls back below lc[i]
        m = n
        for k in range(i + 1, n):
            if lc[k] < lc[i]:
                m = k
                break
        j = i + int(np.argmax(lc[i:m]))
        rise = float(lc[j] - lc[i])
        if rise <= tol or j >= n - 1:
            diag["largest_rejected_rise"] = max(diag["largest_rejected_rise"], rise)
            continue                      # wiggle, or the rise ends the run
        post = lc[j + 1:]
        fall = float(lc[j] - post.min())
        sustained = fall > tol and post[-1] <= post.min() + tol and lc[-1] < lc[i]
        if sustained:
            diag.update({"first_min_rise": round(rise, 2), "second_descent_fall": round(fall, 2)})
            return i, j, diag
        diag["largest_rejected_rise"] = max(diag["largest_rejected_rise"], rise)
    return None, None, diag


def sustained_adv_onset(adv, floor=ADV_FLOOR):
    """First index from which adv accuracy stays >= floor for every later checkpoint."""
    for k in range(len(adv)):
        if np.all(adv[k:] >= floor):
            return k
    return None


def analyze(recs):
    recs = sorted(recs, key=lambda r: r["step"])
    steps = np.array([r["step"] for r in recs])
    acc = np.array([r["clean_acc"] for r in recs])
    adv = np.array([r["adv_acc"] for r in recs])
    lc = np.array([r["lc"]["test"]["mean"] for r in recs])

    span = float(lc.max() - lc.min())
    tol = TOL_FRAC * span
    acc_peak_i = int(np.argmax(acc))
    i, j, diag = find_second_descent(lc, tol)
    second_descent = j is not None

    if second_descent:
        onset_before_acc_peak = bool(steps[j] < steps[acc_peak_i])
        adv_at_onset = float(adv[j])
        adv_max_after = float(adv[j:].max())
        adv_rise = adv_max_after - adv_at_onset
        robustness = bool(adv_rise >= ADV_FLOOR and adv_max_after >= ADV_FLOOR)
    else:
        onset_before_acc_peak = False
        adv_at_onset = adv_max_after = adv_rise = float("nan")
        robustness = bool(adv.max() >= ADV_FLOOR)

    k_adv = sustained_adv_onset(adv)
    lc_still_descending = bool(lc[-1] <= lc.min() + tol and not second_descent)
    acc_still_climbing = bool(acc_peak_i >= len(acc) - 2)

    if second_descent and onset_before_acc_peak and robustness:
        verdict = "PASS"
    elif not second_descent and lc_still_descending and adv.max() < ADV_FLOOR:
        verdict = "NOT ESTABLISHED"
    else:
        verdict = "FAIL"

    out = {
        "verdict": verdict,
        "n_ckpts": len(recs),
        "step_range": [int(steps[0]), int(steps[-1])],
        "lc_tol_5pct_of_range": round(tol, 2),
        "final_clean_acc": round(float(acc[-1]), 4),
        "peak_clean_acc": round(float(acc.max()), 4),
        "clean_acc_peak_step": int(steps[acc_peak_i]),
        "final_adv_acc": round(float(adv[-1]), 4),
        "max_adv_acc": round(float(adv.max()), 4),
        "lc_first_value": round(float(lc[0]), 1),
        "lc_final_value": round(float(lc[-1]), 1),
        "second_lc_descent": second_descent,
        "first_local_min_step": int(steps[i]) if second_descent else None,
        "first_local_min_value": round(float(lc[i]), 1) if second_descent else None,
        "local_max_step": int(steps[j]) if second_descent else None,
        "local_max_value": round(float(lc[j]), 1) if second_descent else None,
        "second_descent_onset_step": int(steps[j]) if second_descent else None,
        "onset_before_clean_acc_peak": onset_before_acc_peak,
        "adv_acc_at_onset": None if second_descent is False else round(adv_at_onset, 4),
        "max_adv_acc_at_or_after_onset": None if second_descent is False else round(adv_max_after, 4),
        "adv_rise_from_onset": None if second_descent is False else round(adv_rise, 4),
        "robustness_rises_during_or_after_descent": robustness,
        "sustained_adv_onset_step": None if k_adv is None else int(steps[k_adv]),
        "adv_onset_at_or_after_second_descent": bool(second_descent and k_adv is not None and k_adv >= j),
        "lc_still_in_first_descent_at_end": lc_still_descending,
        "clean_acc_still_climbing": acc_still_climbing,
        "detector_diagnostics": diag,
    }
    return out


if __name__ == "__main__":
    print(json.dumps(analyze(json.load(open(sys.argv[1]))), indent=1))
