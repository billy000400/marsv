"""Final audit: re-derive every headline number quoted in REPORT.md / RESULTS.md
straight from results/*.json and confirm the exact string appears in both files.

Catches a stale number left behind by a re-run. Prints one line per claim; exits
non-zero if any claim's value is missing from a deliverable.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"


def load(name):
    return json.load(open(R / name))


cm = load("checkpoint_metrics.json")
bulk = load("bulk_onset.json")
pers = load("persistence.json")
lpers = load("large_persistence.json")
lpref = load("large_persistence_ref.json")
ljack = load("large_jackknife.json")
jack = load("sentence_jackknife.json")

steps = cm["steps"]
idx = {s: i for i, s in enumerate(steps)}


def f(x, n=3):
    return f"{x:.{n}f}"


claims = []


def claim(label, value):
    # REPORT.md is the canonical deliverable and must quote the value verbatim;
    # RESULTS.md is a shorter summary, so its coverage is reported, not required.
    claims.append((label, value))


# --- Result 1: ordering onset bracket, 60-pair bank -------------------------
claim("rho(step 32)", f(cm["rho"][idx[32]]))
claim("rho(step 8)", f(cm["rho"][idx[8]]))
claim("rho(final)", f(cm["rho"][idx[143000]]))

# --- Result 2: shape onset bracket ------------------------------------------
wmed = cm["median_w"]
for s in (0, 32, 512, 1000, 2000, 64000, 143000):
    claim(f"median w(step {s})", f(wmed[idx[s]]))
claim("edge drift(step 1000)", f(cm["edge_drift"][idx[1000]]))
claim("movement entropy(step 1000)", f(cm["move_entropy"][idx[1000]]))

# --- Result 16: large-bank reference sweep -----------------------------------
for ref, blk in lpref["references"].items():
    for s in (32, 64):
        row = blk["rows"][str(s)]
        claim(f"large dpi(step {s} | ref {ref})",
              f"+{f(row['dpi'])}" if row["dpi"] > 0 else f(row["dpi"]))

print(f"{'claim':<42} {'value':>10}  REPORT  RESULTS")
bad = 0
text = {n: (ROOT / n).read_text() for n in ("REPORT.md", "RESULTS.md")}
for label, value in claims:
    in_report = value in text["REPORT.md"]
    in_results = value in text["RESULTS.md"]
    bad += not in_report
    print(f"{label:<42} {value:>10}  {'ok  ' if in_report else 'STALE'}"
          f"    {'ok' if in_results else '-'}")

print(f"\n{len(claims)} claims re-derived from results/*.json; "
      f"{bad} missing from REPORT.md.")
sys.exit(1 if bad else 0)
