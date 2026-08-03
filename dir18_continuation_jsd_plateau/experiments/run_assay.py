"""S3/S4 driver: run the frozen pair bank through the plateau assay at one checkpoint.

Usage:  python3 run_assay.py <revision> [--model M] [--calib] [--tag T]
Saves raw 50-point d(t) curves plus per-pair scalar summaries. Never dumps full logits.
"""
import argparse
import json
import os

import numpy as np
import torch

from assay import GRID, run_pair, width
from common import DATA, MODEL, RESULTS, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("revision")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--calib", action="store_true", help="run only the frozen 15-pair calibration set")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--manifest", default="pair_manifest.json")
    a = ap.parse_args()
    tag = a.tag or (("calib_" if a.calib else "") + a.revision + ("" if a.model == MODEL else "_410m"))

    man = json.load(open(os.path.join(RESULTS, a.manifest)))
    valid = torch.tensor(np.load(os.path.join(DATA, "reliability_bank.npz"))["valid"],
                         device="cuda", dtype=torch.long)
    tok, m = load(a.revision, a.model)
    ctx_ids = {c: tok(c)["input_ids"] for c in man["contexts"]}

    idxs = man["calibration_idx"] if a.calib else list(range(len(man["pairs"])))
    curves = np.full((len(idxs), len(man["contexts"]), len(GRID)), np.nan, dtype=np.float32)
    rows = []
    for n, pi in enumerate(idxs):
        p = man["pairs"][pi]
        rec = dict(pair_idx=pi, bin=p["bin"], jsd_A=p["jsd_A"], jsd_B=p["jsd_B"],
                   a_str=p["a_str"], b_str=p["b_str"], w_ctx=[], out_jsd=[], cos0=[], dist0=[])
        for ci, c in enumerate(man["contexts"]):
            ia = torch.tensor([ctx_ids[c] + [p["a"]]], device="cuda")
            ib = torch.tensor([ctx_ids[c] + [p["b_tok"]]], device="cuda")
            assert ia.shape == ib.shape and (ia[0, :-1] == ib[0, :-1]).all(), "pair must differ only in final token"
            r = run_pair(m, ia, ib, layer=0, valid=valid)
            curves[n, ci] = r["d"]
            rec["w_ctx"].append(r["w"])
            rec["out_jsd"].append(r["out_jsd"])
            rec["cos0"].append(r["cos_block0"])
            rec["dist0"].append(r["dist"])
            rec.setdefault("err", []).append(max(r["err0"], r["err1"]))
            rec["fallback"] = bool(r["fallback"])
        ws = np.array(rec["w_ctx"], dtype=float)
        rec["n_valid_ctx"] = int(np.isfinite(ws).sum())
        rec["w"] = float(np.nanmedian(ws)) if rec["n_valid_ctx"] else float("nan")
        rec["out_jsd_med"] = float(np.median(rec["out_jsd"]))
        rows.append(rec)
        if n % 10 == 0:
            print(f"{n+1}/{len(idxs)} {p['a_str']!r}/{p['b_str']!r} w={rec['w']:.3f}", flush=True)

    ws = np.array([r["w"] for r in rows], dtype=float)
    out = dict(revision=a.revision, model=a.model, calib=a.calib, n_pairs=len(rows),
               grid=[float(x) for x in GRID],
               max_endpoint_relerr=float(max(max(r["err"]) for r in rows)),
               valid_curve_rate=float(np.isfinite(ws).mean()),
               iqr_w=float(np.nanpercentile(ws, 75) - np.nanpercentile(ws, 25)),
               median_w=float(np.nanmedian(ws)), rows=rows)
    json.dump(out, open(os.path.join(RESULTS, f"assay_{tag}.json"), "w"), indent=2)
    np.save(os.path.join(RESULTS, f"curves_{tag}.npy"), curves)
    print(f"valid-curve rate {out['valid_curve_rate']:.3f}  median w {out['median_w']:.3f}  "
          f"IQR(w) {out['iqr_w']:.3f}  max endpoint rel-err {out['max_endpoint_relerr']:.2e}")


if __name__ == "__main__":
    main()
