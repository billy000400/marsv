"""Layer sweep driver shared by both experiments.

For every condition (a pair of prompts differing only in the last token) and every interpolation
layer, run the assay and store the d(t) curve at every downstream recording site plus the final
logits. Results go to results/<tag>.npz (curves) and results/<tag>_summary.json (frozen summaries
+ validity checks).
"""
import json
import os

import numpy as np

from assay import SITES, Assay, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def sweep(model, conditions, ts, layers=None):
    """conditions: list of dicts with keys label, ids_A, ids_B (lists of ints).
    Returns curves {f"{label}|L{L}|{site}": d(t)} and checks {label: {...}}."""
    layers = list(range(model.n_layer)) if layers is None else layers
    curves, checks = {}, {}
    for cond in conditions:
        a = np.array(cond["ids_A"], dtype=np.int64)
        b = np.array(cond["ids_B"], dtype=np.int64)
        ck = {"endpoint_err": {}, "d0": {}, "d1": {}, "cos_AB": {}, "norms": {}}
        for L in layers:
            out = model.run_pair(a, b, L, ts)
            for k, v in out["d"].items():
                curves[f"{cond['label']}|L{L}|{k}"] = v.astype(np.float32)
            ck["endpoint_err"][L] = out["endpoint_err"]
            ck["d0"][L] = float(out["d"]["logits"][0])
            ck["d1"][L] = float(out["d"]["logits"][-1])
            ck["cos_AB"][L] = out["cos_AB"]
            ck["norms"][L] = out["norms"]
        checks[cond["label"]] = ck
        print(f"  done {cond['label']}: {len(layers)} interpolation layers", flush=True)
    return curves, checks


def summarize_all(curves, ts, **kw):
    """Frozen per-curve summaries keyed exactly like `curves`."""
    return {k: summarize(ts, v, **kw) for k, v in curves.items()}


def save(tag, curves, checks, summaries, extra=None):
    os.makedirs(RES, exist_ok=True)
    np.savez_compressed(os.path.join(RES, f"{tag}.npz"), **curves)
    payload = {"checks": checks, "summaries": summaries}
    if extra:
        payload.update(extra)
    with open(os.path.join(RES, f"{tag}_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {RES}/{tag}.npz ({len(curves)} curves) and {tag}_summary.json")


def load(tag):
    z = np.load(os.path.join(RES, f"{tag}.npz"))
    curves = {k: z[k] for k in z.files}
    with open(os.path.join(RES, f"{tag}_summary.json")) as f:
        meta = json.load(f)
    return curves, meta


def worst_checks(checks):
    """Collapse per-layer validity checks to the worst value over layers, per condition."""
    out = {}
    for label, ck in checks.items():
        keys = set().union(*[set(v) for v in ck["endpoint_err"].values()])
        out[label] = {k: max(abs(v[k]) for v in ck["endpoint_err"].values()) for k in keys}
        out[label]["max|d0|"] = max(abs(v) for v in ck["d0"].values())
        out[label]["max|1-d1|"] = max(abs(1 - v) for v in ck["d1"].values())
    return out


__all__ = ["sweep", "summarize_all", "save", "load", "worst_checks", "SITES", "Assay", "RES"]


def load_meta(tag):
    """Summaries/checks only — skips decompressing the (large) curve archive."""
    with open(os.path.join(RES, f"{tag}_summary.json")) as f:
        return json.load(f)
