"""Run the frozen 60-pair bank through the plateau assay at ONE pythia-1.4b-deduped checkpoint.

Usage:  python3 run_assay.py <revision> [--manifest M] [--tag T]

Identical assay definitions to dir18 (post-block-0 patch, norm-rescaled SLERP, 50 positions, valid
target IDs, d(t), w, validity rules). This direction adds, per checkpoint:
  * the neighbouring-position output-movement profile m_j (PLAN measurement 5),
  * next-token loss on one frozen sample of 256 held-out corpus rows (timing context only),
  * the published learning rate at that step (timing context only).
Saves raw 50-point d(t) curves and 49-point movement profiles; never dumps full-vocabulary logits.
"""
import argparse
import json
import os

import numpy as np
import torch

from assay import GRID, run_pair
from common import DATA, MODEL, RESULTS, load

LOSS_ROWS, LOSS_LEN = 256, 512   # frozen held-out sample: last 256 rows of split B, first 512 tokens
SEQ, N_SPLIT_ROWS = 2049, 500_000


def pythia_lr(step, peak=2e-4, warmup=1430, total=143000):
    """Published pythia-1.4b schedule: linear warmup then cosine decay to 0.1 x peak."""
    if step < warmup:
        return peak * step / warmup
    f = (step - warmup) / (total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + np.cos(np.pi * f))


@torch.inference_mode()
def heldout_loss(model):
    """Mean next-token cross-entropy (nats) on the frozen 256-row / 512-token held-out sample."""
    arr = np.memmap(os.path.join(DATA, "splitB.bin"), dtype=np.uint16, mode="r",
                    shape=(N_SPLIT_ROWS, SEQ))
    rows = np.asarray(arr[N_SPLIT_ROWS - LOSS_ROWS:, :LOSS_LEN], dtype=np.int64)
    tot, n = 0.0, 0
    for i in range(0, LOSS_ROWS, 4):
        ids = torch.tensor(rows[i:i + 4], device="cuda")
        lg = model(ids).logits[:, :-1].float()
        ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                               ids[:, 1:].reshape(-1), reduction="sum")
        tot += float(ce)
        n += ids[:, 1:].numel()
        del lg
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("revision")
    ap.add_argument("--manifest", default="pair_manifest_top256.json")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or a.revision

    man = json.load(open(os.path.join(RESULTS, a.manifest)))
    valid = torch.tensor(np.load(os.path.join(DATA, "valid_target_ids.npy")),
                         device="cuda", dtype=torch.long)
    tok, m = load(a.revision, MODEL)
    ctx_ids = {c: tok(c)["input_ids"] for c in man["contexts"]}

    n_p, n_c = len(man["pairs"]), len(man["contexts"])
    curves = np.full((n_p, n_c, len(GRID)), np.nan, dtype=np.float32)
    moves = np.full((n_p, n_c, len(GRID) - 1), np.nan, dtype=np.float32)
    rows = []
    for pi, p in enumerate(man["pairs"]):
        rec = dict(pair_idx=pi, bin=p["bin"], jsd_A=p["jsd_A"], jsd_B=p["jsd_B"],
                   a_str=p["a_str"], b_str=p["b_str"], w_ctx=[], out_jsd=[], cos0=[], dist0=[],
                   err=[])
        for ci, c in enumerate(man["contexts"]):
            ia = torch.tensor([ctx_ids[c] + [p["a"]]], device="cuda")
            ib = torch.tensor([ctx_ids[c] + [p["b_tok"]]], device="cuda")
            assert ia.shape == ib.shape and (ia[0, :-1] == ib[0, :-1]).all(), "pair must differ only in final token"
            r = run_pair(m, ia, ib, layer=0, valid=valid)
            curves[pi, ci] = r["d"]
            moves[pi, ci] = r["mv"]
            rec["w_ctx"].append(r["w"])
            rec["out_jsd"].append(r["out_jsd"])
            rec["cos0"].append(r["cos_block0"])
            rec["dist0"].append(r["dist"])
            rec["err"].append(max(r["err0"], r["err1"]))
            rec["fallback"] = bool(r["fallback"])
        ws = np.array(rec["w_ctx"], dtype=float)
        rec["n_valid_ctx"] = int(np.isfinite(ws).sum())
        rec["w"] = float(np.nanmedian(ws)) if rec["n_valid_ctx"] else float("nan")
        rec["out_jsd_med"] = float(np.median(rec["out_jsd"]))
        rows.append(rec)
        if pi % 20 == 0:
            print(f"{pi+1}/{n_p} {p['a_str']!r}/{p['b_str']!r} w={rec['w']:.3f}", flush=True)

    step = int(a.revision.replace("step", ""))
    loss = heldout_loss(m)
    ws = np.array([r["w"] for r in rows], dtype=float)
    out = dict(revision=a.revision, step=step, model=MODEL, n_pairs=len(rows),
               grid=[float(x) for x in GRID], lr=float(pythia_lr(step)), heldout_loss=float(loss),
               max_endpoint_relerr=float(max(max(r["err"]) for r in rows)),
               valid_curve_rate=float(np.isfinite(ws).mean()),
               iqr_w=float(np.nanpercentile(ws, 75) - np.nanpercentile(ws, 25)),
               median_w=float(np.nanmedian(ws)), rows=rows)
    json.dump(out, open(os.path.join(RESULTS, f"assay_{tag}.json"), "w"), indent=2)
    np.save(os.path.join(RESULTS, f"curves_{tag}.npy"), curves)
    np.save(os.path.join(RESULTS, f"moves_{tag}.npy"), moves)
    print(f"valid-curve rate {out['valid_curve_rate']:.3f}  median w {out['median_w']:.3f}  "
          f"IQR(w) {out['iqr_w']:.3f}  loss {loss:.4f}  lr {out['lr']:.3e}  "
          f"max endpoint rel-err {out['max_endpoint_relerr']:.2e}")


if __name__ == "__main__":
    main()
