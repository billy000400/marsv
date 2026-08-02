"""Score the narrow (n_embd 192, nothing frozen) run and merge it into the frozen-assay outputs.

S14's question: the five frozen runs left width monotone in *trainable depth* (0.351 at 12 trainable
blocks -> 0.726 at one usable block), but every one of them cut trainable parameters at the same time,
so depth and capacity are confounded. This run holds depth fixed at all 12 blocks trainable and cuts
capacity instead, by narrowing the model from n_embd 240 to 192 (8.38M -> 5.38M parameters, i.e. the
same trainable-parameter budget as frozen_early's 5.60M). The prediction on record:

  depth account            -> median w near the reference's 0.351
  parameter-count account  -> median w near frozen_early's 0.471

frozen_assay.py recomputes all eight conditions, which does not fit the remaining wall clock, so this
scores the one new condition with the *same* functions, the same 150 pairs and the same seed, then
merges the row into results/frozen_assay_summary.json and results/frozen_assay_raw.npz. The reference
rows it is compared against are the ones already in those files, untouched.
"""
import os, sys, json, itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import self_test
from allpairs_sweep import load_vocab, CONTEXT, N_T
from mlp_gain_probe import N_PAIRS, SEED
from mlp_block_scan import sweep, endpoint_stats, rho, partial_rho
from frozen_assay import load_ckpt, curves, depth_widths, CKPT_ROOT, RES

TAG = "narrow192"


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    self_test()

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[c] for c in CONTEXT + ch], dtype=np.int64) for ch in chars]
    all_pairs = list(itertools.combinations(range(len(chars)), 2))
    rng = np.random.default_rng(SEED)
    pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS, replace=False)]

    blob = json.load(open(os.path.join(RES, "frozen_assay_summary.json")))
    S = blob["summary"]
    old = dict(np.load(os.path.join(RES, "frozen_assay_raw.npz")))

    for which, fn in (("matched", "ckpt_matched.pt"), ("last", "ckpt_last.pt")):
        path = os.path.join(CKPT_ROOT, f"checkpoints_{TAG}", fn)
        if not os.path.exists(path):
            print(f"[skip] {path} missing", flush=True)
            continue
        name = f"{TAG}_{which}"
        if name in S["conditions"] and f"{name}_w" in old:
            print(f"[skip] {name} already scored", flush=True)
            continue
        model, ck = load_ckpt(path, device)
        r = sweep(model, seqs, pairs, ts, device, *endpoint_stats(model, stoi, seqs, device))
        row = {"kind": "narrow", "step": int(ck["step"]), "frozen_blocks": [],
               "n_embd": int(ck["cfg"]["n_embd"]), "val_acc": ck.get("val_acc"), "ckpt": path,
               "median_w": round(float(np.nanmedian(r["w"])), 4),
               "mean_w": round(float(np.nanmean(r["w"])), 4),
               "iqr_w": [round(float(np.nanpercentile(r["w"], 25)), 4),
                         round(float(np.nanpercentile(r["w"], 75)), 4)],
               "strict_frac": round(float(r["strict"].mean()), 4),
               "median_t_star": round(float(np.nanmedian(r["tstar"])), 4),
               "frac_endpoints_differ": round(float(r["differ"].mean()), 4),
               "median_n_argmax": float(np.median(r["n_argmax"])),
               "median_abs_tstar_minus_tflip":
                   round(float(np.nanmedian(np.abs(r["tstar"] - r["tflip"]))), 4),
               "median_max_p": round(float(np.median(r["maxp"])), 4),
               "rho_w_vs_max_p": rho(r["w"], r["maxp"]),
               "partial_rho_w_vs_max_p_given_sep": partial_rho(r["w"], r["maxp"], r["sep"])}
        for ref in ("ref_trained", "ref_matched_step", "ref_init"):
            b = old[f"{ref}_w"]
            dw = r["w"] - b
            row[f"median_paired_dw_vs_{ref}"] = round(float(np.nanmedian(dw)), 4)
            row[f"frac_w_increased_vs_{ref}"] = round(float(np.nanmean(dw > 0)), 4)
        row["depth_median_w"] = depth_widths(model, seqs, pairs, ts, device)
        S["conditions"][name] = row
        for k, v in r.items():
            old[f"{name}_{k}"] = v
        old[f"{name}_curves"] = curves(model, seqs, pairs, ts, device)
        print(json.dumps({name: row}, indent=2), flush=True)
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(RES, "frozen_assay_summary.json"), "w") as f:
        json.dump(blob, f, indent=1)
    np.savez_compressed(os.path.join(RES, "frozen_assay_raw.npz"), **old)
    print("merged into frozen_assay_summary.json / frozen_assay_raw.npz", flush=True)


if __name__ == "__main__":
    main()
