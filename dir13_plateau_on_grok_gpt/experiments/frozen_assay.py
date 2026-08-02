"""Does a network whose blocks 1-4 never trained still build sharp plateaus? (PLAN 5.5 prediction)

train_frozen.py trained two networks that are identical to the reference fresh character run except
that four blocks are held at their step-0 initialization:

  frozen_early  blocks 1-4 frozen  -- the group the MLP-gain probe and per-block scan showed
                                      causally sets the transition sharpness;
  frozen_late   blocks 8-11 frozen -- specificity control, same number of blocks at a depth those
                                      interventions showed contributes almost nothing;
  frozen_deep   blocks 1-7 frozen  -- the successor prediction: after frozen_early merely relocated
                                      the sharpening to blocks 5-8, freeze those too and leave only
                                      blocks 8-11 trainable. Either the sharpening relocates again
                                      (width well below 0.80, with the drop appearing between
                                      injection blocks 8 and 11) or it needs several trainable
                                      blocks below the readout and the paths stay straight.
  frozen_mirror blocks 5-11 frozen -- mirror image of frozen_deep: the same number of blocks frozen
                                      but the trainable capacity at the *bottom* of the stack. Tests
                                      whether width is set by the count of trainable blocks (expect
                                      ~0.558 again, drop between injection blocks 0 and 4, nothing
                                      above) or by their depth (expect a markedly different width).

Here we run the frozen assay on both at *matched validation accuracy* (the first checkpoint whose
val accuracy reaches the reference run's final 0.5502) and at their last checkpoint, against three
reference conditions measured on the same 150 pairs: the reference run at step 0 (untrained), at
step 2500 (the same optimization step at which the frozen runs reach matched accuracy, so that
"sharpness at matched accuracy" is not confounded with "sharpness this early in training"), and at
step 30000 (fully trained). The prediction under test: frozen_early stays near the untrained width
(~0.80) while frozen_late sharpens like the reference (~0.35).

Everything about the assay is the frozen code path (matthew_assay.run_pair, block-0 interpolation,
50 t-values, context "The house was "), and the per-pair statistics are exactly the ones
mlp_block_scan.py records, so the widths are directly comparable to the ablation results.

On the trained reference and the two final frozen models we also re-run Experiment 5's depth control
(inject at blocks 0 / 4 / 8 instead of only block 0). If a frozen network still sharpens, this says
whether it does so at the same depth or has moved the computation elsewhere.

Raw -> results/frozen_assay_raw.npz, stats -> results/frozen_assay_summary.json.
"""
import os, sys, json, time, itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig
from matthew_assay import run_pair, is_plateau, self_test
from allpairs_sweep import load_vocab, load_model, CONTEXT, N_T, FINAL_STEP, INIT_STEP
from mlp_gain_probe import BLOCK, N_PAIRS, SEED
from mlp_block_scan import sweep, endpoint_stats, rho, partial_rho

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
N_CURVES = 20  # pairs whose raw d(t) curves are stored for the figure
CKPT_ROOT = "/tmp/dir13_frozen"  # train_frozen.py --ckpt_root (checkpoints are gitignored scratch)
REF_MATCHED_STEP = 2500  # reference-run checkpoint nearest the frozen runs' matched-accuracy step
# injection depths for the "where is the sharpness made now?" control. 10 and 11 were added for
# frozen_deep, whose only trainable blocks are 8-11: the prediction is about whether the width drop
# appears between injection blocks 8 and 11. Block 11 leaves only ln_f + unembedding downstream.
# 2 was added for frozen_mirror, whose only trainable blocks are 0-4, so its drop should sit there.
DEPTH_BLOCKS = [0, 2, 4, 8, 10, 11]
DEPTH_CONDS = ("ref_trained", "frozen_early_last", "frozen_late_last", "frozen_deep_last",
               "frozen_mirror_last")


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = GPT(GPTConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def curves(model, seqs, pairs, ts, device):
    return np.stack([run_pair(model, seqs[a], seqs[b], BLOCK, ts, device)["d_logit"]
                     for a, b in pairs[:N_CURVES]]).astype(np.float32)


def depth_widths(model, seqs, pairs, ts, device):
    """Median width when the interpolation is injected at each block in DEPTH_BLOCKS.

    Experiment 5's depth control on the reference model: injecting later leaves fewer blocks to
    sharpen the path, so w rises with the injection block and locates where the sharpness is made.
    Re-running it on a frozen network says whether that computation moved."""
    out = {}
    for b in DEPTH_BLOCKS:
        w = [is_plateau(ts, run_pair(model, seqs[i], seqs[j], b, ts, device)["d_logit"])[1]
             for i, j in pairs]
        out[str(b)] = round(float(np.nanmedian(w)), 4)
    return out


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

    conds = []
    for tag, blocks in (("frozen_early", [1, 2, 3, 4]), ("frozen_late", [8, 9, 10, 11]),
                        ("frozen_deep", [1, 2, 3, 4, 5, 6, 7]),
                        ("frozen_mirror", [5, 6, 7, 8, 9, 10, 11])):
        for which, fn in (("matched", "ckpt_matched.pt"), ("last", "ckpt_last.pt")):
            p = os.path.join(CKPT_ROOT, f"checkpoints_{tag}", fn)
            if os.path.exists(p):
                conds.append((f"{tag}_{which}", p, blocks))
            else:
                print(f"[skip] {p} missing", flush=True)

    out, res, raw = {}, {}, {}
    t0 = time.time()

    # reference conditions on the same pairs: trained and untrained reference run.
    # (train_meta_grok_char.json is truncated -- that run crashed while dumping it -- so the
    # reference final val accuracy comes from its history file, which is intact.)
    hist = json.load(open(os.path.join(RES, "train_hist_grok_char.json")))
    ref_acc = hist["val_acc"][-1]
    ref_2500 = hist["val_acc"][hist["step"].index(REF_MATCHED_STEP)]
    for name, step, acc in (("ref_trained", FINAL_STEP, ref_acc),
                            ("ref_matched_step", REF_MATCHED_STEP, ref_2500),
                            ("ref_init", INIT_STEP, None)):
        model = load_model(step, device)
        r = sweep(model, seqs, pairs, ts, device, *endpoint_stats(model, stoi, seqs, device))
        res[name] = r
        raw[name + "_curves"] = curves(model, seqs, pairs, ts, device)
        out[name] = {"kind": "reference", "step": step, "frozen_blocks": [], "val_acc": acc}
        if name in DEPTH_CONDS:
            out[name]["depth_median_w"] = depth_widths(model, seqs, pairs, ts, device)
        del model
        torch.cuda.empty_cache()

    for name, path, blocks in conds:
        model, ck = load_ckpt(path, device)
        r = sweep(model, seqs, pairs, ts, device, *endpoint_stats(model, stoi, seqs, device))
        res[name] = r
        raw[name + "_curves"] = curves(model, seqs, pairs, ts, device)
        out[name] = {"kind": "frozen", "step": int(ck["step"]),
                     "frozen_blocks": blocks,
                     "val_acc": ck.get("val_acc"), "ckpt": path}
        if name in DEPTH_CONDS:
            out[name]["depth_median_w"] = depth_widths(model, seqs, pairs, ts, device)
        del model
        torch.cuda.empty_cache()

    for name, r in res.items():
        out[name].update({
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
            "partial_rho_w_vs_max_p_given_sep": partial_rho(r["w"], r["maxp"], r["sep"])})

    # paired shifts against the two references (same 150 pairs in every condition)
    for ref in ("ref_trained", "ref_matched_step", "ref_init"):
        b = res[ref]["w"]
        for name, r in res.items():
            if name == ref:
                continue
            dw = r["w"] - b
            out[name][f"median_paired_dw_vs_{ref}"] = round(float(np.nanmedian(dw)), 4)
            out[name][f"frac_w_increased_vs_{ref}"] = round(float(np.nanmean(dw > 0)), 4)

    summary = {"context": CONTEXT, "block": BLOCK, "n_t": N_T, "n_pairs": N_PAIRS, "seed": SEED,
               "ref_final_val_acc": ref_acc, "conditions": out}
    print(json.dumps(summary, indent=2), flush=True)
    with open(os.path.join(RES, "frozen_assay_summary.json"), "w") as f:
        json.dump({"summary": summary, "pairs": [[chars[a], chars[b]] for a, b in pairs],
                   "curve_pairs": [[chars[a], chars[b]] for a, b in pairs[:N_CURVES]]}, f, indent=1)
    np.savez_compressed(os.path.join(RES, "frozen_assay_raw.npz"), ts=ts,
                        **{f"{n}_{k}": v for n, r in res.items() for k, v in r.items()}, **raw)
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
