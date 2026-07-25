"""Operator feedback #3 — interpolate from the comma to EVERY other character; is there a plateau?

Same code path as the Matthew-faithful char controls (`run_matthew_ckpts.py`): shared context
"The house was ", endpoint A = context + ',', endpoint B = context + c for each of the other 64
characters in the character vocabulary. 50 evenly spaced interpolation values including both
endpoints, `slerp_rescale` (spherical direction + linear norm), patch ONLY the final position of
the block-L residual stream, run the remaining blocks, score with Matthew's relative distance
d(t) = ||x(t)-x_A|| / (||x(t)-x_A|| + ||x(t)-x_B||) in final-logit space.

Runs on the fresh character GPT:
  * interpolation block 0 at the 6 frozen training phases (emergence across training);
  * every interpolation block 0..11 at the final checkpoint (depth control).

Saves raw d(t) curves + transition widths; plotting lives in plot_comma_sweep.py.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig
from matthew_assay import run_pair, transition_width, is_plateau, self_test

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CKPT_DIR = os.path.join(RES, "checkpoints_grok_char")
CONTEXT = "The house was "
N_T = 50
FINAL_STEP = 30000


def load_vocab():
    """Character vocabulary of the tinyshakespeare corpus, restored from a pilot checkpoint (the
    corpus file itself is no longer on this pod). Both runs built stoi as sorted(set(text))."""
    ck = torch.load(os.path.join(RES, "checkpoints", "ckpt_00000.pt"),
                    map_location="cpu", weights_only=False)
    return ck["stoi"]


def load_model(step, device):
    ck = torch.load(os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pt"),
                    map_location=device, weights_only=False)
    m = GPT(GPTConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    self_test()

    stoi = load_vocab()
    assert "," in stoi, "comma not in vocabulary"
    targets = [c for c in sorted(stoi, key=lambda c: stoi[c]) if c != ","]
    print(f"{len(targets)} target characters", flush=True)

    ts = np.linspace(0.0, 1.0, N_T)
    seq_A = np.array([stoi[c] for c in CONTEXT + ","], dtype=np.int64)
    steps = json.load(open(os.path.join(RES, "frozen_phases_char.json")))["phases"]

    raw = {"ts": ts}
    summary = {"context": CONTEXT, "n_t": N_T, "steps": steps, "targets": targets,
               "by_step": {}, "by_block_final": {}, "checks": {}}

    # (1) interpolation block 0 at each frozen training phase
    for step in steps:
        model = load_model(step, device)
        rows = {}
        for c in targets:
            seq_B = np.array([stoi[x] for x in CONTEXT + c], dtype=np.int64)
            r = run_pair(model, seq_A, seq_B, 0, ts, device)
            ok, w, t_lo, t_hi, dev = is_plateau(ts, r["d_logit"])
            rows[c] = {"w": None if not np.isfinite(w) else round(float(w), 4),
                       "plateau": bool(ok), "t_lo": round(float(t_lo), 4),
                       "t_hi": round(float(t_hi), 4), "iso_dev": round(float(dev), 4)}
            raw[f"s{step}|L0|{stoi[c]}"] = r["d_logit"]
            if step == FINAL_STEP:
                for l, d in r["d_layer"].items():          # downstream resid_post curves
                    raw[f"s{step}|L0|{stoi[c]}|rec{l}"] = d
                summary["checks"][c] = {k: round(v, 8) for k, v in r["endpoint_err"].items()}
                summary["checks"][c]["prefix_err"] = round(r["prefix_err"], 8)
                summary["checks"][c]["d0"] = round(r["d0"], 6)
                summary["checks"][c]["d1"] = round(r["d1"], 6)
        ws = [v["w"] for v in rows.values() if v["w"] is not None]
        n_pl = sum(v["plateau"] for v in rows.values())
        print(f"step {step:6d}: median w={np.median(ws):.3f}  plateau {n_pl}/{len(rows)}", flush=True)
        summary["by_step"][str(step)] = rows
        del model
        torch.cuda.empty_cache()

    # (2) full interpolation-block sweep at the final checkpoint
    model = load_model(FINAL_STEP, device)
    n_layer = len(model.blocks)
    for L in range(1, n_layer):
        rows = {}
        for c in targets:
            seq_B = np.array([stoi[x] for x in CONTEXT + c], dtype=np.int64)
            r = run_pair(model, seq_A, seq_B, L, ts, device)
            w, _, _, dev = transition_width(ts, r["d_logit"])
            rows[c] = {"w": None if not np.isfinite(w) else round(float(w), 4),
                       "iso_dev": round(float(dev), 4)}
            raw[f"s{FINAL_STEP}|L{L}|{stoi[c]}"] = r["d_logit"]
        ws = [v["w"] for v in rows.values() if v["w"] is not None]
        print(f"final ckpt interp block {L:2d}: median w={np.median(ws):.3f}", flush=True)
        summary["by_block_final"][str(L)] = rows
    summary["by_block_final"]["0"] = {c: {"w": v["w"], "iso_dev": v["iso_dev"]}
                                      for c, v in summary["by_step"][str(FINAL_STEP)].items()}

    # (3) how plausible each character is after the context, and how far apart the two endpoint
    #     logit vectors are — to test whether either explains the spread of transition widths
    with torch.no_grad():
        ctx = torch.tensor([[stoi[c] for c in CONTEXT]], device=device)
        p_next = torch.softmax(model(ctx)[0][0, -1], dim=-1).cpu().numpy()
        lg_A = model(torch.from_numpy(seq_A[None]).to(device))[0][0, -1]
        sep = {}
        for c in targets:
            seq_B = np.array([stoi[x] for x in CONTEXT + c], dtype=np.int64)
            lg_B = model(torch.from_numpy(seq_B[None]).to(device))[0][0, -1]
            sep[c] = {"p_next": round(float(p_next[stoi[c]]), 10),
                      "logit_sep": round(float(torch.linalg.norm(lg_A - lg_B)), 4)}
    summary["endpoints_final"] = {"p_next_comma": round(float(p_next[stoi[","]]), 10), "per_char": sep}

    np.savez_compressed(os.path.join(RES, "comma_sweep_raw.npz"), **raw)
    json.dump(summary, open(os.path.join(RES, "comma_sweep_summary.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
