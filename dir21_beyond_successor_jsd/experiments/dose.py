"""Does the block-0 MLP COMPUTE the width trait, or is it just the only early component big enough
to reach the lethal regime?

The component ablation found exactly one spike in 102 early components: mean-ablating the block-0 MLP
erases the per-token width ordering. But it is also the only early component whose removal the model
feels (0.451 bits vs <= 0.007 elsewhere), and the displacement ladder already showed that ANY
disturbance of that size flattens the ordering. This runs a dose-response to separate the two.

Arm A (mlp):     blend the block-0 MLP's final-position output toward its mean with weight alpha.
Arm B (control): add a fixed random direction to the same residual stream at the same position,
                 with its scale binary-searched so the model's output moves the SAME number of bits.

At each dose we re-measure w_hat_u for 12 tokens against 6 anchors and record rho(before, after).
Separated curves in bits space => the block-0 MLP carries the trait. Coincident curves => disturbance
as such kills it. Writes results/dose.json (partial-safe).
"""
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from ablate import measure
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint, jsd_bits
from common import D18, RESULTS

N_ANCHOR = 6
FRAME = FRAMES[0]
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0]
SEED = 0

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


class Dose:
    """Blend the block-0 MLP output toward its mean (alpha), or add c * r to it (control)."""

    def __init__(self, model):
        self.mode = None          # None | "mlp" | "ctrl"
        self.alpha = 0.0
        self.c = 0.0
        self.mean = None
        self.r = None
        self.capture = False
        self.rec = None
        self.h = model.gpt_neox.layers[0].mlp.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.capture:
            self.rec = out[:, -1, :].detach().float().mean(0)
        if self.mode is None:
            return out
        out = out.clone()
        if self.mode == "mlp":
            v = (1 - self.alpha) * out[:, -1, :].float() + self.alpha * self.mean
        else:
            v = out[:, -1, :].float() + self.c * self.r
        out[:, -1, :] = v.to(out.dtype)
        return out


@torch.inference_mode()
def bits_only(model, patcher, ids_all, base_lp):
    """Mean output movement in bits over the 12 tokens, one batched forward, no interpolation."""
    patcher.bank = None
    lp = model(ids_all).logits[:, -1, :].float().log_softmax(-1)
    return float(jsd_bits(base_lp, lp).mean())


def main():
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    used = set(ids_by_str.values())
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    anchors = pool[:: max(1, len(pool) // N_ANCHOR)][:N_ANCHOR]

    toks12 = list(json.load(open(f"{RESULTS}/mode_split.json"))["tokens"].keys())
    ids_by_tok = {s: ids_by_str[s] for s in toks12}

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    pre = tok(FRAME, return_tensors="pt").input_ids.cuda()
    patcher = Patcher(model)
    dose = Dose(model)

    ids_all = torch.cat([pre.repeat(len(ids_by_tok), 1),
                         torch.tensor(list(ids_by_tok.values()),
                                      device=pre.device).unsqueeze(1)], 1)
    ids_mean = torch.cat([pre.repeat(len(ids_by_tok) + len(anchors), 1),
                          torch.tensor(list(ids_by_tok.values()) + anchors,
                                       device=pre.device).unsqueeze(1)], 1)
    dose.capture = True
    with torch.inference_mode():
        patcher.bank = None
        model(ids_mean)
    dose.capture = False
    dose.mean = dose.rec.clone()

    g = torch.Generator(device="cuda").manual_seed(SEED)
    r = torch.randn(model.config.hidden_size, generator=g, device="cuda")
    dose.r = r / r.norm()

    t0 = time.time()
    base_w, base_z = measure(model, patcher, ids_by_tok, anchors, pre)
    base = np.array([base_w[s] for s in toks12])
    base_lp = torch.stack([base_z[s] for s in toks12]).log_softmax(-1)
    print(f"baseline mean {base.mean():.3f} sd {base.std(ddof=1):.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    def score(tag, dosev, bits):
        v = np.array([w[s] for s in toks12])
        ok = np.isfinite(v) & np.isfinite(base)
        rho = float(spearmanr(base[ok], v[ok]).statistic) if ok.sum() > 3 else float("nan")
        row = dict(arm=tag, dose=float(dosev), bits=float(bits), rho=rho,
                   mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)),
                   w=[float(x) for x in v], n_valid=int(ok.sum()))
        print(f"{tag} dose {dosev:.3f}: bits {bits:.3f} rho {rho:+.2f} "
              f"mean {row['mean']:.3f} sd {row['sd']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        return row

    rows = []
    for a in ALPHAS:
        dose.mode, dose.alpha = "mlp", a
        bits = bits_only(model, patcher, ids_all, base_lp)
        w, _ = measure(model, patcher, ids_by_tok, anchors, pre)
        dose.mode = None
        rows.append(score("mlp", a, bits))
        rows[-1]["target_bits"] = None

        # control: binary-search c so the random direction moves the output the same bits
        dose.mode = "ctrl"
        lo, hi = 0.0, 1.0
        for _ in range(40):
            dose.c = hi
            if bits_only(model, patcher, ids_all, base_lp) >= bits:
                break
            lo, hi = hi, hi * 2
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            dose.c = mid
            if bits_only(model, patcher, ids_all, base_lp) < bits:
                lo = mid
            else:
                hi = mid
        dose.c = 0.5 * (lo + hi)
        cb = bits_only(model, patcher, ids_all, base_lp)
        w, _ = measure(model, patcher, ids_by_tok, anchors, pre)
        c_used = dose.c
        dose.mode = None
        rows.append(score("ctrl", c_used, cb))
        rows[-1]["target_bits"] = float(bits)

        json.dump(dict(model=MODEL, revision=REVISION, frame=FRAME, seed=SEED,
                       anchors=[tok.convert_ids_to_tokens(a_) for a_ in anchors],
                       tokens=toks12, base_w=[float(x) for x in base],
                       base_sd=float(np.nanstd(base, ddof=1)), rows=rows),
                  open(f"{RESULTS}/dose.json", "w"), indent=1)
    print("wrote results/dose.json")


if __name__ == "__main__":
    main()
