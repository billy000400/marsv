"""Dose-response with the random control matched to the MLP dose SEPARATELY FOR EACH TOKEN.

Operator feedback (human_feedback.txt): the first dose-response matched the control to the block-0
MLP dose on the MEAN output JSD over the 12 tokens, while the conclusion rests on the ORDERING of
those tokens' widths. If the MLP moves some tokens much more than others, a mean-matched control can
be far too weak on exactly the tokens that carry the ranking.

This rerun matches per prompt. At each dose alpha:
  1. one batched forward gives the MLP arm's output movement b_p for every one of the 18 endpoint
     prompts (12 tokens + 6 anchors);
  2. for each random seed, a vectorised binary search finds a SEPARATE scale c_p per prompt so the
     random direction moves that prompt's output by exactly b_p bits;
  3. widths are re-measured with each prompt carrying its own c_p.
The old mean-matched control is rerun alongside purely to quantify how badly it was mismatched.

Writes results/dose2.json (partial-safe).
"""
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import run_pair
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint, jsd_bits
from common import D18, RESULTS

N_ANCHOR = 6
FRAME = FRAMES[0]
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0]
SEEDS = [0, 1, 2]

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


class Dose:
    """Blend the block-0 MLP output toward its mean (alpha), or add c * r to it (control).

    `c` may be a scalar (one prompt per forward) or a vector with one entry per batch row.
    """

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
            c = self.c if torch.is_tensor(self.c) else torch.tensor(float(self.c), device=out.device)
            v = out[:, -1, :].float() + c.reshape(-1, 1) * self.r
        out[:, -1, :] = v.to(out.dtype)
        return out


@torch.inference_mode()
def bits_per_prompt(model, patcher, ids_all, base_lp):
    """Output movement in bits for every prompt in the batch, one forward."""
    patcher.bank = None
    lp = model(ids_all).logits[:, -1, :].float().log_softmax(-1)
    return jsd_bits(base_lp, lp).float()


def search_scales(model, patcher, dose, ids_all, base_lp, target):
    """Per-prompt scale c_p of the random direction that reproduces target[p] bits."""
    dose.mode = "ctrl"
    n = len(target)
    hi = torch.ones(n, device=target.device)
    for _ in range(40):
        dose.c = hi
        b = bits_per_prompt(model, patcher, ids_all, base_lp)
        low = b < target
        if not low.any():
            break
        hi = torch.where(low, hi * 2, hi)
    lo = torch.zeros_like(hi)
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        dose.c = mid
        b = bits_per_prompt(model, patcher, ids_all, base_lp)
        under = b < target
        lo = torch.where(under, mid, lo)
        hi = torch.where(under, hi, mid)
    dose.c = 0.5 * (lo + hi)
    achieved = bits_per_prompt(model, patcher, ids_all, base_lp)
    c = dose.c.clone()
    dose.mode, dose.c = None, 0.0
    return c, achieved


def search_global(model, patcher, dose, ids_all, base_lp, target_mean):
    """The OLD control: one scale for every prompt, matched on the mean bits (for the diagnostic)."""
    dose.mode = "ctrl"
    hi = 1.0
    for _ in range(40):
        dose.c = hi
        if float(bits_per_prompt(model, patcher, ids_all, base_lp).mean()) >= target_mean:
            break
        hi *= 2
    lo = 0.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        dose.c = mid
        if float(bits_per_prompt(model, patcher, ids_all, base_lp).mean()) < target_mean:
            lo = mid
        else:
            hi = mid
    dose.c = 0.5 * (lo + hi)
    achieved = bits_per_prompt(model, patcher, ids_all, base_lp)
    c = float(dose.c)
    dose.mode, dose.c = None, 0.0
    return c, achieved


def measure(model, patcher, dose, ids_by_tok, anchor_ids, pre, mode, alpha=0.0, cmap=None):
    """w_hat_u against the 6 anchors, with each prompt perturbed by its own scale.

    The perturbation only enters through the endpoint states and endpoint logits: during the
    interpolation the patcher overwrites block 0's final-position output, so the hook is inert there.
    """
    def set_dose(key):
        if mode is None:
            dose.mode = None
        elif mode == "mlp":
            dose.mode, dose.alpha = "mlp", alpha
        else:
            dose.mode, dose.c = "ctrl", float(cmap[key])

    anc = []
    for a in anchor_ids:
        set_dose(("anchor", a))
        anc.append(endpoint(model, patcher,
                            torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1)))
        dose.mode = None
    w = {}
    for s, i in ids_by_tok.items():
        set_dose(s)
        ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
        x, z = endpoint(model, patcher, ids)
        dose.mode = None
        w[s] = float(np.nanmedian([run_pair(model, patcher, ids, x, z, xb, zb)[0]
                                   for xb, zb in anc]))
    return w


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

    keys = list(toks12) + [("anchor", a) for a in anchors]
    ids_all = torch.cat([pre.repeat(len(keys), 1),
                         torch.tensor(list(ids_by_tok.values()) + anchors,
                                      device=pre.device).unsqueeze(1)], 1)

    dose.capture = True
    with torch.inference_mode():
        patcher.bank = None
        model(ids_all)
    dose.capture = False
    dose.mean = dose.rec.clone()

    with torch.inference_mode():
        patcher.bank = None
        base_lp = model(ids_all).logits[:, -1, :].float().log_softmax(-1)

    t0 = time.time()
    base_w = measure(model, patcher, dose, ids_by_tok, anchors, pre, None)
    base = np.array([base_w[s] for s in toks12])
    print(f"baseline mean {base.mean():.3f} sd {base.std(ddof=1):.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    def score(tag, w, bits_tok):
        v = np.array([w[s] for s in toks12])
        ok = np.isfinite(v) & np.isfinite(base)
        rho = float(spearmanr(base[ok], v[ok]).statistic) if ok.sum() > 3 else float("nan")
        row = dict(arm=tag, rho=rho, mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)),
                   w=[float(x) for x in v], bits_tok=[float(b) for b in bits_tok],
                   bits=float(np.mean(bits_tok)), n_valid=int(ok.sum()))
        print(f"  {tag}: bits {row['bits']:.4f} rho {rho:+.2f} mean {row['mean']:.3f} "
              f"sd {row['sd']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        return row

    rows = []
    for a in ALPHAS:
        print(f"alpha {a}", flush=True)
        dose.mode, dose.alpha = "mlp", a
        b_mlp = bits_per_prompt(model, patcher, ids_all, base_lp)
        dose.mode = None
        w_mlp = measure(model, patcher, dose, ids_by_tok, anchors, pre, "mlp", alpha=a)
        r = score(f"mlp a={a}", w_mlp, b_mlp[:len(toks12)].tolist())
        r.update(alpha=a, seed=None, match="none")
        rows.append(r)

        for seed in SEEDS:
            g = torch.Generator(device="cuda").manual_seed(seed)
            rr = torch.randn(model.config.hidden_size, generator=g, device="cuda")
            dose.r = rr / rr.norm()

            c, ach = search_scales(model, patcher, dose, ids_all, base_lp, b_mlp)
            cmap = {k: float(c[j]) for j, k in enumerate(keys)}
            w_c = measure(model, patcher, dose, ids_by_tok, anchors, pre, "ctrl", cmap=cmap)
            r = score(f"ctrl/token seed={seed}", w_c, ach[:len(toks12)].tolist())
            r.update(alpha=a, seed=seed, match="per_token",
                     c=[float(x) for x in c], bits_target=[float(x) for x in b_mlp])
            rows.append(r)

            if seed == SEEDS[0]:                       # diagnostic: the old mean-matched control
                cg, achg = search_global(model, patcher, dose, ids_all, base_lp,
                                         float(b_mlp.mean()))
                cmapg = {k: cg for k in keys}
                w_g = measure(model, patcher, dose, ids_by_tok, anchors, pre, "ctrl", cmap=cmapg)
                r = score(f"ctrl/mean seed={seed}", w_g, achg[:len(toks12)].tolist())
                r.update(alpha=a, seed=seed, match="mean", c=cg,
                         bits_target=[float(x) for x in b_mlp])
                rows.append(r)

        json.dump(dict(model=MODEL, revision=REVISION, frame=FRAME, seeds=SEEDS,
                       anchors=[tok.convert_ids_to_tokens(a_) for a_ in anchors],
                       tokens=toks12, base_w=[float(x) for x in base],
                       base_sd=float(np.nanstd(base, ddof=1)), rows=rows),
                  open(f"{RESULTS}/dose2.json", "w"), indent=1)

    # paired per-token test: |dw| under the MLP dose vs its per-token-matched control
    print("\npaired |dw| MLP vs per-token-matched control", flush=True)
    for a in ALPHAS:
        m = [r for r in rows if r["arm"].startswith("mlp") and r["alpha"] == a][0]
        cs = [r for r in rows if r["match"] == "per_token" and r["alpha"] == a]
        dm = np.abs(np.array(m["w"]) - base)
        dc = np.mean([np.abs(np.array(r["w"]) - base) for r in cs], axis=0)
        p = float(wilcoxon(dm, dc).pvalue)
        print(f"  alpha {a}: |dw| mlp {dm.mean():.3f} ctrl {dc.mean():.3f} p {p:.3f}", flush=True)
    print("wrote results/dose2.json")


if __name__ == "__main__":
    main()
