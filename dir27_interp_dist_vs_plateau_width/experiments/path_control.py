"""Paired interpolation-path control (PLAN addendum), GPT-2 Large, block-0 resid_post.

Usage: DIR27_SITE=block0_resid_post python3 path_control.py
Same endpoints a, b (unpatched last-position outputs of block 0) for two injected paths:
  linear                 r(t) = (1-t) a + t b                      (the main experiment's default)
  matthew_slerp_rescale  port of slerp_rescale from MShinkle/activation_plateau_mechanisms scripts/utils.py
Steps, in order: P1 freeze the pair list -> P2 validate -> P3 run both paths on the frozen pairs.
Saves per-point distances of each readout to both curve endpoints, so both normalizations can be computed offline.
"""
import json
import os

import numpy as np
import torch

import common
from common import RESULTS, SITE, TS, width, y_curve
from matthew_reference_utils import slerp_rescale as reference_slerp_rescale
from s1_sanity import WORDS

assert SITE == "block0_resid_post"
OUT = os.path.join(RESULTS, "path_control")
os.makedirs(OUT, exist_ok=True)
SEED, N_RANDOM, TOK_PER_BATCH = 0, 256, 8
REF_COMMIT = "4222c3efb77fe7983676cf96b3254dc000e8a0bc"
# already reported in REPORT.md: Figure 4a extremes, B = ' large', two size words of Finding 2, Figure 4b
ILLUSTRATIVE = [" pres", " hefty", " large", " huge", " whopping", " money"]
METHODS = ("linear", "matthew_slerp_rescale")


def linear(a, b, t):
    return (1 - t) * a + t * b


def slerp_rescale(a, b, t):
    """Batched port of the reference: a (d,), b (k, d), t (T, 1) -> (k, T, d). Same operations, same order,
    including theta = acos(|cos|). No epsilon or other numerical repair is added."""
    a, b = a.view(1, 1, -1), b.unsqueeze(1)
    na, nb = a.norm(dim=-1, keepdim=True), b.norm(dim=-1, keepdim=True)
    ua, ub = a / na, b / nb
    dot = torch.clamp((ua * ub).sum(dim=-1, keepdim=True), -1.0, 1.0)
    theta = torch.acos(torch.abs(dot))
    s = torch.sin(theta)
    res = torch.sin((1 - t) * theta) / s * ua + torch.sin(t * theta) / s * ub
    return res * ((1 - t) * na + t * nb)


def path(method, a, b, t):
    return linear(a.view(1, 1, -1), b.unsqueeze(1), t) if method == "linear" else slerp_rescale(a, b, t)


def path_length(method, a, b, n):
    """Sampled length of the injected path on an n-interval grid, float64, no model forward."""
    t = torch.linspace(0, 1, n + 1, dtype=torch.float64, device=a.device).view(-1, 1)
    p = path(method, a.double(), b.double(), t)
    return (p[:, 1:] - p[:, :-1]).norm(dim=-1).sum(dim=1)


def main():
    tok, m, dev = common.load()
    r = common.Runner(tok, m, dev)
    assert tok.decode([r.id_a]) == " big"

    # ---- P1: freeze the pair list before any run of the new path ----
    V = m.transformer.wte.weight.shape[0]
    eligible = np.array([i for i in range(V) if i != r.id_a])      # the sweep's vocabulary (s2_sweep.py)
    ids_rand = np.random.default_rng(SEED).choice(eligible, N_RANDOM, replace=False)
    ids_ill = [tok.encode(w)[0] for w in ILLUSTRATIVE]
    assert all(len(tok.encode(w)) == 1 for w in ILLUSTRATIVE)
    meta = dict(model="gpt2-large", dtype="float32", prompt="The house was big", prompt_ids=r.ids,
                intervention_site=SITE, distance_space=SITE, hook_module="GPT2Model.h[0] output",
                readout_blocks=[common.MID_BLOCK, common.FINAL_BLOCK], n_t=len(TS), methods=METHODS,
                normalization_primary="y_fixed = |x(t)-x(0)| / |x(1)-x(0)| (common.y_curve; x(0), x(1) are the "
                                      "patched run's own t=0 and t=1 readouts)",
                normalization_secondary="y_relative = |x(t)-x(0)| / (|x(t)-x(0)| + |x(t)-x(1)|)",
                reference="MShinkle/activation_plateau_mechanisms scripts/utils.py slerp_rescale",
                reference_commit=REF_COMMIT, seed=SEED,
                sampling="np.random.default_rng(0).choice(all token ids except ' big', 256, replace=False)",
                random_ids=ids_rand.tolist(), illustrative_ids=ids_ill, illustrative_tokens=ILLUSTRATIVE,
                illustrative_also_in_random=[int(i) for i in ids_ill if i in set(ids_rand.tolist())])
    json.dump(meta, open(os.path.join(OUT, "pairs.json"), "w"), indent=1)

    t = torch.tensor(TS, device=dev, dtype=torch.float32).view(-1, 1)
    seen = {}
    m.transformer.h[1].register_forward_pre_hook(
        lambda mod, args, kwargs: seen.__setitem__("x", (args[0] if args else kwargs["hidden_states"]).float().clone()),
        with_kwargs=True)
    rel = lambda x, y: float(((x - y).norm(dim=-1) / y.norm(dim=-1)).max())

    mid_a, fin_a = [x.clone() for x in r.run(r.wte[torch.tensor([r.id_a], device=dev)])]
    a, pre_a = r.cap["resid0"][0].clone(), seen["x"][0, :-1].clone()

    def run_pairs(ids, check):
        """Both paths for token ids; returns per-method arrays and updates the worst-case checks."""
        res = {me: dict(dA_mid=[], dB_mid=[], dA_fin=[], dB_fin=[], progress=[], norm_ratio=[]) for me in METHODS}
        D, cos, nb_all = [], [], []
        for b0 in range(0, len(ids), TOK_PER_BATCH):
            ib = torch.tensor(ids[b0:b0 + TOK_PER_BATCH], device=dev)
            mid_b, fin_b = [x.clone() for x in r.run(r.wte[ib])]          # independent unpatched B runs
            b = r.cap["resid0"].clone()
            k = len(ib)
            D.append((b - a).norm(dim=-1))
            cos.append(torch.nn.functional.cosine_similarity(b, a.unsqueeze(0), dim=-1))
            nb_all.append(b.norm(dim=-1))
            for me in METHODS:
                p = path(me, a, b, t)                                        # (k, T, d)
                mid, fin = [x.view(k, len(TS), -1) for x in r.run_patched(p.reshape(-1, p.shape[-1]))]
                x = seen["x"].view(k, len(TS), 4, -1)
                c = check[me]
                c["nonfinite"] += int((~torch.isfinite(p)).sum())
                c["injected_vs_path_max_abs"] = max(c["injected_vs_path_max_abs"], float((x[:, :, -1] - p).abs().max()))
                c["prefix_vs_unpatched_A_max_abs"] = max(c["prefix_vs_unpatched_A_max_abs"],
                                                         float((x[:, :, :-1] - pre_a).abs().max()))
                c["path_t0_vs_a_max_rel_l2"] = max(c["path_t0_vs_a_max_rel_l2"], rel(p[:, 0], a.expand(k, -1)))
                c["path_t1_vs_b_max_rel_l2"] = max(c["path_t1_vs_b_max_rel_l2"], rel(p[:, -1], b))
                for name, got, ref in (("mid_t0", mid[:, 0], mid_a.expand(k, -1)), ("fin_t0", fin[:, 0], fin_a.expand(k, -1)),
                                       ("mid_t1", mid[:, -1], mid_b), ("fin_t1", fin[:, -1], fin_b)):
                    c[f"readout_{name}_vs_unpatched_max_rel_l2"] = max(c[f"readout_{name}_vs_unpatched_max_rel_l2"],
                                                                      rel(got, ref))
                d = res[me]
                d["dA_mid"].append((mid - mid[:, :1]).norm(dim=-1))
                d["dB_mid"].append((mid - mid[:, -1:]).norm(dim=-1))
                d["dA_fin"].append((fin - fin[:, :1]).norm(dim=-1))
                d["dB_fin"].append((fin - fin[:, -1:]).norm(dim=-1))
                d["progress"].append((p - a).norm(dim=-1) / (b - a).norm(dim=-1, keepdim=True))
                nominal = (1 - t.view(1, -1)) * a.norm() + t.view(1, -1) * b.norm(dim=-1, keepdim=True)
                d["norm_ratio"].append(p.norm(dim=-1) / nominal)       # realized / linearly interpolated norm
                assert torch.equal(y_curve(mid), d["dA_mid"][-1] / d["dA_mid"][-1][:, -1:])
                for n in (100, 1000, 2000):
                    d.setdefault(f"L{n}", []).append(path_length(me, a, b, n))
        out = {me: {k_: torch.cat(v).cpu().numpy() for k_, v in res[me].items()} for me in METHODS}
        return out, torch.cat(D).cpu().numpy(), torch.cat(cos).cpu().numpy(), torch.cat(nb_all).cpu().numpy()

    def new_check():
        keys = ["injected_vs_path_max_abs", "prefix_vs_unpatched_A_max_abs", "path_t0_vs_a_max_rel_l2",
                "path_t1_vs_b_max_rel_l2"] + [f"readout_{n}_vs_unpatched_max_rel_l2"
                                              for n in ("mid_t0", "fin_t0", "mid_t1", "fin_t1")]
        return {me: dict({k_: 0.0 for k_ in keys}, nonfinite=0) for me in METHODS}

    # ---- P2: validate before the pilot ----
    val = dict(reference_commit=REF_COMMIT)
    # (i) port vs pinned reference: random nondegenerate vectors (cosines of both signs) and real endpoints
    g = torch.Generator().manual_seed(0)
    va = torch.randn(1280, generator=g).to(dev)
    vb = torch.randn(64, 1280, generator=g).to(dev)
    vb[:16] += 3 * va            # positive cosines
    vb[16:32] -= 3 * va          # negative cosines
    ids_val = [tok.encode(w)[0] for w in WORDS] + ids_ill
    r.run(r.wte[torch.tensor(ids_val, device=dev)])
    b_val = r.cap["resid0"].clone()
    for name, x0, x1 in (("random_vectors", va, vb), ("real_block0_endpoints", a, b_val)):
        port = slerp_rescale(x0, x1, t)
        ref = torch.stack([torch.stack([reference_slerp_rescale(x0.unsqueeze(0), x1[j:j + 1], float(tt)).squeeze(0)
                                        for tt in t.view(-1)]) for j in range(len(x1))])
        val[f"port_vs_reference_max_abs_{name}"] = float((port - ref).abs().max())
        val[f"port_vs_reference_max_rel_l2_{name}"] = rel(port.reshape(-1, port.shape[-1]), ref.reshape(-1, ref.shape[-1]))
        val[f"cosine_range_{name}"] = [float(torch.nn.functional.cosine_similarity(x1, x0.unsqueeze(0), dim=-1).min()),
                                       float(torch.nn.functional.cosine_similarity(x1, x0.unsqueeze(0), dim=-1).max())]
    # (ii) hook, endpoints, raw readouts on the 20 sanity tokens + the illustrative pairs
    chk = new_check()
    vres, vD, vcos, _ = run_pairs(np.array(ids_val), chk)
    val.update(n_tokens=len(ids_val), token_ids=ids_val, checks=chk,
               path_length_over_D={me: dict(n100=(vres[me]["L100"] / vD).tolist(), n1000=(vres[me]["L1000"] / vD).tolist(),
                                            n2000=(vres[me]["L2000"] / vD).tolist()) for me in METHODS},
               path_length_doubling_max_rel_change={me: float(np.abs(vres[me]["L2000"] / vres[me]["L1000"] - 1).max())
                                                    for me in METHODS},
               cosine_min_max=[float(vcos.min()), float(vcos.max())],
               norm_ratio_min_max={me: [float(vres[me]["norm_ratio"].min()), float(vres[me]["norm_ratio"].max())]
                                   for me in METHODS})
    json.dump(val, open(os.path.join(OUT, "validation.json"), "w"), indent=1)
    print(json.dumps({k_: v for k_, v in val.items() if k_ != "path_length_over_D"}, indent=1), flush=True)
    for me in METHODS:
        c = chk[me]
        assert c["nonfinite"] == 0 and c["injected_vs_path_max_abs"] == 0
        # prefix: float32 batch-size noise only (batch-of-1 reference vs batch of 808), as in b0_validate.py
        assert c["prefix_vs_unpatched_A_max_abs"] < 1e-3
        assert max(v for k_, v in c.items() if "rel_l2" in k_) < 1e-3, c
    assert val["port_vs_reference_max_rel_l2_random_vectors"] < 1e-5
    assert val["port_vs_reference_max_rel_l2_real_block0_endpoints"] < 1e-5

    # ---- P3: paired pilot on the frozen pairs (random sample first, illustrative pairs appended) ----
    ids = np.concatenate([ids_rand, np.array(ids_ill)])
    chk = new_check()
    res, D, cos, nb = run_pairs(ids, chk)
    save = dict(ids=ids, is_random=np.arange(len(ids)) < N_RANDOM, D_endpoint=D, cosine=cos,
                norm_a=float(a.norm()), norm_b=nb, ts=TS)
    for me in METHODS:
        d = res[me]
        for k_ in ("dA_mid", "dB_mid", "dA_fin", "dB_fin", "progress", "norm_ratio"):
            save[f"{me}__{k_}"] = d[k_].astype(np.float32)
        for n in (100, 1000, 2000):
            save[f"{me}__L_path_n{n}"] = d[f"L{n}"]
    np.savez(os.path.join(OUT, "pilot.npz"), **save)
    json.dump(dict(n_pairs=len(ids), checks_all_pairs=chk), open(os.path.join(OUT, "pilot_checks.json"), "w"), indent=1)
    print(json.dumps(chk, indent=1))
    print("cos min/max", cos.min(), cos.max(), "n negative", int((cos < 0).sum()))


if __name__ == "__main__":
    main()
