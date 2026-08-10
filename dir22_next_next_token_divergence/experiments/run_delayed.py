"""Delayed-successor plateau: interpolate the ` A`/` B` token embedding, read out
immediately and after the shared successor ` means`.  Runs S1 (validate), S2 (sweep), S3 (metrics).
"""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.225)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

PREFIX = "Use the codebook A = cat and B = dog. Complete: Symbol"
A, B, S = " A", " B", " means"
CAT, DOG = " cat", " dog"
N_T = 101


def one_token(tok, s):
    ids = tok(s)["input_ids"]
    return len(ids) == 1, ids


def slerp_lerp_norm(ea, eb, ts):
    """SLERP the direction, linearly interpolate the L2 norm (Matthew's interpolation)."""
    na, nb = ea.norm(), eb.norm()
    u, v = ea / na, eb / nb
    cos = torch.clamp((u * v).sum(), -1.0, 1.0)
    om = torch.arccos(cos)
    a = torch.tensor(ts, device=ea.device, dtype=ea.dtype).unsqueeze(1)
    if om.abs() < 1e-6:
        d = (1 - a) * u + a * v
        d = d / d.norm(dim=1, keepdim=True)
    else:
        d = (torch.sin((1 - a) * om) * u + torch.sin(a * om) * v) / torch.sin(om)
    return ((1 - a) * na + a * nb) * d, float(om), float(cos)


def jsd(p, q):
    m = 0.5 * (p + q)
    def kl(x, y):
        mask = x > 0
        return float((x[mask] * (np.log(x[mask]) - np.log(y[mask]))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def rel_dist(z, za, zb):
    da = np.linalg.norm(z - za, axis=1)
    db = np.linalg.norm(z - zb, axis=1)
    return da / (da + db)


def first_cross(ts, d, q):
    idx = np.flatnonzero(d >= q)
    return float(ts[idx[0]]) if len(idx) else float("nan")


def width_10_90(ts, d):
    return first_cross(ts, d, 0.9) - first_cross(ts, d, 0.1)


@torch.no_grad()
def sweep(model, ids, sym_pos, vecs, chunk=16):
    """Run the sequence with the symbol-position embedding replaced by each row of `vecs`."""
    wte = model.transformer.wte
    base = wte(torch.tensor([ids], device=DEV))
    out = []
    for i in range(0, len(vecs), chunk):
        v = vecs[i:i + chunk]
        emb = base.repeat(len(v), 1, 1).clone()
        emb[:, sym_pos, :] = v
        lg = model(inputs_embeds=emb).logits[:, -1, :]
        out.append(lg.float().cpu().numpy())
    return np.concatenate(out, 0)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2-large")
    model = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float32)
    model.eval().to(DEV)

    # ---- S1: tokenization + endpoint checks -------------------------------
    single = {s: one_token(tok, s)[0] for s in (A, B, S, CAT, DOG)}
    ids_pre = tok(PREFIX)["input_ids"]
    ids_a = tok(PREFIX + A)["input_ids"]
    ids_b = tok(PREFIX + B)["input_ids"]
    assert ids_a[:-1] == ids_pre and ids_b[:-1] == ids_pre, "prefix retokenized"
    sym_pos = len(ids_pre)                       # index of the interpolated symbol token
    ids_a_s = tok(PREFIX + A + S)["input_ids"]
    ids_b_s = tok(PREFIX + B + S)["input_ids"]
    assert ids_a_s[:-1] == ids_a and ids_b_s[:-1] == ids_b, "successor retokenized"

    tid = {s: tok(s)["input_ids"][0] for s in (S, CAT, DOG)}

    with torch.no_grad():
        lg = {}
        for k, ids in (("A", ids_a), ("B", ids_b), ("A_S", ids_a_s), ("B_S", ids_b_s)):
            lg[k] = model(torch.tensor([ids], device=DEV)).logits[0, -1].float().cpu().numpy()
    pr = {k: np.exp(v - v.max()) / np.exp(v - v.max()).sum() for k, v in lg.items()}

    s1 = dict(
        single_token=single,
        sym_pos=sym_pos,
        top1={k: tok.decode(int(v.argmax())) for k, v in lg.items()},
        p_top1={k: float(v.max()) for k, v in pr.items()},
        immediate_endpoint_jsd_nats=jsd(pr["A"], pr["B"]),
        delayed_endpoint_jsd_nats=jsd(pr["A_S"], pr["B_S"]),
        p_means={k: float(pr[k][tid[S]]) for k in ("A", "B")},
        p_cat={k: float(pr[k][tid[CAT]]) for k in ("A_S", "B_S")},
        p_dog={k: float(pr[k][tid[DOG]]) for k in ("A_S", "B_S")},
    )
    s1["endpoints_valid"] = bool(
        all(single.values())
        and s1["top1"]["A"] == S and s1["top1"]["B"] == S
        and s1["top1"]["A_S"] == CAT and s1["top1"]["B_S"] == DOG)
    print(json.dumps(s1, indent=2))

    # ---- S2: interpolation sweep ------------------------------------------
    ts = np.linspace(0, 1, N_T)
    wte = model.transformer.wte
    ea = wte.weight[ids_a[-1]].detach()
    eb = wte.weight[ids_b[-1]].detach()
    vecs, omega, cos = slerp_lerp_norm(ea, eb, ts)

    z_imm = sweep(model, ids_a, sym_pos, vecs)          # readout at the symbol position
    z_del = sweep(model, ids_a_s, sym_pos, vecs)        # readout after ` means`

    def probs(z):
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    p_imm, p_del = probs(z_imm), probs(z_del)

    # ---- S3: plateau metrics ----------------------------------------------
    d_imm = rel_dist(z_imm, z_imm[0], z_imm[-1])
    d_del = rel_dist(z_del, z_del[0], z_del[-1])
    out = dict(
        s1=s1, omega_rad=omega, cos_ab=cos,
        norm_a=float(ea.norm()), norm_b=float(eb.norm()),
        t=ts.tolist(),
        d_immediate=d_imm.tolist(), d_delayed=d_del.tolist(),
        w_immediate=width_10_90(ts, d_imm), w_delayed=width_10_90(ts, d_del),
        p_means=p_imm[:, tid[S]].tolist(),
        top1_immediate=[tok.decode(int(i)) for i in z_imm.argmax(1)],
        p_cat=p_del[:, tid[CAT]].tolist(), p_dog=p_del[:, tid[DOG]].tolist(),
        top1_delayed=[tok.decode(int(i)) for i in z_del.argmax(1)],
        jsd_immediate_vs_A=[jsd(p_imm[i], p_imm[0]) for i in range(N_T)],
    )
    flip = np.flatnonzero(np.array(out["p_dog"]) > np.array(out["p_cat"]))
    out["flip_t_delayed"] = float(ts[flip[0]]) if len(flip) else None
    out["monotonic_delayed"] = bool(np.all(np.diff(d_del) >= -1e-6))
    out["monotonic_immediate"] = bool(np.all(np.diff(d_imm) >= -1e-6))
    out["t50_immediate"] = first_cross(ts, d_imm, 0.5)
    out["t50_delayed"] = first_cross(ts, d_del, 0.5)
    # Absolute endpoint separation: d(t) is scale-free, so report how large the gap it normalises is.
    out["sep_immediate"] = float(np.linalg.norm(z_imm[0] - z_imm[-1]))
    out["sep_delayed"] = float(np.linalg.norm(z_del[0] - z_del[-1]))
    # Decision margin at the delayed readout: does the sweep ever come close to flipping the top-1?
    srt = np.sort(z_del, axis=1)
    out["top2_margin_delayed"] = (srt[:, -1] - srt[:, -2]).tolist()
    np.save(os.path.join(RESULTS, "logits_delayed.npy"), z_del.astype(np.float32))
    np.save(os.path.join(RESULTS, "logits_immediate.npy"), z_imm.astype(np.float32))
    with open(os.path.join(RESULTS, "delayed.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("w_immediate", out["w_immediate"], "w_delayed", out["w_delayed"],
          "flip_t", out["flip_t_delayed"])


if __name__ == "__main__":
    main()
