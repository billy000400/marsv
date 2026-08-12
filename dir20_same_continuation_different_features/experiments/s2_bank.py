"""S2 (fresh confirmatory plan): build and LOCK matched within-prefix contrasts before any width.

For 300 fresh WikiText-103 *test* spans (seed 31, 20-40 tokens), take the top 24 printable
next tokens as candidate final tokens. For every unordered candidate pair we record, WITHOUT
running any interpolation:

  * successor JSD between the two endpoints' next-token distributions (nats),
  * F = Jaccard distance between the two endpoints' top-64-per-block MLP feature sets (blocks 1-35),
  * final-logit L2 endpoint distance, block-0 endpoint angle, block-0 |log norm ratio|,
  * mean surprisal of the pair's two final tokens under the shared prefix.

Then, within each prefix, pick ONE contrast: two pairs on four distinct final tokens, matched on
JSD and the four confounds, differing in F. Writes results/matched_pairs.json (+ sha256) and
results/matched_features.npz. S3 may only run after this file exists.
"""
import hashlib
import itertools
import json
import os

import numpy as np
import torch
from datasets import load_dataset

from common import RESULTS, blocks, load

SEED = 31
N_PREFIX = 1400          # every eligible wikitext-103 test paragraph (1395 of them)
N_CAND = 24
TOPK_NEURONS = 64
JSD_LO, JSD_HI = 0.005, 0.20
CONFOUNDS = ["logit_dist", "angle0", "lognorm_ratio", "mean_surprisal"]
PRIMARY = dict(version="primary", djsd=0.01, conf=0.50, df=0.10)
RELAXED = dict(version="relaxed", djsd=0.02, conf=0.75, df=0.08)


def get_spans(tok, rng):
    """N_PREFIX random 20-40-token spans from the wikitext-103 test split."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    out = []
    for t in ds["text"]:
        s = t.strip()
        if len(s) < 400 or s.startswith("="):
            continue
        ids = tok(s)["input_ids"]
        n = int(rng.integers(20, 41))
        if len(ids) < n + 10:
            continue
        st = int(rng.integers(0, len(ids) - n))
        out.append(ids[st:st + n])
        if len(out) == N_PREFIX:
            break
    return out


def candidate_tokens(tok, logits):
    """Top N_CAND printable, non-special next tokens."""
    special = set(tok.all_special_ids)
    out = []
    for i in logits.topk(200).indices.tolist():
        if i in special:
            continue
        s = tok.decode([i])
        if s.strip() == "" or not all(c.isprintable() for c in s):
            continue
        out.append(i)
        if len(out) == N_CAND:
            break
    return out


def endpoint_pass(m, ids_batch, wout_norms, n_blocks):
    """One batched forward over the N_CAND complete prompts.

    Returns final-token logits, block-0 resid_post, and the top-64 (block, neuron) feature set
    per prompt, scored by |post-GELU activation| * ||W^out row||_2 in blocks 1..n_blocks-1.
    """
    dev = next(m.parameters()).device
    rec = {}
    hs = []
    bl = blocks(m)

    def mk_resid(hook_out):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hook_out["h0"] = h[:, -1, :].detach().float()
        return hook
    hs.append(bl[0].register_forward_hook(mk_resid(rec)))

    acts = {}

    def mk_act(i):
        def hook(mod, inp, out):
            acts[i] = out[:, -1, :].detach().float()
        return hook
    for i in range(1, n_blocks):
        hs.append(bl[i].mlp.act.register_forward_hook(mk_act(i)))

    with torch.no_grad():
        lg = m(torch.tensor(ids_batch, device=dev), use_cache=False).logits[:, -1, :].float()
    for h in hs:
        h.remove()

    feats = []
    for i in range(1, n_blocks):
        s = acts[i].abs() * wout_norms[i].unsqueeze(0)          # (B, 4d)
        idx = s.topk(TOPK_NEURONS, dim=1).indices                # (B, 64)
        feats.append(idx + i * 100000)                           # unique (block, neuron) id
    return lg, rec["h0"], torch.cat(feats, dim=1).cpu().numpy().astype(np.int32)


def pair_metrics(lg, h0, feats, surp):
    """Metrics for every unordered candidate pair. Returns list of dicts."""
    p = torch.softmax(lg, -1)
    logp = torch.log(p + 1e-45)
    n = lg.shape[0]
    rows = []
    hn = h0 / h0.norm(dim=1, keepdim=True)
    sets = [set(feats[i].tolist()) for i in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        mm = 0.5 * (p[i] + p[j])
        lm = torch.log(mm + 1e-45)
        js = float(0.5 * (p[i] * (logp[i] - lm)).sum() + 0.5 * (p[j] * (logp[j] - lm)).sum())
        inter = len(sets[i] & sets[j])
        union = len(sets[i] | sets[j])
        rows.append(dict(
            i=i, j=j, jsd=js, F=1.0 - inter / union,
            logit_dist=float((lg[i] - lg[j]).norm()),
            angle0=float(torch.arccos(torch.clamp((hn[i] * hn[j]).sum(), -1, 1))),
            lognorm_ratio=float(torch.log(h0[i].norm() / h0[j].norm()).abs()),
            mean_surprisal=0.5 * (surp[i] + surp[j]),
        ))
    return rows


def choose(prefix_rows, sd, rule):
    """Best matched contrast within one prefix under `rule`, or None."""
    best = None
    for a, b in itertools.combinations(prefix_rows, 2):
        if len({a["i"], a["j"], b["i"], b["j"]}) != 4:
            continue
        if abs(a["jsd"] - b["jsd"]) > rule["djsd"]:
            continue
        dist = float(np.sqrt(sum(((a[c] - b[c]) / sd[c]) ** 2 for c in CONFOUNDS)))
        if dist > rule["conf"]:
            continue
        hi, lo = (a, b) if a["F"] >= b["F"] else (b, a)
        df = hi["F"] - lo["F"]
        if df < rule["df"]:
            continue
        cand = (df, -dist, hi, lo, dist)
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    return best


def main():
    rng = np.random.default_rng(SEED)
    tok, m = load("gpt2-large")
    dev = next(m.parameters()).device
    n_blocks = len(blocks(m))
    wout = {i: blocks(m)[i].mlp.c_proj.weight.detach().float().norm(dim=1).to(dev)
            for i in range(1, n_blocks)}          # Conv1D: weight is (4d, d), row j = neuron j

    spans = get_spans(tok, rng)
    print(f"{len(spans)} spans")

    all_rows, prefixes, feat_store = [], [], {}
    for pi, pre in enumerate(spans):
        with torch.no_grad():
            plg = m(torch.tensor([pre], device=dev), use_cache=False).logits[0, -1, :].float()
        cands = candidate_tokens(tok, plg)
        if len(cands) < N_CAND:
            continue
        surp = (-torch.log_softmax(plg, -1)[cands]).tolist()
        ids_batch = [pre + [c] for c in cands]
        lg, h0, feats = endpoint_pass(m, ids_batch, wout, n_blocks)
        rows = pair_metrics(lg, h0, feats, surp)
        for r in rows:
            r["prefix"] = pi
        all_rows.extend(rows)
        prefixes.append(dict(idx=pi, ids=pre, text=tok.decode(pre), cands=cands,
                             cand_str=[tok.decode([c]) for c in cands], surprisal=surp))
        feat_store[str(pi)] = feats
        if (pi + 1) % 50 == 0:
            print(f"  prefix {pi + 1}: {len(all_rows)} candidate pairs")
        del lg, h0
    torch.cuda.empty_cache()

    # Eligibility filters, applied bank-wide before any matching.
    ld = np.array([r["logit_dist"] for r in all_rows])
    ld_p10 = float(np.percentile(ld, 10))
    elig = [r for r in all_rows
            if JSD_LO <= r["jsd"] <= JSD_HI and r["logit_dist"] > ld_p10]
    print(f"{len(all_rows)} candidate pairs -> {len(elig)} eligible "
          f"(JSD in [{JSD_LO},{JSD_HI}], logit_dist > p10={ld_p10:.2f})")

    sd = {c: float(np.std([r[c] for r in elig])) for c in CONFOUNDS}
    by_prefix = {}
    for r in elig:
        by_prefix.setdefault(r["prefix"], []).append(r)

    for rule in (PRIMARY, RELAXED):
        contrasts = []
        for pi, rows in sorted(by_prefix.items()):
            b = choose(rows, sd, rule)
            if b is not None:
                contrasts.append(dict(prefix=pi, dF=b[0], conf_dist=b[4],
                                      high=b[2], low=b[3]))
        print(f"rule={rule['version']}: {len(contrasts)} contrasts")
        if len(contrasts) >= 80:
            break

    pref_by_idx = {p["idx"]: p for p in prefixes}
    out = dict(
        matching_version=rule["version"], rule=rule, seed=SEED, model="gpt2-large",
        n_prefixes=len(prefixes), n_candidate_pairs=len(all_rows), n_eligible=len(elig),
        logit_dist_p10=ld_p10, confound_sd=sd, topk_neurons=TOPK_NEURONS,
        n_contrasts=len(contrasts), contrasts=[], bank_summary=dict(
            jsd=[float(np.percentile([r["jsd"] for r in all_rows], q)) for q in (5, 50, 95)],
            F=[float(np.percentile([r["F"] for r in all_rows], q)) for q in (5, 50, 95)]),
    )
    for c in contrasts:
        p = pref_by_idx[c["prefix"]]
        e = dict(prefix=c["prefix"], text=p["text"], ids=p["ids"],
                 dF=c["dF"], conf_dist=c["conf_dist"])
        for lab in ("high", "low"):
            r = c[lab]
            e[lab] = {k: r[k] for k in
                      ("i", "j", "jsd", "F", "logit_dist", "angle0",
                       "lognorm_ratio", "mean_surprisal")}
            e[lab]["tok_a"] = p["cands"][r["i"]]
            e[lab]["tok_b"] = p["cands"][r["j"]]
            e[lab]["str_a"] = p["cand_str"][r["i"]]
            e[lab]["str_b"] = p["cand_str"][r["j"]]
        out["contrasts"].append(e)

    path = os.path.join(RESULTS, "matched_pairs.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print("matched_pairs.json sha256:", h)
    with open(os.path.join(RESULTS, "matched_pairs.sha256"), "w") as f:
        f.write(h + "\n")
    np.savez_compressed(os.path.join(RESULTS, "matched_features.npz"),
                        **{k: feat_store[k] for k in
                           {str(c["prefix"]) for c in contrasts}})
    # Bank-level scatter data for the balance figure / power reporting.
    np.savez_compressed(os.path.join(RESULTS, "bank_pairs.npz"),
                        jsd=np.array([r["jsd"] for r in all_rows]),
                        F=np.array([r["F"] for r in all_rows]),
                        logit_dist=ld,
                        prefix=np.array([r["prefix"] for r in all_rows]))
    return out


if __name__ == "__main__":
    main()
