"""S1: tokenization checks, endpoint predictions (immediate + five readouts), endpoint JSD."""
import json
import os

import numpy as np
import torch

from common import (ENDPOINT_A, ENDPOINT_B, MODEL, PREFIX, READOUTS, RESULTS,
                    jsd_bits, load)


def single_token(tok, s):
    ids = tok.encode(s)
    return len(ids) == 1, ids


def top5(tok, probs):
    idx = np.argsort(-probs)[:5]
    return [[tok.decode([int(i)]), float(probs[int(i)])] for i in idx]


def main():
    tok, m, dev = load()
    out = {"model": MODEL, "prefix": PREFIX}

    # --- tokenization checks -------------------------------------------------
    checks = {}
    for s in [ENDPOINT_A, ENDPOINT_B] + [a for _, _, a, _ in READOUTS] + [b for _, _, _, b in READOUTS]:
        ok, ids = single_token(tok, s)
        checks[s] = {"single_token": ok, "ids": ids}
    suffix_tok = {}
    for name, suf, _, _ in READOUTS:
        ids = tok.encode(suf)
        suffix_tok[name] = {"suffix": suf, "ids": ids,
                            "pieces": [tok.decode([i]) for i in ids],
                            "n_tokens": len(ids)}
    out["token_checks"] = checks
    out["suffix_tokens"] = suffix_tok
    out["newline_id"] = tok.encode("\n")

    prefix_ids = tok.encode(PREFIX)
    out["prefix_n_tokens"] = len(prefix_ids)
    nl_id = tok.encode("\n")[0]

    def run(ids):
        with torch.no_grad():
            lg = m(torch.tensor([ids], device=dev), use_cache=False).logits[0, -1, :].float().cpu().numpy()
        p = torch.softmax(torch.tensor(lg), dim=-1).numpy()
        return lg, p

    # --- immediate (no readout suffix) --------------------------------------
    imm = {}
    probs_imm = {}
    for lab, ep in (("A", ENDPOINT_A), ("B", ENDPOINT_B)):
        ids = prefix_ids + tok.encode(ep)
        _, p = run(ids)
        probs_imm[lab] = p
        imm[lab] = {"endpoint": ep, "top1": tok.decode([int(np.argmax(p))]),
                    "p_newline": float(p[nl_id]), "top5": top5(tok, p)}
    imm["jsd_bits"] = jsd_bits(probs_imm["A"], probs_imm["B"])
    out["immediate"] = imm

    # --- five downstream readouts -------------------------------------------
    ro = {}
    for name, suf, ans_a, ans_b in READOUTS:
        suf_ids = tok.encode(suf)
        pr = {}
        for lab, ep in (("A", ENDPOINT_A), ("B", ENDPOINT_B)):
            ids = prefix_ids + tok.encode(ep) + suf_ids
            _, p = run(ids)
            pr[lab] = p
        ro[name] = {
            "suffix": suf,
            "A": {"top1": tok.decode([int(np.argmax(pr["A"]))]),
                  "p_expected": float(pr["A"][tok.encode(ans_a)[0]]),
                  "expected": ans_a, "top5": top5(tok, pr["A"])},
            "B": {"top1": tok.decode([int(np.argmax(pr["B"]))]),
                  "p_expected": float(pr["B"][tok.encode(ans_b)[0]]),
                  "expected": ans_b, "top5": top5(tok, pr["B"])},
            "jsd_bits": jsd_bits(pr["A"], pr["B"]),
        }
        ro[name]["A"]["top1_matches_expected"] = ro[name]["A"]["top1"] == ans_a
        ro[name]["B"]["top1_matches_expected"] = ro[name]["B"]["top1"] == ans_b
    out["readouts"] = ro

    path = os.path.join(RESULTS, "s1_endpoints.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    # --- console summary -----------------------------------------------------
    print("prefix tokens:", len(prefix_ids))
    for s, c in checks.items():
        print(f"  single-token {s!r}: {c['single_token']} {c['ids']}")
    for name, d in suffix_tok.items():
        print(f"  suffix {name}: {d['n_tokens']} tokens {d['pieces']}")
    print(f"immediate: A top1={imm['A']['top1']!r} p(nl)={imm['A']['p_newline']:.4f} | "
          f"B top1={imm['B']['top1']!r} p(nl)={imm['B']['p_newline']:.4f} | "
          f"JSD={imm['jsd_bits']:.4f} bits")
    for name, d in ro.items():
        print(f"{name:10s} A {d['A']['top1']!r} {d['A']['p_expected']:.3f} ({d['A']['top1_matches_expected']}) | "
              f"B {d['B']['top1']!r} {d['B']['p_expected']:.3f} ({d['B']['top1_matches_expected']}) | "
              f"JSD={d['jsd_bits']:.3f} bits")
    print("wrote", path)


if __name__ == "__main__":
    main()
