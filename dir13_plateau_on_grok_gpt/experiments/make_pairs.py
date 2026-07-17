"""S3 — Generate and FREEZE natural minimal prompt pairs for the Matthew-style slerp assay.

Each pair: two length-128 sequences `prefix + char_A` vs `prefix + char_B` that are identical
except for the final input character. Prefixes come from held-out (val) Shakespeare text,
deduplicated. Endpoint characters are chosen WITHOUT looking at any interpolation path:

  - char_A = the character actually observed after the prefix in the held-out text;
  - char_B = the model's highest-probability next character that differs from char_A
    (PLAN option 2; option 1 — two chars observed after the same 127-char prefix — is
    infeasible because 127-char prefixes are essentially unique in a 1.1M-char corpus).

Degenerate-pair exclusion (frozen BEFORE the full run): a pair is excluded iff the endpoint
final-logit distance ||x_A - x_B||_2 < 1e-3. Everything is written to prompt_pairs.json.
"""
import os, sys, json, hashlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

PAIR_SEED = 20260717
N_PAIRS = 40
PREFIX_LEN = 127          # + 1 endpoint char = block_size 128
DEGENERATE_THRESH = 1e-3  # frozen numerical threshold on ||x_A - x_B||_2 (final logits)


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = open("/tmp/tinyshakespeare.txt", "rb").read()
    sha = hashlib.sha256(text).hexdigest()
    meta = json.load(open(os.path.join(RES, "train_meta.json")))
    assert sha == meta["corpus_sha256"], "corpus mismatch with training provenance"
    text = text.decode("utf-8")
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data))
    val = data[n:]  # same split as train.py

    ckpt = torch.load(os.path.join(RES, "checkpoints", "ckpt_final.pt"),
                      map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rng = np.random.default_rng(PAIR_SEED)
    # oversample candidate prefix starts; dedup on prefix text
    starts = rng.choice(len(val) - PREFIX_LEN - 1, size=4 * N_PAIRS, replace=False)
    seen_prefix = set()
    pairs, n_degenerate = [], 0
    with torch.no_grad():
        for s in starts:
            if len(pairs) >= N_PAIRS:
                break
            s = int(s)
            prefix_ids = val[s:s + PREFIX_LEN]
            prefix = "".join(itos[i] for i in prefix_ids)
            if prefix in seen_prefix:
                continue
            seen_prefix.add(prefix)
            char_A_id = int(val[s + PREFIX_LEN])  # observed continuation in held-out text
            # model next-char distribution given the prefix
            x = torch.from_numpy(prefix_ids).to(device)[None, :]
            logits, _ = model(x)
            probs = torch.softmax(logits[0, -1], dim=-1).cpu().numpy()
            # char_B = highest-probability char != char_A
            order = np.argsort(-probs)
            char_B_id = int(order[0]) if int(order[0]) != char_A_id else int(order[1])
            # endpoint final logits (full length-128 forwards) for the degeneracy check
            seq_A = np.concatenate([prefix_ids, [char_A_id]])
            seq_B = np.concatenate([prefix_ids, [char_B_id]])
            xb = torch.from_numpy(np.stack([seq_A, seq_B])).to(device)
            lg, _ = model(xb)
            xA, xB = lg[0, -1], lg[1, -1]
            ep_dist = float(torch.linalg.norm(xA - xB))
            if ep_dist < DEGENERATE_THRESH:
                n_degenerate += 1
                continue
            pairs.append({
                "pair_id": len(pairs),
                "val_start": s, "abs_start": int(n + s), "prefix_len": PREFIX_LEN,
                "prefix_tail": prefix[-40:],
                "char_A": itos[char_A_id], "char_B": itos[char_B_id],
                "char_A_id": char_A_id, "char_B_id": char_B_id,
                "p_A": float(probs[char_A_id]), "p_B": float(probs[char_B_id]),
                "endpoint_logit_dist": ep_dist,
                "selection": "A=observed next char in val text; B=model top-1 (top-2 if ==A)",
            })

    out = {
        "seed": PAIR_SEED, "n_pairs": len(pairs), "prefix_len": PREFIX_LEN,
        "split": "val (last 10% of corpus, same split as training)",
        "corpus_sha256": sha, "checkpoint": "ckpt_final.pt",
        "degenerate_threshold_l2_logits": DEGENERATE_THRESH,
        "n_excluded_degenerate": n_degenerate,
        "frozen_before_any_interpolation": True,
        "pairs": pairs,
    }
    with open(os.path.join(RES, "prompt_pairs.json"), "w") as f:
        json.dump(out, f, indent=1)
    dists = [p["endpoint_logit_dist"] for p in pairs]
    pbs = [p["p_B"] for p in pairs]
    print(f"froze {len(pairs)} pairs (excluded {n_degenerate} degenerate); "
          f"endpoint logit dist median {np.median(dists):.2f} range [{min(dists):.2f},{max(dists):.2f}]; "
          f"median p_B {np.median(pbs):.3f}", flush=True)
    print("endpoint char combos:", sorted({(p['char_A'], p['char_B']) for p in pairs})[:20], flush=True)


if __name__ == "__main__":
    main()
