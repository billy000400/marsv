"""Audit every released revision this study uses, to check that step16 is the only bad one.

`step16_forensics.py` proves what revision `step16` contains. This script asks the complementary
question the reader will ask next: are the other 20 revisions in the trajectory genuine, distinct
checkpoints, or is the repository sprinkled with duplicates that our behavioural QC could have
missed wherever two adjacent checkpoints happen to look similar?

The audit is exact and costs almost no bandwidth, because it needs no weights. For each revision we
fetch (a) the published LFS SHA-256 of `model.safetensors` from the Hub API and (b) the safetensors
header (a ~34 KB range request). Then:

  * every revision except step16 has byte-identical header length, tensor layout and
    `__metadata__`, so two such revisions have equal data sections **iff** their files are equal,
    i.e. iff their published SHA-256 agree;
  * therefore all-distinct SHA-256 across that group proves no two of those checkpoints ship the
    same weights, and none of them ships step143000's weights.

step16 is the one revision this argument cannot cover (its header differs, so file inequality is
uninformative about its data) -- which is exactly why it needed the full streamed comparison.

CPU + network only; ~1 MB downloaded in total.
"""

import json
from pathlib import Path

from step16_forensics import REVS as _UNUSED, header, published_digest  # noqa: F401

OUT = Path(__file__).resolve().parent.parent / "results" / "revision_audit.json"

STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000,
         16000, 32000, 64000, 96000, 128000, 143000]


def main():
    rows = {}
    for s in STEPS:
        rev = f"step{s}"
        n, h = header(rev)
        layout = {k: (v["dtype"], tuple(v["shape"]), tuple(v["data_offsets"]))
                  for k, v in h.items() if k != "__metadata__"}
        pub = published_digest(rev)
        rows[rev] = {
            "step": s,
            "header_bytes": n,
            "n_tensors": len(layout),
            "metadata": h.get("__metadata__"),
            "file_size": pub["size"],
            "file_sha256": pub["sha256"],
            "_layout": layout,
        }
        print(f"{rev:>12}  hdr={n}  meta={h.get('__metadata__')}  "
              f"size={pub['size']}  sha256={pub['sha256'][:16]}...")

    ref = rows["step143000"]["_layout"]
    same_layout = [r for r, v in rows.items() if v["_layout"] == ref
                   and v["header_bytes"] == rows["step143000"]["header_bytes"]
                   and v["metadata"] == rows["step143000"]["metadata"]]
    odd = [r for r in rows if r not in same_layout]

    digests = [rows[r]["file_sha256"] for r in same_layout]
    dup = len(digests) - len(set(digests))

    for v in rows.values():
        del v["_layout"]

    out = {
        "repo": "EleutherAI/pythia-1.4b-deduped",
        "n_revisions_audited": len(rows),
        "revisions": rows,
        "comparable_group": same_layout,
        "odd_revisions": odd,
        "n_duplicate_digests_in_group": dup,
        "all_distinct_in_group": dup == 0,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\ncomparable group: {len(same_layout)}/{len(rows)} revisions "
          f"(identical header layout + metadata)")
    print(f"odd revisions: {odd}")
    print(f"duplicate digests within the comparable group: {dup}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
