"""Forensics on the mislabelled `step16` revision of EleutherAI/pythia-1.4b-deduped.

The deliverables report that revision `step16` does not contain step-16 weights (its measured
d(t) curves are bit-identical to step143000's, and its held-out loss is the final model's).
That evidence is behavioural. This script settles it at the source, without trusting the local
model cache: it streams each revision's safetensors payload straight from the Hugging Face CDN
and hashes it.

Checks:
  1. Header comparison across revisions (tensor names, dtypes, shapes, byte offsets, metadata).
  2. SHA-256 of the full tensor-data section (everything after the header) for step16 and a set
     of comparison revisions, streamed in chunks -- nothing is written to disk.
  3. Per-tensor SHA-256 for a sample of tensors, so the match can be localised rather than
     resting on one whole-payload digest.

Negative control comes free from the published LFS digests: step8/step32/step64/step128/
step143000 all have identical header lengths, so their differing file digests imply differing
data sections. The method therefore demonstrably separates genuinely different checkpoints.

CPU + network only; no GPU, no model load.
"""

import hashlib
import json
import struct
import time
import urllib.request
from pathlib import Path

REPO = "EleutherAI/pythia-1.4b-deduped"
BASE = f"https://huggingface.co/{REPO}/resolve/{{}}/model.safetensors"
API = f"https://huggingface.co/api/models/{REPO}/paths-info/{{}}"

OUT = Path(__file__).resolve().parent.parent / "results" / "step16_forensics.json"

# step16 plus its two neighbours (what it should look like) and the final model (what it does
# look like). step128 is a second late-early control.
REVS = ["step8", "step16", "step32", "step64", "step128", "step143000"]
# revisions whose full data section we stream and hash (2.8 GB each)
FULL_HASH = ["step8", "step16", "step32", "step143000"]
# tensors sampled for per-tensor digests: first, middle and last block plus the embeddings
SAMPLE_HINTS = ["embed_in", "layers.0.", "layers.11.", "layers.23.", "embed_out", "final_layer_norm"]

CHUNK = 8 << 20


def _get(url, headers=None, retries=4):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            return urllib.request.urlopen(req, timeout=180)
        except Exception as exc:  # transient CDN failures on multi-GB streams
            if k == retries - 1:
                raise
            print(f"    retry {k + 1} after {type(exc).__name__}")
            time.sleep(5)


def read_range(rev, a, b):
    return _get(BASE.format(rev), {"Range": f"bytes={a}-{b}"}).read()


def header(rev):
    n = struct.unpack("<Q", read_range(rev, 0, 7))[0]
    return n, json.loads(read_range(rev, 8, 7 + n))


def published_digest(rev):
    """LFS sha256 and size as published by the Hub API (independent of what we download)."""
    body = json.dumps({"paths": ["model.safetensors"], "expand": True}).encode()
    req = urllib.request.Request(API.format(rev), data=body,
                                 headers={"Content-Type": "application/json"})
    info = json.load(urllib.request.urlopen(req, timeout=120))[0]
    return {"size": info["size"], "sha256": (info.get("lfs") or {}).get("oid")}


def stream_sha256(rev, start):
    """SHA-256 of the tensor-data section, streamed from the CDN. Never touches disk."""
    h = hashlib.sha256()
    n = 0
    t0 = time.time()
    resp = _get(BASE.format(rev), {"Range": f"bytes={start}-"})
    while True:
        buf = resp.read(CHUNK)
        if not buf:
            break
        h.update(buf)
        n += len(buf)
    print(f"    {rev}: {n / 2**30:.2f} GiB in {time.time() - t0:.0f} s")
    return h.hexdigest(), n


def main():
    res = {"repo": REPO, "revisions": {}}

    print("[1/3] headers + published digests")
    heads = {}
    for rev in REVS:
        n, h = header(rev)
        tensors = {k: v for k, v in h.items() if k != "__metadata__"}
        layout = {k: (v["dtype"], tuple(v["shape"]), tuple(v["data_offsets"]))
                  for k, v in tensors.items()}
        heads[rev] = (n, layout)
        pub = published_digest(rev)
        res["revisions"][rev] = {
            "header_bytes": n,
            "n_tensors": len(tensors),
            "metadata": h.get("__metadata__"),
            "file_size": pub["size"],
            "file_sha256": pub["sha256"],
            "data_start": 8 + n,
        }
        print(f"    {rev}: hdr={n} n={len(tensors)} meta={h.get('__metadata__')} "
              f"size={pub['size']} sha256={pub['sha256'][:16]}...")

    # layout equality: same tensor names/dtypes/shapes/offsets => data sections are comparable
    ref = heads["step143000"][1]
    for rev in REVS:
        res["revisions"][rev]["layout_identical_to_step143000"] = heads[rev][1] == ref

    print("[2/3] streaming sha256 of the tensor-data section")
    for rev in FULL_HASH:
        d, n = stream_sha256(rev, res["revisions"][rev]["data_start"])
        res["revisions"][rev]["data_sha256"] = d
        res["revisions"][rev]["data_bytes"] = n
        print(f"    {rev}: data sha256 = {d}")

    print("[3/3] per-tensor sha256 on a sample")
    _, layout = heads["step143000"]
    names = sorted(layout)
    sample = []
    for hint in SAMPLE_HINTS:
        hits = [k for k in names if hint in k]
        sample += hits[:2]
    sample = sorted(set(sample))
    res["sampled_tensors"] = sample
    per = {rev: {} for rev in FULL_HASH}
    for name in sample:
        a, b = layout[name][2]
        for rev in FULL_HASH:
            s = res["revisions"][rev]["data_start"]
            per[rev][name] = hashlib.sha256(read_range(rev, s + a, s + b - 1)).hexdigest()
        eq = {r: per[r][name] == per["step143000"][name] for r in FULL_HASH}
        print(f"    {name:52s} matches step143000: "
              + ", ".join(f"{r}={eq[r]}" for r in FULL_HASH if r != "step143000"))
    res["per_tensor_sha256"] = per
    res["per_tensor_match_step143000"] = {
        rev: sum(per[rev][n] == per["step143000"][n] for n in sample) for rev in FULL_HASH
    }

    d16 = res["revisions"]["step16"]["data_sha256"]
    res["verdict"] = {
        "step16_data_equals_step143000": d16 == res["revisions"]["step143000"]["data_sha256"],
        "step16_data_equals_step8": d16 == res["revisions"]["step8"]["data_sha256"],
        "step16_data_equals_step32": d16 == res["revisions"]["step32"]["data_sha256"],
        "step16_missing_metadata": res["revisions"]["step16"]["metadata"] is None,
        "header_byte_deficit": (res["revisions"]["step143000"]["header_bytes"]
                                - res["revisions"]["step16"]["header_bytes"]),
    }
    print("verdict:", json.dumps(res["verdict"], indent=2))

    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
