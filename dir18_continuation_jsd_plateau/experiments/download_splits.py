"""Byte-range download of two distant, row-aligned samples of Pythia's released training stream.

The preshuffled deduped Pile is one concatenated uint16 stream of 146,432,000 sequences of exactly
2049 tokens. Verified against the official Megatron .idx header: magic MMIDIDX, version 1, dtype
code 8 = uint16, len = 146,432,000, every listed size 2049, and 34 + 12L + 8D == the .idx file's
actual byte length (1,757,184,042); the .bin shards total 600,078,336,000 bytes == 146,432,000 *
2049 * 2. So the byte offset of global row i is exactly i * 4098 and a row-aligned sample is a plain
byte range -- no unsharding and no 602 GB download.

We never join two rows. The per-chunk ledger makes this resumable: the process is periodically
killed by memory/quota pressure from the other agents sharing this box, so a supervisor reruns it.
"""
import concurrent.futures as cf
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "/tmp/dir18_data"  # local disk: the shared network volume hits EDQUOT under 4 concurrent agents
URL = ("https://huggingface.co/datasets/EleutherAI/pile-deduped-pythia-preshuffled/"
       "resolve/main/document-{:05d}-of-00020.bin")
SHARD_BYTES = 30_000_000_000
ROW_BYTES = 2049 * 2
N_ROWS = 500_000
CHUNK = 16 << 20
WORKERS = 4

# Two distant samples, each wholly inside one shard so no shard-boundary logic is needed.
SPLITS = {
    "A": dict(start_row=1_000_000, shard=0),
    "B": dict(start_row=73_300_000, shard=10),
}


def fetch(url, start, end, retries=6):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                buf = r.read()
            if len(buf) == end - start + 1:
                return buf
        except Exception as e:  # transient CDN / range failures are common at this size
            print(f"  retry {k} @{start}: {e}", flush=True)
    raise RuntimeError(f"range {start}-{end} failed")


def download(name, spec):
    out = os.path.join(DATA, f"split{name}.bin")
    ledger = out + ".done.json"
    nbytes = N_ROWS * ROW_BYTES
    local0 = spec["start_row"] * ROW_BYTES - spec["shard"] * SHARD_BYTES
    assert local0 >= 0 and local0 + nbytes <= SHARD_BYTES, "sample must sit inside one shard"
    url = URL.format(spec["shard"])
    offs = list(range(0, nbytes, CHUNK))
    try:
        got = set(json.load(open(ledger)))
    except Exception:
        got = set()
    if not os.path.exists(out) or os.path.getsize(out) != nbytes:
        with open(out, "wb") as f:
            f.truncate(nbytes)
        got = set()
    todo = [o for o in offs if o not in got]
    print(f"split{name}: {len(got)}/{len(offs)} chunks already done", flush=True)

    def one(off):
        n = min(CHUNK, nbytes - off)
        buf = fetch(url, local0 + off, local0 + off + n - 1)
        with open(out, "r+b") as f:
            f.seek(off)
            f.write(buf)
        return off

    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for fut in cf.as_completed([ex.submit(one, o) for o in todo]):
            got.add(fut.result())
            tmp = ledger + ".tmp"  # atomic: a torn ledger silently re-downloads the whole split
            with open(tmp, "w") as f:
                json.dump(sorted(got), f)
            os.replace(tmp, ledger)
            print(f"split{name}: {len(got)}/{len(offs)} chunks", flush=True)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    for name, spec in SPLITS.items():
        download(name, spec)
    meta = {
        "source": "EleutherAI/pile-deduped-pythia-preshuffled",
        "idx_verified": {"magic": "MMIDIDX", "version": 1, "dtype_code": 8, "dtype": "uint16",
                         "n_sequences": 146432000, "seq_len": 2049,
                         "idx_bytes_predicted": 34 + 12 * 146432000 + 8, "idx_bytes_actual": 1757184042,
                         "bin_bytes_actual": 600078336000,
                         "bin_bytes_predicted": 146432000 * 2049 * 2},
        "splits": {k: dict(v, n_rows=N_ROWS, tokens=N_ROWS * 2049,
                           byte_start=v["start_row"] * ROW_BYTES) for k, v in SPLITS.items()},
    }
    with open(os.path.join(ROOT, "results", "corpus_manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done", flush=True)
