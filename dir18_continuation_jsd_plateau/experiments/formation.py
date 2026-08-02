"""Optional formation subset: does the JSD -> sharpness relationship strengthen during training?

Runs the SAME frozen 75-pair bank at four intermediate pythia-1.4b-deduped checkpoints. Each
checkpoint is fetched to local disk, assayed, and deleted before the next one, so peak disk stays at
one checkpoint (the shared network volume cannot hold four).
"""
import json
import os
import shutil
import subprocess
import sys

STEPS = ["step1000", "step8000", "step32000", "step64000"]
CACHE = "/tmp/hf_formation"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    for rev in STEPS:
        tag = os.path.join(ROOT, "results", f"assay_{rev}.json")
        if os.path.exists(tag):
            print(f"{rev} already done", flush=True)
            continue
        env = dict(os.environ, HF_HOME=CACHE, HF_HUB_DISABLE_XET="1")
        print(f"=== {rev}: downloading", flush=True)
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys;from huggingface_hub import snapshot_download;"
             "snapshot_download('EleutherAI/pythia-1.4b-deduped', revision=sys.argv[1],"
             " allow_patterns=['*.json','model.safetensors','*.txt'])", rev],
            env=env, capture_output=True, text=True)
        if r.returncode:
            print(f"{rev}: DOWNLOAD FAILED\n{r.stderr[-800:]}", flush=True)
            shutil.rmtree(CACHE, ignore_errors=True)
            continue
        print(f"=== {rev}: assaying", flush=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "run_assay.py"), rev],
                           env=env, capture_output=True, text=True, cwd=ROOT)
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-800:], flush=True)
        shutil.rmtree(CACHE, ignore_errors=True)
    print("FORMATION_DONE", flush=True)
