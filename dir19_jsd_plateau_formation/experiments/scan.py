"""Checkpoint scan driver: download one pythia-1.4b-deduped revision, assay it, delete it, repeat.

The shared volume cannot hold many checkpoints, so peak disk stays at one. Checkpoints already
assayed (results/assay_<rev>.json present) are skipped, so the scan is restartable.

Usage:  python3 scan.py step1 step2 step4 ...
"""
import os
import shutil
import subprocess
import sys
import time

CACHE = "/tmp/hf_dir19"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    for rev in sys.argv[1:]:
        if os.path.exists(os.path.join(ROOT, "results", f"assay_{rev}.json")):
            print(f"{rev}: already done", flush=True)
            continue
        env = dict(os.environ, HF_HOME=CACHE, HF_HUB_DISABLE_XET="1", MPLBACKEND="Agg")
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys;from huggingface_hub import snapshot_download;"
             "snapshot_download('EleutherAI/pythia-1.4b-deduped', revision=sys.argv[1],"
             " allow_patterns=['*.json','model.safetensors','*.txt'])", rev],
            env=env, capture_output=True, text=True)
        if r.returncode:
            print(f"{rev}: DOWNLOAD FAILED\n{r.stderr[-600:]}", flush=True)
            shutil.rmtree(CACHE, ignore_errors=True)
            continue
        print(f"{rev}: downloaded in {time.time()-t0:.0f}s, assaying", flush=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "run_assay.py"), rev],
                           env=env, capture_output=True, text=True, cwd=ROOT)
        print(f"{rev}: " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-600:]),
              flush=True)
        print(f"{rev}: total {time.time()-t0:.0f}s", flush=True)
        shutil.rmtree(CACHE, ignore_errors=True)
    print("SCAN_DONE", flush=True)
