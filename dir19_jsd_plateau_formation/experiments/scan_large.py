"""S5 validation: run the frozen 1,000-pair bank at the given revisions.

dir18 already assayed this bank at step0 and step143000 with identical definitions. We add
step64000 (completing the prespecified 64k -> final comparison) and the two checkpoints that define
the ordering onset bracket, step8 and step32.

Usage:  python3 scan_large.py step64000 step8 step32
"""
import os
import shutil
import subprocess
import sys
import time

CACHE = "/tmp/hf_dir19_large"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    for rev in sys.argv[1:]:
        if os.path.exists(os.path.join(ROOT, "results", f"assay_large_{rev}.json")):
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
        print(f"{rev}: download rc={r.returncode} in {time.time()-t0:.0f}s", flush=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "run_assay.py"), rev,
                            "--manifest", "pair_manifest_large.json", "--tag", "large_" + rev],
                           env=env, capture_output=True, text=True, cwd=ROOT)
        print(f"{rev}: " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip()
                            else r.stderr[-600:]), flush=True)
        shutil.rmtree(CACHE, ignore_errors=True)
        print(f"{rev}: total {time.time()-t0:.0f}s", flush=True)
    print("LARGE_DONE", flush=True)
