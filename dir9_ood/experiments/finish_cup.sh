#!/usr/bin/env bash
# Robust finisher: wait for ANY cupenv pip to exit, verify the real cupbearer imports,
# then run the evaluation. Tolerates the very slow ceph FS (long wait window).
cd /mars-vol/marsv/dir9_ood
CUPENV=/mars-vol/marsv/dir9_ood/cupenv
SRC=/mars-vol/marsv/dir9_ood/vendor/cupbearer-main/src
echo "=== waiting for cupenv pip to finish (up to ~50 min) ==="
for i in $(seq 1 600); do
  if ! pgrep -f "cupenv/bin/pip" >/dev/null 2>&1; then echo "pip done after ~$((i*5))s"; break; fi
  sleep 5
done
echo "=== versions (shared base must stay torch 2.9.0+cu130 / numpy 2.3.3; cupenv is separate) ==="
PYTHONPATH=$SRC $CUPENV/bin/python -c "
import numpy, torch
print('cupenv numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())
import datasets, transformers
print('datasets', datasets.__version__, 'transformers', transformers.__version__)
import cupbearer
from cupbearer.detectors import MahalanobisDetector, QuantumEntropyDetector, SpectralSignatureDetector
print('CUPBEARER_IMPORT_OK from', cupbearer.__file__)
" || { echo 'IMPORT_FAILED'; exit 1; }
echo "=== run cup_eval.py (REAL cupbearer detectors on precomputed GPT-2 activations) ==="
PYTHONPATH=$SRC $CUPENV/bin/python experiments/cup_eval.py
echo "=== EXIT $? ==="
