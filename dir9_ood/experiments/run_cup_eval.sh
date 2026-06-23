#!/usr/bin/env bash
# Wait for the cupenv deps install to settle, then run the REAL cupbearer evaluation.
cd /mars-vol/marsv/dir9_ood
CUPENV=/mars-vol/marsv/dir9_ood/cupenv
SRC=/mars-vol/marsv/dir9_ood/vendor/cupbearer-main/src
# Wait until the dep install process is gone (no pip running for cupenv).
for i in $(seq 1 120); do
  if ! pgrep -f "cupenv/bin/pip" >/dev/null 2>&1 && ! pgrep -f "cupenv/bin/python -m pip" >/dev/null 2>&1; then break; fi
  sleep 5
done
echo "=== import check ==="
PYTHONPATH=$SRC $CUPENV/bin/python -c "
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import cupbearer; print('cupbearer', cupbearer.__file__)
from cupbearer.detectors import MahalanobisDetector, QuantumEntropyDetector, SpectralSignatureDetector
print('CUPBEARER_IMPORT_OK')
" || { echo "IMPORT_FAILED"; exit 1; }
echo "=== run cup_eval.py (real cupbearer detectors) ==="
PYTHONPATH=$SRC $CUPENV/bin/python experiments/cup_eval.py
echo "=== EXIT $? ==="
