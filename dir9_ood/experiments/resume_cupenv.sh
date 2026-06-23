#!/usr/bin/env bash
# Resume building the ISOLATED cupbearer env (build_cupenv.sh was interrupted mid torch install).
# numpy<2 already installed; finish torch cu130 + cupbearer GitHub editable install. Shared base
# env (torch 2.9.0+cu130 / numpy 2.3.3) is untouched -- everything goes in $ENV only.
set -eo pipefail
ENV=/mars-vol/marsv/dir9_ood/cupenv
REPO=/mars-vol/marsv/dir9_ood/vendor/cupbearer-main
PY="$ENV/bin/python"
echo "=== [1] torch cu130 (isolated, resume) ==="
"$PY" -m pip install --no-input torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130 || exit 12
echo "=== [2] cupbearer editable from GitHub clone (deps may pull numpy<2-compatible) ==="
"$PY" -m pip install --no-input -e "$REPO" || exit 13
echo "=== [3] verify ==="
"$PY" - <<'PYEOF'
import numpy, torch
print("numpy", numpy.__version__, "torch", torch.__version__, "cuda_build", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.randn(64, 64, device="cuda"); print("matmul ok", float((x@x).sum()) is not None)
import cupbearer
print("cupbearer", getattr(cupbearer, "__version__", "?"), "from", cupbearer.__file__)
from cupbearer.detectors import MahalanobisDetector, QuantumEntropyDetector, SpectralSignatureDetector
print("detectors import ok")
PYEOF
echo "=== DONE BUILD_OK ==="
