#!/usr/bin/env bash
# Build an ISOLATED conda env for cupbearer (operator feedback 2026-06-21).
# Must NOT touch the shared base env's torch 2.9.0+cu130 / numpy 2.3.3.
# A10 / driver 595 / CUDA 13 -> install torch cu130 inside the new env too.
set -eo pipefail
ENV=/mars-vol/marsv/dir9_ood/cupenv
REPO=/mars-vol/marsv/dir9_ood/vendor/cupbearer-main
echo "=== [1] create env $ENV ==="
/opt/conda/bin/mamba create -p "$ENV" python=3.11 pip -y
PY="$ENV/bin/python"
echo "=== [2] torch cu130 + numpy<2 (isolated) ==="
"$PY" -m pip install --no-input "numpy>=1.24,<2" || exit 11
"$PY" -m pip install --no-input torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130 || exit 12
echo "=== [3] cupbearer + deps from GitHub clone ==="
"$PY" -m pip install --no-input -e "$REPO" || exit 13
echo "=== [4] verify ==="
"$PY" - <<'PYEOF'
import numpy, torch
print("numpy", numpy.__version__, "torch", torch.__version__, "cuda_build", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.randn(64, 64, device="cuda"); print("matmul ok", (x@x).sum().item() is not None)
import cupbearer
print("cupbearer", getattr(cupbearer, "__version__", "?"), "from", cupbearer.__file__)
from cupbearer import detectors
print("detectors module ok:", [d for d in dir(detectors) if d[0].isupper()][:20])
PYEOF
echo "=== DONE BUILD_OK ==="
