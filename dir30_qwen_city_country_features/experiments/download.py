"""Download the pinned Qwen model and per-layer transcoder weights (no feature-visualisation binaries)."""
from huggingface_hub import snapshot_download

print(snapshot_download("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca"))
print(snapshot_download("mwhanna/qwen3-0.6b-transcoders-lowl0",
                        revision="28aefe686c09e9bc1a862195b51e145f10868d87",
                        allow_patterns=["*.safetensors", "*.yaml", "README.md"], max_workers=4))
