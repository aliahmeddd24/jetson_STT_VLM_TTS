#!/usr/bin/env bash
# Phase 1: CUDA sanity inside an r35 container with --runtime nvidia (no host changes).
set -euo pipefail
IMG="${1:-g1voice/vlm:r35}"
docker run --rm --runtime nvidia --network none --entrypoint python3 "$IMG" - <<'PY'
import time, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
assert torch.cuda.is_available(), "CUDA not available"
p = torch.cuda.get_device_properties(0)
print("device", p.name, "sm_%d%d" % (p.major, p.minor), "mem %.1f GB" % (p.total_memory / 1e9), "SMs", p.multi_processor_count)
a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): b = a @ a
torch.cuda.synchronize(); dt = time.time() - t
print("fp16 matmul %.1f TFLOP/s" % (20 * 2 * 2048**3 / dt / 1e12))
print("PASS" if (p.major, p.minor) == (8, 7) else "UNEXPECTED ARCH")
PY
