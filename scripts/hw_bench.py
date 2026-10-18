import json
import time

import torch


def bench_matmul(n=4096, dtype=torch.float16, iters=50):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = time.time() - t
    return 2 * n**3 * iters / dt / 1e12


def bench_bandwidth(mb=1024, iters=50):
    x = torch.empty(mb * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    for _ in range(5):
        y.copy_(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = time.time() - t
    return 2 * x.numel() * 4 * iters / dt / 1e9


props = torch.cuda.get_device_properties(0)
result = {
    "gpu": props.name,
    "vram_gb": round(props.total_memory / 1024**3, 2),
    "torch": torch.__version__,
    "fp16_matmul_tflops": round(bench_matmul(dtype=torch.float16), 1),
    "bf16_matmul_tflops": round(bench_matmul(dtype=torch.bfloat16), 1),
    "fp32_matmul_tflops": round(bench_matmul(dtype=torch.float32), 1),
    "memory_bandwidth_gbs": round(bench_bandwidth(), 1),
}
print(json.dumps(result, indent=2))
