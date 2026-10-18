import time

import torch

N = 4096
DURATION = 300
WINDOW = 30

a = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
b = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
for _ in range(10):
    a @ b
torch.cuda.synchronize()

start = window_start = time.time()
count = 0
while time.time() - start < DURATION:
    a @ b
    count += 1
    if count % 50 == 0:
        torch.cuda.synchronize()
        now = time.time()
        if now - window_start >= WINDOW:
            tflops = 2 * N**3 * count / (now - window_start) / 1e12
            print(f"{int(now - start):4d}s  {tflops:.1f} TFLOPs", flush=True)
            window_start, count = now, 0
