# Final training configuration (main run)

| Item | Value |
|---|---|
| Model | 101,026,560 parameters: 12 layers, width 768, 12 query / 4 KV heads, RoPE, SwiGLU 2304, vocab 24,000, context 2048 |
| Data mix | 59% FineWeb-Edu, 40% Stack SQL, 1% pairs (1.18B / 0.8B / 0.02B tokens; 1.0 / 2.5 / 2.45 epochs) |
| Batch | 262,144 tokens per step (micro-batch 1 x 128 accumulation x 2048) |
| Steps | 7,630 (2.0B tokens) |
| Optimizer | AdamW, betas 0.9 / 0.95, weight decay 0.1 (weight matrices only), gradient clip 1.0 |
| Learning rate | peak 1.5e-3, 300 steps linear warmup, cosine decay to 1.5e-4 |
| Precision | bf16 autocast, fp32 weights and loss |
| Checkpoints | every 40 steps (about 14 minutes), latest and previous, milestone every 2,500 steps |
| Evaluation | every 250 steps, 16 fixed windows per source, plus two text samples |
| Speed | about 12,200 tokens/s, about 21.5 s per step, about 45.5 hours |

Launch: `C:\querion-env\Scripts\python.exe -m train.train --wandb --run-name main`

Resume after any interruption: run the same command again.

If the loss spikes or diverges in the first 500 steps: press Ctrl+C and run the same command
with `--lr 1e-3`. The schedule is recomputed from the flag and training continues from the
last checkpoint.

The laptop must stay plugged in. On battery it ran about 17x slower.