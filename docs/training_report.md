# Pretraining report

| Item | Value |
|---|---|
| Steps | 7,630 |
| Tokens trained | 2,000,158,720 |
| Compute time (sum of step times) | 43.3 hours |
| Median speed | 13,012 tokens/s |
| Steps below half of median speed | 1 |
| Loss spikes after warmup (+0.5 over recent average) | 0 |
| Largest gradient norm after warmup | 3.10 |
| Training loss, first step / last 20 steps | 10.204 / 2.174 |

## Validation loss

| Step | Planned mix | FineWeb | Stack SQL | Pairs |
|---|---|---|---|---|
| 250 | 4.3630 | 5.2979 | 3.0315 | 2.4670 |
| 1,000 | 2.8308 | 3.9459 | 1.2312 | 1.0296 |
| 2,500 | 2.5233 | 3.5776 | 1.0112 | 0.8004 |
| 5,000 | 2.3560 | 3.3677 | 0.9056 | 0.6835 |
| 7,630 | 2.2786 | 3.2738 | 0.8518 | 0.6319 |
