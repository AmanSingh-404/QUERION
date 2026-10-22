# Ablation conclusions (Phase 5)

Setup: a 19.5M-parameter proxy model (6 layers, width 384), 100M tokens per run (1,526 steps
of 65,536 tokens), same trainer and data as the main run. All runs are compared on validation
loss re-weighted with the planned 59/40/1 mix, so the data-mix runs are comparable.
Full table: ablations.md. Curves: val_curves.png. Raw logs: logs/.

Noise floor: two identical runs with different seeds differ by 0.0071. Differences under about
0.014 (2x) are treated as ties.

## Decisions

1. Attention: keep grouped-query attention (12 query / 4 KV heads). Full multi-head attention
   was 0.006 better, inside the noise, and needs 3x the KV cache.
2. Positions: keep RoPE. Learned positions were 0.80 worse. At this short horizon the gap is
   probably exaggerated (position tables learn slowly), but RoPE also needs no extra
   parameters and handles other lengths better.
3. Data mix: keep 59% FineWeb-Edu / 40% SQL files / 1% pairs. Moving 18 points to SQL files
   improved Stack SQL loss by 0.095 but cost 0.126 on FineWeb, worsened the planned-mix loss
   by 0.036 (5x the noise floor), and did not improve the pairs loss when the pairs share was
   held at 1% (1.644 vs 1.650). The pairs gain in the first SQL-heavy run came from doubling
   the pairs share, which at full scale would mean about 5 passes over an 8M-token set.
4. Learning rate: peak 1.5e-3 for the 100M model. At proxy scale 3e-3 was best (0.087 better
   than 1e-3) and 6e-3 was far worse (+0.87, unstable), so the optimum sits just below an
   instability cliff. The 100M model is twice as wide and wider models tolerate lower rates,
   so its cliff should be near 3e-3. 1.5e-3 keeps a 2x margin.
5. Batch size: 262,144 tokens per step (7,630 steps for 2.0B tokens) instead of the 524,288
   first planned. Not tested by the ablations: a judgment call (more optimizer steps for the
   same tokens).

## Caveats

- One seed per setting, apart from the base run.
- Short horizon: 100M tokens against 2B. Slow starters (learned positions, low learning rate)
  may catch up later, so their gaps are probably overstated.
- Learning rates do not transfer perfectly across model size and batch size, so the main run
  uses a conservative value and the first 300-500 steps are watched closely.