# Changelog

## v0.6.0 - Phases 6-8: Pretraining, fine-tuning, evaluation
- Main pretraining run: 2B tokens in 43.3 compute hours on the RTX 4050, final validation loss 2.2786 on the planned mix
- Evaluation harness: read-only SQLite executor with timeout and row cap, order-aware result comparator, Spider dev pipeline. The pretrained model scores 4.84% execution accuracy zero-shot
- Supervised fine-tuning with loss on SQL tokens only: 1,597 steps over 130,748 examples from Spider, Gretel and synthetic pairs
- Identifier augmentation for Spider schemas (case styles, plural flips, quote stripping, table shuffle): Spider dev execution accuracy 17.7% to 21.5% (1,034 questions, original dev databases)
- Error analysis: 91% on Spider training databases vs 18% on dev (a generalization gap), 533 of 674 failures are unknown-column errors, glued-keyword bug traced to quoted schemas. Name repair adds 2 to 3 points and is not in the headline numbers
- Weights exported as fp32 safetensors (404 MB), every tensor verified identical after reload. Full write-up in docs/results.md

## v0.5.0 - Phase 5: Ablations
- Nine runs of a 19.5M-parameter proxy model on 100M tokens each: GQA vs MHA (tie within noise 0.007), RoPE vs learned positions (RoPE by 0.80), data mix (kept 59/40/1), learning rate sweep (best 3e-3, unstable at 6e-3)
- Final main-run configuration frozen in docs/final_config.md: peak LR 1.5e-3, 262,144 tokens per step, 7,630 steps
- Throughput watchdog that warns when training slows down, for example on battery power

## v0.4.0 - Phase 4: Training loop
- Resumable trainer: stateless mixed data loader (59% FineWeb-Edu, 40% SQL files, 1% pairs), bf16 autocast, AdamW with warmup and cosine decay, gradient accumulation, atomic checkpoints with automatic fallback, W&B plus local JSONL logging
- A run that is stopped and resumed ends with the same weights as an uninterrupted run (tested)
- Dry run on the real data on the RTX 4050 (500 steps, 8M tokens): validation loss 7.06 to 5.66, about 12.2k tokens/s, 3.6 GB GPU memory. Resuming after Ctrl+C continued with no jump
- Estimated full run: about 46 hours for 2.0B tokens

## v0.3.0 - Phase 3: Model code
- 101,026,560-parameter decoder-only transformer (12 layers, grouped-query attention with 12 query and 4 KV heads, RoPE, RMSNorm, SwiGLU, tied embeddings) with 14 unit tests
- Logits match HuggingFace's Llama implementation exactly (max difference 0.0) when weights are copied over
- Overfits a fixed batch from loss 10.2 to 0.0005 in 200 steps on the RTX 4050 in bf16
- Profiled on the RTX 4050 (6 GB): 13.6k tokens/s at micro-batch 1 without checkpointing, 2.4 GB, about 41 hours estimated for 2B tokens

## v0.2.0 - Phase 2: Data pipeline
- Cleaned (75.7% kept) and deduplicated 1.5 GB of SQL files, verified 67.9% of Gretel pairs by execution, generated 40,000 execution-verified synthetic pairs
- Frozen prompt format and a resumable tokenize-to-shards pipeline: 1.51B unique train tokens in uint16 shards
- Locked Spider dev (1,034 items). The leakage check removed 1 contaminated Stack file and 38 synthetic pairs
- Shard integrity check: every token count in the copied files matches the recorded summaries

## v0.1.0 - Phase 1: Tokenizer
- Normalized 214 MB SQL and English corpus, 8 fixed special tokens
- 24k byte-level BPE tokenizer with 11 tests, shipped as tokenizer/tokenizer.json
- 10% fewer tokens than GPT-2 and 21% fewer than Llama 2 on held-out text-to-SQL; within 1% of Llama 3 using a 5x smaller vocabulary

## v0.0.1 - Phase 0: Setup and workflow
- Repo, folder layout, linting hooks, CI
- Colab and Drive workflow, W&B and Hugging Face logins
- T4 baseline: 24 TFLOPs fp16, 232 GB/s memory bandwidth
- Phase issues, milestones and project board

