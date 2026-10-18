# Changelog


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

