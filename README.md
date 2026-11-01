# QUERION

A 100M-parameter text-to-SQL LLM built from scratch, with its own tokenizer and inference engine

Ask a question in plain English, get SQL that runs. Everything here was built and trained
on a single laptop GPU (RTX 4050, 6 GB): the tokenizer, the data pipeline, the transformer,
pretraining, fine-tuning, the evaluation harness and the decoding engine.

Weights: https://huggingface.co/SinghBro/querion-100m-sql

## Results

Spider dev (1,034 questions), plain execution accuracy on the original dev databases.
Every row is produced by a script (`scripts/collect_benchmarks.py`) from saved
per-question records and written to `benchmarks/accuracy.json`.

| Stage | Execution accuracy | Matches |
|---|---|---|
| Pretrained, zero-shot, greedy | 4.8% | 50 of 1034 |
| + SFT | 17.7% | 183 of 1034 |
| + SFT with identifier augmentation | 21.5% | 222 of 1034 |
| + schema-constrained decoding | 26.2% | 271 of 1034 |
| + 8 samples with execution-guided vote | **35.3%** | 365 of 1034 |

- The last row runs candidate queries on the target database and keeps the result most
  candidates agree on, so it needs database access at inference time. Gold answers are
  never used.
- This is not Spider's official test-suite accuracy (which is stricter), so it is not
  comparable to leaderboard scores.
- Standard error is about 1.5 points. Sampling settings were tuned on 16 held-out
  databases, not on dev. Details and error analysis: `docs/results.md`.

## How it works

1. **Tokenizer.** 24k byte-level BPE trained on SQL and English (see below).
2. **Data.** Cleaned and deduplicated SQL files, execution-verified text-to-SQL pairs and
   a synthetic generator, packed into binary shards. Spider dev is locked and checked for
   leakage.
3. **Model.** 101,026,560 parameters: 12 layers, grouped-query attention (12 query heads,
   4 KV heads), RoPE, RMSNorm, SwiGLU, tied embeddings, 2048-token context. Logits match
   HuggingFace's Llama implementation exactly.
4. **Pretraining.** 2B tokens (59% FineWeb-Edu, 40% SQL from The Stack, 1% text-to-SQL
   pairs) in 43.3 compute hours. Design choices backed by small-scale ablations.
5. **Fine-tuning.** 130,748 examples from Spider train, Gretel and synthetic pairs, loss on
   the SQL tokens only, with identifier augmentation on the Spider schemas.
6. **Inference engine.** KV cache decoding, schema-constrained decoding (every word must be
   an SQL word, a schema word or an alias), and batched sampling with an execution-guided
   vote.

## Limitations

- A small model with little multi-table training data. Most remaining failures pick the
  wrong table: 304 of the 497 unknown-column errors in the constrained run name a column
  from a table the query does not use.
- It reads schemas it has seen far better than new ones (91% on training databases vs 18%
  on unseen ones in an earlier probe).
- English only, SQLite dialect, and no retry on database error messages yet.
- Do not run generated SQL on a database you cannot afford to change. The evaluation uses
  read-only connections.

## Repository

```
tokenizer/   BPE tokenizer and tests          model/     transformer
data/        cleaning, synthetic data, prompt  train/     trainer, SFT, augmentation
eval/        executor, comparator, Spider      infer/     KV cache, constraint, sampling
scripts/     one-command entry points          benchmarks/  machine-generated results
docs/        results, experiments, model card  tests/     unit tests
```

Version history is in `CHANGELOG.md`.

## Tokenizer

A 24,000-token byte-level BPE tokenizer trained from scratch on SQL and English text.
Measured on 1,000 held-out text-to-SQL examples (the `gretelai/synthetic_text_to_sql`
test split, never used for training), 484,665 characters in total.

| Tokenizer | Vocab size | Tokens | Characters per token | QUERION saving |
|---|---|---|---|---|
| QUERION | 24,000 | 129,235 | 3.75 | - |
| GPT-2 | 50,257 | 144,065 | 3.36 | 10.3% |
| Llama 2 | 32,000 | 163,328 | 2.97 | 20.9% |
| Llama 3 | 128,256 | 128,005 | 3.79 | -1.0% |

Notes: the comparison is in-domain, since QUERION's tokenizer is trained on this kind of
text and the others are general-purpose. Llama 3 is about equal while using a vocabulary
5.3x larger. At 100M parameters that matters: a 128k vocabulary would spend about 98M
parameters on embeddings alone (128,256 x 768), versus about 18M for QUERION.