---
language:
- en
pipeline_tag: text-generation
tags:
- text-to-sql
- sql
- from-scratch
- custom-architecture
---

# QUERION 100M (v0.6.0)

A 101,026,560-parameter decoder-only language model trained from scratch for
text-to-SQL, on a single laptop GPU (RTX 4050, 6 GB). Code, training logs and
evaluation harness: https://github.com/AmanSingh-404/QUERION

## Architecture

12 layers, grouped-query attention (12 query heads, 4 KV heads), RoPE, RMSNorm,
SwiGLU, tied input/output embeddings. Context length 2048. Custom 24k byte-level BPE
tokenizer (`tokenizer.json`).

## Training

- Pretraining: 2B tokens (59% FineWeb-Edu, 40% SQL from The Stack, 1% verified
  text-to-SQL pairs), 43.3 compute hours.
- Fine-tuning: 1,597 steps over 130,748 examples from Spider train, Gretel and
  synthetic pairs, with loss on the SQL tokens only. Spider schemas are augmented with
  identifier variants (case style, plural flips, optional quote stripping, table order).

## Evaluation

Spider dev, 1,034 questions. Metric: plain execution accuracy on the original dev
databases (read-only SQLite). This is not Spider's official test-suite accuracy, which is
stricter, so it is not comparable to leaderboard scores.

| Decoding | Execution accuracy |
|---|---|
| Pretrained model, zero-shot, greedy | 4.84% |
| Fine-tuned, no identifier augmentation, greedy | 17.7% |
| This model, greedy | 21.5% |
| + schema-constrained decoding | 26.2% |
| + 8 samples with execution-guided vote | 35.3% |

The last row runs candidate queries on the target database and returns the result most
candidates agree on, so it needs database access at inference time. Gold answers are never
used. The constrained decoding and the vote are in the `infer/` package of the GitHub repo.
Standard error is about 1.5 points. Most remaining failures are wrong table choices: with
the constraint on, 304 of 497 unknown-column errors name a column from a table the query
does not use.

## Prompt format

```
<|bos|><|schema|>{CREATE TABLE statements, one per line}<|question|>{question}<|sql|>
```

The model generates SQL until `<|eos|>`.

## Usage

The architecture is custom, so it does not load through `transformers`. Use the
`model/` package from the GitHub repo:

```python
from safetensors.torch import load_file
from model.config import QUERION_100M
from model.transformer import Transformer

model = Transformer(QUERION_100M)
model.load_state_dict(load_file("model.safetensors"), strict=False)  # lm_head is tied
```

`scripts/eval_spider.py` in the repo shows full generation and scoring.

## Limitations

A small model with little multi-table training data. It reads schemas it has seen
far better than new ones (91% vs 18% in an earlier probe), often invents column names,
and is not a general-purpose assistant. Do not run its SQL on databases you cannot
afford to change, and always run it on read-only connections first.