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
stricter, so it is not comparable to leaderboard scores. Greedy decoding.

| Model | Execution accuracy |
|---|---|
| Pretrained, zero-shot | 4.84% |
| + SFT | 17.7% |
| + SFT with identifier augmentation (this model) | 21.5% |

Standard error is about 1.2 points. Most failures are schema errors: 533 of 674
failing predictions name a column that does not exist.

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