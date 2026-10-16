# Data sources

| Source | Used for | License (re-verify on the dataset card in Phase 2.1) | Notes |
|---|---|---|---|
| b-mc2/sql-create-context | Tokenizer corpus | CC-BY-4.0 | Built from WikiSQL and Spider. Do NOT use for training unless decontaminated against eval sets |
| gretelai/synthetic_text_to_sql | Tokenizer corpus | Apache-2.0 | 100,000 train rows |
| HuggingFaceFW/fineweb-edu (sample-10BT) | Tokenizer corpus | ODC-By 1.0 | First 30,000 documents |

## Tokenizer corpus
- File: corpus.txt on Drive, about 214 MB
- 78,577 + 100,000 SQL rows and 30,000 English documents
- Eval sets (Spider dev, WikiSQL test, custom 300) are never used for training