# Data sources

License check date: 2026-10-17.
"Verified" means confirmed against a published source on that date.
"Verify" means re-check the dataset card before first use (Step 2.2).

## Pretraining (target about 2B tokens)

| Source | URL | Share | License | Status |
|---|---|---|---|---|
| FineWeb-Edu (sample-10BT) | huggingface.co/datasets/HuggingFaceFW/fineweb-edu | about 58% (1.1B tokens) | ODC-By 1.0 | Verify |
| The Stack dedup, SQL subset | huggingface.co/datasets/bigcode/the-stack-dedup | about 40% (0.8B tokens, up to 2 epochs) | Per-file permissive licenses, Terms of Use apply | Verified: gated, terms must be accepted |
| Gretel synthetic_text_to_sql (train) | huggingface.co/datasets/gretelai/synthetic_text_to_sql | about 2% | Apache-2.0 | Verified |

The SQL subset size of The Stack is measured in Step 2.2, and the mix is adjusted to match.

## Fine-tuning

| Source | URL | License | Status |
|---|---|---|---|
| Gretel synthetic_text_to_sql (train) | huggingface.co/datasets/gretelai/synthetic_text_to_sql | Apache-2.0 | Verified |
| WikiSQL (train) | github.com/salesforce/WikiSQL | BSD-3-Clause | Verify |
| Spider (train) | yale-lily.github.io/spider | CC BY-SA 4.0 (copyleft, derived data must stay compatible) | Verified (secondary source), confirm on the Spider page |
| Own synthetic generator | this repo | MIT | Step 2.5 |

## Evaluation (never trained on)

Spider dev, WikiSQL test, and a hand-written 300-question custom set.

## Excluded from training

b-mc2/sql-create-context is built from WikiSQL and Spider, so it would leak
eval questions. It is used only for the tokenizer corpus.

## Tokenizer corpus (done, Phase 1)

- File: corpus.txt on Drive, about 214 MB
- b-mc2/sql-create-context (78,577 rows, CC-BY-4.0, verify), gretelai/synthetic_text_to_sql
  (100,000 rows, Apache-2.0), FineWeb-Edu sample-10BT (first 30,000 documents)