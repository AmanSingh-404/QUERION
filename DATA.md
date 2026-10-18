# Data sources

License check date: 2026-10-17.
"Verified" means confirmed against a published source on that date.
"Verify" means re-check the dataset card before relying on it.
Measured token counts are in docs/data_stats.md.

## Pretraining (about 2.0B training tokens)

| Source | URL | Unique train tokens | Planned tokens | Epochs | License | Status |
|---|---|---|---|---|---|---|
| FineWeb-Edu (sample-10BT) | huggingface.co/datasets/HuggingFaceFW/fineweb-edu | 1,180,011,340 | 1.18B | 1.00 | ODC-By 1.0 | Verify |
| The Stack dedup, SQL subset | huggingface.co/datasets/bigcode/the-stack-dedup | 319,286,475 | 0.80B | 2.51 | Per-file permissive licenses, Terms of Use apply | Verified: gated, terms must be accepted |
| Pairs: Gretel train (verified) + synthetic | huggingface.co/datasets/gretelai/synthetic_text_to_sql, own generator | 8,158,019 | 0.02B | 2.45 | Apache-2.0, MIT | Verified |

## Processing summary

- Stack SQL: 225,879 files downloaded, 171,044 after cleaning (75.7%), 170,640 after
  deduplication (0.2% removed), 170,639 after decontamination.
- Gretel train: 100,000 pairs, 67,852 kept (67.9%) after executing each SQL against its own
  context in SQLite. Non-SELECT tasks and SQL that fails to run were dropped.
- Synthetic: 40,000 pairs generated and verified by execution, 38 removed because the
  question matched a Spider dev question, 39,962 kept.
- Total pairs in training: 107,814.

## Evaluation (never trained on)

- Spider dev: 1,034 items, locked in eval/spider_dev.jsonl (sha256 in eval/MANIFEST.json).
  CC BY-SA 4.0.
- WikiSQL test and a hand-written custom set: added in Phase 8, and the leakage check
  (python -m data.leakage) is re-run when they are added.
- Leakage check result: 1 Stack file (a Spider-derived test.sql) and 38 synthetic questions
  matched eval text and were excluded. Gretel had no matches.

## Fine-tuning (Phase 7)

| Source | URL | License | Status |
|---|---|---|---|
| Gretel synthetic_text_to_sql (train, verified) | huggingface.co/datasets/gretelai/synthetic_text_to_sql | Apache-2.0 | Verified |
| WikiSQL (train) | github.com/salesforce/WikiSQL | BSD-3-Clause | Verify |
| Spider (train) | yale-lily.github.io/spider | CC BY-SA 4.0 (copyleft, derived data must stay compatible) | Verified (secondary source), confirm on the Spider page |
| Own synthetic generator | this repo | MIT | Done |

## Excluded from training

b-mc2/sql-create-context is built from WikiSQL and Spider, so it would leak eval questions.
It was used only for the tokenizer corpus.

## Tokenizer corpus (Phase 1)

- corpus.txt on Drive, about 214 MB
- b-mc2/sql-create-context (78,577 rows, CC-BY-4.0, verify), gretelai/synthetic_text_to_sql
  (100,000 rows, Apache-2.0), FineWeb-Edu sample-10BT (first 30,000 documents)

## Known limitations

- The Stack sample is the first 1.5 GB of the stream, not a random sample.
- Long files dominate the SQL tokens: documents over 2048 tokens are 18% of the documents
  but 78% of the tokens.
- Synthetic pairs come from only about 370 distinct schemas, so they teach SQL mechanics, not
  domain breadth.
- Gretel is LLM-generated. About a third of it was dropped, and what remains can still be
  semantically wrong even if it executes.
- FineWeb-Edu was not scanned for eval leakage (general web text).
- Spider is CC BY-SA 4.0, so datasets derived from it must stay under a compatible license.