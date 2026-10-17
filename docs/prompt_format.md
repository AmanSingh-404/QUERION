# Prompt format (frozen in Phase 2, Step 2.6)

## Text-to-SQL example (fine-tuning and inference)

`<|bos|><|schema|>{schema}<|question|>{question}<|sql|>{sql}<|eos|>`

At inference the prompt ends at `<|sql|>`, and the model generates SQL until `<|eos|>`.

## Rules

- Schema: only CREATE TABLE / CREATE VIEW statements, one statement per line, no
  trailing semicolons. INSERT statements and sample data are never included.
- Every part (schema statements, question, SQL) goes through
  `tokenizer.normalize.normalize`, the same normalization as the tokenizer corpus.
- SQL targets have trailing semicolons removed.
- Fine-tuning loss is computed only on the SQL tokens and `<|eos|>` (Phase 7).
- Context length is 2048 tokens. Examples that do not fit are dropped, never truncated.

## Pretraining documents

`<|bos|>{text}<|eos|>`, documents concatenated and cut into 2048-token windows.
No attention masking across document boundaries (a known simplification).

## Reserved for execute-and-retry (Phase 10)

`...<|sql|>{failed_sql}<|error|>{error message}<|sql|>{fixed_sql}<|eos|>`

## Code

`data/prompt.py` (`format_prompt`, `format_example`), tests in `tests/test_prompt.py`.