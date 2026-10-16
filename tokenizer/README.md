# Tokenizer

Byte-level BPE, 24,000 tokens, trained on a normalized SQL + English corpus.

## Special tokens (fixed IDs)

| ID | Token | Purpose |
|---|---|---|
| 0 | `<|pad|>` | Padding |
| 1 | `<|bos|>` | Start of sequence |
| 2 | `<|eos|>` | End of sequence |
| 3 | `<|schema|>` | Starts the table schema |
| 4 | `<|question|>` | Starts the user question |
| 5 | `<|sql|>` | Starts the SQL answer |
| 6 | `<|error|>` | Starts an execution error message (retry loop) |
| 7 | `<|sep|>` | General separator |

## Prompt layout (frozen in Phase 2)

`<|bos|><|schema|>{schema}<|question|>{question}<|sql|>{sql}<|eos|>`