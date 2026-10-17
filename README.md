# QUERION
A 100M-parameter text-to-SQL LLM built from scratch, with its own tokenizer and inference engine

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