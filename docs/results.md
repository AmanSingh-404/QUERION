# QUERION results (v0.7.0)

QUERION is a 101,026,560-parameter decoder-only language model trained from scratch
for text-to-SQL: custom 24k BPE tokenizer, pretraining on a laptop RTX 4050, supervised
fine-tuning, and a custom inference engine. The weights are unchanged since v0.6.0;
v0.7.0 adds the inference engine.

## Training

- Pretraining: 2B tokens, mix 59% FineWeb-Edu, 40% SQL from The Stack, 1% verified
  text-to-SQL pairs. 43.3 compute hours. Final validation loss 2.2786 on the planned mix.
- Fine-tuning: 1,597 steps over 2 epochs on 130,748 examples, loss on SQL tokens only.
  Sources: Spider train, Gretel (verified by execution), and synthetic pairs.
  Spider schemas are augmented with identifier variants (case style, plural flips,
  optional quote stripping, table order shuffle), 3 variants per example.

## Spider dev (1,034 questions)

Metric: plain execution accuracy on the original dev databases (read-only SQLite,
row order ignored unless the gold query has ORDER BY). This is not Spider's official
test-suite accuracy, which is stricter.

| Stage | Execution accuracy |
|---|---|
| Pretrained, zero-shot, greedy | 4.84% |
| + SFT | 17.7% (183) |
| + SFT with identifier augmentation (v0.6.0 weights) | 21.5% (222) |
| + schema-constrained decoding | 26.2% (271) |
| + 8 samples with execution-guided vote (v0.7.0 default) | 35.3% (365) |

Standard error is about 1.5 points or less. The last stage runs candidate queries on the
target database, so it needs database access at inference time. Gold answers are never
used. It is not comparable to systems that only emit SQL.

## Inference engine

- KV cache: predictions identical to uncached decoding (0 of 100 differ), 1.9x faster
  (73 s vs 142 s per 100 questions). At batch size 1 the model is launch-bound.
- Schema-constrained decoding: every word must be an SQL word, a word from the schema,
  a T1-style alias, or an alias the query defines with AS. Removes invented table names
  and the glued-keyword bug (`FROMconcert`).
- Sampling and vote: 8 candidates decoded in one batch (row 0 greedy, the rest sampled at
  temperature 0.7), each run read-only with a 5 s timeout. Candidates that error or return
  nothing are dropped, and the most common result wins (ties go to the greedy candidate).
  8 samples cost about 1.5x the time of greedy. A full dev run takes about 20 minutes.

## Tuning

Sampling settings were chosen on 300 questions from 16 Spider-train databases held out of
fine-tuning, not on dev. Design decisions up to constrained decoding were informed by dev
errors, so dev numbers are slightly optimistic.

| Setting | Held-out matches (of 300) |
|---|---|
| greedy + constraint | 54 (18.0%) |
| 8 samples, temp 0.7 (chosen) | 79 (26.3%) |
| 16 samples, temp 0.7 | 83 (27.7%), 1.4x the time |
| 8 samples, temp 1.0 | 72 (24.0%) |
| 16 samples, temp 1.0 | 74 (24.7%) |
| 16 samples, temp 0.5 | 76 (25.3%) |

## Findings

- Generalization gap: the SFT model scored 91% on 100 questions from Spider training
  databases and 18% on 100 evenly spread dev questions.
- Identifier augmentation gave +3.8 points on the full dev set. A 100-question sample
  showed no difference (18% vs 18%), within its noise of about 4 points.
- Constrained decoding gave +4.7 points. Of 497 unknown-column errors that remained,
  304 name a column from a table the query does not use, 173 use a column on the wrong
  alias of a table the query does use, and 20 use a table name as a column.
  Wrong table choice is the main limit.
- The vote gave +9.1 points on dev and +8.3 on the held-out probe (18.0% to 26.3%).
  Of the remaining dev failures, 371 queries run with a wrong answer and 296 never run
  (258 with an unknown column).
- Name repair after generation (nearest real name, retry) added only 2 to 3 points on
  100 questions. The code remains in `eval/repair.py`.

## Limitations

- Small model, with little multi-table training data (about 6k Spider examples).
- Wrong table selection is not fixed by decoding rules.
- Execution feedback with error messages (retry) is not implemented yet.
- Numbers are not comparable to official Spider leaderboard scores.