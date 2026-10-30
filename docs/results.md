# QUERION results (v0.6.0)

QUERION is a 101,026,560-parameter decoder-only language model trained from scratch
for text-to-SQL: custom 24k BPE tokenizer, pretraining on a laptop RTX 4050, then
supervised fine-tuning.

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
test-suite accuracy, which is stricter. Greedy decoding.

| Model | Execution accuracy | Unknown column | Unknown table | Syntax error |
|---|---|---|---|---|
| Pretrained, zero-shot | 4.84% | | | |
| + SFT | 17.7% (183) | 537 | 112 | 73 |
| + SFT with identifier augmentation (v0.6.0) | 21.5% (222) | 533 | 75 | 66 |

About 1.2 points of standard error at this sample size.

## Findings

- Generalization gap: the SFT model (before augmentation) scored 91% on 100 questions
  from Spider training databases and 18% on 100 evenly spread dev questions. It reads
  schemas it has seen far better than new ones.
- Most failures are schema errors: in the final model 674 of 1,034 predictions fail to
  run, and 533 of those name a column that does not exist.
- Identifier augmentation gave +3.8 points on the full dev set. A 100-question sample
  showed no difference (18% vs 18%), which is within its noise of about 4 points.
- A name-repair step (replace an unknown table or column with the closest real name and
  retry) added 2 to 3 points on 100 questions. It is not part of the headline numbers.
- A bug found by error analysis: some predictions glue a keyword to a name
  (`FROMconcert`), caused by quoted Spider schemas in the training data.

## Limitations

- Small model, small amount of multi-table training data (about 6k Spider examples).
- No KV cache yet, so decoding is slow.
- Numbers are not comparable to official Spider leaderboard scores.