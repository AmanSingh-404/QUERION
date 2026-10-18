# Data statistics

All numbers are measured from the shard files.

## Token counts and training mix

| Source | Docs | Train tokens | Val tokens | Planned tokens | Epochs | Mix |
|---|---|---|---|---|---|---|
| fineweb | 1,095,802 | 1,180,011,340 | 6,039,043 | 1,180,000,000 | 1.00 | 59.0% |
| stack | 169,785 | 319,286,475 | 1,601,694 | 800,000,000 | 2.51 | 40.0% |
| pairs | 107,274 | 8,158,019 | 41,599 | 20,000,000 | 2.45 | 1.0% |

Unique train tokens: 1,507,455,834. Planned training tokens: 2,000,000,000.

## Document length (tokens, including BOS and EOS)

| Source | Mean | Median | P90 | P99 | Max | Docs over 2048 | Tokens in those |
|---|---|---|---|---|---|---|---|
| fineweb | 1077 | 653 | 2023 | 8230 | 165,201 | 9.8% | 41.9% |
| stack | 1881 | 440 | 3977 | 25407 | 150,511 | 18.2% | 78.0% |
| pairs | 76 | 71 | 113 | 160 | 405 | 0.0% | 0.0% |

## Length histogram (number of documents)

| Source | 0-128 | 128-256 | 256-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384+ |
|---|---|---|---|---|---|---|---|---|---|
| fineweb | 20,074 | 130,384 | 273,303 | 352,715 | 211,798 | 72,800 | 23,701 | 8,569 | 2,458 |
| stack | 22,120 | 34,998 | 34,443 | 27,197 | 20,118 | 14,428 | 8,872 | 4,539 | 3,070 |
| pairs | 103,307 | 3,933 | 34 | 0 | 0 | 0 | 0 | 0 | 0 |

## Integrity checks

All checks passed.
