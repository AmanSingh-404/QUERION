# Error analysis (v0.7.0)

Final system on Spider dev: schema-constrained decoding, 8 samples, execution-guided vote
(35.3%, 365 of 1,034). 667 questions fail. Two more are excluded because their gold query
does not run. Labels come from rules (`eval/taxonomy.py`, `scripts/error_taxonomy.py`) that
compare each failing query with the gold query. Counts are in
`benchmarks/error_taxonomy.json`.

## Where the failures are

| Cause | Failures | Share |
|---|---|---|
| Wrong tables or joins (the query runs but uses other tables than the gold query) | 219 | 32.8% |
| Column from a table the query does not use | 172 | 25.8% |
| Column used on the wrong alias | 73 | 10.9% |
| Unknown table | 13 | 1.9% |
| **Table and join choice, together** | **477** | **71.5%** |
| Same structure, different result | 42 | 6.3% |
| Wrong selected columns | 39 | 5.8% |
| Wrong filter | 30 | 4.5% |
| Syntax error | 23 | 3.4% |
| Wrong aggregate | 22 | 3.3% |
| Invented column name | 13 | 1.9% |
| Wrong query shape (set operation or subquery) | 11 | 1.6% |
| Wrong grouping, ordering or limit | 8 | 1.2% |
| Other | 2 | 0.3% |

## What it shows

- The model gets the mechanics of SQL right. Aggregates, filters, grouping and query shape
  together account for 71 failures (10.6%).
- Most failures are schema linking: choosing the right tables and joining them correctly.
  The first category also includes some valid alternative formulations, for example a
  subquery where the gold query uses a join, so 71.5% is an upper estimate.
- The execution vote cannot fix a table choice that most candidates share. That is why its
  gain (+9 points) is smaller than the size of this group.

## How reliable the labels are

I read 40 random failures (seed 0) next to their gold and predicted SQL. 30 labels were
right as stated, 8 were right but too coarse, and 2 were wrong.

- Too coarse: 3 wrong join keys and 1 wrong sort column (both land in "same structure"),
  1 extra DISTINCT, 1 defensible alternative reading of the question, 1 ambiguous between
  wrong alias and wrong table, 1 whose label name is misleading (right table, column
  borrowed from another table).
- Wrong: 1 "wrong alias" that was really a wrong join table, and 1 "same structure" that
  was a wrong filter column.

So the group sizes above are reliable to within a few points. The "same structure" group
(42) hides join-key and filter-column errors that the rules do not look at.

## Tried, did not help

- Replacing an unknown table or column name with the closest real name after generation:
  +2 to 3 points on 100 questions, and superseded by constrained decoding.
- Trimming stray spaces inside string literals (`" Ward"` becomes `"Ward"`): 11 of 1,034
  final predictions changed, 0 additional matches. Code kept in `eval/tidy.py`, not used.

## Known issues, not fixed

- The word constraint allows DDL words from the schema text (for example `CREATE`)
  anywhere, which produced one syntax error in the sample. At most it affects the 23 syntax
  errors. The grammar-aware constraint planned for Phase 10 replaces it.
- There is no retry on database error messages.

## What would help

More multi-table training data or a larger model, to improve table choice. Decoding rules
cannot add that knowledge.