# Graph Predictor Validation Evidence Status

This directory intentionally preserves the original 36-row scaled-midpoint
batch.  A later stepwise review found that treating the whole batch as one
research milestone was too coarse.  The CSV and metadata have not been deleted,
rewritten, or promoted by this documentation change.

## Accepted selector

The current accepted milestone contains 12 graph-observation rows:

| Scale | Tail count | Layout | Graphs | Rows |
| ---: | ---: | --- | --- | ---: |
| `k=3`, `e=-27` | `7, 8, 9` | `tail_then_head` | sequential, pairwise | 6 |
| `k=3`, `e=-27` | `9` | `head_then_tail` | sequential, pairwise | 2 |
| `k=7`, `e=-31` | `129` | `tail_then_head` | sequential, pairwise | 2 |
| `k=11`, `e=-35` | `2049` | `tail_then_head` | sequential, pairwise | 2 |

These rows were reviewed in the order definition -> RN32 boundary semantics ->
minimal graph discriminator -> layout control -> midpoint boundary -> scale
replication.  Their predicted output bits and exact signed errors match the
candidate observations.

## Provisional rows

The remaining 24 rows also report prediction matches, but they were executed as
part of the earlier batch rather than accepted one rung at a time.  They remain
provisional replication evidence.  `36/36 matched` is a true description of
the batch, but it is not the current accepted milestone.

## Interpretation boundary

- `prediction_matched_observation=true` does not imply that the candidate was
  correctly rounded.
- `preregistered_sum_bits` is the predicted graph output, not a correctly
  rounded target.  This batch does not version a correct-rounding assessment.
- The predictor requires ordered stored leaves and an explicit reduction graph.
  It cannot directly predict or uniquely infer an unknown library or GPU graph.
- The batch does not establish a universal accuracy ordering between sequential
  and pairwise summation.
- Two nonuniform positive-input permutations were explored interactively after
  this batch.  They are not included in these artifacts and are not accepted as
  preregistered validation evidence.
