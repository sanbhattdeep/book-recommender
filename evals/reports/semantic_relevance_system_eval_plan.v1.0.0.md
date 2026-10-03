# Semantic Relevance System Evaluation v1.0

## What this evaluates

This benchmark evaluates the **book recommender**, not the calibrated judge.

The calibrated `semantic_relevance_v0.29.0-r5` judge is treated as the
measurement instrument. Human relevance labels are not used in this system
evaluation.

## Frozen benchmark shape

- 12 semantic queries
- `category = "All"`
- `tone = "All"`
- top 10 recommendations/query
- 120 query-book pairs
- each returned book receives a judge score from 0 to 4

The first benchmark intentionally isolates semantic retrieval quality. Category
filtering and emotional-tone ranking should be evaluated later as separate
dimensions rather than mixing multiple behaviors into one score.

## Primary metrics

1. **Macro mean relevance@10**
   - Average the 10 relevance scores within each query.
   - Average those 12 query means equally.
   - This prevents a query with extra/missing rows from receiving accidental
     extra weight.

2. **Clear-or-strong rate@10**
   - Fraction of recommendations scoring `>= 3`.

3. **Irrelevant rate@10**
   - Fraction scoring `0`.

4. **HitRate@5 (clear/strong)**
   - Fraction of queries with at least one score `>= 3` in the first five.

5. **Top-1 clear/strong rate**
   - Fraction whose first recommendation scores `>= 3`.

6. **MRR@10 (clear/strong)**
   - Reciprocal rank of the first score `>= 3`, macro-averaged across queries.

7. **nDCG@10**
   - Graded relevance gain: `2^score - 1`.
   - Measures whether the stronger recommendations appear earlier in the list.

8. **Catastrophic query rate**
   - Fraction of queries whose mean top-10 score is `< 1.5`.

## Confidence intervals

Use **5,000 query-level cluster bootstrap replicates**, seed `20261002`.

A bootstrap replicate samples 12 queries with replacement. Whenever a query is
sampled, its complete ranked list of 10 recommendations is carried with it.

Do **not** bootstrap the 120 query-book pairs independently. Results from the
same query are correlated, and treating them as independent would make the
confidence interval artificially narrow.

Report percentile 95% confidence intervals for the major aggregate metrics.

## Preregistered release gates

| Gate | Threshold |
|---|---:|
| Macro mean relevance@10 | `>= 2.50` |
| Lower 95% CI for macro mean relevance@10 | `>= 2.00` |
| Clear/strong rate@10 | `>= 60%` |
| Irrelevant rate@10 | `<= 10%` |
| HitRate@5 clear/strong | `>= 90%` |
| Top-1 clear/strong rate | `>= 75%` |
| Mean nDCG@10 | `>= 0.85` |
| Catastrophic query rate | `0` |

Decision:
- **PASS:** all eight gates pass.
- **REVIEW:** one or two fail and catastrophic query rate remains zero.
- **FAIL:** three or more fail, or any catastrophic query is present.

These are **demo benchmark gates**, not production SLAs. They are frozen before
the evaluation run so the thresholds are not chosen after seeing the result.

## Important statistical interpretation

The confidence interval quantifies uncertainty in the **estimated recommender
quality over the benchmark query population**.

It is not:
- the judge's confidence in an individual score;
- judge-vs-human agreement;
- a measure of LLM run-to-run stochasticity.

Judge stochasticity should be measured separately with repeated evaluation
runs after the primary system-quality run.

## Next lifecycle step

1. Freeze this contract and the recommender code snapshot.
2. Collect the 12 x 10 recommendations from the recommender.
3. Score all 120 pairs with the calibrated r5 judge.
4. Calculate query-level and aggregate metrics.
5. Bootstrap 95% CIs.
6. Apply the preregistered gates.
7. Inspect low-scoring queries and recommendations.
8. In a separate stage, repeat the same fixed evaluation to quantify
   non-deterministic judge stability.
