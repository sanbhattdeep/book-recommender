# Semantic Relevance v3 Reranker Experiment Specification

**Version:** `0.1.0`  
**Status:** `DRAFT_TO_FREEZE_BEFORE_V3_DEV_EXECUTION`

## Objective

Checkpoint D established `RERANKER_SUPPORTED`. The v3 experiment isolates one
change only:

```text
existing semantic vector retrieval
        -> top 50 candidates
        -> second-stage reranker
        -> top 10
```

Candidate generation, embeddings, vector index, category filtering, tone
filtering, and the 12 development query texts remain unchanged.

## Frozen development evidence

The 12 current queries are development/diagnostic only. Development evaluation
reuses the frozen 600-row Checkpoint C score map, so v3 development requires
zero new judge calls.

```text
top50_judge_scores_complete.csv
SHA-256
876761c8293615e50072c935b400d8ee31dda08ef12bdb75e7a6d8ecd87cbd35
```

## Reranker

Freeze:

```text
BAAI/bge-reranker-base
revision
2cfc18c9415c912f9d8155881c133215df768a70
```

No project-specific training or fine-tuning is allowed. Use the repository's
Transformers stack with `AutoTokenizer` and
`AutoModelForSequenceClassification`, `trust_remote_code=False`,
safetensors, eval mode, no gradients, maximum input length 512, and
`truncation="only_second"`.

Candidate text:

```text
Title: <title>
Description: <description>
```

If description is empty, use title only. Authors are excluded. The raw single
classification logit is used only for ranking.

## Frozen variants

### v2

Original vector order; take ranks 1-10.

### v3a

Rank all 50 by reranker score descending, then vector rank ascending, then ISBN
ascending. Take 10.

### v3b

Derive reranker rank using the v3a ordering, then calculate:

```text
1/(60 + vector_rank) + 1/(60 + reranker_rank)
```

Both sources have weight 1.0. `k=60` is frozen and must not be tuned on the
development queries.

Rank by RRF score descending, then reranker rank, vector rank, ISBN. Take 10.

## Development comparison

Compute the same eight system metrics for v2, v3a and v3b, plus deep-pool
promotion and query-level win/tie/loss diagnostics.

No release decision is allowed on the 12 development queries.

Exactly one v3 variant is selected with this frozen lexicographic rule:

1. lower catastrophic-query rate;
2. lower irrelevant@10;
3. higher clear/strong@10;
4. higher macro mean@10;
5. higher nDCG@10;
6. if still tied, choose v3b.

No model swap, fine-tuning, RRF-weight tuning or `k` search is permitted after
development results are seen.

## Unseen holdout

After the selected v3 variant and its implementation/configuration are frozen,
create and freeze **24 unseen queries**, `H01-H24`, before generating or viewing
any holdout retrieval output.

Use six strata, four queries each:

| Stratum | Count |
|---|---:|
| Two-core theme | 4 |
| Three-core conjunctive | 4 |
| Core + qualifier | 4 |
| Relationship/role constraint | 4 |
| Hard exclusion/contrast | 4 |
| Abstract nonfiction multi-facet | 4 |

No holdout query may paraphrase or trivially vary any current development
query/theme.

The holdout query file and facet specification are frozen before retrieval.
Semantic rubric and deterministic scoring rules cannot be changed after the
holdout freeze.

## Final evaluation

For each unseen query, v2 and v3 share the same top-50 vector candidate pool:

```text
top50
  |-> v2 original top10
  |-> selected frozen v3 reranker top10
```

Judge the unique union of displayed pairs so shared query-book pairs are judged
once.

The standalone v3 result is primary. Paired v2-v3 analysis is secondary and
cannot override the standalone release decision. Use query-level bootstrap;
no p-values or McNemar test.

## Frozen final release gates

The original eight gates remain unchanged:

- macro mean@10 >= 2.50
- macro-mean 95% CI lower bound >= 2.00
- clear/strong@10 >= 60%
- irrelevant@10 <= 10%
- HitRate@5 >= 90%
- Top1 clear/strong >= 75%
- nDCG@10 >= 0.85
- catastrophic-query rate == 0

Decision policy remains unchanged:

- PASS: all eight pass
- REVIEW: one or two fail and catastrophic rate is zero
- FAIL: three or more fail, or catastrophic rate is non-zero

A positive paired v2->v3 result cannot convert a standalone v3 FAIL into PASS.

## Explicit non-goals

No embedding/index change, retrieval-depth change, judge-as-reranker, training
on the 600 judge-labeled development pairs, RRF tuning, judge-semantic change,
scoring-rule change, or post-hoc release-gate change is allowed.
