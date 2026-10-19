# Semantic Relevance v4a Hybrid Retrieval Experiment Specification

**Version:** `0.1.0`  
**Status:** `FROZEN_BEFORE_V4A_IMPLEMENTATION`

## Objective

The frozen v3 post-hoc diagnosis found 7/17 weak queries `RETRIEVAL_LIMITED`, 4/17 `RERANKER_LIMITED`, and 6/17 `MIXED`. v4a tests a retrieval-first hybrid while preserving the selected v3 reranker.

```text
                     query
                       |
            +----------+----------+
            |                     |
      dense top50            BM25 top50
            |                     |
            +----------+----------+
                       |
                deduplicate union
                       |
             hybrid retrieval RRF
                       |
          frozen v3 BGE reranker
                       |
             final conservative RRF
                       |
                     top10
```

## Frozen v4a choices

- BM25 corpus: runtime `books_with_emotions.csv`, using only `title + description`.
- No authors/categories/emotions and no BM25F field weights.
- Analyzer: NFKC, casefold, remove apostrophes, token pattern `(?u)\b[^\W_]{2,}\b`.
- No stop-word removal, stemming, lemmatization, synonym expansion, query rewriting, decomposition, Boolean exclusion handling, or phrase logic.
- BM25 Okapi: `k1=1.5`, `b=0.75`, Lucene-style non-negative IDF.
- BM25 depth 50, positive-score candidates only.
- Dense depth 50, unchanged. Development reuses the frozen H01-H24 dense top50 artifact.
- Union by ISBN, no truncation before reranking.
- Hybrid retrieval RRF: equal weights, `k=60`.
- Frozen reranker: `BAAI/bge-reranker-base`, revision `2cfc18c9415c912f9d8155881c133215df768a70`.
- Rerank the entire union.
- Final RRF: hybrid retrieval rank + reranker rank, equal weights, `k=60`; top10.
- No parameter sweeps.

## Development protocol

`H01-H24` are development-only. Compare frozen v3b with v4a; no release decision.

For the 17 Stage-B target queries, reuse the fully judged dense top50 and judge only previously-unjudged BM25-only candidates needed to complete the hybrid union. Measure lexical-only discoveries, hybrid oracle headroom, and Stage-C limitation-class transitions.

For all 24 queries, reuse frozen scores and judge only new pairs required by v4a top10. Compute the same eight system metrics used for v3.

Select v3b vs v4a lexicographically: lower catastrophic rate, lower irrelevant@10, higher clear/strong@10, higher macro mean@10, higher nDCG@10; exact tie retains v3b. Retrieval diagnostics explain but cannot override selection.

## New unseen holdout

Only if v4a is selected and its source/config are frozen, author `V4H01-V4H24`, four queries in each of the existing six structural strata. No new query may paraphrase/trivially vary Q01-Q12 or H01-H24. Authoring is blind to retrieval outputs; query/facet files freeze before retrieval.

The 24-query holdout remains a disciplined project benchmark, not a statistically powered production qualification sample.

## Final evaluation

Baseline: frozen v3b. Candidate: frozen v4a. Judge the unique union of displayed top10 pairs. Standalone v4a is primary; paired v3b→v4a is secondary. Use 5,000 query-level bootstrap replicates, seed `20261002`; no p-values/McNemar. The same eight release gates and PASS/REVIEW/FAIL policy remain unchanged.
