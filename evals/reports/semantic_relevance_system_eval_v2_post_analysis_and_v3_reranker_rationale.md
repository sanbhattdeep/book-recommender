# Semantic Relevance System Evaluation v2 — Post-Analysis and v3 Reranker Rationale

**Project:** Book Recommender  
**Evaluation lineage:** semantic relevance system evaluation v1 → v2 controlled comparison  
**Status:** v2 evaluation complete — **FAIL**, but materially improved over v1  
**Investigation status:** residual failure modes characterized; next controlled hypothesis identified  
**Recommended repo location:** `evals/reports/semantic_relevance_system_eval_v2_post_analysis_and_v3_reranker_rationale.md`

---

## 1. Purpose

This document records the analysis performed after semantic relevance system evaluation v2.

The purpose is to preserve an auditable engineering record of:

1. what v1 failed on;
2. the deterministic retrieval-order defect identified from the v1 failure;
3. the controlled v2 fix;
4. the standalone v2 release result;
5. the paired v1-to-v2 evidence;
6. the query-level and recommendation-level residual failure analysis;
7. why the next hypothesis is a reranking stage over the existing top-50 semantic candidates;
8. why reranking is not yet assumed to solve every remaining failure.

This is a **system-evaluation and failure-analysis document**. It does not change the frozen judge, benchmark, release gates, or previously recorded v1/v2 results.

---

## 2. Evaluation lineage

The investigation followed this lifecycle:

```text
frozen recommender v1
        |
        v
system evaluation v1
        |
        v
0/8 release gates passed
        |
        v
query-level failure analysis
        |
        v
deterministic semantic-order defect identified
        |
        v
regression coverage + minimal order-preservation fix
        |
        v
frozen recommender v2
        |
        v
same benchmark + same judge + same gates
        |
        v
system evaluation v2
        |
        v
paired v1-v2 comparison
        |
        v
recommendation-level residual analysis
        |
        v
next controlled hypothesis:
rerank the existing top-50 semantic candidates before final top-10 selection
```

The failed v1 run remains the permanent baseline. It was not overwritten or reinterpreted after the fix.

---

## 3. Frozen comparison design

The v1-to-v2 comparison kept the evaluation contract fixed:

- 12 semantic benchmark queries;
- `category = "All"`;
- `tone = "All"`;
- initial semantic retrieval depth = 50;
- final recommendations = 10;
- 120 query-book pairs per system version;
- same calibrated semantic relevance judge;
- same scoring rubric;
- same system metrics;
- same bootstrap method;
- same release gates;
- no human relevance labels used in system evaluation;
- no judge calls during metric analysis or paired comparison.

For the paired comparison:

- matched unit = query;
- matched queries = 12;
- 5,000 paired bootstrap query-level resamples;
- both versions of a query remain paired within each resample;
- win/tie/loss is calculated per query and oriented so that WIN always means better for v2.

The paired comparison is **secondary evidence**. The authoritative v2 release decision still comes from the frozen standalone v2 metrics and gates.

---

## 4. v1 baseline and the confirmed deterministic defect

### 4.1 v1 baseline

The original system evaluation produced:

| Metric | v1 |
|---|---:|
| Macro mean relevance@10 | 1.333 / 4 |
| Clear/strong rate@10 | 24.2% |
| Irrelevant rate@10 | 42.5% |
| HitRate@5 clear/strong | 58.3% |
| Top-1 clear/strong rate | 16.7% |
| MRR@10 clear/strong | 0.359 |
| nDCG@10 | 0.763 |
| Catastrophic query rate | 66.7% |

**Release decision:** `FAIL`  
**Passed gates:** `0/8`

### 4.2 Root cause found after v1

The recommender performed vector search correctly:

```python
recs = db_books.similarity_search(query, k=initial_top_k)
```

The returned candidate ISBN list therefore had semantic similarity order.

However, the metadata join used a Pandas membership filter similar to:

```python
books[books["isbn13"].isin(books_list)]
```

`isin()` preserved membership but not the semantic ordering of `books_list`.

A subsequent:

```python
.head(final_top_k)
```

therefore selected rows according to incidental source-DataFrame order rather than vector similarity rank.

### 4.3 Deterministic Q09 proof

For the dystopian / authoritarian-control / resistance query, the vector store ranked:

```text
1. 1984
2. The Assault
3. Animal Farm and 1984
...
```

but the v1 recommender returned a different set/order, including books from much lower vector ranks. `1984`, despite being vector rank 1, disappeared from the final top 10.

This established that the major v1 defect was a deterministic application bug at the retrieval-to-metadata boundary, not an LLM-judge failure or statistical artifact.

The v2 change was deliberately narrow: preserve semantic candidate order through the metadata join and final selection.

---

## 5. Standalone v2 result

After the order-preservation fix, v2 produced:

| Metric | v2 | 95% CI where reported |
|---|---:|---:|
| Macro mean relevance@10 | 1.692 | [1.350, 2.059] |
| Clear/strong rate@10 | 33.3% | [20.0%, 49.2%] |
| Irrelevant rate@10 | 30.8% | [18.3%, 43.3%] |
| HitRate@5 clear/strong | 66.7% | — |
| Top-1 clear/strong rate | 33.3% | — |
| MRR@10 clear/strong | 0.508 | — |
| nDCG@10 | 0.836 | — |
| Catastrophic query rate | 25.0% | — |

The frozen release gates were unchanged:

| Gate | v2 |
|---|---|
| Macro mean relevance@10 >= 2.50 | FAIL |
| Macro-mean 95% CI lower bound >= 2.00 | FAIL |
| Clear/strong rate@10 >= 60% | FAIL |
| Irrelevant rate@10 <= 10% | FAIL |
| HitRate@5 >= 90% | FAIL |
| Top-1 clear/strong >= 75% | FAIL |
| nDCG@10 >= 0.85 | FAIL |
| Catastrophic-query rate = 0% | FAIL |

**v2 release decision:** `FAIL`  
**Passed gates:** `0/8`

This is important: **v2 being better than v1 does not imply that v2 is release-ready.**

The paired comparison answers "did the fix improve the system?"  
The frozen release gates answer "is the resulting system good enough?"

---

## 6. Paired v1-to-v2 result

The paired query-level comparison showed:

| Metric | v1 | v2 | Delta | 95% paired bootstrap CI | W/T/L |
|---|---:|---:|---:|---:|---:|
| Macro mean relevance@10 | 1.333 | 1.692 | +0.358 | [+0.158, +0.567] | 9/1/2 |
| Clear/strong rate@10 | 24.2% | 33.3% | +9.2 pp | [+3.3 pp, +15.0 pp] | 8/3/1 |
| Irrelevant rate@10 | 42.5% | 30.8% | -11.7 pp | [-20.8 pp, -2.5 pp] | 8/2/2 |
| HitRate@5 clear/strong | 58.3% | 66.7% | +8.3 pp | [-16.7 pp, +33.3 pp] | 2/9/1 |
| Top-1 clear/strong rate | 16.7% | 33.3% | +16.7 pp | [-16.7 pp, +50.0 pp] | 3/8/1 |
| MRR@10 clear/strong | 0.359 | 0.508 | +0.149 | [-0.101, +0.345] | 8/3/1 |
| nDCG@10 | 0.763 | 0.836 | +0.073 | [-0.034, +0.176] | 8/0/4 |
| Catastrophic query rate | 66.7% | 25.0% | -41.7 pp | [-66.7 pp, -16.7 pp] | 5/7/0 |

### 6.1 Strongest evidence of improvement

Four important metrics have paired confidence intervals entirely on the improvement side of zero:

- macro mean relevance@10;
- clear/strong rate@10;
- irrelevant rate@10;
- catastrophic-query rate.

The macro point estimate increased by roughly 27% relative to v1:

```text
(1.692 - 1.333) / 1.333 ~= 26.9%
```

The catastrophic-query rate fell from 66.7% to 25.0%, equivalent to approximately:

```text
v1: 8 / 12 catastrophic queries
v2: 3 / 12 catastrophic queries
```

The catastrophic W/T/L pattern was `5/7/0`: five queries improved classification, seven were unchanged, and none became newly catastrophic.

### 6.2 Gate transitions

All eight frozen gates remained:

```text
FAIL -> FAIL
```

There were:

```text
FAIL -> PASS: 0
PASS -> PASS: 0
FAIL -> FAIL: 8
PASS -> FAIL: 0
```

The correct conclusion is therefore:

> **The order-preservation fix was successful as an engineering/regression fix, but insufficient as a release-quality fix.**

---

## 7. Recommendation-level integrity checks

The two frozen score artifacts were compared directly.

### 7.1 Frozen score artifact hashes

**v1 `judge_scores.csv` SHA-256**

```text
d079a74c7f0473e46d28f19e37e4241a780804e972f3fa5f25b3a67e62a7bd11
```

**v2 `judge_scores.csv` SHA-256**

```text
4770986e868b26821131d2b36f6cc648f0cab68ae50acda0d1ba4df6d0bd0da1
```

### 7.2 Shared recommendation consistency

Only 24 query-book pairs are shared between v1 and v2.

For all 24 shared pairs:

```text
v1 judge score == v2 judge score
```

Score mismatches:

```text
0 / 24
```

This is a useful integrity check: the measured system difference is driven by changed recommendations/ranks, not by judge-score drift on shared cases.

### 7.3 Membership changed substantially

Per-query top-10 overlap:

| Query | Shared books | Changed membership |
|---|---:|---:|
| Q01 | 3 | 7 |
| Q02 | 2 | 8 |
| Q03 | 2 | 8 |
| Q04 | 3 | 7 |
| Q05 | 1 | 9 |
| Q06 | 2 | 8 |
| Q07 | 5 | 5 |
| Q08 | 1 | 9 |
| Q09 | 2 | 8 |
| Q10 | 2 | 8 |
| Q11 | 1 | 9 |
| Q12 | 0 | 10 |

Across the whole benchmark:

```text
shared query-book pairs: 24 / 120
changed-membership slots: 96 / 120
```

This is consistent with the original defect: v1 was not simply misordering the correct ten books; the order-loss plus truncation changed **which** top-50 candidates survived into the final ten.

### 7.4 Quality of replaced recommendations

The 96 recommendations unique to each version show:

| Recommendation subset | Mean judge score | Clear/strong | Irrelevant |
|---|---:|---:|---:|
| v1-only recommendations | 1.156 | 18.8% | 49.0% |
| v2-only recommendations | 1.604 | 30.2% | 34.4% |

The v2 replacement set is therefore materially better.

This provides recommendation-level evidence that preserving vector order improved the selected candidate set, not merely aggregate metrics.

---

## 8. Residual query-level failure patterns

The remaining failures are not uniform. The analysis identified distinct categories.

### 8.1 Q09 — major recovery, but residual candidate-quality gap

Query:

> A dystopian story about authoritarian control and resistance

Observed change:

```text
mean relevance:       0.4 -> 1.3
irrelevant rate:      80% -> 40%
HitRate@5:              0 -> 1
MRR:                    0 -> 0.333
nDCG:                0.482 -> 0.823
catastrophic:           1 -> 1
```

v2 top results include:

```text
rank 1  1984                  score 2
rank 2  The Assault           score 2
rank 3  Animal Farm and 1984 score 3
```

This is strong confirmation that the v1 order-loss diagnosis was correct.

Q09 remains technically catastrophic only because mean relevance remains below the `< 1.5` threshold. The remaining problem is no longer primarily the metadata-order bug; it is the quality of the broader semantic candidate set.

### 8.2 Q01 — better average set, but a strong lower-ranked top-50 candidate was lost

Query:

> A story about forgiveness and redemption

v1 had:

```text
rank 1  Gilead  score 3
```

v2 contains no clear/strong result. It instead contains several partial matches:

```text
rank 1  Searching for God Knows What  score 2
rank 2  Lila's Child                  score 2
rank 3  A Time to Embrace             score 2
rank 5  Redemption                    score 2
```

Because `Gilead` appeared in v1, it must have existed somewhere in the original vector-search top-50 candidate pool. It is absent from v2's vector-order-preserved top 10.

This is direct evidence that:

> a clear result can exist below vector rank 10 but within the existing top 50.

That observation is central to the reranking hypothesis.

### 8.3 Q03 — candidate set improved, but strong results are buried

Query:

> A book about personal growth and finding purpose

v2 includes three clear results:

```text
rank 7  The Celestine Prophecy         score 3
rank 8  The Drama of the Gifted Child  score 3
rank 9  The Purpose of Your Life       score 3
```

Yet ranks 1-6 contain no clear result.

Observed change:

```text
mean relevance: 1.2 -> 1.7
clear results:    0 -> 3
MRR:              0 -> 0.143
nDCG:          0.890 -> 0.726
```

The candidate set improved, but the best candidates are poorly ordered for the full query intent.

This is a **reranking** failure pattern.

### 8.4 Q10 — strongest current reranking example

Query:

> A book about leadership, teamwork, and building effective organizations

v2 includes:

| Rank | Title | Score |
|---:|---|---:|
| 2 | Leadership and the One Minute Manager | 2 |
| 3 | Leadership in Organizations | 2 |
| 6 | The Leadership Challenge | 2 |
| 7 | The Leader In You | 3 |
| 8 | The Five Dysfunctions of a Team | 3 |
| 9 | Lead Like Jesus | 2 |
| 10 | Behind Closed Doors | 2 |

Observed change:

```text
mean relevance:      0.8 -> 1.7
irrelevant count:      5 -> 2
clear results:         0 -> 2
catastrophic:          1 -> 0

but:

HitRate@5:             0 -> 0
Top-1 clear:           0 -> 0
nDCG:               0.891 -> 0.629
```

This is the clearest evidence that candidate generation can retrieve useful books while vector similarity alone does not order them well enough for the complete multi-facet intent.

Q10 strongly supports a second-stage reranker.

### 8.5 Q11 — earlier clear result, but poorer tail quality

Query:

> An adventure inspired by mythology, gods, and legendary heroes

v2 score pattern:

```text
rank 1   score 0
rank 2   score 3
rank 3   score 3
rank 4   score 0
rank 5   score 2
rank 6   score 0
rank 7   score 2
rank 8   score 2
rank 9   score 0
rank 10  score 3
```

v1 included clear matches such as:

```text
The Epic of Gilgamesh  score 3
Bakkhai                 score 3
Mythology               score 3
```

Only `Mythology` is shared with v2, moving from rank 8 to rank 2.

As with Q01, clear candidates that appeared in v1 necessarily existed somewhere in the original top-50 pool, even though they are outside v2's top 10.

This is additional evidence that a top-50 reranker may recover valuable candidates that vector similarity alone leaves below rank 10.

### 8.6 Q12 — warning case: reranking may not be sufficient

Query:

> A thoughtful book about inequality, poverty, and how society distributes wealth

v2 has:

```text
rank 2  A Framework for Understanding Poverty  score 3
rank 3  War and Peace and War                  score 2
```

but:

```text
8 / 10 v2 recommendations score 0
```

Q12 has **zero top-10 overlap** between v1 and v2.

Its v1 top 10 was also weak:

```text
clear/strong: 0
partial:      3
incidental:   1
irrelevant:   6
```

Therefore the current evidence does **not** show that many strong Q12 candidates are sitting lower in the existing top 50 waiting to be promoted.

Q12 remains a possible **candidate-retrieval / semantic-representation** weakness.

This is why the proposed v3 reranker must be treated as the next controlled hypothesis, not as a claim that reranking will solve every remaining failure.

---

## 9. Failure taxonomy after v2

| Query | Primary current diagnosis | Priority |
|---|---|---:|
| Q01 | Better average set, but a strong lower-ranked candidate is missing from top 10 | High |
| Q03 | Better candidate set; clear matches ranked too low | High |
| Q09 | Major successful recovery; residual candidate-quality weakness | Medium |
| Q10 | Much better candidates; strongest matches ranked too low | Highest |
| Q11 | Useful lower-ranked top-50 candidates appear recoverable; tail quality remains mixed | High |
| Q12 | Severe semantic-tail contamination; possible candidate-recall weakness | Highest |

The post-v2 architecture diagnosis is therefore:

```text
v1
---
vector search
    |
    v
ranked top-50 semantic candidates
    |
    v
metadata membership join
    |
    X  semantic order destroyed
    |
    v
incorrect final top 10


v2
---
vector search
    |
    v
ranked top-50 semantic candidates
    |
    v
order-preserving metadata join
    |
    v
correct vector top 10
    |
    v
remaining weaknesses become visible:
    - strong candidates below vector rank 10
    - clear candidates inside top 10 but ranked too low
    - multi-facet candidate-quality weaknesses
```

---

## 10. Why a reranker is the next controlled hypothesis

The evidence supports keeping v2 candidate generation unchanged and testing a second-stage semantic reranker over the **existing top 50 candidates**.

Proposed high-level flow:

```text
query
  |
  v
vector similarity search
top 50 candidates
  |
  |  unchanged from v2
  v
semantic reranker
(query + richer book representation)
  |
  v
best 10
  |
  v
final recommendations
```

### 10.1 Why not rerank only the existing top 10?

A perfect reorder of v2's current top 10 cannot change:

- macro mean relevance;
- clear/strong rate;
- irrelevant rate;
- catastrophic-query rate;
- macro-mean confidence interval.

Those depend on membership, not ordering.

It could improve rank-sensitive metrics only.

In the current v2 top-10 sets, 11 of 12 queries contain at least one clear/strong result. Therefore an ideal within-top-10 reorder has an upper bound of approximately:

```text
HitRate@5:   11 / 12 = 91.7%
Top-1 clear: 11 / 12 = 91.7%
MRR:                    0.917
nDCG:                   1.000
```

That theoretical reorder would be enough to exceed the frozen gates for:

- HitRate@5;
- Top-1 clear/strong;
- nDCG.

But it would leave all candidate-composition gates unchanged.

Therefore **top-10-only reranking is too narrow**.

### 10.2 Why top-50 reranking is more promising

Q01 and Q11 provide evidence that clear recommendations existed somewhere in the original semantic top-50 pool but were not in v2's vector top 10.

Q03 and Q10 show the complementary failure mode: useful candidates are already in v2's top 10 but are ranked too low.

A top-50 reranker can attack both:

1. **membership problem** — promote strong candidates currently at vector ranks 11-50 into the final ten;
2. **ordering problem** — place the best selected candidates near the top.

This gives the reranker a plausible path to improve both rank-sensitive metrics and final-top-10 composition metrics.

---

## 11. What the evidence does not yet prove

The current analysis does **not** prove that a reranker alone will pass the frozen release gates.

In particular, it does not yet establish:

- how many clear/strong candidates exist within ranks 11-50 for every query;
- whether Q12 has enough relevant candidates anywhere in the current top 50;
- whether the embedding model adequately represents all multi-facet intents;
- whether the indexed book text contains sufficient evidence for those intents;
- whether a reranker can reliably distinguish partial from clear multi-facet matches;
- whether reranking 50 candidates is cost-effective for the intended product;
- whether candidate-generation depth should eventually exceed 50.

The next v3 work should therefore begin with an **offline top-50 diagnostic** before freezing a reranker implementation.

---

## 12. Recommended next experiment

### Step 1 — preserve all existing evidence

Do not modify:

- v1 artifacts;
- v2 artifacts;
- the calibrated judge;
- the 12-query benchmark;
- the release gates;
- the bootstrap configuration.

### Step 2 — inspect the full v2 top-50 candidate pool

For each frozen query, capture at least:

- vector rank;
- ISBN;
- title;
- text used for embedding/retrieval;
- metadata available to a reranker.

The immediate diagnostic question is:

> Are there sufficiently relevant candidates at vector ranks 11-50 to make top-50 reranking a credible improvement strategy across the benchmark?

Q12 should receive particular attention.

### Step 3 — define v3 as a controlled reranking experiment

If the top-50 diagnostic supports the hypothesis:

- keep v2 vector candidate generation unchanged;
- rerank the same 50 candidates;
- select final top 10 only after reranking;
- add deterministic tests for reranker ordering/truncation behavior;
- freeze a new recommender version;
- execute the same frozen benchmark;
- compare v2 vs v3 using the same paired query-level analysis.

### Step 4 — treat retrieval changes as a separate later experiment

If Q12 or other queries show poor relevant-candidate recall even across the full top 50, candidate generation should be investigated separately:

- query representation;
- book text/index representation;
- embedding model;
- candidate depth;
- retrieval strategy.

Do not combine those changes with the first reranker experiment if causal attribution is important.

---

## 13. Decision

### v2 engineering conclusion

**Keep the v2 order-preservation fix.**

Evidence supporting this decision:

- macro relevance increased from 1.333 to 1.692;
- clear/strong rate increased by 9.2 percentage points;
- irrelevant rate decreased by 11.7 percentage points;
- catastrophic-query rate decreased by 41.7 percentage points;
- the corresponding paired intervals for those four metrics remain in the improvement direction;
- 9/12 queries improved on macro relevance;
- 96/120 recommendation slots changed membership;
- v2-only replacements have higher average judge score and lower irrelevance than v1-only replacements;
- all 24 shared query-book pairs retain identical judge scores.

### v2 release conclusion

**Do not treat v2 as release-ready.**

All eight frozen release gates still fail.

### next hypothesis

**Before changing embeddings or retrieval, investigate and then test a semantic reranker over the existing top-50 vector candidates.**

The rationale is evidence-based:

- Q03 and Q10 show good candidates ranked too low;
- Q01 and Q11 show useful candidates can exist below vector rank 10 but inside the original top 50;
- top-10-only reordering cannot improve candidate-composition gates;
- top-50 reranking can potentially improve both membership and ordering;
- Q12 remains an explicit warning that candidate retrieval itself may still be insufficient.

The reranker is therefore the **next controlled experiment**, not the assumed final solution.

---

## 14. Concise case-study summary

A concise way to explain the full sequence:

> I built a frozen semantic-relevance system benchmark for a book recommender using a separately calibrated LLM judge, query-level bootstrap confidence intervals, and preregistered release gates. The first system version failed all eight gates. Failure analysis showed that the vector store returned good candidates in semantic order, but the application destroyed that ordering when joining ISBNs back to Pandas metadata using `isin()`, then truncated the wrong rows. I proved the defect deterministically, added regression coverage, fixed only that boundary, froze v2, and reran the same benchmark.
>
> v2 still failed the absolute release bar, but paired analysis showed material improvement: mean relevance increased, irrelevant recommendations decreased, and catastrophic queries fell from roughly eight of twelve to three of twelve. Recommendation-level comparison showed that 96 of 120 final slots changed membership and the v2 replacement set was substantially better, validating the original defect diagnosis.
>
> The controlled fix then exposed the next failure layer. Some strong candidates were already present but ranked too low, while other strong books appeared in v1 only because they existed deeper in the original top-50 semantic pool. That led to the next controlled hypothesis: keep v2 candidate generation fixed and rerank the full top 50 before selecting the final ten. One query, however, still showed poor candidate quality, so reranking is treated as an experiment rather than assumed to solve all retrieval problems.

---

## 15. Evidence artifacts

Primary repo/runtime artifacts used in this analysis include:

```text
evals/reports/
semantic_relevance_system_eval_v1_failure_investigation.md

evals/runs/semantic_relevance_system_eval_v1/
20261002T152856Z_recommendations/
judge_scores.csv
query_metrics.csv
aggregate_metrics.json
bootstrap_confidence_intervals.json
release_gate_result.json
system_evaluation_report.md

evals/runs/semantic_relevance_system_eval_v2/
20261004T140218Z_recommendations/
judge_scores.csv
query_metrics.csv
aggregate_metrics.json
bootstrap_confidence_intervals.json
release_gate_result.json
system_evaluation_report.md
v1_vs_v2_paired_report.md
v1_vs_v2_paired_comparison.json
v1_vs_v2_query_deltas.csv
v1_vs_v2_gate_transitions.csv
system_evaluation_dashboard.html
```

No new judge calls or human relevance labels were required for this post-analysis.
