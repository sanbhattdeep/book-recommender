# Semantic Relevance System Evaluation v1 — Failure Investigation and Root-Cause Analysis

**Project:** Book Recommender  
**Evaluation:** Semantic relevance system evaluation v1  
**Status:** Baseline evaluation complete — **FAIL**  
**Investigation status:** Root cause identified for a major ranking defect  
**Recommended repo location:** `evals/reports/semantic_relevance_system_eval_v1_failure_investigation.md`

---

## 1. Purpose

This document records the investigation performed after the first frozen semantic-relevance system evaluation of the book recommender failed its preregistered quality gates.

The goals are to preserve an auditable record of how the failure was diagnosed, separate observed evidence from interpretation, identify the confirmed implementation defect before changing the recommender, preserve the failed run as a valid baseline rather than overwriting it, and provide a technically rigorous case study suitable for interview discussion.

This is a **system-evaluation investigation**, not a judge-calibration investigation. The semantic relevance judge had already been calibrated separately and was used here as the measurement instrument for recommender output quality.

---

## 2. Frozen evaluation context

### 2.1 System under test

The evaluated recommender implementation was frozen before execution.

**Frozen recommender SHA-256**

```text
b8f7c1223a525345a7e278fd29e0cd32a5e0c644a9a0b88e5cc9ac319f749974
```

The SHA was rechecked after the failure investigation and confirmed to still match the recommender source used for the baseline evaluation.

### 2.2 Evaluation design

The system benchmark used:

- 12 frozen semantic queries.
- `category = "All"`.
- `tone = "All"`.
- `initial_top_k = 50`.
- `final_top_k = 10`.
- 10 returned recommendations per query.
- 120 total query-book pairs.
- No human relevance labels during system evaluation.
- Frozen semantic relevance judge: `semantic_relevance_v0.29.0-r5`.
- Non-semantic execution amendment: `book-subject-retryfix1`.
- Judge score range: `0..4`.

### 2.3 Frozen scored artifact

The completed 120-pair scoring run produced:

```text
judge_scores.csv SHA-256
d079a74c7f0473e46d28f19e37e4241a780804e972f3fa5f25b3a67e62a7bd11
```

The metric-analysis stage was read-only with respect to this frozen score artifact.

### 2.4 Statistical method

The benchmark preregistered:

- 5,000 bootstrap replicates.
- 95% percentile confidence intervals.
- Seed `20261002`.
- Resampling unit = **query**.
- The 10 recommendations belonging to a query stay together in each resample.

This avoids incorrectly treating the 120 query-book pairs as 120 independent observations.

---

## 3. Baseline evaluation result

The frozen system evaluation failed decisively.

| Metric | Observed | Preregistered gate | Result |
|---|---:|---:|---|
| Macro mean relevance@10 | 1.333 / 4 | >= 2.50 | FAIL |
| 95% CI for macro mean | [0.942, 1.775] | lower bound >= 2.00 | FAIL |
| Clear/strong rate@10 (`score >= 3`) | 24.2% | >= 60% | FAIL |
| Irrelevant rate@10 (`score = 0`) | 42.5% | <= 10% | FAIL |
| HitRate@5 clear/strong | 58.3% | >= 90% | FAIL |
| Top-1 clear/strong rate | 16.7% | >= 75% | FAIL |
| MRR@10 clear/strong | 0.359 | descriptive | — |
| nDCG@10 | 0.763 | >= 0.85 | FAIL |
| Catastrophic query rate | 66.7% | 0% | FAIL |

**Formal decision**

```text
SYSTEM EVALUATION DECISION: FAIL
Passed gates: 0/8
```

The upper end of the 95% CI for mean relevance was only `1.775`, still well below the point-estimate gate of `2.50`. Therefore the result was not merely a narrow miss caused by a small benchmark.

---

## 4. Score distribution

Across the 120 returned recommendations:

| Judge score | Count |
|---:|---:|
| 0 — none | 51 |
| 1 — incidental | 7 |
| 2 — partial | 33 |
| 3 — clear | 29 |
| 4 — strong | 0 |

Key observations:

- 42.5% of all returned recommendations were completely irrelevant under the frozen rubric.
- Only 24.2% were clear-or-strong.
- No recommendation scored `4`.
- The recommender therefore showed a broad inability to satisfy full multi-facet semantic intent, not merely isolated ranking noise.

---

## 5. Query-level failure analysis

The 12 benchmark queries were sorted from worst to best.

| Query | Mean@10 | Clear/strong | Irrelevant | nDCG@10 | Catastrophic |
|---|---:|---:|---:|---:|---:|
| Q09 | 0.4 | 0% | 80% | 0.482 | Yes |
| Q12 | 0.7 | 0% | 60% | 0.498 | Yes |
| Q01 | 0.7 | 10% | 70% | 0.881 | Yes |
| Q10 | 0.8 | 0% | 50% | 0.891 | Yes |
| Q08 | 0.9 | 20% | 60% | 0.807 | Yes |
| Q06 | 1.0 | 30% | 60% | 0.594 | Yes |
| Q03 | 1.2 | 0% | 40% | 0.890 | Yes |
| Q07 | 1.3 | 40% | 50% | 0.661 | Yes |
| Q11 | 1.7 | 30% | 30% | 0.777 | No |
| Q05 | 1.9 | 10% | 10% | 0.841 | No |
| Q04 | 2.7 | 70% | 0% | 0.856 | No |
| Q02 | 2.7 | 80% | 0% | 0.977 | No |

### 5.1 Distinct observed failure patterns

The queries did not fail in one uniform way.

#### Severe candidate-quality + ranking failure

Examples:

- Q09 — dystopian / authoritarian control / resistance.
- Q12 — inequality / poverty / wealth distribution.

These produced almost no clear matches and also low nDCG.

#### Candidate-quality / semantic-coverage failure

Examples:

- Q03 — personal growth / finding purpose.
- Q10 — leadership / teamwork / effective organizations.

Both had poor absolute relevance but relatively high nDCG (`~0.89`).

This is important because this benchmark's nDCG ideal is formed by sorting the same returned ten books by frozen judge score. Therefore high nDCG can coexist with poor retrieval quality: the system may order a weak candidate set reasonably without retrieving sufficiently relevant books.

#### Mixed candidate and ranking failure

Examples:

- Q06 — grief / loss / learning to live again.
- Q07 — complicated parent-child relationship.

Relevant books existed in the top 10, but were mixed with many irrelevant items and were not consistently placed at rank 1.

#### Tail contamination

Q01 produced a clear match at rank 1 but mostly irrelevant recommendations afterwards.

#### Healthy reference case

Q02 performed strongly:

- mean relevance = 2.7,
- clear/strong = 80%,
- irrelevant = 0%,
- top-1 clear,
- nDCG = 0.977.

This showed that the overall pipeline was capable of producing a good result for at least some semantic intents.

---

## 6. Investigation hypothesis

After reviewing the aggregate and query-level results, the next diagnostic question was:

> Are good books absent from the initial semantic candidate pool, or are they present but lost during the recommender's final selection/ranking stage?

This distinction matters because the remediation would be completely different.

### If relevant books are missing from the initial candidate pool

Likely investigation areas:

- embedding model,
- query representation,
- indexed text,
- vector-store configuration,
- corpus content.

### If relevant books are in the candidate pool but disappear or move down

Likely investigation areas:

- metadata join,
- filtering,
- ordering,
- reranking,
- truncation logic.

The recommender implementation was then inspected before changing any code.

---

## 7. Recommender implementation inspection

The frozen function contained the following semantic retrieval flow:

```python
recs = db_books.similarity_search(query, k=initial_top_k)

books_list = [
    int(rec.page_content.strip('"').split()[0])
    for rec in recs
]

book_recs = books[
    books["isbn13"].isin(books_list)
].head(initial_top_k)
```

For `category == "All"` the code then performed:

```python
book_recs = book_recs.head(final_top_k)
```

### 7.1 Critical implementation observation

`db_books.similarity_search(...)` returns documents in semantic similarity order.

However:

```python
books[books["isbn13"].isin(books_list)]
```

uses `isin()` only as a membership filter.

It does **not** reorder the `books` DataFrame to match the ordered `books_list` returned by the vector store.

Therefore the similarity ranking is lost during the metadata join.

The subsequent:

```python
.head(final_top_k)
```

takes the first rows in the original `books` DataFrame order rather than the highest-ranked semantic candidates.

---

## 8. Deterministic proof using Q09

To confirm the source inspection, a read-only diagnostic compared the vector-store ordering with the current recommender output for:

```text
Q09
A dystopian story about authoritarian control and resistance
```

### 8.1 Actual vector-store top 10

The vector search returned:

| Vector rank | Title |
|---:|---|
| 1 | 1984 |
| 2 | The Assault |
| 3 | Animal Farm and 1984 |
| 4 | The Control of Nature |
| 5 | Pacific Edge |
| 6 | Pathologies of Power |
| 7 | High Society |
| 8 | Obasan |
| 9 | Shade's Children (rack) |
| 10 | Democracy in America |

### 8.2 Current recommender top 10

The frozen recommender returned:

| Function rank | Title |
|---:|---|
| 1 | The Devil and Miss Prym |
| 2 | The Darling |
| 3 | Shade's Children (rack) |
| 4 | You Bright and Risen Angels |
| 5 | The Assault |
| 6 | The Tortilla Curtain |
| 7 | The Plague |
| 8 | Youth |
| 9 | Democracy Matters |
| 10 | Animal Farm and 1984 |

The two top-10 sets were not even identical.

### 8.3 Function rank mapped back to vector rank

| Function rank | Vector rank |
|---:|---:|
| 1 | 32 |
| 2 | 29 |
| 3 | 9 |
| 4 | 12 |
| 5 | 2 |
| 6 | 28 |
| 7 | 13 |
| 8 | 22 |
| 9 | 43 |
| 10 | 3 |

Observed facts:

- Function rank 1 was only vector rank 32.
- Function rank 2 was vector rank 29.
- Vector rank 1 (`1984`) disappeared entirely from the final top 10.
- Vector rank 3 appeared only at function rank 10.
- The semantic top-10 ordering and returned top-10 ordering were different.
- The returned top-10 set itself was different from the vector top-10 set.

This provides deterministic evidence that semantic similarity order was lost after retrieval.

---

## 9. Confirmed root cause

### Root cause statement

> The vector database correctly returns candidates in semantic similarity order, but the recommender destroys that order when joining candidate ISBNs back to the book metadata DataFrame using `isin()`. The subsequent `head(final_top_k)` therefore selects books according to the original DataFrame order instead of vector similarity rank.

### Classification

**Type:** deterministic implementation defect  
**Layer:** recommender retrieval-to-metadata join / final selection  
**Not classified as:** judge defect, statistical artifact, or LLM nondeterminism

### Why this matters

The system evaluation initially looked like a broad semantic-retrieval failure.

The diagnostic showed a more specific problem:

```text
vector store retrieves ranked semantic candidates
                 |
                 v
ISBN list preserves semantic order
                 |
                 v
DataFrame membership filter
                 |
                 X  ranking lost here
                 |
                 v
head(final_top_k)
                 |
                 v
incorrect final recommendation list
```

The evaluation therefore successfully exposed a production-style quality defect that would not necessarily appear as an exception or test failure in traditional deterministic testing.

---

## 10. Additional implementation observations

Two additional behaviors were identified while inspecting the function.

### 10.1 Category filtering

When:

```python
category != "All"
```

the implementation filters by category but does not apply `final_top_k` within that branch.

This was not the primary cause of the current semantic-only benchmark because the benchmark froze `category = "All"`.

It should nevertheless receive a separate regression test before broader recommender evaluation.

### 10.2 Tone ordering

Tone-based sorting is applied after the `category == "All"` path has already truncated the candidates to `final_top_k`.

Therefore tone ranking operates only on the prematurely selected subset rather than the full eligible candidate set.

Again, this was not active in the current benchmark because `tone = "All"`, but it is a separate design/implementation concern for future evaluation dimensions.

---

## 11. What the investigation does **not** prove

The confirmed ordering bug is significant, but the current evidence does **not** yet prove that it explains every semantic relevance failure.

Specifically, this investigation does not yet establish:

- that the current embedding model is optimal;
- that all 12 queries have strong relevant candidates in the top 50;
- that preserving vector order alone will pass the preregistered gates;
- that the top-50 candidate recall is sufficient;
- that multi-facet semantic intents are represented adequately in the vector index;
- that category/tone behavior is correct;
- that the judge should be retuned.

Those questions must be evaluated after the deterministic ordering defect is fixed.

---

## 12. Why the failed baseline must be preserved

The current evaluation should **not** be deleted, overwritten, or rerun in-place.

It represents valid evidence about the frozen recommender version.

The correct lifecycle is:

```text
Frozen recommender v1
        |
        v
120-pair system evaluation
        |
        v
0/8 gates PASS
        |
        v
failure diagnostics
        |
        v
confirmed ranking-order defect
        |
        v
regression test + minimal fix
        |
        v
freeze recommender v2
        |
        v
fresh system evaluation
        |
        v
compare v1 vs v2
```

This gives a clean before/after engineering record.

---

## 13. Recommended fix strategy

The implementation change should be narrow:

1. Preserve the vector-store candidate order when joining to book metadata.
2. Do not alter the frozen semantic relevance judge.
3. Do not alter the v1 baseline artifacts.
4. Add deterministic regression tests for ranking preservation.
5. Freeze a new recommender source hash.
6. Run the same semantic system benchmark against the corrected recommender.
7. Compare v1 and v2 metrics query-by-query and gate-by-gate.

A robust implementation should explicitly map each ISBN to its semantic rank before the metadata join, or reconstruct the DataFrame in the exact order returned by the vector store.

The precise code change should be made only after the regression test is in place.

---

## 14. Required regression coverage before re-evaluation

At minimum, add tests for:

### Semantic order preservation

Given vector candidates:

```text
ISBN_A
ISBN_B
ISBN_C
```

the returned metadata rows must remain:

```text
ISBN_A
ISBN_B
ISBN_C
```

regardless of their source DataFrame positions.

### `final_top_k`

For `category = "All"`:

```text
returned rows == first final_top_k semantic candidates
```

### Category behavior

Verify:

- semantic order remains stable after category filtering;
- output is bounded by `final_top_k`.

### Tone behavior

Define and test the intended ordering semantics explicitly:

- candidate generation,
- category filtering,
- tone ordering,
- final truncation.

Avoid relying on DataFrame incidental order at any stage.

### Duplicate / missing ISBN behavior

Specify deterministic handling for:

- duplicate ISBNs,
- candidate ISBN not found in metadata,
- duplicate metadata rows.

---

## 15. Re-evaluation plan

After the fix:

### Freeze a new recommender version

Record:

- source SHA-256,
- git commit,
- regression-test results,
- exact implementation change.

### Keep all other benchmark inputs fixed

For the first controlled comparison, retain:

- same 12 queries,
- same judge,
- same rubric,
- same scoring logic,
- same `initial_top_k = 50`,
- same `final_top_k = 10`,
- same system metrics,
- same bootstrap method,
- same gates.

This isolates the effect of the recommender fix.

### Compare v1 vs v2

Report:

- macro mean relevance@10 delta;
- clear/strong rate delta;
- irrelevant rate delta;
- HitRate@5 delta;
- Top-1 delta;
- MRR delta;
- nDCG delta;
- catastrophic-query-rate delta;
- per-query changes;
- bootstrap CIs;
- gate changes.

Do not reuse the v1 final decision as the v2 decision.

---

## 16. Interview case-study summary

A concise way to explain this investigation:

> I built a system-level semantic relevance benchmark for a book recommender using a separately calibrated LLM judge. Before execution, I froze the recommender, judge, queries, metrics, bootstrap method, and release gates. The recommender failed all eight gates: mean relevance was 1.33/4, 42.5% of results were irrelevant, and two-thirds of the benchmark queries were catastrophic.
>
> Rather than immediately tuning the embeddings or judge, I analyzed the failure by query and separated candidate-quality failures from ranking failures. I then inspected the retrieval boundary and found that the vector database correctly returned 50 candidates in semantic similarity order, but the application joined those ISBNs back to a Pandas DataFrame using `isin()`, which discarded the vector ordering. `head(10)` then returned books according to incidental DataFrame order.
>
> I proved the defect deterministically on a dystopian query: `1984` was vector rank 1 but disappeared from the final top 10, while the application's rank 1 result was only vector rank 32. I preserved the failed evaluation as the baseline, planned a regression test before the fix, and would rerun the same frozen benchmark against the corrected recommender to measure the causal impact.

### What this demonstrates

The case study demonstrates:

- preregistered AI-system evaluation;
- separation of judge calibration from system evaluation;
- LLM-as-a-judge used as a measurement instrument;
- query-cluster bootstrap confidence intervals;
- explicit release gates;
- systematic failure triage;
- distinction between retrieval quality and ranking quality;
- deterministic debugging of an AI-adjacent pipeline;
- reproducibility through hashes and frozen artifacts;
- avoiding post-hoc metric or rubric changes;
- preserving failed runs as valid engineering evidence.

---

## 17. Evidence artifacts

The investigation relied on the following repo/runtime artifacts.

### Frozen evaluation / scoring

```text
evals\runs\semantic_relevance_system_eval_v1\
20261002T152856Z_recommendations\
judge_scores.csv

SHA-256:
d079a74c7f0473e46d28f19e37e4241a780804e972f3fa5f25b3a67e62a7bd11
```

### System-evaluation outputs

```text
query_metrics.csv
aggregate_metrics.json
bootstrap_confidence_intervals.json
release_gate_result.json
largest_quality_failures.csv
system_evaluation_report.md
system_evaluation_dashboard.html
analysis_metadata.json
```

### Console evidence

```text
semantic_relevance_system_eval_analysis.txt
semantic_relevance_system_eval_failure_diagnostic.txt
semantic_recommender_function_diagnostic.txt
semantic_retrieval_order_diagnostic.txt
```

### Frozen recommender source

```text
gradio-dashboard.py

SHA-256:
b8f7c1223a525345a7e278fd29e0cd32a5e0c644a9a0b88e5cc9ac319f749974
```

---

## 18. Audit conclusion

The first semantic-relevance system evaluation produced a valid and useful **FAIL** result.

The subsequent investigation identified a confirmed deterministic implementation defect at the semantic-candidate-to-metadata boundary:

**semantic similarity order is lost during the Pandas `isin()` join.**

The correct next engineering step is not to modify the judge or benchmark. It is to add regression coverage, make a minimal recommender fix that preserves vector order, freeze a new recommender version, and rerun the same benchmark for a controlled v1-to-v2 comparison.

The failed v1 evaluation should remain permanently available as the baseline evidence that led to the defect discovery.
