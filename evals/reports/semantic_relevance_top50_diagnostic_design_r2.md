# Semantic Relevance Top-50 Diagnostic — Design and Analysis Contract

**Project:** Book Recommender  
**Stage:** post-v2 diagnostic, before v3 reranker design  
**Status:** DESIGN — no new judge calls performed by this document  
**Recommended repo location:** `evals/reports/semantic_relevance_top50_diagnostic_design.md`

---

## 1. Purpose

The v2 controlled evaluation established that preserving vector-store semantic order materially improved recommendation quality, but v2 still failed all frozen release gates.

Post-v2 analysis identified two residual patterns:

1. relevant candidates sometimes exist in the current v2 top 10 but are ranked too low;
2. relevant candidates can exist below vector rank 10 but within the original top-50 candidate pool.

The next question is therefore deliberately narrower than "build a reranker":

> **Does the unchanged v2 top-50 semantic candidate pool contain enough relevant material for a top-50-to-top-10 reranker to be a credible next experiment?**

This diagnostic must answer that question before a v3 reranker implementation is selected or tuned.

---

## 2. Scope and non-goals

### In scope

The diagnostic will:

- use the same 12 semantic benchmark queries;
- use the unchanged v2 semantic candidate generator;
- reconstruct and freeze a vector-ranked top 50 for every query using the frozen v2 candidate generator;
- preserve vector rank explicitly;
- reuse existing frozen v1/v2 judge scores where possible;
- score only previously unscored query-book pairs with the same pinned calibrated judge;
- quantify relevance availability at ranks 1-10 and 11-50;
- calculate the theoretical "oracle" headroom available to any reranker restricted to the same top-50 pool;
- identify queries where candidate retrieval itself appears insufficient.

### Out of scope

The diagnostic will not:

- modify the recommender;
- implement a reranker;
- change the embedding model;
- change indexed book text;
- increase retrieval depth above 50;
- change the calibrated semantic-relevance judge;
- change system release gates;
- produce a new release decision;
- use human relevance labels;
- overwrite any v1 or v2 artifact.

---

## 3. Frozen provenance

The diagnostic must use the v2 recommender implementation whose frozen source hash is:

```text
cdd0458ff23bd669185a8c882ceb132438d037af1a29f4dd60bd7fef9c405b60
```

The diagnostic should verify the **recommender source hash**, not require the current Git commit to equal the historical v2 freeze commit. The repository may legitimately contain later analysis/report commits while the recommender source remains unchanged.

The same frozen query file and system-evaluation contract used by v1/v2 must also be verified before candidate collection.

The v2 candidate-generation parameters remain:

```text
queries              = 12
category             = All
tone                 = All
initial_top_k        = 50
final_top_k          = 10
```

The top-50 diagnostic changes only what is *observed*: it records all 50 retrieved candidates rather than truncating the evidence to the final ten.

---

## 4. Diagnostic lifecycle

```text
unchanged v2 candidate generator
        |
        v
collect exact vector top 50
12 x 50 = 600 query-book pairs
        |
        v
reproducibility checks
        |
        +-- current vector ranks 1-10 must reproduce frozen v2 top 10
        |
        +-- historical v1 top-10 containment is recorded as a non-blocking audit
        |
        v
freeze top-50 candidate artifact
        |
        v
join existing v1/v2 frozen judge scores
        |
        v
reuse known scores
expected unique known pairs ~= 216
        |
        v
prepare judge input only for novel pairs
expected novel pairs ~= 384
        |
        v
score novel pairs with same pinned judge
checkpoint/resume enabled
        |
        v
freeze complete 600-pair diagnostic score map
        |
        v
top-50 headroom analysis
        |
        v
decision:
reranker-supported / retrieval-limited / mixed
```

The values `216 known / 384 novel` are expected from the current v1-v2 recommendation-level comparison. The implementation must recompute them rather than hard-code them.

---


## 5. Worked example — what the diagnostic is trying to prove

This example is intentionally illustrative. It uses the known Q10-style failure pattern from v2, but the hypothetical candidates and scores shown at vector ranks 11-50 are **not actual diagnostic results**. Their purpose is to show how the evidence will be interpreted once the real top-50 artifact is collected and scored.

### 5.1 Starting point: what v2 currently sees

Suppose the frozen query is:

> A book about leadership, teamwork, and building effective organizations

The v2 top 10 already tells us something useful:

```text
vector rank 2   partial match   score 2
vector rank 3   partial match   score 2
vector rank 6   partial match   score 2
vector rank 7   clear match     score 3
vector rank 8   clear match     score 3
```

The important problem is that the best known matches are relatively low in the top 10.

That means the current vector search is finding some relevant books, but vector similarity alone is not ordering them optimally for the full multi-facet query.

However, this top-10 view cannot answer the more important question:

> Are even better candidates sitting at vector ranks 11-50?

That is what the top-50 diagnostic is designed to discover.

### 5.2 Phase A: collect the full top 50

The diagnostic records the exact vector-ranked candidate pool:

| Vector rank | Candidate | Previously in v1/v2 top 10? | Judge score known? |
|---:|---|---|---|
| 1 | Candidate A | v2 | Yes |
| 2 | Candidate B | v2 | Yes |
| ... | ... | ... | ... |
| 7 | Known clear candidate | v2 | Yes |
| 8 | Known clear candidate | v2 | Yes |
| 11 | Candidate K | No | No |
| 12 | Candidate L | No | No |
| ... | ... | ... | ... |
| 18 | Candidate R | Maybe v1 | Maybe |
| ... | ... | ... | ... |
| 50 | Candidate AX | No | No |

At this stage there are **no new judge calls**.

The diagnostic first verifies:

```text
ranks 1-10 == frozen v2 top 10
```

and also verifies that every frozen v1 top-10 candidate still exists somewhere in ranks 1-50.

If either check fails, the experiment stops because the candidate pool has drifted.

### 5.3 Phase B: reuse existing scores

Assume the top 50 contains 18 books that were already judged in v1 or v2.

Those scores are reused directly.

For example:

```text
rank 7   known clear candidate     score 3   source=v2
rank 8   known clear candidate     score 3   source=v2
rank 18  known v1 candidate        score 3   source=v1
```

The remaining 32 candidates are marked:

```text
UNSCORED
```

Only those novel query-book pairs move to the next phase.

### 5.4 Phase C: score only the novel candidates

The same pinned calibrated semantic-relevance judge scores the 32 previously unseen candidates.

Imagine the result is:

```text
rank 12  score 2
rank 16  score 3
rank 21  score 4
rank 27  score 3
rank 36  score 1
...
```

Now the complete 50-candidate pool has comparable semantic-relevance scores.

At this point the diagnostic can answer a question that the current v2 evaluation cannot:

> How much good material exists below vector rank 10?

In this illustrative example, the answer would be:

```text
known clear/strong in ranks 1-10:   2
additional clear/strong in 11-50:   4
best deeper candidate score:        4
```

That would be strong evidence that candidate generation is better than the final v2 top 10 suggests.

### 5.5 Phase D: build the diagnostic oracle top 10

The oracle is constructed only for analysis.

It selects the ten highest-scoring books from the 50 and orders them by:

```text
judge score descending
then original vector rank ascending
```

Suppose v2's actual top 10 contains:

```text
scores:
2, 2, 1, 0, 2, 2, 3, 3, 2, 2
```

while the top-50 oracle could select:

```text
scores:
4, 3, 3, 3, 3, 2, 2, 2, 2, 2
```

The important comparison is then:

| Metric | v2 actual | Top-50 oracle |
|---|---:|---:|
| Mean relevance@10 | 1.9 | 2.6 |
| Clear/strong rate | 20% | 50% |
| Irrelevant rate | 10% | 0% |
| Top-1 clear/strong | No | Yes |
| HitRate@5 | Yes/No depending on actual order | Yes |
| nDCG | lower | 1.0 by construction |

Again, these numbers are illustrative only.

The value of the oracle is not that we can deploy it. It uses judge labels and therefore cannot become production ranking logic.

Its value is that it establishes a ceiling:

> If a perfect selector restricted to the same top 50 could perform much better than v2, then the candidate pool contains useful headroom and reranking is a credible next experiment.

### 5.6 What if the result looks like Q12 instead?

Now consider the opposite illustrative outcome.

Suppose a difficult query's full top 50 contains:

```text
score 4: 0 books
score 3: 1 book
score 2: 6 books
score 1: 8 books
score 0: 35 books
```

Even a perfect oracle can only build a weak final top 10.

For example:

```text
oracle clear/strong rate: 10%
oracle mean relevance:    still well below target
oracle irrelevant rate:   still too high
```

In that case, no reranker restricted to those same 50 candidates can manufacture relevant books that the candidate generator failed to retrieve.

That query would be classified as **retrieval-limited**.

The next investigation would then move upstream toward:

```text
query representation
indexed book text
embedding model
candidate depth
retrieval strategy
```

rather than trying to solve the problem with ranking alone.

### 5.7 How the diagnostic leads to a decision

The complete picture is therefore:

```text
                    CURRENT v2
                       |
                       v
              vector-ranked top 10
                       |
                       v
              quality still inadequate
                       |
                       v
          inspect unchanged vector top 50
                       |
             +---------+---------+
             |                   |
             v                   v
      good candidates        good candidates
      exist at 11-50         mostly absent
             |                   |
             v                   v
      oracle top 10          oracle top 10
      improves strongly      remains weak
             |                   |
             v                   v
   RERANKER_SUPPORTED     RETRIEVAL_LIMITED
             \                   /
              \                 /
               +-------+-------+
                       |
                       v
                  MIXED is also
                  possible across
                  different queries
```

A realistic benchmark result may be **MIXED**:

```text
Q03 / Q10:
strong reranking headroom

Q01 / Q11:
useful candidates exist deeper in top 50

Q12:
candidate pool may still be weak
```

That would still justify a controlled reranker experiment because it can improve the ranking-limited queries, while separately documenting that some remaining failures require candidate-retrieval work.

### 5.8 The key distinction

The diagnostic is not asking:

> "Can we make the benchmark look better by sorting on judge scores?"

It is asking:

> "Does the unchanged retrieval system already contain enough relevant candidates for a real reranker to have a plausible opportunity to improve the final top 10?"

The judge-score oracle is only a measurement tool for estimating that opportunity.

---


## 6. Phase A — collect the exact top-50 candidate pool

### 5.1 Collection method

For every frozen query, execute the same v2 semantic retrieval call:

```python
recs = db_books.similarity_search(query, k=50)
```

The collector must record each returned document in exact vector-store order.

The collector must **not** call the final recommender and ask for 50 final recommendations if doing so would exercise category/tone/final-truncation behavior. The purpose is to observe the unchanged semantic candidate generator directly.

### 5.2 Candidate artifact

Primary artifact:

```text
top50_candidates.csv
```

Expected rows:

```text
12 queries x 50 candidates = 600 rows
```

Required fields:

| Field | Meaning |
|---|---|
| `query_id` | frozen query identifier |
| `query` | frozen query text |
| `vector_rank` | 1..50 in retrieval order |
| `isbn13` | candidate book identity |
| `title` | metadata title |
| `description` | book description available to evaluation/reranking |
| `authors` | when available |
| `retrieval_document_text` | exact vector-store document text returned for this candidate |
| `in_frozen_v2_top10` | whether pair appears in frozen v2 top 10 |
| `frozen_v2_rank` | v2 final rank when present |
| `in_frozen_v1_top10` | whether pair appears in frozen v1 top 10 |
| `frozen_v1_rank` | v1 final rank when present |

No semantic relevance score is generated in Phase A.

### 5.3 Required collection assertions

The collector must fail closed if any of these checks fails:

**A. Exactly 12 frozen queries**

Each query must produce exactly 50 candidates.

**B. Exactly 600 query-book rows**

No query may silently return fewer than 50 candidates.

**C. Unique candidate identity within each query**

Duplicate ISBN handling must be explicit. A duplicate cannot silently occupy two ranks.

**D. v2 top-10 reproduction**

For every query:

```text
new top50 vector ranks 1..10
==
frozen v2 top10 ISBNs in the same order
```

This is the strongest practical check that candidate retrieval has not drifted since the v2 evaluation.

**E. historical v1 candidate-pool containment — diagnostic, not a hard gate**

Every frozen v1 top-10 query-book pair historically came from that run's top-50 membership pool. However, ranks 11-50 were never frozen for v1 or v2, so a later top-50 reconstruction cannot require exact historical tail membership.

The collector must therefore record, for every query:

- which historical v1 top-10 pairs are present in the reconstructed top 50;
- their reconstructed vector ranks when present;
- which historical v1 pairs are absent.

Missing historical v1-only candidates do **not** invalidate Checkpoint A as long as all 12 frozen v2 top-10 lists reproduce exactly. They are retained as evidence that deeper candidate membership can vary across retrieval executions.

The hard reproducibility anchor is the frozen v2 top 10, because those ranks were actually persisted.

---

## 7. Phase B — reuse existing frozen judge evidence

### 6.1 Existing score sources

Use the frozen:

```text
v1 judge_scores.csv
v2 judge_scores.csv
```

as immutable score sources.

Scores are joined by:

```text
(query_id, isbn13)
```

not by rank.

### 6.2 Shared-score integrity

If the same query-book pair exists in both v1 and v2 score artifacts:

```text
v1 judge_score must equal v2 judge_score
```

The previous recommendation-level analysis found zero score mismatches among shared pairs. The diagnostic must verify this again programmatically.

If a conflict is found, stop rather than choosing one score arbitrarily.

### 6.3 Reuse artifact

Produce:

```text
top50_score_reuse_map.csv
```

Suggested fields:

```text
query_id
query
vector_rank
isbn13
title
judge_score
score_source
source_rank
```

`score_source` should distinguish at least:

```text
v1
v2
v1_and_v2
UNSCORED
```

---

## 8. Phase C — score only novel top-50 pairs

### 7.1 Why reuse scores

The top-50 pool contains 600 query-book pairs.

The v1 and v2 system evaluations already provide scores for a substantial subset of these pairs. Re-running the calibrated judge on those books would add cost and another opportunity for stochastic variation without adding useful information.

Therefore only rows with:

```text
score_source == UNSCORED
```

will be sent to the judge.

The previous v1/v2 comparison provides an upper bound of 216 unique previously scored pairs, but deep-tail reconstruction can exclude some historical v1-only pairs. Therefore the exact reuse count must be computed after Checkpoint A:

```text
600 total
- N previously scored pairs that are actually present in the reconstructed top 50
= 600 - N novel pairs
```

The exact count must be computed after Phase A.

### 7.2 Judge configuration

Novel candidates must use the **same pinned calibrated semantic relevance judge** used for v1 and v2.

No judge prompt, rubric, facet specification, scoring scale, or execution policy may be changed for this diagnostic.

### 7.3 Execution behavior

The scoring harness should support:

- checkpointing;
- safe resume;
- immutable completed rows;
- duplicate-case detection;
- score range validation;
- provenance hashes;
- zero human relevance labels.

The diagnostic scoring namespace must be separate from the frozen v1/v2 run directories.

### 7.4 Diagnostic score artifact

After combining reused and newly generated scores, create:

```text
top50_judge_scores_complete.csv
```

Expected rows:

```text
600
```

Every `(query_id, isbn13)` pair must have exactly one final diagnostic judge score.

---

## 9. Phase D — top-50 headroom analysis

This stage does **not** evaluate a real v3 system.

It asks what is theoretically possible if a future reranker can choose and order books only from the unchanged top-50 pool.

### 8.1 Per-query candidate-depth metrics

For each query calculate:

```text
clear_or_strong_count_at_10
clear_or_strong_count_at_20
clear_or_strong_count_at_30
clear_or_strong_count_at_40
clear_or_strong_count_at_50

clear_or_strong_count_ranks_11_50
partial_or_better_count_ranks_11_50

best_score_at_10
best_score_ranks_11_50
best_score_at_50

first_clear_or_strong_vector_rank
first_strong_vector_rank
```

Also report score distributions separately for:

```text
ranks 1-10
ranks 11-50
ranks 1-50
```

This directly answers whether deeper retrieval contains usable relevance.

### 8.2 Known-good-candidate rank mapping

For every previously scored clear/strong v1 or v2 recommendation, report its actual vector rank in the new top-50 artifact.

Examples of particular interest from post-v2 analysis include:

```text
Q01: Gilead
Q11: The Epic of Gilgamesh
Q11: Bakkhai
```

The diagnostic should determine their exact vector ranks rather than merely infer that they were somewhere below rank 10.

### 8.3 Oracle top-10 construction

For each query construct a diagnostic **oracle top 10** by:

1. selecting the ten highest judge-scored books from the 50;
2. sorting by judge score descending;
3. using original vector rank ascending as the deterministic tie-breaker.

This is not a proposed production algorithm. It is a ceiling analysis that uses evaluation labels and must never be incorporated into the recommender.

### 8.4 Oracle metrics

Calculate the same semantic-quality metrics on the oracle top 10:

```text
macro mean relevance@10
clear/strong rate@10
irrelevant rate@10
HitRate@5 clear/strong
Top-1 clear/strong
MRR@10 clear/strong
nDCG@10
catastrophic-query rate
```

Compare:

```text
v2 actual top10
vs
top50 oracle top10
```

Do **not** issue a release PASS/FAIL decision for the oracle.

### 8.5 Why the oracle is useful

For any metric that the top-50 oracle still cannot satisfy on this benchmark:

> no reranker restricted to these same 50 candidates can satisfy that metric under the same judge labels.

For any metric that the oracle can satisfy:

> a top-50 reranker has theoretical headroom, but a real reranker is not guaranteed to achieve it.

This gives a strong boundary between:

```text
ranking/selection limitation
```

and:

```text
candidate-pool limitation
```

---

## 10. Primary diagnostic questions

The analysis report must answer these questions explicitly.

### Q1 — Is there relevant material below rank 10?

Across all 12 queries, how many clear/strong candidates exist at vector ranks 11-50?

### Q2 — Is the evidence broad or query-specific?

Are deeper relevant candidates present across most queries, or only a few?

### Q3 — What happens for Q12?

Does Q12 contain additional clear/strong candidates at ranks 11-50?

This is the critical warning case from the v2 post-analysis.

### Q4 — How much oracle headroom exists?

How far do the oracle top-10 metrics improve over v2?

### Q5 — Could a top-50 reranker theoretically cross the existing gates?

Which frozen gate thresholds are theoretically reachable from the current candidate pool?

### Q6 — Which gates remain impossible even with oracle selection?

Those metrics identify limitations that require candidate-generation changes rather than reranking alone.

---

## 11. Decision framework

The diagnostic conclusion should use one of three evidence labels.

### `RERANKER_SUPPORTED`

Use when the top-50 pool contains substantial relevant material below rank 10 and oracle selection shows meaningful headroom in both ranking-sensitive and composition metrics.

Interpretation:

> proceed to a controlled v3 reranker experiment while keeping v2 candidate generation fixed.

### `RETRIEVAL_LIMITED`

Use when the top-50 oracle still has weak relevance composition and important failure queries, especially Q12, do not contain enough relevant candidates.

Interpretation:

> improving candidate generation should precede or accompany reranking.

### `MIXED`

Use when some queries show strong reranking headroom while others remain candidate-pool limited.

Interpretation:

> reranking remains justified as a controlled experiment, but it should not be expected to solve all semantic-relevance failures.

These are **diagnostic labels**, not release decisions.

No numeric diagnostic threshold should be invented after observing the results. The report should show the underlying counts and oracle metrics alongside the label.

---

## 12. Benchmark-contamination boundary

The 12-query benchmark has now been used for:

- v1 system evaluation;
- v1 failure diagnosis;
- v2 controlled evaluation;
- v1-v2 paired analysis;
- recommendation-level diagnosis;
- and, under this design, top-50 relevance analysis.

Therefore these 12 queries should be treated from this point forward as a:

```text
development / diagnostic system benchmark
```

They remain highly useful for:

- controlled v2-v3 comparisons;
- regression detection;
- causal engineering analysis;
- checking whether the reranker addresses known failure modes.

However, after the v3 architecture and implementation are frozen, a **separate unseen system-level holdout** should be executed before making a final unbiased generalization/release claim.

The v3 reranker must not be trained, prompted with, or hard-coded against judge labels from this diagnostic.

---

## 13. Expected artifacts

Recommended run namespace:

```text
evals/runs/semantic_relevance_top50_diagnostic_v1/<run_id>/
```

Expected artifacts:

```text
top50_candidates.csv
top50_collection_metadata.json
top50_collection_lock.json

top50_score_reuse_map.csv
top50_novel_judge_input.csv

top50_judge_scores_new.csv
top50_judge_scores_complete.csv
top50_scoring_lock.json

top50_query_metrics.csv
top50_depth_profile.csv
top50_oracle_top10.csv
top50_oracle_metrics.json
top50_diagnostic_report.md
top50_diagnostic_metadata.json
```

Console wrappers should write their full output to files so diagnostics are not lost to terminal scrollback.

---

## 14. Execution order

The diagnostic should be implemented and reviewed in four checkpoints:

```text
Checkpoint A
collect + freeze top-50 candidates
NO judge calls

Checkpoint B
join/reuse v1-v2 scores
prepare novel-only judge input
NO judge calls

Checkpoint C
score only novel pairs
freeze complete score map

Checkpoint D
analyze depth + oracle headroom
produce diagnostic conclusion
```

Do not proceed from one checkpoint to the next if its integrity checks fail.

---

## 15. Immediate next implementation step

The first code change should implement **Checkpoint A only**.

It should:

- load the frozen 12 queries;
- load the backend without launching Gradio;
- directly call the unchanged vector store for `k=50`;
- preserve exact vector rank;
- join book metadata without altering order;
- compare ranks 1-10 against frozen v2 top-10 output;
- verify all frozen v1 top-10 candidates are contained within top 50;
- write and hash the 600-row candidate artifact;
- perform **zero judge calls**.

Only after that artifact passes review should the score-reuse/scoring stages be implemented.


---

## Design amendment after Checkpoint A r1

Checkpoint A r1 reproduced the frozen v2 head through Q07 but found one historical
v1-only candidate absent from the newly reconstructed Q07 top 50. This exposed an
overly strong assumption in the original design: historical ranks 11-50 were not
frozen, so exact deep-tail containment cannot be enforced retroactively.

The amended policy is:

```text
frozen v2 ranks 1-10 exact reproduction  -> HARD GATE
historical v1 top-10 containment          -> DIAGNOSTIC / NON-BLOCKING
```

The 600-row top-50 artifact produced by the successful amended Checkpoint A is
then frozen and becomes the authoritative candidate pool for all later
top-50 diagnostic stages.
