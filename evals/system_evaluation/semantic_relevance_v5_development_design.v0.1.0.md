# Semantic Relevance V5 — Factorized Development Design

## Decision

V5 will not begin with one large architecture change. Frozen V4H evidence shows two different failure mechanisms: some queries had clear/strong material inside the candidate pool but outside the displayed top 10, while others had no clear/strong result anywhere in the frozen V4B full union. The next development cycle therefore isolates **candidate generation (R)** from **ranking/display selection (K)**.

The four frozen experiment cells are:

| System | Candidate generation | Ranking | Purpose |
|---|---|---|---|
| B | R0 = frozen V4B | K0 = frozen V4B | baseline |
| R | R1 = new candidate generation | K0 | retrieval-only effect |
| K | R0 | K1 = new ranking | ranking-only effect |
| RK | R1 | K1 | combined effect / interaction |

R1 and K1 are not selected from V4H query-by-query behavior. Their exact implementations, models/prompts, candidate budgets, merge rules, scoring rules, and tests must be frozen in the **next package before fresh V5D authoring or execution**.

## Why factorize instead of shipping a combined V5 change

The V4H closeout established:

- 4/10 original `NO_CLEAR_TOP10` queries had `HEADROOM_FOUND_11_50`;
- 0/10 were rescued only at ranks 51-end;
- 6/10 had `NO_CLEAR_THROUGH_FULL_UNION`.

That is evidence for both ranking/display and candidate-generation work. Changing both at once would make a win or loss difficult to attribute. A 2×2 design measures retrieval and ranking effects separately and also exposes interaction.

## V4H authority boundary

V4H is consumed and remains a frozen **FAIL**. It may justify the *classes* of hypotheses investigated in V5, but it may not be used to choose V5 parameters, prompts, model revisions, thresholds, query-specific rules, or examples. V4H query text, facets, candidate IDs, and judge labels are prohibited from V5 tuning.

## Fresh V5 development selection set

V5 uses a new 24-query selection set, `V5D01-V5D24`, with four queries in each of the six established strata. Authoring is blind to catalog contents, retrieval outputs, candidate-system outputs, and judge outputs. Exact/paraphrased reuse of H01-H24 or V4H01-V4H24 is prohibited.

The V5D query and facet files are frozen before any V5D retrieval. Once the first semantic V5D result exists, no implementation or parameter change is permitted for this V5 design; a change requires a new version and treats V5D as consumed evidence.

## Candidate budget and diagnostic depth

R1 may produce at most 100 unique pre-rerank candidates per query, matching the V4B dense50 + BM25-50 maximum. This prevents "more candidates" from being the only experimental difference. Any multi-source merge, deduplication, and truncation must be deterministic and frozen with R1.

Candidate-generation diagnostics are evaluated at depth 50. System selection is still based on displayed top-10 quality; retrieval diagnostics explain *why* a system moved but cannot override selection.

## System selection

The existing lexicographic development rule is retained:

1. lower catastrophic-query rate;
2. lower irrelevant rate at 10;
3. higher clear/strong rate at 10;
4. higher macro mean relevance at 10;
5. higher nDCG at 10.

All four systems are run on the same frozen V5D set. If the baseline B is exactly tied for best on every criterion, retain B. Otherwise prefer fewer changed factors on an exact tie. A remaining exact R-vs-K tie uses lower median system inference latency excluding judge time; if that also ties, no candidate is selected without a separately preregistered tie-break.

## Final holdout

Only a non-baseline V5 candidate selected on V5D may proceed to a new unseen `V5H01-V5H24` holdout. The candidate is frozen before V5H authoring. The original eight release gates remain unchanged.

## Next step

Freeze the exact **R1 candidate-generation** and **K1 ranking** specifications and implementations—including all model IDs/revisions, prompts/templates, depths, merge rules, scoring/aggregation, tie-breaking, and repeatability tests—before V5D authoring or semantic execution.
