# Semantic relevance top-50 diagnostic — Checkpoint A

This package implements **Checkpoint A only** from the top-50 diagnostic design.

It is a read-only diagnostic against the frozen v2 semantic candidate generator.
It makes **zero judge calls** and changes neither the recommender nor the frozen
v1/v2 evaluation artifacts.

## What it proves

For each of the 12 frozen semantic queries, the collector directly requests:

```python
db_books.similarity_search(query, k=50)
```

and records the exact vector order.

Before freezing the 600-row candidate artifact, it requires both:

1. vector ranks 1-10 exactly reproduce the frozen v2 final top 10, in order;
2. every frozen v1 top-10 query-book pair still exists somewhere in the current
   vector top 50.

If either invariant fails, the script stops. That protects the diagnostic from
silently analyzing a drifted vector store or environment.

## Important Git behavior

The collector verifies the frozen v2 recommender **source hash**:

```text
cdd0458ff23bd669185a8c882ceb132438d037af1a29f4dd60bd7fef9c405b60
```

It deliberately does **not** require current `HEAD` to equal the historical v2
freeze commit. The repository may contain legitimate later analysis/report
commits after v2 was merged; the system-under-test source must remain frozen.

## Overlay

From the repository root, extract this package and copy its contents over the
repo. It adds:

```text
collect_semantic_relevance_top50_checkpoint_a.ps1
evals/system_evaluation/collect_semantic_relevance_top50_candidates.py
```

No existing v1/v2 run artifact is overwritten.

## Run

From the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\collect_semantic_relevance_top50_checkpoint_a.ps1 2>&1 |
  Tee-Object -FilePath .\semantic_relevance_top50_checkpoint_a.txt
```

Expected successful ending:

```text
TOP-50 DIAGNOSTIC CHECKPOINT A: PASS
Candidates: 600
Judge calls performed: 0
Human relevance labels used: NO
```

The run is created under:

```text
evals/runs/semantic_relevance_top50_diagnostic_v1/<timestamp>_candidates/
```

Primary artifacts:

```text
top50_candidates.csv
top50_reproduction_checks.csv
top50_collection_metadata.json
top50_collection_lock.json
```

## Stop boundary

Do **not** begin score reuse or novel candidate judging after this command.
Upload `semantic_relevance_top50_checkpoint_a.txt` for review first.

Checkpoint B will only be prepared after the candidate-pool invariants pass.
