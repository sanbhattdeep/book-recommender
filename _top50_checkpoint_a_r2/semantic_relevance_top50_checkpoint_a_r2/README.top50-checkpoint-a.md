# Semantic relevance top-50 diagnostic — Checkpoint A r2

This package implements **Checkpoint A only** from the top-50 diagnostic design.

It is a read-only diagnostic using the frozen v2 semantic candidate generator.
It makes **zero judge calls** and changes neither the recommender nor the frozen
v1/v2 evaluation artifacts.

## Important correction from r1

The original Checkpoint A treated this as a hard invariant:

> every historical v1 top-10 candidate must still appear in a later reconstructed top 50.

That is too strong. The historical ranks 11-50 were never frozen. A v1 book is
proof that it belonged to the **historical v1** top-50 pool, but it is not proof
that the same deep-tail candidate must appear in a fresh reconstruction.

Checkpoint A r2 therefore uses:

**Hard reproducibility gate**

```text
current vector ranks 1-10 == frozen v2 top 10, in exact order
```

for all 12 queries.

**Diagnostic-only audit**

```text
how many historical v1 top-10 pairs still appear in the current reconstructed top 50?
```

Missing historical v1-only candidates are recorded in the check CSV, metadata,
lock file, and console output, but do not abort collection.

## What is being frozen

The historical v2 ranks 11-50 were never persisted. This diagnostic therefore
creates a **fresh top-50 reconstruction from the frozen v2 candidate generator**
and frozen benchmark inputs.

Once Checkpoint A succeeds, that 600-row reconstruction is frozen and becomes
the candidate pool for Checkpoints B-D.

## Hard checks

The collector requires:

1. frozen query hash unchanged;
2. frozen evaluation-contract hash unchanged;
3. frozen v2 recommender source hash unchanged;
4. frozen v1 and v2 recommendation artifacts unchanged;
5. exactly 12 queries x 50 candidates = 600 rows;
6. no duplicate query-book pair within a query;
7. exact frozen v2 top-10 reproduction for all 12 queries.

Historical v1 containment is reported but is not a hard gate.

## Frozen recommender source

```text
cdd0458ff23bd669185a8c882ceb132438d037af1a29f4dd60bd7fef9c405b60
```

Current Git `HEAD` is intentionally not required to equal the historical v2
freeze commit; later analysis/report commits are allowed.

## Run

From the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\collect_semantic_relevance_top50_checkpoint_a.ps1 2>&1 |
  Tee-Object -FilePath .\semantic_relevance_top50_checkpoint_a_r2.txt
```

Expected successful ending:

```text
Frozen v2 top-10 reproduction: 12/12 PASS
Top-50 rows:                   600/600

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
Upload the captured console output for review first.
