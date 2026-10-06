# Semantic relevance top-50 diagnostic — Checkpoint C

Checkpoint C is the **first top-50 diagnostic stage that calls the calibrated
LLM judge**.

It scores only the 387 novel query-book pairs frozen by Checkpoint B and reuses
the 213 historical v1/v2 scores without rejudging them.

## Frozen Checkpoint B inputs

```text
top50_score_reuse_map.csv
SHA-256:
603ca8e093fa7720c9cd9d9bb7787780316875f33ec37652894b44195fa565b4

top50_novel_judge_input.csv
SHA-256:
d1205259ff9399c112c282d7142bd12589a8d3a59cadda8e65f526be32401b82
```

Expected split:

```text
reused = 213
novel  = 387
total  = 600
```

## Judge contract

Checkpoint C uses the same calibrated judge contract as v1/v2:

```text
semantic_relevance_v0.29.0-r5
facet spec v0.10.1
same semantic scoring implementation
same book-subject-retryfix1 execution-only amendment when applicable
human relevance labels = NOT USED
```

Before creating the judge model, the script verifies the Checkpoint A/B locks,
artifact hashes, frozen semantic scoring hash, facet-spec hash, judge-config
hash, contract hash, and frozen judge / accepted retryfix1 provenance.

## Restart safety

Every completed novel judgment is immediately persisted to:

```text
top50_judge_scores_new.csv
```

On rerun, the scorer verifies every persisted case against the frozen novel
input and **does not rejudge completed case IDs**.

If the process or machine stops after N cases, rerun the same command.

## Optional bounded invocation

The wrapper supports an optional case cap:

```powershell
.\score_semantic_relevance_top50_checkpoint_c.ps1 -MaxNew 10
```

This scores at most ten additional cases and exits successfully with a partial
checkpoint. A later run resumes from those persisted results.

With no `-MaxNew`, all remaining novel cases are attempted.

## Full run

From the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\score_semantic_relevance_top50_checkpoint_c.ps1 2>&1 |
  Tee-Object -FilePath .\semantic_relevance_top50_checkpoint_c.txt
```

The wrapper keeps Windows system/display sleep blocked while scoring and clears
the execution-state override on exit.

## Successful final outputs

After 387/387 novel cases complete:

```text
top50_judge_scores_new.csv
top50_judge_scores_complete.csv
top50_judge_scoring_metadata.json
top50_scoring_lock.json
```

`top50_judge_scores_complete.csv` contains all 600 frozen candidates:

```text
213 reused historical judgments
387 newly judged cases
```

The complete score map is the frozen input for Checkpoint D.

## Deliberate stop boundary

Checkpoint C does **not** calculate:

- aggregate top-50 metrics;
- depth profiles;
- oracle top-10 metrics;
- bootstrap confidence intervals;
- release gates.

Upload the final console output before Checkpoint D.
