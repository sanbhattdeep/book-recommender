# Semantic relevance top-50 diagnostic — Checkpoint B

Checkpoint B prepares frozen-score reuse and the **novel-only judge input** for
the top-50 diagnostic. It makes **zero judge calls**.

## Frozen inputs

Checkpoint B is bound to the successful Checkpoint A candidate artifact:

```text
evals/runs/semantic_relevance_top50_diagnostic_v1/
20261005T113402Z_candidates/top50_candidates.csv

SHA-256:
679dc2a8a3607db8d98e4ee868b5d7fc6566d11cb235eebed24e4676240abbad
```

It also verifies the frozen v1/v2 judge inputs and score files by SHA-256.

## Reuse policy

A historical score is reused only when all of the following hold:

1. `(query_id, isbn13)` matches a current top-50 candidate;
2. the historical v1/v2 semantic outcomes do not conflict;
3. the current `query`, `title`, `description`, and `authors` payload exactly
   matches the frozen historical judge input.

A payload mismatch is treated as a provenance failure. The script does not
silently reuse the score and does not silently rejudge the changed payload.

## Expected deterministic counts

For the frozen Checkpoint A artifact:

```text
historical v1/v2 shared pairs     24
historical semantic conflicts      0
payload mismatches                  0

reusable v1-only pairs             93
reusable v2-only pairs             96
reusable v1+v2 pairs               24
-------------------------------------
reusable total                    213

novel pairs                       387
total                             600
```

These values are recomputed, then asserted. They are not used to manufacture
the output.

## Outputs

Checkpoint B writes into the existing Checkpoint A run directory:

```text
top50_score_reuse_map.csv
top50_novel_judge_input.csv
top50_existing_score_conflicts.csv
top50_payload_mismatches.csv
top50_score_reuse_metadata.json
top50_score_reuse_lock.json
```

`top50_novel_judge_input.csv` is the only set that should be sent to the
calibrated judge in Checkpoint C.

## Run

From the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\prepare_semantic_relevance_top50_checkpoint_b.ps1 2>&1 |
  Tee-Object -FilePath .\semantic_relevance_top50_checkpoint_b.txt
```

Expected successful integrity summary:

```text
Historical shared v1/v2 pairs: 24/24
Historical score conflicts:    0
Historical payload mismatches: 0
Reusable pairs total:          213/600
Novel judge-input pairs:       387/600

TOP-50 DIAGNOSTIC CHECKPOINT B: PASS
Judge calls performed: 0
```

## Stop boundary

Do **not** start Checkpoint C judging automatically.

Upload `semantic_relevance_top50_checkpoint_b.txt` for review first. The
novel-input hash and per-query novel counts should be reviewed before any new
judge calls are made.
