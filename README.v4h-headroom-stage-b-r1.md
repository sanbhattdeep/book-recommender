# V4H NO_CLEAR_THROUGH_50 Headroom Diagnostic — Stage B r1

V4H is consumed diagnostic evidence and the frozen V4B release decision remains `FAIL`.

Stage A classified the 10 `NO_CLEAR_TOP10` queries using the frozen V4B ranks 11–50:

- 4/10: `HEADROOM_FOUND_11_50`
- 6/10: `NO_CLEAR_THROUGH_50`

Stage B is allowed only for those six unresolved queries. This package does **not** perform retrieval, reranking, judging, model calls, release analysis, or release-gate recalculation.

It derives the exact six-query target set from the frozen Stage-A classification artifact and freezes every candidate at V4B rank 51 through the end of each query's already-frozen full-union ranking. The four Stage-A headroom-found queries are explicitly excluded.

## Step 1 — zero-call preflight

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\prepare_semantic_relevance_v4h_headroom_stage_b.ps1 2>&1 | `
  Tee-Object -FilePath .\semantic_relevance_v4h_headroom_stage_b_prepare_output.txt
```

Inspect this output before candidate freezing.

## Step 2 — candidate freeze (only after preflight approval)

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\freeze_semantic_relevance_v4h_headroom_stage_b.ps1 2>&1 | `
  Tee-Object -FilePath .\semantic_relevance_v4h_headroom_stage_b_freeze_output.txt
```

The freeze writes:

- `v4h_no_clear_headroom_stage_b_candidates.csv`
- `v4h_no_clear_headroom_stage_b_query_inventory.csv`
- `v4h_no_clear_headroom_stage_b_metadata.json`
- `v4h_no_clear_headroom_stage_b.lock.json`

Only after this candidate inventory is frozen should a separate package prepare exact same-query/same-ISBN score reuse and fresh Stage-B judging input.

## Interpretation boundary

Finding a clear/strong result (`judge_score >= 3`) in ranks 51–end would demonstrate deeper full-union ranking headroom. Conversely, `NO_CLEAR_THROUGH_FULL_UNION` may be concluded only after every frozen Stage-B candidate for a query has a complete frozen score (or the frozen union has no ranks beyond 50). Stage B cannot change the V4B release `FAIL`.
