# System evaluation v2 — core analysis r2

This package keeps the system-evaluation story focused on the statistical tools
we want to explain clearly in an interview: point estimates, bootstrap confidence
intervals, paired deltas, win/tie/loss consistency, and frozen release gates.

## Authoritative benchmark layer

It preserves the original frozen evaluation:

- same query-level metric definitions;
- same nDCG definition;
- same 5,000 query-cluster bootstrap replicates;
- same seed `20261002`;
- same 95% percentile confidence intervals;
- same eight preregistered release gates;
- same PASS / REVIEW / FAIL policy.

## Paired v1 → v2 layer

The matched comparison reports:

- v1 and v2 metric values;
- v2-v1 effect-size deltas;
- 5,000-replicate paired query-bootstrap 95% CIs;
- per-metric query win/tie/loss counts;
- per-query delta CSV;
- v1→v2 release-gate transitions using the unchanged frozen gate definitions.

The matched unit is the query. Both versions of a query remain paired in every
bootstrap resample. Recommendation-rank rows are not treated as independent
matched observations.

## Visual layer

The dashboard includes:

- v2 KPI cards;
- v2 bootstrap-CI chart;
- v1→v2 paired metric table with paired CIs and W/T/L;
- v1→v2 release-gate transition table;
- per-query mean-relevance chart;
- score distribution;
- current v2 release gates;
- five lowest-quality queries;
- lowest-scoring recommendations.

Output:

```text
evals/runs/semantic_relevance_system_eval_v2/20261004T140218Z_recommendations/system_evaluation_dashboard.html
```

Additional paired outputs:

```text
evals/runs/semantic_relevance_system_eval_v2/20261004T140218Z_recommendations/v1_vs_v2_paired_report.md
evals/runs/semantic_relevance_system_eval_v2/20261004T140218Z_recommendations/v1_vs_v2_paired_comparison.json
evals/runs/semantic_relevance_system_eval_v2/20261004T140218Z_recommendations/v1_vs_v2_query_deltas.csv
evals/runs/semantic_relevance_system_eval_v2/20261004T140218Z_recommendations/v1_vs_v2_gate_transitions.csv
```

## Run

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\analyze_semantic_relevance_system_eval_v2.ps1 2>&1 |
  Tee-Object -FilePath .\semantic_relevance_system_eval_v2_analysis_r2.txt
```

Judge execution remains disabled and no human relevance labels are consumed by
this analysis stage.
