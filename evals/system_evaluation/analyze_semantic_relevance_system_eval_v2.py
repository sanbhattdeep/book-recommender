from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

CONTRACT_FILE = EVALS / "system_evaluation/semantic_relevance_system_eval.v1.0.0.json"
ANALYSIS_INPUT_FILE = EVALS / "system_evaluation/semantic_relevance_system_eval_analysis_input.v2.0.0.json"
RUN_DIR = EVALS / "runs/semantic_relevance_system_eval_v2/20261004T140218Z_recommendations"
SCORES_FILE = RUN_DIR / "judge_scores.csv"
SCORING_LOCK_FILE = RUN_DIR / "judge_scoring_lock.json"
EXECUTION_PATCH_FILE = EVALS / "system_evaluation/semantic_relevance_system_eval_execution_patch.retryfix1.json"

QUERY_METRICS_FILE = RUN_DIR / "query_metrics.csv"
AGGREGATE_FILE = RUN_DIR / "aggregate_metrics.json"
BOOTSTRAP_FILE = RUN_DIR / "bootstrap_confidence_intervals.json"
GATES_FILE = RUN_DIR / "release_gate_result.json"
FAILURES_FILE = RUN_DIR / "largest_quality_failures.csv"
REPORT_FILE = RUN_DIR / "system_evaluation_report.md"
METADATA_FILE = RUN_DIR / "analysis_metadata.json"

CI_METRICS = [
    "macro_mean_relevance_at_10",
    "clear_or_strong_rate_at_10",
    "irrelevant_rate_at_10",
    "hit_rate_at_5_clear_or_strong",
    "top1_clear_or_strong_rate",
    "mrr_at_10_clear_or_strong",
    "ndcg_at_10",
]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))

def ndcg_for_scores(scores: list[int]) -> float:
    gains = np.array([(2 ** int(s)) - 1 for s in scores], dtype=float)
    discounts = np.log2(np.arange(2, len(scores) + 2, dtype=float))
    dcg = float(np.sum(gains / discounts))
    ideal = np.sort(gains)[::-1]
    idcg = float(np.sum(ideal / discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg

def query_summary(group: pd.DataFrame) -> dict[str, Any]:
    g = group.sort_values("rank", kind="stable").copy()
    scores = g["judge_score"].astype(int).tolist()
    if len(scores) != 10:
        raise ValueError(f"{g['query_id'].iloc[0]}: expected 10 scores, got {len(scores)}")

    clear = np.array(scores) >= 3
    zero = np.array(scores) == 0
    clear_ranks = [i + 1 for i, matched in enumerate(clear) if matched]
    first_clear_rank = min(clear_ranks) if clear_ranks else None
    mrr = 0.0 if first_clear_rank is None else 1.0 / first_clear_rank

    return {
        "query_id": str(g["query_id"].iloc[0]),
        "query": str(g["query"].iloc[0]),
        "mean_relevance_at_10": float(np.mean(scores)),
        "median_relevance_at_10": float(np.median(scores)),
        "clear_or_strong_rate_at_10": float(np.mean(clear)),
        "irrelevant_rate_at_10": float(np.mean(zero)),
        "hit_at_5_clear_or_strong": int(np.any(clear[:5])),
        "top1_clear_or_strong": int(bool(clear[0])),
        "first_clear_or_strong_rank": first_clear_rank,
        "mrr_at_10_clear_or_strong": float(mrr),
        "ndcg_at_10": float(ndcg_for_scores(scores)),
        "catastrophic_query": int(float(np.mean(scores)) < 1.5),
        "score_0_count": int(sum(s == 0 for s in scores)),
        "score_1_count": int(sum(s == 1 for s in scores)),
        "score_2_count": int(sum(s == 2 for s in scores)),
        "score_3_count": int(sum(s == 3 for s in scores)),
        "score_4_count": int(sum(s == 4 for s in scores)),
    }

def aggregate_from_query_metrics(qm: pd.DataFrame) -> dict[str, float]:
    return {
        "macro_mean_relevance_at_10": float(qm["mean_relevance_at_10"].mean()),
        "clear_or_strong_rate_at_10": float(qm["clear_or_strong_rate_at_10"].mean()),
        "irrelevant_rate_at_10": float(qm["irrelevant_rate_at_10"].mean()),
        "hit_rate_at_5_clear_or_strong": float(qm["hit_at_5_clear_or_strong"].mean()),
        "top1_clear_or_strong_rate": float(qm["top1_clear_or_strong"].mean()),
        "mrr_at_10_clear_or_strong": float(qm["mrr_at_10_clear_or_strong"].mean()),
        "ndcg_at_10": float(qm["ndcg_at_10"].mean()),
        "catastrophic_query_rate": float(qm["catastrophic_query"].mean()),
    }

def bootstrap_query_clusters(qm: pd.DataFrame, replicates: int, seed: int, confidence_level: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    metric_columns = {
        "macro_mean_relevance_at_10": "mean_relevance_at_10",
        "clear_or_strong_rate_at_10": "clear_or_strong_rate_at_10",
        "irrelevant_rate_at_10": "irrelevant_rate_at_10",
        "hit_rate_at_5_clear_or_strong": "hit_at_5_clear_or_strong",
        "top1_clear_or_strong_rate": "top1_clear_or_strong",
        "mrr_at_10_clear_or_strong": "mrr_at_10_clear_or_strong",
        "ndcg_at_10": "ndcg_at_10",
    }
    arrays = {m: qm[c].astype(float).to_numpy() for m, c in metric_columns.items()}
    dist = {m: np.empty(replicates, dtype=float) for m in metric_columns}
    n = len(qm)

    for i in range(replicates):
        idx = rng.integers(0, n, size=n)
        for m, values in arrays.items():
            dist[m][i] = float(np.mean(values[idx]))

    alpha = 1.0 - confidence_level
    lo = alpha / 2.0
    hi = 1.0 - lo
    return {
        m: {
            "lower_95": float(np.quantile(values, lo)),
            "upper_95": float(np.quantile(values, hi)),
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_standard_error": float(np.std(values, ddof=1)),
            "replicates": replicates,
            "seed": seed,
            "resampling_unit": "query",
            "method": "percentile cluster bootstrap",
        }
        for m, values in dist.items()
    }

def gate_pass(actual: float, op: str, expected: float) -> bool:
    if op == ">=":
        return actual >= expected
    if op == "<=":
        return actual <= expected
    if op == "==":
        return actual == expected
    raise ValueError(f"Unsupported gate operator: {op}")

def main() -> None:
    print("Semantic relevance system evaluation v2 - metrics and uncertainty")
    print("-------------------------------------------------------------")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")

    contract = load_json(CONTRACT_FILE)
    analysis_input = load_json(ANALYSIS_INPUT_FILE)
    scoring_lock = load_json(SCORING_LOCK_FILE)
    execution_patch = load_json(EXECUTION_PATCH_FILE)

    if scoring_lock.get("status") != "JUDGE_SCORES_FROZEN_BEFORE_METRIC_ANALYSIS":
        raise ValueError("Judge scores are not frozen for metric analysis.")
    if int(scoring_lock.get("completed_pairs", -1)) != 120:
        raise ValueError("Expected 120 completed judge scores.")
    if scoring_lock.get("human_relevance_labels_used") is not False:
        raise ValueError("System evaluation unexpectedly used human labels.")

    actual_sha = sha256(SCORES_FILE)
    expected_sha = analysis_input["judge_scores_sha256"]
    if actual_sha != expected_sha:
        raise ValueError(f"judge_scores.csv changed after freeze: expected={expected_sha} actual={actual_sha}")
    if scoring_lock.get("judge_scores_sha256") != actual_sha:
        raise ValueError("Scoring lock hash disagrees with judge_scores.csv.")
    if analysis_input.get("status") != "FROZEN_BEFORE_SYSTEM_EVAL_ANALYSIS":
        raise ValueError("V2 analysis input is not frozen.")
    if analysis_input.get("system_eval_variant") != "v2-orderfix":
        raise ValueError("Unexpected system-evaluation variant.")

    if scoring_lock.get("system_eval_variant") != "v2-orderfix":
        raise ValueError("Scoring lock is not for v2-orderfix.")
    if scoring_lock.get("judge_execution_patch") != "book-subject-retryfix1":
        raise ValueError("V2 scoring lock does not record retryfix1.")

    required_patch = {
        "status": "APPLIED_NON_SEMANTIC_EXECUTION_PATCH",
        "patch_id": "book-subject-retryfix1",
        "semantic_rules_changed": False,
        "scoring_rules_changed": False,
    }
    for key, expected in required_patch.items():
        if execution_patch.get(key) != expected:
            raise ValueError(
                f"Execution patch mismatch for {key}: "
                f"expected={expected!r} actual={execution_patch.get(key)!r}"
            )

    scores = pd.read_csv(
        SCORES_FILE,
        dtype={"case_id": str, "query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    scores["rank"] = pd.to_numeric(scores["rank"], errors="raise").astype(int)
    scores["judge_score"] = pd.to_numeric(scores["judge_score"], errors="raise").astype(int)

    if len(scores) != 120 or scores["case_id"].duplicated().any():
        raise ValueError("Expected exactly 120 unique scored cases.")
    if not scores["judge_score"].between(0, 4).all():
        raise ValueError("Judge scores must all be in [0, 4].")

    counts = scores.groupby("query_id").size().to_dict()
    if len(counts) != 12 or any(v != 10 for v in counts.values()):
        raise ValueError(f"Expected 12 x 10 scored pairs; got {counts}")

    qm = pd.DataFrame([
        query_summary(group)
        for _, group in scores.groupby("query_id", sort=True)
    ])
    qm.to_csv(QUERY_METRICS_FILE, index=False, encoding="utf-8")

    aggregate = aggregate_from_query_metrics(qm)
    aggregate.update({
        "query_count": 12,
        "recommendations_per_query": 10,
        "query_book_pairs": 120,
        "overall_mean_relevance": float(scores["judge_score"].mean()),
        "overall_median_relevance": float(scores["judge_score"].median()),
        "score_distribution": {str(s): int((scores["judge_score"] == s).sum()) for s in range(5)},
        "score_distribution_rate": {str(s): float((scores["judge_score"] == s).mean()) for s in range(5)},
    })

    cfg = contract["statistical_uncertainty"]
    ci = bootstrap_query_clusters(
        qm,
        replicates=int(cfg["bootstrap_replicates"]),
        seed=int(cfg["seed"]),
        confidence_level=float(cfg["confidence_level"]),
    )
    aggregate["macro_mean_relevance_at_10_ci_lower_95"] = ci["macro_mean_relevance_at_10"]["lower_95"]
    aggregate["macro_mean_relevance_at_10_ci_upper_95"] = ci["macro_mean_relevance_at_10"]["upper_95"]

    AGGREGATE_FILE.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    BOOTSTRAP_FILE.write_text(json.dumps({
        "method": "query-level cluster bootstrap percentile confidence interval",
        "confidence_level": float(cfg["confidence_level"]),
        "replicates": int(cfg["bootstrap_replicates"]),
        "seed": int(cfg["seed"]),
        "resampling_unit": "query",
        "query_clusters": 12,
        "within_query_recommendations_kept_together": True,
        "confidence_intervals": ci,
        "interpretation": (
            "Intervals quantify sampling uncertainty across benchmark queries. "
            "They do not measure judge-vs-human agreement or run-to-run LLM stochasticity."
        ),
    }, indent=2) + "\n", encoding="utf-8")

    gate_results = {}
    failed = []
    for name, gate in contract["release_gates"].items():
        actual = float(aggregate[gate["metric"]])
        passed = gate_pass(actual, gate["op"], float(gate["value"]))
        gate_results[name] = {
            "passed": passed,
            "metric": gate["metric"],
            "actual": actual,
            "operator": gate["op"],
            "threshold": gate["value"],
            "rationale": gate.get("rationale"),
        }
        if not passed:
            failed.append(name)

    catastrophic = aggregate["catastrophic_query_rate"]
    if not failed:
        decision = "PASS"
    elif len(failed) <= 2 and catastrophic == 0.0:
        decision = "REVIEW"
    else:
        decision = "FAIL"

    GATES_FILE.write_text(json.dumps({
        "decision": decision,
        "passed_gate_count": len(gate_results) - len(failed),
        "total_gate_count": len(gate_results),
        "failed_gate_count": len(failed),
        "failed_gates": failed,
        "checks": gate_results,
        "decision_policy": contract["decision_policy"],
        "gate_scope": "Interview/demo system-evaluation benchmark gates; not production SLAs.",
    }, indent=2) + "\n", encoding="utf-8")

    scores[[
        "case_id","query_id","query","rank","isbn13","title",
        "judge_score","match_level","judge_reason","scoring_explanation"
    ]].sort_values(
        ["judge_score","query_id","rank"],
        ascending=[True, True, True],
        kind="stable",
    ).to_csv(FAILURES_FILE, index=False, encoding="utf-8")

    METADATA_FILE.write_text(json.dumps({
        "status": "SYSTEM_EVALUATION_ANALYSIS_COMPLETE",
        "system_eval_variant": "v2-orderfix",
        "comparison_baseline_run": "evals/runs/semantic_relevance_system_eval_v1/20261002T152856Z_recommendations",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge_scores_sha256": actual_sha,
        "judge_execution_patch": execution_patch["patch_id"],
        "judge_calls_performed_during_analysis": 0,
        "human_relevance_labels_used": False,
        "bootstrap_replicates": int(cfg["bootstrap_replicates"]),
        "bootstrap_seed": int(cfg["seed"]),
        "decision": decision,
    }, indent=2) + "\n", encoding="utf-8")

    def pct(x: float) -> str:
        return f"{x:.1%}"

    ci_rows = []
    for metric in CI_METRICS:
        value = aggregate[metric]
        lower = ci[metric]["lower_95"]
        upper = ci[metric]["upper_95"]
        if metric == "macro_mean_relevance_at_10":
            ci_rows.append(f"| `{metric}` | {value:.3f} | [{lower:.3f}, {upper:.3f}] |")
        else:
            ci_rows.append(f"| `{metric}` | {pct(value)} | [{pct(lower)}, {pct(upper)}] |")

    gate_rows = [
        f"| {'PASS' if row['passed'] else 'FAIL'} | `{name}` | {row['actual']:.4f} | `{row['operator']} {row['threshold']}` |"
        for name, row in gate_results.items()
    ]

    worst = qm.sort_values(
        ["mean_relevance_at_10","ndcg_at_10"],
        ascending=[True, True],
        kind="stable",
    ).head(5)

    worst_rows = [
        f"| {row.query_id} | {row.mean_relevance_at_10:.2f} | {row.clear_or_strong_rate_at_10:.0%} | "
        f"{row.irrelevant_rate_at_10:.0%} | {row.ndcg_at_10:.3f} | {row['query']} |"
        for _, row in worst.iterrows()
    ]

    REPORT_FILE.write_text(f"""# Semantic Relevance System Evaluation v2 Report

**Decision:** **{decision}**

**System under test:** frozen `semantic-recommender-v2-orderfix`  
**Measurement instrument:** `semantic_relevance_v0.29.0-r5` + `book-subject-retryfix1` execution amendment  
**Queries:** 12  
**Recommendations/query:** 10  
**Judged pairs:** 120  
**Human relevance labels used:** No  

## Aggregate metrics with 95% query-cluster bootstrap CIs

| Metric | Point estimate | 95% CI |
|---|---:|---:|
{chr(10).join(ci_rows)}

Additional point estimates:

- Catastrophic query rate: **{aggregate['catastrophic_query_rate']:.1%}**
- Overall median relevance: **{aggregate['overall_median_relevance']:.2f}/4**
- Score distribution: **{aggregate['score_distribution']}**

## Statistical method

Confidence intervals use **{int(cfg['bootstrap_replicates']):,}**
query-level cluster-bootstrap replicates with seed `{int(cfg['seed'])}`.
The 10 recommendations belonging to a query remain together in every resample.

This estimates sampling uncertainty across queries. It does not measure
judge-vs-human agreement and does not measure LLM run-to-run stochasticity.

## Preregistered release gates

| Result | Gate | Actual | Requirement |
|---|---|---:|---:|
{chr(10).join(gate_rows)}

**Passed:** {len(gate_results) - len(failed)}/{len(gate_results)}  
**Failed:** {len(failed)}/{len(gate_results)}

These are interview/demo benchmark gates frozen before execution, not production SLAs.

## Five lowest-quality benchmark queries

| Query | Mean@10 | Score >=3 | Score 0 | nDCG@10 | Text |
|---|---:|---:|---:|---:|---|
{chr(10).join(worst_rows)}

## Debugging

Start with `query_metrics.csv`, then `largest_quality_failures.csv`, then the
corresponding `judge_scores.csv` row and its `scoring_explanation`,
`facet_assessments_json`, `component_evidence_ledger_json`, and
`facet_evidence_json`.

## Evidence integrity

Accepted frozen score artifact SHA-256:

```text
{actual_sha}
```

All v2 scores were produced under the recorded non-semantic
`book-subject-retryfix1` execution amendment. The patch manifest states that
semantic rules and deterministic scoring rules were unchanged.

No judge calls were performed during metric calculation or bootstrapping.
""", encoding="utf-8")

    print()
    print("SYSTEM EVALUATION METRICS")
    print("-------------------------")
    print(f"Macro mean relevance@10:   {aggregate['macro_mean_relevance_at_10']:.3f} "
          f"95% CI [{ci['macro_mean_relevance_at_10']['lower_95']:.3f}, "
          f"{ci['macro_mean_relevance_at_10']['upper_95']:.3f}]")
    print(f"Clear/strong rate@10:      {aggregate['clear_or_strong_rate_at_10']:.1%} "
          f"95% CI [{ci['clear_or_strong_rate_at_10']['lower_95']:.1%}, "
          f"{ci['clear_or_strong_rate_at_10']['upper_95']:.1%}]")
    print(f"Irrelevant rate@10:        {aggregate['irrelevant_rate_at_10']:.1%} "
          f"95% CI [{ci['irrelevant_rate_at_10']['lower_95']:.1%}, "
          f"{ci['irrelevant_rate_at_10']['upper_95']:.1%}]")
    print(f"HitRate@5 clear/strong:     {aggregate['hit_rate_at_5_clear_or_strong']:.1%}")
    print(f"Top-1 clear/strong rate:    {aggregate['top1_clear_or_strong_rate']:.1%}")
    print(f"MRR@10 clear/strong:        {aggregate['mrr_at_10_clear_or_strong']:.3f}")
    print(f"nDCG@10:                    {aggregate['ndcg_at_10']:.3f}")
    print(f"Catastrophic query rate:    {aggregate['catastrophic_query_rate']:.1%}")

    print()
    print("RELEASE GATES")
    print("-------------")
    for name, row in gate_results.items():
        print(f"{'PASS' if row['passed'] else 'FAIL'}  {name}: "
              f"actual={row['actual']:.4f} {row['operator']} {row['threshold']}")

    print()
    print(f"SYSTEM EVALUATION DECISION: {decision}")
    print(f"Passed gates: {len(gate_results) - len(failed)}/{len(gate_results)}")
    print("Judge calls during analysis: 0")
    print("Human relevance labels used: NO")
    print(f"Report: {REPORT_FILE}")

if __name__ == "__main__":
    main()
