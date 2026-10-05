from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_semantic_relevance_system_eval_v2 import (
    aggregate_from_query_metrics,
    query_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

V1_RUN = REPO_ROOT / "evals/runs/semantic_relevance_system_eval_v1/20261002T152856Z_recommendations"
V2_RUN = REPO_ROOT / "evals/runs/semantic_relevance_system_eval_v2/20261004T140218Z_recommendations"

V1_SCORES = V1_RUN / "judge_scores.csv"
V2_SCORES = V2_RUN / "judge_scores.csv"
V1_GATES = V1_RUN / "release_gate_result.json"
V2_GATES = V2_RUN / "release_gate_result.json"

EXPECTED_V1_SHA = "d079a74c7f0473e46d28f19e37e4241a780804e972f3fa5f25b3a67e62a7bd11"
EXPECTED_V2_SHA = "4770986e868b26821131d2b36f6cc648f0cab68ae50acda0d1ba4df6d0bd0da1"

OUTPUT_QUERY_DELTAS = V2_RUN / "v1_vs_v2_query_deltas.csv"
OUTPUT_GATE_TRANSITIONS = V2_RUN / "v1_vs_v2_gate_transitions.csv"
OUTPUT_JSON = V2_RUN / "v1_vs_v2_paired_comparison.json"
OUTPUT_REPORT = V2_RUN / "v1_vs_v2_paired_report.md"
OUTPUT_METADATA = V2_RUN / "v1_vs_v2_paired_analysis_metadata.json"

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261002
CONFIDENCE_LEVEL = 0.95

METRICS = {
    "macro_mean_relevance_at_10": {
        "query_column": "mean_relevance_at_10",
        "higher_is_better": True,
    },
    "clear_or_strong_rate_at_10": {
        "query_column": "clear_or_strong_rate_at_10",
        "higher_is_better": True,
    },
    "irrelevant_rate_at_10": {
        "query_column": "irrelevant_rate_at_10",
        "higher_is_better": False,
    },
    "hit_rate_at_5_clear_or_strong": {
        "query_column": "hit_at_5_clear_or_strong",
        "higher_is_better": True,
    },
    "top1_clear_or_strong_rate": {
        "query_column": "top1_clear_or_strong",
        "higher_is_better": True,
    },
    "mrr_at_10_clear_or_strong": {
        "query_column": "mrr_at_10_clear_or_strong",
        "higher_is_better": True,
    },
    "ndcg_at_10": {
        "query_column": "ndcg_at_10",
        "higher_is_better": True,
    },
    "catastrophic_query_rate": {
        "query_column": "catastrophic_query",
        "higher_is_better": False,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_scores(path: Path, expected_sha: str) -> pd.DataFrame:
    actual = sha256(path)
    if actual != expected_sha:
        raise ValueError(
            f"Frozen score hash mismatch for {path}: "
            f"expected={expected_sha} actual={actual}"
        )

    frame = pd.read_csv(
        path,
        dtype={"case_id": str, "query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
    frame["judge_score"] = (
        pd.to_numeric(frame["judge_score"], errors="raise").astype(int)
    )

    if len(frame) != 120 or frame["case_id"].duplicated().any():
        raise ValueError("Expected exactly 120 unique scored cases.")
    if not frame["judge_score"].between(0, 4).all():
        raise ValueError("Judge scores must all be in [0, 4].")
    return frame


def make_query_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        query_summary(group)
        for _, group in scores.groupby("query_id", sort=True)
    ])


def paired_bootstrap(
    differences_by_metric: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(next(iter(differences_by_metric.values())))
    indices = rng.integers(0, n, size=(replicates, n))

    alpha = 1.0 - CONFIDENCE_LEVEL
    lo = alpha / 2.0
    hi = 1.0 - lo

    result = {}
    for name, differences in differences_by_metric.items():
        dist = np.mean(differences[indices], axis=1)
        result[name] = {
            "lower_95": float(np.quantile(dist, lo)),
            "upper_95": float(np.quantile(dist, hi)),
            "bootstrap_mean": float(np.mean(dist)),
            "bootstrap_standard_error": float(np.std(dist, ddof=1)),
            "replicates": replicates,
            "seed": seed,
            "resampling_unit": "matched query",
            "versions_kept_paired": True,
            "method": "paired percentile query bootstrap",
        }
    return result


def build_gate_transitions(
    v1_gates: dict[str, Any],
    v2_gates: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    v1_checks = v1_gates.get("checks", {})
    v2_checks = v2_gates.get("checks", {})
    if list(v1_checks.keys()) != list(v2_checks.keys()):
        raise ValueError("v1 and v2 release gate names/order do not match.")

    transitions: list[dict[str, Any]] = []
    counts = {
        "fail_to_pass": 0,
        "pass_to_pass": 0,
        "fail_to_fail": 0,
        "pass_to_fail": 0,
    }

    for gate_name in v1_checks:
        v1 = v1_checks[gate_name]
        v2 = v2_checks[gate_name]
        if (
            v1.get("metric") != v2.get("metric")
            or v1.get("operator") != v2.get("operator")
            or float(v1.get("threshold")) != float(v2.get("threshold"))
        ):
            raise ValueError(f"Gate definition changed between v1 and v2: {gate_name}")

        v1_passed = bool(v1["passed"])
        v2_passed = bool(v2["passed"])
        if not v1_passed and v2_passed:
            transition = "FAIL->PASS"
            counts["fail_to_pass"] += 1
        elif v1_passed and v2_passed:
            transition = "PASS->PASS"
            counts["pass_to_pass"] += 1
        elif not v1_passed and not v2_passed:
            transition = "FAIL->FAIL"
            counts["fail_to_fail"] += 1
        else:
            transition = "PASS->FAIL"
            counts["pass_to_fail"] += 1

        transitions.append({
            "gate": gate_name,
            "metric": v1["metric"],
            "operator": v1["operator"],
            "threshold": float(v1["threshold"]),
            "v1_actual": float(v1["actual"]),
            "v1_passed": v1_passed,
            "v2_actual": float(v2["actual"]),
            "v2_passed": v2_passed,
            "transition": transition,
        })

    return transitions, counts


def fmt_metric(name: str, value: float) -> str:
    if name in {
        "clear_or_strong_rate_at_10",
        "irrelevant_rate_at_10",
        "hit_rate_at_5_clear_or_strong",
        "top1_clear_or_strong_rate",
        "catastrophic_query_rate",
    }:
        return f"{value:.1%}"
    return f"{value:.3f}"


def fmt_delta(name: str, value: float) -> str:
    if name in {
        "clear_or_strong_rate_at_10",
        "irrelevant_rate_at_10",
        "hit_rate_at_5_clear_or_strong",
        "top1_clear_or_strong_rate",
        "catastrophic_query_rate",
    }:
        return f"{value * 100:+.1f} pp"
    return f"{value:+.3f}"


def main() -> None:
    print("Semantic relevance system evaluation - paired v1 vs v2 comparison")
    print("----------------------------------------------------------------")
    print("Inference unit: matched query")
    print("Matched queries: 12")
    print("Paired bootstrap: 5000 query-level replicates")
    print("Win/tie/loss: query-level, improvement-oriented")
    print("Gate transitions: v1 -> v2, frozen gate definitions")
    print("Judge calls: 0")
    print("Human relevance labels: NOT USED")

    v1_scores = load_scores(V1_SCORES, EXPECTED_V1_SHA)
    v2_scores = load_scores(V2_SCORES, EXPECTED_V2_SHA)

    v1_qm = make_query_metrics(v1_scores)
    v2_qm = make_query_metrics(v2_scores)

    if list(v1_qm["query_id"]) != list(v2_qm["query_id"]):
        raise ValueError("v1 and v2 query IDs are not aligned.")
    if list(v1_qm["query"]) != list(v2_qm["query"]):
        raise ValueError("v1 and v2 query text differs.")

    v1_agg = aggregate_from_query_metrics(v1_qm)
    v2_agg = aggregate_from_query_metrics(v2_qm)

    query_delta_rows = []
    metric_results: dict[str, Any] = {}
    differences_by_metric: dict[str, np.ndarray] = {}

    for metric, spec in METRICS.items():
        column = spec["query_column"]
        a = v1_qm[column].astype(float).to_numpy()
        b = v2_qm[column].astype(float).to_numpy()
        raw_diff = b - a
        improvement = raw_diff if spec["higher_is_better"] else -raw_diff
        differences_by_metric[metric] = raw_diff

        for i, qid in enumerate(v1_qm["query_id"]):
            query_delta_rows.append({
                "query_id": qid,
                "query": str(v1_qm.iloc[i]["query"]),
                "metric": metric,
                "v1_value": float(a[i]),
                "v2_value": float(b[i]),
                "raw_delta_v2_minus_v1": float(raw_diff[i]),
                "improvement_oriented_delta": float(improvement[i]),
                "outcome": (
                    "WIN" if improvement[i] > 1e-15
                    else "LOSS" if improvement[i] < -1e-15
                    else "TIE"
                ),
            })

        raw_aggregate_delta = float(v2_agg[metric] - v1_agg[metric])
        metric_results[metric] = {
            "v1": float(v1_agg[metric]),
            "v2": float(v2_agg[metric]),
            "raw_delta_v2_minus_v1": raw_aggregate_delta,
            "improvement_oriented_delta": (
                raw_aggregate_delta if spec["higher_is_better"] else -raw_aggregate_delta
            ),
            "higher_is_better": bool(spec["higher_is_better"]),
            "wins": int(np.sum(improvement > 1e-15)),
            "ties": int(np.sum(np.abs(improvement) <= 1e-15)),
            "losses": int(np.sum(improvement < -1e-15)),
        }

    bootstrap = paired_bootstrap(
        differences_by_metric,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
    )

    for metric, result in metric_results.items():
        result["paired_bootstrap_95_ci_raw_delta"] = {
            "lower": bootstrap[metric]["lower_95"],
            "upper": bootstrap[metric]["upper_95"],
        }
        result["paired_bootstrap_standard_error"] = bootstrap[metric][
            "bootstrap_standard_error"
        ]

    v1_gates = load_json(V1_GATES)
    v2_gates = load_json(V2_GATES)
    gate_transitions, gate_transition_counts = build_gate_transitions(v1_gates, v2_gates)

    pd.DataFrame(query_delta_rows).to_csv(
        OUTPUT_QUERY_DELTAS,
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(gate_transitions).to_csv(
        OUTPUT_GATE_TRANSITIONS,
        index=False,
        encoding="utf-8",
    )

    payload = {
        "status": "PAIRED_V1_V2_ANALYSIS_COMPLETE",
        "analysis_role": "secondary_comparison",
        "cannot_modify_preregistered_v2_release_decision": True,
        "v1_run": str(V1_RUN.relative_to(REPO_ROOT)).replace("\\", "/"),
        "v2_run": str(V2_RUN.relative_to(REPO_ROOT)).replace("\\", "/"),
        "v1_judge_scores_sha256": sha256(V1_SCORES),
        "v2_judge_scores_sha256": sha256(V2_SCORES),
        "matched_query_count": 12,
        "paired_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "resampling_unit": "matched query",
            "versions_kept_paired": True,
        },
        "metrics": metric_results,
        "gate_transitions": {
            "v1_decision": v1_gates.get("decision"),
            "v2_decision": v2_gates.get("decision"),
            "counts": gate_transition_counts,
            "gates": gate_transitions,
        },
        "limitations": [
            "Only 12 matched benchmark queries are available.",
            (
                "Inference is across this benchmark query set and does not "
                "establish universal improvement for all possible queries."
            ),
            (
                "Paired bootstrap intervals do not measure judge run-to-run "
                "stochasticity."
            ),
            (
                "The paired comparison is secondary evidence and does not "
                "change the preregistered v2 release decision."
            ),
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = []
    for metric, result in metric_results.items():
        ci = result["paired_bootstrap_95_ci_raw_delta"]
        rows.append(
            "| " + metric + " | "
            + fmt_metric(metric, result["v1"]) + " | "
            + fmt_metric(metric, result["v2"]) + " | "
            + fmt_delta(metric, result["raw_delta_v2_minus_v1"]) + " | "
            + "[" + fmt_delta(metric, ci["lower"]) + ", "
            + fmt_delta(metric, ci["upper"]) + "] | "
            + f"{result['wins']}/{result['ties']}/{result['losses']} |"
        )

    gate_rows = []
    for row in gate_transitions:
        gate_rows.append(
            f"| {row['gate']} | {'PASS' if row['v1_passed'] else 'FAIL'} | "
            f"{'PASS' if row['v2_passed'] else 'FAIL'} | {row['transition']} |"
        )

    report = """# Semantic Relevance System Evaluation — v1 vs v2 Comparison

## Claim boundary

The authoritative v2 release decision comes only from the original frozen
benchmark metrics, bootstrap method, eight release gates, and decision policy.

This comparison is secondary evidence showing the size and consistency of the
v1-to-v2 change. It cannot override the v2 release decision.

## Paired design

- Matched unit: query.
- Matched queries: 12.
- Paired bootstrap: 5,000 query-level resamples, seed `20261002`.
- Both versions of each selected query remain paired.
- No recommendation-rank rows are treated as matched observations.
- Win/tie/loss is calculated per matched query after orienting each metric so
  that WIN always means better for v2.

## Metric deltas

`Delta` is v2 - v1. For irrelevant rate and catastrophic-query rate, negative
delta means improvement.

| Metric | v1 | v2 | Delta | 95% paired bootstrap CI | W/T/L |
|---|---:|---:|---:|---:|---:|
__ROWS__

## Release-gate transitions

Gate definitions and thresholds are unchanged. These transitions show whether
v2 moved each frozen gate from pass/fail to a different state; they do not
redefine the v2 release decision.

| Gate | v1 | v2 | Transition |
|---|---:|---:|---:|
__GATES__

Transition counts:

- FAIL→PASS: __FAIL_TO_PASS__
- PASS→PASS: __PASS_TO_PASS__
- FAIL→FAIL: __FAIL_TO_FAIL__
- PASS→FAIL: __PASS_TO_FAIL__

## Interpretation

Effect size, paired confidence intervals, win/tie/loss counts, and gate
transitions are the comparison evidence. With only 12 query clusters,
uncertainty can remain substantial.

The paired bootstrap represents variability across benchmark queries. It does
not measure judge run-to-run stochasticity.

No judge calls and no human relevance labels are used in this comparison.
"""
    report = report.replace("__ROWS__", "\n".join(rows))
    report = report.replace("__GATES__", "\n".join(gate_rows))
    report = report.replace("__FAIL_TO_PASS__", str(gate_transition_counts["fail_to_pass"]))
    report = report.replace("__PASS_TO_PASS__", str(gate_transition_counts["pass_to_pass"]))
    report = report.replace("__FAIL_TO_FAIL__", str(gate_transition_counts["fail_to_fail"]))
    report = report.replace("__PASS_TO_FAIL__", str(gate_transition_counts["pass_to_fail"]))
    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    OUTPUT_METADATA.write_text(
        json.dumps({
            "status": "PAIRED_V1_V2_ANALYSIS_COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "v1_judge_scores_sha256": sha256(V1_SCORES),
            "v2_judge_scores_sha256": sha256(V2_SCORES),
            "judge_calls_performed": 0,
            "human_relevance_labels_used": False,
            "paired_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "matched_query_count": 12,
            "gate_transition_count": len(gate_transitions),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("PAIRED V1 VS V2 COMPARISON: PASS")
    print("Matched queries: 12")
    print("Paired bootstrap replicates: 5000")
    print(
        "Gate transitions: "
        f"FAIL->PASS={gate_transition_counts['fail_to_pass']}, "
        f"PASS->PASS={gate_transition_counts['pass_to_pass']}, "
        f"FAIL->FAIL={gate_transition_counts['fail_to_fail']}, "
        f"PASS->FAIL={gate_transition_counts['pass_to_fail']}"
    )
    print("Judge calls during comparison: 0")
    print("Human relevance labels used: NO")
    print(f"Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
