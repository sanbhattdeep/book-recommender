from __future__ import annotations

import hashlib
import json
import math
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

EXPECTED_V1_SHA = "d079a74c7f0473e46d28f19e37e4241a780804e972f3fa5f25b3a67e62a7bd11"
EXPECTED_V2_SHA = "4770986e868b26821131d2b36f6cc648f0cab68ae50acda0d1ba4df6d0bd0da1"

OUTPUT_QUERY_DELTAS = V2_RUN / "v1_vs_v2_query_deltas.csv"
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
        "binary_query_metric": False,
    },
    "clear_or_strong_rate_at_10": {
        "query_column": "clear_or_strong_rate_at_10",
        "higher_is_better": True,
        "binary_query_metric": False,
    },
    "irrelevant_rate_at_10": {
        "query_column": "irrelevant_rate_at_10",
        "higher_is_better": False,
        "binary_query_metric": False,
    },
    "hit_rate_at_5_clear_or_strong": {
        "query_column": "hit_at_5_clear_or_strong",
        "higher_is_better": True,
        "binary_query_metric": True,
    },
    "top1_clear_or_strong_rate": {
        "query_column": "top1_clear_or_strong",
        "higher_is_better": True,
        "binary_query_metric": True,
    },
    "mrr_at_10_clear_or_strong": {
        "query_column": "mrr_at_10_clear_or_strong",
        "higher_is_better": True,
        "binary_query_metric": False,
    },
    "ndcg_at_10": {
        "query_column": "ndcg_at_10",
        "higher_is_better": True,
        "binary_query_metric": False,
    },
    "catastrophic_query_rate": {
        "query_column": "catastrophic_query",
        "higher_is_better": False,
        "binary_query_metric": True,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def exact_mcnemar_pvalue(
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[float, int, int]:
    """Two-sided exact McNemar test using discordant matched queries only."""
    a = np.asarray(v1, dtype=int)
    b = np.asarray(v2, dtype=int)

    n_01 = int(np.sum((a == 0) & (b == 1)))
    n_10 = int(np.sum((a == 1) & (b == 0)))
    discordant = n_01 + n_10

    if discordant == 0:
        return 1.0, n_01, n_10

    smaller = min(n_01, n_10)
    tail = sum(
        math.comb(discordant, k)
        for k in range(0, smaller + 1)
    ) / (2 ** discordant)

    return min(1.0, 2.0 * tail), n_01, n_10


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
    print("McNemar: exact, binary query outcomes only")
    print("Other p-value tests: NOT USED")
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
    differences_by_metric = {}

    for metric, spec in METRICS.items():
        column = spec["query_column"]
        a = v1_qm[column].astype(float).to_numpy()
        b = v2_qm[column].astype(float).to_numpy()
        raw_diff = b - a
        improvement = (
            raw_diff if spec["higher_is_better"] else -raw_diff
        )
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
        result = {
            "v1": float(v1_agg[metric]),
            "v2": float(v2_agg[metric]),
            "raw_delta_v2_minus_v1": raw_aggregate_delta,
            "improvement_oriented_delta": (
                raw_aggregate_delta
                if spec["higher_is_better"]
                else -raw_aggregate_delta
            ),
            "higher_is_better": bool(spec["higher_is_better"]),
            "wins": int(np.sum(improvement > 1e-15)),
            "ties": int(np.sum(np.abs(improvement) <= 1e-15)),
            "losses": int(np.sum(improvement < -1e-15)),
            "binary_query_metric": bool(spec["binary_query_metric"]),
        }

        if spec["binary_query_metric"]:
            p, n_01, n_10 = exact_mcnemar_pvalue(a, b)

            if spec["higher_is_better"]:
                improved = n_01
                regressed = n_10
            else:
                improved = n_10
                regressed = n_01

            result["mcnemar"] = {
                "method": "two-sided exact McNemar",
                "v1_0_v2_1": n_01,
                "v1_1_v2_0": n_10,
                "discordant_pairs": n_01 + n_10,
                "improved_queries": improved,
                "regressed_queries": regressed,
                "exact_two_sided_p_value": float(p),
                "role": "secondary exploratory diagnostic; not a release gate",
            }

        metric_results[metric] = result

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

    pd.DataFrame(query_delta_rows).to_csv(
        OUTPUT_QUERY_DELTAS,
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
        "hypothesis_testing": {
            "mcnemar_only": True,
            "scope": [
                "hit_rate_at_5_clear_or_strong",
                "top1_clear_or_strong_rate",
                "catastrophic_query_rate",
            ],
            "other_p_value_tests_used": False,
            "interpretation": (
                "McNemar is retained only as a compact secondary diagnostic "
                "for matched binary query outcomes. No significance result "
                "changes the release decision."
            ),
        },
        "metrics": metric_results,
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
                "McNemar results are exploratory secondary diagnostics, not "
                "preregistered release criteria."
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

    mcnemar_rows = []
    for metric, result in metric_results.items():
        if "mcnemar" not in result:
            continue
        m = result["mcnemar"]
        mcnemar_rows.append(
            f"| {metric} | {m['improved_queries']} | "
            f"{m['regressed_queries']} | {m['discordant_pairs']} | "
            f"{m['exact_two_sided_p_value']:.4f} |"
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
- No general p-value testing is performed.

## Metric deltas

`Delta` is v2 - v1. For irrelevant rate and catastrophic-query rate, negative
delta means improvement.

| Metric | v1 | v2 | Delta | 95% paired bootstrap CI | W/T/L |
|---|---:|---:|---:|---:|---:|
__ROWS__

## Exact McNemar diagnostics

McNemar is retained only for binary query-level outcomes. The most useful
numbers are the discordant directions: queries that improved versus queries
that regressed. Exact p-values are shown only as a compact secondary
diagnostic and are **not release gates**.

| Binary metric | Improved queries | Regressed queries | Discordant | Exact two-sided p |
|---|---:|---:|---:|---:|
__MCNEMAR__

## Interpretation

Effect size and paired confidence intervals are the primary comparison
evidence. With only 12 query clusters, uncertainty can remain substantial.

The paired bootstrap represents variability across benchmark queries. It does
not measure judge run-to-run stochasticity.

No judge calls and no human relevance labels are used in this comparison.
"""
    report = report.replace("__ROWS__", "\n".join(rows))
    report = report.replace("__MCNEMAR__", "\n".join(mcnemar_rows))
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
            "mcnemar_binary_metrics": 3,
            "other_p_value_tests_used": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("PAIRED V1 VS V2 COMPARISON: PASS")
    print("Matched queries: 12")
    print("Paired bootstrap replicates: 5000")
    print("Exact McNemar binary diagnostics: COMPLETE (3 metrics)")
    print("Other p-value tests: NOT USED")
    print("Judge calls during comparison: 0")
    print("Human relevance labels used: NO")
    print(f"Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
