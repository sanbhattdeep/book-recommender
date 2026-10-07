from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

SPEC_FILE = EVALS / "system_evaluation/semantic_relevance_v3_reranker_experiment.v0.1.0.json"
SPEC_LOCK_FILE = EVALS / "system_evaluation/semantic_relevance_v3_reranker_experiment.v0.1.0.lock.json"
TOP50_RUN = EVALS / "runs/semantic_relevance_top50_diagnostic_v1/20261005T113402Z_candidates"
SCORE_MAP_FILE = TOP50_RUN / "top50_judge_scores_complete.csv"

EXPECTED_SPEC_SHA256 = "5bb1f07f68db828b8db1a8d6d1c62831a5309651b7c62c91d04184bf41bb9261"
EXPECTED_SCORE_MAP_SHA256 = "876761c8293615e50072c935b400d8ee31dda08ef12bdb75e7a6d8ecd87cbd35"

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261002
CONFIDENCE_LEVEL = 0.95

METRIC_ORDER = [
    ("catastrophic_query_rate", False),
    ("irrelevant_rate_at_10", False),
    ("clear_or_strong_rate_at_10", True),
    ("macro_mean_relevance_at_10", True),
    ("ndcg_at_10", True),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ndcg_for_scores(scores: list[int]) -> float:
    gains = np.array([(2 ** int(s)) - 1 for s in scores], dtype=float)
    discounts = np.log2(np.arange(2, len(scores) + 2, dtype=float))
    dcg = float(np.sum(gains / discounts))
    ideal = np.sort(gains)[::-1]
    idcg = float(np.sum(ideal / discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg


def query_summary(group: pd.DataFrame) -> dict[str, Any]:
    g = group.sort_values("rank", kind="stable")
    scores = g["judge_score"].astype(int).tolist()
    if len(scores) != 10:
        raise ValueError(f"{g['query_id'].iloc[0]} expected 10 rows, got {len(scores)}")
    clear = np.asarray(scores) >= 3
    zero = np.asarray(scores) == 0
    clear_ranks = [i + 1 for i, value in enumerate(clear) if value]
    first_clear = min(clear_ranks) if clear_ranks else None
    return {
        "query_id": str(g["query_id"].iloc[0]),
        "query": str(g["query"].iloc[0]),
        "mean_relevance_at_10": float(np.mean(scores)),
        "clear_or_strong_rate_at_10": float(np.mean(clear)),
        "irrelevant_rate_at_10": float(np.mean(zero)),
        "hit_at_5_clear_or_strong": int(np.any(clear[:5])),
        "top1_clear_or_strong": int(bool(clear[0])),
        "mrr_at_10_clear_or_strong": 0.0 if first_clear is None else 1.0 / first_clear,
        "ndcg_at_10": float(ndcg_for_scores(scores)),
        "catastrophic_query": int(float(np.mean(scores)) < 1.5),
    }


def aggregate(qm: pd.DataFrame) -> dict[str, float]:
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


def bootstrap(qm: pd.DataFrame) -> dict[str, dict[str, float]]:
    mapping = {
        "macro_mean_relevance_at_10": "mean_relevance_at_10",
        "clear_or_strong_rate_at_10": "clear_or_strong_rate_at_10",
        "irrelevant_rate_at_10": "irrelevant_rate_at_10",
        "hit_rate_at_5_clear_or_strong": "hit_at_5_clear_or_strong",
        "top1_clear_or_strong_rate": "top1_clear_or_strong",
        "mrr_at_10_clear_or_strong": "mrr_at_10_clear_or_strong",
        "ndcg_at_10": "ndcg_at_10",
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(qm)
    indices = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
    alpha = 1.0 - CONFIDENCE_LEVEL
    result: dict[str, dict[str, float]] = {}
    for metric, col in mapping.items():
        values = qm[col].astype(float).to_numpy()
        dist = np.mean(values[indices], axis=1)
        result[metric] = {
            "lower_95": float(np.quantile(dist, alpha / 2.0)),
            "upper_95": float(np.quantile(dist, 1.0 - alpha / 2.0)),
        }
    return result


def load_ranked(path: Path, variant: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"query_id", "query", "rank", "vector_rank", "isbn13", "title"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{variant} missing columns: {sorted(missing)}")
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
    frame["vector_rank"] = pd.to_numeric(frame["vector_rank"], errors="raise").astype(int)
    if len(frame) != 120:
        raise ValueError(f"{variant}: expected 120 recommendations.")
    if frame.duplicated(["query_id", "isbn13"]).any():
        raise ValueError(f"{variant}: duplicate query-book pair.")
    counts = frame.groupby("query_id").size()
    if len(counts) != 12 or not (counts == 10).all():
        raise ValueError(f"{variant}: expected 12x10.")
    frame["variant"] = variant
    return frame


def join_labels(recs: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_cols = ["query_id", "isbn13", "judge_score", "match_level"]
    joined = recs.merge(
        labels[label_cols],
        on=["query_id", "isbn13"],
        how="left",
        validate="one_to_one",
    )
    if joined["judge_score"].isna().any():
        missing = joined.loc[joined["judge_score"].isna(), ["query_id", "isbn13"]]
        raise ValueError(f"Missing frozen labels for recommendations: {missing.to_dict('records')}")
    joined["judge_score"] = pd.to_numeric(joined["judge_score"], errors="raise").astype(int)
    return joined


def choose_variant(metrics: dict[str, dict[str, float]]) -> str:
    a = metrics["v3a_cross_encoder"]
    b = metrics["v3b_rrf_hybrid"]
    for metric, higher_is_better in METRIC_ORDER:
        av = float(a[metric])
        bv = float(b[metric])
        if abs(av - bv) <= 1e-15:
            continue
        if higher_is_better:
            return "v3a_cross_encoder" if av > bv else "v3b_rrf_hybrid"
        return "v3a_cross_encoder" if av < bv else "v3b_rrf_hybrid"
    return "v3b_rrf_hybrid"


def query_outcomes(base_qm: pd.DataFrame, candidate_qm: pd.DataFrame, metric: str, higher: bool) -> dict[str, int]:
    merged = base_qm[["query_id", metric]].merge(
        candidate_qm[["query_id", metric]],
        on="query_id",
        suffixes=("_v2", "_candidate"),
        validate="one_to_one",
    )
    delta = merged[f"{metric}_candidate"] - merged[f"{metric}_v2"]
    improvement = delta if higher else -delta
    return {
        "wins": int((improvement > 1e-15).sum()),
        "ties": int((np.abs(improvement) <= 1e-15).sum()),
        "losses": int((improvement < -1e-15).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    print("Semantic relevance v3 development analysis")
    print("------------------------------------------")
    print("Judge calls: 0")
    print("Human relevance labels: NOT USED")
    print("Release decision: NOT PRODUCED")
    print("Development benchmark only")
    print()

    if sha256(SPEC_FILE) != EXPECTED_SPEC_SHA256:
        raise ValueError("Frozen v3 spec hash mismatch.")
    lock = load_json(SPEC_LOCK_FILE)
    if lock.get("status") != "FROZEN_BEFORE_V3_DEV_EXECUTION":
        raise ValueError("v3 spec is not frozen.")
    if sha256(SCORE_MAP_FILE) != EXPECTED_SCORE_MAP_SHA256:
        raise ValueError("Frozen Checkpoint C score map changed.")

    metadata_file = run_dir / "v3_dev_reranker_run_metadata.json"
    metadata = load_json(metadata_file)
    if metadata.get("judge_labels_loaded") is not False:
        raise ValueError("Reranker execution unexpectedly loaded judge labels.")
    if int(metadata.get("judge_calls_performed", -1)) != 0:
        raise ValueError("Reranker execution unexpectedly performed judge calls.")
    if metadata.get("repeated_inference_rank_order_identical") is not True:
        raise ValueError("Reranker reproducibility contract failed.")

    for filename, expected in metadata["outputs"].items():
        actual = sha256(run_dir / filename)
        if actual != expected:
            raise ValueError(f"Reranker output changed: {filename}")

    labels = pd.read_csv(
        SCORE_MAP_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    labels["vector_rank"] = pd.to_numeric(labels["vector_rank"], errors="raise").astype(int)
    labels["judge_score"] = pd.to_numeric(labels["judge_score"], errors="raise").astype(int)

    # Frozen v2 baseline is reconstructed from the same 600-row score map.
    v2 = labels[labels["vector_rank"] <= 10].copy()
    v2["rank"] = v2["vector_rank"].astype(int)
    v2["variant"] = "v2_baseline"

    v3a = join_labels(
        load_ranked(run_dir / "v3_dev_recommendations_v3a.csv", "v3a_cross_encoder"),
        labels,
    )
    v3b = join_labels(
        load_ranked(run_dir / "v3_dev_recommendations_v3b.csv", "v3b_rrf_hybrid"),
        labels,
    )

    systems = {
        "v2_baseline": v2,
        "v3a_cross_encoder": v3a,
        "v3b_rrf_hybrid": v3b,
    }

    query_metrics: dict[str, pd.DataFrame] = {}
    aggregates: dict[str, dict[str, float]] = {}
    cis: dict[str, dict[str, dict[str, float]]] = {}

    query_rows: list[dict[str, Any]] = []
    for variant, frame in systems.items():
        qm = pd.DataFrame([
            query_summary(group)
            for _, group in frame.groupby("query_id", sort=True)
        ])
        qm["variant"] = variant
        query_metrics[variant] = qm
        aggregates[variant] = aggregate(qm)
        cis[variant] = bootstrap(qm)
        query_rows.extend(qm.to_dict("records"))

    pd.DataFrame(query_rows).to_csv(
        run_dir / "v3_dev_query_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    # Additional headroom/preservation diagnostics.
    v2_ids = {
        (str(row.query_id), str(row.isbn13))
        for row in v2.itertuples()
    }
    diagnostics: dict[str, Any] = {}
    for variant in ("v3a_cross_encoder", "v3b_rrf_hybrid"):
        frame = systems[variant]
        ids = {
            (str(row.query_id), str(row.isbn13))
            for row in frame.itertuples()
        }
        promoted = frame[frame["vector_rank"] > 10]
        dropped_v2_clear = v2[
            (v2["judge_score"] >= 3)
            & ~v2.apply(lambda r: (str(r["query_id"]), str(r["isbn13"])) in ids, axis=1)
        ]
        diagnostics[variant] = {
            "deep_pool_promotions_into_top10": int(len(promoted)),
            "clear_or_strong_promotions_from_vector_ranks_11_50": int(
                ((promoted["judge_score"] >= 3)).sum()
            ),
            "clear_or_strong_candidates_from_original_top10_dropped": int(len(dropped_v2_clear)),
            "shared_with_v2_top10": int(len(ids & v2_ids)),
            "query_wtl_vs_v2": {
                "mean_relevance_at_10": query_outcomes(
                    query_metrics["v2_baseline"],
                    query_metrics[variant],
                    "mean_relevance_at_10",
                    True,
                ),
                "irrelevant_rate_at_10": query_outcomes(
                    query_metrics["v2_baseline"],
                    query_metrics[variant],
                    "irrelevant_rate_at_10",
                    False,
                ),
                "ndcg_at_10": query_outcomes(
                    query_metrics["v2_baseline"],
                    query_metrics[variant],
                    "ndcg_at_10",
                    True,
                ),
            },
        }

    selected = choose_variant(aggregates)

    comparison = {
        "status": "V3_DEV_VARIANT_COMPARISON_COMPLETE",
        "analysis_role": "development_only",
        "release_decision": None,
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "spec_sha256": sha256(SPEC_FILE),
        "frozen_score_map_sha256": sha256(SCORE_MAP_FILE),
        "reranker_run_metadata_sha256": sha256(metadata_file),
        "aggregates": aggregates,
        "bootstrap_95_ci": cis,
        "diagnostics": diagnostics,
        "selection_rule": load_json(SPEC_FILE)["variant_selection_rule"],
        "selected_variant": selected,
        "final_holdout_required": True,
    }

    json_file = run_dir / "v3_dev_variant_comparison.json"
    json_file.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    def fmt_pct(x: float) -> str:
        return f"{x:.1%}"

    rows = []
    for variant in ("v2_baseline", "v3a_cross_encoder", "v3b_rrf_hybrid"):
        a = aggregates[variant]
        rows.append(
            f"| {variant} | {a['macro_mean_relevance_at_10']:.3f} | "
            f"{fmt_pct(a['clear_or_strong_rate_at_10'])} | "
            f"{fmt_pct(a['irrelevant_rate_at_10'])} | "
            f"{fmt_pct(a['hit_rate_at_5_clear_or_strong'])} | "
            f"{fmt_pct(a['top1_clear_or_strong_rate'])} | "
            f"{a['mrr_at_10_clear_or_strong']:.3f} | "
            f"{a['ndcg_at_10']:.3f} | "
            f"{fmt_pct(a['catastrophic_query_rate'])} |"
        )

    report = f"""# Semantic Relevance v3 Development Reranker Comparison

**Role:** Development/diagnostic only  
**Release decision:** Not produced  
**Judge calls during development analysis:** 0  
**Human relevance labels used:** No  
**Selected variant:** `{selected}`

| Variant | Mean@10 | Clear/strong | Irrelevant | Hit@5 | Top1 | MRR | nDCG | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Selection

The selected variant follows the preregistered lexicographic rule in the frozen
v0.1.0 experiment specification. No model, RRF parameter, or gate tuning is
performed here.

## Boundary

This 12-query result is not a release claim. Freeze the selected v3
implementation/configuration before authoring and freezing the 24-query unseen
holdout.
"""
    report_file = run_dir / "v3_dev_variant_comparison.md"
    report_file.write_text(report, encoding="utf-8")

    selected_lock = {
        "status": "V3_DEV_CANDIDATE_SELECTED_BEFORE_UNSEEN_HOLDOUT",
        "selected_variant": selected,
        "spec_sha256": sha256(SPEC_FILE),
        "spec_lock_sha256": sha256(SPEC_LOCK_FILE),
        "frozen_score_map_sha256": sha256(SCORE_MAP_FILE),
        "reranker_run_metadata_sha256": sha256(metadata_file),
        "comparison_sha256": sha256(json_file),
        "model_id": metadata["model_id"],
        "model_revision": metadata["model_revision"],
        "release_decision_produced": False,
        "unseen_holdout_required": True,
    }
    selected_lock_file = run_dir / "v3_selected_candidate_lock.json"
    selected_lock_file.write_text(
        json.dumps(selected_lock, indent=2) + "\n",
        encoding="utf-8",
    )

    print("V3 DEVELOPMENT METRICS")
    print("----------------------")
    for variant in ("v2_baseline", "v3a_cross_encoder", "v3b_rrf_hybrid"):
        a = aggregates[variant]
        print(
            f"{variant:21s} mean={a['macro_mean_relevance_at_10']:.3f} "
            f"clear={a['clear_or_strong_rate_at_10']:.1%} "
            f"irrelevant={a['irrelevant_rate_at_10']:.1%} "
            f"ndcg={a['ndcg_at_10']:.3f} "
            f"catastrophic={a['catastrophic_query_rate']:.1%}"
        )

    print()
    print(f"Selected variant: {selected}")
    print("Release decision: NOT PRODUCED")
    print("Judge calls: 0")
    print("V3 DEVELOPMENT ANALYSIS: PASS")


if __name__ == "__main__":
    main()
