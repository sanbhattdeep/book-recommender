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

RUN_DIR = (
    EVALS
    / "runs/semantic_relevance_top50_diagnostic_v1"
    / "20261005T113402Z_candidates"
)

COMPLETE_SCORES_FILE = RUN_DIR / "top50_judge_scores_complete.csv"
NEW_SCORES_FILE = RUN_DIR / "top50_judge_scores_new.csv"
SCORING_LOCK_FILE = RUN_DIR / "top50_scoring_lock.json"
A_LOCK_FILE = RUN_DIR / "top50_collection_lock.json"
B_LOCK_FILE = RUN_DIR / "top50_score_reuse_lock.json"

BLANKSPAN_PATCH_FILE = (
    EVALS
    / "system_evaluation/semantic_relevance_top50_execution_patch.blankspanfix1.json"
)
FINALIZE_PATCH_FILE = (
    EVALS
    / "system_evaluation/semantic_relevance_top50_execution_patch.finalizecaseidfix1.json"
)
CHECKPOINT_C_HARNESS_FILE = (
    EVALS / "system_evaluation/score_semantic_relevance_top50_novel_candidates.py"
)
CONTRACT_FILE = (
    EVALS / "system_evaluation/semantic_relevance_system_eval.v1.0.0.json"
)
V2_SCORES_FILE = (
    EVALS
    / "runs/semantic_relevance_system_eval_v2"
    / "20261004T140218Z_recommendations"
    / "judge_scores.csv"
)

QUERY_METRICS_FILE = RUN_DIR / "top50_query_metrics.csv"
DEPTH_PROFILE_FILE = RUN_DIR / "top50_depth_profile.csv"
ORACLE_TOP10_FILE = RUN_DIR / "top50_oracle_top10.csv"
KNOWN_CANDIDATES_FILE = RUN_DIR / "top50_known_candidate_map.csv"
ACTUAL_VS_ORACLE_FILE = RUN_DIR / "top50_actual_vs_oracle_query_metrics.csv"
ORACLE_METRICS_FILE = RUN_DIR / "top50_oracle_metrics.json"
HEADROOM_JSON_FILE = RUN_DIR / "top50_headroom_analysis.json"
REPORT_FILE = RUN_DIR / "top50_diagnostic_report.md"
METADATA_FILE = RUN_DIR / "top50_headroom_analysis_metadata.json"
LOCK_FILE = RUN_DIR / "top50_headroom_analysis_lock.json"

EXPECTED_COMPLETE_SHA256 = (
    "876761c8293615e50072c935b400d8ee31dda08ef12bdb75e7a6d8ecd87cbd35"
)
EXPECTED_NEW_SCORES_SHA256 = (
    "87b016fcd16415e996f9ba1cd57cbd92573d0374f499be936f5b9691401cb87b"
)
EXPECTED_SCORING_LOCK_SHA256 = (
    "e733a176f52550fb6c067934181e76ccf6f5f80cbb601e2758f87c059f853cfd"
)
EXPECTED_BLANKSPAN_PATCH_SHA256 = (
    "235ce6495f2ac8a3fcea7877b124d94a6f6311ee2a11f31397a9beea765e8f45"
)
EXPECTED_FINALIZE_PATCH_SHA256 = (
    "efbc268492297448899738fa235004d7ddee917dffd6b043378051dfa28b0fa8"
)
EXPECTED_CHECKPOINT_C_HARNESS_SHA256 = (
    "62227c8d7dfbda0e0152e76eb1fe4919a373cd81948364da102c57716b1dbb76"
)
EXPECTED_V2_SCORES_SHA256 = (
    "4770986e868b26821131d2b36f6cc648f0cab68ae50acda0d1ba4df6d0bd0da1"
)

EXPECTED_A_CANDIDATES_SHA256 = (
    "679dc2a8a3607db8d98e4ee868b5d7fc6566d11cb235eebed24e4676240abbad"
)
EXPECTED_B_REUSE_SHA256 = (
    "603ca8e093fa7720c9cd9d9bb7787780316875f33ec37652894b44195fa565b4"
)
EXPECTED_B_NOVEL_SHA256 = (
    "d1205259ff9399c112c282d7142bd12589a8d3a59cadda8e65f526be32401b82"
)

EXPECTED_ROWS = 600
EXPECTED_QUERIES = 12
EXPECTED_PER_QUERY = 50

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261002
CONFIDENCE_LEVEL = 0.95

DEPTHS = (10, 20, 30, 40, 50)

KNOWN_CANDIDATES = [
    ("Q01", "Gilead"),
    ("Q09", "1984"),
    ("Q09", "The Assault"),
    ("Q09", "Animal Farm and 1984"),
    ("Q11", "The Epic of Gilgamesh"),
    ("Q11", "Bakkhai"),
    ("Q11", "Mythology"),
    ("Q12", "A Framework for Understanding Poverty"),
    ("Q12", "Nickel and Dimed"),
    ("Q12", "The Working Poor"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_hash(label: str, path: Path, expected: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: expected={expected} actual={actual} path={path}"
        )
    print(f"PASS  {label} hash unchanged")
    return actual


def qsort_value(query_id: str) -> int:
    digits = "".join(ch for ch in str(query_id) if ch.isdigit())
    return int(digits) if digits else 10**9


def ndcg_for_scores(scores: list[int]) -> float:
    gains = np.array([(2 ** int(s)) - 1 for s in scores], dtype=float)
    discounts = np.log2(np.arange(2, len(scores) + 2, dtype=float))
    dcg = float(np.sum(gains / discounts))
    ideal = np.sort(gains)[::-1]
    idcg = float(np.sum(ideal / discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg


def query_metrics_from_ranked_top10(
    frame: pd.DataFrame,
    *,
    rank_column: str,
) -> dict[str, Any]:
    g = frame.sort_values(rank_column, kind="stable").copy()
    scores = g["judge_score"].astype(int).tolist()
    if len(scores) != 10:
        raise ValueError(
            f"{g['query_id'].iloc[0]}: expected 10 rows, got {len(scores)}"
        )

    clear = np.array(scores) >= 3
    zero = np.array(scores) == 0
    clear_ranks = [i + 1 for i, matched in enumerate(clear) if matched]
    first_clear = min(clear_ranks) if clear_ranks else None

    return {
        "query_id": str(g["query_id"].iloc[0]),
        "query": str(g["query"].iloc[0]),
        "mean_relevance_at_10": float(np.mean(scores)),
        "median_relevance_at_10": float(np.median(scores)),
        "clear_or_strong_rate_at_10": float(np.mean(clear)),
        "irrelevant_rate_at_10": float(np.mean(zero)),
        "hit_at_5_clear_or_strong": int(np.any(clear[:5])),
        "top1_clear_or_strong": int(bool(clear[0])),
        "first_clear_or_strong_rank": first_clear,
        "mrr_at_10_clear_or_strong": (
            0.0 if first_clear is None else 1.0 / first_clear
        ),
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
        "macro_mean_relevance_at_10": float(
            qm["mean_relevance_at_10"].astype(float).mean()
        ),
        "clear_or_strong_rate_at_10": float(
            qm["clear_or_strong_rate_at_10"].astype(float).mean()
        ),
        "irrelevant_rate_at_10": float(
            qm["irrelevant_rate_at_10"].astype(float).mean()
        ),
        "hit_rate_at_5_clear_or_strong": float(
            qm["hit_at_5_clear_or_strong"].astype(float).mean()
        ),
        "top1_clear_or_strong_rate": float(
            qm["top1_clear_or_strong"].astype(float).mean()
        ),
        "mrr_at_10_clear_or_strong": float(
            qm["mrr_at_10_clear_or_strong"].astype(float).mean()
        ),
        "ndcg_at_10": float(qm["ndcg_at_10"].astype(float).mean()),
        "catastrophic_query_rate": float(
            qm["catastrophic_query"].astype(float).mean()
        ),
    }


def bootstrap_macro_mean(
    qm: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    values = qm["mean_relevance_at_10"].astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(values)
    indices = rng.integers(0, n, size=(replicates, n))
    dist = np.mean(values[indices], axis=1)
    alpha = 1.0 - CONFIDENCE_LEVEL
    return {
        "lower_95": float(np.quantile(dist, alpha / 2.0)),
        "upper_95": float(np.quantile(dist, 1.0 - alpha / 2.0)),
        "bootstrap_mean": float(np.mean(dist)),
        "bootstrap_standard_error": float(np.std(dist, ddof=1)),
        "replicates": replicates,
        "seed": seed,
        "resampling_unit": "query",
        "method": "percentile query-cluster bootstrap",
    }


def gate_threshold_results(
    metrics: dict[str, float],
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for gate_name, gate in contract["release_gates"].items():
        metric = gate["metric"]
        actual = float(metrics[metric])
        threshold = float(gate["value"])
        op = str(gate["op"])
        if op == ">=":
            met = actual >= threshold
        elif op == "<=":
            met = actual <= threshold
        elif op == "==":
            met = actual == threshold
        else:
            raise ValueError(f"Unsupported operator {op!r}")

        results[gate_name] = {
            "metric": metric,
            "actual": actual,
            "operator": op,
            "threshold": threshold,
            "threshold_met_by_oracle": bool(met),
            "interpretation": (
                "Diagnostic theoretical headroom only. This is not a release "
                "gate decision for an implemented reranker."
            ),
        }
    return results


def load_and_validate_complete_scores() -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    check_hash(
        "Checkpoint C complete score map",
        COMPLETE_SCORES_FILE,
        EXPECTED_COMPLETE_SHA256,
    )
    check_hash(
        "Checkpoint C novel score artifact",
        NEW_SCORES_FILE,
        EXPECTED_NEW_SCORES_SHA256,
    )
    check_hash(
        "Checkpoint C scoring lock",
        SCORING_LOCK_FILE,
        EXPECTED_SCORING_LOCK_SHA256,
    )
    check_hash(
        "blank-span execution amendment",
        BLANKSPAN_PATCH_FILE,
        EXPECTED_BLANKSPAN_PATCH_SHA256,
    )
    check_hash(
        "finalization case-ID amendment",
        FINALIZE_PATCH_FILE,
        EXPECTED_FINALIZE_PATCH_SHA256,
    )
    check_hash(
        "Checkpoint C scoring harness",
        CHECKPOINT_C_HARNESS_FILE,
        EXPECTED_CHECKPOINT_C_HARNESS_SHA256,
    )
    check_hash(
        "historical v2 judge scores",
        V2_SCORES_FILE,
        EXPECTED_V2_SCORES_SHA256,
    )

    scoring_lock = load_json(SCORING_LOCK_FILE)
    if (
        scoring_lock.get("status")
        != "TOP50_JUDGE_SCORES_FROZEN_BEFORE_HEADROOM_ANALYSIS"
    ):
        raise ValueError("Checkpoint C scoring lock is not frozen for analysis.")
    if int(scoring_lock.get("reused_pairs", -1)) != 213:
        raise ValueError("Checkpoint C lock reusable count changed.")
    if int(scoring_lock.get("newly_judged_pairs", -1)) != 387:
        raise ValueError("Checkpoint C lock novel count changed.")
    if int(scoring_lock.get("complete_pairs", -1)) != 600:
        raise ValueError("Checkpoint C lock complete count changed.")
    if scoring_lock.get("human_relevance_labels_used") is not False:
        raise ValueError("Checkpoint C unexpectedly used human relevance labels.")
    if scoring_lock.get("aggregate_metrics_calculated") is not False:
        raise ValueError("Aggregate metrics were already calculated before D.")
    if scoring_lock.get("oracle_metrics_calculated") is not False:
        raise ValueError("Oracle metrics were already calculated before D.")

    if scoring_lock.get("complete_score_map_sha256") != EXPECTED_COMPLETE_SHA256:
        raise ValueError("Checkpoint C lock complete-map SHA mismatch.")
    if scoring_lock.get("new_judge_scores_sha256") != EXPECTED_NEW_SCORES_SHA256:
        raise ValueError("Checkpoint C lock new-score SHA mismatch.")

    blank = load_json(BLANKSPAN_PATCH_FILE)
    if blank.get("patch_id") != "top50-hard-exclusion-blank-span-normalization1":
        raise ValueError("Unexpected blank-span execution amendment.")
    for key in (
        "semantic_rules_changed",
        "scoring_rules_changed",
        "judge_prompt_changed",
        "judge_config_changed",
        "valid_span_evidence_changed",
    ):
        if blank.get(key) is not False:
            raise ValueError(f"blankspanfix1 changed forbidden field: {key}")

    finalize = load_json(FINALIZE_PATCH_FILE)
    if finalize.get("patch_id") != "top50-finalize-case-id-derivation1":
        raise ValueError("Unexpected finalization amendment.")
    for key in (
        "semantic_rules_changed",
        "scoring_rules_changed",
        "judge_source_changed",
        "judge_config_changed",
        "judge_calls_required",
        "new_judge_scores_changed",
    ):
        if finalize.get(key) is not False:
            raise ValueError(f"finalizecaseidfix1 changed forbidden field: {key}")

    a_lock = load_json(A_LOCK_FILE)
    b_lock = load_json(B_LOCK_FILE)
    if a_lock.get("top50_candidates_sha256") != EXPECTED_A_CANDIDATES_SHA256:
        raise ValueError("Checkpoint A frozen candidate SHA changed.")
    if b_lock.get("score_reuse_map_sha256") != EXPECTED_B_REUSE_SHA256:
        raise ValueError("Checkpoint B frozen reuse-map SHA changed.")
    if b_lock.get("novel_judge_input_sha256") != EXPECTED_B_NOVEL_SHA256:
        raise ValueError("Checkpoint B frozen novel-input SHA changed.")

    frame = pd.read_csv(
        COMPLETE_SCORES_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    required = {
        "diagnostic_case_id",
        "query_id",
        "query",
        "vector_rank",
        "isbn13",
        "title",
        "description",
        "authors",
        "score_source",
        "source_case_id",
        "judge_score",
        "match_level",
        "has_relevant_evidence",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Complete score map missing columns: {sorted(missing)}")

    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected 600 rows, got {len(frame)}")
    if frame["diagnostic_case_id"].duplicated().any():
        raise ValueError("Duplicate diagnostic_case_id in complete score map.")
    if frame.duplicated(["query_id", "isbn13"]).any():
        raise ValueError("Duplicate query-book pair in complete score map.")

    frame["vector_rank"] = pd.to_numeric(
        frame["vector_rank"], errors="raise"
    ).astype(int)
    frame["judge_score"] = pd.to_numeric(
        frame["judge_score"], errors="raise"
    ).astype(int)

    if not frame["judge_score"].between(0, 4).all():
        raise ValueError("Judge scores must be in 0..4.")

    counts = frame.groupby("query_id").size()
    if len(counts) != EXPECTED_QUERIES or not (counts == EXPECTED_PER_QUERY).all():
        raise ValueError(f"Expected 12 x 50 rows; got {counts.to_dict()}")

    for qid, group in frame.groupby("query_id"):
        ranks = sorted(group["vector_rank"].tolist())
        if ranks != list(range(1, 51)):
            raise ValueError(f"{qid}: vector ranks are not exactly 1..50")

    source_counts = frame["score_source"].value_counts().to_dict()
    if int(source_counts.get("checkpoint_c_new", 0)) != 387:
        raise ValueError(f"Expected 387 checkpoint_c_new rows; got {source_counts}")
    if 600 - int(source_counts.get("checkpoint_c_new", 0)) != 213:
        raise ValueError(f"Expected 213 reused rows; got {source_counts}")

    # The reconstructed top 10 must be exactly the historical v2 top 10 and
    # must carry the same judge scores. This is the strongest reproducibility
    # anchor for the diagnostic tail.
    v2 = pd.read_csv(
        V2_SCORES_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    v2["rank"] = pd.to_numeric(v2["rank"], errors="raise").astype(int)
    v2["judge_score"] = pd.to_numeric(
        v2["judge_score"], errors="raise"
    ).astype(int)

    top10 = frame.loc[frame["vector_rank"] <= 10].copy()
    if len(top10) != 120:
        raise ValueError("Complete map does not contain 120 rank<=10 rows.")

    left = top10[
        ["query_id", "vector_rank", "isbn13", "judge_score"]
    ].rename(columns={
        "vector_rank": "rank",
        "judge_score": "top50_judge_score",
    })
    right = v2[
        ["query_id", "rank", "isbn13", "judge_score"]
    ].rename(columns={"judge_score": "v2_judge_score"})

    check = left.merge(
        right,
        on=["query_id", "rank"],
        how="outer",
        suffixes=("_top50", "_v2"),
        indicator=True,
        validate="one_to_one",
    )

    if not check["_merge"].eq("both").all():
        raise ValueError("Top50 rank<=10 identities do not align with frozen v2.")
    if not (check["isbn13_top50"] == check["isbn13_v2"]).all():
        raise ValueError("Top50 rank<=10 ISBN order differs from frozen v2.")
    if not (check["top50_judge_score"] == check["v2_judge_score"]).all():
        raise ValueError("Top50 rank<=10 judge scores differ from frozen v2.")

    print("PASS  reconstructed top10 identity and scores match frozen v2 120/120")
    return frame, scoring_lock, load_json(CONTRACT_FILE)


def make_depth_profile(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for qid, group in scores.groupby("query_id", sort=False):
        g = group.sort_values("vector_rank", kind="stable").copy()
        query = str(g["query"].iloc[0])

        top10 = g.loc[g["vector_rank"] <= 10]
        deep = g.loc[g["vector_rank"].between(11, 50)]

        clear_all = g.loc[g["judge_score"] >= 3]
        strong_all = g.loc[g["judge_score"] == 4]
        clear_deep = deep.loc[deep["judge_score"] >= 3]
        partial_deep = deep.loc[deep["judge_score"] >= 2]

        row: dict[str, Any] = {
            "query_id": str(qid),
            "query": query,
            "first_clear_or_strong_vector_rank": (
                None if clear_all.empty else int(clear_all["vector_rank"].min())
            ),
            "first_strong_vector_rank": (
                None if strong_all.empty else int(strong_all["vector_rank"].min())
            ),
            "best_score_top10": int(top10["judge_score"].max()),
            "best_score_ranks_11_50": int(deep["judge_score"].max()),
            "best_score_top50": int(g["judge_score"].max()),
            "clear_or_strong_count_ranks_11_50": int(len(clear_deep)),
            "partial_or_better_count_ranks_11_50": int(len(partial_deep)),
            "clear_or_strong_ranks_11_50": ",".join(
                str(x) for x in clear_deep["vector_rank"].astype(int).tolist()
            ),
            "partial_or_better_ranks_11_50": ",".join(
                str(x) for x in partial_deep["vector_rank"].astype(int).tolist()
            ),
        }

        for depth in DEPTHS:
            d = g.loc[g["vector_rank"] <= depth]
            row[f"clear_or_strong_count_at_{depth}"] = int(
                (d["judge_score"] >= 3).sum()
            )
            row[f"partial_or_better_count_at_{depth}"] = int(
                (d["judge_score"] >= 2).sum()
            )

        for lo, hi, label in [
            (1, 10, "ranks_1_10"),
            (11, 50, "ranks_11_50"),
            (1, 50, "ranks_1_50"),
        ]:
            d = g.loc[g["vector_rank"].between(lo, hi)]
            for score in range(5):
                row[f"{label}_score_{score}_count"] = int(
                    (d["judge_score"] == score).sum()
                )

        rows.append(row)

    result = pd.DataFrame(rows)
    result["_q"] = result["query_id"].map(qsort_value)
    return result.sort_values("_q").drop(columns="_q").reset_index(drop=True)


def build_oracle(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for _, group in scores.groupby("query_id", sort=False):
        g = group.sort_values(
            ["judge_score", "vector_rank"],
            ascending=[False, True],
            kind="stable",
        ).head(10).copy()
        g["oracle_rank"] = np.arange(1, 11)
        g["came_from_deep_pool"] = g["vector_rank"] > 10
        rows.append(g)

    oracle = pd.concat(rows, ignore_index=True)
    oracle["_q"] = oracle["query_id"].map(qsort_value)
    return (
        oracle.sort_values(["_q", "oracle_rank"], kind="stable")
        .drop(columns="_q")
        .reset_index(drop=True)
    )


def build_known_candidate_map(
    scores: pd.DataFrame,
    oracle: pd.DataFrame,
) -> pd.DataFrame:
    oracle_ids = set(oracle["diagnostic_case_id"].astype(str))
    rows: list[dict[str, Any]] = []

    for query_id, expected_title in KNOWN_CANDIDATES:
        q = scores.loc[scores["query_id"] == query_id].copy()
        exact = q.loc[
            q["title"].str.strip().str.casefold()
            == expected_title.strip().casefold()
        ]

        if exact.empty:
            contains = q.loc[
                q["title"].str.casefold().str.contains(
                    expected_title.casefold(),
                    regex=False,
                )
            ]
            match = contains
            match_mode = "contains" if not contains.empty else "not_found"
        else:
            match = exact
            match_mode = "exact"

        if match.empty:
            rows.append({
                "query_id": query_id,
                "expected_title": expected_title,
                "found": False,
                "match_mode": "not_found",
                "actual_title": "",
                "isbn13": "",
                "vector_rank": "",
                "judge_score": "",
                "match_level": "",
                "in_actual_v2_top10": False,
                "in_oracle_top10": False,
            })
            continue

        for _, item in match.iterrows():
            case_id = str(item["diagnostic_case_id"])
            rows.append({
                "query_id": query_id,
                "expected_title": expected_title,
                "found": True,
                "match_mode": match_mode,
                "actual_title": str(item["title"]),
                "isbn13": str(item["isbn13"]),
                "vector_rank": int(item["vector_rank"]),
                "judge_score": int(item["judge_score"]),
                "match_level": str(item["match_level"]),
                "in_actual_v2_top10": int(item["vector_rank"]) <= 10,
                "in_oracle_top10": case_id in oracle_ids,
            })

    return pd.DataFrame(rows)


def pct(x: float) -> str:
    return f"{x:.1%}"


def fmt_metric(name: str, value: float) -> str:
    if name in {
        "clear_or_strong_rate_at_10",
        "irrelevant_rate_at_10",
        "hit_rate_at_5_clear_or_strong",
        "top1_clear_or_strong_rate",
        "catastrophic_query_rate",
    }:
        return pct(value)
    return f"{value:.3f}"


def main() -> None:
    print("Semantic relevance top-50 diagnostic - Checkpoint D")
    print("---------------------------------------------------")
    print("Purpose: depth/headroom/oracle analysis of frozen 600-row score map")
    print("Judge calls: 0")
    print("Human relevance labels: NOT USED")
    print("Benchmark role: development/diagnostic")
    print("Release decision: NOT PRODUCED")
    print()

    scores, scoring_lock, contract = load_and_validate_complete_scores()

    # -------------------------
    # Actual frozen v2 top-10
    # -------------------------
    actual_top10 = scores.loc[scores["vector_rank"] <= 10].copy()
    actual_qm = pd.DataFrame([
        query_metrics_from_ranked_top10(group, rank_column="vector_rank")
        for _, group in actual_top10.groupby("query_id", sort=False)
    ])
    actual_qm["_q"] = actual_qm["query_id"].map(qsort_value)
    actual_qm = actual_qm.sort_values("_q").drop(columns="_q").reset_index(drop=True)
    actual_agg = aggregate_from_query_metrics(actual_qm)
    actual_ci = bootstrap_macro_mean(
        actual_qm,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    actual_agg["macro_mean_relevance_at_10_ci_lower_95"] = actual_ci["lower_95"]
    actual_agg["macro_mean_relevance_at_10_ci_upper_95"] = actual_ci["upper_95"]

    # -------------------------
    # Depth profile
    # -------------------------
    depth = make_depth_profile(scores)

    # -------------------------
    # Oracle top 10
    # -------------------------
    oracle = build_oracle(scores)
    oracle_qm = pd.DataFrame([
        query_metrics_from_ranked_top10(group, rank_column="oracle_rank")
        for _, group in oracle.groupby("query_id", sort=False)
    ])
    oracle_qm["_q"] = oracle_qm["query_id"].map(qsort_value)
    oracle_qm = (
        oracle_qm.sort_values("_q").drop(columns="_q").reset_index(drop=True)
    )
    oracle_agg = aggregate_from_query_metrics(oracle_qm)
    oracle_ci = bootstrap_macro_mean(
        oracle_qm,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    oracle_agg["macro_mean_relevance_at_10_ci_lower_95"] = oracle_ci["lower_95"]
    oracle_agg["macro_mean_relevance_at_10_ci_upper_95"] = oracle_ci["upper_95"]

    gate_headroom = gate_threshold_results(oracle_agg, contract)
    all_frozen_thresholds_met = all(
        row["threshold_met_by_oracle"]
        for row in gate_headroom.values()
    )

    # Query-level actual-vs-oracle comparison.
    merged_qm = actual_qm.merge(
        oracle_qm,
        on=["query_id", "query"],
        suffixes=("_actual_v2", "_oracle"),
        validate="one_to_one",
    )
    for metric in [
        "mean_relevance_at_10",
        "clear_or_strong_rate_at_10",
        "irrelevant_rate_at_10",
        "hit_at_5_clear_or_strong",
        "top1_clear_or_strong",
        "mrr_at_10_clear_or_strong",
        "ndcg_at_10",
        "catastrophic_query",
    ]:
        merged_qm[f"{metric}_delta_oracle_minus_actual"] = (
            merged_qm[f"{metric}_oracle"].astype(float)
            - merged_qm[f"{metric}_actual_v2"].astype(float)
        )

    actual_ids = set(actual_top10["diagnostic_case_id"].astype(str))
    oracle_ids = set(oracle["diagnostic_case_id"].astype(str))
    deep_oracle = oracle.loc[oracle["vector_rank"] > 10].copy()
    oracle_deep_count = int(len(deep_oracle))
    queries_with_deep_oracle = int(deep_oracle["query_id"].nunique())

    aggregate_deltas = {
        key: float(oracle_agg[key] - actual_agg[key])
        for key in [
            "macro_mean_relevance_at_10",
            "clear_or_strong_rate_at_10",
            "irrelevant_rate_at_10",
            "hit_rate_at_5_clear_or_strong",
            "top1_clear_or_strong_rate",
            "mrr_at_10_clear_or_strong",
            "ndcg_at_10",
            "catastrophic_query_rate",
        ]
    }

    any_oracle_change = any(
        abs(value) > 1e-15
        for value in aggregate_deltas.values()
    ) or oracle_deep_count > 0

    if all_frozen_thresholds_met:
        classification = "RERANKER_SUPPORTED"
        classification_reason = (
            "A perfect score-aware reranking of the frozen top-50 pool meets "
            "every original frozen system-evaluation threshold. This establishes "
            "headroom in the existing candidate pool; it does not prove that a "
            "production reranker can realize that ceiling."
        )
    elif any_oracle_change:
        classification = "MIXED"
        classification_reason = (
            "Oracle reranking improves the frozen v2 result but at least one "
            "original frozen threshold remains unreachable from this top-50 "
            "pool. Both ranking and candidate-generation limitations remain."
        )
    else:
        classification = "RETRIEVAL_LIMITED"
        classification_reason = (
            "Oracle reranking produces no meaningful change from the frozen v2 "
            "top-10 result, so the observed limitation lies in candidate "
            "generation rather than within-pool ordering."
        )

    # Save query metrics with both actual and oracle fields.
    query_out = merged_qm.copy()
    query_out.to_csv(QUERY_METRICS_FILE, index=False, encoding="utf-8")
    depth.to_csv(DEPTH_PROFILE_FILE, index=False, encoding="utf-8")

    oracle_columns = [
        "query_id",
        "query",
        "oracle_rank",
        "vector_rank",
        "diagnostic_case_id",
        "isbn13",
        "title",
        "judge_score",
        "match_level",
        "score_source",
        "came_from_deep_pool",
    ]
    oracle[oracle_columns].to_csv(
        ORACLE_TOP10_FILE,
        index=False,
        encoding="utf-8",
    )

    known = build_known_candidate_map(scores, oracle)
    known.to_csv(KNOWN_CANDIDATES_FILE, index=False, encoding="utf-8")
    merged_qm.to_csv(
        ACTUAL_VS_ORACLE_FILE,
        index=False,
        encoding="utf-8",
    )

    oracle_payload = {
        "status": "TOP50_ORACLE_ANALYSIS_COMPLETE",
        "analysis_role": "diagnostic_theoretical_ceiling",
        "release_decision": None,
        "oracle_algorithm": (
            "Within each query, sort the frozen top-50 candidates by judge_score "
            "descending, break ties by original vector_rank ascending, take 10."
        ),
        "not_a_production_algorithm": True,
        "actual_v2_metrics": actual_agg,
        "actual_v2_macro_mean_bootstrap": actual_ci,
        "oracle_metrics": oracle_agg,
        "oracle_macro_mean_bootstrap": oracle_ci,
        "aggregate_delta_oracle_minus_actual": aggregate_deltas,
        "frozen_gate_threshold_headroom": gate_headroom,
        "all_original_frozen_thresholds_met_by_oracle": (
            all_frozen_thresholds_met
        ),
        "oracle_top10_membership": {
            "rows": 120,
            "deep_pool_rows_from_vector_ranks_11_50": oracle_deep_count,
            "queries_using_at_least_one_deep_pool_candidate": queries_with_deep_oracle,
            "actual_v2_and_oracle_shared_case_count": len(actual_ids & oracle_ids),
            "oracle_only_case_count": len(oracle_ids - actual_ids),
        },
        "warning": (
            "The oracle uses judge scores that would not be available to a "
            "production recommender. It measures ceiling/headroom only."
        ),
    }
    ORACLE_METRICS_FILE.write_text(
        json.dumps(oracle_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    # Pool-level raw evidence summary.
    deep = scores.loc[scores["vector_rank"].between(11, 50)]
    headroom_payload = {
        "status": "TOP50_HEADROOM_ANALYSIS_COMPLETE",
        "classification": classification,
        "classification_reason": classification_reason,
        "release_decision": None,
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "benchmark_role": "development/diagnostic",
        "final_v3_claim_requires_unseen_holdout": True,
        "frozen_complete_score_map_sha256": EXPECTED_COMPLETE_SHA256,
        "candidate_pool": {
            "queries": 12,
            "candidates_per_query": 50,
            "pairs": 600,
            "rank_11_50_pairs": int(len(deep)),
            "clear_or_strong_pairs_rank_11_50": int(
                (deep["judge_score"] >= 3).sum()
            ),
            "partial_or_better_pairs_rank_11_50": int(
                (deep["judge_score"] >= 2).sum()
            ),
            "queries_with_clear_or_strong_in_ranks_11_50": int(
                deep.loc[deep["judge_score"] >= 3, "query_id"].nunique()
            ),
            "queries_with_any_clear_or_strong_in_top50": int(
                scores.loc[scores["judge_score"] >= 3, "query_id"].nunique()
            ),
        },
        "actual_v2_metrics": actual_agg,
        "oracle_metrics": oracle_agg,
        "aggregate_delta_oracle_minus_actual": aggregate_deltas,
        "oracle_threshold_headroom": gate_headroom,
    }
    HEADROOM_JSON_FILE.write_text(
        json.dumps(headroom_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    # Render concise report.
    metric_names = [
        "macro_mean_relevance_at_10",
        "clear_or_strong_rate_at_10",
        "irrelevant_rate_at_10",
        "hit_rate_at_5_clear_or_strong",
        "top1_clear_or_strong_rate",
        "mrr_at_10_clear_or_strong",
        "ndcg_at_10",
        "catastrophic_query_rate",
    ]
    metric_rows = []
    for name in metric_names:
        metric_rows.append(
            f"| `{name}` | {fmt_metric(name, actual_agg[name])} | "
            f"{fmt_metric(name, oracle_agg[name])} | "
            f"{fmt_metric(name, aggregate_deltas[name])} |"
        )

    gate_rows = []
    for name, row in gate_headroom.items():
        gate_rows.append(
            f"| `{name}` | {row['actual']:.4f} | "
            f"`{row['operator']} {row['threshold']}` | "
            f"{'YES' if row['threshold_met_by_oracle'] else 'NO'} |"
        )

    depth_rows = []
    for _, row in depth.iterrows():
        depth_rows.append(
            f"| {row['query_id']} | "
            f"{row['clear_or_strong_count_at_10']} | "
            f"{row['clear_or_strong_count_at_20']} | "
            f"{row['clear_or_strong_count_at_30']} | "
            f"{row['clear_or_strong_count_at_40']} | "
            f"{row['clear_or_strong_count_at_50']} | "
            f"{row['first_clear_or_strong_vector_rank']} | "
            f"{row['best_score_ranks_11_50']} |"
        )

    REPORT_FILE.write_text(
        f"""# Semantic Relevance Top-50 Diagnostic — Checkpoint D

## Status

**Classification:** `{classification}`

{classification_reason}

This is a **development/diagnostic headroom analysis**, not a release decision.
The 12-query benchmark has now been used for iterative diagnosis; a final v3
system claim requires an unseen holdout.

## Frozen evidence

- Complete 600-row score map SHA-256: `{EXPECTED_COMPLETE_SHA256}`
- Reused frozen judgments: 213
- Newly judged top-50 pairs: 387
- Judge calls during Checkpoint D: 0
- Human relevance labels used: No

## Actual v2 top-10 vs top-50 oracle

| Metric | Frozen v2 actual | Oracle top10 | Oracle - actual |
|---|---:|---:|---:|
{chr(10).join(metric_rows)}

Actual macro mean 95% query-bootstrap CI:
`[{actual_ci['lower_95']:.3f}, {actual_ci['upper_95']:.3f}]`

Oracle macro mean 95% query-bootstrap CI:
`[{oracle_ci['lower_95']:.3f}, {oracle_ci['upper_95']:.3f}]`

## Oracle definition

For each query, the diagnostic oracle sorts the frozen top-50 candidates by:

1. `judge_score` descending;
2. original `vector_rank` ascending as the tie-breaker;
3. keep the first 10.

This is a **theoretical ceiling**, not a deployable reranking algorithm. A
production reranker does not have access to these judge labels.

The oracle top-10 uses **{oracle_deep_count}** candidates from reconstructed
vector ranks 11-50 across **{queries_with_deep_oracle}** queries.

## Frozen-threshold headroom

These rows ask only whether the theoretical oracle reaches each **original
frozen threshold**. They do not produce a release PASS/FAIL decision.

| Frozen threshold | Oracle actual | Requirement | Reached? |
|---|---:|---:|---:|
{chr(10).join(gate_rows)}

## Clear/strong evidence by retrieval depth

| Query | @10 | @20 | @30 | @40 | @50 | First >=3 rank | Best score 11-50 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(depth_rows)}

Full per-query distributions, first strong ranks, and deep-pool rank lists are
in `top50_depth_profile.csv`.

## Diagnostic interpretation

- `RERANKER_SUPPORTED`: the perfect within-top50 oracle reaches every original
  frozen threshold. This supports trying a real top50→top10 reranker next.
- `MIXED`: oracle reranking helps, but at least one frozen threshold remains
  unreachable from the current pool.
- `RETRIEVAL_LIMITED`: within-pool oracle reranking produces no meaningful
  change, indicating that candidate generation is the limiting layer.

No post-hoc release thresholds are introduced.

## Known candidate mapping

See `top50_known_candidate_map.csv` for diagnostic candidates such as `Gilead`,
`1984`, `The Epic of Gilgamesh`, `Bakkhai`, `A Framework for Understanding
Poverty`, `Nickel and Dimed`, and `The Working Poor`.

## Next experimental boundary

If the classification supports reranking, the next experiment should train or
configure a reranker **without using this 12-query benchmark as a final holdout**.
Final system qualification must use unseen queries.
""",
        encoding="utf-8",
    )

    metadata = {
        "status": "TOP50_CHECKPOINT_D_ANALYSIS_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_id": "semantic_relevance_top50_diagnostic_v1",
        "stage": "CHECKPOINT_D_TOP50_HEADROOM_ANALYSIS",
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "release_decision_produced": False,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resampling_unit": "query",
        "classification": classification,
        "benchmark_role": "development/diagnostic",
        "unseen_holdout_required_for_final_v3_claim": True,
        "input_hashes": {
            "complete_score_map_sha256": EXPECTED_COMPLETE_SHA256,
            "new_judge_scores_sha256": EXPECTED_NEW_SCORES_SHA256,
            "checkpoint_c_scoring_lock_sha256": EXPECTED_SCORING_LOCK_SHA256,
            "blankspan_patch_sha256": EXPECTED_BLANKSPAN_PATCH_SHA256,
            "finalization_patch_sha256": EXPECTED_FINALIZE_PATCH_SHA256,
            "checkpoint_c_harness_sha256": EXPECTED_CHECKPOINT_C_HARNESS_SHA256,
        },
    }
    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    output_paths = [
        QUERY_METRICS_FILE,
        DEPTH_PROFILE_FILE,
        ORACLE_TOP10_FILE,
        KNOWN_CANDIDATES_FILE,
        ACTUAL_VS_ORACLE_FILE,
        ORACLE_METRICS_FILE,
        HEADROOM_JSON_FILE,
        REPORT_FILE,
        METADATA_FILE,
    ]

    output_hashes = {
        path.name: sha256(path)
        for path in output_paths
    }

    lock = {
        "status": "TOP50_CHECKPOINT_D_FROZEN",
        "diagnostic_id": "semantic_relevance_top50_diagnostic_v1",
        "stage": "CHECKPOINT_D_TOP50_HEADROOM_ANALYSIS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_complete_score_map_sha256": EXPECTED_COMPLETE_SHA256,
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "release_decision_produced": False,
        "classification": classification,
        "output_hashes": output_hashes,
        "benchmark_role": "development/diagnostic",
        "unseen_holdout_required_for_final_v3_claim": True,
    }
    LOCK_FILE.write_text(
        json.dumps(lock, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("CHECKPOINT D RESULTS")
    print("--------------------")
    print(f"Classification:             {classification}")
    print(
        f"Deep-pool oracle rows:      {oracle_deep_count}/120 "
        f"across {queries_with_deep_oracle}/12 queries"
    )
    print(
        "Top50 queries with >=1 clear/strong: "
        f"{headroom_payload['candidate_pool']['queries_with_any_clear_or_strong_in_top50']}/12"
    )
    print(
        "Deep ranks11-50 clear/strong pairs: "
        f"{headroom_payload['candidate_pool']['clear_or_strong_pairs_rank_11_50']}"
    )
    print(
        f"Actual macro mean@10:       {actual_agg['macro_mean_relevance_at_10']:.3f}"
    )
    print(
        f"Oracle macro mean@10:       {oracle_agg['macro_mean_relevance_at_10']:.3f}"
    )
    print(
        f"Actual clear/strong@10:     {actual_agg['clear_or_strong_rate_at_10']:.1%}"
    )
    print(
        f"Oracle clear/strong@10:     {oracle_agg['clear_or_strong_rate_at_10']:.1%}"
    )
    print(
        f"Actual irrelevant@10:       {actual_agg['irrelevant_rate_at_10']:.1%}"
    )
    print(
        f"Oracle irrelevant@10:       {oracle_agg['irrelevant_rate_at_10']:.1%}"
    )
    print(
        f"Actual catastrophic rate:   {actual_agg['catastrophic_query_rate']:.1%}"
    )
    print(
        f"Oracle catastrophic rate:   {oracle_agg['catastrophic_query_rate']:.1%}"
    )
    print()
    print("No oracle release PASS/FAIL decision was produced.")
    print("Judge calls performed: 0")
    print("Human relevance labels used: NO")
    print()
    print("TOP-50 DIAGNOSTIC CHECKPOINT D: PASS")
    print(f"Report: {REPORT_FILE}")
    print(f"Analysis lock: {LOCK_FILE}")
    print("Review Checkpoint D before designing any v3 reranker experiment.")


if __name__ == "__main__":
    main()
