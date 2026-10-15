from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

HOLDOUT_RUN = EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"
STAGE_A_RUN = EVALS / "runs/semantic_relevance_v3_posthoc_diagnostic/v1_zero_judge"
STAGE_B_RUN = EVALS / "runs/semantic_relevance_v3_posthoc_diagnostic/v2_deep_candidate_prepare"
OUT_DIR = EVALS / "runs/semantic_relevance_v3_posthoc_diagnostic/v3_classification"

CONTRACT_FILE = (
    EVALS
    / "system_evaluation"
    / "semantic_relevance_v3_posthoc_stage_b_contract.v0.1.0.json"
)
STAGE_A_LOCK = STAGE_A_RUN / "diagnostic_lock.json"
STAGE_A_QUERY_DIAG = STAGE_A_RUN / "query_diagnosis.csv"

STAGE_B_INPUT_LOCK = STAGE_B_RUN / "deep_candidate_input_lock.json"
STAGE_B_SCORING_LOCK = STAGE_B_RUN / "deep_candidate_scoring_lock.json"
TARGETS_FILE = STAGE_B_RUN / "target_queries.csv"
COMPLETE_MAP_FILE = STAGE_B_RUN / "target_top50_complete_score_map.csv"

FINAL_ANALYSIS_LOCK = HOLDOUT_RUN / "v3_final_holdout_analysis_lock.json"
V3_QUERY_METRICS = HOLDOUT_RUN / "v3_holdout_query_metrics.csv"

CLASSIFICATION_FILE = OUT_DIR / "query_retrieval_reranker_classification.csv"
STRATUM_SUMMARY_FILE = OUT_DIR / "stratum_classification_summary.csv"
REPORT_FILE = OUT_DIR / "posthoc_retrieval_vs_reranker_report.md"
METADATA_FILE = OUT_DIR / "classification_metadata.json"
LOCK_FILE = OUT_DIR / "classification_lock.json"

EXPECTED_CONTRACT_SHA256 = (
    "3394bf61c05ce496e940601a56f1eefb98fd2b2f45b0e799e1c7915691c94483"
)
EXPECTED_STAGE_A_LOCK_SHA256 = (
    "913e54028d02cf59d6c8b3ffbeb9f02ed97c0159f6ec10b1f78187b0301e731d"
)
EXPECTED_STAGE_B_INPUT_LOCK_SHA256 = (
    "ca7340bfc21db93a8b70cdafcf591d6b13a8a31ade55f2de1c954358195af832"
)
EXPECTED_STAGE_B_SCORING_LOCK_SHA256 = (
    "af67cc8090d0613973da4c927b573d29bb956be34d4c44255594cb12bde6fd1f"
)
EXPECTED_COMPLETE_MAP_SHA256 = (
    "723fa27ecb8947334d25b3b7423a6597da39d594eb7ec2f979056b1c3cc516e5"
)
EXPECTED_FINAL_ANALYSIS_LOCK_SHA256 = (
    "09b77bc0bd46b6ac88c79240657bb67f746c09959011f9bdea650d2d3bbd3a20"
)

EXPECTED_TARGET_QUERIES = 17
EXPECTED_TOP50_ROWS = 850
EPS = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pct(value: float) -> str:
    return f"{value:.1%}"


def verify_frozen_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    frozen = [
        ("diagnostic contract", CONTRACT_FILE, EXPECTED_CONTRACT_SHA256),
        ("Stage-A lock", STAGE_A_LOCK, EXPECTED_STAGE_A_LOCK_SHA256),
        ("Stage-B input lock", STAGE_B_INPUT_LOCK, EXPECTED_STAGE_B_INPUT_LOCK_SHA256),
        ("Stage-B scoring lock", STAGE_B_SCORING_LOCK, EXPECTED_STAGE_B_SCORING_LOCK_SHA256),
        ("Stage-B complete top50 map", COMPLETE_MAP_FILE, EXPECTED_COMPLETE_MAP_SHA256),
        ("final v3 analysis lock", FINAL_ANALYSIS_LOCK, EXPECTED_FINAL_ANALYSIS_LOCK_SHA256),
    ]
    for label, path, expected in frozen:
        require(path.exists(), f"Missing {label}: {path}")
        actual = sha256(path)
        require(actual == expected, f"{label} changed: expected={expected} actual={actual}")
        print(f"PASS  frozen {label} hash unchanged")

    contract = load_json(CONTRACT_FILE)
    require(
        contract.get("status") == "FROZEN_BEFORE_POSTHOC_DEEP_JUDGING",
        "Diagnostic classification contract status changed.",
    )
    require(
        contract["authority_boundary"]["may_modify_frozen_v3_release_decision"] is False,
        "Diagnostic authority boundary changed.",
    )
    require(
        contract["authority_boundary"]["frozen_v3_decision"] == "FAIL",
        "Frozen v3 decision in diagnostic contract changed.",
    )

    stage_a_lock = load_json(STAGE_A_LOCK)
    require(
        stage_a_lock.get("status") == "V3_POSTHOC_DIAGNOSTIC_STAGE_A_FROZEN",
        "Stage A is not frozen.",
    )
    require(
        stage_a_lock.get("release_decision_modified") is False,
        "Stage A modified the release decision.",
    )
    require(
        sha256(STAGE_A_QUERY_DIAG)
        == stage_a_lock["outputs"]["query_diagnosis.csv"],
        "Stage-A query diagnosis changed.",
    )

    stage_b_input = load_json(STAGE_B_INPUT_LOCK)
    require(
        stage_b_input.get("status")
        == "V3_POSTHOC_DEEP_DIAGNOSTIC_INPUT_FROZEN_BEFORE_JUDGING",
        "Stage-B input status changed.",
    )
    require(
        sha256(TARGETS_FILE) == stage_b_input["outputs"]["target_queries.csv"],
        "Frozen target-query set changed.",
    )

    stage_b_scoring = load_json(STAGE_B_SCORING_LOCK)
    require(
        stage_b_scoring.get("status")
        == "V3_POSTHOC_DEEP_JUDGE_SCORES_FROZEN_BEFORE_CLASSIFICATION",
        "Stage-B deep judgments are not frozen before classification.",
    )
    require(
        stage_b_scoring.get("target_top50_complete_score_map_sha256")
        == EXPECTED_COMPLETE_MAP_SHA256,
        "Stage-B scoring lock disagrees with complete map hash.",
    )
    require(
        stage_b_scoring.get("classification_calculated") is False,
        "Stage-B scoring unexpectedly calculated classification.",
    )
    require(
        stage_b_scoring.get("frozen_v3_release_decision_modified") is False,
        "Stage-B scoring modified frozen v3 decision.",
    )

    final_lock = load_json(FINAL_ANALYSIS_LOCK)
    require(
        final_lock.get("status") == "V3_FINAL_UNSEEN_HOLDOUT_ANALYSIS_FROZEN",
        "Final v3 analysis is not frozen.",
    )
    require(final_lock.get("v3_final_decision") == "FAIL", "Unexpected frozen v3 decision.")
    expected_v3_qm = final_lock["outputs"]["v3_holdout_query_metrics.csv"]
    require(
        sha256(V3_QUERY_METRICS) == expected_v3_qm,
        "Frozen v3 query metrics changed.",
    )

    targets = pd.read_csv(
        TARGETS_FILE,
        dtype={"query_id": str},
        encoding="utf-8",
    )
    complete = pd.read_csv(
        COMPLETE_MAP_FILE,
        dtype={"query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    v3q = pd.read_csv(
        V3_QUERY_METRICS,
        dtype={"query_id": str},
        encoding="utf-8",
    )

    require(len(targets) == EXPECTED_TARGET_QUERIES, "Expected 17 frozen target queries.")
    require(len(complete) == EXPECTED_TOP50_ROWS, "Expected frozen 17x50 complete map.")
    require(complete["query_id"].nunique() == EXPECTED_TARGET_QUERIES, "Complete map query count changed.")

    counts = complete.groupby("query_id").size()
    require((counts == 50).all(), f"Complete map is not exact 17x50: {counts.to_dict()}")

    complete["vector_rank"] = pd.to_numeric(
        complete["vector_rank"], errors="raise"
    ).astype(int)
    complete["judge_score"] = pd.to_numeric(
        complete["judge_score"], errors="raise"
    ).astype(int)
    require(
        complete["judge_score"].between(0, 4).all(),
        "Complete map score outside 0..4.",
    )

    require(
        set(targets["query_id"]) == set(complete["query_id"]),
        "Target query IDs differ from complete-map query IDs.",
    )

    return targets, complete, v3q, contract


def classify(
    *,
    oracle_mean: float,
    oracle_clear_rate: float,
    oracle_irrelevant_rate: float,
    top50_clear_count: int,
) -> tuple[str, str]:
    if oracle_mean < 1.5 - EPS or top50_clear_count == 0:
        reasons = []
        if oracle_mean < 1.5 - EPS:
            reasons.append("oracle_mean_relevance_at_10 < 1.5")
        if top50_clear_count == 0:
            reasons.append("top50_clear_or_strong_count == 0")
        return (
            "RETRIEVAL_LIMITED",
            "; ".join(reasons),
        )

    if (
        oracle_mean >= 2.5 - EPS
        and oracle_clear_rate >= 0.60 - EPS
        and oracle_irrelevant_rate <= 0.10 + EPS
    ):
        return (
            "RERANKER_LIMITED",
            "oracle top10 meets frozen core headroom criteria: "
            "mean>=2.5, clear>=60%, irrelevant<=10%",
        )

    return (
        "MIXED",
        "top50 has some headroom, but oracle top10 is neither catastrophically "
        "retrieval-limited nor sufficient for the frozen reranker-limited criteria",
    )


def main() -> None:
    print("Semantic relevance v3 - POST-HOC retrieval vs reranker classification")
    print("---------------------------------------------------------------------")
    print("Evidence: frozen Stage-B complete top50 score map")
    print("Judge execution: DISABLED")
    print("Model execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release metrics/gates: NOT RECALCULATED")
    print("Frozen v3 release decision: UNCHANGED")
    print()

    targets, complete, v3q, contract = verify_frozen_inputs()
    print("PASS  all classification inputs and authority boundaries verified")

    v3_small = v3q[
        [
            "query_id",
            "mean_relevance_at_10",
            "clear_or_strong_rate_at_10",
            "irrelevant_rate_at_10",
            "ndcg_at_10",
        ]
    ].rename(
        columns={
            "mean_relevance_at_10": "frozen_v3_mean_relevance_at_10",
            "clear_or_strong_rate_at_10": "frozen_v3_clear_or_strong_rate_at_10",
            "irrelevant_rate_at_10": "frozen_v3_irrelevant_rate_at_10",
            "ndcg_at_10": "frozen_v3_ndcg_at_10",
        }
    )

    target_info = targets[
        [
            "query_id",
            "stratum",
            "query",
            "v2_mean",
            "v3_mean",
            "mean_delta",
            "v3_mean_below_1_5",
            "diagnostic_class",
        ]
    ].merge(
        v3_small,
        on="query_id",
        how="left",
        validate="one_to_one",
    )

    require(
        target_info["frozen_v3_mean_relevance_at_10"].notna().all(),
        "Missing frozen v3 query metrics for a target query.",
    )
    require(
        (
            target_info["v3_mean"].astype(float)
            - target_info["frozen_v3_mean_relevance_at_10"].astype(float)
        ).abs().max()
        <= 1e-12,
        "Stage-A v3 mean disagrees with frozen final-analysis query metrics.",
    )

    rows: list[dict[str, Any]] = []

    for _, target in target_info.sort_values("query_id", kind="stable").iterrows():
        qid = str(target["query_id"])
        q = complete[complete["query_id"] == qid].copy()
        require(len(q) == 50, f"{qid}: expected exactly 50 candidates.")

        q = q.sort_values(
            ["judge_score", "vector_rank", "isbn13"],
            ascending=[False, True, True],
            kind="stable",
        )
        oracle = q.head(10).copy()

        top50_clear = int((q["judge_score"] >= 3).sum())
        deep_clear = int(
            ((q["vector_rank"] >= 11) & (q["judge_score"] >= 3)).sum()
        )
        clear_ranks = q.loc[q["judge_score"] >= 3, "vector_rank"].astype(int).tolist()
        first_clear_rank = min(clear_ranks) if clear_ranks else None

        oracle_mean = float(oracle["judge_score"].mean())
        oracle_clear_rate = float((oracle["judge_score"] >= 3).mean())
        oracle_irrelevant_rate = float((oracle["judge_score"] == 0).mean())
        best_score = int(q["judge_score"].max())

        classification, reason = classify(
            oracle_mean=oracle_mean,
            oracle_clear_rate=oracle_clear_rate,
            oracle_irrelevant_rate=oracle_irrelevant_rate,
            top50_clear_count=top50_clear,
        )

        v3_mean = float(target["frozen_v3_mean_relevance_at_10"])
        v3_clear = float(target["frozen_v3_clear_or_strong_rate_at_10"])

        rows.append(
            {
                "query_id": qid,
                "stratum": str(target["stratum"]),
                "query": str(target["query"]),
                "stage_a_diagnostic_class": str(target["diagnostic_class"]),
                "frozen_v2_mean_relevance_at_10": float(target["v2_mean"]),
                "frozen_v3_mean_relevance_at_10": v3_mean,
                "frozen_v3_clear_or_strong_rate_at_10": v3_clear,
                "frozen_v3_irrelevant_rate_at_10": float(
                    target["frozen_v3_irrelevant_rate_at_10"]
                ),
                "frozen_v3_ndcg_at_10": float(target["frozen_v3_ndcg_at_10"]),
                "best_score_top50": best_score,
                "first_clear_or_strong_vector_rank": first_clear_rank,
                "clear_or_strong_count_top50": top50_clear,
                "clear_or_strong_count_ranks_11_50": deep_clear,
                "oracle_mean_relevance_at_10": oracle_mean,
                "oracle_clear_or_strong_rate_at_10": oracle_clear_rate,
                "oracle_irrelevant_rate_at_10": oracle_irrelevant_rate,
                "v3_to_oracle_mean_headroom": oracle_mean - v3_mean,
                "v3_to_oracle_clear_rate_headroom": oracle_clear_rate - v3_clear,
                "posthoc_limitation_class": classification,
                "classification_reason": reason,
            }
        )

    result = pd.DataFrame(rows)

    require(len(result) == EXPECTED_TARGET_QUERIES, "Classification did not produce 17 rows.")
    require(
        set(result["posthoc_limitation_class"])
        <= {"RETRIEVAL_LIMITED", "RERANKER_LIMITED", "MIXED"},
        "Unexpected classification label.",
    )

    counts = (
        result["posthoc_limitation_class"]
        .value_counts()
        .reindex(
            ["RETRIEVAL_LIMITED", "RERANKER_LIMITED", "MIXED"],
            fill_value=0,
        )
    )

    stratum_summary = (
        result.groupby(["stratum", "posthoc_limitation_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=["RETRIEVAL_LIMITED", "RERANKER_LIMITED", "MIXED"],
            fill_value=0,
        )
        .reset_index()
    )

    headroom_summary = (
        result.groupby("stratum", as_index=False)
        .agg(
            target_queries=("query_id", "size"),
            mean_v3_relevance=("frozen_v3_mean_relevance_at_10", "mean"),
            mean_oracle_relevance=("oracle_mean_relevance_at_10", "mean"),
            mean_oracle_headroom=("v3_to_oracle_mean_headroom", "mean"),
            mean_top50_clear_count=("clear_or_strong_count_top50", "mean"),
            mean_deep_clear_count=("clear_or_strong_count_ranks_11_50", "mean"),
        )
    )
    stratum_summary = stratum_summary.merge(
        headroom_summary,
        on="stratum",
        how="left",
        validate="one_to_one",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(CLASSIFICATION_FILE, index=False, encoding="utf-8")
    stratum_summary.to_csv(STRATUM_SUMMARY_FILE, index=False, encoding="utf-8")

    class_table = "\n".join(
        f"| {r.query_id} | {r.stratum} | {r.frozen_v3_mean_relevance_at_10:.2f} | "
        f"{r.oracle_mean_relevance_at_10:.2f} | "
        f"{int(r.clear_or_strong_count_top50)} | "
        f"{int(r.clear_or_strong_count_ranks_11_50)} | "
        f"{r.posthoc_limitation_class} |"
        for _, r in result.iterrows()
    )

    stratum_table = "\n".join(
        f"| {r.stratum} | {int(r.RETRIEVAL_LIMITED)} | "
        f"{int(r.RERANKER_LIMITED)} | {int(r.MIXED)} | "
        f"{r.mean_v3_relevance:.2f} | {r.mean_oracle_relevance:.2f} | "
        f"{r.mean_oracle_headroom:+.2f} |"
        for _, r in stratum_summary.iterrows()
    )

    report = f"""# Semantic Relevance v3 — Post-hoc Retrieval vs Reranker Diagnosis

## Authority boundary

This is **post-hoc v4-development evidence only**. It consumes the frozen
17×50 Stage-B score map and applies classification rules that were frozen before
the 624 new deep-candidate judgments.

It does **not** recalculate v3 release gates and cannot revise the frozen final
v3 decision: **FAIL (0/8 gates)**.

## Overall diagnosis

- `RETRIEVAL_LIMITED`: **{int(counts['RETRIEVAL_LIMITED'])}/17**
- `RERANKER_LIMITED`: **{int(counts['RERANKER_LIMITED'])}/17**
- `MIXED`: **{int(counts['MIXED'])}/17**

Interpretation:

- `RETRIEVAL_LIMITED` means even ideal top-10 selection from the frozen vector
  top-50 remains catastrophically weak (`oracle mean < 1.5`) or the top-50
  contains no clear/strong candidate.
- `RERANKER_LIMITED` means the top-50 contains enough judged material for an
  oracle top-10 to satisfy the pre-frozen **core headroom criteria**:
  mean ≥2.5, clear/strong ≥60%, irrelevant ≤10%. This is not a release PASS.
- `MIXED` means there is meaningful candidate-pool headroom, but neither
  retrieval alone nor reranking alone cleanly explains the weakness.

## Query-level classification

| Query | Stratum | v3 mean | Oracle mean | Clear in top50 | Clear at ranks 11-50 | Diagnosis |
|---|---|---:|---:|---:|---:|---|
{class_table}

## Stratum summary

| Stratum | Retrieval-limited | Reranker-limited | Mixed | v3 mean | Oracle mean | Mean headroom |
|---|---:|---:|---:|---:|---:|---:|
{stratum_table}

## v4 design use

The classification is intended to choose the next engineering hypothesis:

- retrieval-heavy work is justified where the frozen top-50 itself lacks
  sufficient relevant material;
- reranker-heavy work is justified where strong candidates exist but the frozen
  v3b selection fails to surface them;
- mixed cases should not be forced into a single-cause explanation.

No new judge calls, model calls, human labels, release metrics, or release
decisions are produced by this classification stage.
"""
    REPORT_FILE.write_text(report, encoding="utf-8")

    metadata = {
        "status": "V3_POSTHOC_RETRIEVAL_VS_RERANKER_CLASSIFICATION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "stage_b_scoring_lock_sha256": EXPECTED_STAGE_B_SCORING_LOCK_SHA256,
        "source_complete_map_sha256": EXPECTED_COMPLETE_MAP_SHA256,
        "target_query_count": EXPECTED_TARGET_QUERIES,
        "classification_counts": {
            "RETRIEVAL_LIMITED": int(counts["RETRIEVAL_LIMITED"]),
            "RERANKER_LIMITED": int(counts["RERANKER_LIMITED"]),
            "MIXED": int(counts["MIXED"]),
        },
        "judge_calls_performed": 0,
        "model_calls_performed": 0,
        "human_relevance_labels_used": False,
        "release_metrics_recalculated": False,
        "release_gates_recalculated": False,
        "frozen_v3_release_decision_modified": False,
        "next_stage": "V4_ARCHITECTURE_DECISION_FROM_FROZEN_POSTHOC_EVIDENCE",
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    outputs = [CLASSIFICATION_FILE, STRATUM_SUMMARY_FILE, REPORT_FILE, METADATA_FILE]
    lock_payload = {
        "status": "V3_POSTHOC_RETRIEVAL_VS_RERANKER_CLASSIFICATION_FROZEN",
        "diagnostic_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "stage_b_scoring_lock_sha256": EXPECTED_STAGE_B_SCORING_LOCK_SHA256,
        "source_complete_map_sha256": EXPECTED_COMPLETE_MAP_SHA256,
        "target_query_count": EXPECTED_TARGET_QUERIES,
        "classification_counts": {
            "RETRIEVAL_LIMITED": int(counts["RETRIEVAL_LIMITED"]),
            "RERANKER_LIMITED": int(counts["RERANKER_LIMITED"]),
            "MIXED": int(counts["MIXED"]),
        },
        "judge_calls_performed": 0,
        "model_calls_performed": 0,
        "human_relevance_labels_used": False,
        "release_metrics_recalculated": False,
        "release_gates_recalculated": False,
        "frozen_v3_release_decision_modified": False,
        "outputs": {p.name: sha256(p) for p in outputs},
        "next_stage": "V4_ARCHITECTURE_DECISION_FROM_FROZEN_POSTHOC_EVIDENCE",
    }

    if LOCK_FILE.exists():
        existing = load_json(LOCK_FILE)
        require(existing == lock_payload, "Existing classification lock differs.")
    else:
        LOCK_FILE.write_text(
            json.dumps(lock_payload, indent=2) + "\n",
            encoding="utf-8",
        )

    print()
    print("V3 POST-HOC RETRIEVAL VS RERANKER CLASSIFICATION: PASS")
    print(f"Target queries classified:  {EXPECTED_TARGET_QUERIES}/{EXPECTED_TARGET_QUERIES}")
    print(f"RETRIEVAL_LIMITED:          {int(counts['RETRIEVAL_LIMITED'])}")
    print(f"RERANKER_LIMITED:           {int(counts['RERANKER_LIMITED'])}")
    print(f"MIXED:                      {int(counts['MIXED'])}")
    print("Judge calls performed:      0")
    print("Model calls performed:      0")
    print("Release metrics recalculated: NO")
    print("Frozen v3 decision modified:  NO")
    print(f"Classification lock SHA-256: {sha256(LOCK_FILE)}")
    print(f"Report: {REPORT_FILE}")
    print("Next: v4 architecture decision from frozen post-hoc evidence.")


if __name__ == "__main__":
    main()
