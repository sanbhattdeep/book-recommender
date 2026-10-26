from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"
SYS = ROOT / "evals/system_evaluation"

CLASSIFICATION_LOCK = RUN / "v4h_headroom_stage_a_classification.lock.json"
QUERY_CLASSIFICATION = RUN / "v4h_headroom_stage_a_query_classification.csv"
RERANKING_LOCK = RUN / "v4h_reranking.lock.json"
V4B_FULL = RUN / "v4h_v4b_ranked_full_union.csv"
FINAL_RELEASE_LOCK = RUN / "v4h_final_release_analysis.lock.json"
CONTRACT = SYS / "semantic_relevance_v4h_no_clear_headroom_stage_b.v0.1.0.json"

CANDIDATES = RUN / "v4h_no_clear_headroom_stage_b_candidates.csv"
QUERY_INVENTORY = RUN / "v4h_no_clear_headroom_stage_b_query_inventory.csv"
METADATA = RUN / "v4h_no_clear_headroom_stage_b_metadata.json"
LOCK = RUN / "v4h_no_clear_headroom_stage_b.lock.json"

EXPECTED_CLASSIFICATION_LOCK_SHA256 = "906b5975345c9d9eed1273ecc4ccd0611880b14d6326bf9f0e74b46f7d63d948"
EXPECTED_RERANKING_LOCK_SHA256 = "6d3a7014f5995d978a90df627c720c0b81aa2e8b28e8bba99cfa2c64d27811b5"
EXPECTED_FINAL_RELEASE_LOCK_SHA256 = "7e7d73920452ec39ea72947d2b3762a02aaa1b2bb0bcd7470c7e21dbb3a25ab0"
EXPECTED_CONTRACT_SHA256 = "c26af8da897995b817d9eeffef56f94a2fd4cab99dca16777d85f569759a4476"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")


def verify_boundary() -> tuple[list[str], list[str], pd.DataFrame, dict[str, Any]]:
    verify_raw(CLASSIFICATION_LOCK, EXPECTED_CLASSIFICATION_LOCK_SHA256, "Stage-A classification lock")
    verify_raw(RERANKING_LOCK, EXPECTED_RERANKING_LOCK_SHA256, "V4H reranking lock")
    verify_raw(FINAL_RELEASE_LOCK, EXPECTED_FINAL_RELEASE_LOCK_SHA256, "V4H final release-analysis lock")
    verify_raw(CONTRACT, EXPECTED_CONTRACT_SHA256, "Stage-B definition contract")

    clock = load_json(CLASSIFICATION_LOCK)
    req(clock.get("status") == "V4H_HEADROOM_STAGE_A_CLASSIFICATION_FROZEN",
        "Unexpected Stage-A classification lock status")
    req(clock.get("frozen_release_decision") == "FAIL", "Frozen V4B release decision changed")
    req(clock.get("release_decision_modified") is False, "Stage-A modified the release decision")
    req(clock.get("classification_counts") == {
        "HEADROOM_FOUND_11_50": 4,
        "NO_CLEAR_THROUGH_50": 6,
    }, f"Unexpected Stage-A classification counts: {clock.get('classification_counts')}")
    req(clock.get("new_judge_calls") == 0, "Stage-A classification unexpectedly made judge calls")
    req(clock.get("new_model_calls") == 0, "Stage-A classification unexpectedly made model calls")
    req(clock.get("new_retrieval_calls") == 0, "Stage-A classification unexpectedly made retrieval calls")
    req(clock.get("new_reranker_calls") == 0, "Stage-A classification unexpectedly made reranker calls")
    req(clock.get("release_metrics_calculated") is False, "Stage-A classification recalculated release metrics")

    outputs = clock.get("outputs", {})
    expected_q_sha = outputs.get(QUERY_CLASSIFICATION.name)
    req(isinstance(expected_q_sha, str) and len(expected_q_sha) == 64,
        "Stage-A classification lock does not bind query-classification artifact")
    verify_raw(QUERY_CLASSIFICATION, expected_q_sha, "Stage-A query classification")

    q = pd.read_csv(QUERY_CLASSIFICATION, dtype=str, keep_default_na=False, encoding="utf-8")
    req({"query_id", "classification"}.issubset(q.columns), "Stage-A query classification missing required columns")
    req(len(q) == 10, f"Expected 10 Stage-A query rows, found {len(q)}")
    req(not q["query_id"].duplicated().any(), "Duplicate query_id in Stage-A classification")
    allowed = {"HEADROOM_FOUND_11_50", "NO_CLEAR_THROUGH_50"}
    req(set(q["classification"]).issubset(allowed), "Unexpected Stage-A classification value")

    unresolved = q.loc[q["classification"] == "NO_CLEAR_THROUGH_50", "query_id"].tolist()
    resolved = q.loc[q["classification"] == "HEADROOM_FOUND_11_50", "query_id"].tolist()
    req(len(unresolved) == 6, f"Expected 6 NO_CLEAR_THROUGH_50 queries, found {len(unresolved)}")
    req(len(resolved) == 4, f"Expected 4 HEADROOM_FOUND_11_50 queries, found {len(resolved)}")
    req(not (set(unresolved) & set(resolved)), "Resolved and unresolved Stage-A query sets overlap")

    rlock = load_json(RERANKING_LOCK)
    req(rlock.get("status") == "V4H_V3B_V4B_RERANKING_FROZEN_BEFORE_DISPLAY_UNION_OR_JUDGING",
        "Unexpected V4H reranking lock status")
    full_sha = rlock.get("v4b_ranked_full_union_sha256")
    req(isinstance(full_sha, str) and len(full_sha) == 64,
        "V4H reranking lock missing full-union SHA-256")
    verify_raw(V4B_FULL, full_sha, "V4B frozen ranked full union")

    full = pd.read_csv(V4B_FULL, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"query_id", "query", "stratum", "isbn13", "title", "description", "v4b_rank", "reranker_score", "hybrid_retrieval_rank"}
    missing = required - set(full.columns)
    req(not missing, f"V4B full ranking missing columns: {sorted(missing)}")
    full["v4b_rank"] = pd.to_numeric(full["v4b_rank"], errors="raise").astype(int)
    req(not full.duplicated(["query_id", "isbn13"]).any(), "Duplicate V4B full-union query/ISBN pair")

    for qid in unresolved:
        g = full[full["query_id"] == qid].sort_values("v4b_rank", kind="stable")
        req(len(g) >= 50, f"{qid} has fewer than 50 frozen V4B candidates")
        ranks = g["v4b_rank"].tolist()
        req(ranks == list(range(1, max(ranks) + 1)), f"{qid} frozen V4B ranks are not contiguous 1..end")

    for path in [CANDIDATES, QUERY_INVENTORY, METADATA, LOCK]:
        req(not path.exists(), f"Refusing Stage-B preflight because output already exists: {path}")

    return unresolved, resolved, full, rlock


def main() -> None:
    print("Semantic relevance V4H - headroom Stage-B candidate-freeze preflight")
    print("--------------------------------------------------------------------")
    print("Role: consumed-holdout diagnostic only")
    print("Judge/model/retrieval/reranker calls: DISABLED")
    print("Release metrics/gates: NOT CALCULATED")
    print("Frozen V4B release decision modification: DISABLED")
    print()

    unresolved, resolved, full, _ = verify_boundary()
    deeper = full[full["query_id"].isin(unresolved) & (full["v4b_rank"] >= 51)]

    print(f"PASS  frozen Stage-A unresolved target set derived: {len(unresolved)} queries")
    print("INFO  Stage-B target query IDs: " + ", ".join(unresolved))
    print(f"PASS  Stage-A headroom-found queries excluded: {len(resolved)} queries")
    print(f"INFO  frozen ranks 51-end available before freeze: {len(deeper)} query-book pairs")
    print()
    print("V4H HEADROOM STAGE-B CANDIDATE-FREEZE PREFLIGHT: PASS")
    print("Judge/model/retrieval/reranker calls performed: 0")
    print("Release metrics calculated: NO")
    print("Frozen V4B release decision modified: NO")
    print("Next: freeze ranks 51-end for the six NO_CLEAR_THROUGH_50 queries only.")


if __name__ == "__main__":
    main()
