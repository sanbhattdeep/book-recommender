from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
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


def verify_boundary() -> tuple[list[str], list[str], pd.DataFrame, dict[str, Any], str]:
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

    outputs = clock.get("outputs", {})
    expected_q_sha = outputs.get(QUERY_CLASSIFICATION.name)
    req(isinstance(expected_q_sha, str) and len(expected_q_sha) == 64,
        "Stage-A classification lock does not bind query-classification artifact")
    verify_raw(QUERY_CLASSIFICATION, expected_q_sha, "Stage-A query classification")

    q = pd.read_csv(QUERY_CLASSIFICATION, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(q) == 10, f"Expected 10 Stage-A query rows, found {len(q)}")
    req(not q["query_id"].duplicated().any(), "Duplicate query_id in Stage-A classification")
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

    return unresolved, resolved, full, rlock, expected_q_sha


def main() -> None:
    print("Semantic relevance V4H - freeze no-clear headroom Stage-B candidates")
    print("-------------------------------------------------------------------")
    print("Judge/model/retrieval/reranker calls: DISABLED")
    print("Release decision modification: DISABLED")
    print()

    unresolved, resolved, full, rlock, qclass_sha = verify_boundary()
    for path in [CANDIDATES, QUERY_INVENTORY, METADATA, LOCK]:
        req(not path.exists(), f"Refusing to overwrite existing Stage-B artifact: {path}")

    selected = full[full["query_id"].isin(unresolved) & (full["v4b_rank"] >= 51)].copy()
    selected = selected.sort_values(["query_id", "v4b_rank"], kind="stable")
    req(not set(selected["query_id"]).intersection(resolved),
        "Stage-B candidate inventory includes Stage-A HEADROOM_FOUND query")
    req((selected["v4b_rank"] >= 51).all(), "Stage-B candidate inventory includes rank <=50")

    keep = [c for c in [
        "query_id", "query", "stratum", "v4b_rank", "hybrid_retrieval_rank",
        "dense_rank", "bm25_rank", "candidate_source", "isbn13", "title",
        "description", "authors", "reranker_score"
    ] if c in full.columns]
    selected[keep].to_csv(CANDIDATES, index=False, encoding="utf-8")

    inventory_rows: list[dict[str, Any]] = []
    for qid in unresolved:
        g = full[full["query_id"] == qid].sort_values("v4b_rank", kind="stable")
        max_rank = int(g["v4b_rank"].max())
        deep_count = int((g["v4b_rank"] >= 51).sum())
        req(deep_count == max(0, max_rank - 50), f"{qid} Stage-B rank count mismatch")
        inventory_rows.append({
            "query_id": qid,
            "stratum": g.iloc[0]["stratum"],
            "max_v4b_rank": max_rank,
            "stage_b_rank_start": 51 if deep_count else "",
            "stage_b_rank_end": max_rank if deep_count else "",
            "stage_b_candidate_count": deep_count,
            "union_exhausted_at_50": deep_count == 0,
        })

    inventory = pd.DataFrame(inventory_rows)
    req(len(inventory) == 6, "Expected six Stage-B query inventory rows")
    req(int(inventory["stage_b_candidate_count"].sum()) == len(selected),
        "Stage-B candidate total does not match per-query inventory")
    QUERY_INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(QUERY_INVENTORY, index=False, encoding="utf-8")

    candidate_sha = sha256(CANDIDATES)
    inventory_sha = sha256(QUERY_INVENTORY)
    target_counts = {
        row["query_id"]: int(row["stage_b_candidate_count"])
        for row in inventory_rows
    }

    metadata = {
        "status": "V4H_NO_CLEAR_HEADROOM_STAGE_B_CANDIDATES_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic",
        "source_stage_a_classification_lock_sha256": EXPECTED_CLASSIFICATION_LOCK_SHA256,
        "source_stage_a_query_classification_sha256": qclass_sha,
        "source_v4b_ranked_full_union_sha256": rlock["v4b_ranked_full_union_sha256"],
        "target_query_ids": unresolved,
        "excluded_headroom_found_query_ids": resolved,
        "target_query_count": len(unresolved),
        "rank_window": "51-end",
        "candidate_count": len(selected),
        "candidate_counts_by_query": target_counts,
        "candidate_sha256": candidate_sha,
        "query_inventory_sha256": inventory_sha,
        "judge_calls": 0,
        "model_calls": 0,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "release_metrics_calculated": False,
        "release_decision": "FAIL",
        "release_decision_modified": False,
        "next_stage": "PREPARE_STAGE_B_SCORE_REUSE_AND_FRESH_JUDGING_INPUT",
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V4H_NO_CLEAR_HEADROOM_STAGE_B_CANDIDATES_FROZEN_BEFORE_JUDGING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic",
        "diagnostic_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "stage_a_classification_lock_sha256": EXPECTED_CLASSIFICATION_LOCK_SHA256,
        "stage_a_query_classification_sha256": qclass_sha,
        "reranking_lock_sha256": EXPECTED_RERANKING_LOCK_SHA256,
        "source_v4b_ranked_full_union_sha256": rlock["v4b_ranked_full_union_sha256"],
        "target_query_ids": unresolved,
        "excluded_headroom_found_query_ids": resolved,
        "target_query_count": len(unresolved),
        "rank_window": "51-end",
        "candidate_count": len(selected),
        "candidate_counts_by_query": target_counts,
        "candidate_sha256": candidate_sha,
        "query_inventory_sha256": inventory_sha,
        "judge_calls": 0,
        "model_calls": 0,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "release_metrics_calculated": False,
        "frozen_release_decision": "FAIL",
        "release_decision_modified": False,
        "metadata_sha256": sha256(METADATA),
        "next_stage": "PREPARE_STAGE_B_SCORE_REUSE_AND_FRESH_JUDGING_INPUT",
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print("PASS  exact Stage-B target set frozen from Stage-A NO_CLEAR_THROUGH_50 classification")
    print("PASS  all four Stage-A HEADROOM_FOUND_11_50 queries excluded")
    print("PASS  exact Stage-B rank window frozen: 51-end of each query's frozen V4B full union")
    print(f"PASS  exact Stage-B candidate inventory frozen: {len(selected)} pairs")
    for row in inventory_rows:
        print(f"INFO  {row['query_id']}: max_rank={row['max_v4b_rank']} stage_b_candidates={row['stage_b_candidate_count']}")
    print(f"Stage-B candidates SHA-256: {candidate_sha}")
    print(f"Stage-B query inventory SHA-256: {inventory_sha}")
    print(f"Stage-B candidate lock SHA-256: {sha256(LOCK)}")
    print("Judge/model/retrieval/reranker calls performed: 0")
    print("Release metrics calculated: NO")
    print("Frozen V4B release decision modified: NO")
    print("V4H HEADROOM STAGE-B CANDIDATE FREEZE: PASS")
    print("Next: prepare exact score reuse and fresh judging input for the frozen Stage-B pairs.")


if __name__ == "__main__":
    main()
