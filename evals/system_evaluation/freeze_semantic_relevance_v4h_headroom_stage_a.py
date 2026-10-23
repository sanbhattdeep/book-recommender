from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"
SYS = ROOT / "evals/system_evaluation"

FAILURE_LOCK = RUN/"v4h_failure_analysis.lock.json"
FAILURE_QUERY_DIAG = RUN/"v4h_failure_query_diagnostics.csv"
FINAL_RELEASE_LOCK = RUN/"v4h_final_release_analysis.lock.json"
RERANKING_LOCK = RUN/"v4h_reranking.lock.json"
V4B_FULL = RUN/"v4h_v4b_ranked_full_union.csv"
JUDGE_SCORES = RUN/"v4h_judge_scores.csv"
CONTRACT = SYS/"semantic_relevance_v4h_no_clear_headroom_stage_a.v0.1.0.json"
CANDIDATES = RUN/"v4h_no_clear_headroom_stage_a_candidates.csv"
METADATA = RUN/"v4h_no_clear_headroom_stage_a_metadata.json"
LOCK = RUN/"v4h_no_clear_headroom_stage_a.lock.json"


TARGET_QUERY_IDS = ['V4H07', 'V4H09', 'V4H10', 'V4H11', 'V4H13', 'V4H16', 'V4H19', 'V4H21', 'V4H22', 'V4H23']
EXPECTED_FAILURE_LOCK_SHA256 = "4e0f10f3d70fae6154c20d32622c21f437749d30835e892736cafa7d6ab3fc90"
EXPECTED_FAILURE_QUERY_DIAG_SHA256 = "09c1c74f49d09ae76fb95c2a38ca7d69197e29caa7737875d87de946362363c8"
EXPECTED_FINAL_RELEASE_LOCK_SHA256 = "7e7d73920452ec39ea72947d2b3762a02aaa1b2bb0bcd7470c7e21dbb3a25ab0"
EXPECTED_RERANKING_LOCK_SHA256 = "6d3a7014f5995d978a90df627c720c0b81aa2e8b28e8bba99cfa2c64d27811b5"
EXPECTED_JUDGE_SCORES_SHA256 = "8f8249a609287477a7f094ef774dd8f8b2af4b9b9b142d58e6c22123ae15f906"
EXPECTED_CONTRACT_SHA256 = "e47636abed2ac834c69ad9ce17c8c32fa4803b9f0cbca86dfd44f8a83ab7639d"
EXPECTED_STAGE_A_ROWS = 400

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

def verify_boundary() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    verify_raw(FAILURE_LOCK, EXPECTED_FAILURE_LOCK_SHA256, "V4H failure-analysis lock")
    verify_raw(FAILURE_QUERY_DIAG, EXPECTED_FAILURE_QUERY_DIAG_SHA256, "V4H failure query diagnostics")
    verify_raw(FINAL_RELEASE_LOCK, EXPECTED_FINAL_RELEASE_LOCK_SHA256, "V4H final release-analysis lock")
    verify_raw(RERANKING_LOCK, EXPECTED_RERANKING_LOCK_SHA256, "V4H reranking lock")
    verify_raw(JUDGE_SCORES, EXPECTED_JUDGE_SCORES_SHA256, "V4H frozen judge scores")
    verify_raw(CONTRACT, EXPECTED_CONTRACT_SHA256, "V4H headroom Stage-A contract")

    flock = load_json(FAILURE_LOCK)
    req(flock.get("status") == "V4H_POST_RELEASE_FAILURE_ANALYSIS_FROZEN",
        "Unexpected failure-analysis lock status")
    req(flock.get("frozen_release_decision") == "FAIL", "Frozen release decision changed")
    req(flock.get("release_decision_modified") is False, "Release decision was modified")
    req(flock.get("v4h_consumed") is True, "V4H is not marked consumed")

    rlock = load_json(RERANKING_LOCK)
    req(rlock.get("status") == "V4H_V3B_V4B_RERANKING_FROZEN_BEFORE_DISPLAY_UNION_OR_JUDGING",
        "Unexpected reranking lock status")
    full_sha = rlock.get("v4b_ranked_full_union_sha256")
    req(isinstance(full_sha, str) and len(full_sha) == 64,
        "Reranking lock missing frozen V4B full-union SHA-256")
    verify_raw(V4B_FULL, full_sha, "V4B frozen ranked full union")

    diag = pd.read_csv(FAILURE_QUERY_DIAG, dtype=str, keep_default_na=False, encoding="utf-8")
    observed = diag.loc[diag["v4_failure_signature"] == "NO_CLEAR_TOP10", "query_id"].tolist()
    req(observed == TARGET_QUERY_IDS,
        f"NO_CLEAR_TOP10 target set changed: observed={observed}")
    req((pd.to_numeric(
        diag.loc[diag["query_id"].isin(TARGET_QUERY_IDS), "v4_clear_count"],
        errors="raise"
    ) == 0).all(), "A target query no longer has v4_clear_count=0")

    full = pd.read_csv(V4B_FULL, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id","query","stratum","isbn13","title","description",
        "v4b_rank","reranker_score","hybrid_retrieval_rank"
    }
    missing = required - set(full.columns)
    req(not missing, f"V4B full ranking missing columns: {sorted(missing)}")
    full["v4b_rank"] = pd.to_numeric(full["v4b_rank"], errors="raise").astype(int)
    req(not full.duplicated(["query_id","isbn13"]).any(), "Duplicate V4B full-union pair")
    for qid in TARGET_QUERY_IDS:
        g = full[full["query_id"] == qid]
        req(len(g) >= 50, f"{qid} has fewer than 50 V4B ranked candidates")
        req(set(range(1,51)).issubset(set(g["v4b_rank"].tolist())),
            f"{qid} missing a rank in 1..50")

    scores = pd.read_csv(JUDGE_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    scores["judge_score"] = pd.to_numeric(scores["judge_score"], errors="raise").astype(int)
    for qid in TARGET_QUERY_IDS:
        g = scores[
            (scores["query_id"] == qid) &
            (scores["in_v4b"].str.lower().isin(["true","1","yes"]))
        ].copy()
        g["rank"] = pd.to_numeric(g["v4b_rank"], errors="raise").astype(int)
        g = g[g["rank"].between(1,10)]
        req(len(g) == 10, f"{qid} does not have exact judged V4B top10")
        req((g["judge_score"] < 3).all(), f"{qid} unexpectedly has clear/strong V4B top10")

    return diag, full, rlock


def main() -> None:
    print("Semantic relevance V4H - freeze no-clear headroom Stage-A candidates")
    print("-------------------------------------------------------------------")
    print("Judge/model/retrieval/reranker calls: DISABLED")
    print("Release decision modification: DISABLED")
    print()

    diag, full, rlock = verify_boundary()
    for path in [CANDIDATES, METADATA, LOCK]:
        req(not path.exists(), f"Refusing to overwrite existing artifact: {path}")

    selected = full[
        full["query_id"].isin(TARGET_QUERY_IDS) &
        full["v4b_rank"].between(11,50)
    ].copy()
    selected = selected.sort_values(["query_id","v4b_rank"], kind="stable")
    req(len(selected) == EXPECTED_STAGE_A_ROWS, "Stage-A candidate count changed")
    req((selected.groupby("query_id").size() == 40).all(), "Expected 40 candidates/query")

    keep = [
        c for c in [
            "query_id","query","stratum","v4b_rank","hybrid_retrieval_rank",
            "dense_rank","bm25_rank","candidate_source","isbn13","title",
            "description","authors","reranker_score"
        ] if c in selected.columns
    ]
    selected[keep].to_csv(CANDIDATES, index=False, encoding="utf-8")
    candidate_sha = sha256(CANDIDATES)

    metadata = {
        "status": "V4H_NO_CLEAR_HEADROOM_STAGE_A_CANDIDATES_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic",
        "target_query_ids": TARGET_QUERY_IDS,
        "target_query_count": 10,
        "rank_window": [11,50],
        "candidates_per_query": 40,
        "candidate_count": len(selected),
        "candidate_sha256": candidate_sha,
        "source_v4b_ranked_full_union_sha256": rlock["v4b_ranked_full_union_sha256"],
        "failure_query_diagnostics_sha256": EXPECTED_FAILURE_QUERY_DIAG_SHA256,
        "judge_calls": 0,
        "model_calls": 0,
        "release_decision": "FAIL",
        "release_decision_modified": False,
        "next_stage": "PREPARE_FRESH_FROZEN_JUDGE_INPUT_FOR_400_STAGE_A_PAIRS",
    }
    METADATA.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")

    lock = {
        "status": "V4H_NO_CLEAR_HEADROOM_STAGE_A_CANDIDATES_FROZEN_BEFORE_JUDGING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "failure_analysis_lock_sha256": EXPECTED_FAILURE_LOCK_SHA256,
        "final_release_analysis_lock_sha256": EXPECTED_FINAL_RELEASE_LOCK_SHA256,
        "reranking_lock_sha256": EXPECTED_RERANKING_LOCK_SHA256,
        "source_v4b_ranked_full_union_sha256": rlock["v4b_ranked_full_union_sha256"],
        "candidate_count": len(selected),
        "candidate_sha256": candidate_sha,
        "target_query_ids": TARGET_QUERY_IDS,
        "rank_window": [11,50],
        "judge_calls": 0,
        "release_decision_modified": False,
        "metadata_sha256": sha256(METADATA),
        "next_stage": "PREPARE_FRESH_FROZEN_JUDGE_INPUT_FOR_400_STAGE_A_PAIRS",
    }
    LOCK.write_text(json.dumps(lock, indent=2)+"\n", encoding="utf-8")

    print("PASS  exact target set frozen: " + ", ".join(TARGET_QUERY_IDS))
    print("PASS  exact Stage-A rank window frozen: 11-50")
    print("PASS  exact candidate inventory frozen: 400 pairs")
    print(f"Stage-A candidates SHA-256: {candidate_sha}")
    print(f"Stage-A candidate lock SHA-256: {sha256(LOCK)}")
    print("Judge calls performed: 0")
    print("Frozen V4B release decision modified: NO")
    print("V4H HEADROOM STAGE-A CANDIDATE FREEZE: PASS")
    print("Next: prepare fresh frozen judge input for these 400 pairs.")

if __name__ == "__main__":
    main()
