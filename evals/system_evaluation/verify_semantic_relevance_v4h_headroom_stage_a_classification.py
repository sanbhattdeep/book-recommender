from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"
SYS = ROOT / "evals/system_evaluation"

CANDIDATES = RUN / "v4h_no_clear_headroom_stage_a_candidates.csv"
CANDIDATE_LOCK = RUN / "v4h_no_clear_headroom_stage_a.lock.json"
SCORE_INVENTORY = RUN / "v4h_headroom_stage_a_score_inventory.csv"
JUDGING_INPUT = RUN / "v4h_headroom_stage_a_judging_input.csv"
JUDGING_INPUT_LOCK = RUN / "v4h_headroom_stage_a_judging_input.lock.json"
EXISTING_SCORES = RUN / "v4h_judge_scores.csv"
EXISTING_SCORING_LOCK = RUN / "v4h_judge_scoring.lock.json"
FRESH_SCORES = RUN / "v4h_headroom_stage_a_fresh_judge_scores.csv"
FRESH_SCORING_LOCK = RUN / "v4h_headroom_stage_a_fresh_judge_scoring.lock.json"
FAILURE_LOCK = RUN / "v4h_failure_analysis.lock.json"
FINAL_RELEASE_LOCK = RUN / "v4h_final_release_analysis.lock.json"
STAGE_A_CONTRACT = SYS / "semantic_relevance_v4h_no_clear_headroom_stage_a.v0.1.0.json"
JUDGING_CONTRACT = SYS / "semantic_relevance_v4h_headroom_stage_a_judging_contract.v0.1.0.json"
EXECUTION_CONTRACT = SYS / "semantic_relevance_v4h_headroom_stage_a_execution_contract.v0.1.0.json"
CLASSIFICATION_CONTRACT = SYS / "semantic_relevance_v4h_headroom_stage_a_classification_contract.v0.1.0.json"

COMPLETE_SCORES = RUN / "v4h_headroom_stage_a_complete_scores.csv"
QUERY_CLASSIFICATION = RUN / "v4h_headroom_stage_a_query_classification.csv"
SUMMARY = RUN / "v4h_headroom_stage_a_summary.json"
REPORT = RUN / "v4h_headroom_stage_a_report.md"
METADATA = RUN / "v4h_headroom_stage_a_classification_metadata.json"
CLASSIFICATION_LOCK = RUN / "v4h_headroom_stage_a_classification.lock.json"

TARGET_QUERY_IDS = ["V4H07","V4H09","V4H10","V4H11","V4H13","V4H16","V4H19","V4H21","V4H22","V4H23"]
EXPECTED = {
    "candidates": "2f2bfe05e1c21bb6a7b8c6dbe6c37fca8c42b9985ebeb74fcda710f5b629a68d",
    "candidate_lock": "cb0ee919fd4a830d6d7e30f7acc91854470354684768ab9cf4762ec289f1fdfa",
    "score_inventory": "fbbc796bc885842e5f9e63ce9ea7d2c7540bdeea09aa34b51c44a80988807cda",
    "judging_input": "eb57cfead0f79cedb9f350b3be683a5b063ab62355cc9074e7df264fc4fe14c8",
    "judging_input_lock": "67d1dfbc2852adf96c97de0a34603010f06bab37aa77854b21c6dd5325653173",
    "existing_scores": "8f8249a609287477a7f094ef774dd8f8b2af4b9b9b142d58e6c22123ae15f906",
    "existing_scoring_lock": "ac7a96c89eb2d4a51eda1b56756ed3b943311e7f300b661659cff51285423a8f",
    "fresh_scores": "5a1c988e0efb6898e721f700261e834ed1994de107338dc49e68549aa9e1ca02",
    "fresh_scoring_lock": "b64ff7d8828fcd4d8938c487d48335c0837aaca0d6439f42b91579467a5c57f9",
    "failure_lock": "4e0f10f3d70fae6154c20d32622c21f437749d30835e892736cafa7d6ab3fc90",
    "final_release_lock": "7e7d73920452ec39ea72947d2b3762a02aaa1b2bb0bcd7470c7e21dbb3a25ab0",
    "stage_a_contract": "e47636abed2ac834c69ad9ce17c8c32fa4803b9f0cbca86dfd44f8a83ab7639d",
    "judging_contract": "75810d2451809943a1e5d586029aa27ab42f87113a028273d5ee582496cf9422",
    "execution_contract": "5385d221e2c80968dd976b4bf18abdbda6bbc3c8b30db264167f572c78b47dbb",
    "classification_contract": "16228e03a1148d298aa0e2529265d3ddeee021062882b6c043e6cf9bdd33a5a9",
}

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

def canonical_isbn(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    req(bool(text), "ISBN is empty")
    return text

def verify_boundary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checks = [
        (CANDIDATES, "candidates", "Stage-A candidate inventory"),
        (CANDIDATE_LOCK, "candidate_lock", "Stage-A candidate lock"),
        (SCORE_INVENTORY, "score_inventory", "Stage-A score inventory"),
        (JUDGING_INPUT, "judging_input", "Stage-A fresh judging input"),
        (JUDGING_INPUT_LOCK, "judging_input_lock", "Stage-A judging-input lock"),
        (EXISTING_SCORES, "existing_scores", "frozen V4H judge scores"),
        (EXISTING_SCORING_LOCK, "existing_scoring_lock", "frozen V4H judge scoring lock"),
        (FRESH_SCORES, "fresh_scores", "Stage-A fresh judge scores"),
        (FRESH_SCORING_LOCK, "fresh_scoring_lock", "Stage-A fresh scoring lock"),
        (FAILURE_LOCK, "failure_lock", "V4H failure-analysis lock"),
        (FINAL_RELEASE_LOCK, "final_release_lock", "V4H final release-analysis lock"),
        (STAGE_A_CONTRACT, "stage_a_contract", "Stage-A definition contract"),
        (JUDGING_CONTRACT, "judging_contract", "Stage-A judging-preparation contract"),
        (EXECUTION_CONTRACT, "execution_contract", "Stage-A execution contract"),
        (CLASSIFICATION_CONTRACT, "classification_contract", "Stage-A classification contract"),
    ]
    for path, key, label in checks:
        verify_raw(path, EXPECTED[key], label)

    failure = load_json(FAILURE_LOCK)
    req(failure.get("status") == "V4H_POST_RELEASE_FAILURE_ANALYSIS_FROZEN", "Unexpected failure-analysis lock status")
    req(failure.get("frozen_release_decision") == "FAIL", "Frozen V4B release decision changed")
    req(failure.get("release_decision_modified") is False, "Failure analysis modified release decision")
    req(failure.get("v4h_consumed") is True, "V4H is not marked consumed")

    final = load_json(FINAL_RELEASE_LOCK)
    req(final.get("status") == "V4H_FINAL_RELEASE_ANALYSIS_FROZEN", "Unexpected final release-analysis lock status")
    req(final.get("v4b_final_release_decision") == "FAIL", "V4B final release decision changed")

    fresh_lock = load_json(FRESH_SCORING_LOCK)
    req(fresh_lock.get("status") == "V4H_HEADROOM_STAGE_A_FRESH_JUDGE_SCORES_FROZEN", "Unexpected Stage-A fresh-score lock status")
    req(int(fresh_lock.get("fresh_pair_count", -1)) == 340, "Fresh pair count changed")
    req(int(fresh_lock.get("reused_exact_pair_count_external", -1)) == 60, "Reused pair count changed")
    req(int(fresh_lock.get("total_stage_a_pair_count", -1)) == 400, "Stage-A pair count changed")
    req(fresh_lock.get("fresh_judge_scores_sha256") == EXPECTED["fresh_scores"], "Fresh-score lock does not bind expected score artifact")
    req(fresh_lock.get("human_relevance_labels_used") is False, "Fresh-score lock says human labels were used")
    req(fresh_lock.get("release_metrics_calculated") is False, "Fresh-score lock says release metrics were calculated")
    req(fresh_lock.get("release_decision_modified") is False, "Fresh-score lock says release decision was modified")

    inventory = pd.read_csv(SCORE_INVENTORY, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(inventory) == 400, "Expected exactly 400 Stage-A inventory rows")
    required = {"case_id","query_id","stratum","v4b_rank","isbn13","score_source","reused_judge_score"}
    req(required <= set(inventory.columns), f"Stage-A score inventory missing columns: {sorted(required-set(inventory.columns))}")
    inventory["isbn13"] = inventory["isbn13"].map(canonical_isbn)
    inventory["v4b_rank"] = pd.to_numeric(inventory["v4b_rank"], errors="raise").astype(int)
    req(not inventory["case_id"].duplicated().any(), "Duplicate Stage-A case_id")
    req(not inventory.duplicated(["query_id","isbn13"]).any(), "Duplicate Stage-A query-book pair")
    req(sorted(inventory["query_id"].unique()) == TARGET_QUERY_IDS, "Stage-A target query set changed")
    req(inventory["v4b_rank"].between(11,50).all(), "Stage-A inventory contains rank outside 11-50")
    req((inventory.groupby("query_id").size() == 40).all(), "Expected 40 Stage-A candidates per query")
    counts = inventory["score_source"].value_counts().to_dict()
    req(counts == {"fresh_judge_required": 340, "reused_frozen_v4h_display_score": 60}, f"Stage-A score-source partition changed: {counts}")

    fresh = pd.read_csv(FRESH_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    fresh_required = {"case_id","query_id","isbn13","v4b_rank","judge_score","match_level"}
    req(fresh_required <= set(fresh.columns), f"Fresh score artifact missing columns: {sorted(fresh_required-set(fresh.columns))}")
    req(len(fresh) == 340, "Expected exactly 340 fresh Stage-A scores")
    fresh["isbn13"] = fresh["isbn13"].map(canonical_isbn)
    fresh["v4b_rank"] = pd.to_numeric(fresh["v4b_rank"], errors="raise").astype(int)
    fresh["judge_score"] = pd.to_numeric(fresh["judge_score"], errors="raise").astype(int)
    req(fresh["judge_score"].between(0,4).all(), "Fresh judge score outside 0..4")
    req(not fresh["case_id"].duplicated().any(), "Duplicate fresh case_id")
    req(not fresh.duplicated(["query_id","isbn13"]).any(), "Duplicate fresh query-book pair")

    reused = inventory[inventory["score_source"] == "reused_frozen_v4h_display_score"].copy()
    fresh_inventory = inventory[inventory["score_source"] == "fresh_judge_required"].copy()
    req(set(fresh["case_id"]) == set(fresh_inventory["case_id"]), "Fresh score case set does not equal frozen fresh inventory")
    for row in fresh.itertuples(index=False):
        src = fresh_inventory.set_index("case_id").loc[str(row.case_id)]
        req(str(row.query_id) == str(src["query_id"]), f"{row.case_id} query mismatch")
        req(canonical_isbn(row.isbn13) == canonical_isbn(src["isbn13"]), f"{row.case_id} ISBN mismatch")
        req(int(row.v4b_rank) == int(src["v4b_rank"]), f"{row.case_id} rank mismatch")

    reused_scores = pd.to_numeric(reused["reused_judge_score"], errors="raise").astype(int)
    req(reused_scores.between(0,4).all(), "Reused Stage-A score outside 0..4")

    existing = pd.read_csv(EXISTING_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    req({"query_id","isbn13","judge_score"} <= set(existing.columns), "Frozen V4H score artifact missing required columns")
    existing["isbn13"] = existing["isbn13"].map(canonical_isbn)
    existing["judge_score"] = pd.to_numeric(existing["judge_score"], errors="raise").astype(int)
    req(not existing.duplicated(["query_id","isbn13"]).any(), "Frozen V4H score artifact has duplicate pair")
    lookup = existing.set_index(["query_id","isbn13"])["judge_score"].to_dict()
    for row in reused.itertuples(index=False):
        key = (str(row.query_id), canonical_isbn(row.isbn13))
        req(key in lookup, f"Reused pair missing from frozen V4H scores: {key}")
        req(int(row.reused_judge_score) == int(lookup[key]), f"Reused score mismatch for {key}")

    return inventory, fresh, existing


def main() -> None:
    print("Semantic relevance V4H - headroom Stage-A classification preflight")
    print("-----------------------------------------------------------------")
    print("Role: consumed-holdout diagnostic only")
    print("Judge/model/retrieval/reranker calls: DISABLED")
    print("Release metrics/gates: NOT CALCULATED")
    print("Frozen V4B release decision modification: DISABLED")
    print()
    verify_boundary()
    req(not CLASSIFICATION_LOCK.exists(), "Stage-A classification lock already exists")
    for path in [COMPLETE_SCORES, QUERY_CLASSIFICATION, SUMMARY, REPORT, METADATA]:
        req(not path.exists(), f"Refusing to overwrite existing Stage-A classification output: {path}")
    print("PASS  Stage-A 400-pair score inventory verified: 60 exact reused + 340 fresh")
    print("PASS  all 340 fresh Stage-A scores frozen and bound by final scoring lock")
    print("PASS  reused scores independently agree with frozen V4H score artifact")
    print("PASS  classification rule remains score >=3 in ranks 11-50")
    print()
    print("V4H HEADROOM STAGE-A CLASSIFICATION PREFLIGHT: PASS")
    print("Judge/model/retrieval/reranker calls performed: 0")
    print("Release metrics calculated: NO")
    print("Frozen V4B release decision modified: NO")
    print("Next: run deterministic Stage-A score combination and query classification.")

if __name__ == "__main__":
    main()
