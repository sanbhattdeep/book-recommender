from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

SCORE_INVENTORY = RUN / "v4a_dev_display_score_inventory.csv"
NOVEL_INPUT = RUN / "v4a_dev_novel_display_pairs_to_judge.csv"
JUDGING_INPUT_LOCK = RUN / "v4a_dev_judging_input.lock.json"
NOVEL_SCORES = RUN / "v4a_dev_novel_judge_scores.csv"
NOVEL_SCORING_LOCK = RUN / "v4a_dev_novel_judge_scoring.lock.json"

COMPLETE_MAP = RUN / "v4a_dev_complete_display_score_map.csv"
FINALIZATION_METADATA = RUN / "v4a_dev_score_finalization_metadata.json"
FINALIZATION_LOCK = RUN / "v4a_dev_complete_score_map.lock.json"

EXPECTED = {
    "score_inventory": "363ef15e6e8cfa07afcab6e73b20d0d2e21d778235f1800ab388a22f6ac8f818",
    "novel_input": "3d33db6b52314b1a022705c0545798ded6cf17a5e3e4355aad19b5cd43dcdd56",
    "judging_input_lock": "551d22ffa5c8e3d04aa5457675d47c40abed19d6162efc66dde2a8c329a95c97",
    "novel_scores": "87b3a425e793e402f313268b653e76b8e162baec6ddb5d9167b7835a7de19044",
    "novel_scoring_lock": "3be5b00e48d057093f8b5c8e98ca29f66f7c88e70441043d601487960c26e8bf",
}

EXPECTED_TOTAL = 347
EXPECTED_REUSED = 268
EXPECTED_NOVEL = 79
EXPECTED_QUERIES = 24
EXPECTED_SYSTEM_ROWS = 240


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bool_value(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def verify_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checks = [
        ("score inventory", SCORE_INVENTORY, EXPECTED["score_inventory"]),
        ("novel judging input", NOVEL_INPUT, EXPECTED["novel_input"]),
        ("judging-input lock", JUDGING_INPUT_LOCK, EXPECTED["judging_input_lock"]),
        ("novel judge scores", NOVEL_SCORES, EXPECTED["novel_scores"]),
        ("novel scoring lock", NOVEL_SCORING_LOCK, EXPECTED["novel_scoring_lock"]),
    ]
    for label, path, expected in checks:
        require(path.exists(), f"Missing frozen input: {path}")
        actual = sha256(path)
        require(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
        print(f"PASS  frozen {label} SHA-256 verified")

    input_lock = load_json(JUDGING_INPUT_LOCK)
    require(
        input_lock.get("status") == "V4A_DEV_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "Unexpected judging-input lock status",
    )
    require(int(input_lock.get("display_union_rows", -1)) == EXPECTED_TOTAL, "Judging-input total changed")
    require(int(input_lock.get("reused_frozen_scores", -1)) == EXPECTED_REUSED, "Judging-input reused count changed")
    require(int(input_lock.get("novel_pairs_to_judge", -1)) == EXPECTED_NOVEL, "Judging-input novel count changed")
    require(input_lock.get("judge_execution_performed") is False, "Judging-input lock must represent pre-judge freeze")

    scoring_lock = load_json(NOVEL_SCORING_LOCK)
    require(
        scoring_lock.get("status") == "V4A_DEV_NOVEL_JUDGE_SCORES_FROZEN_BEFORE_FINALIZATION",
        "Unexpected novel scoring-lock status",
    )
    require(int(scoring_lock.get("novel_pairs", -1)) == EXPECTED_NOVEL, "Novel scoring-lock count changed")
    require(scoring_lock.get("novel_judge_scores_sha256") == EXPECTED["novel_scores"], "Novel score hash differs inside scoring lock")
    require(scoring_lock.get("human_relevance_labels_used") is False, "Novel scoring lock says human labels were used")
    require(scoring_lock.get("release_metrics_calculated") is False, "Novel scoring stage calculated release metrics")
    require(scoring_lock.get("development_selection_decision_produced") is False, "Novel scoring stage produced a development decision")

    inventory = pd.read_csv(SCORE_INVENTORY, dtype=str, keep_default_na=False, encoding="utf-8")
    required_inventory = {
        "query_id", "query", "stratum", "isbn13", "title", "description", "authors",
        "pair_source", "in_v3b", "v3b_rank", "in_v4a", "v4a_rank", "score_status",
        "score_source", "judge_score", "match_level", "judge_reason", "scoring_explanation",
        "facet_assessments_json", "facet_evidence_json", "component_evidence_ledger_json",
    }
    missing = required_inventory - set(inventory.columns)
    require(not missing, f"Score inventory missing columns: {sorted(missing)}")
    require(len(inventory) == EXPECTED_TOTAL, f"Expected {EXPECTED_TOTAL} score-inventory rows")
    require(not inventory.duplicated(["query_id", "isbn13"]).any(), "Duplicate pair in score inventory")
    require(inventory["query_id"].nunique() == EXPECTED_QUERIES, "Score inventory is not H01-H24")
    require(int((inventory["score_status"] == "reused").sum()) == EXPECTED_REUSED, "Reused count changed")
    require(int((inventory["score_status"] == "novel").sum()) == EXPECTED_NOVEL, "Novel count changed")

    in_v3b = inventory["in_v3b"].map(bool_value)
    in_v4a = inventory["in_v4a"].map(bool_value)
    require(int(in_v3b.sum()) == EXPECTED_SYSTEM_ROWS, "v3b membership is not exactly 240 rows")
    require(int(in_v4a.sum()) == EXPECTED_SYSTEM_ROWS, "v4a membership is not exactly 240 rows")
    require(inventory.loc[in_v3b, "score_status"].eq("reused").all(), "A v3b displayed pair is not reusable")
    require(inventory.loc[inventory["score_status"] == "novel", "pair_source"].eq("v4a_only").all(), "Novel inventory contains non-v4a-only pair")

    novel_input = pd.read_csv(NOVEL_INPUT, dtype=str, keep_default_na=False, encoding="utf-8")
    require(len(novel_input) == EXPECTED_NOVEL, "Novel input is not 79 rows")
    require(not novel_input["case_id"].duplicated().any(), "Duplicate novel input case IDs")
    require(not novel_input.duplicated(["query_id", "isbn13"]).any(), "Duplicate novel input pair")

    scores = pd.read_csv(NOVEL_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    score_required = {
        "case_id", "query_id", "query", "stratum", "isbn13", "title", "authors",
        "pair_source", "in_v3b", "v3b_rank", "in_v4a", "v4a_rank", "match_level",
        "judge_score", "judge_reason", "scoring_explanation", "facet_assessments_json",
        "facet_evidence_json", "component_evidence_ledger_json",
    }
    missing = score_required - set(scores.columns)
    require(not missing, f"Novel judge scores missing columns: {sorted(missing)}")
    require(len(scores) == EXPECTED_NOVEL, "Novel score artifact is not 79 rows")
    require(not scores["case_id"].duplicated().any(), "Duplicate novel score case IDs")
    require(not scores.duplicated(["query_id", "isbn13"]).any(), "Duplicate novel judged pair")
    values = pd.to_numeric(scores["judge_score"], errors="raise").astype(int)
    require(values.between(0, 4).all(), "Novel judge score outside 0..4")

    expected_cases = novel_input.set_index("case_id", drop=False)
    actual_cases = scores.set_index("case_id", drop=False)
    require(set(expected_cases.index) == set(actual_cases.index), "Novel judged case set differs from frozen input")
    for field in ["query_id", "query", "stratum", "isbn13", "title", "pair_source", "in_v3b", "v3b_rank", "in_v4a", "v4a_rank"]:
        left = expected_cases.loc[actual_cases.index, field].astype(str)
        right = actual_cases[field].astype(str)
        require((left == right).all(), f"Novel judged payload differs from frozen input: {field}")

    print("PASS  frozen reuse/novel boundary verified: 268 reused + 79 judged = 347")
    return inventory, novel_input, scores


def build_complete_map(inventory: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    score_payload = scores[[
        "query_id", "isbn13", "judge_score", "match_level", "judge_reason",
        "scoring_explanation", "facet_assessments_json", "facet_evidence_json",
        "component_evidence_ledger_json",
    ]].copy().rename(columns={
        "judge_score": "new_judge_score",
        "match_level": "new_match_level",
        "judge_reason": "new_judge_reason",
        "scoring_explanation": "new_scoring_explanation",
        "facet_assessments_json": "new_facet_assessments_json",
        "facet_evidence_json": "new_facet_evidence_json",
        "component_evidence_ledger_json": "new_component_evidence_ledger_json",
    })

    merged = inventory.merge(score_payload, on=["query_id", "isbn13"], how="left", validate="one_to_one")
    reused = merged["score_status"].eq("reused")
    novel = merged["score_status"].eq("novel")

    require(merged.loc[reused, "new_judge_score"].isna().all(), "A reused pair unexpectedly has a new judge score")
    require(merged.loc[novel, "new_judge_score"].notna().all(), "A novel pair is missing its frozen new judgment")

    # Use object staging to remain safe under pandas 3 nullable/string defaults.
    for col in ["final_score_source", "final_judge_score", "final_match_level", "final_judge_reason",
                "final_scoring_explanation", "final_facet_assessments_json", "final_facet_evidence_json",
                "final_component_evidence_ledger_json"]:
        merged[col] = pd.Series([None] * len(merged), dtype="object")

    merged.loc[reused, "final_score_source"] = merged.loc[reused, "score_source"].astype("object")
    merged.loc[reused, "final_judge_score"] = merged.loc[reused, "judge_score"].astype("object")
    merged.loc[reused, "final_match_level"] = merged.loc[reused, "match_level"].astype("object")
    merged.loc[reused, "final_judge_reason"] = merged.loc[reused, "judge_reason"].astype("object")
    merged.loc[reused, "final_scoring_explanation"] = merged.loc[reused, "scoring_explanation"].astype("object")
    merged.loc[reused, "final_facet_assessments_json"] = merged.loc[reused, "facet_assessments_json"].astype("object")
    merged.loc[reused, "final_facet_evidence_json"] = merged.loc[reused, "facet_evidence_json"].astype("object")
    merged.loc[reused, "final_component_evidence_ledger_json"] = merged.loc[reused, "component_evidence_ledger_json"].astype("object")

    merged.loc[novel, "final_score_source"] = "v4a_dev_novel_judged"
    merged.loc[novel, "final_judge_score"] = merged.loc[novel, "new_judge_score"].astype("object")
    merged.loc[novel, "final_match_level"] = merged.loc[novel, "new_match_level"].astype("object")
    merged.loc[novel, "final_judge_reason"] = merged.loc[novel, "new_judge_reason"].astype("object")
    merged.loc[novel, "final_scoring_explanation"] = merged.loc[novel, "new_scoring_explanation"].astype("object")
    merged.loc[novel, "final_facet_assessments_json"] = merged.loc[novel, "new_facet_assessments_json"].astype("object")
    merged.loc[novel, "final_facet_evidence_json"] = merged.loc[novel, "new_facet_evidence_json"].astype("object")
    merged.loc[novel, "final_component_evidence_ledger_json"] = merged.loc[novel, "new_component_evidence_ledger_json"].astype("object")

    final_scores = pd.to_numeric(merged["final_judge_score"], errors="raise").astype(int)
    require(final_scores.between(0, 4).all(), "Complete-map judge score outside 0..4")
    merged["final_judge_score"] = final_scores

    source_counts = merged["final_score_source"].value_counts().to_dict()
    require(int(source_counts.get("v3_holdout_frozen", 0)) == 242, "Expected 242 holdout-reused scores")
    require(int(source_counts.get("v3_posthoc_deep_frozen", 0)) == 26, "Expected 26 posthoc-reused scores")
    require(int(source_counts.get("v4a_dev_novel_judged", 0)) == 79, "Expected 79 newly judged scores")

    columns = [
        "query_id", "query", "stratum", "isbn13", "title", "description", "authors",
        "pair_source", "in_v3b", "v3b_rank", "in_v4a", "v4a_rank",
        "final_score_source", "final_judge_score", "final_match_level", "final_judge_reason",
        "final_scoring_explanation", "final_facet_assessments_json", "final_facet_evidence_json",
        "final_component_evidence_ledger_json",
    ]
    complete = merged[columns].copy()
    complete["_v3"] = complete["v3b_rank"].replace("", "9999").astype(int)
    complete["_v4"] = complete["v4a_rank"].replace("", "9999").astype(int)
    complete = complete.sort_values(["query_id", "_v3", "_v4", "isbn13"], kind="stable").drop(columns=["_v3", "_v4"]).reset_index(drop=True)

    require(len(complete) == EXPECTED_TOTAL, "Complete score map row count changed")
    require(not complete.duplicated(["query_id", "isbn13"]).any(), "Duplicate pair in complete score map")
    return complete


def main() -> None:
    print("Semantic relevance v4a - deterministic development score finalization")
    print("-------------------------------------------------------------------")
    print("Judge execution: DISABLED")
    print("Model execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Development comparison: NOT CALCULATED")
    print("Development selection: NOT PRODUCED")
    print("Release metrics/gates: NOT CALCULATED")
    print()

    inventory, _, scores = verify_inputs()
    complete = build_complete_map(inventory, scores)

    for path in [COMPLETE_MAP, FINALIZATION_METADATA, FINALIZATION_LOCK]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing finalization artifact: {path}")

    COMPLETE_MAP.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(COMPLETE_MAP, index=False, encoding="utf-8")

    source_counts = {k: int(v) for k, v in complete["final_score_source"].value_counts().to_dict().items()}
    metadata = {
        "status": "V4A_DEV_SCORE_FINALIZATION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_query_set": "H01-H24",
        "score_inventory_sha256": EXPECTED["score_inventory"],
        "novel_judging_input_sha256": EXPECTED["novel_input"],
        "judging_input_lock_sha256": EXPECTED["judging_input_lock"],
        "novel_judge_scores_sha256": EXPECTED["novel_scores"],
        "novel_judge_scoring_lock_sha256": EXPECTED["novel_scoring_lock"],
        "displayed_pair_count": EXPECTED_TOTAL,
        "reused_frozen_scores": EXPECTED_REUSED,
        "new_novel_scores": EXPECTED_NOVEL,
        "score_source_counts": source_counts,
        "complete_score_map_sha256": sha256(COMPLETE_MAP),
        "judge_calls_performed_during_finalization": 0,
        "model_calls_performed_during_finalization": 0,
        "human_relevance_labels_used": False,
        "development_comparison_calculated": False,
        "development_selection_decision_produced": False,
        "release_metrics_calculated": False,
        "release_gates_calculated": False,
        "next_stage": "v3b_vs_v4a_development_comparison_and_candidate_selection",
    }
    FINALIZATION_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V4A_DEV_COMPLETE_DISPLAY_SCORE_MAP_FROZEN_BEFORE_COMPARISON",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_query_set": "H01-H24",
        "score_inventory_sha256": EXPECTED["score_inventory"],
        "novel_judging_input_sha256": EXPECTED["novel_input"],
        "judging_input_lock_sha256": EXPECTED["judging_input_lock"],
        "novel_judge_scores_sha256": EXPECTED["novel_scores"],
        "novel_judge_scoring_lock_sha256": EXPECTED["novel_scoring_lock"],
        "displayed_pair_count": EXPECTED_TOTAL,
        "reused_frozen_scores": EXPECTED_REUSED,
        "new_novel_scores": EXPECTED_NOVEL,
        "score_source_counts": source_counts,
        "complete_score_map_sha256": sha256(COMPLETE_MAP),
        "finalization_metadata_sha256": sha256(FINALIZATION_METADATA),
        "judge_calls_performed_during_finalization": 0,
        "model_calls_performed_during_finalization": 0,
        "human_relevance_labels_used": False,
        "development_comparison_calculated": False,
        "development_selection_decision_produced": False,
        "release_metrics_calculated": False,
        "release_gates_calculated": False,
        "next_stage": "v3b_vs_v4a_development_comparison_and_candidate_selection",
    }
    FINALIZATION_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print("PASS  complete displayed-pair score map finalized: 347/347")
    print(f"INFO  score sources: {source_counts}")
    print("PASS  complete score map frozen before development comparison")
    print()
    print("V4A DEVELOPMENT SCORE FINALIZATION: PASS")
    print(f"Complete score map SHA-256: {sha256(COMPLETE_MAP)}")
    print(f"Finalization lock SHA-256: {sha256(FINALIZATION_LOCK)}")
    print("Judge calls during finalization: 0")
    print("Model calls during finalization: 0")
    print("Human relevance labels used: NO")
    print("Development comparison calculated: NO")
    print("Development selection produced: NO")
    print("Release metrics calculated: NO")
    print("Next: v3b-vs-v4a development comparison and candidate selection.")


if __name__ == "__main__":
    main()
