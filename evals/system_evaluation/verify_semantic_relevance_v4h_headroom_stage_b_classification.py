from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"
SYS = ROOT / "evals/system_evaluation"

STAGE_A_CLASSIFICATION_LOCK = RUN / "v4h_headroom_stage_a_classification.lock.json"
STAGE_A_QUERY_CLASSIFICATION = RUN / "v4h_headroom_stage_a_query_classification.csv"
CANDIDATES = RUN / "v4h_no_clear_headroom_stage_b_candidates.csv"
QUERY_INVENTORY = RUN / "v4h_no_clear_headroom_stage_b_query_inventory.csv"
CANDIDATE_LOCK = RUN / "v4h_no_clear_headroom_stage_b.lock.json"
SCORE_INVENTORY = RUN / "v4h_headroom_stage_b_score_inventory.csv"
JUDGING_INPUT = RUN / "v4h_headroom_stage_b_judging_input.csv"
JUDGING_INPUT_LOCK = RUN / "v4h_headroom_stage_b_judging_input.lock.json"
FRESH_SCORES = RUN / "v4h_headroom_stage_b_fresh_judge_scores.csv"
FRESH_SCORING_LOCK = RUN / "v4h_headroom_stage_b_fresh_judge_scoring.lock.json"
V4H_DISPLAY_SCORES = RUN / "v4h_judge_scores.csv"
V4H_DISPLAY_SCORING_LOCK = RUN / "v4h_judge_scoring.lock.json"
STAGE_A_FRESH_SCORES = RUN / "v4h_headroom_stage_a_fresh_judge_scores.csv"
STAGE_A_FRESH_SCORING_LOCK = RUN / "v4h_headroom_stage_a_fresh_judge_scoring.lock.json"
FINAL_RELEASE_LOCK = RUN / "v4h_final_release_analysis.lock.json"
STAGE_B_DEFINITION_CONTRACT = SYS / "semantic_relevance_v4h_no_clear_headroom_stage_b.v0.1.0.json"
JUDGING_CONTRACT = SYS / "semantic_relevance_v4h_headroom_stage_b_judging_contract.v0.1.0.json"
EXECUTION_CONTRACT = SYS / "semantic_relevance_v4h_headroom_stage_b_execution_contract.v0.1.0.json"
CLASSIFICATION_CONTRACT = SYS / "semantic_relevance_v4h_headroom_stage_b_classification_contract.v0.1.0.json"

COMPLETE_SCORES = RUN / "v4h_headroom_stage_b_complete_scores.csv"
QUERY_CLASSIFICATION = RUN / "v4h_headroom_stage_b_query_classification.csv"
OVERALL_DISPOSITION = RUN / "v4h_headroom_overall_query_disposition.csv"
SUMMARY = RUN / "v4h_headroom_stage_b_summary.json"
REPORT = RUN / "v4h_headroom_stage_b_report.md"
METADATA = RUN / "v4h_headroom_stage_b_classification_metadata.json"
CLASSIFICATION_LOCK = RUN / "v4h_headroom_stage_b_classification.lock.json"

TARGET_QUERY_IDS = ["V4H10", "V4H11", "V4H13", "V4H16", "V4H21", "V4H23"]
EXPECTED = {
    "stage_a_classification_lock": "906b5975345c9d9eed1273ecc4ccd0611880b14d6326bf9f0e74b46f7d63d948",
    "candidates": "910f1a3f05487ab95f83cf3a7c6415750957a35ff9cf0e95a697b209a58c255f",
    "query_inventory": "bca3146d181d52ee6c3e8f43db3795ccf79ce764799fc0089d34067b89ad302b",
    "candidate_lock": "eb598bec298d8dd617d7b291a3274b9e1c4ae3dcf5f45c8bf98708a6a179023f",
    "score_inventory": "661933691c032d714a16a4e972f3754a15ca8bfac41d0108c0f4953f2cf7815a",
    "judging_input": "ac5b832a2feb7c8f26dcc964cccf6b33000147f005482047ea0c36e4e51bb591",
    "judging_input_lock": "49ae1b46c247a94fbae27cee9ad8b1987ca07e1fcaa24522f62137527e5d9392",
    "fresh_scores": "2f91833d2351250e1841a88351af35c44d08ea6d0a8efd8b62e3fa75242eb428",
    "fresh_scoring_lock": "33dabcfeff65aacb3f0ab69f7a12a13d4412ee3ec67dc1c348dc2b1ba47d12d9",
    "v4h_display_scores": "8f8249a609287477a7f094ef774dd8f8b2af4b9b9b142d58e6c22123ae15f906",
    "v4h_display_scoring_lock": "ac7a96c89eb2d4a51eda1b56756ed3b943311e7f300b661659cff51285423a8f",
    "stage_a_fresh_scores": "5a1c988e0efb6898e721f700261e834ed1994de107338dc49e68549aa9e1ca02",
    "stage_a_fresh_scoring_lock": "b64ff7d8828fcd4d8938c487d48335c0837aaca0d6439f42b91579467a5c57f9",
    "final_release_lock": "7e7d73920452ec39ea72947d2b3762a02aaa1b2bb0bcd7470c7e21dbb3a25ab0",
    "stage_b_definition_contract": "c26af8da897995b817d9eeffef56f94a2fd4cab99dca16777d85f569759a4476",
    "judging_contract": "45ceaac688a3e6f6c733c5343a38590652cbfa9fa0236db68e1b4e6835322aef",
    "execution_contract": "f93427e7734e4dfb9392800959b48bc9a22c283525afd1f35f8442e764ede1ce",
    "classification_contract": "90c0d5295989f5517fa079ec16215cbb1b44f661fe5919cc3065babd89f9dda5",
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


def load_score_frame(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    req({"query_id", "isbn13", "judge_score"} <= set(frame.columns),
        f"{label} missing required score columns")
    frame["isbn13"] = frame["isbn13"].map(canonical_isbn)
    frame["judge_score"] = pd.to_numeric(frame["judge_score"], errors="raise").astype(int)
    req(frame["judge_score"].between(0, 4).all(), f"{label} contains score outside 0..4")
    req(not frame.duplicated(["query_id", "isbn13"]).any(),
        f"{label} has duplicate query-book pair")
    return frame


def verify_boundary() -> dict[str, Any]:
    checks = [
        (STAGE_A_CLASSIFICATION_LOCK, "stage_a_classification_lock", "Stage-A classification lock"),
        (CANDIDATES, "candidates", "Stage-B candidate inventory"),
        (QUERY_INVENTORY, "query_inventory", "Stage-B query inventory"),
        (CANDIDATE_LOCK, "candidate_lock", "Stage-B candidate lock"),
        (SCORE_INVENTORY, "score_inventory", "Stage-B score inventory"),
        (JUDGING_INPUT, "judging_input", "Stage-B fresh judging input"),
        (JUDGING_INPUT_LOCK, "judging_input_lock", "Stage-B judging-input lock"),
        (FRESH_SCORES, "fresh_scores", "Stage-B fresh judge scores"),
        (FRESH_SCORING_LOCK, "fresh_scoring_lock", "Stage-B fresh scoring lock"),
        (V4H_DISPLAY_SCORES, "v4h_display_scores", "frozen V4H displayed-union scores"),
        (V4H_DISPLAY_SCORING_LOCK, "v4h_display_scoring_lock", "frozen V4H displayed-union scoring lock"),
        (STAGE_A_FRESH_SCORES, "stage_a_fresh_scores", "frozen Stage-A fresh scores"),
        (STAGE_A_FRESH_SCORING_LOCK, "stage_a_fresh_scoring_lock", "frozen Stage-A fresh scoring lock"),
        (FINAL_RELEASE_LOCK, "final_release_lock", "V4H final release-analysis lock"),
        (STAGE_B_DEFINITION_CONTRACT, "stage_b_definition_contract", "Stage-B definition contract"),
        (JUDGING_CONTRACT, "judging_contract", "Stage-B judging-preparation contract"),
        (EXECUTION_CONTRACT, "execution_contract", "Stage-B execution contract"),
        (CLASSIFICATION_CONTRACT, "classification_contract", "Stage-B classification contract"),
    ]
    for path, key, label in checks:
        verify_raw(path, EXPECTED[key], label)

    stage_a_lock = load_json(STAGE_A_CLASSIFICATION_LOCK)
    req(stage_a_lock.get("status") == "V4H_HEADROOM_STAGE_A_CLASSIFICATION_FROZEN",
        "Unexpected Stage-A classification lock status")
    req(stage_a_lock.get("frozen_release_decision") == "FAIL",
        "Frozen V4B release decision changed")
    req(stage_a_lock.get("release_decision_modified") is False,
        "Stage-A classification modified release decision")
    req(stage_a_lock.get("classification_counts") == {
        "HEADROOM_FOUND_11_50": 4,
        "NO_CLEAR_THROUGH_50": 6,
    }, f"Unexpected Stage-A classification counts: {stage_a_lock.get('classification_counts')}")

    q_sha = stage_a_lock.get("outputs", {}).get(STAGE_A_QUERY_CLASSIFICATION.name)
    req(isinstance(q_sha, str) and len(q_sha) == 64,
        "Stage-A classification lock does not bind query classification")
    verify_raw(STAGE_A_QUERY_CLASSIFICATION, q_sha, "Stage-A query classification")
    stage_a_q = pd.read_csv(STAGE_A_QUERY_CLASSIFICATION, dtype=str,
                            keep_default_na=False, encoding="utf-8")
    req(len(stage_a_q) == 10, "Expected 10 Stage-A query classifications")
    req(not stage_a_q["query_id"].duplicated().any(), "Duplicate Stage-A query classification")
    unresolved = stage_a_q.loc[
        stage_a_q["classification"] == "NO_CLEAR_THROUGH_50", "query_id"
    ].tolist()
    req(unresolved == TARGET_QUERY_IDS,
        f"Stage-B target set changed: expected={TARGET_QUERY_IDS} actual={unresolved}")

    candidate_lock = load_json(CANDIDATE_LOCK)
    req(candidate_lock.get("status") == "V4H_NO_CLEAR_HEADROOM_STAGE_B_CANDIDATES_FROZEN_BEFORE_JUDGING",
        "Unexpected Stage-B candidate lock status")
    req(candidate_lock.get("target_query_ids") == TARGET_QUERY_IDS,
        "Stage-B candidate-lock target set changed")
    req(int(candidate_lock.get("candidate_count", -1)) == 252,
        "Stage-B candidate count changed")
    req(candidate_lock.get("rank_window") == "51-end", "Stage-B rank window changed")
    req(candidate_lock.get("release_decision_modified") is False,
        "Stage-B candidate lock says release decision changed")

    qinv = pd.read_csv(QUERY_INVENTORY, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(qinv) == 6, "Expected six Stage-B query-inventory rows")
    req(qinv["query_id"].tolist() == TARGET_QUERY_IDS, "Stage-B query inventory order/set changed")
    qinv["stage_b_candidate_count"] = pd.to_numeric(
        qinv["stage_b_candidate_count"], errors="raise"
    ).astype(int)
    req(int(qinv["stage_b_candidate_count"].sum()) == 252,
        "Stage-B per-query candidate counts do not total 252")

    inventory = pd.read_csv(SCORE_INVENTORY, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"case_id", "query_id", "stratum", "v4b_rank", "isbn13", "score_source", "reused_judge_score"}
    req(required <= set(inventory.columns),
        f"Stage-B score inventory missing columns: {sorted(required-set(inventory.columns))}")
    req(len(inventory) == 252, "Expected exactly 252 Stage-B inventory rows")
    inventory["isbn13"] = inventory["isbn13"].map(canonical_isbn)
    inventory["v4b_rank"] = pd.to_numeric(inventory["v4b_rank"], errors="raise").astype(int)
    req(inventory["v4b_rank"].ge(51).all(), "Stage-B inventory contains rank below 51")
    req(not inventory["case_id"].duplicated().any(), "Duplicate Stage-B case_id")
    req(not inventory.duplicated(["query_id", "isbn13"]).any(), "Duplicate Stage-B query-book pair")
    req(inventory["query_id"].drop_duplicates().tolist() == TARGET_QUERY_IDS,
        "Stage-B inventory target query order/set changed")
    actual_counts = inventory.groupby("query_id", sort=False).size().to_dict()
    expected_counts = dict(zip(qinv["query_id"], qinv["stage_b_candidate_count"]))
    req(actual_counts == expected_counts,
        f"Stage-B inventory per-query counts changed: expected={expected_counts} actual={actual_counts}")
    source_counts = inventory["score_source"].value_counts().to_dict()
    req(source_counts == {
        "fresh_judge_required": 251,
        "reused_frozen_v4h_display_score": 1,
    }, f"Stage-B score-source partition changed: {source_counts}")

    fresh_lock = load_json(FRESH_SCORING_LOCK)
    req(fresh_lock.get("status") == "V4H_HEADROOM_STAGE_B_FRESH_JUDGE_SCORES_FROZEN",
        "Unexpected Stage-B fresh-score lock status")
    req(int(fresh_lock.get("fresh_pair_count", -1)) == 251, "Fresh Stage-B pair count changed")
    req(int(fresh_lock.get("reused_exact_pair_count_external", -1)) == 1,
        "Stage-B reused pair count changed")
    req(int(fresh_lock.get("total_stage_b_pair_count", -1)) == 252,
        "Stage-B total pair count changed")
    req(fresh_lock.get("fresh_judge_scores_sha256") == EXPECTED["fresh_scores"],
        "Stage-B fresh lock does not bind expected score artifact")
    req(fresh_lock.get("human_relevance_labels_used") is False,
        "Stage-B fresh lock says human labels were used")
    req(fresh_lock.get("release_metrics_calculated") is False,
        "Stage-B fresh lock says release metrics were calculated")
    req(fresh_lock.get("release_decision_modified") is False,
        "Stage-B fresh lock says release decision changed")

    final = load_json(FINAL_RELEASE_LOCK)
    req(final.get("status") == "V4H_FINAL_RELEASE_ANALYSIS_FROZEN",
        "Unexpected V4H final release-analysis lock status")
    req(final.get("v4b_final_release_decision") == "FAIL",
        "V4B final release decision changed")

    fresh = load_score_frame(FRESH_SCORES, "Stage-B fresh scores")
    req(len(fresh) == 251, "Expected exactly 251 Stage-B fresh scores")
    req({"case_id", "v4b_rank", "match_level"} <= set(fresh.columns),
        "Stage-B fresh score artifact missing diagnostic columns")
    fresh["v4b_rank"] = pd.to_numeric(fresh["v4b_rank"], errors="raise").astype(int)
    req(fresh["v4b_rank"].ge(51).all(), "Stage-B fresh scores contain rank below 51")
    req(not fresh["case_id"].duplicated().any(), "Duplicate Stage-B fresh case_id")

    fresh_inventory = inventory[inventory["score_source"] == "fresh_judge_required"].copy()
    req(set(fresh["case_id"]) == set(fresh_inventory["case_id"]),
        "Stage-B fresh score case set differs from frozen fresh inventory")
    inv_by_case = fresh_inventory.set_index("case_id")
    for row in fresh.itertuples(index=False):
        src = inv_by_case.loc[str(row.case_id)]
        req(str(row.query_id) == str(src["query_id"]), f"{row.case_id} query mismatch")
        req(canonical_isbn(row.isbn13) == canonical_isbn(src["isbn13"]),
            f"{row.case_id} ISBN mismatch")
        req(int(row.v4b_rank) == int(src["v4b_rank"]), f"{row.case_id} rank mismatch")

    display = load_score_frame(V4H_DISPLAY_SCORES, "V4H displayed-union scores")
    stage_a_fresh = load_score_frame(STAGE_A_FRESH_SCORES, "Stage-A fresh scores")
    display_lookup = display.set_index(["query_id", "isbn13"])["judge_score"].to_dict()
    stage_a_lookup = stage_a_fresh.set_index(["query_id", "isbn13"])["judge_score"].to_dict()

    reused = inventory[inventory["score_source"] != "fresh_judge_required"].copy()
    req(len(reused) == 1, "Expected exactly one Stage-B reused score")
    reused_scores = pd.to_numeric(reused["reused_judge_score"], errors="raise").astype(int)
    req(reused_scores.between(0, 4).all(), "Reused Stage-B score outside 0..4")
    for row in reused.itertuples(index=False):
        key = (str(row.query_id), canonical_isbn(row.isbn13))
        if row.score_source == "reused_frozen_v4h_display_score":
            req(key in display_lookup, f"Reused displayed pair missing from frozen V4H scores: {key}")
            req(int(row.reused_judge_score) == int(display_lookup[key]),
                f"Reused displayed score mismatch for {key}")
        elif row.score_source == "reused_frozen_stage_a_fresh_score":
            req(key in stage_a_lookup, f"Reused Stage-A pair missing from frozen Stage-A scores: {key}")
            req(int(row.reused_judge_score) == int(stage_a_lookup[key]),
                f"Reused Stage-A score mismatch for {key}")
        else:
            raise ValueError(f"Unexpected Stage-B reused score source: {row.score_source}")

    return {
        "inventory": inventory,
        "query_inventory": qinv,
        "fresh": fresh,
        "stage_a_query_classification": stage_a_q,
        "reused": reused,
    }


def main() -> None:
    print("Semantic relevance V4H - headroom Stage-B classification preflight")
    print("-----------------------------------------------------------------")
    print("Role: consumed-holdout diagnostic only")
    print("Judge/model/retrieval/reranker calls: DISABLED")
    print("Release metrics/gates: NOT CALCULATED")
    print("Frozen V4B release decision modification: DISABLED")
    print()
    verify_boundary()
    req(not CLASSIFICATION_LOCK.exists(), "Stage-B classification lock already exists")
    for path in [COMPLETE_SCORES, QUERY_CLASSIFICATION, OVERALL_DISPOSITION, SUMMARY, REPORT, METADATA]:
        req(not path.exists(), f"Refusing to overwrite existing Stage-B classification output: {path}")
    print("PASS  Stage-B 252-pair score inventory verified: 1 exact reused + 251 fresh")
    print("PASS  all 251 fresh Stage-B scores frozen and bound by final scoring lock")
    print("PASS  exact reused score independently agrees with its frozen source")
    print("PASS  Stage-A unresolved target set preserved: 6 queries")
    print("PASS  classification rule remains score >=3 in ranks 51-end")
    print()
    print("V4H HEADROOM STAGE-B CLASSIFICATION PREFLIGHT: PASS")
    print("Judge/model/retrieval/reranker calls performed: 0")
    print("Release metrics calculated: NO")
    print("Frozen V4B release decision modified: NO")
    print("Next: combine 1 exact reused + 251 fresh scores and classify Stage-B headroom.")


if __name__ == "__main__":
    main()
