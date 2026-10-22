from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
HOLDOUT_RUN = EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"

NOVEL_INPUT = RUN / "v4a_retrieval_diagnostic_novel_bm25_pairs_to_judge.csv"
SCORE_INVENTORY = RUN / "v4a_retrieval_diagnostic_score_inventory.csv"
JUDGING_INPUT_LOCK = RUN / "v4a_retrieval_diagnostic_input.lock.json"
PARTIAL_SCORES = RUN / "v4a_retrieval_diagnostic_novel_bm25_judge_scores.csv"
QUARANTINE = RUN / "v4a_retrieval_diagnostic_quarantined_cases.json"
JUDGING_METADATA = RUN / "v4a_retrieval_diagnostic_novel_judging_metadata.json"
SCORING_LOCK = RUN / "v4a_retrieval_diagnostic_novel_judge_scoring.lock.json"
FREEZE_LOCK = RUN / "v4a_retrieval_diagnostic_partial_judging.freeze.lock.json"

HOLDOUT_SCORING_LOCK = HOLDOUT_RUN / "holdout_judge_scoring_lock.json"
SPAN_PATCH = EVALS / "system_evaluation/semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"
TIMEOUT_PATCH = REPO_ROOT / "v4a_retrieval_diagnostic_deepeval_timeout_patch1.json"
QUARANTINE_PATCH = REPO_ROOT / "v4a_retrieval_diagnostic_case_quarantine_patch1.json"

EXPECTED = {
    "novel_input": "d9607305d0e0bd00d268bec34acdf2a4af0e23e22c27756ca27460fd97ff3c74",
    "score_inventory": "250841f0bf9cefc5a461e09f3963842187bdac8bfdaafe03e7e07487d07c7e16",
    "judging_input_lock": "da34d259dacd70348878c2c1e61fa9d702dd536b58df88ffc129a3fb4e6be4e2",
    "holdout_scoring_lock": "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777",
    "span_patch": "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266",
    "timeout_patch": "b250e7448cb0ca53573dc088cb1e3f3c4a2c7372ee1848627483e53f5a7c8a40",
    "quarantine_patch": "eabc05dd299cefd8c605d2db98427aecdf5bdb6013bda5a89fdac9be3b59d56c",
}

EXPECTED_NOVEL = 675
EXPECTED_PERSISTED = 674
KNOWN_CASE = "V4A_RETRDIAG_H14_9781582972947"
KNOWN_QID = "H14"
KNOWN_ISBN = "9781582972947"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_frozen(path: Path, expected: str, label: str) -> None:
    require(path.exists(), f"Missing frozen input: {path}")
    actual = sha256(path)
    require(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} SHA-256 verified")


def main() -> None:
    print("Semantic relevance v4a - freeze partial/quarantined retrieval-diagnostic judging state")
    print("---------------------------------------------------------------------------------")
    print("Judge execution: DISABLED")
    print("Model execution: DISABLED")
    print("Semantic score assignment: DISABLED")
    print("Diagnostic analysis: DISABLED")
    print("Development selection: FROZEN / NOT REOPENED")
    print("Release analysis: DISABLED")
    print()

    for path, key, label in [
        (NOVEL_INPUT, "novel_input", "novel BM25 judging input"),
        (SCORE_INVENTORY, "score_inventory", "diagnostic score inventory"),
        (JUDGING_INPUT_LOCK, "judging_input_lock", "diagnostic judging-input lock"),
        (HOLDOUT_SCORING_LOCK, "holdout_scoring_lock", "v3 holdout scoring lock"),
        (SPAN_PATCH, "span_patch", "span-echo execution patch"),
        (TIMEOUT_PATCH, "timeout_patch", "DeepEval timeout execution patch"),
        (QUARANTINE_PATCH, "quarantine_patch", "case quarantine execution patch"),
    ]:
        verify_frozen(path, EXPECTED[key], label)

    require(PARTIAL_SCORES.exists(), f"Missing partial scores: {PARTIAL_SCORES}")
    require(QUARANTINE.exists(), f"Missing quarantine record: {QUARANTINE}")
    require(JUDGING_METADATA.exists(), f"Missing judging metadata: {JUDGING_METADATA}")
    require(not SCORING_LOCK.exists(), "A final diagnostic scoring lock already exists; partial-state freeze is no longer appropriate")
    require(not FREEZE_LOCK.exists(), f"Refusing to overwrite existing partial-state freeze lock: {FREEZE_LOCK}")

    novel = pd.read_csv(NOVEL_INPUT, dtype=str, keep_default_na=False, encoding="utf-8")
    require(len(novel) == EXPECTED_NOVEL, f"Expected {EXPECTED_NOVEL} frozen novel cases, got {len(novel)}")
    require(not novel["case_id"].duplicated().any(), "Duplicate frozen novel case_id")
    require(not novel.duplicated(["query_id", "isbn13"]).any(), "Duplicate frozen novel query/book pair")

    scores = pd.read_csv(PARTIAL_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    required_score_cols = {"case_id", "query_id", "isbn13", "judge_score"}
    missing = required_score_cols - set(scores.columns)
    require(not missing, f"Partial score file missing columns: {sorted(missing)}")
    require(len(scores) == EXPECTED_PERSISTED, f"Expected {EXPECTED_PERSISTED} persisted scores, got {len(scores)}")
    require(not scores["case_id"].duplicated().any(), "Duplicate persisted case_id")
    require(not scores.duplicated(["query_id", "isbn13"]).any(), "Duplicate persisted query/book pair")
    vals = pd.to_numeric(scores["judge_score"], errors="raise").astype(int)
    require(vals.between(0, 4).all(), "Persisted judge score outside 0..4")

    frozen_ids = set(novel["case_id"].astype(str))
    score_ids = set(scores["case_id"].astype(str))
    require(score_ids <= frozen_ids, "Persisted score exists outside frozen novel input")
    require(KNOWN_CASE not in score_ids, "Known quarantined case unexpectedly has a persisted score")

    quarantine = load_json(QUARANTINE)
    require(isinstance(quarantine, list), "Quarantine artifact must be a JSON array")
    require(len(quarantine) == 1, f"Expected exactly one quarantined case, got {len(quarantine)}")
    q = quarantine[0]
    require(str(q.get("case_id")) == KNOWN_CASE, "Unexpected quarantined case_id")
    require(str(q.get("query_id")) == KNOWN_QID, "Unexpected quarantined query_id")
    require(str(q.get("isbn13")) == KNOWN_ISBN, "Unexpected quarantined ISBN")
    require(q.get("status") == "QUARANTINED_UNSCORED", "Quarantined case status changed")
    require(str(q.get("patch_id")) == "v4a-retrdiag-case-timeout-quarantine1", "Quarantine patch id changed")
    quarantine_ids = {str(q["case_id"])}
    require(not (score_ids & quarantine_ids), "A quarantined case also has a score")
    require(score_ids | quarantine_ids == frozen_ids, "Persisted + quarantined cases do not exactly cover frozen 675-case input")

    metadata = load_json(JUDGING_METADATA)
    require(metadata.get("status") == "PARTIAL_QUARANTINED_UNSCORED", f"Unexpected metadata status: {metadata.get('status')}")
    require(int(metadata.get("novel_pairs", -1)) == EXPECTED_NOVEL, "Metadata novel pair count changed")
    require(int(metadata.get("completed_novel_pairs", -1)) == EXPECTED_PERSISTED, "Metadata persisted count changed")
    require(int(metadata.get("quarantined_unscored_pairs", -1)) == 1, "Metadata quarantine count changed")
    require(metadata.get("human_relevance_labels_used") is False, "Metadata says human labels were used")
    require(metadata.get("development_selection_reopened") is False, "Metadata says development selection was reopened")
    require(metadata.get("release_metrics_calculated") is False, "Metadata says release metrics were calculated")
    require(metadata.get("release_gates_calculated") is False, "Metadata says release gates were calculated")
    require(metadata.get("release_decision_produced") is False, "Metadata says release decision was produced")

    payload = {
        "status": "V4A_RETRIEVAL_DIAGNOSTIC_PARTIAL_JUDGING_STATE_FROZEN_BEFORE_SENSITIVITY_ANALYSIS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_role": "explanatory_only",
        "frozen_novel_pairs": EXPECTED_NOVEL,
        "persisted_scored_pairs": EXPECTED_PERSISTED,
        "quarantined_unscored_pairs": 1,
        "quarantined_case_id": KNOWN_CASE,
        "quarantined_query_id": KNOWN_QID,
        "quarantined_isbn13": KNOWN_ISBN,
        "quarantined_score_assigned": False,
        "quarantined_case_required_for_analysis": True,
        "frozen_inputs": {
            "novel_bm25_judging_input_sha256": EXPECTED["novel_input"],
            "score_inventory_sha256": EXPECTED["score_inventory"],
            "judging_input_lock_sha256": EXPECTED["judging_input_lock"],
            "holdout_judge_scoring_lock_sha256": EXPECTED["holdout_scoring_lock"],
            "span_echo_patch_sha256": EXPECTED["span_patch"],
            "timeout_patch_sha256": EXPECTED["timeout_patch"],
            "quarantine_patch_sha256": EXPECTED["quarantine_patch"],
        },
        "partial_outputs": {
            "v4a_retrieval_diagnostic_novel_bm25_judge_scores.csv": sha256(PARTIAL_SCORES),
            "v4a_retrieval_diagnostic_quarantined_cases.json": sha256(QUARANTINE),
            "v4a_retrieval_diagnostic_novel_judging_metadata.json": sha256(JUDGING_METADATA),
        },
        "authority_boundary": {
            "judge_calls_performed_during_freeze": 0,
            "model_calls_performed_during_freeze": 0,
            "semantic_score_assigned_to_quarantine": False,
            "human_relevance_labels_used": False,
            "development_selection_reopened": False,
            "release_metrics_calculated": False,
            "release_gates_calculated": False,
            "release_decision_produced": False,
        },
        "next_stage": "PRE_REGISTER_QUARANTINE_SCORE_SENSITIVITY_ANALYSIS_OVER_ALL_ALLOWED_SCORES_0_TO_4",
    }
    FREEZE_LOCK.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("PASS  frozen 675-case coverage verified: 674 persisted scores + 1 explicitly quarantined/unscored")
    print(f"PASS  known quarantine remains unscored and required: {KNOWN_CASE}")
    print("PASS  no final scoring lock exists")
    print("PASS  authority boundary verified: selection/release remain untouched")
    print()
    print("V4A RETRIEVAL DIAGNOSTIC PARTIAL JUDGING STATE FREEZE: PASS")
    print(f"Partial score SHA-256: {sha256(PARTIAL_SCORES)}")
    print(f"Quarantine SHA-256:    {sha256(QUARANTINE)}")
    print(f"Metadata SHA-256:      {sha256(JUDGING_METADATA)}")
    print(f"Freeze lock SHA-256:   {sha256(FREEZE_LOCK)}")
    print("Judge calls during freeze: 0")
    print("Model calls during freeze: 0")
    print("Quarantined score assigned: NO")
    print("Development selection reopened: NO")
    print("Release analysis: NO")
    print("Next: pre-register deterministic sensitivity analysis for quarantine score in {0,1,2,3,4} before calculating diagnostic results.")


if __name__ == "__main__":
    main()
