from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
SYSTEM = EVALS / "system_evaluation"
RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
V3_POSTHOC_RUN = EVALS / "runs/semantic_relevance_v3_posthoc_diagnostic/v2_deep_candidate_prepare"
V3_CLASS_RUN = EVALS / "runs/semantic_relevance_v3_posthoc_diagnostic/v3_classification"

SPEC = SYSTEM / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json"
HYBRID_UNION = RUN / "v4a_dev_hybrid_union.csv"
INVENTORY = RUN / "v4a_retrieval_diagnostic_score_inventory.csv"
INPUT_LOCK = RUN / "v4a_retrieval_diagnostic_input.lock.json"
PARTIAL_SCORES = RUN / "v4a_retrieval_diagnostic_novel_bm25_judge_scores.csv"
QUARANTINE = RUN / "v4a_retrieval_diagnostic_quarantined_cases.json"
JUDGING_METADATA = RUN / "v4a_retrieval_diagnostic_novel_judging_metadata.json"
PARTIAL_FREEZE_LOCK = RUN / "v4a_retrieval_diagnostic_partial_judging.freeze.lock.json"
FINAL_SCORING_LOCK = RUN / "v4a_retrieval_diagnostic_novel_judge_scoring.lock.json"
SENSITIVITY_CONTRACT = SYSTEM / "semantic_relevance_v4a_retrieval_diagnostic_sensitivity_contract.v0.1.0.json"
SENSITIVITY_INPUT_LOCK = RUN / "v4a_retrieval_diagnostic_sensitivity_input.lock.json"

V3_COMPLETE_MAP = V3_POSTHOC_RUN / "target_top50_complete_score_map.csv"
V3_CLASSIFICATION_LOCK = V3_CLASS_RUN / "classification_lock.json"
V3_CLASSIFICATION = V3_CLASS_RUN / "query_retrieval_reranker_classification.csv"
V4A_COMPARISON_LOCK = RUN / "v4a_dev_comparison_selection.lock.json"

EXPECTED = {
    "spec": "6aba9c8354f201ad41044dd82d4476612f81ebd27a5ce4f8bf4f327a8b3281b9",
    "hybrid_union": "71be13c297f6449e6a53c54c7b9db590f6f033d69a11f16b21377797d9390155",
    "inventory": "250841f0bf9cefc5a461e09f3963842187bdac8bfdaafe03e7e07487d07c7e16",
    "input_lock": "da34d259dacd70348878c2c1e61fa9d702dd536b58df88ffc129a3fb4e6be4e2",
    "partial_scores": "ceb643cbff823007beb1f698b9fd42a78af70ead73e1c8bc90de94d14764be47",
    "quarantine": "ce16b3da279d4435d3b2eea7cf1aa190e6e804ecb6e275e8c12001703cbb0a38",
    "judging_metadata": "0a065f0dd163865776550420527822601162ddb321f981ca999fcce11932032c",
    "partial_freeze_lock": "ae22f150f5f64f228f7970010103b1229c7b4c100f348f55faec172034319e55",
    "v3_complete_map": "723fa27ecb8947334d25b3b7423a6597da39d594eb7ec2f979056b1c3cc516e5",
    "v3_class_lock": "cbaaaa1795773e46867ee5c135822b4b559c7805eee8e4264609dd4c9a913385",
    "v4a_comparison_lock": "c9a33feabf061e9bec373cbf5a5bdfb8b83ac03fd700f9fa55ffdfaa77e5056f",
    "contract": "96b0a1a8a4365b8cdd75018235964dbd46f018cd8dfd086e06fbb8d8b02c9f90",
}

KNOWN_CASE = "V4A_RETRDIAG_H14_9781582972947"
KNOWN_QID = "H14"
KNOWN_ISBN = "9781582972947"
ALLOWED_SCORES = [0, 1, 2, 3, 4]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_lf_sha256(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_raw(path: Path, expected: str, label: str) -> None:
    require(path.exists(), f"Missing frozen input: {path}")
    actual = sha256(path)
    require(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")


def verify_text(path: Path, expected: str, label: str) -> None:
    require(path.exists(), f"Missing frozen input: {path}")
    raw = sha256(path)
    if raw == expected:
        mode = "raw"
    else:
        norm = canonical_lf_sha256(path)
        require(norm == expected, f"{label} hash mismatch: expected={expected} raw={raw} canonical_lf={norm}")
        mode = "canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")


def main() -> None:
    print("Semantic relevance v4a - pre-register quarantine sensitivity analysis")
    print("------------------------------------------------------------------")
    print("Judge execution: DISABLED")
    print("Model execution: DISABLED")
    print("Human labels: DISABLED")
    print("Diagnostic calculations: DISABLED")
    print("Quarantine score assignment: DISABLED")
    print("Development selection: FROZEN / NOT REOPENED")
    print("Release analysis: DISABLED")
    print()

    require(not SENSITIVITY_INPUT_LOCK.exists(), f"Refusing to overwrite existing sensitivity input lock: {SENSITIVITY_INPUT_LOCK}")
    require(not FINAL_SCORING_LOCK.exists(), "Final diagnostic scoring lock unexpectedly exists")

    verify_text(SPEC, EXPECTED["spec"], "v4a frozen spec")
    for path, key, label in [
        (HYBRID_UNION, "hybrid_union", "v4a hybrid union"),
        (INVENTORY, "inventory", "diagnostic score inventory"),
        (INPUT_LOCK, "input_lock", "diagnostic input lock"),
        (PARTIAL_SCORES, "partial_scores", "674-case partial score artifact"),
        (QUARANTINE, "quarantine", "single-case quarantine artifact"),
        (JUDGING_METADATA, "judging_metadata", "partial judging metadata"),
        (PARTIAL_FREEZE_LOCK, "partial_freeze_lock", "partial judging freeze lock"),
        (V3_COMPLETE_MAP, "v3_complete_map", "v3 posthoc complete top50 map"),
        (V3_CLASSIFICATION_LOCK, "v3_class_lock", "v3 Stage-C classification lock"),
        (V4A_COMPARISON_LOCK, "v4a_comparison_lock", "v4a comparison/selection lock"),
        (SENSITIVITY_CONTRACT, "contract", "sensitivity contract"),
    ]:
        verify_raw(path, EXPECTED[key], label)

    freeze = load_json(PARTIAL_FREEZE_LOCK)
    require(
        freeze.get("status") == "V4A_RETRIEVAL_DIAGNOSTIC_PARTIAL_JUDGING_STATE_FROZEN_BEFORE_SENSITIVITY_ANALYSIS",
        "Unexpected partial freeze status",
    )
    require(int(freeze.get("persisted_scored_pairs", -1)) == 674, "Partial freeze persisted count changed")
    require(int(freeze.get("quarantined_unscored_pairs", -1)) == 1, "Partial freeze quarantine count changed")
    require(freeze.get("quarantined_case_id") == KNOWN_CASE, "Partial freeze case changed")
    require(freeze.get("quarantined_score_assigned") is False, "Partial freeze says quarantine was scored")

    scores = pd.read_csv(PARTIAL_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    require(len(scores) == 674, f"Expected 674 partial scores, got {len(scores)}")
    require(KNOWN_CASE not in set(scores["case_id"].astype(str)), "Quarantined case unexpectedly has a persisted score")

    quarantine = load_json(QUARANTINE)
    require(isinstance(quarantine, list) and len(quarantine) == 1, "Expected exactly one quarantine record")
    q = quarantine[0]
    require(q.get("case_id") == KNOWN_CASE, "Quarantine case changed")
    require(q.get("query_id") == KNOWN_QID, "Quarantine query changed")
    require(str(q.get("isbn13")) == KNOWN_ISBN, "Quarantine ISBN changed")
    require(q.get("status") == "QUARANTINED_UNSCORED", "Quarantine status changed")

    inventory = pd.read_csv(INVENTORY, dtype=str, keep_default_na=False, encoding="utf-8")
    require(len(inventory) == 1576, f"Expected 1576 diagnostic union pairs, got {len(inventory)}")
    require(inventory["query_id"].nunique() == 17, "Expected 17 diagnostic queries")
    pair = inventory[(inventory["query_id"] == KNOWN_QID) & (inventory["isbn13"].astype(str) == KNOWN_ISBN)]
    require(len(pair) == 1, "Quarantined pair not uniquely present in diagnostic inventory")
    require(pair.iloc[0]["candidate_source"] == "bm25_only", "Quarantined pair is no longer BM25-only")

    class_lock = load_json(V3_CLASSIFICATION_LOCK)
    require(class_lock.get("status") == "V3_POSTHOC_RETRIEVAL_VS_RERANKER_CLASSIFICATION_FROZEN", "Unexpected Stage-C classification status")
    expected_class_hash = class_lock.get("outputs", {}).get("query_retrieval_reranker_classification.csv")
    require(expected_class_hash, "Stage-C lock missing classification CSV hash")
    require(V3_CLASSIFICATION.exists(), f"Missing Stage-C classification CSV: {V3_CLASSIFICATION}")
    require(sha256(V3_CLASSIFICATION) == expected_class_hash, "Stage-C classification CSV changed")

    comparison = load_json(V4A_COMPARISON_LOCK)
    require(comparison.get("status") == "V4A_DEVELOPMENT_COMPARISON_AND_SELECTION_FROZEN", "Unexpected v4a comparison lock status")

    contract = load_json(SENSITIVITY_CONTRACT)
    require(contract.get("status") == "FROZEN_BEFORE_SENSITIVITY_ANALYSIS", "Sensitivity contract status changed")
    require(contract["frozen_case"]["case_id"] == KNOWN_CASE, "Sensitivity contract case changed")
    require(contract["frozen_case"]["must_remain_unscored"] is True, "Contract permits quarantine score assignment")
    require(contract["sensitivity_scenarios"]["allowed_scores"] == ALLOWED_SCORES, "Sensitivity score set changed")
    require(contract["authority_boundary"]["may_call_judge_or_model"] is False, "Contract permits model execution")
    require(contract["authority_boundary"]["may_assign_actual_score_to_quarantine"] is False, "Contract permits score imputation")
    require(contract["authority_boundary"]["may_reopen_v4a_development_selection"] is False, "Contract permits selection reopening")
    require(contract["authority_boundary"]["may_calculate_release_metrics_or_gates"] is False, "Contract permits release analysis")

    payload = {
        "status": "V4A_RETRIEVAL_DIAGNOSTIC_SENSITIVITY_CONTRACT_FROZEN_BEFORE_ANALYSIS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": EXPECTED["contract"],
        "partial_judging_freeze_lock_sha256": EXPECTED["partial_freeze_lock"],
        "diagnostic_score_inventory_sha256": EXPECTED["inventory"],
        "partial_novel_score_sha256": EXPECTED["partial_scores"],
        "quarantine_sha256": EXPECTED["quarantine"],
        "hybrid_union_sha256": EXPECTED["hybrid_union"],
        "v3_stage_c_classification_lock_sha256": EXPECTED["v3_class_lock"],
        "v4a_comparison_selection_lock_sha256": EXPECTED["v4a_comparison_lock"],
        "quarantined_case_id": KNOWN_CASE,
        "quarantined_actual_score_assigned": False,
        "hypothetical_scores_to_analyze": ALLOWED_SCORES,
        "judge_calls_performed": 0,
        "model_calls_performed": 0,
        "human_relevance_labels_used": False,
        "diagnostic_results_calculated": False,
        "development_selection_reopened": False,
        "release_metrics_calculated": False,
        "release_gates_calculated": False,
        "release_decision_produced": False,
        "next_stage": "RUN_DETERMINISTIC_FIVE_SCENARIO_RETRIEVAL_DIAGNOSTIC_SENSITIVITY_ANALYSIS",
    }
    SENSITIVITY_INPUT_LOCK.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("PASS  1575 actual scores + 1 required quarantine boundary verified")
    print("PASS  hypothetical scenario set frozen exactly to {0,1,2,3,4}")
    print("PASS  oracle ordering and Stage-C-equivalent classification rules frozen")
    print("PASS  authority boundary verified: explanatory only; selection/release untouched")
    print()
    print("V4A RETRIEVAL DIAGNOSTIC SENSITIVITY PRE-REGISTRATION: PASS")
    print(f"Sensitivity contract SHA-256: {sha256(SENSITIVITY_CONTRACT)}")
    print(f"Sensitivity input lock SHA-256: {sha256(SENSITIVITY_INPUT_LOCK)}")
    print("Judge calls performed: 0")
    print("Model calls performed: 0")
    print("Actual quarantine score assigned: NO")
    print("Diagnostic results calculated: NO")
    print("Development selection reopened: NO")
    print("Release analysis: NO")
    print("Next: deterministic five-scenario sensitivity analysis with zero judge/model calls.")


if __name__ == "__main__":
    main()
