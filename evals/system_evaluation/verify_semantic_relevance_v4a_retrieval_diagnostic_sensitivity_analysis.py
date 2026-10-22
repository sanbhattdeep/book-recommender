from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
SYSTEM = EVALS / "system_evaluation"
RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
V3_CLASS_RUN = EVALS / "runs/semantic_relevance_v3_posthoc_diagnostic/v3_classification"

CONTRACT = SYSTEM / "semantic_relevance_v4a_retrieval_diagnostic_sensitivity_contract.v0.1.0.json"
SENSITIVITY_INPUT_LOCK = RUN / "v4a_retrieval_diagnostic_sensitivity_input.lock.json"
INVENTORY = RUN / "v4a_retrieval_diagnostic_score_inventory.csv"
PARTIAL_SCORES = RUN / "v4a_retrieval_diagnostic_novel_bm25_judge_scores.csv"
QUARANTINE = RUN / "v4a_retrieval_diagnostic_quarantined_cases.json"
PARTIAL_FREEZE_LOCK = RUN / "v4a_retrieval_diagnostic_partial_judging.freeze.lock.json"
V3_CLASS_LOCK = V3_CLASS_RUN / "classification_lock.json"
V3_CLASSIFICATION = V3_CLASS_RUN / "query_retrieval_reranker_classification.csv"
V4A_COMPARISON_LOCK = RUN / "v4a_dev_comparison_selection.lock.json"
FINAL_SCORING_LOCK = RUN / "v4a_retrieval_diagnostic_novel_judge_scoring.lock.json"

OUT_QUERY = RUN / "v4a_retrieval_diagnostic_sensitivity_query_results.csv"
OUT_SUMMARY = RUN / "v4a_retrieval_diagnostic_sensitivity_scenario_summary.csv"
OUT_INVARIANCE = RUN / "v4a_retrieval_diagnostic_sensitivity_invariance.json"
OUT_REPORT = RUN / "v4a_retrieval_diagnostic_sensitivity_report.md"
OUT_METADATA = RUN / "v4a_retrieval_diagnostic_sensitivity_metadata.json"
OUT_LOCK = RUN / "v4a_retrieval_diagnostic_sensitivity_analysis.lock.json"

EXPECTED = {
    "contract": "96b0a1a8a4365b8cdd75018235964dbd46f018cd8dfd086e06fbb8d8b02c9f90",
    "sensitivity_input_lock": "c7aafd11e4620fe0e9031b6367c1137e28efa1372d3891c3be64558e01c15b04",
    "inventory": "250841f0bf9cefc5a461e09f3963842187bdac8bfdaafe03e7e07487d07c7e16",
    "partial_scores": "ceb643cbff823007beb1f698b9fd42a78af70ead73e1c8bc90de94d14764be47",
    "quarantine": "ce16b3da279d4435d3b2eea7cf1aa190e6e804ecb6e275e8c12001703cbb0a38",
    "partial_freeze_lock": "ae22f150f5f64f228f7970010103b1229c7b4c100f348f55faec172034319e55",
    "v3_class_lock": "cbaaaa1795773e46867ee5c135822b4b559c7805eee8e4264609dd4c9a913385",
    "v4a_comparison_lock": "c9a33feabf061e9bec373cbf5a5bdfb8b83ac03fd700f9fa55ffdfaa77e5056f"
}

KNOWN_CASE = "V4A_RETRDIAG_H14_9781582972947"
KNOWN_QID = "H14"
KNOWN_ISBN = "9781582972947"
SCENARIOS = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}

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
    print("Semantic relevance v4a - retrieval-diagnostic sensitivity analysis preflight")
    print("---------------------------------------------------------------------------")
    print("Judge execution: DISABLED")
    print("Model execution: DISABLED")
    print("Human labels: DISABLED")
    print("Diagnostic calculations: DISABLED")
    print("Actual quarantine score assignment: DISABLED")
    print("Development selection: FROZEN / NOT REOPENED")
    print("Release analysis: DISABLED")
    print()

    for out in [OUT_QUERY, OUT_SUMMARY, OUT_INVARIANCE, OUT_REPORT, OUT_METADATA, OUT_LOCK]:
        require(not out.exists(), f"Refusing to proceed because analysis output already exists: {out}")
    require(not FINAL_SCORING_LOCK.exists(), "Final diagnostic judge scoring lock unexpectedly exists")

    verify_text(CONTRACT, EXPECTED["contract"], "sensitivity contract")
    for path, key, label in [
        (SENSITIVITY_INPUT_LOCK, "sensitivity_input_lock", "sensitivity input lock"),
        (INVENTORY, "inventory", "diagnostic score inventory"),
        (PARTIAL_SCORES, "partial_scores", "674-case partial score artifact"),
        (QUARANTINE, "quarantine", "single-case quarantine artifact"),
        (PARTIAL_FREEZE_LOCK, "partial_freeze_lock", "partial judging freeze lock"),
        (V3_CLASS_LOCK, "v3_class_lock", "v3 Stage-C classification lock"),
        (V4A_COMPARISON_LOCK, "v4a_comparison_lock", "v4a comparison/selection lock"),
    ]:
        verify_raw(path, EXPECTED[key], label)

    contract = load_json(CONTRACT)
    require(contract.get("status") == "FROZEN_BEFORE_SENSITIVITY_ANALYSIS", "Unexpected sensitivity contract status")
    require(contract["sensitivity_scenarios"]["mapping"] == SCENARIOS, "Scenario mapping changed")
    require(contract["frozen_case"]["case_id"] == KNOWN_CASE, "Frozen sensitivity case changed")
    require(contract["frozen_case"]["must_remain_unscored"] is True, "Contract no longer requires quarantine to remain unscored")
    require(contract["authority_boundary"]["may_call_judge_or_model"] is False, "Contract allows judge/model calls")
    require(contract["authority_boundary"]["may_reopen_v4a_development_selection"] is False, "Contract allows selection reopening")
    require(contract["authority_boundary"]["may_calculate_release_metrics_or_gates"] is False, "Contract allows release analysis")

    silock = load_json(SENSITIVITY_INPUT_LOCK)
    require(silock.get("status") == "V4A_RETRIEVAL_DIAGNOSTIC_SENSITIVITY_CONTRACT_FROZEN_BEFORE_ANALYSIS", "Unexpected sensitivity input lock status")
    require(silock.get("contract_sha256") == EXPECTED["contract"], "Sensitivity input lock contract hash changed")
    require(silock.get("quarantined_case_id") == KNOWN_CASE, "Sensitivity input lock case changed")
    require(silock.get("quarantined_actual_score_assigned") is False, "Sensitivity input lock says quarantine score was assigned")
    require(silock.get("hypothetical_scores_to_analyze") == [0,1,2,3,4], "Sensitivity input lock scenario set changed")
    require(silock.get("diagnostic_results_calculated") is False, "Sensitivity input lock says results were already calculated")
    require(silock.get("development_selection_reopened") is False, "Sensitivity input lock says selection reopened")
    require(silock.get("release_metrics_calculated") is False, "Sensitivity input lock says release metrics were calculated")

    freeze = load_json(PARTIAL_FREEZE_LOCK)
    require(int(freeze.get("persisted_scored_pairs", -1)) == 674, "Partial freeze scored count changed")
    require(int(freeze.get("quarantined_unscored_pairs", -1)) == 1, "Partial freeze quarantine count changed")
    require(freeze.get("quarantined_case_id") == KNOWN_CASE, "Partial freeze quarantine case changed")
    require(freeze.get("quarantined_score_assigned") is False, "Partial freeze says quarantine score was assigned")

    inventory = pd.read_csv(INVENTORY, dtype=str, keep_default_na=False, encoding="utf-8")
    require(len(inventory) == 1576, f"Expected 1576 diagnostic union pairs, got {len(inventory)}")
    require(inventory["query_id"].nunique() == 17, "Expected 17 diagnostic queries")
    require(not inventory.duplicated(["query_id", "isbn13"]).any(), "Duplicate pair in diagnostic inventory")
    required_inv = {
        "query_id","query","stratum","isbn13","candidate_source","dense_rank","bm25_rank",
        "hybrid_retrieval_rank","score_status","judge_score"
    }
    require(not (required_inv - set(inventory.columns)), f"Inventory missing columns: {sorted(required_inv - set(inventory.columns))}")

    partial = pd.read_csv(PARTIAL_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    require(len(partial) == 674, f"Expected 674 persisted novel scores, got {len(partial)}")
    require(not partial.duplicated(["query_id", "isbn13"]).any(), "Duplicate pair in partial novel scores")
    require(KNOWN_CASE not in set(partial["case_id"].astype(str)), "Quarantined case has a persisted score")
    vals = pd.to_numeric(partial["judge_score"], errors="raise").astype(int)
    require(vals.between(0,4).all(), "Partial score outside 0..4")

    q = load_json(QUARANTINE)
    require(isinstance(q, list) and len(q) == 1, "Expected exactly one quarantine row")
    require(q[0].get("case_id") == KNOWN_CASE, "Quarantine case changed")
    require(q[0].get("status") == "QUARANTINED_UNSCORED", "Quarantine status changed")

    inv_reused = inventory["score_status"].eq("reused")
    inv_novel = inventory["score_status"].eq("novel")
    require(int(inv_reused.sum()) == 901, f"Expected 901 reused inventory scores, got {int(inv_reused.sum())}")
    require(int(inv_novel.sum()) == 675, f"Expected 675 novel inventory rows, got {int(inv_novel.sum())}")
    reused_vals = pd.to_numeric(inventory.loc[inv_reused, "judge_score"], errors="raise").astype(int)
    require(reused_vals.between(0,4).all(), "Reused inventory score outside 0..4")
    require(inventory.loc[inv_novel, "judge_score"].astype(str).str.strip().eq("").all(), "Novel inventory unexpectedly contains score")

    partial_pairs = set(zip(partial["query_id"].astype(str), partial["isbn13"].astype(str)))
    novel_pairs = set(zip(inventory.loc[inv_novel, "query_id"].astype(str), inventory.loc[inv_novel, "isbn13"].astype(str)))
    quarantine_pair = (KNOWN_QID, KNOWN_ISBN)
    require(partial_pairs | {quarantine_pair} == novel_pairs, "674 persisted + quarantine do not exactly cover the 675 novel inventory pairs")
    require(partial_pairs.isdisjoint({quarantine_pair}), "Quarantine pair overlaps persisted scores")

    class_lock = load_json(V3_CLASS_LOCK)
    require(class_lock.get("status") == "V3_POSTHOC_RETRIEVAL_VS_RERANKER_CLASSIFICATION_FROZEN", "Unexpected V3 Stage-C classification status")
    class_hash = class_lock.get("outputs", {}).get("query_retrieval_reranker_classification.csv")
    require(class_hash, "V3 Stage-C lock does not contain classification CSV hash")
    require(V3_CLASSIFICATION.exists(), f"Missing V3 Stage-C classification CSV: {V3_CLASSIFICATION}")
    require(sha256(V3_CLASSIFICATION) == class_hash, "V3 Stage-C classification CSV changed")
    cls = pd.read_csv(V3_CLASSIFICATION, dtype=str, keep_default_na=False, encoding="utf-8")
    require(len(cls) == 17, f"Expected 17 Stage-C classifications, got {len(cls)}")
    require(set(cls["query_id"].astype(str)) == set(inventory["query_id"].astype(str)), "Stage-C query set differs from sensitivity query set")
    require(set(cls["posthoc_limitation_class"]) <= {"RETRIEVAL_LIMITED","RERANKER_LIMITED","MIXED"}, "Unexpected Stage-C class")

    comparison = load_json(V4A_COMPARISON_LOCK)
    require(comparison.get("status") == "V4A_DEVELOPMENT_COMPARISON_AND_SELECTION_FROZEN", "Unexpected V4A comparison lock status")
    require(comparison.get("selected_candidate") == "v3b", "Frozen development selection is no longer v3b")
    require(comparison.get("release_decision_produced") is False, "Comparison lock unexpectedly contains a release decision")

    print("PASS  frozen 1575 actual-score boundary + 1 unscored quarantine verified")
    print("PASS  all five hypothetical scenarios remain counterfactual only")
    print("PASS  exact 17-query Stage-C baseline classification set verified")
    print("PASS  frozen development selection remains v3b and cannot be reopened")
    print()
    print("V4A RETRIEVAL DIAGNOSTIC SENSITIVITY ANALYSIS PREFLIGHT: PASS")
    print("Sensitivity scenarios frozen: S0=0, S1=1, S2=2, S3=3, S4=4")
    print("Actual quarantine score assigned: NO")
    print("Diagnostic calculations performed: NO")
    print("Judge calls performed: 0")
    print("Model calls performed: 0")
    print("Development selection reopened: NO")
    print("Release analysis: NO")
    print("Next: execute deterministic five-scenario sensitivity analysis.")

if __name__ == "__main__":
    main()
