from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYS = ROOT / "evals/system_evaluation"
V4H_RUN = ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"

SPEC_JSON = SYS / "semantic_relevance_v5_development_design.v0.1.0.json"
SPEC_MD = SYS / "semantic_relevance_v5_development_design.v0.1.0.md"
V5_LOCK = SYS / "semantic_relevance_v5_development_design.v0.1.0.lock.json"
V4B_DESIGN_LOCK = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.lock.json"
V4B_IMPLEMENTATION_LOCK = SYS / "semantic_relevance_v4b_implementation.lock.json"
V4H_FINAL_LOCK = V4H_RUN / "v4h_final_release_analysis.lock.json"
V4H_HEADROOM_CLOSEOUT_LOCK = V4H_RUN / "v4h_headroom_diagnostic_closeout.lock.json"

EXPECTED = {
    "spec_json": "ef21af8375d3606cf53038f6986a8cd7e3c389fee8345770a39aff73d4cc4551",
    "spec_md": "81e896a4b3c74c3a026c5c7379a651e244705064d5cc2499041630517accc7e5",
    "v4b_design_lock": "e60ca1032f7fa733f3e7a51ec87a229d7454645aa6109b5e557588af50383c6d",
    "v4b_implementation_lock": "f30a61e7ca7a5e07766c3f2baa3d4a65abcedb48cdedf23ee6d5c0e05b3171b0",
    "v4h_final_lock": "7e7d73920452ec39ea72947d2b3762a02aaa1b2bb0bcd7470c7e21dbb3a25ab0",
    "v4h_headroom_closeout_lock": "df16cc592716337c5eb27bdf2610e7ba94ebb2c12ec4bb70ca0f875d88f96dad",
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def lf_sha256(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()

def req(x: bool, msg: str) -> None:
    if not x: raise ValueError(msg)

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def verify_text(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    raw = sha256(path)
    if raw == expected:
        mode = "raw"
    else:
        norm = lf_sha256(path)
        req(norm == expected, f"{label} hash mismatch: expected={expected} raw={raw} lf={norm}")
        mode = "canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")

def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

def main() -> None:
    print("Semantic relevance V5 - development design freeze")
    print("-------------------------------------------------")
    print("V5 implementation execution: DISABLED")
    print("V5D authoring/execution: DISABLED")
    print("Retrieval/model/reranker/judge calls: DISABLED")
    print("Development selection: NOT CALCULATED")
    print("Release analysis: DISABLED")
    print()

    req(not V5_LOCK.exists(), f"Refusing to overwrite existing V5 design lock: {V5_LOCK}")
    verify_text(SPEC_JSON, EXPECTED["spec_json"], "V5 JSON design spec")
    verify_text(SPEC_MD, EXPECTED["spec_md"], "V5 Markdown design spec")
    verify_raw(V4B_DESIGN_LOCK, EXPECTED["v4b_design_lock"], "V4B design lock")
    verify_raw(V4B_IMPLEMENTATION_LOCK, EXPECTED["v4b_implementation_lock"], "V4B implementation lock")
    verify_raw(V4H_FINAL_LOCK, EXPECTED["v4h_final_lock"], "V4H final release-analysis lock")
    verify_raw(V4H_HEADROOM_CLOSEOUT_LOCK, EXPECTED["v4h_headroom_closeout_lock"], "V4H headroom diagnostic closeout lock")

    spec = load(SPEC_JSON)
    v4b = load(V4B_DESIGN_LOCK)
    v4bi = load(V4B_IMPLEMENTATION_LOCK)
    final = load(V4H_FINAL_LOCK)
    closeout = load(V4H_HEADROOM_CLOSEOUT_LOCK)

    req(v4b.get("status") == "V4B_DESIGN_FROZEN_BEFORE_IMPLEMENTATION", "Unexpected V4B design status")
    req(v4bi.get("status") == "V4B_IMPLEMENTATION_FROZEN_BEFORE_DEVELOPMENT_RANKING", "Unexpected V4B implementation status")
    req(final.get("status") == "V4H_FINAL_RELEASE_ANALYSIS_FROZEN", "Unexpected V4H final lock status")
    req(final.get("v4b_final_release_decision") == "FAIL", "Frozen V4B V4H release decision changed")
    req(closeout.get("status") == "V4H_HEADROOM_DIAGNOSTIC_CLOSED_FROZEN", "Unexpected V4H headroom closeout status")
    req(closeout.get("final_headroom_disposition_counts") == {
        "HEADROOM_FOUND_11_50": 4,
        "HEADROOM_FOUND_51_END": 0,
        "NO_CLEAR_THROUGH_FULL_UNION": 6,
    }, "V4H headroom disposition changed")
    req(closeout.get("release_decision_modified") is False, "V4H closeout changed release decision")

    req(spec.get("status") == "PROPOSED_FOR_FREEZE_BEFORE_V5_COMPONENT_IMPLEMENTATION", "Unexpected V5 spec status")
    req(spec["authority_boundary"]["v4h_role"] == "consumed holdout; diagnostic evidence only", "V4H authority boundary changed")
    req(spec["authority_boundary"]["v4h_may_select_parameters_models_prompts_or_thresholds"] is False, "V4H tuning unexpectedly allowed")
    req(spec["fresh_v5_development_set"]["query_count"] == 24, "V5D query count changed")
    req(spec["fresh_v5_development_set"]["queries_per_stratum"] == 4, "V5D stratum balance changed")
    req(spec["fresh_v5_development_set"]["query_and_facet_files_frozen_before_any_v5d_retrieval"] is True, "V5D freeze ordering changed")
    req(spec["factorial_program"]["systems"] == [
        {"id":"B","retrieval":"R0","ranking":"K0","role":"baseline"},
        {"id":"R","retrieval":"R1","ranking":"K0","role":"retrieval-only candidate"},
        {"id":"K","retrieval":"R0","ranking":"K1","role":"ranking-only candidate"},
        {"id":"RK","retrieval":"R1","ranking":"K1","role":"combined candidate"},
    ], "V5 factorial matrix changed")
    req(spec["factorial_program"]["factors"]["R"]["candidate_budget_rule"].startswith("R1 may emit at most 100"), "R1 candidate budget boundary changed")
    req(spec["evaluation"]["candidate_diagnostic_depth"] == 50, "V5 candidate diagnostic depth changed")
    req(spec["development_selection_rule"]["no_post_result_parameter_search"] is True, "Post-result parameter search unexpectedly allowed")
    req(spec["implementation_freeze_requirements"]["freeze_before"] == "V5D query authoring and any V5D retrieval/model execution", "Component-freeze ordering changed")
    req(spec["final_unseen_holdout"]["same_original_eight_release_gates"] is True, "Release gates changed")

    payload = {
        "status": "V5_DEVELOPMENT_DESIGN_FROZEN_BEFORE_COMPONENT_IMPLEMENTATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "program_id": spec["program_id"],
        "version": spec["version"],
        "spec_json_sha256": EXPECTED["spec_json"],
        "spec_md_sha256": EXPECTED["spec_md"],
        "v4b_design_lock_sha256": EXPECTED["v4b_design_lock"],
        "v4b_implementation_lock_sha256": EXPECTED["v4b_implementation_lock"],
        "v4h_final_release_analysis_lock_sha256": EXPECTED["v4h_final_lock"],
        "v4h_headroom_closeout_lock_sha256": EXPECTED["v4h_headroom_closeout_lock"],
        "v4h_consumed": True,
        "frozen_v4b_release_decision": "FAIL",
        "factorial_systems": ["B", "R", "K", "RK"],
        "fresh_v5d_query_count": 24,
        "v5d_queries_per_stratum": 4,
        "r1_max_unique_candidate_budget": 100,
        "candidate_diagnostic_depth": 50,
        "r1_k1_exact_specs_frozen_before_v5d_authoring": True,
        "v4h_tuning_allowed": False,
        "retrieval_calls_performed": 0,
        "model_calls_performed": 0,
        "reranker_calls_performed": 0,
        "judge_calls_performed": 0,
        "development_selection_calculated": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "next_stage": spec["next_stage"],
    }
    V5_LOCK.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("PASS  V4H remains consumed diagnostic evidence only; V4B FAIL remains frozen")
    print("PASS  V5 factorized B/R/K/RK program frozen")
    print("PASS  fresh V5D24 protocol frozen: 6 strata x 4 queries")
    print("PASS  R1 candidate budget capped at V4B maximum of 100 unique candidates/query")
    print("PASS  exact R1/K1 implementations required before V5D authoring or execution")
    print("PASS  original release gates preserved")
    print()
    print("V5 DEVELOPMENT DESIGN FREEZE: PASS")
    print(f"V5 JSON spec SHA-256: {sha256(SPEC_JSON)}")
    print(f"V5 Markdown spec SHA-256: {sha256(SPEC_MD)}")
    print(f"V5 design lock SHA-256: {sha256(V5_LOCK)}")
    print("Retrieval/model/reranker/judge calls performed: 0")
    print("Development selection calculated: NO")
    print("Release metrics calculated: NO")
    print("Frozen V4B release decision modified: NO")
    print("Next: freeze exact V5 R1 candidate-generation and K1 ranking specs/implementations before V5D authoring.")

if __name__ == "__main__":
    main()
