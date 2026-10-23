from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from typing import Any
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT/"evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))
SYS = EVALS/"system_evaluation"
RUN = ROOT/"evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"
HIST_RUN = ROOT/"evals/runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"

from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec, validate_query_facet_spec

JUDGING_INPUT = RUN/"v4h_headroom_stage_a_judging_input.csv"
JUDGING_INPUT_LOCK = RUN/"v4h_headroom_stage_a_judging_input.lock.json"
SCORE_INVENTORY = RUN/"v4h_headroom_stage_a_score_inventory.csv"
V4H_FACET_FILE = EVALS/"facets/semantic_relevance/semantic_relevance_v4_holdout_query_facets.v0.1.0.json"
V4H_AUTHORING_LOCK = SYS/"semantic_relevance_v4_holdout_authoring.lock.json"
HISTORICAL_SCORING_LOCK = HIST_RUN/"holdout_judge_scoring_lock.json"
JUDGE_FILE = EVALS/"semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS/"semantic_relevance_facet_scoring.py"
JUDGE_CONFIG_FILE = EVALS/"judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
RUBRIC_FILE = EVALS/"rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
SPAN_ECHO_PATCH_FILE = SYS/"semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"
ORIGINAL_DUP_PATCH_FILE = SYS/"semantic_relevance_v4h_execution_patch.duplicate-candidate-id-normalization1.json"
CARRY_DUP_PATCH_FILE = SYS/"semantic_relevance_v4h_headroom_stage_a_execution_patch.duplicate-candidate-id-carryforward1.json"
EXEC_CONTRACT = SYS/"semantic_relevance_v4h_headroom_stage_a_execution_contract.v0.1.0.json"
FRESH_SCORES = RUN/"v4h_headroom_stage_a_fresh_judge_scores.csv"
FINAL_LOCK = RUN/"v4h_headroom_stage_a_fresh_judge_scoring.lock.json"


TARGET_QUERY_IDS = ['V4H07', 'V4H09', 'V4H10', 'V4H11', 'V4H13', 'V4H16', 'V4H19', 'V4H21', 'V4H22', 'V4H23']
EXPECTED_FRESH_ROWS = 340
EXPECTED_JUDGING_INPUT_SHA256 = "eb57cfead0f79cedb9f350b3be683a5b063ab62355cc9074e7df264fc4fe14c8"
EXPECTED_JUDGING_INPUT_LOCK_SHA256 = "67d1dfbc2852adf96c97de0a34603010f06bab37aa77854b21c6dd5325653173"
EXPECTED_SCORE_INVENTORY_SHA256 = "fbbc796bc885842e5f9e63ce9ea7d2c7540bdeea09aa34b51c44a80988807cda"
EXPECTED_V4H_FACET_SHA256 = "dbdb7c5d5603e17164cf6ef880df912f6196663387885e1f5773e54b97d2b106"
EXPECTED_V4H_AUTHORING_LOCK_SHA256 = "14cd3a8a9f6bf22666dd4c74a150e85f82a53789ccaa5bd2870d606b66ef83e7"
EXPECTED_V3_SCORING_LOCK_SHA256 = "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777"
EXPECTED_SPAN_ECHO_PATCH_SHA256 = "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266"
EXPECTED_ORIGINAL_DUP_PATCH_SHA256 = "09b2acb1d3809da6ed4667d1e3f7b042dc86f85b6f69b732bb9017136f0b2b94"
EXPECTED_CARRY_PATCH_SHA256 = "5e38542976fcb091a9f6461d2be11a34716d542b592bd96228e51146ac40ac73"
EXPECTED_EXEC_CONTRACT_SHA256 = "5385d221e2c80968dd976b4bf18abdbda6bbc3c8b30db264167f572c78b47dbb"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def lf_sha256(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))

def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

def verify_text(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    raw, norm = sha256(path), lf_sha256(path)
    req(raw == expected or norm == expected,
        f"{label} hash mismatch: expected={expected} raw={raw} lf={norm}")
    print(f"PASS  {label} {'raw' if raw == expected else 'canonical-LF'} SHA-256 verified")

def canonical_isbn(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    req(bool(text), "ISBN is empty")
    return text

def load_facet_specs(payload: dict[str, Any]) -> dict[str, QueryFacetSpec]:
    raw_queries = payload.get("queries")
    req(isinstance(raw_queries, list), "Facet specification must contain queries list")
    specs = {}
    for raw in raw_queries:
        qid = str(raw["query_id"])
        spec = QueryFacetSpec(
            query_id=qid,
            query=str(raw["query"]),
            facets=[QueryFacet(**f) for f in raw["facets"]],
        )
        validate_query_facet_spec(spec)
        specs[qid] = spec
    req(all(qid in specs for qid in TARGET_QUERY_IDS), "Target query missing from frozen facets")
    return specs

def verify_measurement_instrument() -> dict[str, Any]:
    verify_raw(HISTORICAL_SCORING_LOCK, EXPECTED_V3_SCORING_LOCK_SHA256,
               "V3 frozen holdout scoring lock")
    hlock = load_json(HISTORICAL_SCORING_LOCK)
    req(hlock.get("status") == "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS",
        "Unexpected historical scoring-lock status")
    req(hlock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5",
        "Frozen judge candidate changed")
    req(hlock.get("human_relevance_labels_used") is False,
        "Historical measurement instrument used human labels")

    for path, key, label in [
        (JUDGE_FILE, "judge_source_sha256", "frozen judge source"),
        (SCORING_FILE, "scoring_source_sha256", "frozen deterministic scoring"),
        (JUDGE_CONFIG_FILE, "judge_config_sha256", "frozen judge config"),
        (RUBRIC_FILE, "rubric_sha256", "frozen rubric"),
    ]:
        expected = str(hlock.get(key, ""))
        req(bool(expected), f"Historical lock missing {key}")
        verify_text(path, expected, label)

    verify_raw(V4H_FACET_FILE, EXPECTED_V4H_FACET_SHA256, "V4H facet specification")
    verify_raw(V4H_AUTHORING_LOCK, EXPECTED_V4H_AUTHORING_LOCK_SHA256, "V4H authoring lock")
    verify_text(SPAN_ECHO_PATCH_FILE, EXPECTED_SPAN_ECHO_PATCH_SHA256,
                "accepted span-echo execution patch")
    verify_raw(ORIGINAL_DUP_PATCH_FILE, EXPECTED_ORIGINAL_DUP_PATCH_SHA256,
               "accepted original duplicate-ID execution patch")
    verify_raw(CARRY_DUP_PATCH_FILE, EXPECTED_CARRY_PATCH_SHA256,
               "Stage-A duplicate-ID carry-forward amendment")
    verify_raw(EXEC_CONTRACT, EXPECTED_EXEC_CONTRACT_SHA256,
               "Stage-A judge-execution contract")

    opatch = load_json(ORIGINAL_DUP_PATCH_FILE)
    req(opatch.get("patch_id") == "v4h-stage-a-duplicate-candidate-id-normalization1",
        "Unexpected original duplicate-ID patch ID")
    cpatch = load_json(CARRY_DUP_PATCH_FILE)
    req(cpatch.get("based_on_patch_sha256") == EXPECTED_ORIGINAL_DUP_PATCH_SHA256,
        "Carry-forward amendment does not bind original duplicate-ID patch")
    for key in [
        "judge_source_modified","judge_prompt_modified","judge_model_modified",
        "judge_temperature_modified","semantic_rules_changed",
        "deterministic_scoring_rules_changed","facet_spec_changed",
        "judge_config_changed","rubric_changed","candidate_set_changed",
        "release_metrics_or_gates_changed","release_decision_modified",
    ]:
        req(cpatch.get(key) is False, f"Carry-forward amendment changes forbidden contract: {key}")

    config = load_json(JUDGE_CONFIG_FILE)
    req(config.get("version") == "0.29.0-r5", "Judge config version changed")
    req(config.get("provider") == "ollama", "Judge provider changed")
    req(float(config.get("temperature")) == 0.0, "Judge temperature changed")
    rubric = load_json(RUBRIC_FILE)
    req(rubric.get("status") == "frozen", "Rubric is not frozen")
    req(rubric.get("version") == "0.1.0", "Rubric version changed")
    specs = load_facet_specs(load_json(V4H_FACET_FILE))
    print("PASS  frozen v0.29.0-r5 measurement instrument verified")
    print("PASS  target V4H facets validated against frozen judge schema")
    return {"lock": hlock, "config": config, "rubric": rubric, "facet_specs": specs}

def verify_input_boundary() -> tuple[pd.DataFrame, dict[str, Any]]:
    verify_raw(JUDGING_INPUT, EXPECTED_JUDGING_INPUT_SHA256, "Stage-A fresh judging input")
    verify_raw(JUDGING_INPUT_LOCK, EXPECTED_JUDGING_INPUT_LOCK_SHA256, "Stage-A judging-input lock")
    verify_raw(SCORE_INVENTORY, EXPECTED_SCORE_INVENTORY_SHA256, "Stage-A score inventory")

    lock = load_json(JUDGING_INPUT_LOCK)
    req(lock.get("status") == "V4H_HEADROOM_STAGE_A_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "Unexpected Stage-A judging-input lock status")
    req(int(lock.get("fresh_pair_count", -1)) == EXPECTED_FRESH_ROWS, "Fresh pair count changed")
    req(int(lock.get("reused_exact_pair_count", -1)) == 60, "Reused pair count changed")
    req(int(lock.get("total_stage_a_pair_count", -1)) == 400, "Stage-A total pair count changed")
    req(lock.get("fresh_judging_input_sha256") == EXPECTED_JUDGING_INPUT_SHA256,
        "Input lock fresh-input hash changed")
    req(lock.get("score_inventory_sha256") == EXPECTED_SCORE_INVENTORY_SHA256,
        "Input lock score-inventory hash changed")
    req(lock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5", "Judge candidate changed")
    req(lock.get("historical_score_reuse_cross_query_allowed") is False,
        "Cross-query score reuse unexpectedly allowed")
    req(lock.get("approximate_score_reuse_allowed") is False,
        "Approximate score reuse unexpectedly allowed")
    req(lock.get("judge_execution_performed") is False,
        "Input lock says judge already executed")
    req(lock.get("release_decision_modified") is False,
        "Input lock says release decision changed")

    frame = pd.read_csv(JUDGING_INPUT, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"case_id","query_id","query","stratum","isbn13","title","description","v4b_rank"}
    missing = required - set(frame.columns)
    req(not missing, f"Fresh input missing columns: {sorted(missing)}")
    req(len(frame) == EXPECTED_FRESH_ROWS, f"Expected 340 fresh rows, got {len(frame)}")
    frame["isbn13"] = frame["isbn13"].map(canonical_isbn)
    frame["v4b_rank_num"] = pd.to_numeric(frame["v4b_rank"], errors="raise").astype(int)
    req(frame["v4b_rank_num"].between(11,50).all(), "Fresh input contains rank outside 11-50")
    req(not frame["case_id"].duplicated().any(), "Duplicate fresh case ID")
    req(not frame.duplicated(["query_id","isbn13"]).any(), "Duplicate fresh query-book pair")
    req(set(frame["query_id"]).issubset(set(TARGET_QUERY_IDS)), "Fresh input contains non-target query")
    req(frame["description"].str.strip().ne("").all(), "Fresh input has empty description")
    return frame, lock


def main() -> None:
    print("Semantic relevance V4H - headroom Stage-A judge execution preflight")
    print("-----------------------------------------------------------------")
    print("Role: consumed-holdout diagnostic only")
    print("Fresh cases: 340")
    print("Judge execution: DISABLED")
    print("Release decision modification: DISABLED")
    print()

    frame, _ = verify_input_boundary()
    verify_measurement_instrument()
    req(not FINAL_LOCK.exists(), "Final Stage-A fresh-score lock already exists")
    if FRESH_SCORES.exists():
        existing = pd.read_csv(FRESH_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
        req(set(existing["case_id"]).issubset(set(frame["case_id"])),
            "Persisted Stage-A score contains case outside frozen fresh input")
        print(f"INFO  resume-state score artifact exists: {len(existing)}/340 persisted")
    else:
        print("INFO  no Stage-A fresh-score artifact exists yet")

    print()
    print("V4H HEADROOM STAGE-A JUDGE PREFLIGHT: PASS")
    print("Fresh frozen pairs: 340")
    print("Measurement instrument: semantic_relevance_v0.29.0-r5")
    print("Resume-safe persistence: ENABLED")
    print("Judge calls performed during preflight: 0")
    print("Release decision modified: NO")
    print("Next: run a 3-case smoke test.")

if __name__ == "__main__":
    main()
