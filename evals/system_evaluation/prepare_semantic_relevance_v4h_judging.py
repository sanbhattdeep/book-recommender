from __future__ import annotations

import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))
SYS = EVALS / "system_evaluation"
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"
HIST_RUN = REPO_ROOT / "evals/runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"

DISPLAY_UNION = RUN / "v4h_v3b_v4b_display_union.csv"
DISPLAY_UNION_LOCK = RUN / "v4h_v3b_v4b_display_union.lock.json"
V4H_AUTHORING_LOCK = SYS / "semantic_relevance_v4_holdout_authoring.lock.json"
V4H_FACET_FILE = EVALS / "facets/semantic_relevance/semantic_relevance_v4_holdout_query_facets.v0.1.0.json"

HISTORICAL_SCORING_LOCK = HIST_RUN / "holdout_judge_scoring_lock.json"
JUDGE_FILE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS / "semantic_relevance_facet_scoring.py"
JUDGE_CONFIG_FILE = EVALS / "judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
RUBRIC_FILE = EVALS / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
SPAN_ECHO_PATCH_FILE = SYS / "semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"
JUDGING_CONTRACT = SYS / "semantic_relevance_v4h_judging_contract.v0.1.0.json"

JUDGING_INPUT = RUN / "v4h_judging_input.csv"
JUDGING_INPUT_LOCK = RUN / "v4h_judging_input.lock.json"
SCORES = RUN / "v4h_judge_scores.csv"
SCORING_LOCK = RUN / "v4h_judge_scoring.lock.json"

from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec, validate_query_facet_spec


EXPECTED = {
    "display_union": "444fa3722dc1e8815e111ab95db9a35b501ae9e57d692d560c6c5fe5955f096a",
    "display_union_lock": "bbb477a60ea123071ee9ec659cac63f987ef5f817a23e7788567ef650fa7387c",
    "v4h_facet": "dbdb7c5d5603e17164cf6ef880df912f6196663387885e1f5773e54b97d2b106",
    "v4h_authoring_lock": "14cd3a8a9f6bf22666dd4c74a150e85f82a53789ccaa5bd2870d606b66ef83e7",
    "v3_holdout_scoring_lock": "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777",
    "span_echo_patch": "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266"
}
EXPECTED_ROWS = 382
EXPECTED_QUERY_IDS = ["V4H01", "V4H02", "V4H03", "V4H04", "V4H05", "V4H06", "V4H07", "V4H08", "V4H09", "V4H10", "V4H11", "V4H12", "V4H13", "V4H14", "V4H15", "V4H16", "V4H17", "V4H18", "V4H19", "V4H20", "V4H21", "V4H22", "V4H23", "V4H24"]
EXPECTED_CONTRACT_SHA256 = "918991e145fdeb71391cc42b035b48f485b027c226d831ea4b04ee14cce80469"

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
    raw = sha256(path)
    norm = lf_sha256(path)
    req(raw == expected or norm == expected,
        f"{label} hash mismatch: expected={expected} raw={raw} lf={norm}")
    print(f"PASS  {label} {'raw' if raw == expected else 'canonical-LF'} SHA-256 verified")

def canonical_isbn(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text:
        raise ValueError("ISBN is empty")
    return text

def load_facet_specs(payload: dict[str, Any]) -> dict[str, QueryFacetSpec]:
    raw_queries = payload.get("queries")
    req(isinstance(raw_queries, list), "Facet specification must contain a queries list")
    specs = {}
    for raw_query in raw_queries:
        qid = str(raw_query["query_id"])
        req(qid not in specs, f"Duplicate query ID in facet spec: {qid}")
        spec = QueryFacetSpec(
            query_id=qid,
            query=str(raw_query["query"]),
            facets=[QueryFacet(**f) for f in raw_query["facets"]],
        )
        validate_query_facet_spec(spec)
        for facet in spec.facets:
            req(bool(facet.required_components), f"{qid}/{facet.facet_id} missing required_components")
            ids = [c.component_id for c in facet.required_components]
            req(len(ids) == len(set(ids)), f"{qid}/{facet.facet_id} duplicate component IDs")
        specs[qid] = spec
    req(sorted(specs) == EXPECTED_QUERY_IDS, "Facet spec query IDs changed")
    return specs

def verify_measurement_instrument() -> dict[str, Any]:
    verify_raw(HISTORICAL_SCORING_LOCK, EXPECTED["v3_holdout_scoring_lock"], "V3 frozen holdout scoring lock")
    hlock = load_json(HISTORICAL_SCORING_LOCK)
    req(hlock.get("status") == "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS",
        "Unexpected historical scoring-lock status")
    req(hlock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5",
        "Frozen judge candidate changed")
    req(hlock.get("human_relevance_labels_used") is False,
        "Historical frozen judge used human relevance labels")

    for path, key, label in [
        (JUDGE_FILE, "judge_source_sha256", "frozen judge source"),
        (SCORING_FILE, "scoring_source_sha256", "frozen deterministic scoring"),
        (JUDGE_CONFIG_FILE, "judge_config_sha256", "frozen judge config"),
        (RUBRIC_FILE, "rubric_sha256", "frozen rubric"),
    ]:
        expected = str(hlock.get(key, ""))
        req(expected, f"Historical scoring lock missing {key}")
        verify_text(path, expected, label)

    verify_raw(V4H_FACET_FILE, EXPECTED["v4h_facet"], "V4H facet specification")
    facet_payload = load_json(V4H_FACET_FILE)
    specs = load_facet_specs(facet_payload)

    verify_text(SPAN_ECHO_PATCH_FILE, EXPECTED["span_echo_patch"], "accepted span-echo execution patch")
    patch = load_json(SPAN_ECHO_PATCH_FILE)
    req(patch.get("status") == "APPLIED_NON_SEMANTIC_EXECUTION_PATCH",
        "Unexpected span-echo patch status")
    req(patch.get("patch_id") == "v3-posthoc-prominence-span-echo-normalization1",
        "Unexpected span-echo patch ID")
    for key in ["judge_source_modified","judge_prompt_modified","semantic_rules_changed",
                "deterministic_scoring_rules_changed","facet_spec_changed",
                "judge_config_changed","rubric_changed","release_metrics_or_gates_changed"]:
        req(patch.get(key) is False, f"Span-echo patch changed forbidden contract: {key}")

    config = load_json(JUDGE_CONFIG_FILE)
    req(config.get("version") == "0.29.0-r5", "Judge config version changed")
    req(config.get("provider") == "ollama", "Judge provider changed")
    req(float(config.get("temperature")) == 0.0, "Judge temperature changed")
    req(config.get("rubric_id") == "semantic-recommendation-relevance", "Judge rubric ID changed")

    rubric = load_json(RUBRIC_FILE)
    req(rubric.get("status") == "frozen", "Rubric is not frozen")
    req(rubric.get("version") == "0.1.0", "Rubric version changed")

    print("PASS  frozen v0.29.0-r5 measurement instrument verified")
    print("PASS  V4H facet specification validated against frozen judge schema")
    return {"lock": hlock, "config": config, "rubric": rubric, "facet_specs": specs}

def verify_display_union() -> pd.DataFrame:
    verify_raw(DISPLAY_UNION, EXPECTED["display_union"], "V4H displayed-pair union")
    verify_raw(DISPLAY_UNION_LOCK, EXPECTED["display_union_lock"], "V4H displayed-pair union lock")
    verify_raw(V4H_AUTHORING_LOCK, EXPECTED["v4h_authoring_lock"], "V4H authoring lock")
    verify_raw(JUDGING_CONTRACT, EXPECTED_CONTRACT_SHA256, "V4H judging contract")

    lock = load_json(DISPLAY_UNION_LOCK)
    req(lock.get("status") == "V4H_DISPLAY_UNION_FROZEN_BEFORE_JUDGING",
        "Unexpected display-union lock status")
    req(int(lock.get("union_rows", -1)) == EXPECTED_ROWS, "Display-union row count changed")
    req(lock.get("display_union_sha256") == EXPECTED["display_union"], "Display-union lock hash mismatch")
    req(lock.get("historical_score_reuse_allowed") is False, "Historical score reuse unexpectedly allowed")
    req(lock.get("all_unique_pairs_require_judging") is True, "Not all unique pairs require judging")
    req(lock.get("judge_execution_performed") is False, "Display-union lock says judge already ran")
    req(lock.get("human_relevance_labels_used") is False, "Display-union lock says human labels used")
    req(lock.get("release_metrics_calculated") is False, "Display-union lock says metrics already calculated")

    frame = pd.read_csv(DISPLAY_UNION, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id","query","stratum","isbn13","title","description","authors",
        "pair_source","in_v3b","v3b_rank","in_v4b","v4b_rank"
    }
    missing = required - set(frame.columns)
    req(not missing, f"Display union missing columns: {sorted(missing)}")
    req(len(frame) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} union rows, got {len(frame)}")
    frame["isbn13"] = frame["isbn13"].map(canonical_isbn)
    req(not frame.duplicated(["query_id","isbn13"]).any(), "Duplicate union query-book pairs")
    req(sorted(frame["query_id"].unique()) == EXPECTED_QUERY_IDS, "Display union query IDs changed")
    req(frame["pair_source"].isin(["both","v3b_only","v4b_only"]).all(), "Unexpected pair_source")
    forbidden = {"human_score","human_reason","review_status","label","relevance_label"}
    req(not (forbidden & set(frame.columns)),
        f"Human/label-like columns present in judge inventory: {sorted(forbidden & set(frame.columns))}")
    req(frame["description"].str.strip().ne("").all(), "Judge inventory contains empty description")
    return frame

def deterministic_case_id(query_id: str, isbn13: str) -> str:
    return f"V4H_JUDGE_{query_id}_{canonical_isbn(isbn13)}"


def main() -> None:
    print("Semantic relevance V4H - frozen judge-input preparation")
    print("-------------------------------------------------------")
    print("Judge execution: DISABLED")
    print("Historical score reuse: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release metrics/decision: NOT CALCULATED")
    print()

    req(not JUDGING_INPUT.exists(), f"Refusing to overwrite existing judging input: {JUDGING_INPUT}")
    req(not JUDGING_INPUT_LOCK.exists(), f"Refusing to overwrite existing judging-input lock: {JUDGING_INPUT_LOCK}")
    req(not SCORES.exists(), f"Judge score file already exists: {SCORES}")
    req(not SCORING_LOCK.exists(), f"Judge scoring lock already exists: {SCORING_LOCK}")

    instrument = verify_measurement_instrument()
    union = verify_display_union()

    # Exact query text must match the frozen V4H facet spec.
    specs = instrument["facet_specs"]
    for qid, group in union.groupby("query_id", sort=True):
        req(group["query"].nunique() == 1, f"{qid} has multiple query texts")
        req(str(group["query"].iloc[0]) == specs[qid].query,
            f"{qid} query text disagrees with frozen facet specification")

    judge_input = union.copy()
    judge_input.insert(
        0,
        "case_id",
        [
            deterministic_case_id(qid, isbn)
            for qid, isbn in zip(judge_input["query_id"], judge_input["isbn13"])
        ],
    )
    req(not judge_input["case_id"].duplicated().any(), "Duplicate deterministic case IDs")
    req(len(judge_input) == EXPECTED_ROWS, "Judge-input row count changed")
    judge_input = judge_input.sort_values(["query_id","isbn13"], kind="stable").reset_index(drop=True)
    judge_input.to_csv(JUDGING_INPUT, index=False, encoding="utf-8")

    lock = {
        "status": "V4H_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_set": "V4H01-V4H24",
        "query_count": 24,
        "case_count": EXPECTED_ROWS,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "display_union_sha256": EXPECTED["display_union"],
        "display_union_lock_sha256": EXPECTED["display_union_lock"],
        "v4h_facet_spec_sha256": EXPECTED["v4h_facet"],
        "historical_measurement_instrument_lock_sha256": EXPECTED["v3_holdout_scoring_lock"],
        "span_echo_patch_sha256": EXPECTED["span_echo_patch"],
        "judging_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "judging_input_sha256": sha256(JUDGING_INPUT),
        "case_id_rule": "V4H_JUDGE_<query_id>_<canonical_isbn13>",
        "historical_score_reuse_allowed": False,
        "fresh_pairs_to_judge": EXPECTED_ROWS,
        "judge_execution_performed": False,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "next_stage": "EXECUTE_FROZEN_V0_29_R5_JUDGE_ON_ALL_382_V4H_DISPLAYED_PAIRS",
    }
    JUDGING_INPUT_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print(f"PASS  deterministic judge input frozen: {len(judge_input)} cases")
    print("PASS  every V4H displayed pair requires fresh frozen-judge scoring")
    print("PASS  no human relevance labels are present in the judge input")
    print()
    print("V4H JUDGING INPUT PREPARATION: PASS")
    print(f"Judging input SHA-256: {sha256(JUDGING_INPUT)}")
    print(f"Judging-input lock SHA-256: {sha256(JUDGING_INPUT_LOCK)}")
    print("Judge calls performed: 0")
    print("Human labels used: NO")
    print("Release metrics calculated: NO")
    print("Next: execute the frozen v0.29.0-r5 judge on all 382 cases with resume-safe persistence.")

if __name__ == "__main__":
    main()
