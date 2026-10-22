from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

import pandas as pd
from deepeval.models import OllamaModel

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

import semantic_relevance_facet_judge as judge_module
from semantic_relevance_facet_judge import (
    generate_validated_semantic_verdict,
    serialize_facet_assessments,
    serialize_facet_evidence,
    supported_core_facet_texts,
    supported_evidence_by_id,
)
from semantic_relevance_facet_scoring import compute_facet_score
from run_judge_v0_29_r5_development import load_facet_specs

RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
HOLDOUT_RUN = EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"

NOVEL_INPUT = RUN / "v4a_retrieval_diagnostic_novel_bm25_pairs_to_judge.csv"
SCORE_INVENTORY = RUN / "v4a_retrieval_diagnostic_score_inventory.csv"
JUDGING_INPUT_LOCK = RUN / "v4a_retrieval_diagnostic_input.lock.json"
HOLDOUT_SCORING_LOCK = HOLDOUT_RUN / "holdout_judge_scoring_lock.json"

HOLDOUT_FACET_FILE = EVALS / "facets/semantic_relevance/semantic_relevance_v3_holdout_query_facets.v0.1.0.json"
JUDGE_FILE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS / "semantic_relevance_facet_scoring.py"
JUDGE_CONFIG_FILE = EVALS / "judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
RUBRIC_FILE = EVALS / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
SPAN_ECHO_PATCH_FILE = EVALS / "system_evaluation/semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"
TIMEOUT_PATCH_FILE = REPO_ROOT / "v4a_retrieval_diagnostic_deepeval_timeout_patch1.json"
QUARANTINE_PATCH_FILE = REPO_ROOT / "v4a_retrieval_diagnostic_case_quarantine_patch1.json"

NEW_SCORES_FILE = RUN / "v4a_retrieval_diagnostic_novel_bm25_judge_scores.csv"
METADATA_FILE = RUN / "v4a_retrieval_diagnostic_novel_judging_metadata.json"
SCORING_LOCK_FILE = RUN / "v4a_retrieval_diagnostic_novel_judge_scoring.lock.json"
QUARANTINE_FILE = RUN / "v4a_retrieval_diagnostic_quarantined_cases.json"

EXPECTED_NOVEL_INPUT_SHA256 = "d9607305d0e0bd00d268bec34acdf2a4af0e23e22c27756ca27460fd97ff3c74"
EXPECTED_SCORE_INVENTORY_SHA256 = "250841f0bf9cefc5a461e09f3963842187bdac8bfdaafe03e7e07487d07c7e16"
EXPECTED_JUDGING_INPUT_LOCK_SHA256 = "da34d259dacd70348878c2c1e61fa9d702dd536b58df88ffc129a3fb4e6be4e2"
EXPECTED_HOLDOUT_SCORING_LOCK_SHA256 = "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777"
EXPECTED_SPAN_ECHO_PATCH_SHA256 = "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266"
SPAN_ECHO_PATCH_ID = "v3-posthoc-prominence-span-echo-normalization1"
TIMEOUT_PATCH_ID = "v4a-retrdiag-deepeval-per-attempt-timeout-900s1"
EXPECTED_TIMEOUT_PATCH_SHA256 = "b250e7448cb0ca53573dc088cb1e3f3c4a2c7372ee1848627483e53f5a7c8a40"
EXPECTED_PER_ATTEMPT_TIMEOUT_SECONDS = "900"
QUARANTINE_PATCH_ID = "v4a-retrdiag-case-timeout-quarantine1"
KNOWN_QUARANTINED_CASE = "V4A_RETRDIAG_H14_9781582972947"
EXPECTED_QUARANTINE_PATCH_SHA256 = "eabc05dd299cefd8c605d2db98427aecdf5bdb6013bda5a89fdac9be3b59d56c"
EXPECTED_NOVEL = 675
EXPECTED_TARGET_QUERIES = 17

AUDIT_COLUMNS = [
    "case_id", "query_id", "query", "stratum", "isbn13", "title", "authors",
    "candidate_source", "bm25_rank", "bm25_score", "hybrid_retrieval_rank",
    "has_relevant_evidence", "evidence_text", "matched_concept", "match_level",
    "judge_score", "judge_reason", "semantic_generation_mode",
    "semantic_stage_retry_count", "prominence_span_echo_normalization_count",
    "book_subject_summary", "book_subject_span_ids_json", "book_subject_reason",
    "book_subject_analysis_retry_count", "deterministic_direct_cue_count",
    "deterministic_cue_polarity_blocked_count", "composite_verification_attempt_count",
    "composite_verification_count", "hard_exclusion_precheck_attempt_count",
    "hard_exclusion_precheck_trigger_count", "core_facet_count", "incidental_core_count",
    "meaningful_core_count", "meaningful_core_coverage", "strong_core_count",
    "direct_core_count", "entailed_core_count", "adjacent_core_count",
    "unsupported_core_count", "qualifier_count", "satisfied_qualifier_count",
    "qualifier_cap_applied", "clear_rule_applied", "scoring_explanation",
    "facet_assessments_json", "facet_evidence_json", "component_evidence_ledger_json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def package_ver(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def canonical_optional_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return "" if text.strip().casefold() in {"", "nan", "none", "null", "<na>"} else text


def verify_frozen_text(label: str, path: Path, expected: str) -> dict[str, str]:
    raw = sha256(path)
    norm = lf_normalized_sha256(path)
    if raw == expected:
        mode = "raw_bytes"
    elif norm == expected:
        mode = "lf_normalized_line_endings"
    else:
        raise ValueError(f"{label} hash mismatch: expected={expected} raw={raw} lf={norm}")
    print(f"PASS  frozen {label} verified ({mode})")
    return {
        "frozen_sha256": expected,
        "current_raw_sha256": raw,
        "lf_normalized_sha256": norm,
        "verification_mode": mode,
    }


PATCH_STATS = {"normalized_entries": 0, "validator_calls_with_normalization": 0}
_ORIGINAL_PROMINENCE_VALIDATOR = judge_module._validate_prominence_result


def _normalize_exact_prominence_span_echoes(result: Any, spans: dict[str, str]) -> int:
    normalized: list[str] = []
    repaired = 0
    for value in result.supporting_span_ids:
        token = str(value).strip()
        if token in spans:
            normalized.append(token)
            continue
        matches = [sid for sid, text in spans.items() if token == f"{sid}: {text}".strip()]
        if len(matches) == 1:
            normalized.append(matches[0])
            repaired += 1
        else:
            normalized.append(str(value))
    if repaired:
        result.supporting_span_ids = normalized
    return repaired


def _patched_validate_prominence_result(result: Any, spans: dict[str, str], config: dict[str, Any]) -> None:
    repaired = _normalize_exact_prominence_span_echoes(result, spans)
    if repaired:
        PATCH_STATS["normalized_entries"] += repaired
        PATCH_STATS["validator_calls_with_normalization"] += 1
    _ORIGINAL_PROMINENCE_VALIDATOR(result=result, spans=spans, config=config)


def apply_span_echo_execution_patch() -> None:
    require(sha256(SPAN_ECHO_PATCH_FILE) == EXPECTED_SPAN_ECHO_PATCH_SHA256, "Span-echo patch record changed")
    patch = load_json(SPAN_ECHO_PATCH_FILE)
    expected = {
        "status": "APPLIED_NON_SEMANTIC_EXECUTION_PATCH",
        "patch_id": SPAN_ECHO_PATCH_ID,
        "judge_source_modified": False,
        "judge_prompt_modified": False,
        "semantic_rules_changed": False,
        "deterministic_scoring_rules_changed": False,
        "facet_spec_changed": False,
        "judge_config_changed": False,
        "rubric_changed": False,
        "release_metrics_or_gates_changed": False,
        "frozen_v3_release_decision_modified": False,
    }
    for key, value in expected.items():
        require(patch.get(key) == value, f"Span-echo patch contract changed: {key}")

    class Probe:
        def __init__(self, ids):
            self.supporting_span_ids = ids

    spans = {"S3": "alpha", "S4": "beta"}
    p = Probe(["S3: alpha", "S4: beta"])
    require(
        _normalize_exact_prominence_span_echoes(p, spans) == 2
        and p.supporting_span_ids == ["S3", "S4"],
        "Span-echo patch self-test failed",
    )
    p = Probe(["S3: ALPHA", "S99: alpha", "not-an-id"])
    require(_normalize_exact_prominence_span_echoes(p, spans) == 0, "Span-echo patch over-normalized")
    PATCH_STATS["normalized_entries"] = 0
    PATCH_STATS["validator_calls_with_normalization"] = 0
    judge_module._validate_prominence_result = _patched_validate_prominence_result


def verify_timeout_execution_patch() -> dict[str, Any]:
    require(TIMEOUT_PATCH_FILE.exists(), f"Missing timeout execution patch: {TIMEOUT_PATCH_FILE}")
    require(sha256(TIMEOUT_PATCH_FILE) == EXPECTED_TIMEOUT_PATCH_SHA256, "Timeout execution patch record changed")
    patch = load_json(TIMEOUT_PATCH_FILE)
    expected = {
        "status": "APPLIED_NON_SEMANTIC_EXECUTION_PATCH",
        "patch_id": TIMEOUT_PATCH_ID,
        "environment_variable": "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
        "patched_seconds": 900,
        "resume_only": True,
        "judge_source_modified": False,
        "judge_prompt_modified": False,
        "judge_model_modified": False,
        "judge_temperature_modified": False,
        "judge_retry_semantics_modified": False,
        "semantic_rules_changed": False,
        "deterministic_scoring_rules_changed": False,
        "facet_spec_changed": False,
        "judge_config_changed": False,
        "rubric_changed": False,
        "candidate_set_changed": False,
        "development_selection_reopened": False,
        "release_metrics_or_gates_changed": False,
        "release_decision_modified": False,
    }
    for key, value in expected.items():
        require(patch.get(key) == value, f"Timeout execution patch contract changed: {key}")
    actual_timeout = os.environ.get("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE")
    require(
        actual_timeout == EXPECTED_PER_ATTEMPT_TIMEOUT_SECONDS,
        "Frozen timeout execution patch requires "
        f"DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE={EXPECTED_PER_ATTEMPT_TIMEOUT_SECONDS}; "
        f"got {actual_timeout!r}",
    )
    print(
        "PASS  non-semantic DeepEval per-attempt timeout patch enabled: "
        f"{EXPECTED_PER_ATTEMPT_TIMEOUT_SECONDS}s"
    )
    return {
        "patch_id": TIMEOUT_PATCH_ID,
        "patch_sha256": EXPECTED_TIMEOUT_PATCH_SHA256,
        "per_attempt_timeout_seconds": int(EXPECTED_PER_ATTEMPT_TIMEOUT_SECONDS),
    }


def verify_quarantine_execution_patch() -> dict[str, Any]:
    require(QUARANTINE_PATCH_FILE.exists(), f"Missing quarantine execution patch: {QUARANTINE_PATCH_FILE}")
    require(sha256(QUARANTINE_PATCH_FILE) == EXPECTED_QUARANTINE_PATCH_SHA256, "Quarantine execution patch record changed")
    patch = load_json(QUARANTINE_PATCH_FILE)
    expected = {
        "status": "APPLIED_NON_SEMANTIC_EXECUTION_PATCH",
        "patch_id": QUARANTINE_PATCH_ID,
        "known_quarantined_case_id": KNOWN_QUARANTINED_CASE,
        "resume_only": True,
        "score_assigned_to_quarantined_case": False,
        "quarantined_case_excluded_from_final_analysis": False,
        "judge_source_modified": False,
        "judge_prompt_modified": False,
        "judge_model_modified": False,
        "judge_temperature_modified": False,
        "judge_retry_semantics_modified": False,
        "semantic_rules_changed": False,
        "deterministic_scoring_rules_changed": False,
        "facet_spec_changed": False,
        "judge_config_changed": False,
        "rubric_changed": False,
        "candidate_set_changed": False,
        "development_selection_reopened": False,
        "release_metrics_or_gates_changed": False,
        "release_decision_modified": False,
    }
    for key, value in expected.items():
        require(patch.get(key) == value, f"Quarantine execution patch contract changed: {key}")
    print(f"PASS  non-semantic case quarantine patch enabled: {KNOWN_QUARANTINED_CASE}")
    return patch


def exception_chain_contains_timeout(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError) or "timed out after" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def load_quarantine() -> list[dict[str, Any]]:
    if not QUARANTINE_FILE.exists():
        return []
    payload = load_json(QUARANTINE_FILE)
    require(isinstance(payload, list), "Quarantine file must contain a JSON list")
    return payload


def persist_quarantine(rows: list[dict[str, Any]]) -> None:
    QUARANTINE_FILE.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def matched_core_concepts(spec, verdict) -> str | None:
    matched = supported_core_facet_texts(spec=spec, verdict=verdict)
    return "; ".join(matched) if matched else None


def combined_evidence_text(spec, evidence_by_id: dict[str, str], score: int) -> str | None:
    if score == 0 or not evidence_by_id:
        return None
    parts = []
    for facet in spec.facets:
        evidence = evidence_by_id.get(facet.facet_id)
        if evidence:
            parts.append(f"{facet.facet_id}: {evidence}")
    return " | ".join(parts) if parts else None


def verify_inputs() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    checks = [
        (NOVEL_INPUT, EXPECTED_NOVEL_INPUT_SHA256, "novel BM25 diagnostic judging input"),
        (SCORE_INVENTORY, EXPECTED_SCORE_INVENTORY_SHA256, "diagnostic score inventory"),
        (JUDGING_INPUT_LOCK, EXPECTED_JUDGING_INPUT_LOCK_SHA256, "diagnostic judging-input lock"),
        (HOLDOUT_SCORING_LOCK, EXPECTED_HOLDOUT_SCORING_LOCK_SHA256, "v3 holdout scoring lock"),
        (SPAN_ECHO_PATCH_FILE, EXPECTED_SPAN_ECHO_PATCH_SHA256, "span-echo patch record"),
    ]
    for path, expected, label in checks:
        require(path.exists(), f"Missing frozen input: {path}")
        require(sha256(path) == expected, f"{label} hash mismatch")
        print(f"PASS  {label} SHA-256 verified")

    input_lock = load_json(JUDGING_INPUT_LOCK)
    require(
        input_lock.get("status") == "V4A_RETRIEVAL_DIAGNOSTIC_INPUT_FROZEN_BEFORE_JUDGING",
        "Unexpected retrieval-diagnostic input-lock status",
    )
    require(int(input_lock.get("target_query_count", -1)) == EXPECTED_TARGET_QUERIES, "Target-query count changed")
    require(int(input_lock.get("diagnostic_union_rows", -1)) == 1576, "Diagnostic union row count changed")
    require(int(input_lock.get("dense_branch_rows", -1)) == 850, "Dense branch row count changed")
    require(int(input_lock.get("bm25_only_rows", -1)) == 726, "BM25-only row count changed")
    require(int(input_lock.get("reused_scores", -1)) == 901, "Reused score count changed")
    require(
        int(input_lock.get("novel_bm25_only_pairs_to_judge", -1)) == EXPECTED_NOVEL,
        "Novel BM25 diagnostic count changed",
    )
    require(input_lock.get("score_inventory_sha256") == EXPECTED_SCORE_INVENTORY_SHA256, "Input-lock inventory hash mismatch")
    require(input_lock.get("novel_bm25_judging_input_sha256") == EXPECTED_NOVEL_INPUT_SHA256, "Input-lock novel hash mismatch")
    require(input_lock.get("judge_execution_performed") is False, "Input lock says judge execution already occurred")
    require(input_lock.get("model_execution_performed") is False, "Input lock says model execution already occurred")
    require(input_lock.get("human_relevance_labels_used") is False, "Input lock says human labels were used")
    require(input_lock.get("development_selection_reopened") is False, "Input lock reopens frozen development selection")
    require(input_lock.get("release_metrics_calculated") is False, "Input lock says release metrics were calculated")
    require(input_lock.get("release_decision_produced") is False, "Input lock says release decision was produced")

    holdout_lock = load_json(HOLDOUT_SCORING_LOCK)
    require(
        holdout_lock.get("status") == "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS",
        "Unexpected frozen holdout scoring status",
    )
    require(holdout_lock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5", "Judge candidate changed")

    runtime_provenance: dict[str, Any] = {}
    for label, path, key in [
        ("judge source", JUDGE_FILE, "judge_source_sha256"),
        ("deterministic scoring", SCORING_FILE, "scoring_source_sha256"),
        ("judge config", JUDGE_CONFIG_FILE, "judge_config_sha256"),
        ("rubric", RUBRIC_FILE, "rubric_sha256"),
    ]:
        expected = str(holdout_lock.get(key, ""))
        require(expected, f"Holdout scoring lock missing {key}")
        runtime_provenance[key] = verify_frozen_text(label, path, expected)

    expected_facet_sha = str(holdout_lock.get("holdout_facet_spec_sha256", ""))
    require(expected_facet_sha, "Holdout scoring lock missing holdout_facet_spec_sha256")
    runtime_provenance["holdout_facet_spec_sha256"] = verify_frozen_text(
        "holdout facet spec", HOLDOUT_FACET_FILE, expected_facet_sha
    )

    config = load_json(JUDGE_CONFIG_FILE)
    require(config.get("version") == "0.29.0-r5", "Unexpected judge config version")
    require(float(config.get("temperature", -1)) == 0.0, "Judge temperature changed")

    facet_specs = load_facet_specs(load_json(HOLDOUT_FACET_FILE))
    require(len(facet_specs) == 24, "Expected 24 frozen H01-H24 facet specs")

    novel = pd.read_csv(NOVEL_INPUT, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "case_id", "query_id", "query", "stratum", "isbn13", "title", "description", "authors",
        "candidate_source", "bm25_rank", "bm25_score", "hybrid_retrieval_rank",
    }
    missing = required - set(novel.columns)
    require(not missing, f"Novel diagnostic input missing columns: {sorted(missing)}")
    require(len(novel) == EXPECTED_NOVEL, f"Expected {EXPECTED_NOVEL} novel BM25-only pairs")
    require(not novel["case_id"].duplicated().any(), "Duplicate novel diagnostic case IDs")
    require(not novel.duplicated(["query_id", "isbn13"]).any(), "Duplicate novel diagnostic query-book pairs")
    require(novel["candidate_source"].eq("bm25_only").all(), "Novel diagnostic input contains non-BM25-only pair")
    require(novel["case_id"].str.startswith("V4A_RETRDIAG_").all(), "Unexpected diagnostic case-id prefix")
    bm25_rank = pd.to_numeric(novel["bm25_rank"], errors="raise").astype(int)
    require(bm25_rank.between(1, 50).all(), "Novel diagnostic BM25 rank outside 1..50")
    hybrid_rank = pd.to_numeric(novel["hybrid_retrieval_rank"], errors="raise").astype(int)
    require(hybrid_rank.between(1, 100).all(), "Novel diagnostic hybrid retrieval rank outside 1..100")
    pd.to_numeric(novel["bm25_score"], errors="raise")
    require(
        novel["query_id"].nunique() <= EXPECTED_TARGET_QUERIES
        and novel["query_id"].str.match(r"H(?:0[1-9]|1[0-9]|2[0-4])$").all(),
        "Unexpected query IDs in novel diagnostic input",
    )
    novel["rank"] = bm25_rank

    for qid, group in novel.groupby("query_id", sort=False):
        require(qid in facet_specs, f"Unknown query ID: {qid}")
        require(set(group["query"].astype(str)) == {facet_specs[qid].query}, f"{qid}: query differs from frozen facet spec")

    return novel, config, runtime_provenance


def persist(rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows, columns=AUDIT_COLUMNS).to_csv(NEW_SCORES_FILE, index=False, encoding="utf-8")


def load_existing(novel: pd.DataFrame) -> list[dict[str, Any]]:
    if not NEW_SCORES_FILE.exists():
        return []
    frame = pd.read_csv(NEW_SCORES_FILE, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = set(AUDIT_COLUMNS) - set(frame.columns)
    require(not missing, f"Existing partial score file missing columns: {sorted(missing)}")
    require(not frame["case_id"].duplicated().any(), "Duplicate persisted case IDs")
    valid = novel.set_index("case_id", drop=False)
    for row in frame.to_dict(orient="records"):
        cid = str(row["case_id"])
        require(cid in valid.index, f"Persisted case outside frozen input: {cid}")
        frozen = valid.loc[cid]
        for field in ["query_id", "query", "isbn13", "title", "candidate_source", "bm25_rank", "hybrid_retrieval_rank"]:
            require(str(row[field]) == str(frozen[field]), f"Persisted case {cid} changed {field}")
        require(0 <= int(row["judge_score"]) <= 4, f"Persisted score outside 0..4 for {cid}")
    require(len(frame) <= EXPECTED_NOVEL, "Persisted score file has too many rows")
    return frame[AUDIT_COLUMNS].to_dict(orient="records")


def write_metadata(*, config: dict[str, Any], runtime_provenance: dict[str, Any], completed: int, status: str, error: str | None = None, quarantined: int = 0) -> None:
    payload = {
        "evaluation_id": "semantic-relevance-v4a-retrieval-diagnostic",
        "stage": "NOVEL_BM25_ONLY_DIAGNOSTIC_JUDGING",
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_role": "explanatory_only",
        "target_query_count": EXPECTED_TARGET_QUERIES,
        "judging_input_lock_sha256": EXPECTED_JUDGING_INPUT_LOCK_SHA256,
        "novel_bm25_judging_input_sha256": EXPECTED_NOVEL_INPUT_SHA256,
        "score_inventory_sha256": EXPECTED_SCORE_INVENTORY_SHA256,
        "holdout_judge_scoring_lock_sha256": EXPECTED_HOLDOUT_SCORING_LOCK_SHA256,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "judge_model": config["model"],
        "judge_base_url": config["base_url"],
        "temperature": config["temperature"],
        "runtime_provenance": runtime_provenance,
        "execution_patch_id": SPAN_ECHO_PATCH_ID,
        "execution_patch_sha256": EXPECTED_SPAN_ECHO_PATCH_SHA256,
        "timeout_execution_patch_id": TIMEOUT_PATCH_ID,
        "timeout_execution_patch_sha256": EXPECTED_TIMEOUT_PATCH_SHA256,
        "deepeval_per_attempt_timeout_seconds": int(EXPECTED_PER_ATTEMPT_TIMEOUT_SECONDS),
        "prominence_span_echo_normalized_entries_this_process": int(PATCH_STATS["normalized_entries"]),
        "prominence_validator_calls_with_normalization_this_process": int(PATCH_STATS["validator_calls_with_normalization"]),
        "novel_pairs": EXPECTED_NOVEL,
        "completed_novel_pairs": completed,
        "remaining_novel_pairs": EXPECTED_NOVEL - completed,
        "quarantined_unscored_pairs": quarantined,
        "quarantine_execution_patch_id": QUARANTINE_PATCH_ID,
        "quarantine_execution_patch_sha256": EXPECTED_QUARANTINE_PATCH_SHA256,
        "human_relevance_labels_used": False,
        "development_selection_reopened": False,
        "release_metrics_calculated": False,
        "release_gates_calculated": False,
        "release_decision_produced": False,
        "python_version": platform.python_version(),
        "deepeval_version": package_ver("deepeval"),
        "pandas_version": package_ver("pandas"),
        "error": error,
    }
    METADATA_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume persisted cases after a pure execution failure")
    args = parser.parse_args()

    print("Semantic relevance v4a - 17-query retrieval diagnostic novel BM25-only judging")
    print("----------------------------------------------------------------------------")
    print(f"Frozen novel BM25-only pairs: {EXPECTED_NOVEL}")
    print("Diagnostic role: EXPLANATORY ONLY")
    print("Human relevance labels: NOT USED")
    print("Development selection: FROZEN / NOT REOPENED")
    print("Release metrics/gates: NOT CALCULATED")
    print()

    novel, config, runtime_provenance = verify_inputs()
    apply_span_echo_execution_patch()
    print("PASS  non-semantic prominence span-echo normalization patch enabled")
    facet_specs = load_facet_specs(load_json(HOLDOUT_FACET_FILE))
    rubric = load_json(RUBRIC_FILE)

    results = load_existing(novel)
    if results and not args.resume:
        raise RuntimeError(
            "Persisted retrieval-diagnostic novel scores already exist. "
            "Use --resume only after a pure execution failure."
        )
    require(args.resume, "Timeout-patched r2 package is resume-only; pass --resume.")
    require(results, "Timeout-patched r2 package requires persisted partial scores from the prior execution failure.")
    timeout_patch = verify_timeout_execution_patch()
    verify_quarantine_execution_patch()

    completed_ids = {str(r["case_id"]) for r in results}
    frozen_ids = set(novel["case_id"].astype(str))
    require(completed_ids <= frozen_ids, "Persisted cases fall outside frozen novel diagnostic input")
    remaining = novel[~novel["case_id"].isin(completed_ids)].copy()
    require(KNOWN_QUARANTINED_CASE in set(remaining["case_id"].astype(str)), "Known quarantine case is not the next unresolved frozen input")

    quarantine = load_quarantine()
    quarantined_ids = {str(r.get("case_id", "")) for r in quarantine}
    require(not (quarantined_ids & completed_ids), "A quarantined case already has a persisted score")
    require(quarantined_ids <= frozen_ids, "Quarantine file contains case outside frozen input")
    if KNOWN_QUARANTINED_CASE not in quarantined_ids:
        qrow = novel.loc[novel["case_id"].astype(str) == KNOWN_QUARANTINED_CASE].iloc[0]
        quarantine.append({
            "case_id": KNOWN_QUARANTINED_CASE,
            "query_id": str(qrow["query_id"]),
            "isbn13": str(qrow["isbn13"]),
            "status": "QUARANTINED_UNSCORED",
            "reason": "Repeated core-prominence execution timeout on the same frozen case: 360s attempts and 900s attempt exhausted; no semantic score assigned.",
            "patch_id": QUARANTINE_PATCH_ID,
        })
        persist_quarantine(quarantine)
        quarantined_ids.add(KNOWN_QUARANTINED_CASE)
        print(f"QUARANTINED without scoring: {KNOWN_QUARANTINED_CASE} (previous repeated timeout evidence)")

    remaining = remaining[~remaining["case_id"].isin(quarantined_ids)].copy()

    print(f"Starting/resuming frozen judge run: {len(completed_ids)}/{EXPECTED_NOVEL} persisted; {len(quarantined_ids)} quarantined/unscored")
    print("Do not inspect or tune semantic results until all 675 novel diagnostic pairs are frozen.")

    judge_model = None
    if not remaining.empty:
        judge_model = OllamaModel(
            model=config["model"],
            base_url=config["base_url"],
            temperature=config["temperature"],
        )

    write_metadata(
        config=config,
        runtime_provenance=runtime_provenance,
        completed=len(completed_ids),
        status="RUNNING" if not remaining.empty else "PARTIAL_QUARANTINED",
        quarantined=len(quarantined_ids),
    )

    try:
        for _, row in remaining.iterrows():
            cid = str(row["case_id"])
            ordinal = len(completed_ids) + 1
            print(f"Processing: {ordinal}/{EXPECTED_NOVEL} case_id={cid}", flush=True)
            spec = facet_specs[str(row["query_id"])]
            before = int(PATCH_STATS["normalized_entries"])
            try:
                semantic_result = generate_validated_semantic_verdict(
                    judge_model=judge_model,
                    row=row,
                    spec=spec,
                    rubric=rubric,
                    config=config,
                )
            except Exception as case_exc:
                if not exception_chain_contains_timeout(case_exc):
                    raise
                quarantine.append({
                    "case_id": cid,
                    "query_id": str(row["query_id"]),
                    "isbn13": str(row["isbn13"]),
                    "status": "QUARANTINED_UNSCORED",
                    "reason": f"Pure execution timeout under frozen 900s per-attempt patch: {type(case_exc).__name__}: {case_exc}",
                    "patch_id": QUARANTINE_PATCH_ID,
                })
                persist_quarantine(quarantine)
                quarantined_ids.add(cid)
                print(f"QUARANTINED without scoring: {cid}; continuing with remaining frozen cases", flush=True)
                write_metadata(
                    config=config,
                    runtime_provenance=runtime_provenance,
                    completed=len(completed_ids),
                    status="RUNNING_WITH_QUARANTINE",
                    quarantined=len(quarantined_ids),
                )
                continue
            verdict = semantic_result.verdict
            score_result = compute_facet_score(spec=spec, verdict=verdict)
            positive_evidence = supported_evidence_by_id(
                spec=spec,
                verdict=verdict,
                evidence_by_id=semantic_result.evidence_by_id,
            )
            has_relevant = score_result.score > 0

            results.append({
                "case_id": cid,
                "query_id": str(row["query_id"]),
                "query": str(row["query"]),
                "stratum": str(row["stratum"]),
                "isbn13": str(row["isbn13"]),
                "title": str(row["title"]),
                "authors": canonical_optional_text(row.get("authors", "")),
                "candidate_source": str(row["candidate_source"]),
                "bm25_rank": int(row["bm25_rank"]),
                "bm25_score": float(row["bm25_score"]),
                "hybrid_retrieval_rank": int(row["hybrid_retrieval_rank"]),
                "has_relevant_evidence": has_relevant,
                "evidence_text": combined_evidence_text(spec, positive_evidence, score_result.score),
                "matched_concept": matched_core_concepts(spec, verdict) if has_relevant else None,
                "match_level": score_result.label,
                "judge_score": score_result.score,
                "judge_reason": verdict.overall_reason,
                "semantic_generation_mode": semantic_result.generation_mode,
                "semantic_stage_retry_count": semantic_result.stage_retry_count,
                "prominence_span_echo_normalization_count": int(PATCH_STATS["normalized_entries"]) - before,
                "book_subject_summary": semantic_result.book_subject_analysis.primary_subject_summary,
                "book_subject_span_ids_json": json.dumps(
                    semantic_result.book_subject_analysis.primary_subject_span_ids,
                    ensure_ascii=False,
                ),
                "book_subject_reason": semantic_result.book_subject_analysis.reason,
                "book_subject_analysis_retry_count": semantic_result.subject_analysis_retry_count,
                "deterministic_direct_cue_count": semantic_result.deterministic_direct_cue_count,
                "deterministic_cue_polarity_blocked_count": semantic_result.deterministic_cue_polarity_blocked_count,
                "composite_verification_attempt_count": semantic_result.composite_verification_attempt_count,
                "composite_verification_count": semantic_result.composite_verification_count,
                "hard_exclusion_precheck_attempt_count": semantic_result.hard_exclusion_precheck_attempt_count,
                "hard_exclusion_precheck_trigger_count": semantic_result.hard_exclusion_precheck_trigger_count,
                "core_facet_count": score_result.core_facet_count,
                "incidental_core_count": score_result.incidental_core_count,
                "meaningful_core_count": score_result.meaningful_core_count,
                "meaningful_core_coverage": score_result.meaningful_core_coverage,
                "strong_core_count": score_result.strong_core_count,
                "direct_core_count": score_result.direct_core_count,
                "entailed_core_count": score_result.entailed_core_count,
                "adjacent_core_count": score_result.adjacent_core_count,
                "unsupported_core_count": score_result.unsupported_core_count,
                "qualifier_count": score_result.qualifier_count,
                "satisfied_qualifier_count": score_result.satisfied_qualifier_count,
                "qualifier_cap_applied": score_result.qualifier_cap_applied,
                "clear_rule_applied": score_result.clear_rule_applied,
                "scoring_explanation": score_result.explanation,
                "facet_assessments_json": serialize_facet_assessments(spec=spec, verdict=verdict),
                "facet_evidence_json": serialize_facet_evidence(semantic_result.candidate_evidence_by_id),
                "component_evidence_ledger_json": json.dumps(
                    semantic_result.component_evidence_ledger_by_id,
                    ensure_ascii=False,
                ),
            })
            completed_ids.add(cid)
            persist(results)
            if len(completed_ids) % 25 == 0 or len(completed_ids) == EXPECTED_NOVEL:
                print(f"Checkpoint: {len(completed_ids)}/{EXPECTED_NOVEL} persisted")
            write_metadata(
                config=config,
                runtime_provenance=runtime_provenance,
                completed=len(completed_ids),
                status="RUNNING",
                quarantined=len(quarantined_ids),
            )
    except Exception as exc:
        persist(results)
        write_metadata(
            config=config,
            runtime_provenance=runtime_provenance,
            completed=len(completed_ids),
            status="PARTIAL_EXECUTION_FAILURE",
            error=f"{type(exc).__name__}: {exc}",
            quarantined=len(quarantined_ids),
        )
        raise

    if len(results) + len(quarantined_ids) == EXPECTED_NOVEL and quarantined_ids:
        persist(results)
        persist_quarantine(quarantine)
        write_metadata(
            config=config,
            runtime_provenance=runtime_provenance,
            completed=len(results),
            status="PARTIAL_QUARANTINED_UNSCORED",
            quarantined=len(quarantined_ids),
        )
        print()
        print("V4A 17-QUERY RETRIEVAL DIAGNOSTIC NOVEL JUDGING: PARTIAL / QUARANTINED")
        print(f"Novel BM25-only scores persisted: {len(results)}/{EXPECTED_NOVEL}")
        print(f"Quarantined unscored cases: {len(quarantined_ids)}")
        for q in quarantine:
            print(f"  - {q['case_id']}: {q['reason']}")
        print("Scoring lock created: NO")
        print("Development selection reopened: NO")
        print("Release metrics calculated: NO")
        print("Next: resolve quarantined case(s) under a separately reviewed execution package before final diagnostic analysis.")
        return

    require(len(results) == EXPECTED_NOVEL, "Retrieval-diagnostic judging did not complete 675/675")
    scores = pd.DataFrame(results, columns=AUDIT_COLUMNS)
    score_values = pd.to_numeric(scores["judge_score"], errors="raise").astype(int)
    require(score_values.between(0, 4).all(), "Diagnostic score outside 0..4")
    patch_counts = pd.to_numeric(scores["prominence_span_echo_normalization_count"], errors="raise").astype(int)
    require((patch_counts >= 0).all(), "Negative span-echo normalization count")

    lock = {
        "status": "V4A_RETRIEVAL_DIAGNOSTIC_NOVEL_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_role": "explanatory_only",
        "target_query_count": EXPECTED_TARGET_QUERIES,
        "judging_input_lock_sha256": EXPECTED_JUDGING_INPUT_LOCK_SHA256,
        "novel_bm25_judging_input_sha256": EXPECTED_NOVEL_INPUT_SHA256,
        "score_inventory_sha256": EXPECTED_SCORE_INVENTORY_SHA256,
        "holdout_judge_scoring_lock_sha256": EXPECTED_HOLDOUT_SCORING_LOCK_SHA256,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "execution_patch_id": SPAN_ECHO_PATCH_ID,
        "execution_patch_sha256": EXPECTED_SPAN_ECHO_PATCH_SHA256,
        "timeout_execution_patch_id": TIMEOUT_PATCH_ID,
        "timeout_execution_patch_sha256": EXPECTED_TIMEOUT_PATCH_SHA256,
        "deepeval_per_attempt_timeout_seconds": int(EXPECTED_PER_ATTEMPT_TIMEOUT_SECONDS),
        "holdout_facet_spec_sha256": load_json(HOLDOUT_SCORING_LOCK)["holdout_facet_spec_sha256"],
        "judge_source_sha256": load_json(HOLDOUT_SCORING_LOCK)["judge_source_sha256"],
        "scoring_source_sha256": load_json(HOLDOUT_SCORING_LOCK)["scoring_source_sha256"],
        "judge_config_sha256": load_json(HOLDOUT_SCORING_LOCK)["judge_config_sha256"],
        "rubric_sha256": load_json(HOLDOUT_SCORING_LOCK)["rubric_sha256"],
        "novel_pairs": EXPECTED_NOVEL,
        "novel_judge_scores_sha256": sha256(NEW_SCORES_FILE),
        "prominence_span_echo_normalized_cases": int((patch_counts > 0).sum()),
        "prominence_span_echo_normalized_entries": int(patch_counts.sum()),
        "human_relevance_labels_used": False,
        "development_selection_reopened": False,
        "release_metrics_calculated": False,
        "release_gates_calculated": False,
        "release_decision_produced": False,
        "next_stage": "FINALIZE_V4A_17_QUERY_HYBRID_UNION_SCORE_MAP_AND_RUN_RETRIEVAL_DIAGNOSTIC",
    }
    if SCORING_LOCK_FILE.exists():
        existing = load_json(SCORING_LOCK_FILE)
        for key, value in lock.items():
            if key != "frozen_at_utc":
                require(existing.get(key) == value, f"Existing diagnostic scoring lock mismatch: {key}")
    else:
        SCORING_LOCK_FILE.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    write_metadata(
        config=config,
        runtime_provenance=runtime_provenance,
        completed=EXPECTED_NOVEL,
        status="COMPLETE",
        quarantined=0,
    )
    print()
    print("V4A 17-QUERY RETRIEVAL DIAGNOSTIC NOVEL JUDGING: PASS")
    print(f"Novel BM25-only pairs judged: {EXPECTED_NOVEL}/{EXPECTED_NOVEL}")
    print(f"Novel score SHA-256: {sha256(NEW_SCORES_FILE)}")
    print(f"Scoring lock SHA-256: {sha256(SCORING_LOCK_FILE)}")
    print(f"Prominence span-echo normalized cases: {int((patch_counts > 0).sum())}")
    print(f"Prominence span-echo normalized entries: {int(patch_counts.sum())}")
    print("Human relevance labels used: NO")
    print("Development selection reopened: NO")
    print("Release metrics calculated: NO")
    print("Release decision produced: NO")
    print("Next: deterministic finalization of the complete 17-query hybrid-union diagnostic score map.")


if __name__ == "__main__":
    main()
