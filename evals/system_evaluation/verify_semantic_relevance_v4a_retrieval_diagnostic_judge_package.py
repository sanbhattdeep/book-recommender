from __future__ import annotations

import hashlib
import json
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
HOLDOUT_SCORING_LOCK = HOLDOUT_RUN / "holdout_judge_scoring_lock.json"
HOLDOUT_FACET_FILE = EVALS / "facets/semantic_relevance/semantic_relevance_v3_holdout_query_facets.v0.1.0.json"
JUDGE_FILE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS / "semantic_relevance_facet_scoring.py"
JUDGE_CONFIG_FILE = EVALS / "judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
RUBRIC_FILE = EVALS / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
SPAN_ECHO_PATCH_FILE = EVALS / "system_evaluation/semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"
TIMEOUT_PATCH_FILE = REPO_ROOT / "v4a_retrieval_diagnostic_deepeval_timeout_patch1.json"
QUARANTINE_PATCH_FILE = REPO_ROOT / "v4a_retrieval_diagnostic_case_quarantine_patch1.json"
PARTIAL_SCORES = RUN / "v4a_retrieval_diagnostic_novel_bm25_judge_scores.csv"
RUNNER = EVALS / "system_evaluation/score_semantic_relevance_v4a_retrieval_diagnostic_novel_pairs.py"

EXPECTED = {
    "novel_input": "d9607305d0e0bd00d268bec34acdf2a4af0e23e22c27756ca27460fd97ff3c74",
    "score_inventory": "250841f0bf9cefc5a461e09f3963842187bdac8bfdaafe03e7e07487d07c7e16",
    "judging_input_lock": "da34d259dacd70348878c2c1e61fa9d702dd536b58df88ffc129a3fb4e6be4e2",
    "holdout_scoring_lock": "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777",
    "span_echo_patch": "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266",
    "timeout_patch": "b250e7448cb0ca53573dc088cb1e3f3c4a2c7372ee1848627483e53f5a7c8a40",
    "quarantine_patch": "eabc05dd299cefd8c605d2db98427aecdf5bdb6013bda5a89fdac9be3b59d56c",
    "runner": "485b15818ada565a4ddea38fd0ae9dafc127d3f7913823479d999a4346231480",
}
EXPECTED_NOVEL = 675
EXPECTED_PERSISTED = 402
KNOWN_QUARANTINED_CASE = "V4A_RETRDIAG_H14_9781582972947"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_sha256(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_text(label: str, path: Path, expected: str) -> None:
    raw = sha256(path)
    norm = lf_sha256(path)
    req(raw == expected or norm == expected, f"{label} hash mismatch: expected={expected} raw={raw} lf={norm}")
    mode = "raw" if raw == expected else "canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")


def main() -> None:
    print("Semantic relevance v4a - 17-query retrieval diagnostic judge package preflight")
    print("----------------------------------------------------------------------------")
    print("Judge execution:          DISABLED")
    print("Human labels:             DISABLED")
    print("Development selection:    FROZEN / NOT REOPENED")
    print("Release analysis:         DISABLED")
    print()

    for label, path, key in [
        ("novel BM25 diagnostic judging input", NOVEL_INPUT, "novel_input"),
        ("diagnostic score inventory", SCORE_INVENTORY, "score_inventory"),
        ("diagnostic judging-input lock", JUDGING_INPUT_LOCK, "judging_input_lock"),
        ("v3 holdout scoring lock", HOLDOUT_SCORING_LOCK, "holdout_scoring_lock"),
        ("span-echo execution patch", SPAN_ECHO_PATCH_FILE, "span_echo_patch"),
        ("DeepEval timeout execution patch", TIMEOUT_PATCH_FILE, "timeout_patch"),
        ("case quarantine execution patch", QUARANTINE_PATCH_FILE, "quarantine_patch"),
        ("retrieval-diagnostic judge runner", RUNNER, "runner"),
    ]:
        req(path.exists(), f"Missing {label}: {path}")
        actual = sha256(path)
        req(actual == EXPECTED[key], f"{label} hash mismatch: expected={EXPECTED[key]} actual={actual}")
        print(f"PASS  {label} SHA-256 verified")

    lock = load_json(JUDGING_INPUT_LOCK)
    req(lock.get("status") == "V4A_RETRIEVAL_DIAGNOSTIC_INPUT_FROZEN_BEFORE_JUDGING", "Unexpected diagnostic input-lock status")
    req(lock.get("diagnostic_role") == "explanatory_only", "Diagnostic authority boundary changed")
    req(int(lock.get("target_query_count", -1)) == 17, "Target-query count changed")
    req(int(lock.get("diagnostic_union_rows", -1)) == 1576, "Diagnostic union row count changed")
    req(int(lock.get("dense_branch_rows", -1)) == 850, "Dense branch row count changed")
    req(int(lock.get("bm25_only_rows", -1)) == 726, "BM25-only row count changed")
    req(int(lock.get("reused_scores", -1)) == 901, "Reused score count changed")
    req(int(lock.get("novel_bm25_only_pairs_to_judge", -1)) == EXPECTED_NOVEL, "Novel diagnostic count changed")
    req(lock.get("judge_execution_performed") is False, "Input lock says judging already occurred")
    req(lock.get("model_execution_performed") is False, "Input lock says model execution already occurred")
    req(lock.get("human_relevance_labels_used") is False, "Input lock says human labels were used")
    req(lock.get("development_selection_reopened") is False, "Frozen development selection was reopened")
    req(lock.get("release_metrics_calculated") is False, "Input lock says release metrics were calculated")
    req(lock.get("release_decision_produced") is False, "Input lock says a release decision was produced")

    hlock = load_json(HOLDOUT_SCORING_LOCK)
    req(hlock.get("status") == "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS", "Unexpected v3 holdout scoring lock status")
    req(hlock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5", "Judge candidate changed")
    req(hlock.get("human_relevance_labels_used") is False, "Frozen v3 judge runtime used human labels")

    for label, path, key in [
        ("frozen judge source", JUDGE_FILE, "judge_source_sha256"),
        ("frozen deterministic scoring", SCORING_FILE, "scoring_source_sha256"),
        ("frozen judge config", JUDGE_CONFIG_FILE, "judge_config_sha256"),
        ("frozen rubric", RUBRIC_FILE, "rubric_sha256"),
    ]:
        expected = str(hlock.get(key, ""))
        req(expected, f"Holdout scoring lock missing {key}")
        verify_text(label, path, expected)

    expected_facet_sha = str(hlock.get("holdout_facet_spec_sha256", ""))
    req(expected_facet_sha, "Holdout scoring lock missing holdout_facet_spec_sha256")
    verify_text("frozen H01-H24 facet spec", HOLDOUT_FACET_FILE, expected_facet_sha)

    timeout_patch = load_json(TIMEOUT_PATCH_FILE)
    req(timeout_patch.get("status") == "APPLIED_NON_SEMANTIC_EXECUTION_PATCH", "Unexpected timeout patch status")
    req(timeout_patch.get("patch_id") == "v4a-retrdiag-deepeval-per-attempt-timeout-900s1", "Unexpected timeout patch ID")
    req(timeout_patch.get("patched_seconds") == 900, "Timeout patch seconds changed")
    req(timeout_patch.get("resume_only") is True, "Timeout patch must remain resume-only")
    for key in [
        "judge_source_modified", "judge_prompt_modified", "judge_model_modified",
        "judge_temperature_modified", "judge_retry_semantics_modified",
        "semantic_rules_changed", "deterministic_scoring_rules_changed",
        "facet_spec_changed", "judge_config_changed", "rubric_changed",
        "candidate_set_changed", "development_selection_reopened",
        "release_metrics_or_gates_changed", "release_decision_modified",
    ]:
        req(timeout_patch.get(key) is False, f"Timeout patch changed forbidden contract: {key}")
    print("PASS  timeout patch contract verified: 900s / resume-only / non-semantic")

    qpatch = load_json(QUARANTINE_PATCH_FILE)
    req(qpatch.get("status") == "APPLIED_NON_SEMANTIC_EXECUTION_PATCH", "Unexpected quarantine patch status")
    req(qpatch.get("patch_id") == "v4a-retrdiag-case-timeout-quarantine1", "Unexpected quarantine patch ID")
    req(qpatch.get("known_quarantined_case_id") == KNOWN_QUARANTINED_CASE, "Known quarantine case changed")
    req(qpatch.get("observed_persisted_scores_before_patch") == EXPECTED_PERSISTED, "Persisted-count provenance changed")
    req(qpatch.get("resume_only") is True, "Quarantine patch must remain resume-only")
    req(qpatch.get("score_assigned_to_quarantined_case") is False, "Quarantine patch must not assign a score")
    req(qpatch.get("quarantined_case_excluded_from_final_analysis") is False, "Quarantined case must remain required before final analysis")
    req(qpatch.get("final_analysis_allowed_with_unresolved_quarantine") is False, "Final analysis must remain blocked while quarantine is unresolved")
    for key in [
        "judge_source_modified", "judge_prompt_modified", "judge_model_modified",
        "judge_temperature_modified", "judge_retry_semantics_modified",
        "semantic_rules_changed", "deterministic_scoring_rules_changed",
        "facet_spec_changed", "judge_config_changed", "rubric_changed",
        "candidate_set_changed", "development_selection_reopened",
        "release_metrics_or_gates_changed", "release_decision_modified",
    ]:
        req(qpatch.get(key) is False, f"Quarantine patch changed forbidden contract: {key}")
    print(f"PASS  quarantine patch contract verified: {KNOWN_QUARANTINED_CASE} remains unscored and required")

    patch = load_json(SPAN_ECHO_PATCH_FILE)
    req(patch.get("status") == "APPLIED_NON_SEMANTIC_EXECUTION_PATCH", "Unexpected span-echo patch status")
    req(patch.get("patch_id") == "v3-posthoc-prominence-span-echo-normalization1", "Unexpected span-echo patch ID")
    req(patch.get("semantic_rules_changed") is False, "Span-echo patch changed semantic rules")
    req(patch.get("deterministic_scoring_rules_changed") is False, "Span-echo patch changed scoring rules")

    frame = pd.read_csv(NOVEL_INPUT, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "case_id", "query_id", "query", "stratum", "isbn13", "title", "description", "authors",
        "candidate_source", "bm25_rank", "bm25_score", "hybrid_retrieval_rank",
    }
    missing = required - set(frame.columns)
    req(not missing, f"Novel diagnostic input missing columns: {sorted(missing)}")
    req(len(frame) == EXPECTED_NOVEL, f"Expected {EXPECTED_NOVEL} frozen novel BM25-only pairs, got {len(frame)}")
    req(not frame["case_id"].duplicated().any(), "Duplicate novel diagnostic case IDs")
    req(not frame.duplicated(["query_id", "isbn13"]).any(), "Duplicate novel diagnostic query-book pairs")
    req(frame["candidate_source"].eq("bm25_only").all(), "Novel diagnostic input includes a non-BM25-only pair")
    req(frame["case_id"].str.startswith("V4A_RETRDIAG_").all(), "Unexpected diagnostic case-id prefix")
    bm25 = pd.to_numeric(frame["bm25_rank"], errors="raise").astype(int)
    hybrid = pd.to_numeric(frame["hybrid_retrieval_rank"], errors="raise").astype(int)
    req(bm25.between(1, 50).all(), "Novel diagnostic BM25 rank outside 1..50")
    req(hybrid.between(1, 100).all(), "Novel diagnostic hybrid rank outside 1..100")
    pd.to_numeric(frame["bm25_score"], errors="raise")
    req(frame["query_id"].nunique() <= 17, "Novel diagnostic input contains more than 17 queries")
    req(frame["query_id"].str.match(r"H(?:0[1-9]|1[0-9]|2[0-4])$").all(), "Unexpected query IDs in novel input")
    print(f"PASS  frozen diagnostic judging structure verified: {len(frame)} pairs across {frame['query_id'].nunique()} queries")

    req(PARTIAL_SCORES.exists(), "Expected persisted partial score file is missing; r2 is resume-only")
    partial = pd.read_csv(PARTIAL_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    req("case_id" in partial.columns and "judge_score" in partial.columns, "Partial score file missing required columns")
    req(not partial["case_id"].duplicated().any(), "Duplicate case IDs in partial score file")
    req(set(partial["case_id"]) <= set(frame["case_id"]), "Partial score file contains case outside frozen judging input")
    scores = pd.to_numeric(partial["judge_score"], errors="raise").astype(int)
    req(scores.between(0, 4).all(), "Partial score file contains judge score outside 0..4")
    req(len(partial) == EXPECTED_PERSISTED, f"r3 expects exactly {EXPECTED_PERSISTED} persisted scores, got {len(partial)}")
    req(KNOWN_QUARANTINED_CASE in set(frame["case_id"]), "Known quarantine case missing from frozen novel input")
    req(KNOWN_QUARANTINED_CASE not in set(partial["case_id"]), "Known quarantine case unexpectedly already has a score")
    print(f"PASS  persisted partial score file verified: {len(partial)}/{EXPECTED_NOVEL} cases")
    print(f"PASS  known timeout case remains unresolved and unscored: {KNOWN_QUARANTINED_CASE}")

    print()
    print("V4A 17-QUERY RETRIEVAL DIAGNOSTIC JUDGE PACKAGE PREFLIGHT: PASS")
    print(f"Novel BM25-only pairs frozen: {EXPECTED_NOVEL}")
    print(f"Persisted partial cases:      {len(partial)}")
    print("Timeout execution patch:      900s / RESUME ONLY")
    print(f"Known case quarantine:         {KNOWN_QUARANTINED_CASE} / UNSCORED")
    print("Judge execution:              NO")
    print("Human labels used:            NO")
    print("Development selection:        NOT REOPENED")
    print("Release analysis:             NO")
    print("Next: resume judging the remaining frozen cases while preserving the repeatedly timed-out case as explicitly unscored quarantine.")


if __name__ == "__main__":
    main()
