from __future__ import annotations
import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from deepeval.models import OllamaModel

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT/"evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))
SYS = EVALS/"system_evaluation"
RUN = ROOT/"evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"
HIST_RUN = ROOT/"evals/runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"

import semantic_relevance_facet_judge as facet_judge
from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec, compute_facet_score, validate_query_facet_spec

JUDGING_INPUT = RUN/"v4h_headroom_stage_b_judging_input.csv"
JUDGING_INPUT_LOCK = RUN/"v4h_headroom_stage_b_judging_input.lock.json"
SCORE_INVENTORY = RUN/"v4h_headroom_stage_b_score_inventory.csv"
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
EXEC_CONTRACT = SYS/"semantic_relevance_v4h_headroom_stage_b_execution_contract.v0.1.0.json"

FRESH_SCORES = RUN/"v4h_headroom_stage_b_fresh_judge_scores.csv"
METADATA = RUN/"v4h_headroom_stage_b_judging_metadata.json"
FINAL_LOCK = RUN/"v4h_headroom_stage_b_fresh_judge_scoring.lock.json"


TARGET_QUERY_IDS = ['V4H10', 'V4H11', 'V4H13', 'V4H16', 'V4H21', 'V4H23']
EXPECTED_FRESH_ROWS = 251
EXPECTED_JUDGING_INPUT_SHA256 = "ac5b832a2feb7c8f26dcc964cccf6b33000147f005482047ea0c36e4e51bb591"
EXPECTED_JUDGING_INPUT_LOCK_SHA256 = "49ae1b46c247a94fbae27cee9ad8b1987ca07e1fcaa24522f62137527e5d9392"
EXPECTED_SCORE_INVENTORY_SHA256 = "661933691c032d714a16a4e972f3754a15ca8bfac41d0108c0f4953f2cf7815a"
EXPECTED_V4H_FACET_SHA256 = "dbdb7c5d5603e17164cf6ef880df912f6196663387885e1f5773e54b97d2b106"
EXPECTED_V4H_AUTHORING_LOCK_SHA256 = "14cd3a8a9f6bf22666dd4c74a150e85f82a53789ccaa5bd2870d606b66ef83e7"
EXPECTED_V3_SCORING_LOCK_SHA256 = "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777"
EXPECTED_SPAN_ECHO_PATCH_SHA256 = "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266"
EXPECTED_ORIGINAL_DUP_PATCH_SHA256 = "09b2acb1d3809da6ed4667d1e3f7b042dc86f85b6f69b732bb9017136f0b2b94"
EXPECTED_CARRY_PATCH_SHA256 = "5e38542976fcb091a9f6461d2be11a34716d542b592bd96228e51146ac40ac73"
EXPECTED_EXEC_CONTRACT_SHA256 = "f93427e7734e4dfb9392800959b48bc9a22c283525afd1f35f8442e764ede1ce"

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
               "accepted Stage-A duplicate-ID carry-forward amendment")
    verify_raw(EXEC_CONTRACT, EXPECTED_EXEC_CONTRACT_SHA256,
               "Stage-B judge-execution contract")

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
    verify_raw(JUDGING_INPUT, EXPECTED_JUDGING_INPUT_SHA256, "Stage-B fresh judging input")
    verify_raw(JUDGING_INPUT_LOCK, EXPECTED_JUDGING_INPUT_LOCK_SHA256, "Stage-B judging-input lock")
    verify_raw(SCORE_INVENTORY, EXPECTED_SCORE_INVENTORY_SHA256, "Stage-B score inventory")

    lock = load_json(JUDGING_INPUT_LOCK)
    req(lock.get("status") == "V4H_HEADROOM_STAGE_B_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "Unexpected Stage-B judging-input lock status")
    req(int(lock.get("fresh_pair_count", -1)) == EXPECTED_FRESH_ROWS, "Fresh pair count changed")
    req(int(lock.get("reused_exact_pair_count", -1)) == 1, "Reused pair count changed")
    req(int(lock.get("total_stage_b_pair_count", -1)) == 252, "Stage-B total pair count changed")
    req(lock.get("fresh_judging_input_sha256") == EXPECTED_JUDGING_INPUT_SHA256,
        "Input lock fresh-input hash changed")
    req(lock.get("score_inventory_sha256") == EXPECTED_SCORE_INVENTORY_SHA256,
        "Input lock score-inventory hash changed")
    req(lock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5", "Judge candidate changed")
    req(lock.get("cross_query_score_reuse_allowed") is False,
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
    req(len(frame) == EXPECTED_FRESH_ROWS, f"Expected 251 fresh rows, got {len(frame)}")
    frame["isbn13"] = frame["isbn13"].map(canonical_isbn)
    frame["v4b_rank_num"] = pd.to_numeric(frame["v4b_rank"], errors="raise").astype(int)
    req(frame["v4b_rank_num"].ge(51).all(), "Fresh input contains rank outside 51-end")
    req(not frame["case_id"].duplicated().any(), "Duplicate fresh case ID")
    req(not frame.duplicated(["query_id","isbn13"]).any(), "Duplicate fresh query-book pair")
    req(set(frame["query_id"]).issubset(set(TARGET_QUERY_IDS)), "Fresh input contains non-target query")
    req(frame["description"].str.strip().ne("").all(), "Fresh input has empty description")
    return frame, lock


RESULT_BASE = [
    "case_id","query_id","query","stratum","isbn13","title","authors",
    "v4b_rank","hybrid_retrieval_rank","dense_rank","bm25_rank","candidate_source","reranker_score",
]
RESULT_DIAGNOSTIC = [
    "judge_score","match_level","judge_reason",
    "semantic_generation_mode","semantic_stage_retry_count",
    "book_subject_summary","book_subject_span_ids_json","book_subject_reason",
    "book_subject_analysis_retry_count",
    "deterministic_direct_cue_count","deterministic_cue_polarity_blocked_count",
    "composite_verification_attempt_count","composite_verification_count",
    "hard_exclusion_precheck_attempt_count","hard_exclusion_precheck_trigger_count",
    "core_facet_count","incidental_core_count","meaningful_core_count",
    "meaningful_core_coverage","strong_core_count","direct_core_count","entailed_core_count",
    "adjacent_core_count","unsupported_core_count","qualifier_count","satisfied_qualifier_count",
    "qualifier_cap_applied","clear_rule_applied","scoring_explanation",
    "facet_assessments_json","facet_evidence_json","component_evidence_ledger_json",
]
RESULT_COLUMNS = RESULT_BASE + RESULT_DIAGNOSTIC

def input_value(row: pd.Series, col: str) -> str:
    return str(row[col]) if col in row.index else ""

def install_execution_patches() -> None:
    original_prominence_validator = facet_judge._validate_prominence_result
    def patched_prominence_validator(result: Any, spans: dict[str,str], config: dict[str,Any]) -> None:
        normalized = []
        for raw in list(result.supporting_span_ids):
            text = str(raw).strip()
            if text in spans:
                normalized.append(text)
                continue
            replacement = None
            for sid, span_text in spans.items():
                if text == f"{sid}: {span_text}":
                    replacement = sid
                    break
            normalized.append(replacement if replacement is not None else raw)
        result.supporting_span_ids = normalized
        original_prominence_validator(result=result, spans=spans, config=config)
    facet_judge._validate_prominence_result = patched_prominence_validator

    original_candidate_normalizer = facet_judge.normalize_candidate_span_ids
    def patched_candidate_normalizer(selection: Any, spans: dict[str,str]) -> list[str]:
        raw_ids = [str(span_id).strip() for span_id in selection.evidence_span_ids]
        if facet_judge.NO_EVIDENCE_SPAN in raw_ids:
            return original_candidate_normalizer(selection=selection, spans=spans)
        if len(raw_ids) != len(set(raw_ids)):
            unique_ids = list(dict.fromkeys(raw_ids))
            print("INFO  non-semantic duplicate candidate-ID normalization: "
                  f"{raw_ids} -> {unique_ids}")
            normalized_selection = facet_judge.EvidenceSelection(
                evidence_span_ids=unique_ids,
                reason=selection.reason,
            )
            return original_candidate_normalizer(selection=normalized_selection, spans=spans)
        return original_candidate_normalizer(selection=selection, spans=spans)
    facet_judge.normalize_candidate_span_ids = patched_candidate_normalizer

def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

def write_metadata(status: str, completed: int, total: int, *, last_case: str|None=None, error: str|None=None) -> None:
    payload = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic",
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "fresh_total_cases": total,
        "fresh_completed_cases": completed,
        "fresh_remaining_cases": total-completed,
        "last_completed_case_id": last_case,
        "error": error,
        "judging_input_sha256": EXPECTED_JUDGING_INPUT_SHA256,
        "score_inventory_sha256": EXPECTED_SCORE_INVENTORY_SHA256,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "release_decision_modified": False,
    }
    METADATA.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")

def validate_existing_scores(scores: pd.DataFrame, judging: pd.DataFrame) -> pd.DataFrame:
    req(set(RESULT_COLUMNS) <= set(scores.columns), "Existing fresh score file schema changed")
    req(not scores["case_id"].duplicated().any(), "Duplicate case IDs in fresh score file")
    req(set(scores["case_id"]) <= set(judging["case_id"]), "Fresh score contains case outside frozen input")
    if len(scores):
        vals = pd.to_numeric(scores["judge_score"], errors="raise").astype(int)
        req(vals.between(0,4).all(), "Existing fresh score outside 0..4")
        expected = judging.set_index("case_id")[["query_id","isbn13","v4b_rank"]]
        for row in scores.itertuples(index=False):
            src = expected.loc[str(row.case_id)]
            req(str(row.query_id) == str(src["query_id"]), f"{row.case_id} query mismatch")
            req(canonical_isbn(row.isbn13) == canonical_isbn(src["isbn13"]), f"{row.case_id} ISBN mismatch")
            req(int(row.v4b_rank) == int(src["v4b_rank"]), f"{row.case_id} rank mismatch")
    return scores[RESULT_COLUMNS].copy()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of currently-unscored fresh pairs to process; omit for full continuation.")
    args = parser.parse_args()
    if args.limit is not None:
        req(args.limit > 0, "--limit must be > 0")

    print("Semantic relevance V4H - headroom Stage-B frozen judge execution")
    print("---------------------------------------------------------------")
    print("Role: consumed-holdout diagnostic only")
    print("Fresh pair count: 251")
    print("Exact reused pair count: 1 (not judged here)")
    print("Human relevance labels: NOT USED")
    print("Retrieval/reranker calls: 0")
    print("Release metrics/decision: NOT CALCULATED / NOT MODIFIED")
    print("Persistence: after every completed fresh case")
    print()

    req(not FINAL_LOCK.exists(), "Final Stage-B fresh-score lock already exists; judging is frozen")
    judging, _ = verify_input_boundary()
    instrument = verify_measurement_instrument()
    config, rubric, specs = instrument["config"], instrument["rubric"], instrument["facet_specs"]

    install_execution_patches()
    print("PASS  accepted non-semantic prominence span-echo normalization installed")
    print("PASS  duplicate candidate-ID normalization carry-forward installed")

    if FRESH_SCORES.exists():
        existing = pd.read_csv(FRESH_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
        results = validate_existing_scores(existing, judging)
        print(f"PASS  resume-safe persisted fresh scores verified: {len(results)}/{len(judging)}")
    else:
        results = pd.DataFrame(columns=RESULT_COLUMNS)
        print("INFO  no persisted Stage-B fresh scores found; starting at fresh case 1")

    completed = set(results["case_id"].astype(str))
    remaining = judging[~judging["case_id"].isin(completed)].copy()
    if args.limit is not None:
        remaining = remaining.head(args.limit)

    judge_model = OllamaModel(
        model=config["model"],
        base_url=config["base_url"],
        temperature=config["temperature"],
    )

    write_metadata("RUNNING", len(results), len(judging),
                   last_case=(None if len(results)==0 else str(results.iloc[-1]["case_id"])))

    try:
        for _, row in remaining.iterrows():
            case_id, qid = str(row["case_id"]), str(row["query_id"])
            spec = specs[qid]
            semantic_result = facet_judge.generate_validated_semantic_verdict(
                judge_model=judge_model,
                row=row,
                spec=spec,
                rubric=rubric,
                config=config,
            )
            score = compute_facet_score(spec=spec, verdict=semantic_result.verdict)
            out = {
                "case_id": case_id,
                "query_id": qid,
                "query": str(row["query"]),
                "stratum": str(row["stratum"]),
                "isbn13": canonical_isbn(row["isbn13"]),
                "title": str(row["title"]),
                "authors": input_value(row, "authors"),
                "v4b_rank": int(row["v4b_rank"]),
                "hybrid_retrieval_rank": input_value(row, "hybrid_retrieval_rank"),
                "dense_rank": input_value(row, "dense_rank"),
                "bm25_rank": input_value(row, "bm25_rank"),
                "candidate_source": input_value(row, "candidate_source"),
                "reranker_score": input_value(row, "reranker_score"),
                "judge_score": int(score.score),
                "match_level": str(score.label),
                "judge_reason": str(semantic_result.verdict.overall_reason),
                "semantic_generation_mode": str(semantic_result.generation_mode),
                "semantic_stage_retry_count": int(semantic_result.stage_retry_count),
                "book_subject_summary": str(semantic_result.book_subject_analysis.primary_subject_summary),
                "book_subject_span_ids_json": json.dumps(
                    semantic_result.book_subject_analysis.primary_subject_span_ids, ensure_ascii=False
                ),
                "book_subject_reason": str(semantic_result.book_subject_analysis.reason),
                "book_subject_analysis_retry_count": int(semantic_result.subject_analysis_retry_count),
                "deterministic_direct_cue_count": int(semantic_result.deterministic_direct_cue_count),
                "deterministic_cue_polarity_blocked_count": int(semantic_result.deterministic_cue_polarity_blocked_count),
                "composite_verification_attempt_count": int(semantic_result.composite_verification_attempt_count),
                "composite_verification_count": int(semantic_result.composite_verification_count),
                "hard_exclusion_precheck_attempt_count": int(semantic_result.hard_exclusion_precheck_attempt_count),
                "hard_exclusion_precheck_trigger_count": int(semantic_result.hard_exclusion_precheck_trigger_count),
                "core_facet_count": int(score.core_facet_count),
                "incidental_core_count": int(score.incidental_core_count),
                "meaningful_core_count": int(score.meaningful_core_count),
                "meaningful_core_coverage": float(score.meaningful_core_coverage),
                "strong_core_count": int(score.strong_core_count),
                "direct_core_count": int(score.direct_core_count),
                "entailed_core_count": int(score.entailed_core_count),
                "adjacent_core_count": int(score.adjacent_core_count),
                "unsupported_core_count": int(score.unsupported_core_count),
                "qualifier_count": int(score.qualifier_count),
                "satisfied_qualifier_count": int(score.satisfied_qualifier_count),
                "qualifier_cap_applied": bool(score.qualifier_cap_applied),
                "clear_rule_applied": "" if score.clear_rule_applied is None else str(score.clear_rule_applied),
                "scoring_explanation": str(score.explanation),
                "facet_assessments_json": facet_judge.serialize_facet_assessments(
                    spec=spec, verdict=semantic_result.verdict
                ),
                "facet_evidence_json": facet_judge.serialize_facet_evidence(
                    semantic_result.candidate_evidence_by_id
                ),
                "component_evidence_ledger_json": json.dumps(
                    semantic_result.component_evidence_ledger_by_id, ensure_ascii=False
                ),
            }
            results = pd.concat([results, pd.DataFrame([out])], ignore_index=True)
            results = validate_existing_scores(results, judging)
            atomic_write_csv(results, FRESH_SCORES)
            write_metadata("RUNNING", len(results), len(judging), last_case=case_id)
            print(f"{len(results):3d}/{len(judging)} {case_id}: "
                  f"score={score.score} label={score.label} retries={semantic_result.stage_retry_count}")
    except Exception as exc:
        last_case = None if len(results)==0 else str(results.iloc[-1]["case_id"])
        write_metadata("PARTIAL_FAILED", len(results), len(judging), last_case=last_case, error=repr(exc))
        print()
        print(f"PARTIAL  persisted fresh scores: {len(results)}/{len(judging)}")
        print(f"FAILED   current fresh case was not persisted: {repr(exc)}")
        print("Resume by rerunning the same PowerShell command after reviewing the failure.")
        raise

    if len(results) < len(judging):
        write_metadata("PARTIAL", len(results), len(judging),
                       last_case=(None if len(results)==0 else str(results.iloc[-1]["case_id"])))
        print()
        print(f"V4H HEADROOM STAGE-B JUDGE EXECUTION: PARTIAL ({len(results)}/{len(judging)} fresh persisted)")
        print("Final fresh-score lock: NOT WRITTEN")
        print("Release metrics calculated: NO")
        print("Frozen V4B release decision modified: NO")
        print("Rerun without --limit to continue.")
        return

    results = validate_existing_scores(results, judging)
    req(len(results) == EXPECTED_FRESH_ROWS, "Complete fresh score count changed")
    req(set(results["case_id"]) == set(judging["case_id"]), "Complete fresh case set mismatch")
    results = results.sort_values(["query_id","v4b_rank"], kind="stable").reset_index(drop=True)
    atomic_write_csv(results, FRESH_SCORES)
    write_metadata("COMPLETE_FROZEN", len(results), len(judging), last_case=str(results.iloc[-1]["case_id"]))

    counts = pd.to_numeric(results["judge_score"], errors="raise").astype(int).value_counts().sort_index().to_dict()
    lock = {
        "status": "V4H_HEADROOM_STAGE_B_FRESH_JUDGE_SCORES_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic",
        "fresh_pair_count": EXPECTED_FRESH_ROWS,
        "reused_exact_pair_count_external": 1,
        "total_stage_b_pair_count": 252,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "judging_input_sha256": EXPECTED_JUDGING_INPUT_SHA256,
        "judging_input_lock_sha256": EXPECTED_JUDGING_INPUT_LOCK_SHA256,
        "score_inventory_sha256": EXPECTED_SCORE_INVENTORY_SHA256,
        "execution_contract_sha256": EXPECTED_EXEC_CONTRACT_SHA256,
        "span_echo_patch_sha256": EXPECTED_SPAN_ECHO_PATCH_SHA256,
        "original_duplicate_id_patch_sha256": EXPECTED_ORIGINAL_DUP_PATCH_SHA256,
        "carryforward_duplicate_id_patch_sha256": EXPECTED_CARRY_PATCH_SHA256,
        "fresh_judge_scores_sha256": sha256(FRESH_SCORES),
        "judging_metadata_sha256": sha256(METADATA),
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "release_decision_modified": False,
        "next_stage": "COMBINE_60_REUSED_PLUS_251_FRESH_AND_CLASSIFY_STAGE_B_HEADROOM",
    }
    FINAL_LOCK.write_text(json.dumps(lock, indent=2)+"\n", encoding="utf-8")

    print()
    print("PASS  all 251 fresh Stage-B pairs judged")
    print("PASS  fresh Stage-B score set frozen")
    print("V4H HEADROOM STAGE-B JUDGE EXECUTION: PASS")
    print(f"Fresh judge scores SHA-256: {sha256(FRESH_SCORES)}")
    print(f"Fresh scoring lock SHA-256: {sha256(FINAL_LOCK)}")
    print(f"Fresh score counts: {counts}")
    print("Human relevance labels used: NO")
    print("Release metrics calculated: NO")
    print("Frozen V4B release decision modified: NO")
    print("Next: combine 1 exact reused score + 251 fresh scores and classify Stage-B headroom.")

if __name__ == "__main__":
    main()
