from __future__ import annotations

import argparse
import hashlib, json, sys, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from deepeval.models import OllamaModel

# The frozen judge/scoring modules live directly under <repo>/evals.
# Add that directory to sys.path BEFORE importing them.
REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

import semantic_relevance_facet_judge as facet_judge
from semantic_relevance_facet_scoring import (
    QueryFacet,
    QueryFacetSpec,
    compute_facet_score,
    validate_query_facet_spec,
)

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
METADATA = RUN / "v4h_judging_metadata.json"
SCORING_LOCK = RUN / "v4h_judge_scoring.lock.json"


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


RESULT_COLUMNS = [
    "case_id","query_id","query","stratum","isbn13","title","authors",
    "pair_source","in_v3b","v3b_rank","in_v4b","v4b_rank",
    "judge_score","match_level","judge_reason",
    "semantic_generation_mode","semantic_stage_retry_count",
    "book_subject_summary","book_subject_span_ids_json","book_subject_reason",
    "book_subject_analysis_retry_count",
    "deterministic_direct_cue_count","deterministic_cue_polarity_blocked_count",
    "composite_verification_attempt_count","composite_verification_count",
    "hard_exclusion_precheck_attempt_count","hard_exclusion_precheck_trigger_count",
    "core_facet_count","incidental_core_count","meaningful_core_count",
    "meaningful_core_coverage","strong_core_count","direct_core_count",
    "entailed_core_count","adjacent_core_count","unsupported_core_count",
    "qualifier_count","satisfied_qualifier_count","qualifier_cap_applied",
    "clear_rule_applied","scoring_explanation",
    "facet_assessments_json","facet_evidence_json","component_evidence_ledger_json",
]

def install_nonsemantic_execution_patches() -> None:
    # Accepted historical patch: normalize exact rendered prominence span
    # echoes back to their supplied S# ID before the original validator.
    original_prominence_validator = facet_judge._validate_prominence_result

    def patched_prominence_validator(
        result: Any,
        spans: dict[str, str],
        config: dict[str, Any],
    ) -> None:
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

    # V4H resume patch: if Stage A repeats an otherwise-valid real S# ID,
    # de-duplicate exact repetitions while preserving first-occurrence order.
    # NONE and every other validation rule are delegated unchanged.
    original_candidate_normalizer = facet_judge.normalize_candidate_span_ids

    def patched_candidate_normalizer(
        selection: Any,
        spans: dict[str, str],
    ) -> list[str]:
        raw_ids = [str(span_id).strip() for span_id in selection.evidence_span_ids]

        if facet_judge.NO_EVIDENCE_SPAN in raw_ids:
            return original_candidate_normalizer(selection=selection, spans=spans)

        if len(raw_ids) != len(set(raw_ids)):
            unique_ids = list(dict.fromkeys(raw_ids))
            print(
                "INFO  non-semantic duplicate candidate-ID normalization: "
                f"{raw_ids} -> {unique_ids}"
            )
            normalized_selection = facet_judge.EvidenceSelection(
                evidence_span_ids=unique_ids,
                reason=selection.reason,
            )
            return original_candidate_normalizer(
                selection=normalized_selection,
                spans=spans,
            )

        return original_candidate_normalizer(selection=selection, spans=spans)

    facet_judge.normalize_candidate_span_ids = patched_candidate_normalizer

def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

def write_metadata(status: str, completed: int, total: int, *, last_case: str | None = None,
                   error: str | None = None) -> None:
    payload = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "query_set": "V4H01-V4H24",
        "total_cases": total,
        "completed_cases": completed,
        "remaining_cases": total-completed,
        "last_completed_case_id": last_case,
        "error": error,
        "judging_input_sha256": sha256(JUDGING_INPUT),
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
    }
    METADATA.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

def verify_judging_input_boundary() -> tuple[pd.DataFrame, dict[str,Any]]:
    instrument = verify_measurement_instrument()
    union = verify_display_union()

    req(JUDGING_INPUT.exists(), "Frozen judging input is missing; run preparation first")
    req(JUDGING_INPUT_LOCK.exists(), "Frozen judging-input lock is missing; run preparation first")
    ilock = load_json(JUDGING_INPUT_LOCK)
    req(ilock.get("status") == "V4H_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "Unexpected judging-input lock status")
    req(int(ilock.get("case_count",-1)) == EXPECTED_ROWS, "Judging-input case count changed")
    req(ilock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5", "Judge candidate changed")
    req(ilock.get("display_union_sha256") == EXPECTED["display_union"], "Judging lock display-union hash changed")
    req(ilock.get("v4h_facet_spec_sha256") == EXPECTED["v4h_facet"], "Judging lock facet hash changed")
    req(ilock.get("historical_score_reuse_allowed") is False, "Historical score reuse unexpectedly allowed")
    req(ilock.get("judge_execution_performed") is False, "Input lock says judge already ran before freeze")
    req(ilock.get("human_relevance_labels_used") is False, "Input lock says human labels used")
    req(ilock.get("judging_input_sha256") == sha256(JUDGING_INPUT), "Judging input changed after freeze")

    frame = pd.read_csv(JUDGING_INPUT, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(frame) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} judging cases, got {len(frame)}")
    req(not frame["case_id"].duplicated().any(), "Duplicate judging case IDs")
    req(not frame.duplicated(["query_id","isbn13"]).any(), "Duplicate judging query-book pairs")
    req(set(frame["case_id"]) == {
        deterministic_case_id(qid,isbn)
        for qid,isbn in zip(frame["query_id"],frame["isbn13"])
    }, "Judging case IDs do not follow frozen deterministic rule")
    req(sorted(frame["query_id"].unique()) == EXPECTED_QUERY_IDS, "Judging query IDs changed")

    # Exact pair identity must still equal frozen display union.
    a = set(zip(frame["query_id"].astype(str),frame["isbn13"].map(canonical_isbn)))
    b = set(zip(union["query_id"].astype(str),union["isbn13"].map(canonical_isbn)))
    req(a == b, "Judging input pair set differs from frozen display union")
    return frame, instrument

def validate_existing_scores(scores: pd.DataFrame, judging: pd.DataFrame) -> pd.DataFrame:
    req(set(RESULT_COLUMNS) <= set(scores.columns), "Existing score file schema changed")
    req(not scores["case_id"].duplicated().any(), "Duplicate case IDs in existing score file")
    req(set(scores["case_id"]) <= set(judging["case_id"]), "Existing score contains case outside frozen input")
    if len(scores):
        vals = pd.to_numeric(scores["judge_score"], errors="raise").astype(int)
        req(vals.between(0,4).all(), "Existing score outside 0..4")
        expected_pairs = judging.set_index("case_id")[["query_id","isbn13"]]
        for row in scores.itertuples(index=False):
            src = expected_pairs.loc[str(row.case_id)]
            req(str(row.query_id) == str(src["query_id"]), f"Existing {row.case_id} query_id mismatch")
            req(canonical_isbn(row.isbn13) == canonical_isbn(src["isbn13"]),
                f"Existing {row.case_id} isbn mismatch")
    return scores[RESULT_COLUMNS].copy()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional number of currently-unscored cases to process; omit for full run.")
    args = parser.parse_args()
    if args.limit is not None:
        req(args.limit > 0, "--limit must be > 0")

    print("Semantic relevance V4H - frozen v0.29.0-r5 judge execution")
    print("----------------------------------------------------------")
    print("Historical score reuse: NO")
    print("Human relevance labels: NOT USED")
    print("Retrieval/reranker calls: 0")
    print("Release metrics/decision: NOT CALCULATED")
    print("Persistence: after every completed case")
    print()

    req(not SCORING_LOCK.exists(), "Final V4H judge scoring lock already exists; judging is already frozen")
    judging, instrument = verify_judging_input_boundary()
    config = instrument["config"]
    rubric = instrument["rubric"]
    specs = instrument["facet_specs"]

    install_nonsemantic_execution_patches()
    print("PASS  accepted non-semantic prominence span-echo normalization installed")
    print("PASS  V4H non-semantic duplicate candidate-ID normalization installed")
    print("PASS  V4H non-semantic duplicate candidate-ID normalization installed")

    if SCORES.exists():
        existing = pd.read_csv(SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
        results = validate_existing_scores(existing, judging)
        print(f"PASS  resume-safe persisted scores verified: {len(results)}/{len(judging)}")
    else:
        results = pd.DataFrame(columns=RESULT_COLUMNS)
        print("INFO  no persisted V4H judge scores found; starting at case 1")

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

    processed_now = 0
    try:
        for _, row in remaining.iterrows():
            case_id = str(row["case_id"])
            qid = str(row["query_id"])
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
                "authors": str(row["authors"]),
                "pair_source": str(row["pair_source"]),
                "in_v3b": str(row["in_v3b"]),
                "v3b_rank": str(row["v3b_rank"]),
                "in_v4b": str(row["in_v4b"]),
                "v4b_rank": str(row["v4b_rank"]),
                "judge_score": int(score.score),
                "match_level": str(score.label),
                "judge_reason": str(semantic_result.verdict.overall_reason),
                "semantic_generation_mode": str(semantic_result.generation_mode),
                "semantic_stage_retry_count": int(semantic_result.stage_retry_count),
                "book_subject_summary": str(semantic_result.book_subject_analysis.primary_subject_summary),
                "book_subject_span_ids_json": json.dumps(
                    semantic_result.book_subject_analysis.primary_subject_span_ids,
                    ensure_ascii=False,
                ),
                "book_subject_reason": str(semantic_result.book_subject_analysis.reason),
                "book_subject_analysis_retry_count": int(semantic_result.subject_analysis_retry_count),
                "deterministic_direct_cue_count": int(semantic_result.deterministic_direct_cue_count),
                "deterministic_cue_polarity_blocked_count": int(
                    semantic_result.deterministic_cue_polarity_blocked_count
                ),
                "composite_verification_attempt_count": int(
                    semantic_result.composite_verification_attempt_count
                ),
                "composite_verification_count": int(semantic_result.composite_verification_count),
                "hard_exclusion_precheck_attempt_count": int(
                    semantic_result.hard_exclusion_precheck_attempt_count
                ),
                "hard_exclusion_precheck_trigger_count": int(
                    semantic_result.hard_exclusion_precheck_trigger_count
                ),
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
                    semantic_result.component_evidence_ledger_by_id,
                    ensure_ascii=False,
                ),
            }

            results = pd.concat([results, pd.DataFrame([out])], ignore_index=True)
            results = validate_existing_scores(results, judging)
            atomic_write_csv(results, SCORES)
            processed_now += 1
            completed_count = len(results)
            write_metadata("RUNNING", completed_count, len(judging), last_case=case_id)
            print(
                f"{completed_count:3d}/{len(judging)} {case_id}: "
                f"score={score.score} label={score.label} retries={semantic_result.stage_retry_count}"
            )
    except Exception as exc:
        last_case = None if len(results)==0 else str(results.iloc[-1]["case_id"])
        write_metadata("PARTIAL_FAILED", len(results), len(judging), last_case=last_case, error=repr(exc))
        print()
        print(f"PARTIAL  persisted judge scores: {len(results)}/{len(judging)}")
        print(f"FAILED   current case was not persisted: {repr(exc)}")
        print("Resume by rerunning the same PowerShell command after reviewing the failure.")
        raise

    # A --limit run is deliberately partial and cannot freeze final scores.
    if len(results) < len(judging):
        write_metadata("PARTIAL", len(results), len(judging),
                       last_case=(None if len(results)==0 else str(results.iloc[-1]["case_id"])))
        print()
        print(f"V4H JUDGE EXECUTION: PARTIAL ({len(results)}/{len(judging)} persisted)")
        print("Final scoring lock: NOT WRITTEN")
        print("Release metrics calculated: NO")
        print("Rerun without --limit to continue.")
        return

    # Complete-set integrity before final freeze.
    results = validate_existing_scores(results, judging)
    req(len(results) == EXPECTED_ROWS, "Complete judge score count changed")
    req(set(results["case_id"]) == set(judging["case_id"]), "Complete score case set mismatch")
    results = results.sort_values(["query_id","isbn13"], kind="stable").reset_index(drop=True)
    atomic_write_csv(results, SCORES)

    lock = {
        "status": "V4H_JUDGE_SCORES_FROZEN_BEFORE_RELEASE_ANALYSIS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_set": "V4H01-V4H24",
        "query_count": 24,
        "case_count": EXPECTED_ROWS,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "judging_input_sha256": sha256(JUDGING_INPUT),
        "judging_input_lock_sha256": sha256(JUDGING_INPUT_LOCK),
        "display_union_sha256": EXPECTED["display_union"],
        "v4h_facet_spec_sha256": EXPECTED["v4h_facet"],
        "historical_measurement_instrument_lock_sha256": EXPECTED["v3_holdout_scoring_lock"],
        "span_echo_patch_sha256": EXPECTED["span_echo_patch"],
        "judge_scores_sha256": sha256(SCORES),
        "judging_metadata_sha256": sha256(METADATA),
        "historical_score_reuse_performed": False,
        "fresh_judged_pairs": EXPECTED_ROWS,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "next_stage": "CALCULATE_FROZEN_V4H_STANDALONE_AND_PAIRED_RELEASE_METRICS",
    }
    # Update complete metadata before hashing it into final lock.
    write_metadata("COMPLETE_FROZEN_BEFORE_RELEASE_ANALYSIS", len(results), len(judging),
                   last_case=str(results.iloc[-1]["case_id"]))
    lock["judging_metadata_sha256"] = sha256(METADATA)
    SCORING_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    counts = pd.to_numeric(results["judge_score"], errors="raise").astype(int).value_counts().sort_index().to_dict()
    print()
    print("PASS  all 382 unique V4H displayed pairs freshly judged")
    print("PASS  complete judge score set frozen before release analysis")
    print()
    print("V4H JUDGE EXECUTION: PASS")
    print(f"Judge scores SHA-256: {sha256(SCORES)}")
    print(f"Judge scoring lock SHA-256: {sha256(SCORING_LOCK)}")
    print(f"Score counts: {counts}")
    print("Historical score reuse performed: NO")
    print("Human relevance labels used: NO")
    print("Release metrics calculated: NO")
    print("Next: compute frozen standalone V4B release metrics and secondary paired V3B-vs-V4B analysis.")

if __name__ == "__main__":
    main()
