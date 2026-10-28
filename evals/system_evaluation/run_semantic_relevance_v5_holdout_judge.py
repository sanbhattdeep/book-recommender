from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from deepeval.models import OllamaModel

# This runner lives under evals/system_evaluation while the frozen judge/scoring
# modules live directly under evals. Add that parent directory explicitly so
# execution is independent of the caller's current PYTHONPATH.
EVALS_DIR = Path(__file__).resolve().parents[1]
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

import semantic_relevance_facet_judge as judge_module
from semantic_relevance_facet_scoring import (
    compute_facet_score,
    derive_facet_support,
)
import verify_semantic_relevance_v5_holdout_judge_execution as boundary

RUN = boundary.RUN
SCORES = boundary.SCORES
METADATA = boundary.METADATA
SCORE_LOCK = boundary.SCORE_LOCK
EXPECTED_VERIFIER = "b1a9ebedb4886b15445d9dfc0550aa15e89b7674791d0e2d77e73a515911a8f7"
CASE_COUNT = boundary.CASE_COUNT
SMOKE_QUERY_IDS = ["V5H01", "V5H17", "V5H21"]

RESULT_COLUMNS = [
    "case_id", "query_id", "query", "stratum", "isbn13", "title", "authors",
    "in_B", "B_rank", "in_K", "K_rank", "system_count",
    "has_relevant_evidence", "evidence_text", "matched_concept", "match_level",
    "judge_score", "judge_reason",
    "semantic_generation_mode", "semantic_stage_retry_count",
    "book_subject_summary", "book_subject_span_ids_json", "book_subject_reason",
    "book_subject_analysis_retry_count", "deterministic_direct_cue_count",
    "deterministic_cue_polarity_blocked_count",
    "composite_verification_attempt_count", "composite_verification_count",
    "hard_exclusion_precheck_attempt_count", "hard_exclusion_precheck_trigger_count",
    "core_facet_count", "incidental_core_count", "meaningful_core_count",
    "meaningful_core_coverage", "strong_core_count", "direct_core_count",
    "entailed_core_count", "adjacent_core_count", "unsupported_core_count",
    "qualifier_count", "satisfied_qualifier_count", "qualifier_cap_applied",
    "clear_rule_applied", "scoring_explanation",
    "facet_assessments_json", "facet_evidence_json",
    "component_evidence_ledger_json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_sha256(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def install_nonsemantic_normalizations() -> None:
    # ---------------------------------------------------------------------
    # Repair 1: duplicate Stage-A real S# IDs only.
    # ---------------------------------------------------------------------
    original_normalize = judge_module.normalize_candidate_span_ids

    def normalize_candidate_span_ids_carryforward(selection, spans):
        ids = [str(x).strip() for x in selection.evidence_span_ids]
        if ids != [judge_module.NO_EVIDENCE_SPAN] and judge_module.NO_EVIDENCE_SPAN not in ids:
            deduped = []
            for item in ids:
                if item not in deduped:
                    deduped.append(item)
            if deduped != ids:
                selection = selection.model_copy(deep=True)
                selection.evidence_span_ids = deduped
        return original_normalize(selection, spans)

    judge_module.normalize_candidate_span_ids = normalize_candidate_span_ids_carryforward

    # ---------------------------------------------------------------------
    # Repair 2: prominence supporting_span_ids exact rendered-line echo only.
    # Reimplement the frozen function's bounded loop solely to insert the
    # accepted structural normalization before the original validator.
    # ---------------------------------------------------------------------
    def normalize_prominence_ids(result, spans):
        valid = set(spans)
        rendered = {
            f"{sid}: {text}".strip(): sid
            for sid, text in spans.items()
        }
        changed = False
        normalized = []
        for raw in list(result.supporting_span_ids):
            value = str(raw).strip()
            if value in valid:
                normalized.append(value)
            elif value in rendered:
                normalized.append(rendered[value])
                changed = True
            else:
                normalized.append(value)
        if changed:
            result = result.model_copy(deep=True)
            result.supporting_span_ids = normalized
        return result

    def assess_core_prominence_carryforward(
        judge_model, facet, evidence_text, book_subject, spans, config
    ):
        last_error = None
        for attempt in range(1, judge_module.MAX_STAGE_ATTEMPTS + 1):
            try:
                generated = judge_model.generate(
                    prompt=judge_module.build_prominence_prompt(
                        facet=facet,
                        evidence_text=evidence_text,
                        book_subject=book_subject,
                        spans=spans,
                        config=config,
                    ),
                    schema=judge_module.ProminenceAssessment,
                )
                result = judge_module.unpack_generated_model(
                    generated, judge_module.ProminenceAssessment
                )
                result = normalize_prominence_ids(result, spans)
                judge_module._validate_prominence_result(
                    result=result, spans=spans, config=config
                )
                return result, attempt - 1
            except Exception as error:
                last_error = error
        raise judge_module.JudgeOutputValidationError(
            "Unable to obtain valid core prominence assessment against frozen book subject."
        ) from last_error

    judge_module.assess_core_prominence = assess_core_prominence_carryforward


def load_existing(input_frame: pd.DataFrame) -> list[dict[str, Any]]:
    if not SCORES.exists():
        return []
    df = pd.read_csv(SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    req(list(df.columns) == RESULT_COLUMNS, "Existing V5H judge score schema changed")
    req(not df["case_id"].duplicated().any(), "Existing V5H scores contain duplicate case_id")
    valid_ids = set(input_frame["case_id"].astype(str))
    req(set(df["case_id"].astype(str)).issubset(valid_ids), "Existing scores contain case IDs outside frozen input")
    # Validate immutable identity fields against frozen input.
    expected = input_frame.set_index("case_id")
    for row in df.itertuples(index=False):
        src = expected.loc[str(row.case_id)]
        for col in ["query_id", "query", "stratum", "isbn13", "title", "authors",
                    "in_B", "B_rank", "in_K", "K_rank", "system_count"]:
            req(str(getattr(row, col)) == str(src[col]), f"Persisted identity drift for {row.case_id}/{col}")
        score = int(row.judge_score)
        req(score in [0, 1, 2, 3, 4], f"Persisted invalid score for {row.case_id}")
    return df[RESULT_COLUMNS].to_dict(orient="records")


def persist(results: list[dict[str, Any]]) -> None:
    pd.DataFrame(results, columns=RESULT_COLUMNS).to_csv(
        SCORES, index=False, encoding="utf-8"
    )


def write_metadata(
    *, status: str, completed: int, mode: str, last_case_id: str | None = None,
    error: str | None = None
) -> None:
    payload = {
        "status": status,
        "updated_at_utc": now(),
        "query_set": "V5H01-V5H24",
        "total_cases": CASE_COUNT,
        "completed_cases": completed,
        "remaining_cases": CASE_COUNT - completed,
        "mode": mode,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "source_judging_input_sha256": boundary.EXPECTED["novel"],
        "source_judging_input_lock_sha256": boundary.EXPECTED["jinput_lock"],
        "judge_execution_contract_sha256": boundary.EXPECTED["contract"],
        "nonsemantic_execution_patch_sha256": boundary.EXPECTED["exec_patch"],
        "resume_safe_persistence": True,
        "persist_after_every_successful_case": True,
        "failing_case_persisted": False,
        "historical_score_reuse_performed": False,
        "retrieval_calls_performed": 0,
        "reranker_inference_performed": False,
        "human_relevance_labels_used": False,
        "holdout_metrics_calculated": False,
        "release_gates_calculated": False,
        "release_decision_produced": False,
        "last_completed_case_id": last_case_id,
        "error": error,
    }
    METADATA.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def supported_evidence_text(spec, verdict, evidence_by_id):
    positive = judge_module.supported_evidence_by_id(
        spec=spec, verdict=verdict, evidence_by_id=evidence_by_id
    )
    parts = []
    for facet in spec.facets:
        text = positive.get(facet.facet_id)
        if text:
            parts.append(f"{facet.facet_id}: {text}")
    return " | ".join(parts) if parts else ""


def one_result(row, spec, judge_model, rubric, config) -> dict[str, Any]:
    semantic = judge_module.generate_validated_semantic_verdict(
        judge_model=judge_model,
        row=row,
        spec=spec,
        rubric=rubric,
        config=config,
    )
    verdict = semantic.verdict
    score = compute_facet_score(spec=spec, verdict=verdict)

    matched = judge_module.supported_core_facet_texts(spec=spec, verdict=verdict)
    matched_concept = " | ".join(matched) if matched else ""
    evidence_text = supported_evidence_text(spec, verdict, semantic.evidence_by_id)

    return {
        "case_id": str(row["case_id"]),
        "query_id": str(row["query_id"]),
        "query": str(row["query"]),
        "stratum": str(row["stratum"]),
        "isbn13": str(row["isbn13"]),
        "title": str(row["title"]),
        "authors": str(row["authors"]),
        "in_B": str(row["in_B"]),
        "B_rank": str(row["B_rank"]),
        "in_K": str(row["in_K"]),
        "K_rank": str(row["K_rank"]),
        "system_count": str(row["system_count"]),
        "has_relevant_evidence": bool(score.score > 0),
        "evidence_text": evidence_text,
        "matched_concept": matched_concept,
        "match_level": score.label,
        "judge_score": score.score,
        "judge_reason": verdict.overall_reason,
        "semantic_generation_mode": semantic.generation_mode,
        "semantic_stage_retry_count": semantic.stage_retry_count,
        "book_subject_summary": semantic.book_subject_analysis.primary_subject_summary,
        "book_subject_span_ids_json": json.dumps(
            semantic.book_subject_analysis.primary_subject_span_ids, ensure_ascii=False
        ),
        "book_subject_reason": semantic.book_subject_analysis.reason,
        "book_subject_analysis_retry_count": semantic.subject_analysis_retry_count,
        "deterministic_direct_cue_count": semantic.deterministic_direct_cue_count,
        "deterministic_cue_polarity_blocked_count": semantic.deterministic_cue_polarity_blocked_count,
        "composite_verification_attempt_count": semantic.composite_verification_attempt_count,
        "composite_verification_count": semantic.composite_verification_count,
        "hard_exclusion_precheck_attempt_count": semantic.hard_exclusion_precheck_attempt_count,
        "hard_exclusion_precheck_trigger_count": semantic.hard_exclusion_precheck_trigger_count,
        "core_facet_count": score.core_facet_count,
        "incidental_core_count": score.incidental_core_count,
        "meaningful_core_count": score.meaningful_core_count,
        "meaningful_core_coverage": score.meaningful_core_coverage,
        "strong_core_count": score.strong_core_count,
        "direct_core_count": score.direct_core_count,
        "entailed_core_count": score.entailed_core_count,
        "adjacent_core_count": score.adjacent_core_count,
        "unsupported_core_count": score.unsupported_core_count,
        "qualifier_count": score.qualifier_count,
        "satisfied_qualifier_count": score.satisfied_qualifier_count,
        "qualifier_cap_applied": score.qualifier_cap_applied,
        "clear_rule_applied": score.clear_rule_applied or "",
        "scoring_explanation": score.explanation,
        "facet_assessments_json": judge_module.serialize_facet_assessments(
            spec=spec, verdict=verdict
        ),
        "facet_evidence_json": judge_module.serialize_facet_evidence(
            semantic.candidate_evidence_by_id
        ),
        "component_evidence_ledger_json": json.dumps(
            semantic.component_evidence_ledger_by_id, ensure_ascii=False
        ),
    }


def finalize_if_complete(results: list[dict[str, Any]]) -> bool:
    if len(results) != CASE_COUNT:
        return False

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    req(not df["case_id"].duplicated().any(), "Cannot freeze duplicate V5H judge cases")
    req(len(df) == CASE_COUNT, "Cannot freeze incomplete V5H score set")
    # Persist stable source-input order in the final file.
    input_df = pd.read_csv(
        boundary.NOVEL_INPUT, dtype=str, keep_default_na=False, encoding="utf-8"
    )
    order = {cid: i for i, cid in enumerate(input_df["case_id"].astype(str))}
    df["_order"] = df["case_id"].astype(str).map(order)
    req(df["_order"].notna().all(), "Final V5H scores contain unknown case IDs")
    df = df.sort_values("_order", kind="stable").drop(columns=["_order"])
    df.to_csv(SCORES, index=False, encoding="utf-8")

    dist = {
        int(k): int(v)
        for k, v in pd.to_numeric(df["judge_score"]).value_counts().sort_index().items()
    }
    req(sum(dist.values()) == CASE_COUNT, "Final V5H score distribution count mismatch")

    write_metadata(
        status="COMPLETED_FROZEN_BEFORE_RELEASE_ANALYSIS",
        completed=CASE_COUNT,
        mode="full",
        last_case_id=str(df.iloc[-1]["case_id"]),
    )

    lock = {
        "status": "V5H_JUDGE_SCORES_FROZEN_BEFORE_RELEASE_ANALYSIS",
        "frozen_at_utc": now(),
        "query_set": "V5H01-V5H24",
        "query_count": 24,
        "case_count": CASE_COUNT,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "judge_execution_contract_sha256": boundary.EXPECTED["contract"],
        "source_judging_input_sha256": boundary.EXPECTED["novel"],
        "source_judging_input_lock_sha256": boundary.EXPECTED["jinput_lock"],
        "score_inventory_sha256": boundary.EXPECTED["inventory"],
        "nonsemantic_execution_patch_sha256": boundary.EXPECTED["exec_patch"],
        "judge_scores_sha256": sha256(SCORES),
        "judging_metadata_sha256": sha256(METADATA),
        "score_counts": {str(k): v for k, v in dist.items()},
        "historical_score_reuse_performed": False,
        "fresh_judge_cases_completed": CASE_COUNT,
        "retrieval_calls_performed": 0,
        "reranker_inference_performed": False,
        "human_relevance_labels_used": False,
        "holdout_metrics_calculated": False,
        "release_gates_calculated": False,
        "release_decision_produced": False,
        "next_stage": "PREPARE_V5H_B_VS_K_FINAL_RELEASE_ANALYSIS_USING_ORIGINAL_EIGHT_GATES",
    }
    SCORE_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return True


def select_smoke(remaining: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for qid in SMOKE_QUERY_IDS:
        g = remaining[remaining["query_id"] == qid]
        if len(g):
            parts.append(g.head(1))
    if not parts:
        return remaining.head(0)
    return pd.concat(parts, ignore_index=False).head(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = parser.parse_args()

    verifier_path = Path(boundary.__file__).resolve()
    req(
        sha256(verifier_path) == EXPECTED_VERIFIER
        or lf_sha256(verifier_path) == EXPECTED_VERIFIER,
        "V5H judge-execution verifier changed after package freeze",
    )

    frame, specs, config = boundary.validate(allow_existing_scores=True)
    rubric = boundary.load_json(boundary.RUBRIC)

    install_nonsemantic_normalizations()

    results = load_existing(frame)
    completed = {str(x["case_id"]) for x in results}
    remaining = frame[~frame["case_id"].astype(str).isin(completed)].copy()

    if args.mode == "smoke":
        work = select_smoke(remaining)
    else:
        work = remaining

    print("Semantic relevance V5H - frozen v0.29.0-r5 judge execution")
    print("----------------------------------------------------------")
    print(f"Mode: {args.mode.upper()}")
    print(f"Frozen cases: {CASE_COUNT}")
    print(f"Resume-safe persisted scores verified: {len(results)}/{CASE_COUNT}")
    print("Retrieval/reranker calls: 0")
    print("Historical score reuse: NO")
    print("Human relevance labels: NOT USED")
    print("Holdout/release metrics and gates: NOT CALCULATED")
    print("Two accepted non-semantic format repairs: ENABLED")
    print()

    if len(work) == 0:
        print("No currently-unscored cases selected for this mode.")
        if len(results) == CASE_COUNT:
            if finalize_if_complete(results):
                print("PASS  complete V5H judge score set frozen before release analysis")
        return

    judge_model = OllamaModel(
        model=config["model"],
        base_url=config["base_url"],
        temperature=config["temperature"],
    )

    write_metadata(
        status="RUNNING",
        completed=len(results),
        mode=args.mode,
        last_case_id=(str(results[-1]["case_id"]) if results else None),
    )

    try:
        for _, row in work.iterrows():
            cid = str(row["case_id"])
            qid = str(row["query_id"])
            # The failing case is appended only AFTER a complete successful
            # semantic generation + deterministic score, so failures are not persisted.
            result = one_result(row, specs[qid], judge_model, rubric, config)
            results.append(result)
            completed.add(cid)
            persist(results)
            write_metadata(
                status="RUNNING",
                completed=len(results),
                mode=args.mode,
                last_case_id=cid,
            )

            facet_by_id = {f.facet_id: f for f in specs[qid].facets}
            # Recreate support summary for console audit from serialized verdict
            # is unnecessary; final deterministic score/label is sufficient here.
            print(
                f"{len(results):>3}/{CASE_COUNT} {cid}: "
                f"score={result['judge_score']} level={result['match_level']} "
                f"retries={result['semantic_stage_retry_count']} "
                f"subject_retries={result['book_subject_analysis_retry_count']} "
                f"direct_cues={result['deterministic_direct_cue_count']} "
                f"composites={result['composite_verification_count']} "
                f"precheck_triggers={result['hard_exclusion_precheck_trigger_count']}"
            )

    except Exception as error:
        write_metadata(
            status="PARTIAL_FAILED",
            completed=len(results),
            mode=args.mode,
            last_case_id=(str(results[-1]["case_id"]) if results else None),
            error=f"{type(error).__name__}: {error}",
        )
        raise

    if len(results) == CASE_COUNT:
        req(args.mode == "full", "Smoke mode must not create final score lock")
        finalize_if_complete(results)
        df = pd.read_csv(SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
        counts = {
            int(k): int(v)
            for k, v in pd.to_numeric(df["judge_score"]).value_counts().sort_index().items()
        }
        print()
        print(f"PASS  all {CASE_COUNT} unique V5H displayed pairs freshly judged")
        print("PASS  complete V5H judge score set frozen before release analysis")
        print()
        print("V5H JUDGE EXECUTION: PASS")
        print(f"Judge scores SHA-256: {sha256(SCORES)}")
        print(f"Judge scoring lock SHA-256: {sha256(SCORE_LOCK)}")
        print(f"Score counts: {counts}")
        print("Holdout metrics calculated: NO")
        print("Release gates calculated: NO")
        print("Release decision produced: NO")
        print("Next: prepare B-vs-K final V5H release analysis using the original eight gates.")
    else:
        write_metadata(
            status=("SMOKE_COMPLETE" if args.mode == "smoke" else "PARTIAL"),
            completed=len(results),
            mode=args.mode,
            last_case_id=str(results[-1]["case_id"]),
        )
        print()
        print(f"Persisted V5H scores: {len(results)}/{CASE_COUNT}")
        print("Final judge-score lock written: NO")
        print("Holdout metrics calculated: NO")
        print("Release gates calculated: NO")
        print("Release decision produced: NO")
        if args.mode == "smoke":
            print("V5H JUDGE SMOKE: PASS")
            print("Next: run FULL mode; it will resume from these persisted smoke cases.")


if __name__ == "__main__":
    main()
