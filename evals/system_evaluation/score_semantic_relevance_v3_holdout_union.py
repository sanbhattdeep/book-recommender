from __future__ import annotations

import argparse
import hashlib
import json
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

from semantic_relevance_facet_judge import (
    generate_validated_semantic_verdict,
    serialize_facet_assessments,
    serialize_facet_evidence,
    supported_core_facet_texts,
    supported_evidence_by_id,
)
from semantic_relevance_facet_scoring import compute_facet_score
from run_judge_v0_29_r5_development import load_facet_specs

RUN_DIR = (
    EVALS
    / "runs"
    / "semantic_relevance_v3_holdout"
    / "20261007T124150Z_retrieval"
)
UNION_FILE = RUN_DIR / "holdout_displayed_pair_union.csv"
UNION_LOCK_FILE = RUN_DIR / "holdout_displayed_pair_union_lock.json"
JUDGING_INPUT_LOCK_FILE = RUN_DIR / "holdout_judging_input_lock.json"
EOL_PATCH_FILE = (
    EVALS
    / "system_evaluation"
    / "semantic_relevance_v3_holdout_execution_patch.eolnorm1.json"
)

SCORES_FILE = RUN_DIR / "holdout_judge_scores.csv"
METADATA_FILE = RUN_DIR / "holdout_judge_scoring_metadata.json"
SCORING_LOCK_FILE = RUN_DIR / "holdout_judge_scoring_lock.json"

HOLDOUT_FACET_FILE = (
    EVALS
    / "facets"
    / "semantic_relevance"
    / "semantic_relevance_v3_holdout_query_facets.v0.1.0.json"
)
REFERENCE_C_LOCK_FILE = (
    EVALS
    / "runs"
    / "semantic_relevance_top50_diagnostic_v1"
    / "20261005T113402Z_candidates"
    / "top50_scoring_lock.json"
)

JUDGE_FILE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS / "semantic_relevance_facet_scoring.py"
JUDGE_CONFIG_FILE = (
    EVALS
    / "judge_configs"
    / "semantic_relevance_judge.v0.29.0-r5.json"
)
RUBRIC_FILE = (
    EVALS
    / "rubrics"
    / "semantic_relevance"
    / "semantic_relevance_rubric.v0.1.0.json"
)

EXPECTED_UNION_SHA256 = (
    "885850ae548bba5af4fb24fefd74a7dbedbf961241a7090b10e24e1d5598030b"
)
EXPECTED_UNION_LOCK_SHA256 = (
    "fd5697398a80200386c38d1f4bedf0ebd4dfcfa3c677915c7597829f899a6da9"
)
EXPECTED_TOTAL = 322
EXPECTED_QUERIES = 24

AUDIT_COLUMNS = [
    "case_id",
    "query_id",
    "query",
    "stratum",
    "vector_rank",
    "isbn13",
    "title",
    "authors",
    "in_v2_top10",
    "v2_rank",
    "in_v3_top10",
    "v3_rank",
    "displayed_by_both",
    "has_relevant_evidence",
    "evidence_text",
    "matched_concept",
    "match_level",
    "judge_score",
    "judge_reason",
    "semantic_generation_mode",
    "semantic_stage_retry_count",
    "book_subject_summary",
    "book_subject_span_ids_json",
    "book_subject_reason",
    "book_subject_analysis_retry_count",
    "deterministic_direct_cue_count",
    "deterministic_cue_polarity_blocked_count",
    "composite_verification_attempt_count",
    "composite_verification_count",
    "hard_exclusion_precheck_attempt_count",
    "hard_exclusion_precheck_trigger_count",
    "core_facet_count",
    "incidental_core_count",
    "meaningful_core_count",
    "meaningful_core_coverage",
    "strong_core_count",
    "direct_core_count",
    "entailed_core_count",
    "adjacent_core_count",
    "unsupported_core_count",
    "qualifier_count",
    "satisfied_qualifier_count",
    "qualifier_cap_applied",
    "clear_rule_applied",
    "scoring_explanation",
    "facet_assessments_json",
    "facet_evidence_json",
    "component_evidence_ledger_json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def lf_normalized_sha256(path: Path) -> str:
    """SHA-256 after normalizing text line endings only: CRLF/CR -> LF."""
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def verify_checkpoint_text_hash(
    *,
    label: str,
    path: Path,
    expected: str,
) -> dict[str, str]:
    """
    Verify frozen textual runtime identity while tolerating checkout-only EOL
    conversion. Any byte difference beyond CRLF/CR versus LF still fails.
    """
    raw_sha = sha256(path)
    normalized_sha = lf_normalized_sha256(path)

    if raw_sha == expected:
        mode = "raw_bytes"
    elif normalized_sha == expected:
        mode = "lf_normalized_line_endings"
    else:
        raise ValueError(
            f"{label} changed since frozen Checkpoint C: "
            f"expected={expected} raw={raw_sha} lf_normalized={normalized_sha}"
        )

    print(
        f"PASS  frozen {label} matches Checkpoint C "
        f"(verification={mode})"
    )
    return {
        "checkpoint_c_sha256": expected,
        "current_raw_sha256": raw_sha,
        "lf_normalized_sha256": normalized_sha,
        "verification_mode": mode,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def package_ver(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bool_value(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def int_or_blank(value: Any) -> int | str:
    text = str(value).strip()
    return "" if not text else int(text)


def matched_core_concepts(spec, verdict) -> str | None:
    matched = supported_core_facet_texts(spec=spec, verdict=verdict)
    return "; ".join(matched) if matched else None


def combined_evidence_text(spec, evidence_by_id, overall_score: int) -> str | None:
    if overall_score == 0 or not evidence_by_id:
        return None
    parts = []
    for facet in spec.facets:
        evidence = evidence_by_id.get(facet.facet_id)
        if evidence:
            parts.append(f"{facet.facet_id}: {evidence}")
    return " | ".join(parts) if parts else None


def persist(rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows, columns=AUDIT_COLUMNS).to_csv(
        SCORES_FILE,
        index=False,
        encoding="utf-8",
    )


def load_existing() -> list[dict[str, Any]]:
    if not SCORES_FILE.exists():
        return []

    frame = pd.read_csv(
        SCORES_FILE,
        dtype={
            "case_id": str,
            "query_id": str,
            "isbn13": str,
        },
        keep_default_na=False,
        encoding="utf-8",
    )
    missing = set(AUDIT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Existing holdout_judge_scores.csv is incompatible; missing {sorted(missing)}"
        )

    require(
        not frame["case_id"].duplicated().any(),
        "Existing holdout judge scores contain duplicate case IDs.",
    )
    return frame[AUDIT_COLUMNS].to_dict(orient="records")


def write_metadata(
    *,
    config: dict[str, Any],
    status: str,
    completed: int,
    error: str | None = None,
) -> None:
    payload = {
        "evaluation_id": "semantic-relevance-v3-unseen-holdout",
        "stage": "ONE_TIME_FROZEN_JUDGE_SCORING",
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(RUN_DIR.relative_to(REPO_ROOT)),
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "judge_source_sha256": load_json(REFERENCE_C_LOCK_FILE)["judge_source_sha256"],
        "scoring_source_sha256": load_json(REFERENCE_C_LOCK_FILE)["scoring_source_sha256"],
        "holdout_facet_spec_sha256": sha256(HOLDOUT_FACET_FILE),
        "judge_config_sha256": load_json(REFERENCE_C_LOCK_FILE)["judge_config_sha256"],
        "rubric_sha256": load_json(REFERENCE_C_LOCK_FILE)["rubric_sha256"],
        "verification_patch_id": "v3-holdout-checkpoint-c-eol-normalization1",
        "verification_patch_sha256": sha256(EOL_PATCH_FILE),
        "current_runtime_raw_sha256": {
            "judge_source_sha256": sha256(JUDGE_FILE),
            "scoring_source_sha256": sha256(SCORING_FILE),
            "judge_config_sha256": sha256(JUDGE_CONFIG_FILE),
            "rubric_sha256": sha256(RUBRIC_FILE),
        },
        "displayed_pair_union_sha256": sha256(UNION_FILE),
        "judging_input_lock_sha256": sha256(JUDGING_INPUT_LOCK_FILE),
        "judge_model": config["model"],
        "judge_base_url": config["base_url"],
        "temperature": config["temperature"],
        "total_unique_pairs": EXPECTED_TOTAL,
        "completed_pairs": completed,
        "remaining_pairs": EXPECTED_TOTAL - completed,
        "human_relevance_labels_used": False,
        "aggregate_metrics_calculated": False,
        "paired_analysis_calculated": False,
        "release_decision_produced": False,
        "python_version": platform.python_version(),
        "deepeval_version": package_ver("deepeval"),
        "pandas_version": package_ver("pandas"),
        "error": error,
    }
    METADATA_FILE.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_inputs() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    require(sha256(UNION_FILE) == EXPECTED_UNION_SHA256, "Displayed-pair union changed.")
    require(
        sha256(UNION_LOCK_FILE) == EXPECTED_UNION_LOCK_SHA256,
        "Displayed-pair union lock changed.",
    )

    input_lock = load_json(JUDGING_INPUT_LOCK_FILE)
    require(
        input_lock.get("status")
        == "V3_HOLDOUT_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "Judge input has not been frozen.",
    )
    require(
        input_lock.get("displayed_pair_union_sha256") == EXPECTED_UNION_SHA256,
        "Judging-input lock union hash mismatch.",
    )
    require(
        int(input_lock.get("unique_pairs_to_judge", -1)) == EXPECTED_TOTAL,
        "Judging-input lock pair count changed.",
    )
    require(
        input_lock.get("judge_execution_performed") is False,
        "Judging-input lock says judge execution already occurred.",
    )

    reference = load_json(REFERENCE_C_LOCK_FILE)
    require(
        reference.get("status")
        == "TOP50_JUDGE_SCORES_FROZEN_BEFORE_HEADROOM_ANALYSIS",
        "Reference Checkpoint-C lock status changed.",
    )

    patch = load_json(EOL_PATCH_FILE)
    require(
        patch.get("status") == "APPLIED_NON_SEMANTIC_VERIFICATION_PATCH",
        "EOL verification patch status changed.",
    )
    require(
        patch.get("patch_id") == "v3-holdout-checkpoint-c-eol-normalization1",
        "Unexpected EOL verification patch ID.",
    )
    require(
        input_lock.get("verification_patch_id")
        == "v3-holdout-checkpoint-c-eol-normalization1",
        "Judging-input lock EOL verification patch ID mismatch.",
    )
    require(
        input_lock.get("verification_patch_sha256") == sha256(EOL_PATCH_FILE),
        "Judging-input lock EOL verification patch hash mismatch.",
    )

    checks = [
        ("judge source", JUDGE_FILE, "judge_source_sha256"),
        ("deterministic scoring", SCORING_FILE, "scoring_source_sha256"),
        ("judge config", JUDGE_CONFIG_FILE, "judge_config_sha256"),
        ("rubric", RUBRIC_FILE, "rubric_sha256"),
    ]
    runtime_provenance: dict[str, dict[str, str]] = {}
    for label, path, key in checks:
        expected = str(reference.get(key, ""))
        require(expected, f"Reference C lock missing {key}.")
        require(
            input_lock.get(key) == expected,
            f"Judging-input lock runtime hash mismatch: {key}",
        )
        current = verify_checkpoint_text_hash(
            label=label,
            path=path,
            expected=expected,
        )
        runtime_provenance[key] = current
        require(
            input_lock.get("runtime_byte_provenance", {}).get(key) == current,
            f"Runtime byte provenance changed after judging-input freeze: {key}",
        )

    require(
        sha256(HOLDOUT_FACET_FILE) == input_lock.get("holdout_facet_spec_sha256"),
        "Frozen holdout facet spec changed after judge-input freeze.",
    )

    config = load_json(JUDGE_CONFIG_FILE)
    require(config.get("version") == "0.29.0-r5", "Unexpected judge version.")
    require(float(config.get("temperature", -1)) == 0.0, "Judge temperature changed.")

    facet_payload = load_json(HOLDOUT_FACET_FILE)
    facet_specs = load_facet_specs(facet_payload)
    require(len(facet_specs) == EXPECTED_QUERIES, "Expected 24 holdout facet specs.")

    frame = pd.read_csv(
        UNION_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    require(len(frame) == EXPECTED_TOTAL, "Displayed-pair union row count changed.")
    require(not frame["case_id"].duplicated().any(), "Duplicate case IDs.")
    require(
        not frame.duplicated(["query_id", "isbn13"]).any(),
        "Duplicate query-book pairs.",
    )

    frame = frame.copy()
    frame["vector_rank"] = pd.to_numeric(
        frame["vector_rank"],
        errors="raise",
    ).astype(int)

    # Compatibility field used by the frozen judge pipeline. It is not a
    # displayed-system rank and is not used in final metric ordering.
    frame["rank"] = frame["vector_rank"].astype(int)

    for qid, group in frame.groupby("query_id", sort=False):
        require(qid in facet_specs, f"Unknown query_id: {qid}")
        require(
            set(group["query"].astype(str)) == {facet_specs[qid].query},
            f"{qid}: query differs from frozen facet spec.",
        )

    return frame, config, facet_specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume persisted cases after a pure execution failure.",
    )
    args = parser.parse_args()

    print("Semantic relevance v3 - ONE-TIME unseen holdout judge scoring")
    print("-------------------------------------------------------------")
    print("Frozen unique displayed pairs: 322")
    print("Human relevance labels: NOT USED")
    print("Aggregate metrics: NOT CALCULATED")
    print("Paired analysis: NOT CALCULATED")
    print("Release decision: NOT PRODUCED")
    print()

    frame, config, facet_specs = verify_inputs()
    rubric = load_json(RUBRIC_FILE)

    results = load_existing()
    if results and not args.resume:
        raise RuntimeError(
            "Persisted holdout judge scores already exist. "
            "Do not rerun or overwrite the unseen holdout. "
            "Use --resume only after a pure execution failure."
        )

    completed_ids = {str(row["case_id"]) for row in results}
    expected_ids = set(frame["case_id"].astype(str))
    require(
        completed_ids <= expected_ids,
        "Persisted scores contain case IDs outside the frozen judging union.",
    )

    remaining = frame[~frame["case_id"].isin(completed_ids)].copy()

    if len(completed_ids) == EXPECTED_TOTAL:
        print("All 322 frozen pairs are already persisted; finalizing lock only.")
    else:
        print(f"Starting/resuming frozen judge run: {len(completed_ids)}/{EXPECTED_TOTAL} persisted")
        print("Do not inspect or tune semantic results until the complete run finishes.")

    judge_model = None
    if not remaining.empty:
        judge_model = OllamaModel(
            model=config["model"],
            base_url=config["base_url"],
            temperature=config["temperature"],
        )

    write_metadata(
        config=config,
        status="RUNNING" if remaining.shape[0] else "COMPLETE",
        completed=len(completed_ids),
    )

    try:
        for _, row in remaining.iterrows():
            case_id = str(row["case_id"])
            spec = facet_specs[str(row["query_id"])]

            semantic_result = generate_validated_semantic_verdict(
                judge_model=judge_model,
                row=row,
                spec=spec,
                rubric=rubric,
                config=config,
            )
            verdict = semantic_result.verdict
            score_result = compute_facet_score(
                spec=spec,
                verdict=verdict,
            )

            positive_evidence = supported_evidence_by_id(
                spec=spec,
                verdict=verdict,
                evidence_by_id=semantic_result.evidence_by_id,
            )
            has_relevant = score_result.score > 0

            results.append(
                {
                    "case_id": case_id,
                    "query_id": str(row["query_id"]),
                    "query": str(row["query"]),
                    "stratum": str(row["stratum"]),
                    "vector_rank": int(row["vector_rank"]),
                    "isbn13": str(row["isbn13"]),
                    "title": str(row["title"]),
                    "authors": str(row.get("authors", "")),
                    "in_v2_top10": bool_value(row["in_v2_top10"]),
                    "v2_rank": int_or_blank(row["v2_rank"]),
                    "in_v3_top10": bool_value(row["in_v3_top10"]),
                    "v3_rank": int_or_blank(row["v3_rank"]),
                    "displayed_by_both": bool_value(row["displayed_by_both"]),
                    "has_relevant_evidence": has_relevant,
                    "evidence_text": combined_evidence_text(
                        spec,
                        positive_evidence,
                        score_result.score,
                    ),
                    "matched_concept": (
                        matched_core_concepts(spec, verdict)
                        if has_relevant
                        else None
                    ),
                    "match_level": score_result.label,
                    "judge_score": score_result.score,
                    "judge_reason": verdict.overall_reason,
                    "semantic_generation_mode": semantic_result.generation_mode,
                    "semantic_stage_retry_count": semantic_result.stage_retry_count,
                    "book_subject_summary": (
                        semantic_result.book_subject_analysis.primary_subject_summary
                    ),
                    "book_subject_span_ids_json": json.dumps(
                        semantic_result.book_subject_analysis.primary_subject_span_ids,
                        ensure_ascii=False,
                    ),
                    "book_subject_reason": semantic_result.book_subject_analysis.reason,
                    "book_subject_analysis_retry_count": (
                        semantic_result.subject_analysis_retry_count
                    ),
                    "deterministic_direct_cue_count": (
                        semantic_result.deterministic_direct_cue_count
                    ),
                    "deterministic_cue_polarity_blocked_count": (
                        semantic_result.deterministic_cue_polarity_blocked_count
                    ),
                    "composite_verification_attempt_count": (
                        semantic_result.composite_verification_attempt_count
                    ),
                    "composite_verification_count": (
                        semantic_result.composite_verification_count
                    ),
                    "hard_exclusion_precheck_attempt_count": (
                        semantic_result.hard_exclusion_precheck_attempt_count
                    ),
                    "hard_exclusion_precheck_trigger_count": (
                        semantic_result.hard_exclusion_precheck_trigger_count
                    ),
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
                    "facet_assessments_json": serialize_facet_assessments(
                        spec=spec,
                        verdict=verdict,
                    ),
                    "facet_evidence_json": serialize_facet_evidence(
                        semantic_result.candidate_evidence_by_id
                    ),
                    "component_evidence_ledger_json": json.dumps(
                        semantic_result.component_evidence_ledger_by_id,
                        ensure_ascii=False,
                    ),
                }
            )

            completed_ids.add(case_id)
            persist(results)
            write_metadata(
                config=config,
                status="RUNNING",
                completed=len(completed_ids),
            )

            # Intentionally suppress per-case semantic score/title output on
            # the one-time unseen holdout. Only progress is printed.
            if len(completed_ids) % 10 == 0 or len(completed_ids) == EXPECTED_TOTAL:
                print(f"Checkpoint: {len(completed_ids)}/{EXPECTED_TOTAL} persisted")

    except Exception as exc:
        write_metadata(
            config=config,
            status="PARTIAL",
            completed=len(completed_ids),
            error=f"{type(exc).__name__}: {exc}",
        )
        print()
        print(f"HOLDOUT JUDGE STATUS: PARTIAL ({len(completed_ids)}/{EXPECTED_TOTAL})")
        print("Pure execution failures may be resumed with the explicit --resume path.")
        print("Do not inspect/tune semantic results.")
        raise

    require(
        len(completed_ids) == EXPECTED_TOTAL,
        f"Judge run ended with {len(completed_ids)}/{EXPECTED_TOTAL} persisted.",
    )

    scores = pd.read_csv(
        SCORES_FILE,
        dtype={
            "case_id": str,
            "query_id": str,
            "isbn13": str,
        },
        keep_default_na=False,
        encoding="utf-8",
    )
    require(len(scores) == EXPECTED_TOTAL, "Final judge-score row count changed.")
    require(not scores["case_id"].duplicated().any(), "Duplicate final case IDs.")
    score_values = pd.to_numeric(scores["judge_score"], errors="raise").astype(int)
    require(score_values.between(0, 4).all(), "Judge scores outside frozen 0..4 scale.")

    lock = {
        "status": "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "judge_source_sha256": load_json(REFERENCE_C_LOCK_FILE)["judge_source_sha256"],
        "scoring_source_sha256": load_json(REFERENCE_C_LOCK_FILE)["scoring_source_sha256"],
        "holdout_facet_spec_sha256": sha256(HOLDOUT_FACET_FILE),
        "judge_config_sha256": load_json(REFERENCE_C_LOCK_FILE)["judge_config_sha256"],
        "rubric_sha256": load_json(REFERENCE_C_LOCK_FILE)["rubric_sha256"],
        "verification_patch_id": "v3-holdout-checkpoint-c-eol-normalization1",
        "verification_patch_sha256": sha256(EOL_PATCH_FILE),
        "current_runtime_raw_sha256": {
            "judge_source_sha256": sha256(JUDGE_FILE),
            "scoring_source_sha256": sha256(SCORING_FILE),
            "judge_config_sha256": sha256(JUDGE_CONFIG_FILE),
            "rubric_sha256": sha256(RUBRIC_FILE),
        },
        "displayed_pair_union_sha256": sha256(UNION_FILE),
        "judging_input_lock_sha256": sha256(JUDGING_INPUT_LOCK_FILE),
        "holdout_judge_scores_sha256": sha256(SCORES_FILE),
        "unique_pairs_judged": EXPECTED_TOTAL,
        "query_count": EXPECTED_QUERIES,
        "judge_score_min": int(score_values.min()),
        "judge_score_max": int(score_values.max()),
        "human_relevance_labels_used": False,
        "aggregate_metrics_calculated": False,
        "paired_analysis_calculated": False,
        "release_decision_produced": False,
        "next_stage": "FINAL_V2_V3_HOLDOUT_ANALYSIS",
    }

    if SCORING_LOCK_FILE.exists():
        existing = load_json(SCORING_LOCK_FILE)
        immutable = [key for key in lock if key != "frozen_at_utc"]
        for key in immutable:
            require(
                existing.get(key) == lock.get(key),
                f"Existing scoring-lock mismatch: {key}",
            )
    else:
        SCORING_LOCK_FILE.write_text(
            json.dumps(lock, indent=2) + "\n",
            encoding="utf-8",
        )

    write_metadata(
        config=config,
        status="COMPLETE",
        completed=EXPECTED_TOTAL,
    )

    print()
    print("V3 UNSEEN HOLDOUT JUDGE SCORING: PASS")
    print(f"Unique pairs judged: {EXPECTED_TOTAL}/{EXPECTED_TOTAL}")
    print(f"Judge scores SHA-256: {sha256(SCORES_FILE)}")
    print(f"Judge scoring lock SHA-256: {sha256(SCORING_LOCK_FILE)}")
    print("Human relevance labels used: NO")
    print("Aggregate metrics calculated: NO")
    print("Paired analysis calculated: NO")
    print("Release decision produced: NO")
    print("Next: final frozen v2/v3 holdout analysis.")


if __name__ == "__main__":
    main()
