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

CONTRACT_FILE = EVALS / "system_evaluation/semantic_relevance_system_eval.v1.0.0.json"
SYSTEM_INPUT_LOCK_FILE = EVALS / "system_evaluation/semantic_relevance_system_eval_input_lock.v1.0.0.json"
SCORING_INPUT_FILE = EVALS / "system_evaluation/semantic_relevance_system_eval_scoring_input.v2.0.0.json"
EXECUTION_PATCH_FILE = EVALS / "system_evaluation/semantic_relevance_system_eval_execution_patch.retryfix1.json"
RECOMMENDER_V2_LOCK_FILE = EVALS / "system_evaluation/semantic_recommender_v2_lock.json"
EXPECTED_V2_RECOMMENDER_SHA256 = "cdd0458ff23bd669185a8c882ceb132438d037af1a29f4dd60bd7fef9c405b60"
EXPECTED_V2_GIT_COMMIT = "91fd9d2a0ebde9f66e7eb6b826631db70b63a165"
BASELINE_V1_RUN = "evals/runs/semantic_relevance_system_eval_v1/20261002T152856Z_recommendations"

RUBRIC_FILE = EVALS / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
FACET_SPEC_FILE = EVALS / "facets/semantic_relevance/semantic_relevance_query_facets.v0.10.1.json"
JUDGE_CONFIG_FILE = EVALS / "judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
JUDGE_FILE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS / "semantic_relevance_facet_scoring.py"

AUDIT_COLUMNS = [
    "case_id","query_id","query","rank","isbn13","title","authors",
    "has_relevant_evidence","evidence_text","matched_concept","match_level",
    "judge_score","judge_reason","semantic_generation_mode",
    "semantic_stage_retry_count","book_subject_summary",
    "book_subject_span_ids_json","book_subject_reason",
    "book_subject_analysis_retry_count","deterministic_direct_cue_count",
    "deterministic_cue_polarity_blocked_count",
    "composite_verification_attempt_count","composite_verification_count",
    "hard_exclusion_precheck_attempt_count",
    "hard_exclusion_precheck_trigger_count","core_facet_count",
    "incidental_core_count","meaningful_core_count","meaningful_core_coverage",
    "strong_core_count","direct_core_count","entailed_core_count",
    "adjacent_core_count","unsupported_core_count","qualifier_count",
    "satisfied_qualifier_count","qualifier_cap_applied","clear_rule_applied",
    "scoring_explanation","facet_assessments_json","facet_evidence_json",
    "component_evidence_ledger_json",
]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))

def package_ver(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None

def resolve_collection_run(value: str | None) -> Path:
    scoring_input = load_json(SCORING_INPUT_FILE)
    frozen = Path(scoring_input["collection_run"])
    if not frozen.is_absolute():
        frozen = REPO_ROOT / frozen

    if value:
        supplied = Path(value)
        if not supplied.is_absolute():
            supplied = REPO_ROOT / supplied
        if supplied.resolve() != frozen.resolve():
            raise ValueError(
                "Supplied collection run does not match frozen scoring input: "
                f"{supplied} != {frozen}"
            )

    if not frozen.exists():
        raise FileNotFoundError(frozen)
    return frozen

def verify_frozen_inputs(run_dir: Path):
    system_lock = load_json(SYSTEM_INPUT_LOCK_FILE)
    scoring_input = load_json(SCORING_INPUT_FILE)
    collection_lock = load_json(
        run_dir / "recommendation_collection_lock.json"
    )
    recommender_v2_lock = load_json(RECOMMENDER_V2_LOCK_FILE)

    if system_lock.get("status") != "FROZEN_BEFORE_SYSTEM_EVAL_EXECUTION":
        raise ValueError("Original system input lock is not frozen.")

    if scoring_input.get("status") != "FROZEN_BEFORE_JUDGE_SCORING":
        raise ValueError("V2 scoring input is not frozen.")

    if scoring_input.get("system_eval_variant") != "v2-orderfix":
        raise ValueError("Unexpected v2 scoring-input variant.")

    if collection_lock.get("status") != (
        "RECOMMENDATIONS_FROZEN_BEFORE_JUDGE_SCORING"
    ):
        raise ValueError("V2 recommendation collection is not frozen.")

    if collection_lock.get("system_eval_variant") != "v2-orderfix":
        raise ValueError("Unexpected v2 collection variant.")

    if recommender_v2_lock.get("status") != "FROZEN_FOR_SYSTEM_EVALUATION":
        raise ValueError("V2 recommender lock is not frozen.")

    if recommender_v2_lock.get("source_sha256") != (
        EXPECTED_V2_RECOMMENDER_SHA256
    ):
        raise ValueError("V2 recommender-lock source SHA mismatch.")

    # The frozen output artifacts are the scoring-stage source of truth.
    checks = [
        (
            "recommendations",
            run_dir / "recommendations.csv",
            scoring_input["recommendations_sha256"],
        ),
        (
            "judge input",
            run_dir / "judge_input.csv",
            scoring_input["judge_input_sha256"],
        ),
        ("scoring", SCORING_FILE, system_lock["scoring_sha256"]),
        ("facet spec", FACET_SPEC_FILE, system_lock["facet_spec_sha256"]),
        ("judge config", JUDGE_CONFIG_FILE, system_lock["judge_config_sha256"]),
        ("contract", CONTRACT_FILE, system_lock["contract_file_sha256"]),
    ]

    for label, path, expected in checks:
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"{label} hash mismatch: expected={expected} actual={actual}"
            )
        print(f"PASS  frozen {label} hash unchanged")

    # Cross-check the collection lock against the newly frozen scoring input.
    if collection_lock.get("recommendations_sha256") != (
        scoring_input["recommendations_sha256"]
    ):
        raise ValueError(
            "V2 collection/scoring-input recommendations SHA mismatch."
        )

    if collection_lock.get("judge_input_sha256") != (
        scoring_input["judge_input_sha256"]
    ):
        raise ValueError(
            "V2 collection/scoring-input judge-input SHA mismatch."
        )

    if collection_lock.get("recommender_sha256") != (
        EXPECTED_V2_RECOMMENDER_SHA256
    ):
        raise ValueError("V2 collection recommender SHA mismatch.")

    if scoring_input.get("recommender_sha256") != (
        EXPECTED_V2_RECOMMENDER_SHA256
    ):
        raise ValueError("V2 scoring-input recommender SHA mismatch.")

    if collection_lock.get("expected_execution_git_commit") != (
        EXPECTED_V2_GIT_COMMIT
    ):
        raise ValueError("V2 collection execution Git commit mismatch.")

    if scoring_input.get("execution_git_commit") != EXPECTED_V2_GIT_COMMIT:
        raise ValueError("V2 scoring-input execution Git commit mismatch.")

    if scoring_input.get("baseline_v1_run") != BASELINE_V1_RUN:
        raise ValueError("V2 scoring input baseline-v1 link mismatch.")

    print("PASS  v2 collection/scoring provenance cross-checks")

    # Keep the original frozen judge / retryfix1 acceptance contract unchanged.
    frozen_judge_sha = system_lock["judge_sha256"]
    current_judge_sha = sha256(JUDGE_FILE)

    if current_judge_sha == frozen_judge_sha:
        print("PASS  frozen judge hash unchanged")
    else:
        patch = load_json(EXECUTION_PATCH_FILE)
        required = {
            "status": "APPLIED_NON_SEMANTIC_EXECUTION_PATCH",
            "patch_id": "book-subject-retryfix1",
            "from_judge_sha256": frozen_judge_sha,
            "to_judge_sha256": current_judge_sha,
            "completed_cases_before_patch": 102,
            "completed_cases_using_repaired_path": 0,
            "semantic_rules_changed": False,
            "scoring_rules_changed": False,
        }
        for key, expected in required.items():
            if patch.get(key) != expected:
                raise ValueError(
                    f"Execution patch mismatch for {key}: "
                    f"expected={expected!r} actual={patch.get(key)!r}"
                )
        print(
            "PASS  book-subject-retryfix1 accepted: "
            "semantic/scoring rules unchanged"
        )

    return scoring_input

def validate_judge_input(frame: pd.DataFrame, facet_specs) -> pd.DataFrame:
    required = {"query_id","query","rank","isbn13","title","description"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"judge_input.csv missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["query"] = frame["query"].astype(str)
    frame["rank"] = frame["rank"].astype(int)
    frame["isbn13"] = frame["isbn13"].astype(str)

    if len(frame) != 120:
        raise ValueError(f"Expected 120 frozen pairs; got {len(frame)}.")
    if frame[["query_id","rank"]].duplicated().any():
        raise ValueError("Duplicate query_id/rank pairs found.")

    counts = frame.groupby("query_id").size().to_dict()
    if len(counts) != 12 or any(v != 10 for v in counts.values()):
        raise ValueError(f"Expected 12 queries x 10 recommendations; got {counts}")

    for query_id, group in frame.groupby("query_id", sort=False):
        if query_id not in facet_specs:
            raise ValueError(f"Unknown query_id: {query_id}")
        if set(group["query"].astype(str)) != {facet_specs[query_id].query}:
            raise ValueError(f"{query_id}: query text differs from frozen facet spec.")
        ranks = sorted(group["rank"].tolist())
        if ranks != list(range(1, 11)):
            raise ValueError(f"{query_id}: ranks must be exactly 1..10; got {ranks}")

    frame["case_id"] = frame.apply(
        lambda r: f"SYS_{r['query_id']}_R{int(r['rank']):02d}",
        axis=1,
    )
    if "authors" not in frame.columns:
        frame["authors"] = ""
    return frame

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

def persist(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows, columns=AUDIT_COLUMNS).to_csv(
        path, index=False, encoding="utf-8"
    )

def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(
        path,
        dtype={"case_id": str, "query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    missing = set(AUDIT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Existing judge_scores.csv incompatible; missing {sorted(missing)}"
        )
    return frame[AUDIT_COLUMNS].to_dict(orient="records")

def write_metadata(run_dir, config, status, completed, error=None):
    payload = {
        "evaluation_id": "semantic-relevance-system-eval",
        "evaluation_version": "1.0.0",
        "system_eval_variant": "v2-orderfix",
        "comparison_baseline_run": BASELINE_V1_RUN,
        "stage": "CALIBRATED_JUDGE_SCORING",
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_run": str(run_dir.relative_to(REPO_ROOT)),
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "judge_execution_patch": ("book-subject-retryfix1" if EXECUTION_PATCH_FILE.exists() else None),
        "judge_source_sha256": sha256(JUDGE_FILE),
        "judge_model": config["model"],
        "judge_base_url": config["base_url"],
        "temperature": config["temperature"],
        "human_relevance_labels_used": False,
        "total_pairs": 120,
        "completed_pairs": completed,
        "python_version": platform.python_version(),
        "deepeval_version": package_ver("deepeval"),
        "pandas_version": package_ver("pandas"),
        "error": error,
    }
    (run_dir / "judge_scoring_metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-run", default=None)
    args = parser.parse_args()

    print("Semantic relevance system evaluation v2 - calibrated judge scoring")
    print("--------------------------------------------------------------")
    print("Human relevance labels: NOT USED")

    run_dir = resolve_collection_run(args.collection_run)
    verify_frozen_inputs(run_dir)

    rubric = load_json(RUBRIC_FILE)
    config = load_json(JUDGE_CONFIG_FILE)
    facet_payload = load_json(FACET_SPEC_FILE)
    facet_specs = load_facet_specs(facet_payload)

    if config.get("version") != "0.29.0-r5":
        raise ValueError("Expected judge config version 0.29.0-r5.")
    if facet_payload.get("version") != "0.10.1":
        raise ValueError("Expected facet spec version 0.10.1.")

    frame = pd.read_csv(
        run_dir / "judge_input.csv",
        dtype={"query_id": str, "isbn13": str},
        encoding="utf-8",
    )
    frame = validate_judge_input(frame, facet_specs)

    results_file = run_dir / "judge_scores.csv"
    results = load_existing(results_file)
    completed_ids = {str(r["case_id"]) for r in results}
    remaining = frame[~frame["case_id"].isin(completed_ids)].copy()

    judge_model = None
    if not remaining.empty:
        judge_model = OllamaModel(
            model=config["model"],
            base_url=config["base_url"],
            temperature=config["temperature"],
        )

    write_metadata(
        run_dir, config,
        "RUNNING" if len(completed_ids) < 120 else "COMPLETE",
        len(completed_ids),
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
            score_result = compute_facet_score(spec=spec, verdict=verdict)

            positive_evidence = supported_evidence_by_id(
                spec=spec,
                verdict=verdict,
                evidence_by_id=semantic_result.evidence_by_id,
            )
            has_relevant = score_result.score > 0

            results.append({
                "case_id": case_id,
                "query_id": str(row["query_id"]),
                "query": str(row["query"]),
                "rank": int(row["rank"]),
                "isbn13": str(row["isbn13"]),
                "title": str(row["title"]),
                "authors": str(row.get("authors", "")),
                "has_relevant_evidence": has_relevant,
                "evidence_text": combined_evidence_text(
                    spec, positive_evidence, score_result.score
                ),
                "matched_concept": (
                    matched_core_concepts(spec, verdict)
                    if has_relevant else None
                ),
                "match_level": score_result.label,
                "judge_score": score_result.score,
                "judge_reason": verdict.overall_reason,
                "semantic_generation_mode": semantic_result.generation_mode,
                "semantic_stage_retry_count": semantic_result.stage_retry_count,
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
                "facet_assessments_json": serialize_facet_assessments(
                    spec=spec, verdict=verdict
                ),
                "facet_evidence_json": serialize_facet_evidence(
                    semantic_result.candidate_evidence_by_id
                ),
                "component_evidence_ledger_json": json.dumps(
                    semantic_result.component_evidence_ledger_by_id,
                    ensure_ascii=False,
                ),
            })

            completed_ids.add(case_id)
            persist(results_file, results)
            write_metadata(run_dir, config, "RUNNING", len(completed_ids))

            print(
                f"{len(completed_ids):3d}/120 {case_id}: "
                f"score={score_result.score} level={score_result.label} "
                f"title={str(row['title'])[:70]}"
            )
            if len(completed_ids) % 10 == 0:
                print(f"Checkpoint: {len(completed_ids)}/120 persisted")

    except Exception as exc:
        write_metadata(
            run_dir, config, "PARTIAL", len(completed_ids),
            f"{type(exc).__name__}: {exc}"
        )
        print(f"Status: PARTIAL ({len(completed_ids)}/120)")
        print("Rerun the SAME command to resume.")
        raise

    if len(completed_ids) != 120:
        raise RuntimeError(
            f"Judge scoring ended with {len(completed_ids)}/120 completed."
        )

    final_frame = pd.read_csv(
        results_file,
        dtype={"case_id": str, "query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    final_frame["_q"] = final_frame["query_id"].str.extract(r"(\d+)").astype(int)
    final_frame = (
        final_frame.sort_values(["_q","rank"], kind="stable")
        .drop(columns=["_q"])
    )
    final_frame.to_csv(results_file, index=False, encoding="utf-8")

    lock = {
        "status": "JUDGE_SCORES_FROZEN_BEFORE_METRIC_ANALYSIS",
        "system_eval_variant": "v2-orderfix",
        "comparison_baseline_run": BASELINE_V1_RUN,
        "recommender_sha256": EXPECTED_V2_RECOMMENDER_SHA256,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "judge_execution_patch": ("book-subject-retryfix1" if EXECUTION_PATCH_FILE.exists() else None),
        "judge_source_sha256": sha256(JUDGE_FILE),
        "judge_scores_file": "judge_scores.csv",
        "judge_scores_sha256": sha256(results_file),
        "recommendations_sha256": scoring_input["recommendations_sha256"],
        "judge_input_sha256": scoring_input["judge_input_sha256"],
        "completed_pairs": 120,
        "human_relevance_labels_used": False,
        "aggregate_metrics_calculated": False,
        "bootstrap_confidence_intervals_calculated": False,
    }
    (run_dir / "judge_scoring_lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    write_metadata(run_dir, config, "COMPLETE", 120)

    print()
    print("CALIBRATED JUDGE SCORING: PASS")
    print(f"Pairs scored: 120/120")
    print(f"Judge scores SHA-256: {sha256(results_file)}")
    print("Human relevance labels used: NO")
    print("Aggregate metrics calculated: NO")
    print("Bootstrap confidence intervals calculated: NO")
    print("Next: aggregate metrics + query-level bootstrap 95% CIs + release gates.")

if __name__ == "__main__":
    main()
