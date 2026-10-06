from __future__ import annotations

"""
Semantic relevance top-50 diagnostic — Checkpoint C.

This stage:
- verifies the frozen Checkpoint A/B artifacts;
- scores ONLY the 387 novel query-book pairs;
- uses the same calibrated semantic_relevance_v0.29.0-r5 judge contract;
- persists every completed novel judgment immediately;
- safely resumes without re-judging completed cases;
- creates a frozen 600-row complete score map after all 387 novel cases finish.

This stage DOES NOT calculate aggregate metrics, oracle metrics, bootstrap CIs,
or release decisions. Those belong to Checkpoint D.
"""

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
    / "runs/semantic_relevance_top50_diagnostic_v1"
    / "20261005T113402Z_candidates"
)

TOP50_CANDIDATES_FILE = RUN_DIR / "top50_candidates.csv"
CHECKPOINT_A_LOCK_FILE = RUN_DIR / "top50_collection_lock.json"

SCORE_REUSE_MAP_FILE = RUN_DIR / "top50_score_reuse_map.csv"
NOVEL_JUDGE_INPUT_FILE = RUN_DIR / "top50_novel_judge_input.csv"
CHECKPOINT_B_LOCK_FILE = RUN_DIR / "top50_score_reuse_lock.json"

NEW_SCORES_FILE = RUN_DIR / "top50_judge_scores_new.csv"
COMPLETE_SCORES_FILE = RUN_DIR / "top50_judge_scores_complete.csv"
SCORING_METADATA_FILE = RUN_DIR / "top50_judge_scoring_metadata.json"
SCORING_LOCK_FILE = RUN_DIR / "top50_scoring_lock.json"

SYSTEM_INPUT_LOCK_FILE = (
    EVALS / "system_evaluation/semantic_relevance_system_eval_input_lock.v1.0.0.json"
)
EXECUTION_PATCH_FILE = (
    EVALS
    / "system_evaluation/semantic_relevance_system_eval_execution_patch.retryfix1.json"
)

RUBRIC_FILE = (
    EVALS / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
)
FACET_SPEC_FILE = (
    EVALS
    / "facets/semantic_relevance/semantic_relevance_query_facets.v0.10.1.json"
)
JUDGE_CONFIG_FILE = (
    EVALS
    / "judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
)
JUDGE_FILE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS / "semantic_relevance_facet_scoring.py"
CONTRACT_FILE = (
    EVALS / "system_evaluation/semantic_relevance_system_eval.v1.0.0.json"
)

V1_JUDGE_SCORES_FILE = (
    EVALS
    / "runs/semantic_relevance_system_eval_v1"
    / "20261002T152856Z_recommendations"
    / "judge_scores.csv"
)
V2_JUDGE_SCORES_FILE = (
    EVALS
    / "runs/semantic_relevance_system_eval_v2"
    / "20261004T140218Z_recommendations"
    / "judge_scores.csv"
)

EXPECTED_TOP50_SHA256 = (
    "679dc2a8a3607db8d98e4ee868b5d7fc6566d11cb235eebed24e4676240abbad"
)
EXPECTED_REUSE_MAP_SHA256 = (
    "603ca8e093fa7720c9cd9d9bb7787780316875f33ec37652894b44195fa565b4"
)
EXPECTED_NOVEL_INPUT_SHA256 = (
    "d1205259ff9399c112c282d7142bd12589a8d3a59cadda8e65f526be32401b82"
)
EXPECTED_V1_SCORES_SHA256 = (
    "d079a74c7f0473e46d28f19e37e4241a780804e972f3fa5f25b3a67e62a7bd11"
)
EXPECTED_V2_SCORES_SHA256 = (
    "4770986e868b26821131d2b36f6cc648f0cab68ae50acda0d1ba4df6d0bd0da1"
)

EXPECTED_TOTAL_PAIRS = 600
EXPECTED_REUSED_PAIRS = 213
EXPECTED_NOVEL_PAIRS = 387
EXPECTED_QUERY_COUNT = 12

# Full audit columns for newly judged cases. "rank" is retained for judge-code
# compatibility and is equal to vector_rank for this diagnostic.
NEW_AUDIT_COLUMNS = [
    "diagnostic_case_id",
    "case_id",
    "query_id",
    "query",
    "rank",
    "vector_rank",
    "isbn13",
    "title",
    "authors",
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

COMPLETE_COLUMNS = [
    "diagnostic_case_id",
    "query_id",
    "query",
    "vector_rank",
    "isbn13",
    "title",
    "description",
    "authors",
    "score_source",
    "source_case_id",
    "judge_score",
    "match_level",
    "has_relevant_evidence",
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


def check_hash(label: str, path: Path, expected: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: expected={expected} actual={actual} path={path}"
        )
    print(f"PASS  {label} hash unchanged")


def text_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def canonical_optional_authors(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text.strip().casefold() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def load_and_verify_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    """Verify A/B locks and immutable artifacts before any model is created."""

    a_lock = load_json(CHECKPOINT_A_LOCK_FILE)
    b_lock = load_json(CHECKPOINT_B_LOCK_FILE)

    if a_lock.get("status") != "TOP50_CANDIDATES_FROZEN_BEFORE_SCORE_REUSE":
        raise ValueError("Checkpoint A lock is not frozen for score reuse.")

    if b_lock.get("status") != "TOP50_SCORE_REUSE_FROZEN_BEFORE_NOVEL_JUDGING":
        raise ValueError("Checkpoint B lock is not frozen before novel judging.")

    if int(a_lock.get("judge_calls_performed", -1)) != 0:
        raise ValueError("Checkpoint A unexpectedly reports judge calls.")

    if int(b_lock.get("judge_calls_performed", -1)) != 0:
        raise ValueError("Checkpoint B unexpectedly reports judge calls.")

    if int(b_lock.get("reusable_pairs", -1)) != EXPECTED_REUSED_PAIRS:
        raise ValueError("Checkpoint B reusable-pair count changed.")

    if int(b_lock.get("novel_pairs", -1)) != EXPECTED_NOVEL_PAIRS:
        raise ValueError("Checkpoint B novel-pair count changed.")

    if int(b_lock.get("total_pairs", -1)) != EXPECTED_TOTAL_PAIRS:
        raise ValueError("Checkpoint B total-pair count changed.")

    if int(b_lock.get("historical_score_conflicts", -1)) != 0:
        raise ValueError("Checkpoint B lock reports historical score conflicts.")

    if int(b_lock.get("payload_mismatches", -1)) != 0:
        raise ValueError("Checkpoint B lock reports payload mismatches.")

    check_hash(
        "Checkpoint A top-50 candidates",
        TOP50_CANDIDATES_FILE,
        EXPECTED_TOP50_SHA256,
    )
    check_hash(
        "Checkpoint B score-reuse map",
        SCORE_REUSE_MAP_FILE,
        EXPECTED_REUSE_MAP_SHA256,
    )
    check_hash(
        "Checkpoint B novel judge input",
        NOVEL_JUDGE_INPUT_FILE,
        EXPECTED_NOVEL_INPUT_SHA256,
    )
    check_hash(
        "frozen v1 judge scores",
        V1_JUDGE_SCORES_FILE,
        EXPECTED_V1_SCORES_SHA256,
    )
    check_hash(
        "frozen v2 judge scores",
        V2_JUDGE_SCORES_FILE,
        EXPECTED_V2_SCORES_SHA256,
    )

    # Cross-check the hashes stored in the B lock as well.
    if b_lock.get("top50_candidates_sha256") != EXPECTED_TOP50_SHA256:
        raise ValueError("Checkpoint B lock top50 hash mismatch.")
    if b_lock.get("score_reuse_map_sha256") != EXPECTED_REUSE_MAP_SHA256:
        raise ValueError("Checkpoint B lock reuse-map hash mismatch.")
    if b_lock.get("novel_judge_input_sha256") != EXPECTED_NOVEL_INPUT_SHA256:
        raise ValueError("Checkpoint B lock novel-input hash mismatch.")

    top50 = pd.read_csv(
        TOP50_CANDIDATES_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    reuse = pd.read_csv(
        SCORE_REUSE_MAP_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    novel = pd.read_csv(
        NOVEL_JUDGE_INPUT_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    for frame in [top50, reuse, novel]:
        if "authors" in frame.columns:
            frame["authors"] = frame["authors"].map(canonical_optional_authors)

    if len(top50) != EXPECTED_TOTAL_PAIRS:
        raise ValueError(f"Expected 600 top-50 rows; got {len(top50)}")
    if len(reuse) != EXPECTED_TOTAL_PAIRS:
        raise ValueError(f"Expected 600 reuse-map rows; got {len(reuse)}")
    if len(novel) != EXPECTED_NOVEL_PAIRS:
        raise ValueError(f"Expected 387 novel rows; got {len(novel)}")

    required_novel = {
        "diagnostic_case_id",
        "query_id",
        "query",
        "vector_rank",
        "isbn13",
        "title",
        "description",
        "authors",
    }
    missing = required_novel - set(novel.columns)
    if missing:
        raise ValueError(f"Novel judge input missing columns: {sorted(missing)}")

    novel["vector_rank"] = novel["vector_rank"].astype(int)
    novel["rank"] = novel["vector_rank"]
    novel["case_id"] = novel["diagnostic_case_id"].astype(str)

    if novel["diagnostic_case_id"].duplicated().any():
        raise ValueError("Novel judge input contains duplicate case IDs.")

    if novel.duplicated(["query_id", "isbn13"]).any():
        raise ValueError("Novel judge input contains duplicate query-book pairs.")

    if set(reuse["score_source"]) - {"v1", "v2", "v1_and_v2", "UNSCORED"}:
        raise ValueError("Unexpected score_source in reuse map.")

    uns = reuse.loc[reuse["score_source"] == "UNSCORED"].copy()
    if len(uns) != EXPECTED_NOVEL_PAIRS:
        raise ValueError("Reuse map no longer contains exactly 387 UNSCORED rows.")

    if set(uns["diagnostic_case_id"]) != set(novel["diagnostic_case_id"]):
        raise ValueError(
            "Checkpoint B novel input does not exactly match UNSCORED reuse-map cases."
        )

    if int((reuse["score_source"] != "UNSCORED").sum()) != EXPECTED_REUSED_PAIRS:
        raise ValueError("Reuse map no longer contains exactly 213 reusable rows.")

    return top50, reuse, novel, a_lock, b_lock


def verify_judge_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify the same pinned semantic/scoring contract used by v1/v2."""

    system_lock = load_json(SYSTEM_INPUT_LOCK_FILE)

    if system_lock.get("status") != "FROZEN_BEFORE_SYSTEM_EVAL_EXECUTION":
        raise ValueError("Original system-evaluation input lock is not frozen.")

    checks = [
        ("scoring", SCORING_FILE, system_lock["scoring_sha256"]),
        ("facet spec", FACET_SPEC_FILE, system_lock["facet_spec_sha256"]),
        ("judge config", JUDGE_CONFIG_FILE, system_lock["judge_config_sha256"]),
        ("contract", CONTRACT_FILE, system_lock["contract_file_sha256"]),
    ]

    for label, path, expected in checks:
        check_hash(f"frozen {label}", path, expected)

    frozen_judge_sha = system_lock["judge_sha256"]
    current_judge_sha = sha256(JUDGE_FILE)

    execution_patch_id = None
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
        execution_patch_id = "book-subject-retryfix1"
        print(
            "PASS  book-subject-retryfix1 accepted: semantic/scoring rules unchanged"
        )

    rubric = load_json(RUBRIC_FILE)
    config = load_json(JUDGE_CONFIG_FILE)
    facet_payload = load_json(FACET_SPEC_FILE)

    if config.get("version") != "0.29.0-r5":
        raise ValueError("Expected judge config version 0.29.0-r5.")
    if facet_payload.get("version") != "0.10.1":
        raise ValueError("Expected facet spec version 0.10.1.")

    # Record, but do not invent a new post-hoc rubric pin beyond the original
    # v1/v2 contract checks.
    print(f"INFO  rubric SHA-256: {sha256(RUBRIC_FILE)}")

    config["_execution_patch_id"] = execution_patch_id
    return rubric, config, facet_payload


def validate_novel_queries(
    novel: pd.DataFrame,
    facet_specs: dict[str, Any],
) -> pd.DataFrame:
    """Ensure every novel row still matches the frozen 12-query facet spec."""

    frame = novel.copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["query"] = frame["query"].astype(str)
    frame["isbn13"] = frame["isbn13"].astype(str)
    frame["title"] = frame["title"].astype(str)
    frame["description"] = frame["description"].astype(str)
    frame["authors"] = frame["authors"].map(canonical_optional_authors)
    frame["vector_rank"] = frame["vector_rank"].astype(int)
    frame["rank"] = frame["vector_rank"]
    frame["case_id"] = frame["diagnostic_case_id"].astype(str)

    query_ids = set(frame["query_id"])
    if len(query_ids) != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"Expected novel rows across 12 queries; got {sorted(query_ids)}"
        )

    for query_id, group in frame.groupby("query_id", sort=False):
        if query_id not in facet_specs:
            raise ValueError(f"Unknown query_id in novel input: {query_id}")
        if set(group["query"]) != {facet_specs[query_id].query}:
            raise ValueError(
                f"{query_id}: novel query text differs from frozen facet spec."
            )

        ranks = group["vector_rank"].tolist()
        if any(rank < 1 or rank > 50 for rank in ranks):
            raise ValueError(f"{query_id}: invalid vector rank in novel input.")

    return frame


def matched_core_concepts(spec, verdict) -> str | None:
    matched = supported_core_facet_texts(spec=spec, verdict=verdict)
    return "; ".join(matched) if matched else None


def combined_evidence_text(
    spec,
    evidence_by_id: dict[str, str],
    overall_score: int,
) -> str | None:
    if overall_score == 0 or not evidence_by_id:
        return None

    parts: list[str] = []
    for facet in spec.facets:
        evidence = evidence_by_id.get(facet.facet_id)
        if evidence:
            parts.append(f"{facet.facet_id}: {evidence}")
    return " | ".join(parts) if parts else None


def persist_new_scores(rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows, columns=NEW_AUDIT_COLUMNS).to_csv(
        NEW_SCORES_FILE,
        index=False,
        encoding="utf-8",
    )


def load_existing_new_scores(
    novel: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Load partial results and prove they still correspond to frozen inputs."""

    if not NEW_SCORES_FILE.exists():
        return []

    frame = pd.read_csv(
        NEW_SCORES_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    missing = set(NEW_AUDIT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Existing new-score artifact is incompatible; missing {sorted(missing)}"
        )

    frame = frame[NEW_AUDIT_COLUMNS].copy()

    if frame["diagnostic_case_id"].duplicated().any():
        raise ValueError("Existing new-score artifact has duplicate case IDs.")

    if len(frame) > EXPECTED_NOVEL_PAIRS:
        raise ValueError("Existing new-score artifact has more than 387 rows.")

    valid = novel.set_index("diagnostic_case_id", drop=False)

    for row in frame.to_dict("records"):
        case_id = str(row["diagnostic_case_id"])
        if case_id not in valid.index:
            raise ValueError(
                f"Existing scored case is not in frozen novel input: {case_id}"
            )

        frozen = valid.loc[case_id]
        checks = {
            "query_id": str(frozen["query_id"]),
            "query": str(frozen["query"]),
            "isbn13": str(frozen["isbn13"]),
            "title": str(frozen["title"]),
            "vector_rank": str(int(frozen["vector_rank"])),
        }
        for field, expected in checks.items():
            actual = str(row[field])
            if actual != expected:
                raise ValueError(
                    f"Existing scored case {case_id} changed field {field}: "
                    f"expected={expected!r} actual={actual!r}"
                )

        score = int(row["judge_score"])
        if score < 0 or score > 4:
            raise ValueError(
                f"Existing scored case {case_id} has invalid score={score}"
            )

    return frame.to_dict(orient="records")


def write_scoring_metadata(
    *,
    config: dict[str, Any],
    status: str,
    completed: int,
    error: str | None = None,
) -> None:
    payload = {
        "diagnostic_id": "semantic_relevance_top50_diagnostic_v1",
        "stage": "CHECKPOINT_C_NOVEL_JUDGE_SCORING",
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(RUN_DIR.relative_to(REPO_ROOT)),
        "top50_candidates_sha256": sha256(TOP50_CANDIDATES_FILE),
        "score_reuse_map_sha256": sha256(SCORE_REUSE_MAP_FILE),
        "novel_judge_input_sha256": sha256(NOVEL_JUDGE_INPUT_FILE),
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "judge_execution_patch": config.get("_execution_patch_id"),
        "judge_source_sha256": sha256(JUDGE_FILE),
        "scoring_source_sha256": sha256(SCORING_FILE),
        "facet_spec_sha256": sha256(FACET_SPEC_FILE),
        "judge_config_sha256": sha256(JUDGE_CONFIG_FILE),
        "rubric_sha256": sha256(RUBRIC_FILE),
        "judge_model": config["model"],
        "judge_base_url": config["base_url"],
        "temperature": config["temperature"],
        "total_candidate_pairs": EXPECTED_TOTAL_PAIRS,
        "reused_pairs": EXPECTED_REUSED_PAIRS,
        "novel_pairs": EXPECTED_NOVEL_PAIRS,
        "completed_novel_pairs": completed,
        "remaining_novel_pairs": EXPECTED_NOVEL_PAIRS - completed,
        "human_relevance_labels_used": False,
        "aggregate_metrics_calculated": False,
        "oracle_metrics_calculated": False,
        "bootstrap_confidence_intervals_calculated": False,
        "python_version": platform.python_version(),
        "deepeval_version": package_ver("deepeval"),
        "pandas_version": package_ver("pandas"),
        "error": error,
    }

    SCORING_METADATA_FILE.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def create_complete_score_map(
    *,
    top50: pd.DataFrame,
    reuse: pd.DataFrame,
    new_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Create the frozen 600-row analysis input.

    Reused rows take their semantic outcome from Checkpoint B. Novel rows take
    their semantic outcome from the completed Checkpoint C score artifact.
    """

    base = top50[
        [
            "diagnostic_case_id",
            "query_id",
            "query",
            "vector_rank",
            "isbn13",
            "title",
            "description",
            "authors",
        ]
    ].copy()

    base["vector_rank"] = base["vector_rank"].astype(int)
    base["authors"] = base["authors"].map(canonical_optional_authors)

    reuse_small = reuse[
        [
            "diagnostic_case_id",
            "score_source",
            "reused_judge_score",
            "reused_match_level",
            "reused_has_relevant_evidence",
            "v1_case_id",
            "v2_case_id",
        ]
    ].copy()

    merged = base.merge(
        reuse_small,
        on="diagnostic_case_id",
        how="left",
        validate="one_to_one",
    )

    new_small = new_scores[
        [
            "diagnostic_case_id",
            "judge_score",
            "match_level",
            "has_relevant_evidence",
        ]
    ].copy()
    new_small = new_small.rename(
        columns={
            "judge_score": "new_judge_score",
            "match_level": "new_match_level",
            "has_relevant_evidence": "new_has_relevant_evidence",
        }
    )

    merged = merged.merge(
        new_small,
        on="diagnostic_case_id",
        how="left",
        validate="one_to_one",
    )

    def source_case_id(row: pd.Series) -> str:
        if row["score_source"] == "v1_and_v2":
            return str(row["v2_case_id"])
        if row["score_source"] == "v2":
            return str(row["v2_case_id"])
        if row["score_source"] == "v1":
            return str(row["v1_case_id"])
        return str(row["diagnostic_case_id"])

    merged["source_case_id"] = merged.apply(source_case_id, axis=1)

    reused_mask = merged["score_source"] != "UNSCORED"
    novel_mask = ~reused_mask

    merged["judge_score"] = ""
    merged["match_level"] = ""
    merged["has_relevant_evidence"] = ""

    merged.loc[reused_mask, "judge_score"] = merged.loc[
        reused_mask, "reused_judge_score"
    ]
    merged.loc[reused_mask, "match_level"] = merged.loc[
        reused_mask, "reused_match_level"
    ]
    merged.loc[reused_mask, "has_relevant_evidence"] = merged.loc[
        reused_mask, "reused_has_relevant_evidence"
    ]

    merged.loc[novel_mask, "judge_score"] = merged.loc[
        novel_mask, "new_judge_score"
    ]
    merged.loc[novel_mask, "match_level"] = merged.loc[
        novel_mask, "new_match_level"
    ]
    merged.loc[novel_mask, "has_relevant_evidence"] = merged.loc[
        novel_mask, "new_has_relevant_evidence"
    ]
    merged.loc[novel_mask, "score_source"] = "checkpoint_c_new"

    if len(merged) != EXPECTED_TOTAL_PAIRS:
        raise ValueError(f"Complete score map expected 600 rows; got {len(merged)}")

    if merged["diagnostic_case_id"].duplicated().any():
        raise ValueError("Complete score map contains duplicate case IDs.")

    if merged["judge_score"].astype(str).eq("").any():
        missing = merged.loc[
            merged["judge_score"].astype(str).eq(""),
            ["diagnostic_case_id", "query_id", "vector_rank"],
        ]
        raise ValueError(
            "Complete score map has missing judge scores:\n"
            + missing.to_string(index=False)
        )

    scores = merged["judge_score"].astype(int)
    if ((scores < 0) | (scores > 4)).any():
        raise ValueError("Complete score map contains score outside 0..4.")

    if int((merged["score_source"] == "checkpoint_c_new").sum()) != (
        EXPECTED_NOVEL_PAIRS
    ):
        raise ValueError("Complete score map does not contain 387 new scores.")

    if int((merged["score_source"] != "checkpoint_c_new").sum()) != (
        EXPECTED_REUSED_PAIRS
    ):
        raise ValueError("Complete score map does not contain 213 reused scores.")

    counts = merged.groupby("query_id").size()
    if len(counts) != 12 or not (counts == 50).all():
        raise ValueError(
            f"Complete score map must contain 12 x 50 rows; got {counts.to_dict()}"
        )

    return (
        merged[COMPLETE_COLUMNS]
        .sort_values(["query_id", "vector_rank"], kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
        help=(
            "Optional cap on NEW cases scored in this invocation. "
            "0 means score all remaining cases."
        ),
    )
    args = parser.parse_args()

    if args.max_new < 0:
        raise ValueError("--max-new must be >= 0.")

    print("Semantic relevance top-50 diagnostic - Checkpoint C")
    print("---------------------------------------------------")
    print("Purpose: score ONLY frozen novel top-50 pairs")
    print("Judge: semantic_relevance_v0.29.0-r5")
    print(f"Novel pairs: {EXPECTED_NOVEL_PAIRS}")
    print(f"Reused pairs: {EXPECTED_REUSED_PAIRS}")
    print("Human relevance labels: NOT USED")
    print("Aggregate/oracle metrics: NOT CALCULATED")
    print()

    top50, reuse, novel, _, _ = load_and_verify_inputs()
    rubric, config, facet_payload = verify_judge_contract()
    facet_specs = load_facet_specs(facet_payload)
    novel = validate_novel_queries(novel, facet_specs)

    results = load_existing_new_scores(novel)
    completed_ids = {str(row["diagnostic_case_id"]) for row in results}

    if not completed_ids.issubset(set(novel["diagnostic_case_id"])):
        raise ValueError("Existing results contain case IDs outside frozen novel input.")

    remaining = novel.loc[
        ~novel["diagnostic_case_id"].isin(completed_ids)
    ].copy()

    already_completed = len(completed_ids)
    print()
    print("CHECKPOINT C PREFLIGHT")
    print("----------------------")
    print(f"Frozen novel input:             {len(novel)}/387")
    print(f"Previously completed novel:     {already_completed}/387")
    print(f"Remaining before this run:      {len(remaining)}/387")
    print(f"Checkpoint B reusable evidence: {EXPECTED_REUSED_PAIRS}/600")
    print("Existing score reuse conflicts: 0")
    print("Human relevance labels used:    NO")

    if args.max_new > 0:
        remaining = remaining.head(args.max_new).copy()
        print(f"Invocation new-case cap:         {args.max_new}")
    else:
        print("Invocation new-case cap:         NONE (all remaining)")

    judge_model = None
    if not remaining.empty:
        judge_model = OllamaModel(
            model=config["model"],
            base_url=config["base_url"],
            temperature=config["temperature"],
        )

    write_scoring_metadata(
        config=config,
        status=(
            "RUNNING"
            if already_completed < EXPECTED_NOVEL_PAIRS
            else "COMPLETE"
        ),
        completed=already_completed,
    )

    try:
        for _, row in remaining.iterrows():
            case_id = str(row["diagnostic_case_id"])
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

            results.append(
                {
                    "diagnostic_case_id": case_id,
                    "case_id": case_id,
                    "query_id": str(row["query_id"]),
                    "query": str(row["query"]),
                    "rank": int(row["vector_rank"]),
                    "vector_rank": int(row["vector_rank"]),
                    "isbn13": str(row["isbn13"]),
                    "title": str(row["title"]),
                    "authors": canonical_optional_authors(
                        row.get("authors", "")
                    ),
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
                    "book_subject_reason": (
                        semantic_result.book_subject_analysis.reason
                    ),
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
                    "satisfied_qualifier_count": (
                        score_result.satisfied_qualifier_count
                    ),
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
            persist_new_scores(results)

            completed = len(completed_ids)
            write_scoring_metadata(
                config=config,
                status="RUNNING",
                completed=completed,
            )

            print(
                f"{completed:3d}/387 {case_id}: "
                f"score={score_result.score} level={score_result.label} "
                f"title={str(row['title'])[:68]}"
            )

            if completed % 10 == 0:
                print(f"Checkpoint: {completed}/387 novel judgments persisted")

    except Exception as exc:
        write_scoring_metadata(
            config=config,
            status="PARTIAL",
            completed=len(completed_ids),
            error=f"{type(exc).__name__}: {exc}",
        )
        print()
        print(f"Status: PARTIAL ({len(completed_ids)}/387 novel cases)")
        print("Rerun the SAME command to resume; completed cases will not be rejudged.")
        raise

    # A deliberate --max-new cap can leave a healthy partial result without
    # representing it as an execution failure.
    if len(completed_ids) < EXPECTED_NOVEL_PAIRS:
        write_scoring_metadata(
            config=config,
            status="PARTIAL_LIMIT_REACHED",
            completed=len(completed_ids),
        )
        print()
        print("CHECKPOINT C PARTIAL: requested invocation completed successfully")
        print(f"Persisted novel judgments: {len(completed_ids)}/387")
        print("No complete-score artifact or scoring lock has been created yet.")
        print("Rerun the SAME command to continue.")
        return

    # Sort and rewrite the final new-score artifact deterministically.
    final_new = pd.read_csv(
        NEW_SCORES_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    final_new["vector_rank"] = final_new["vector_rank"].astype(int)
    final_new["rank"] = final_new["rank"].astype(int)
    final_new["_q"] = (
        final_new["query_id"].str.extract(r"(\d+)")[0].astype(int)
    )
    final_new = (
        final_new.sort_values(
            ["_q", "vector_rank"],
            kind="stable",
        )
        .drop(columns=["_q"])
        .reset_index(drop=True)
    )
    final_new.to_csv(
        NEW_SCORES_FILE,
        index=False,
        encoding="utf-8",
    )

    if len(final_new) != EXPECTED_NOVEL_PAIRS:
        raise RuntimeError(
            f"Novel scoring completed with {len(final_new)}/387 rows."
        )

    complete = create_complete_score_map(
        top50=top50,
        reuse=reuse,
        new_scores=final_new,
    )
    complete.to_csv(
        COMPLETE_SCORES_FILE,
        index=False,
        encoding="utf-8",
    )

    lock = {
        "status": "TOP50_JUDGE_SCORES_FROZEN_BEFORE_HEADROOM_ANALYSIS",
        "diagnostic_id": "semantic_relevance_top50_diagnostic_v1",
        "run_directory": str(RUN_DIR.relative_to(REPO_ROOT)),
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "judge_execution_patch": config.get("_execution_patch_id"),
        "judge_source_sha256": sha256(JUDGE_FILE),
        "scoring_source_sha256": sha256(SCORING_FILE),
        "facet_spec_sha256": sha256(FACET_SPEC_FILE),
        "judge_config_sha256": sha256(JUDGE_CONFIG_FILE),
        "rubric_sha256": sha256(RUBRIC_FILE),
        "top50_candidates_sha256": sha256(TOP50_CANDIDATES_FILE),
        "score_reuse_map_sha256": sha256(SCORE_REUSE_MAP_FILE),
        "novel_judge_input_sha256": sha256(NOVEL_JUDGE_INPUT_FILE),
        "new_judge_scores_file": NEW_SCORES_FILE.name,
        "new_judge_scores_sha256": sha256(NEW_SCORES_FILE),
        "complete_score_map_file": COMPLETE_SCORES_FILE.name,
        "complete_score_map_sha256": sha256(COMPLETE_SCORES_FILE),
        "reused_pairs": EXPECTED_REUSED_PAIRS,
        "newly_judged_pairs": EXPECTED_NOVEL_PAIRS,
        "complete_pairs": EXPECTED_TOTAL_PAIRS,
        "human_relevance_labels_used": False,
        "aggregate_metrics_calculated": False,
        "oracle_metrics_calculated": False,
        "bootstrap_confidence_intervals_calculated": False,
        "next_stage": "CHECKPOINT_D_TOP50_HEADROOM_ANALYSIS",
    }

    SCORING_LOCK_FILE.write_text(
        json.dumps(lock, indent=2) + "\n",
        encoding="utf-8",
    )

    write_scoring_metadata(
        config=config,
        status="COMPLETE",
        completed=EXPECTED_NOVEL_PAIRS,
    )

    print()
    print("CHECKPOINT C INTEGRITY")
    print("----------------------")
    print(f"Reused frozen judgments:     {EXPECTED_REUSED_PAIRS}/600")
    print(f"New novel judgments:         {EXPECTED_NOVEL_PAIRS}/600")
    print(f"Complete scored candidates:  {len(complete)}/600")
    print("Judge score range:           0..4")
    print("Human relevance labels used: NO")
    print("Aggregate metrics calculated:NO")
    print("Oracle metrics calculated:   NO")
    print()
    print("TOP-50 DIAGNOSTIC CHECKPOINT C: PASS")
    print(f"Run directory: {RUN_DIR}")
    print(f"New judge scores SHA-256: {sha256(NEW_SCORES_FILE)}")
    print(f"Complete score map SHA-256: {sha256(COMPLETE_SCORES_FILE)}")
    print("Next: review Checkpoint C before top-50 headroom/oracle analysis.")


if __name__ == "__main__":
    main()
