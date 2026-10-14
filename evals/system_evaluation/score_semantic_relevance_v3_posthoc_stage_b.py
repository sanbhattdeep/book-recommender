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

HOLDOUT_RUN = EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"
STAGE_B_RUN = EVALS / "runs/semantic_relevance_v3_posthoc_diagnostic/v2_deep_candidate_prepare"

INPUT_LOCK = STAGE_B_RUN / "deep_candidate_input_lock.json"
TARGETS_FILE = STAGE_B_RUN / "target_queries.csv"
INVENTORY_FILE = STAGE_B_RUN / "target_top50_inventory.csv"
REUSED_FILE = STAGE_B_RUN / "reused_displayed_scores.csv"
NOVEL_FILE = STAGE_B_RUN / "novel_deep_candidates_to_judge.csv"
CONTRACT_FILE = (
    EVALS
    / "system_evaluation"
    / "semantic_relevance_v3_posthoc_stage_b_contract.v0.1.0.json"
)

HOLDOUT_SCORING_LOCK = HOLDOUT_RUN / "holdout_judge_scoring_lock.json"
HOLDOUT_FACET_FILE = (
    EVALS
    / "facets"
    / "semantic_relevance"
    / "semantic_relevance_v3_holdout_query_facets.v0.1.0.json"
)
EOL_PATCH_FILE = (
    EVALS
    / "system_evaluation"
    / "semantic_relevance_v3_holdout_execution_patch.eolnorm1.json"
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

NEW_SCORES_FILE = STAGE_B_RUN / "novel_deep_judge_scores.csv"
COMPLETE_MAP_FILE = STAGE_B_RUN / "target_top50_complete_score_map.csv"
METADATA_FILE = STAGE_B_RUN / "deep_candidate_judging_metadata.json"
SCORING_LOCK_FILE = STAGE_B_RUN / "deep_candidate_scoring_lock.json"

EXPECTED_INPUT_LOCK_SHA256 = (
    "ca7340bfc21db93a8b70cdafcf591d6b13a8a31ade55f2de1c954358195af832"
)
EXPECTED_CONTRACT_SHA256 = (
    "3394bf61c05ce496e940601a56f1eefb98fd2b2f45b0e799e1c7915691c94483"
)
EXPECTED_HOLDOUT_SCORING_LOCK_SHA256 = (
    "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777"
)

EXPECTED_TARGET_QUERIES = 17
EXPECTED_TOTAL_PAIRS = 850
EXPECTED_REUSED = 226
EXPECTED_NOVEL = 624

AUDIT_COLUMNS = [
    "case_id",
    "query_id",
    "query",
    "stratum",
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def verify_checkpoint_text_hash(
    *,
    label: str,
    path: Path,
    expected: str,
) -> dict[str, str]:
    raw_sha = sha256(path)
    normalized_sha = lf_normalized_sha256(path)

    if raw_sha == expected:
        mode = "raw_bytes"
    elif normalized_sha == expected:
        mode = "lf_normalized_line_endings"
    else:
        raise ValueError(
            f"{label} differs from the frozen holdout judge runtime: "
            f"expected={expected} raw={raw_sha} lf_normalized={normalized_sha}"
        )

    print(f"PASS  frozen {label} verified ({mode})")
    return {
        "frozen_sha256": expected,
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


def canonical_optional_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text.strip().casefold() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


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
    parts = []
    for facet in spec.facets:
        evidence = evidence_by_id.get(facet.facet_id)
        if evidence:
            parts.append(f"{facet.facet_id}: {evidence}")
    return " | ".join(parts) if parts else None


def persist(rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows, columns=AUDIT_COLUMNS).to_csv(
        NEW_SCORES_FILE,
        index=False,
        encoding="utf-8",
    )


def load_existing(novel: pd.DataFrame) -> list[dict[str, Any]]:
    if not NEW_SCORES_FILE.exists():
        return []

    frame = pd.read_csv(
        NEW_SCORES_FILE,
        dtype={"case_id": str, "query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    missing = set(AUDIT_COLUMNS) - set(frame.columns)
    require(not missing, f"Existing partial score file missing columns: {sorted(missing)}")
    require(not frame["case_id"].duplicated().any(), "Duplicate persisted diagnostic case IDs.")

    valid = novel.set_index("case_id", drop=False)
    for row in frame.to_dict(orient="records"):
        case_id = str(row["case_id"])
        require(case_id in valid.index, f"Persisted case not in frozen novel input: {case_id}")
        frozen = valid.loc[case_id]
        checks = {
            "query_id": str(frozen["query_id"]),
            "query": str(frozen["query"]),
            "isbn13": str(frozen["isbn13"]),
            "title": str(frozen["title"]),
            "vector_rank": str(int(frozen["vector_rank"])),
        }
        for field, expected in checks.items():
            require(
                str(row[field]) == expected,
                f"Persisted case {case_id} changed {field}.",
            )
        score = int(row["judge_score"])
        require(0 <= score <= 4, f"Persisted case {case_id} score outside 0..4.")

    require(len(frame) <= EXPECTED_NOVEL, "Persisted file has too many rows.")
    return frame[AUDIT_COLUMNS].to_dict(orient="records")


def verify_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    require(sha256(INPUT_LOCK) == EXPECTED_INPUT_LOCK_SHA256, "Stage-B input lock changed.")
    require(sha256(CONTRACT_FILE) == EXPECTED_CONTRACT_SHA256, "Stage-B contract changed.")
    require(
        sha256(HOLDOUT_SCORING_LOCK) == EXPECTED_HOLDOUT_SCORING_LOCK_SHA256,
        "Frozen holdout judge scoring lock changed.",
    )

    input_lock = load_json(INPUT_LOCK)
    require(
        input_lock.get("status")
        == "V3_POSTHOC_DEEP_DIAGNOSTIC_INPUT_FROZEN_BEFORE_JUDGING",
        "Stage-B diagnostic input is not frozen before judging.",
    )
    require(int(input_lock.get("target_query_count", -1)) == EXPECTED_TARGET_QUERIES, "Target query count changed.")
    require(int(input_lock.get("target_top50_pairs", -1)) == EXPECTED_TOTAL_PAIRS, "Target top50 pair count changed.")
    require(int(input_lock.get("reused_score_count", -1)) == EXPECTED_REUSED, "Reused score count changed.")
    require(int(input_lock.get("novel_score_count", -1)) == EXPECTED_NOVEL, "Novel score count changed.")
    require(input_lock.get("judge_execution_performed") is False, "Input lock says judging already occurred.")
    require(input_lock.get("release_decision_modified") is False, "Input lock modified v3 release decision.")

    for filename, path in [
        ("target_queries.csv", TARGETS_FILE),
        ("target_top50_inventory.csv", INVENTORY_FILE),
        ("reused_displayed_scores.csv", REUSED_FILE),
        ("novel_deep_candidates_to_judge.csv", NOVEL_FILE),
    ]:
        expected = input_lock["outputs"][filename]
        require(sha256(path) == expected, f"Frozen Stage-B input changed: {filename}")
    print("PASS  frozen Stage-B diagnostic input verified")

    holdout_lock = load_json(HOLDOUT_SCORING_LOCK)
    require(
        holdout_lock.get("status") == "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS",
        "Frozen holdout judge scoring status changed.",
    )
    require(
        holdout_lock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5",
        "Judge candidate changed.",
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

    runtime_provenance = {}
    for label, path, key in [
        ("judge source", JUDGE_FILE, "judge_source_sha256"),
        ("deterministic scoring", SCORING_FILE, "scoring_source_sha256"),
        ("judge config", JUDGE_CONFIG_FILE, "judge_config_sha256"),
        ("rubric", RUBRIC_FILE, "rubric_sha256"),
    ]:
        expected = str(holdout_lock.get(key, ""))
        require(expected, f"Holdout scoring lock missing {key}.")
        runtime_provenance[key] = verify_checkpoint_text_hash(
            label=label,
            path=path,
            expected=expected,
        )

    require(
        sha256(HOLDOUT_FACET_FILE) == holdout_lock.get("holdout_facet_spec_sha256"),
        "Frozen holdout facet spec changed.",
    )

    config = load_json(JUDGE_CONFIG_FILE)
    require(config.get("version") == "0.29.0-r5", "Unexpected judge config version.")
    require(float(config.get("temperature", -1)) == 0.0, "Judge temperature changed.")

    facet_specs = load_facet_specs(load_json(HOLDOUT_FACET_FILE))
    require(len(facet_specs) == 24, "Expected 24 frozen holdout facet specs.")

    novel = pd.read_csv(
        NOVEL_FILE,
        dtype={"case_id": str, "query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    inventory = pd.read_csv(
        INVENTORY_FILE,
        dtype={"query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )

    require(len(novel) == EXPECTED_NOVEL, f"Expected {EXPECTED_NOVEL} novel rows.")
    require(len(inventory) == EXPECTED_TOTAL_PAIRS, f"Expected {EXPECTED_TOTAL_PAIRS} inventory rows.")
    require(not novel["case_id"].duplicated().any(), "Duplicate novel diagnostic case IDs.")
    require(not novel.duplicated(["query_id", "isbn13"]).any(), "Duplicate novel query-book pairs.")

    novel["vector_rank"] = pd.to_numeric(novel["vector_rank"], errors="raise").astype(int)
    require((novel["vector_rank"] > 10).all(), "Novel candidate found inside vector top10.")
    novel["rank"] = novel["vector_rank"]

    for qid, group in novel.groupby("query_id", sort=False):
        require(qid in facet_specs, f"Unknown frozen holdout query: {qid}")
        require(
            set(group["query"].astype(str)) == {facet_specs[qid].query},
            f"{qid}: diagnostic query text differs from frozen holdout facet spec.",
        )

    return novel, inventory, config, runtime_provenance


def write_metadata(
    *,
    config: dict[str, Any],
    runtime_provenance: dict[str, Any],
    completed: int,
    status: str,
    error: str | None = None,
) -> None:
    payload = {
        "diagnostic_id": "semantic-relevance-v3-posthoc-deep-candidate-diagnostic",
        "stage": "POSTHOC_DEEP_CANDIDATE_JUDGING",
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_b_input_lock_sha256": EXPECTED_INPUT_LOCK_SHA256,
        "diagnostic_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "holdout_judge_scoring_lock_sha256": EXPECTED_HOLDOUT_SCORING_LOCK_SHA256,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "judge_model": config["model"],
        "judge_base_url": config["base_url"],
        "temperature": config["temperature"],
        "runtime_provenance": runtime_provenance,
        "target_query_count": EXPECTED_TARGET_QUERIES,
        "target_top50_pairs": EXPECTED_TOTAL_PAIRS,
        "reused_frozen_scores": EXPECTED_REUSED,
        "novel_pairs": EXPECTED_NOVEL,
        "completed_novel_pairs": completed,
        "remaining_novel_pairs": EXPECTED_NOVEL - completed,
        "human_relevance_labels_used": False,
        "classification_calculated": False,
        "release_metrics_recalculated": False,
        "release_gates_recalculated": False,
        "frozen_v3_release_decision_modified": False,
        "python_version": platform.python_version(),
        "deepeval_version": package_ver("deepeval"),
        "pandas_version": package_ver("pandas"),
        "error": error,
    }
    METADATA_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_complete_map(
    inventory: pd.DataFrame,
    new_scores: pd.DataFrame,
) -> pd.DataFrame:
    base = inventory.copy()
    base["vector_rank"] = pd.to_numeric(base["vector_rank"], errors="raise").astype(int)

    ns = new_scores[
        [
            "case_id",
            "query_id",
            "isbn13",
            "judge_score",
            "match_level",
            "judge_reason",
            "scoring_explanation",
            "facet_assessments_json",
            "facet_evidence_json",
            "component_evidence_ledger_json",
        ]
    ].copy()
    ns = ns.rename(
        columns={
            "judge_score": "new_judge_score",
            "match_level": "new_match_level",
            "judge_reason": "new_judge_reason",
            "scoring_explanation": "new_scoring_explanation",
            "facet_assessments_json": "new_facet_assessments_json",
            "facet_evidence_json": "new_facet_evidence_json",
            "component_evidence_ledger_json": "new_component_evidence_ledger_json",
        }
    )

    merged = base.merge(
        ns,
        on=["query_id", "isbn13"],
        how="left",
        validate="one_to_one",
    )

    existing_mask = merged["has_existing_frozen_score"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    novel_mask = ~existing_mask

    require(int(existing_mask.sum()) == EXPECTED_REUSED, "Complete map reused count changed.")
    require(int(novel_mask.sum()) == EXPECTED_NOVEL, "Complete map novel count changed.")
    require(
        merged.loc[novel_mask, "new_judge_score"].notna().all(),
        "Complete map is missing a novel deep judgment.",
    )

    merged["score_source"] = ""
    merged["judge_score"] = ""
    merged["final_match_level"] = ""
    merged["final_judge_reason"] = ""
    merged["final_scoring_explanation"] = ""
    merged["final_facet_assessments_json"] = ""
    merged["final_facet_evidence_json"] = ""
    merged["final_component_evidence_ledger_json"] = ""

    merged.loc[existing_mask, "score_source"] = "frozen_displayed_pair_score"
    merged.loc[existing_mask, "judge_score"] = merged.loc[
        existing_mask, "existing_judge_score"
    ]
    merged.loc[existing_mask, "final_match_level"] = merged.loc[
        existing_mask, "match_level"
    ]
    merged.loc[existing_mask, "final_judge_reason"] = merged.loc[
        existing_mask, "judge_reason"
    ]
    merged.loc[existing_mask, "final_scoring_explanation"] = merged.loc[
        existing_mask, "scoring_explanation"
    ]
    merged.loc[existing_mask, "final_facet_assessments_json"] = merged.loc[
        existing_mask, "facet_assessments_json"
    ]
    merged.loc[existing_mask, "final_facet_evidence_json"] = merged.loc[
        existing_mask, "facet_evidence_json"
    ]
    merged.loc[existing_mask, "final_component_evidence_ledger_json"] = merged.loc[
        existing_mask, "component_evidence_ledger_json"
    ]

    merged.loc[novel_mask, "score_source"] = "posthoc_deep_new"
    merged.loc[novel_mask, "judge_score"] = merged.loc[
        novel_mask, "new_judge_score"
    ]
    merged.loc[novel_mask, "final_match_level"] = merged.loc[
        novel_mask, "new_match_level"
    ]
    merged.loc[novel_mask, "final_judge_reason"] = merged.loc[
        novel_mask, "new_judge_reason"
    ]
    merged.loc[novel_mask, "final_scoring_explanation"] = merged.loc[
        novel_mask, "new_scoring_explanation"
    ]
    merged.loc[novel_mask, "final_facet_assessments_json"] = merged.loc[
        novel_mask, "new_facet_assessments_json"
    ]
    merged.loc[novel_mask, "final_facet_evidence_json"] = merged.loc[
        novel_mask, "new_facet_evidence_json"
    ]
    merged.loc[novel_mask, "final_component_evidence_ledger_json"] = merged.loc[
        novel_mask, "new_component_evidence_ledger_json"
    ]

    scores = pd.to_numeric(merged["judge_score"], errors="raise").astype(int)
    require(scores.between(0, 4).all(), "Complete map score outside 0..4.")
    require(len(merged) == EXPECTED_TOTAL_PAIRS, "Complete map row count changed.")

    counts = merged.groupby("query_id").size()
    require(
        len(counts) == EXPECTED_TARGET_QUERIES and (counts == 50).all(),
        "Complete map is not 17x50.",
    )

    columns = [
        "query_id",
        "query",
        "stratum",
        "vector_rank",
        "isbn13",
        "title",
        "description",
        "authors",
        "score_source",
        "judge_score",
        "final_match_level",
        "final_judge_reason",
        "final_scoring_explanation",
        "final_facet_assessments_json",
        "final_facet_evidence_json",
        "final_component_evidence_ledger_json",
    ]
    return (
        merged[columns]
        .sort_values(["query_id", "vector_rank", "isbn13"], kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume persisted cases after a pure execution failure.",
    )
    args = parser.parse_args()

    print("Semantic relevance v3 - POST-HOC deep-candidate judging")
    print("--------------------------------------------------------")
    print("Purpose: diagnose frozen top50 headroom for v4 development")
    print(f"Target queries: {EXPECTED_TARGET_QUERIES}")
    print(f"Reused frozen scores: {EXPECTED_REUSED}")
    print(f"Novel deep candidates: {EXPECTED_NOVEL}")
    print("Human relevance labels: NOT USED")
    print("Classification: NOT CALCULATED")
    print("Release metrics/gates: NOT RECALCULATED")
    print("Frozen v3 release decision: UNCHANGED")
    print()

    novel, inventory, config, runtime_provenance = verify_inputs()
    facet_specs = load_facet_specs(load_json(HOLDOUT_FACET_FILE))
    rubric = load_json(RUBRIC_FILE)

    results = load_existing(novel)
    if results and not args.resume:
        raise RuntimeError(
            "Persisted post-hoc deep judgments already exist. "
            "Do not restart or overwrite them. Use --resume only after a pure "
            "execution failure."
        )

    completed_ids = {str(row["case_id"]) for row in results}
    frozen_ids = set(novel["case_id"].astype(str))
    require(completed_ids <= frozen_ids, "Persisted cases fall outside frozen novel input.")

    remaining = novel[~novel["case_id"].isin(completed_ids)].copy()

    print(f"Starting/resuming deep diagnostic judge run: {len(completed_ids)}/{EXPECTED_NOVEL} persisted")
    print("Do not inspect or tune semantic results until all novel pairs are frozen.")

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
        status="RUNNING" if not remaining.empty else "COMPLETE",
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

            results.append(
                {
                    "case_id": case_id,
                    "query_id": str(row["query_id"]),
                    "query": str(row["query"]),
                    "stratum": str(row["stratum"]),
                    "vector_rank": int(row["vector_rank"]),
                    "isbn13": str(row["isbn13"]),
                    "title": str(row["title"]),
                    "authors": canonical_optional_text(row.get("authors", "")),
                    "has_relevant_evidence": has_relevant,
                    "evidence_text": combined_evidence_text(
                        spec,
                        positive_evidence,
                        score_result.score,
                    ),
                    "matched_concept": (
                        matched_core_concepts(spec, verdict) if has_relevant else None
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

            if len(completed_ids) % 10 == 0 or len(completed_ids) == EXPECTED_NOVEL:
                print(f"Checkpoint: {len(completed_ids)}/{EXPECTED_NOVEL} persisted")

            write_metadata(
                config=config,
                runtime_provenance=runtime_provenance,
                completed=len(completed_ids),
                status="RUNNING",
            )

    except Exception as exc:
        persist(results)
        write_metadata(
            config=config,
            runtime_provenance=runtime_provenance,
            completed=len(completed_ids),
            status="PARTIAL_EXECUTION_FAILURE",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    require(len(results) == EXPECTED_NOVEL, "Novel deep judging did not complete 624/624.")

    new_scores = pd.DataFrame(results, columns=AUDIT_COLUMNS)
    score_values = pd.to_numeric(new_scores["judge_score"], errors="raise").astype(int)
    require(score_values.between(0, 4).all(), "Novel deep score outside 0..4.")

    complete = create_complete_map(inventory, new_scores)
    complete.to_csv(COMPLETE_MAP_FILE, index=False, encoding="utf-8")

    lock = {
        "status": "V3_POSTHOC_DEEP_JUDGE_SCORES_FROZEN_BEFORE_CLASSIFICATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "stage_b_input_lock_sha256": EXPECTED_INPUT_LOCK_SHA256,
        "holdout_judge_scoring_lock_sha256": EXPECTED_HOLDOUT_SCORING_LOCK_SHA256,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "holdout_facet_spec_sha256": sha256(HOLDOUT_FACET_FILE),
        "judge_source_sha256": load_json(HOLDOUT_SCORING_LOCK)["judge_source_sha256"],
        "scoring_source_sha256": load_json(HOLDOUT_SCORING_LOCK)["scoring_source_sha256"],
        "judge_config_sha256": load_json(HOLDOUT_SCORING_LOCK)["judge_config_sha256"],
        "rubric_sha256": load_json(HOLDOUT_SCORING_LOCK)["rubric_sha256"],
        "target_query_count": EXPECTED_TARGET_QUERIES,
        "target_top50_pairs": EXPECTED_TOTAL_PAIRS,
        "reused_frozen_scores": EXPECTED_REUSED,
        "new_posthoc_scores": EXPECTED_NOVEL,
        "novel_deep_judge_scores_sha256": sha256(NEW_SCORES_FILE),
        "target_top50_complete_score_map_sha256": sha256(COMPLETE_MAP_FILE),
        "human_relevance_labels_used": False,
        "classification_calculated": False,
        "release_metrics_recalculated": False,
        "release_gates_recalculated": False,
        "frozen_v3_release_decision_modified": False,
        "next_stage": "POSTHOC_RETRIEVAL_VS_RERANKER_CLASSIFICATION",
    }

    if SCORING_LOCK_FILE.exists():
        existing = load_json(SCORING_LOCK_FILE)
        for key, value in lock.items():
            if key != "frozen_at_utc":
                require(existing.get(key) == value, f"Existing scoring lock mismatch: {key}")
    else:
        SCORING_LOCK_FILE.write_text(
            json.dumps(lock, indent=2) + "\n",
            encoding="utf-8",
        )

    write_metadata(
        config=config,
        runtime_provenance=runtime_provenance,
        completed=EXPECTED_NOVEL,
        status="COMPLETE",
    )

    print()
    print("V3 POST-HOC DEEP-CANDIDATE JUDGING: PASS")
    print(f"Novel pairs judged: {EXPECTED_NOVEL}/{EXPECTED_NOVEL}")
    print(f"Reused frozen scores: {EXPECTED_REUSED}")
    print(f"Complete target top50 map: {EXPECTED_TOTAL_PAIRS}/{EXPECTED_TOTAL_PAIRS}")
    print(f"Novel score SHA-256: {sha256(NEW_SCORES_FILE)}")
    print(f"Complete map SHA-256: {sha256(COMPLETE_MAP_FILE)}")
    print(f"Scoring lock SHA-256: {sha256(SCORING_LOCK_FILE)}")
    print("Human relevance labels used: NO")
    print("Classification calculated: NO")
    print("Frozen v3 release decision modified: NO")
    print("Next: post-hoc retrieval-vs-reranker classification.")


if __name__ == "__main__":
    main()
