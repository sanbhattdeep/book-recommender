from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

HOLDOUT_RUN = EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"
STAGE_B_RUN = EVALS / "runs/semantic_relevance_v3_posthoc_diagnostic/v2_deep_candidate_prepare"

INPUT_LOCK = STAGE_B_RUN / "deep_candidate_input_lock.json"
INVENTORY_FILE = STAGE_B_RUN / "target_top50_inventory.csv"
NOVEL_INPUT_FILE = STAGE_B_RUN / "novel_deep_candidates_to_judge.csv"
NOVEL_SCORES_FILE = STAGE_B_RUN / "novel_deep_judge_scores.csv"
JUDGING_METADATA_FILE = STAGE_B_RUN / "deep_candidate_judging_metadata.json"

COMPLETE_MAP_FILE = STAGE_B_RUN / "target_top50_complete_score_map.csv"
FINALIZATION_METADATA_FILE = STAGE_B_RUN / "deep_candidate_finalization_metadata.json"
SCORING_LOCK_FILE = STAGE_B_RUN / "deep_candidate_scoring_lock.json"

CONTRACT_FILE = (
    EVALS
    / "system_evaluation"
    / "semantic_relevance_v3_posthoc_stage_b_contract.v0.1.0.json"
)
SPAN_ECHO_PATCH_FILE = (
    EVALS
    / "system_evaluation"
    / "semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"
)
FINALIZATION_PATCH_FILE = (
    EVALS
    / "system_evaluation"
    / "semantic_relevance_v3_posthoc_execution_patch.pandas3-finalization-dtype1.json"
)
HOLDOUT_SCORING_LOCK = HOLDOUT_RUN / "holdout_judge_scoring_lock.json"

EXPECTED_INPUT_LOCK_SHA256 = "ca7340bfc21db93a8b70cdafcf591d6b13a8a31ade55f2de1c954358195af832"
EXPECTED_CONTRACT_SHA256 = "3394bf61c05ce496e940601a56f1eefb98fd2b2f45b0e799e1c7915691c94483"
EXPECTED_SPAN_ECHO_PATCH_SHA256 = "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266"
EXPECTED_FINALIZATION_PATCH_SHA256 = "a38569f44f101fca988682c05a7e03e4c9a50f796443b96360841ca050c3891d"
EXPECTED_HOLDOUT_SCORING_LOCK_SHA256 = "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777"
EXPECTED_NOVEL_SCORES_SHA256 = "5ec85be03a8fe1260810d132f6a2e0e52c572ccf25cbb878d1e29019b8bba07f"

EXPECTED_TARGET_QUERIES = 17
EXPECTED_TOTAL_PAIRS = 850
EXPECTED_REUSED = 226
EXPECTED_NOVEL = 624


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def object_column(length: int) -> pd.Series:
    """Create a pandas-3-safe mixed/nullable staging column."""
    return pd.Series([None] * length, dtype="object")


def verify_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frozen = [
        ("Stage-B input lock", INPUT_LOCK, EXPECTED_INPUT_LOCK_SHA256),
        ("Stage-B contract", CONTRACT_FILE, EXPECTED_CONTRACT_SHA256),
        ("span-echo execution patch", SPAN_ECHO_PATCH_FILE, EXPECTED_SPAN_ECHO_PATCH_SHA256),
        ("finalization dtype patch", FINALIZATION_PATCH_FILE, EXPECTED_FINALIZATION_PATCH_SHA256),
        ("holdout scoring lock", HOLDOUT_SCORING_LOCK, EXPECTED_HOLDOUT_SCORING_LOCK_SHA256),
        ("624-case novel score artifact", NOVEL_SCORES_FILE, EXPECTED_NOVEL_SCORES_SHA256),
    ]
    for label, path, expected in frozen:
        require(path.exists(), f"Missing {label}: {path}")
        actual = sha256(path)
        require(actual == expected, f"{label} changed: expected={expected} actual={actual}")
        print(f"PASS  frozen {label} hash unchanged")

    input_lock = load_json(INPUT_LOCK)
    require(
        input_lock.get("status")
        == "V3_POSTHOC_DEEP_DIAGNOSTIC_INPUT_FROZEN_BEFORE_JUDGING",
        "Stage-B input lock status changed.",
    )
    require(int(input_lock.get("target_query_count", -1)) == EXPECTED_TARGET_QUERIES, "Target query count changed.")
    require(int(input_lock.get("target_top50_pairs", -1)) == EXPECTED_TOTAL_PAIRS, "Target top50 pair count changed.")
    require(int(input_lock.get("reused_score_count", -1)) == EXPECTED_REUSED, "Reused score count changed.")
    require(int(input_lock.get("novel_score_count", -1)) == EXPECTED_NOVEL, "Novel score count changed.")

    for filename, path in [
        ("target_top50_inventory.csv", INVENTORY_FILE),
        ("novel_deep_candidates_to_judge.csv", NOVEL_INPUT_FILE),
    ]:
        expected = input_lock["outputs"][filename]
        require(sha256(path) == expected, f"Frozen Stage-B input changed: {filename}")

    span_patch = load_json(SPAN_ECHO_PATCH_FILE)
    require(
        span_patch.get("patch_id") == "v3-posthoc-prominence-span-echo-normalization1",
        "Unexpected span-echo patch ID.",
    )
    require(span_patch.get("semantic_rules_changed") is False, "Span-echo patch changed semantic rules.")
    require(span_patch.get("deterministic_scoring_rules_changed") is False, "Span-echo patch changed scoring rules.")

    final_patch = load_json(FINALIZATION_PATCH_FILE)
    require(
        final_patch.get("patch_id") == "v3-posthoc-pandas3-finalization-dtype1",
        "Unexpected finalization patch ID.",
    )
    require(final_patch.get("judge_calls") == 0, "Finalization patch unexpectedly permits judge calls.")
    require(final_patch.get("semantic_rules_changed") is False, "Finalization patch changed semantic rules.")
    require(final_patch.get("deterministic_scoring_rules_changed") is False, "Finalization patch changed scoring rules.")

    inventory = pd.read_csv(
        INVENTORY_FILE,
        dtype={"query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    novel_input = pd.read_csv(
        NOVEL_INPUT_FILE,
        dtype={"case_id": str, "query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    scores = pd.read_csv(
        NOVEL_SCORES_FILE,
        dtype={"case_id": str, "query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )

    require(len(inventory) == EXPECTED_TOTAL_PAIRS, "Inventory is not 850 rows.")
    require(len(novel_input) == EXPECTED_NOVEL, "Novel input is not 624 rows.")
    require(len(scores) == EXPECTED_NOVEL, "Persisted novel scores are not 624 rows.")
    require(not scores["case_id"].duplicated().any(), "Duplicate persisted case IDs.")
    require(not scores.duplicated(["query_id", "isbn13"]).any(), "Duplicate persisted query-book scores.")

    expected_cases = set(novel_input["case_id"])
    actual_cases = set(scores["case_id"])
    require(actual_cases == expected_cases, "Persisted 624 cases do not exactly match frozen novel input.")

    for field in ["query_id", "query", "isbn13", "title", "vector_rank"]:
        frozen = novel_input.set_index("case_id")[field].astype(str)
        actual = scores.set_index("case_id")[field].astype(str)
        require(
            (frozen.loc[actual.index] == actual).all(),
            f"Persisted score payload differs from frozen novel input: {field}",
        )

    score_values = pd.to_numeric(scores["judge_score"], errors="raise").astype(int)
    require(score_values.between(0, 4).all(), "Persisted novel score outside 0..4.")

    require(
        "prominence_span_echo_normalization_count" in scores.columns,
        "Persisted score artifact lacks span-echo audit column.",
    )
    patch_counts = pd.to_numeric(
        scores["prominence_span_echo_normalization_count"],
        errors="raise",
    ).astype(int)
    require((patch_counts >= 0).all(), "Negative span-echo normalization count.")

    counts = inventory.groupby("query_id").size()
    require(
        len(counts) == EXPECTED_TARGET_QUERIES and (counts == 50).all(),
        "Inventory is not exact 17x50.",
    )

    return inventory, novel_input, scores, input_lock


def build_complete_map(inventory: pd.DataFrame, new_scores: pd.DataFrame) -> pd.DataFrame:
    base = inventory.copy()
    base["vector_rank"] = pd.to_numeric(base["vector_rank"], errors="raise").astype(int)

    score_cols = [
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
    ns = new_scores[score_cols].copy().rename(
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

    require(int(existing_mask.sum()) == EXPECTED_REUSED, "Complete-map reused count changed.")
    require(int(novel_mask.sum()) == EXPECTED_NOVEL, "Complete-map novel count changed.")
    require(
        merged.loc[novel_mask, "new_judge_score"].notna().all(),
        "Complete map is missing a novel deep judgment.",
    )

    # pandas 3 defaults empty text columns to a dedicated StringDtype. Numeric
    # assignment into such a column raises TypeError. Use explicit object
    # staging columns, then canonicalize judge_score to integer 0..4.
    merged["score_source"] = object_column(len(merged))
    merged["judge_score"] = object_column(len(merged))
    merged["final_match_level"] = object_column(len(merged))
    merged["final_judge_reason"] = object_column(len(merged))
    merged["final_scoring_explanation"] = object_column(len(merged))
    merged["final_facet_assessments_json"] = object_column(len(merged))
    merged["final_facet_evidence_json"] = object_column(len(merged))
    merged["final_component_evidence_ledger_json"] = object_column(len(merged))

    merged.loc[existing_mask, "score_source"] = "frozen_displayed_pair_score"
    merged.loc[existing_mask, "judge_score"] = (
        merged.loc[existing_mask, "existing_judge_score"].astype("object")
    )
    merged.loc[existing_mask, "final_match_level"] = (
        merged.loc[existing_mask, "match_level"].astype("object")
    )
    merged.loc[existing_mask, "final_judge_reason"] = (
        merged.loc[existing_mask, "judge_reason"].astype("object")
    )
    merged.loc[existing_mask, "final_scoring_explanation"] = (
        merged.loc[existing_mask, "scoring_explanation"].astype("object")
    )
    merged.loc[existing_mask, "final_facet_assessments_json"] = (
        merged.loc[existing_mask, "facet_assessments_json"].astype("object")
    )
    merged.loc[existing_mask, "final_facet_evidence_json"] = (
        merged.loc[existing_mask, "facet_evidence_json"].astype("object")
    )
    merged.loc[existing_mask, "final_component_evidence_ledger_json"] = (
        merged.loc[existing_mask, "component_evidence_ledger_json"].astype("object")
    )

    merged.loc[novel_mask, "score_source"] = "posthoc_deep_new"
    merged.loc[novel_mask, "judge_score"] = (
        merged.loc[novel_mask, "new_judge_score"].astype("object")
    )
    merged.loc[novel_mask, "final_match_level"] = (
        merged.loc[novel_mask, "new_match_level"].astype("object")
    )
    merged.loc[novel_mask, "final_judge_reason"] = (
        merged.loc[novel_mask, "new_judge_reason"].astype("object")
    )
    merged.loc[novel_mask, "final_scoring_explanation"] = (
        merged.loc[novel_mask, "new_scoring_explanation"].astype("object")
    )
    merged.loc[novel_mask, "final_facet_assessments_json"] = (
        merged.loc[novel_mask, "new_facet_assessments_json"].astype("object")
    )
    merged.loc[novel_mask, "final_facet_evidence_json"] = (
        merged.loc[novel_mask, "new_facet_evidence_json"].astype("object")
    )
    merged.loc[novel_mask, "final_component_evidence_ledger_json"] = (
        merged.loc[novel_mask, "new_component_evidence_ledger_json"].astype("object")
    )

    scores = pd.to_numeric(merged["judge_score"], errors="raise").astype(int)
    require(scores.between(0, 4).all(), "Complete map score outside 0..4.")
    merged["judge_score"] = scores

    require(len(merged) == EXPECTED_TOTAL_PAIRS, "Complete map row count changed.")
    counts = merged.groupby("query_id").size()
    require(
        len(counts) == EXPECTED_TARGET_QUERIES and (counts == 50).all(),
        "Complete map is not exact 17x50.",
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
    print("Semantic relevance v3 - Stage B deterministic FINALIZATION")
    print("-----------------------------------------------------------")
    print("Input novel judgments: 624/624 already persisted")
    print("Judge execution: DISABLED")
    print("Model execution: DISABLED")
    print("Classification: NOT CALCULATED")
    print("Release metrics/gates: NOT RECALCULATED")
    print("Frozen v3 release decision: UNCHANGED")
    print()

    inventory, novel_input, new_scores, input_lock = verify_inputs()
    print("PASS  exact 624-case score artifact verified before finalization")

    complete = build_complete_map(inventory, new_scores)
    COMPLETE_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(COMPLETE_MAP_FILE, index=False, encoding="utf-8")

    require(len(complete) == EXPECTED_TOTAL_PAIRS, "Written complete map is not 850 rows.")

    patch_counts = pd.to_numeric(
        new_scores["prominence_span_echo_normalization_count"],
        errors="raise",
    ).astype(int)

    metadata = {
        "status": "V3_POSTHOC_DEEP_CANDIDATE_FINALIZATION_COMPLETE",
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_b_input_lock_sha256": EXPECTED_INPUT_LOCK_SHA256,
        "diagnostic_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "source_novel_deep_judge_scores_sha256": EXPECTED_NOVEL_SCORES_SHA256,
        "span_echo_execution_patch_sha256": EXPECTED_SPAN_ECHO_PATCH_SHA256,
        "finalization_patch_sha256": EXPECTED_FINALIZATION_PATCH_SHA256,
        "target_query_count": EXPECTED_TARGET_QUERIES,
        "target_top50_pairs": EXPECTED_TOTAL_PAIRS,
        "reused_frozen_scores": EXPECTED_REUSED,
        "new_posthoc_scores": EXPECTED_NOVEL,
        "complete_map_sha256": sha256(COMPLETE_MAP_FILE),
        "prominence_span_echo_normalized_cases": int((patch_counts > 0).sum()),
        "prominence_span_echo_normalized_entries": int(patch_counts.sum()),
        "judge_calls_performed_during_finalization": 0,
        "model_calls_performed_during_finalization": 0,
        "human_relevance_labels_used": False,
        "classification_calculated": False,
        "release_metrics_recalculated": False,
        "release_gates_recalculated": False,
        "frozen_v3_release_decision_modified": False,
        "next_stage": "POSTHOC_RETRIEVAL_VS_RERANKER_CLASSIFICATION",
    }
    FINALIZATION_METADATA_FILE.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    lock = {
        "status": "V3_POSTHOC_DEEP_JUDGE_SCORES_FROZEN_BEFORE_CLASSIFICATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "stage_b_input_lock_sha256": EXPECTED_INPUT_LOCK_SHA256,
        "holdout_judge_scoring_lock_sha256": EXPECTED_HOLDOUT_SCORING_LOCK_SHA256,
        "span_echo_execution_patch_id": "v3-posthoc-prominence-span-echo-normalization1",
        "span_echo_execution_patch_sha256": EXPECTED_SPAN_ECHO_PATCH_SHA256,
        "finalization_patch_id": "v3-posthoc-pandas3-finalization-dtype1",
        "finalization_patch_sha256": EXPECTED_FINALIZATION_PATCH_SHA256,
        "target_query_count": EXPECTED_TARGET_QUERIES,
        "target_top50_pairs": EXPECTED_TOTAL_PAIRS,
        "reused_frozen_scores": EXPECTED_REUSED,
        "new_posthoc_scores": EXPECTED_NOVEL,
        "novel_deep_judge_scores_sha256": EXPECTED_NOVEL_SCORES_SHA256,
        "target_top50_complete_score_map_sha256": sha256(COMPLETE_MAP_FILE),
        "finalization_metadata_sha256": sha256(FINALIZATION_METADATA_FILE),
        "prominence_span_echo_normalized_cases": int((patch_counts > 0).sum()),
        "prominence_span_echo_normalized_entries": int(patch_counts.sum()),
        "judge_calls_performed_during_finalization": 0,
        "model_calls_performed_during_finalization": 0,
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

    print()
    print("V3 POST-HOC DEEP-CANDIDATE FINALIZATION: PASS")
    print(f"Persisted novel scores verified: {EXPECTED_NOVEL}/{EXPECTED_NOVEL}")
    print(f"Reused frozen scores:           {EXPECTED_REUSED}")
    print(f"Complete target top50 map:      {len(complete)}/{EXPECTED_TOTAL_PAIRS}")
    print(f"Novel score SHA-256:            {EXPECTED_NOVEL_SCORES_SHA256}")
    print(f"Complete map SHA-256:           {sha256(COMPLETE_MAP_FILE)}")
    print(f"Scoring lock SHA-256:           {sha256(SCORING_LOCK_FILE)}")
    print(
        "Prominence span-echo normalized cases:   "
        f"{int((patch_counts > 0).sum())}"
    )
    print(
        "Prominence span-echo normalized entries: "
        f"{int(patch_counts.sum())}"
    )
    print("Judge calls during finalization: 0")
    print("Model calls during finalization: 0")
    print("Classification calculated:       NO")
    print("Frozen v3 release decision modified: NO")
    print("Next: post-hoc retrieval-vs-reranker classification.")


if __name__ == "__main__":
    main()
