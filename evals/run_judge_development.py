"""
Run development Semantic Recommendation Relevance Judge v0.26.0 on the consumed 90-case development set.

Typical usage
-------------
Deterministic scoring tests:

    uv run python evals/test_facet_scoring.py

Targeted development diagnostics:

    uv run python evals/run_judge_development.py `
      --case-id Q02_R50 `
      --case-id Q04_R50 `
      --case-id Q07_R05

Full 90-case development run:

    uv run python evals/run_judge_development.py

Resume an interrupted run:

    uv run python evals/run_judge_development.py `
      --resume evals/runs/semantic_relevance_v0_26_development/<RUN_ID>

v0.26 architecture
-----------------
    frozen query facets
            ↓
    definition-aware per-facet candidate evidence selection
            ↓
    isolated definition-aware single-span verification
            ↓
    conservative multi-span composition recovery when needed
            ↓
    definition-aware core-only prominence assessment
            ↓
    Python derives absent/incidental/meaningful/strong
            ↓
    Python computes final 0-4 score

The LLM never directly outputs the legacy facet support level or final score.

This dataset is no longer unseen. It combines the previously consumed validation and final-holdout splits and is development/regression evidence only.
"""


from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import pandas as pd
from deepeval.models import OllamaModel

from semantic_relevance_facet_judge import (
    generate_validated_semantic_verdict,
    serialize_facet_assessments,
    serialize_facet_evidence,
    supported_core_facet_texts,
    supported_evidence_by_id,
)
from semantic_relevance_facet_scoring import (
    QueryFacet,
    QueryFacetSpec,
    compute_facet_score,
    derive_facet_support,
    validate_query_facet_spec,
)


# =============================================================================
# Version pins
# =============================================================================
#
# These pins are intentionally explicit.
#
# Once a versioned artifact has participated in a meaningful evaluation run,
# do not silently edit that file. Create a new version instead.
# =============================================================================

JUDGE_CONFIG_VERSION = "0.26.0"
EVALUATION_DATASET_VERSION = "2.1.0"
JUDGE_CALIBRATION_DATASET_VERSION = "0.5.0"
EVALUATION_DATASET_ROLE = "post_holdout_development"
EVALUATION_SCOPE = "consumed_unseen_pool_development"
RUBRIC_VERSION = "0.1.0"
FACET_SPEC_VERSION = "0.9.1"


# =============================================================================
# Repository paths
# =============================================================================
#
# Expected script location:
#
#     <repo>/evals/run_judge_development.py
#
# Therefore parents[1] is the repository root.
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"

DATASET_FILE = (
    EVALS_DIR
    / "datasets"
    / f"semantic_relevance_v0.26_development.v{EVALUATION_DATASET_VERSION}.csv"
)

RUBRIC_FILE = (
    EVALS_DIR
    / "rubrics"
    / "semantic_relevance"
    / f"semantic_relevance_rubric.v{RUBRIC_VERSION}.json"
)

FACET_SPEC_FILE = (
    EVALS_DIR
    / "facets"
    / "semantic_relevance"
    / f"semantic_relevance_query_facets.v{FACET_SPEC_VERSION}.json"
)

JUDGE_CONFIG_FILE = (
    EVALS_DIR
    / "judge_configs"
    / f"semantic_relevance_judge.v{JUDGE_CONFIG_VERSION}.json"
)

RUNS_DIR = (
    EVALS_DIR
    / "runs"
    / "semantic_relevance_v0_26_development"
)


# =============================================================================
# Stable result schema
# =============================================================================
#
# The first seven fields preserve the interface expected by the existing
# analyzer. The remaining fields expose v0.9 isolated evidence-verification decisions
# so failures can be audited without reconstructing the LLM prompts.
# =============================================================================

RESULT_COLUMNS = [
    # Backward-compatible summary fields
    "case_id",
    "has_relevant_evidence",
    "evidence_text",
    "matched_concept",
    "match_level",
    "judge_score",
    "judge_reason",

    # v0.9 execution provenance
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

    # Deterministic support / aggregation diagnostics
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

    # Complete per-facet stage audit artifacts
    "facet_assessments_json",
    "facet_evidence_json",
    "component_evidence_ledger_json",
]


# =============================================================================
# Generic file helpers
# =============================================================================

def load_json(
    path: Path,
) -> dict[str, Any]:
    """
    Load one UTF-8 JSON artifact.

    Including the path in the exception is useful when multiple versioned JSON
    files exist next to each other.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def installed_package_version(
    package_name: str,
) -> str | None:
    """
    Return a package version for run provenance.

    A missing package-version record should not itself prevent a local run.
    """

    try:
        return package_version(
            package_name
        )
    except PackageNotFoundError:
        return None


# =============================================================================
# Frozen facet-spec loading
# =============================================================================

def load_facet_specs(
    facet_payload: dict[str, Any],
) -> dict[str, QueryFacetSpec]:
    """
    Parse the versioned facet JSON into validated QueryFacetSpec objects.

    Returns
    -------
    dict[str, QueryFacetSpec]
        Mapping keyed by query_id.

    Why key by query_id?
    --------------------
    Every candidate for a given query must reuse EXACTLY the same facet
    decomposition. The candidate book must never influence query decomposition.
    """

    raw_queries = facet_payload.get(
        "queries"
    )

    if not isinstance(
        raw_queries,
        list,
    ):
        raise ValueError(
            "Facet specification must contain a 'queries' list."
        )

    specs: dict[str, QueryFacetSpec] = {}

    for raw_query in raw_queries:

        query_id = str(
            raw_query["query_id"]
        )

        if query_id in specs:
            raise ValueError(
                f"Duplicate query_id in facet specification: {query_id}"
            )

        spec = QueryFacetSpec(
            query_id=query_id,
            query=str(
                raw_query["query"]
            ),
            facets=[
                QueryFacet(
                    **raw_facet
                )
                for raw_facet
                in raw_query["facets"]
            ],
        )

        # This deterministic validation catches duplicate facet IDs and facet
        # specs with no core facet.
        validate_query_facet_spec(
            spec
        )

        # v0.26.0 contract: every production facet must carry a frozen,
        # non-empty canonical component set with unique IDs.
        for facet in spec.facets:
            if not facet.required_components:
                raise ValueError(
                    f"{query_id}/{facet.facet_id}: v0.26 facet is missing required_components."
                )
            component_ids = [
                component.component_id for component in facet.required_components
            ]
            if len(component_ids) != len(set(component_ids)):
                raise ValueError(
                    f"{query_id}/{facet.facet_id}: duplicate canonical component IDs: "
                    f"{component_ids}."
                )

        specs[
            query_id
        ] = spec

    return specs


# =============================================================================
# Input/provenance validation
# =============================================================================

def validate_inputs(
    dataset: pd.DataFrame,
    rubric: dict[str, Any],
    config: dict[str, Any],
    facet_payload: dict[str, Any],
    facet_specs: dict[str, QueryFacetSpec],
) -> None:
    """
    Validate all versioned artifacts before making the first LLM call.

    These checks prevent subtle provenance bugs such as:

    - dataset v0.2.0 with a judge config that says dataset v0.1.0;
    - a modified query being evaluated with an old frozen facet decomposition;
    - accidentally running a draft facet specification;
    - resuming with a judge config that points at a different rubric.
    """

    # -------------------------------------------------------------------------
    # Dataset schema required by this runner.
    #
    # Human fields are validated here because this is a labelled development
    # dataset, but they are NEVER placed in the judge prompt.
    # -------------------------------------------------------------------------
    required_columns = {
        "case_id",
        "query_id",
        "query",
        "title",
        "authors",
        "description",
        "human_score",
        "human_reason",
        "review_status",
        "dataset_version",
        "rubric_version",
    }

    missing_columns = (
        required_columns
        - set(
            dataset.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing dataset columns: "
            f"{sorted(missing_columns)}"
        )

    # case_id remains the stable join key between blind judge output and the
    # later human-vs-judge analysis.
    if dataset[
        "case_id"
    ].duplicated().any():

        duplicates = dataset.loc[
            dataset[
                "case_id"
            ].duplicated(
                keep=False
            ),
            "case_id",
        ].tolist()

        raise ValueError(
            "Duplicate case_id values found: "
            f"{duplicates}"
        )

    # -------------------------------------------------------------------------
    # v0.26 consumed-development contract: 60 historical U_ cases +
    # 30 consumed fresh-validation U2_ cases. The untouched holdout is absent.
    # -------------------------------------------------------------------------
    if len(dataset) != 90:
        raise ValueError(
            "The v0.26 development dataset must contain exactly 90 consumed cases; "
            f"found {len(dataset)}."
        )

    prefixes_ok = dataset["case_id"].astype(str).str.startswith(("U_", "U2_"))
    if not prefixes_ok.all():
        raise ValueError(
            "v0.26 development case IDs must use historical U_ or consumed-validation U2_ prefixes."
        )
    if int(dataset["case_id"].astype(str).str.startswith("U_").sum()) != 60:
        raise ValueError("Expected exactly 60 historical U_ development cases.")
    if int(dataset["case_id"].astype(str).str.startswith("U2_").sum()) != 30:
        raise ValueError("Expected exactly 30 consumed U2_ validation cases.")

    # -------------------------------------------------------------------------
    # Human gold-label completeness.
    # -------------------------------------------------------------------------
    if not dataset[
        "human_score"
    ].notna().all():
        raise ValueError(
            "Calibration dataset contains missing human_score values."
        )

    if not dataset[
        "human_score"
    ].isin(
        [0, 1, 2, 3, 4]
    ).all():
        raise ValueError(
            "human_score must contain only 0, 1, 2, 3, or 4."
        )

    # -------------------------------------------------------------------------
    # Dataset/rubric provenance.
    # -------------------------------------------------------------------------
    if set(
        dataset[
            "dataset_version"
        ].astype(str)
    ) != {
        EVALUATION_DATASET_VERSION
    }:
        raise ValueError(
            "Development dataset rows do not all declare "
            f"dataset version {EVALUATION_DATASET_VERSION}."
        )

    if set(
        dataset[
            "rubric_version"
        ].astype(str)
    ) != {
        RUBRIC_VERSION
    }:
        raise ValueError(
            "Dataset rows do not all declare "
            f"rubric version {RUBRIC_VERSION}."
        )

    if str(
        rubric.get(
            "version"
        )
    ) != RUBRIC_VERSION:
        raise ValueError(
            "Rubric file declares version "
            f"{rubric.get('version')!r}; "
            f"expected {RUBRIC_VERSION!r}."
        )

    # -------------------------------------------------------------------------
    # Judge-config provenance.
    # -------------------------------------------------------------------------
    if str(
        config.get(
            "version"
        )
    ) != JUDGE_CONFIG_VERSION:
        raise ValueError(
            "Judge config declares version "
            f"{config.get('version')!r}; "
            f"expected {JUDGE_CONFIG_VERSION!r}."
        )

    if str(
        config.get(
            "dataset_version"
        )
    ) != JUDGE_CALIBRATION_DATASET_VERSION:
        raise ValueError(
            "Judge v0.26.0 config remains pinned to its original "
            "calibration/development dataset version "
            f"{JUDGE_CALIBRATION_DATASET_VERSION}; found "
            f"{config.get('dataset_version')!r}."
        )

    if str(
        config.get(
            "rubric_version"
        )
    ) != RUBRIC_VERSION:
        raise ValueError(
            "Judge config rubric_version does not match "
            f"{RUBRIC_VERSION}."
        )

    if str(
        config.get(
            "facet_spec_version"
        )
    ) != FACET_SPEC_VERSION:
        raise ValueError(
            "Judge config facet_spec_version does not match "
            f"{FACET_SPEC_VERSION}."
        )

    if config.get(
        "provider"
    ) != "ollama":
        raise ValueError(
            "Judge provider must be 'ollama' for this runner."
        )

    # -------------------------------------------------------------------------
    # Frozen facet-spec provenance.
    # -------------------------------------------------------------------------
    if str(
        facet_payload.get(
            "version"
        )
    ) != FACET_SPEC_VERSION:
        raise ValueError(
            "Facet specification declares version "
            f"{facet_payload.get('version')!r}; "
            f"expected {FACET_SPEC_VERSION!r}."
        )

    if facet_payload.get(
        "status"
    ) != "frozen":
        raise ValueError(
            "Facet specification must have status='frozen' before running "
            "Judge v0.26.0."
        )

    if str(
        facet_payload.get(
            "rubric_version"
        )
    ) != RUBRIC_VERSION:
        raise ValueError(
            "Facet specification rubric_version does not match "
            f"{RUBRIC_VERSION}."
        )

    # -------------------------------------------------------------------------
    # Query/facet alignment.
    #
    # This is critical. A query cannot silently change while retaining an old
    # facet decomposition.
    # -------------------------------------------------------------------------
    dataset_query_ids = set(
        dataset[
            "query_id"
        ].astype(str)
    )

    facet_query_ids = set(
        facet_specs
    )

    missing_facet_specs = (
        dataset_query_ids
        - facet_query_ids
    )

    if missing_facet_specs:
        raise ValueError(
            "Dataset contains query IDs with no frozen facet definition: "
            f"{sorted(missing_facet_specs)}"
        )

    for query_id, group in dataset.groupby(
        "query_id"
    ):

        query_id = str(
            query_id
        )

        unique_query_texts = set(
            group[
                "query"
            ].astype(str)
        )

        if len(
            unique_query_texts
        ) != 1:
            raise ValueError(
                f"Dataset query_id {query_id} has multiple query texts."
            )

        dataset_query_text = next(
            iter(
                unique_query_texts
            )
        )

        frozen_query_text = (
            facet_specs[
                query_id
            ].query
        )

        if (
            dataset_query_text
            != frozen_query_text
        ):
            raise ValueError(
                f"Query text mismatch for {query_id}.\n"
                f"Dataset: {dataset_query_text}\n"
                f"Facets:  {frozen_query_text}"
            )


# =============================================================================
# Git/run metadata helpers
# =============================================================================

def git_commit_sha() -> str | None:
    """
    Return the current Git commit SHA when available.

    The SHA lets us associate an evaluation run with a concrete repository
    revision. Commit important evaluation-infrastructure changes before a
    meaningful run so the SHA is actually useful.
    """

    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            text=True,
        ).strip()

    except Exception:
        # Missing Git metadata should not prevent local development evaluation.
        return None


def create_run_dir(
    resume: str | None,
) -> Path:
    """
    Create a timestamped run directory, or resolve --resume.

    UTC run IDs remain unambiguous across machines and time zones.
    """

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if resume:

        run_dir = Path(
            resume
        )

        # Support either an absolute path or repository-relative path.
        if not run_dir.is_absolute():
            run_dir = (
                REPO_ROOT
                / run_dir
            )

        if not run_dir.exists():
            raise FileNotFoundError(
                "Resume directory not found: "
                f"{run_dir}"
            )

        return run_dir

    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_dir = (
        RUNS_DIR
        / run_id
    )

    run_dir.mkdir(
        parents=False,
        exist_ok=False,
    )

    return run_dir


def metadata_file_for(
    run_dir: Path,
) -> Path:
    return (
        run_dir
        / "run_metadata.json"
    )


def validate_resume_metadata(
    run_dir: Path,
) -> None:
    """
    Reject attempts to resume a run created with a different evaluation contract.

    This prevents a single run directory from containing a mixture of v0.5 and
    older facet-judge results, or different facet/dataset versions.
    """

    metadata_file = metadata_file_for(
        run_dir
    )

    if not metadata_file.exists():
        # A newly created run has no metadata yet.
        return

    metadata = load_json(
        metadata_file
    )

    expected_values = {
        "judge_config_version": (
            JUDGE_CONFIG_VERSION
        ),
        "calibration_dataset_version": (
            JUDGE_CALIBRATION_DATASET_VERSION
        ),
        "evaluation_dataset_version": (
            EVALUATION_DATASET_VERSION
        ),
        "evaluation_dataset_role": (
            EVALUATION_DATASET_ROLE
        ),
        "evaluation_scope": (
            EVALUATION_SCOPE
        ),
        "source_splits_consumed": True,
        "development_dataset_post_holdout": True,
        "judge_behavior_frozen": False,
        "rubric_version": (
            RUBRIC_VERSION
        ),
        "facet_spec_version": (
            FACET_SPEC_VERSION
        ),
    }

    mismatches = []

    for key, expected in expected_values.items():

        actual = metadata.get(
            key
        )

        if str(
            actual
        ) != str(
            expected
        ):
            mismatches.append(
                f"{key}: run={actual!r}, expected={expected!r}"
            )

    if mismatches:
        raise ValueError(
            "Cannot resume this run because its provenance does not match "
            "the v0.26 development runner:\n- "
            + "\n- ".join(
                mismatches
            )
        )


def write_metadata(
    run_dir: Path,
    config: dict[str, Any],
    status: str,
    completed_cases: int,
    total_cases: int,
) -> None:
    """
    Persist run provenance and current progress.

    This file is rewritten after every successful case. If a local-model run
    fails midway, the partial run still records exactly which artifacts and
    model settings were in use.
    """

    metadata = {
        "run_id": (
            run_dir.name
        ),

        "status": (
            status
        ),

        "updated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        # ---------------------------------------------------------------------
        # Code provenance
        # ---------------------------------------------------------------------
        "git_commit_sha": (
            git_commit_sha()
        ),

        # ---------------------------------------------------------------------
        # Versioned evaluation artifacts
        # ---------------------------------------------------------------------
        "rubric_version": (
            RUBRIC_VERSION
        ),

        "calibration_dataset_version": (
            JUDGE_CALIBRATION_DATASET_VERSION
        ),
        "evaluation_dataset_version": (
            EVALUATION_DATASET_VERSION
        ),
        "evaluation_dataset_role": (
            EVALUATION_DATASET_ROLE
        ),
        "evaluation_scope": (
            EVALUATION_SCOPE
        ),
        "source_splits_consumed": True,
        "development_dataset_post_holdout": True,
        "judge_behavior_frozen": False,

        "judge_config_version": (
            JUDGE_CONFIG_VERSION
        ),

        "facet_spec_version": (
            FACET_SPEC_VERSION
        ),

        "evaluation_dataset_file": (
            str(DATASET_FILE.relative_to(REPO_ROOT))
        ),

        "facet_scoring_module": (
            "evals/semantic_relevance_facet_scoring.py"
        ),

        "facet_judge_module": (
            "evals/semantic_relevance_facet_judge.py"
        ),

        # ---------------------------------------------------------------------
        # Judge runtime configuration
        # ---------------------------------------------------------------------
        "judge_provider": (
            config[
                "provider"
            ]
        ),

        "judge_model": (
            config[
                "model"
            ]
        ),

        "judge_base_url": (
            config[
                "base_url"
            ]
        ),

        "temperature": (
            config[
                "temperature"
            ]
        ),

        # ---------------------------------------------------------------------
        # Runtime/library versions
        # ---------------------------------------------------------------------
        "python_version": (
            platform.python_version()
        ),

        "deepeval_version": (
            installed_package_version(
                "deepeval"
            )
        ),

        "pydantic_version": (
            installed_package_version(
                "pydantic"
            )
        ),

        "pandas_version": (
            installed_package_version(
                "pandas"
            )
        ),

        # ---------------------------------------------------------------------
        # Run progress
        # ---------------------------------------------------------------------
        "completed_cases": (
            completed_cases
        ),

        "total_cases": (
            total_cases
        ),
    }

    metadata_file_for(
        run_dir
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# Result serialization helpers
# =============================================================================

def matched_core_concepts(
    spec: QueryFacetSpec,
    verdict,
) -> str | None:
    """
    Legacy matched_concept summary from Python-derived positive CORE support.
    """

    matched = supported_core_facet_texts(
        spec=spec,
        verdict=verdict,
    )

    if not matched:
        return None

    return "; ".join(
        matched
    )


def combined_evidence_text(
    spec: QueryFacetSpec,
    evidence_by_id: dict[str, str],
    overall_score: int,
) -> str | None:
    """
    Produce a readable legacy evidence summary for positive overall scores.

    Full per-facet evidence—including qualifier-only evidence on score-0
    cases—remains available in `facet_evidence_json`.

    Returning null here for score 0 keeps the legacy compatibility fields
    internally intuitive:

        score 0
        has_relevant_evidence=False
        evidence_text=None
        matched_concept=None
    """

    if (
        overall_score == 0
        or not evidence_by_id
    ):
        return None

    parts = []

    for facet in spec.facets:

        evidence = evidence_by_id.get(
            facet.facet_id
        )

        if evidence:
            parts.append(
                f"{facet.facet_id}: {evidence}"
            )

    if not parts:
        return None

    return " | ".join(
        parts
    )


def persist_results(
    results_file: Path,
    results: list[dict[str, Any]],
) -> None:
    """
    Persist after every successful case.

    Long local-Ollama runs should be safely resumable without losing completed
    cases.
    """

    pd.DataFrame(
        results,
        columns=RESULT_COLUMNS,
    ).to_csv(
        results_file,
        index=False,
        encoding="utf-8",
    )


def load_existing_results(
    results_file: Path,
) -> list[dict[str, Any]]:
    """
    Load persisted compatible rows when --resume is used.

    A schema mismatch fails rather than silently mixing result formats.
    """

    if not results_file.exists():
        return []

    existing_results = pd.read_csv(
        results_file,
        dtype={
            "case_id": str,
        },
        encoding="utf-8",
        keep_default_na=False,
    )

    missing_columns = (
        set(
            RESULT_COLUMNS
        )
        - set(
            existing_results.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Existing judge_results.csv is not compatible with "
            "Judge v0.26.0. Missing columns: "
            f"{sorted(missing_columns)}"
        )

    return existing_results[
        RESULT_COLUMNS
    ].to_dict(
        orient="records"
    )


# =============================================================================
# Main post-holdout development loop
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run development facet-based Judge Config v0.26.0 on the "
            "90-case consumed development set using local Ollama."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Score only the first N currently unscored cases. "
            "Useful for development smoke tests."
        ),
    )

    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        dest="case_ids",
        help=(
            "Evaluate only the requested case ID. May be supplied multiple "
            "times for a targeted development/regression slice."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume an existing compatible run directory instead of creating "
            "a new run."
        ),
    )

    args = parser.parse_args()

    if (
        args.limit is not None
        and args.limit < 1
    ):
        raise ValueError(
            "--limit must be >= 1."
        )

    # =========================================================================
    # 1. Verify all versioned artifacts exist.
    # =========================================================================

    for required_file in [
        DATASET_FILE,
        RUBRIC_FILE,
        FACET_SPEC_FILE,
        JUDGE_CONFIG_FILE,
    ]:

        if not required_file.exists():
            raise FileNotFoundError(
                "Required file not found: "
                f"{required_file}"
            )

    # =========================================================================
    # 2. Load dataset + evaluation artifacts.
    # =========================================================================

    dataset = pd.read_csv(
        DATASET_FILE,
        dtype={
            "case_id": str,
            "query_id": str,
            "isbn13": str,
        },
        encoding="utf-8",
    )

    rubric = load_json(
        RUBRIC_FILE
    )

    config = load_json(
        JUDGE_CONFIG_FILE
    )

    facet_payload = load_json(
        FACET_SPEC_FILE
    )

    facet_specs = load_facet_specs(
        facet_payload
    )

    validate_inputs(
        dataset=dataset,
        rubric=rubric,
        config=config,
        facet_payload=facet_payload,
        facet_specs=facet_specs,
    )

    # =========================================================================
    # 3. Create a new run directory or resolve a resume directory.
    # =========================================================================

    run_dir = create_run_dir(
        args.resume
    )

    # If this is a resumed directory, ensure it was created under the same
    # evaluation contract.
    validate_resume_metadata(
        run_dir
    )

    results_file = (
        run_dir
        / "judge_results.csv"
    )

    results = load_existing_results(
        results_file
    )

    completed_case_ids = {
        str(
            result[
                "case_id"
            ]
        )
        for result
        in results
    }

    # Select only cases that have not already been persisted.
    remaining_cases = dataset[
        ~dataset[
            "case_id"
        ]
        .astype(str)
        .isin(
            completed_case_ids
        )
    ].copy()

    # Optional targeted diagnostic subset.
    #
    # All current Q01-Q12 cases are development data, so targeted case runs are
    # useful for testing known failure modes before paying the cost of a full
    # local-model run.
    if args.case_ids:

        requested_case_ids = {
            str(case_id)
            for case_id
            in args.case_ids
        }

        available_case_ids = set(
            dataset[
                "case_id"
            ].astype(str)
        )

        unknown_case_ids = (
            requested_case_ids
            - available_case_ids
        )

        if unknown_case_ids:
            raise ValueError(
                "Unknown --case-id values: "
                f"{sorted(unknown_case_ids)}"
            )

        remaining_cases = remaining_cases[
            remaining_cases[
                "case_id"
            ]
            .astype(str)
            .isin(
                requested_case_ids
            )
        ].copy()

    # --limit applies to CURRENTLY UNSCORED cases only, matching the previous
    # runner's resume behavior.
    if args.limit is not None:
        remaining_cases = remaining_cases.head(
            args.limit
        )

    # =========================================================================
    # 4. Initialize the local Ollama judge.
    # =========================================================================

    judge_model = OllamaModel(
        model=config[
            "model"
        ],
        base_url=config[
            "base_url"
        ],
        temperature=config[
            "temperature"
        ],
    )

    total_cases = len(
        dataset
    )

    write_metadata(
        run_dir=run_dir,
        config=config,
        status="RUNNING",
        completed_cases=len(
            completed_case_ids
        ),
        total_cases=total_cases,
    )

    # =========================================================================
    # 5. Evaluate each remaining query-book pair.
    # =========================================================================
    #
    # Per frozen facet:
    #     A. LLM selects one candidate source span.
    #     B. A separate isolated LLM call verifies facet vs exact span.
    #     C. If supported CORE facet, a separate call assesses prominence.
    #
    # Python then derives absent/incidental/meaningful/strong and computes 0-4.
    #
    # Human labels are never passed to any judge stage.
    # =========================================================================

    try:

        for _, row in remaining_cases.iterrows():

            case_id = str(
                row[
                    "case_id"
                ]
            )

            query_id = str(
                row[
                    "query_id"
                ]
            )

            spec = facet_specs[
                query_id
            ]

            # -----------------------------------------------------------------
            # Stage 1: LLM evaluates each frozen facet independently.
            # -----------------------------------------------------------------
            semantic_result = generate_validated_semantic_verdict(
                judge_model=judge_model,
                row=row,
                spec=spec,
                rubric=rubric,
                config=config,
            )

            verdict = semantic_result.verdict

            # -----------------------------------------------------------------
            # Stage 2: deterministic Python owns the final score.
            #
            # The LLM has no opportunity to output or override this value.
            # -----------------------------------------------------------------
            score_result = compute_facet_score(
                spec=spec,
                verdict=verdict,
            )

            # -----------------------------------------------------------------
            # Exact source evidence was selected inside the same grounded facet
            # assessment that produced relation/prominence.
            # -----------------------------------------------------------------
            evidence_by_id = (
                semantic_result.evidence_by_id
            )

            positive_evidence_by_id = supported_evidence_by_id(
                spec=spec,
                verdict=verdict,
                evidence_by_id=evidence_by_id,
            )

            # -----------------------------------------------------------------
            # Legacy compatibility fields.
            #
            # `has_relevant_evidence` refers to OVERALL relevance, not merely a
            # positive qualifier. A historical book unrelated to war can have
            # qualifier evidence while still correctly receiving overall 0.
            # -----------------------------------------------------------------
            has_relevant_evidence = (
                score_result.score
                > 0
            )

            matched_concept = (
                matched_core_concepts(
                    spec=spec,
                    verdict=verdict,
                )
                if has_relevant_evidence
                else None
            )

            evidence_text = combined_evidence_text(
                spec=spec,
                evidence_by_id=positive_evidence_by_id,
                overall_score=score_result.score,
            )

            # Full detail is persisted independently of compatibility fields.
            facet_assessments_json = serialize_facet_assessments(
                spec=spec,
                verdict=verdict,
            )

            facet_evidence_json = serialize_facet_evidence(
                semantic_result.candidate_evidence_by_id
            )

            component_evidence_ledger_json = json.dumps(
                semantic_result.component_evidence_ledger_by_id,
                ensure_ascii=False,
            )

            results.append(
                {
                    # ---------------------------------------------------------
                    # Existing analyzer-compatible fields.
                    # ---------------------------------------------------------
                    "case_id": (
                        case_id
                    ),

                    "has_relevant_evidence": (
                        has_relevant_evidence
                    ),

                    "evidence_text": (
                        evidence_text
                    ),

                    "matched_concept": (
                        matched_concept
                    ),

                    "match_level": (
                        score_result.label
                    ),

                    "judge_score": (
                        score_result.score
                    ),

                    "judge_reason": (
                        verdict.overall_reason
                    ),

                    "semantic_generation_mode": (
                        semantic_result.generation_mode
                    ),

                    "semantic_stage_retry_count": (
                        semantic_result.stage_retry_count
                    ),

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

                    # ---------------------------------------------------------
                    # Deterministic scoring diagnostics.
                    # ---------------------------------------------------------
                    "core_facet_count": (
                        score_result.core_facet_count
                    ),

                    "incidental_core_count": (
                        score_result.incidental_core_count
                    ),

                    "meaningful_core_count": (
                        score_result.meaningful_core_count
                    ),

                    "meaningful_core_coverage": (
                        score_result.meaningful_core_coverage
                    ),

                    "strong_core_count": (
                        score_result.strong_core_count
                    ),

                    "direct_core_count": (
                        score_result.direct_core_count
                    ),

                    "entailed_core_count": (
                        score_result.entailed_core_count
                    ),

                    "adjacent_core_count": (
                        score_result.adjacent_core_count
                    ),

                    "unsupported_core_count": (
                        score_result.unsupported_core_count
                    ),

                    "qualifier_count": (
                        score_result.qualifier_count
                    ),

                    "satisfied_qualifier_count": (
                        score_result.satisfied_qualifier_count
                    ),

                    "qualifier_cap_applied": (
                        score_result.qualifier_cap_applied
                    ),

                    "clear_rule_applied": (
                        score_result.clear_rule_applied
                    ),

                    "scoring_explanation": (
                        score_result.explanation
                    ),

                    # ---------------------------------------------------------
                    # Complete facet-level audit artifacts.
                    # ---------------------------------------------------------
                    "facet_assessments_json": (
                        facet_assessments_json
                    ),

                    "facet_evidence_json": (
                        facet_evidence_json
                    ),

                    "component_evidence_ledger_json": (
                        component_evidence_ledger_json
                    ),
                }
            )

            completed_case_ids.add(
                case_id
            )

            # Persist the successful case BEFORE moving to the next case.
            persist_results(
                results_file=results_file,
                results=results,
            )

            write_metadata(
                run_dir=run_dir,
                config=config,
                status="RUNNING",
                completed_cases=len(
                    completed_case_ids
                ),
                total_cases=total_cases,
            )

            facet_by_id = {
                facet.facet_id: facet
                for facet
                in spec.facets
            }

            facet_summary = ", ".join(
                (
                    f"{assessment.facet_id}="
                    f"{derive_facet_support(facet_by_id[assessment.facet_id], assessment).value}/"
                    f"{assessment.verification_relation.value}/"
                    f"{assessment.prominence.value}/"
                    f"candidates={len(assessment.candidate_evidence_span_ids)}"
                )
                for assessment
                in verdict.assessments
            )

            print(
                f"{len(completed_case_ids):>2}/{total_cases} "
                f"{case_id}: "
                f"level={score_result.label}, "
                f"score={score_result.score}, "
                f"semantic_mode={semantic_result.generation_mode}, "
                f"direct_cues={semantic_result.deterministic_direct_cue_count}, "
                f"polarity_blocked={semantic_result.deterministic_cue_polarity_blocked_count}, "
                f"composite_verifiers={semantic_result.composite_verification_attempt_count}, "
                f"composites={semantic_result.composite_verification_count}, "
                f"prechecks={semantic_result.hard_exclusion_precheck_attempt_count}, "
                f"precheck_triggers={semantic_result.hard_exclusion_precheck_trigger_count}, "
                f"facets=[{facet_summary}]"
            )

    except Exception:

        # Persist FAILED status before surfacing the original exception. The
        # successfully written result rows remain resumable.
        write_metadata(
            run_dir=run_dir,
            config=config,
            status="FAILED",
            completed_cases=len(
                completed_case_ids
            ),
            total_cases=total_cases,
        )

        raise

    # =========================================================================
    # 6. Finalize run status.
    # =========================================================================

    unique_completed_cases = len(
        completed_case_ids
    )

    final_status = (
        "COMPLETED"
        if unique_completed_cases
        == total_cases
        else "PARTIAL"
    )

    write_metadata(
        run_dir=run_dir,
        config=config,
        status=final_status,
        completed_cases=unique_completed_cases,
        total_cases=total_cases,
    )

    print()

    print(
        f"Run directory: "
        f"{run_dir}"
    )

    print(
        f"Status: "
        f"{final_status}"
    )

    print(
        f"Completed: "
        f"{unique_completed_cases}/"
        f"{total_cases}"
    )


if __name__ == "__main__":
    main()
