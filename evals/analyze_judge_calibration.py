"""
Run blind LLM-judge calibration for Semantic Recommendation Relevance.

This runner is designed for Judge Config v0.4.0.

What changed in v0.4.0
----------------------
Earlier judge versions returned only:

    score
    reason

v0.4.0 uses a richer structured verdict:

    has_relevant_evidence
    match_level
    score
    reason

The judge first classifies the semantic match as one of:

    none       -> 0
    incidental -> 1
    partial    -> 2
    clear      -> 3
    strong     -> 4

The Python runner then validates that the judge's boolean, match level,
and numeric score are mutually consistent.

This is important because the earlier smoke tests showed that the local
judge sometimes described a case as having "no explicit evidence" while
still assigning score 1. The structured verdict makes those internal
inconsistencies detectable.

Blind-evaluation rule
---------------------
The judge may see only:

    query
    title
    authors
    description
    rubric
    judge instructions

It must NOT see:

    human_score
    human_reason
    review_status
    retrieval_rank
    candidate_source
    query_slice

Typical usage
-------------
Smoke test the first five remaining cases:

    uv run python evals/run_judge_calibration.py --limit 5

Resume an existing run:

    uv run python evals/run_judge_calibration.py `
      --resume evals/runs/semantic_relevance/<RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Literal

import pandas as pd
from deepeval.models import OllamaModel
from pydantic import BaseModel, Field


# =============================================================================
# Version pins
# =============================================================================
#
# These constants define the exact evaluation artifacts used by this runner.
#
# Once a judge configuration has been used for a real calibration run,
# do not silently modify it. Create a new version instead.
# =============================================================================

JUDGE_CONFIG_VERSION = "0.4.0"
DATASET_VERSION = "0.1.0"
RUBRIC_VERSION = "0.1.0"


# =============================================================================
# Repository paths
# =============================================================================

# Expected script location:
#   <repo>/evals/run_judge_calibration.py
#
# Therefore parents[1] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"

DATASET_FILE = (
    EVALS_DIR
    / "datasets"
    / f"semantic_relevance_calibration.v{DATASET_VERSION}.csv"
)

RUBRIC_FILE = (
    EVALS_DIR
    / "rubrics"
    / "semantic_relevance"
    / f"semantic_relevance_rubric.v{RUBRIC_VERSION}.json"
)

JUDGE_CONFIG_FILE = (
    EVALS_DIR
    / "judge_configs"
    / f"semantic_relevance_judge.v{JUDGE_CONFIG_VERSION}.json"
)

RUNS_DIR = EVALS_DIR / "runs" / "semantic_relevance"


# =============================================================================
# Structured judge output
# =============================================================================

MatchLevel = Literal[
    "none",
    "incidental",
    "partial",
    "clear",
    "strong",
]


class JudgeVerdict(BaseModel):
    """
    Structured semantic classification returned by the LLM judge.

    The numeric score is still requested from the model, but the runner
    independently validates it against match_level using the versioned
    score_mapping from the judge config.
    """

    has_relevant_evidence: bool = Field(
        description=(
            "True only when the supplied description contains actual positive "
            "evidence for at least one requested concept or close semantic "
            "equivalent."
        )
    )

    match_level: MatchLevel = Field(
        description=(
            "Semantic match classification: none, incidental, partial, clear, "
            "or strong."
        )
    )

    score: Literal[0, 1, 2, 3, 4] = Field(
        description="Numeric score corresponding to match_level."
    )

    reason: str = Field(
        min_length=1,
        description=(
            "Brief evidence-based explanation for the selected match level."
        )
    )


# =============================================================================
# File/config helpers
# =============================================================================

def load_json(path: Path) -> dict:
    """Load a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_inputs(
    dataset: pd.DataFrame,
    rubric: dict,
    config: dict,
) -> None:
    """
    Validate dataset, rubric, and judge-config compatibility before any model
    calls are made.
    """

    required_columns = {
        "case_id",
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

    missing_columns = required_columns - set(dataset.columns)

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {sorted(missing_columns)}"
        )

    if dataset["case_id"].duplicated().any():
        duplicates = dataset.loc[
            dataset["case_id"].duplicated(keep=False),
            "case_id",
        ].tolist()

        raise ValueError(
            f"Duplicate case_id values found: {duplicates}"
        )

    if not dataset["human_score"].notna().all():
        raise ValueError(
            "Calibration dataset contains missing human_score values."
        )

    if not dataset["human_score"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError(
            "human_score must contain only 0, 1, 2, 3, or 4."
        )

    if set(dataset["dataset_version"].astype(str)) != {DATASET_VERSION}:
        raise ValueError(
            f"Dataset rows do not all declare dataset version "
            f"{DATASET_VERSION}."
        )

    if set(dataset["rubric_version"].astype(str)) != {RUBRIC_VERSION}:
        raise ValueError(
            f"Dataset rows do not all declare rubric version "
            f"{RUBRIC_VERSION}."
        )

    if rubric.get("version") != RUBRIC_VERSION:
        raise ValueError(
            f"Rubric file declares version {rubric.get('version')!r}; "
            f"expected {RUBRIC_VERSION!r}."
        )

    if config.get("version") != JUDGE_CONFIG_VERSION:
        raise ValueError(
            f"Judge config declares version {config.get('version')!r}; "
            f"expected {JUDGE_CONFIG_VERSION!r}."
        )

    if config.get("rubric_version") != RUBRIC_VERSION:
        raise ValueError(
            "Judge config rubric_version does not match RUBRIC_VERSION."
        )

    if config.get("dataset_version") != DATASET_VERSION:
        raise ValueError(
            "Judge config dataset_version does not match DATASET_VERSION."
        )

    if config.get("provider") != "ollama":
        raise ValueError(
            "Judge provider must be 'ollama' for this runner."
        )

    expected_mapping = {
        "none": 0,
        "incidental": 1,
        "partial": 2,
        "clear": 3,
        "strong": 4,
    }

    if config.get("score_mapping") != expected_mapping:
        raise ValueError(
            "Judge config score_mapping is missing or does not match "
            f"the expected mapping: {expected_mapping}"
        )


# =============================================================================
# Rubric formatting
# =============================================================================

def format_rubric(rubric: dict) -> str:
    """
    Convert the versioned rubric JSON into readable prompt text.

    The rubric JSON remains the source of truth. We do not duplicate the rubric
    definitions inside this Python file.
    """

    lines: list[str] = [
        f"RUBRIC: {rubric['name']}",
        "",
        "PURPOSE",
        rubric["purpose"],
        "",
        "EVIDENCE RULES",
    ]

    for rule in rubric.get("evidence_rules", []):
        lines.append(f"- {rule}")

    lines.extend(
        [
            "",
            "SCORING DEFINITIONS",
        ]
    )

    for score in [4, 3, 2, 1, 0]:
        item = rubric["scores"][str(score)]

        lines.append(
            f"{score} — {item['label']}: {item['definition']}"
        )

        # Synthetic anchor examples help humans and LLM judges interpret each
        # score consistently.
        for example in item.get("anchor_examples", []):
            lines.append(
                f"  Anchor query: {example['query']}"
            )
            lines.append(
                f"  Anchor description: "
                f"{example['book_description']}"
            )
            lines.append(
                f"  Why this score: {example['why']}"
            )

        lines.append("")

    lines.append("BOUNDARY GUIDANCE")

    for boundary in rubric.get("boundary_examples", []):
        lines.append(
            f"- {boundary['boundary']}: "
            f"{boundary['distinction']}"
        )

    lines.extend(
        [
            "",
            "TIE-BREAK RULE",
            rubric["tie_break_rule"],
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Prompt construction
# =============================================================================

def build_prompt(
    row: pd.Series,
    rubric_text: str,
    config: dict,
) -> str:
    """
    Build the blind prompt for one query-book evaluation unit.

    IMPORTANT:
    Never replace this explicit allow-list with `row.to_dict()`.

    The row contains the human gold score and human reason, which must remain
    hidden from the judge.
    """

    case = {
        "query": str(row["query"]),
        "title": str(row["title"]),
        "authors": str(row["authors"]),
        "description": str(row["description"]),
    }

    judge_instructions = "\n".join(
        f"- {instruction}"
        for instruction in config["instructions"]
    )

    return f"""You are an independent evaluator of a semantic book recommender.

Judge exactly ONE query-book pair using the supplied rubric and decision ladder.

JUDGE INSTRUCTIONS
{judge_instructions}

{rubric_text}

CASE TO EVALUATE

User query:
{case["query"]}

Recommended book title:
{case["title"]}

Authors:
{case["authors"]}

Book description:
{case["description"]}

Return the structured verdict requested by the response schema.
"""


# =============================================================================
# Judge-verdict consistency checks
# =============================================================================

def validate_verdict(
    verdict: JudgeVerdict,
    config: dict,
) -> int:
    """
    Validate semantic consistency between:

        has_relevant_evidence
        match_level
        score

    Returns
    -------
    int
        The deterministic score derived from match_level.

    Why derive the score again?
    ---------------------------
    The LLM is responsible for the semantic classification. Python is
    responsible for enforcing the mechanical mapping from classification to
    score. This prevents an internally inconsistent verdict such as:

        match_level = "none"
        score = 1
    """

    score_mapping = config["score_mapping"]

    derived_score = score_mapping[
        verdict.match_level
    ]

    # Mechanical mapping must always hold.
    if verdict.score != derived_score:
        raise ValueError(
            "Judge returned inconsistent match_level/score: "
            f"match_level={verdict.match_level!r}, "
            f"score={verdict.score}, "
            f"expected_score={derived_score}."
        )

    # No evidence must map to "none".
    if (
        verdict.has_relevant_evidence is False
        and verdict.match_level != "none"
    ):
        raise ValueError(
            "Judge returned has_relevant_evidence=False but "
            f"match_level={verdict.match_level!r}. "
            "Expected match_level='none'."
        )

    # Any non-none level requires positive evidence.
    if (
        verdict.has_relevant_evidence is True
        and verdict.match_level == "none"
    ):
        raise ValueError(
            "Judge returned has_relevant_evidence=True but "
            "match_level='none'."
        )

    return derived_score


# =============================================================================
# Run metadata / resumability
# =============================================================================

def git_commit_sha() -> str | None:
    """
    Return the current Git commit SHA when available.
    """

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()

    except Exception:
        # Missing Git metadata should not block a local calibration run.
        return None


def create_run_dir(
    resume: str | None,
) -> Path:
    """
    Create a new timestamped run directory or resolve an existing run to resume.
    """

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if resume:
        run_dir = Path(
            resume
        )

        if not run_dir.is_absolute():
            run_dir = (
                REPO_ROOT
                / run_dir
            )

        if not run_dir.exists():
            raise FileNotFoundError(
                f"Resume directory not found: {run_dir}"
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


def write_metadata(
    run_dir: Path,
    config: dict,
    status: str,
    completed_cases: int,
    total_cases: int,
) -> None:
    """
    Persist run provenance and progress.

    This file is rewritten throughout execution so partial runs remain
    reproducible.
    """

    metadata = {
        "run_id": run_dir.name,
        "status": status,
        "updated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        # Code provenance.
        "git_commit_sha": git_commit_sha(),

        # Versioned evaluation artifacts.
        "rubric_version": RUBRIC_VERSION,
        "calibration_dataset_version": DATASET_VERSION,
        "judge_config_version": JUDGE_CONFIG_VERSION,

        # Judge runtime configuration.
        "judge_provider": config["provider"],
        "judge_model": config["model"],
        "judge_base_url": config["base_url"],
        "temperature": config["temperature"],

        # Runtime/library versions.
        "python_version": platform.python_version(),
        "deepeval_version": package_version("deepeval"),

        # Progress.
        "completed_cases": completed_cases,
        "total_cases": total_cases,
    }

    metadata_file = (
        run_dir
        / "run_metadata.json"
    )

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# DeepEval/Ollama output normalization
# =============================================================================

def unpack_generation(
    generated,
) -> JudgeVerdict:
    """
    Normalize DeepEval structured generation into JudgeVerdict.

    Depending on the DeepEval/model integration version, structured generation
    may return:
    - the Pydantic object directly,
    - a dict,
    - or a tuple whose first item is the structured result.
    """

    if isinstance(
        generated,
        tuple,
    ):
        verdict = generated[0]
    else:
        verdict = generated

    if isinstance(
        verdict,
        dict,
    ):
        verdict = JudgeVerdict(
            **verdict
        )

    if not isinstance(
        verdict,
        JudgeVerdict,
    ):
        raise TypeError(
            f"Unexpected judge response type: {type(verdict)!r}"
        )

    return verdict


# =============================================================================
# Main calibration loop
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Blindly score semantic relevance calibration cases "
            "with Judge Config v0.4.0 using a local Ollama model."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Score only the first N remaining cases. "
            "Useful for smoke testing."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume an existing run directory instead of "
            "creating a new run."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. Ensure required versioned artifacts exist.
    # -------------------------------------------------------------------------
    for required_file in [
        DATASET_FILE,
        RUBRIC_FILE,
        JUDGE_CONFIG_FILE,
    ]:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_file}"
            )

    # -------------------------------------------------------------------------
    # 2. Load dataset, rubric, and judge configuration.
    # -------------------------------------------------------------------------
    dataset = pd.read_csv(
        DATASET_FILE,
        dtype={
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

    validate_inputs(
        dataset=dataset,
        rubric=rubric,
        config=config,
    )

    # -------------------------------------------------------------------------
    # 3. Create or resume the run.
    # -------------------------------------------------------------------------
    run_dir = create_run_dir(
        args.resume
    )

    results_file = (
        run_dir
        / "judge_results.csv"
    )

    if results_file.exists():
        existing_results = pd.read_csv(
            results_file,
            encoding="utf-8",
        )

        completed_case_ids = set(
            existing_results[
                "case_id"
            ].astype(str)
        )

        results = (
            existing_results
            .to_dict(
                orient="records"
            )
        )

    else:
        completed_case_ids = set()
        results = []

    remaining_cases = dataset[
        ~dataset[
            "case_id"
        ]
        .astype(str)
        .isin(completed_case_ids)
    ].copy()

    # --limit applies only to currently unscored cases.
    if args.limit is not None:
        remaining_cases = remaining_cases.head(
            args.limit
        )

    # -------------------------------------------------------------------------
    # 4. Initialize local Ollama judge.
    # -------------------------------------------------------------------------
    judge_model = OllamaModel(
        model=config["model"],
        base_url=config["base_url"],
        temperature=config["temperature"],
    )

    rubric_text = format_rubric(
        rubric
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

    # -------------------------------------------------------------------------
    # 5. Score each remaining query-book pair.
    # -------------------------------------------------------------------------
    for _, row in remaining_cases.iterrows():

        prompt = build_prompt(
            row=row,
            rubric_text=rubric_text,
            config=config,
        )

        generated = judge_model.generate(
            prompt=prompt,
            schema=JudgeVerdict,
        )

        verdict = unpack_generation(
            generated
        )

        # Derive and validate the numeric score mechanically.
        judge_score = validate_verdict(
            verdict=verdict,
            config=config,
        )

        results.append(
            {
                "case_id": str(
                    row["case_id"]
                ),
                "has_relevant_evidence": (
                    verdict.has_relevant_evidence
                ),
                "match_level": (
                    verdict.match_level
                ),
                "judge_score": (
                    judge_score
                ),
                "judge_reason": (
                    verdict.reason
                ),
            }
        )

        # Persist after every case so long local-model runs are resumable.
        pd.DataFrame(
            results
        ).to_csv(
            results_file,
            index=False,
            encoding="utf-8",
        )

        write_metadata(
            run_dir=run_dir,
            config=config,
            status="RUNNING",
            completed_cases=len(
                results
            ),
            total_cases=total_cases,
        )

        print(
            f"{len(results):>2}/{total_cases} "
            f"{row['case_id']}: "
            f"evidence={verdict.has_relevant_evidence}, "
            f"level={verdict.match_level}, "
            f"score={judge_score}"
        )

    # -------------------------------------------------------------------------
    # 6. Finalize run status.
    # -------------------------------------------------------------------------
    unique_completed_cases = len(
        {
            str(
                result["case_id"]
            )
            for result in results
        }
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
        completed_cases=(
            unique_completed_cases
        ),
        total_cases=total_cases,
    )

    print()
    print(
        f"Run directory: {run_dir}"
    )
    print(
        f"Status: {final_status}"
    )
    print(
        f"Completed: "
        f"{unique_completed_cases}/{total_cases}"
    )


if __name__ == "__main__":
    main()
