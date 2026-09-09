"""
Run blind LLM-judge calibration for Semantic Recommendation Relevance.

This runner is designed for Judge Config v0.5.0.

Purpose
-------
The script asks a local Ollama LLM judge to score each query-book pair
in the human-labelled calibration dataset.

The LLM must classify semantic relevance using:

    none       -> 0
    incidental -> 1
    partial    -> 2
    clear      -> 3
    strong     -> 4

Judge Config v0.5.0 adds explicit evidence grounding.

For every positive relevance judgment, the judge must identify:

    has_relevant_evidence = True
    evidence_text
    matched_concept
    match_level
    score
    reason

Example:

    {
        "has_relevant_evidence": true,
        "evidence_text": "finding redemption",
        "matched_concept": "redemption",
        "match_level": "clear",
        "score": 3,
        "reason": "..."
    }

Python then validates that:

1. match_level and score agree.
2. evidence=False maps to none/0.
3. evidence=True supplies evidence_text.
4. evidence=True supplies matched_concept.
5. evidence_text actually exists in the supplied book description.

Why this matters
----------------
Earlier judge versions sometimes reasoned:

    "this could be seen as redemption"

and then treated that speculation as actual evidence.

v0.5.0 forces the model to point to concrete supplied text before a
positive relevance judgment can be accepted.

Blind-evaluation rule
---------------------
The judge may see only:

    query
    title
    authors
    description
    rubric
    judge instructions

The judge must NOT see:

    human_score
    human_reason
    review_status
    retrieval_rank
    candidate_source
    query_slice

Typical usage
-------------
Smoke test:

    uv run python evals/run_judge_calibration.py --limit 5

Resume an existing run:

    uv run python evals/run_judge_calibration.py `
      --resume evals/runs/semantic_relevance/<RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import unicodedata
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
# Once an artifact version has been used for a real calibration run, do not
# silently edit that version. Create a new version instead.
# =============================================================================

JUDGE_CONFIG_VERSION = "0.5.0"
DATASET_VERSION = "0.1.0"
RUBRIC_VERSION = "0.1.0"


# =============================================================================
# Repository paths
# =============================================================================
#
# Expected script location:
#
#   <repo>/evals/run_judge_calibration.py
#
# Therefore parents[1] is the repository root.
# =============================================================================

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

RUNS_DIR = (
    EVALS_DIR
    / "runs"
    / "semantic_relevance"
)


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
    Structured semantic-relevance verdict returned by the LLM judge.

    v0.5.0 requires explicit grounding for any positive relevance judgment.
    """

    has_relevant_evidence: bool = Field(
        description=(
            "True only when the supplied book description contains actual "
            "positive evidence for at least one concept requested by the user, "
            "or for a close semantic equivalent. Speculative or inferred "
            "connections do not count as evidence."
        )
    )

    evidence_text: str | None = Field(
        default=None,
        description=(
            "The shortest relevant excerpt or phrase from the supplied book "
            "description that supports has_relevant_evidence=True. "
            "The text must come directly from the supplied description. "
            "Do not paraphrase or invent evidence. "
            "Use null when has_relevant_evidence=False."
        )
    )

    matched_concept: str | None = Field(
        default=None,
        description=(
            "The concept requested by the user, or close semantic equivalent, "
            "that the evidence_text is claimed to support. "
            "Examples include 'redemption', 'crime investigation', "
            "'parent-child relationship', or 'astronomy'. "
            "Use null when has_relevant_evidence=False."
        )
    )

    match_level: MatchLevel = Field(
        description=(
            "Overall semantic relevance classification. "
            "'none' means no meaningful evidence of the requested concepts; "
            "'incidental' means genuine but peripheral overlap; "
            "'partial' means one central part is meaningfully satisfied while "
            "another central part is missing; "
            "'clear' means the overall request is clearly satisfied but one "
            "important aspect is weaker, implicit, secondary, or missing; "
            "'strong' means the book directly and strongly satisfies the "
            "central semantic intent."
        )
    )

    score: Literal[0, 1, 2, 3, 4] = Field(
        description=(
            "Numeric score corresponding exactly to match_level: "
            "none=0, incidental=1, partial=2, clear=3, strong=4."
        )
    )

    reason: str = Field(
        min_length=1,
        description=(
            "A concise evidence-based explanation for the verdict. "
            "Explain why the supplied description supports the selected "
            "match_level. Do not speculate about themes or plot elements "
            "that are not established by the supplied description."
        )
    )

class EvidenceExtraction(BaseModel):
    evidence_text: str | None = Field(
        default=None,
        description=(
            "A short exact contiguous excerpt copied verbatim from the supplied "
            "book description that supports the already-decided semantic match. "
            "Do not paraphrase. Return null if no such excerpt exists."
        ),
    )

# =============================================================================
# File/config helpers
# =============================================================================

class JudgeVerdictValidationError(ValueError):
    """
    Base exception for an LLM-generated verdict that violates one of our
    deterministic validation rules.

    These errors are considered repairable because the model can be asked to
    regenerate a corrected structured verdict.
    """


class EvidenceGroundingError(JudgeVerdictValidationError):
    """
    Raised when evidence_text is not an exact excerpt from the supplied
    description.
    """

def load_json(path: Path) -> dict:
    """
    Load a UTF-8 JSON file.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_inputs(
    dataset: pd.DataFrame,
    rubric: dict,
    config: dict,
) -> None:
    """
    Validate compatibility between:

    - calibration dataset,
    - rubric,
    - judge configuration.

    These checks prevent accidental provenance problems such as running
    dataset v0.1.0 with rubric v0.2.0 while still believing the run used
    rubric v0.1.0.
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

    missing_columns = (
        required_columns
        - set(dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: "
            f"{sorted(missing_columns)}"
        )

    # case_id is our unique join key when judge results are later compared
    # against human labels.
    if dataset["case_id"].duplicated().any():

        duplicates = dataset.loc[
            dataset["case_id"].duplicated(
                keep=False
            ),
            "case_id",
        ].tolist()

        raise ValueError(
            f"Duplicate case_id values found: "
            f"{duplicates}"
        )

    # Calibration should only begin after human labelling is complete.
    if not dataset["human_score"].notna().all():
        raise ValueError(
            "Calibration dataset contains "
            "missing human_score values."
        )

    if not dataset["human_score"].isin(
        [0, 1, 2, 3, 4]
    ).all():
        raise ValueError(
            "human_score must contain only "
            "0, 1, 2, 3, or 4."
        )

    # Verify dataset provenance.
    if set(
        dataset[
            "dataset_version"
        ].astype(str)
    ) != {DATASET_VERSION}:

        raise ValueError(
            "Dataset rows do not all declare "
            f"dataset version {DATASET_VERSION}."
        )

    # Verify the rubric version used to create the gold labels.
    if set(
        dataset[
            "rubric_version"
        ].astype(str)
    ) != {RUBRIC_VERSION}:

        raise ValueError(
            "Dataset rows do not all declare "
            f"rubric version {RUBRIC_VERSION}."
        )

    if rubric.get(
        "version"
    ) != RUBRIC_VERSION:

        raise ValueError(
            "Rubric file declares version "
            f"{rubric.get('version')!r}; "
            f"expected {RUBRIC_VERSION!r}."
        )

    if config.get(
        "version"
    ) != JUDGE_CONFIG_VERSION:

        raise ValueError(
            "Judge config declares version "
            f"{config.get('version')!r}; "
            f"expected {JUDGE_CONFIG_VERSION!r}."
        )

    if config.get(
        "rubric_version"
    ) != RUBRIC_VERSION:

        raise ValueError(
            "Judge config rubric_version does "
            "not match RUBRIC_VERSION."
        )

    if config.get(
        "dataset_version"
    ) != DATASET_VERSION:

        raise ValueError(
            "Judge config dataset_version does "
            "not match DATASET_VERSION."
        )

    if config.get(
        "provider"
    ) != "ollama":

        raise ValueError(
            "Judge provider must be 'ollama' "
            "for this runner."
        )

    expected_score_mapping = {
        "none": 0,
        "incidental": 1,
        "partial": 2,
        "clear": 3,
        "strong": 4,
    }

    if config.get(
        "score_mapping"
    ) != expected_score_mapping:

        raise ValueError(
            "Judge config score_mapping is missing "
            "or does not match the expected mapping:\n"
            f"{expected_score_mapping}"
        )


# =============================================================================
# Rubric formatting
# =============================================================================

def format_rubric(
    rubric: dict,
) -> str:
    """
    Convert the versioned rubric JSON into readable prompt text.

    The JSON rubric remains the source of truth. The scoring rules are not
    duplicated in this Python file.
    """

    lines: list[str] = [
        f"RUBRIC: {rubric['name']}",
        "",
        "PURPOSE",
        rubric["purpose"],
        "",
        "EVIDENCE RULES",
    ]

    for rule in rubric.get(
        "evidence_rules",
        [],
    ):
        lines.append(
            f"- {rule}"
        )

    lines.extend(
        [
            "",
            "SCORING DEFINITIONS",
        ]
    )

    # Present strongest match first.
    for score in [
        4,
        3,
        2,
        1,
        0,
    ]:

        item = rubric[
            "scores"
        ][str(score)]

        lines.append(
            f"{score} — "
            f"{item['label']}: "
            f"{item['definition']}"
        )

        # Anchor examples help both human and LLM judges interpret each
        # rubric level consistently.
        for example in item.get(
            "anchor_examples",
            [],
        ):

            lines.append(
                f"  Anchor query: "
                f"{example['query']}"
            )

            lines.append(
                f"  Anchor description: "
                f"{example['book_description']}"
            )

            lines.append(
                f"  Why this score: "
                f"{example['why']}"
            )

        lines.append("")

    lines.append(
        "BOUNDARY GUIDANCE"
    )

    # Boundary examples are especially useful for adjacent scores such as
    # 2 vs 3, where most real disagreement is expected.
    for boundary in rubric.get(
        "boundary_examples",
        [],
    ):

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

    return "\n".join(
        lines
    )


# =============================================================================
# Prompt construction
# =============================================================================

def build_prompt(
    row: pd.Series,
    rubric_text: str,
    config: dict,
) -> str:
    """
    Build the blind LLM prompt for one query-book pair.

    IMPORTANT
    ---------
    Do NOT replace this explicit field allow-list with:

        row.to_dict()

    because the dataset row also contains:

        human_score
        human_reason

    Those fields are gold labels and must never enter the judge prompt.
    """

    case = {
        "query": str(
            row["query"]
        ),
        "title": str(
            row["title"]
        ),
        "authors": str(
            row["authors"]
        ),
        "description": str(
            row["description"]
        ),
    }

    judge_instructions = "\n".join(
        f"- {instruction}"
        for instruction
        in config["instructions"]
    )

    return f"""
You are an independent evaluator of a semantic book recommender.

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

IMPORTANT EVIDENCE REQUIREMENT

If has_relevant_evidence=True:

- evidence_text must be a short, exact phrase copied from the supplied
  Book description.
- Do not paraphrase evidence_text.
- matched_concept must identify the user-requested concept or close
  semantic equivalent that the evidence supports.

If there is no actual textual evidence:

- has_relevant_evidence must be False
- evidence_text must be null
- matched_concept must be null
- match_level must be "none"
- score must be 0

Return the structured verdict requested by the response schema.
""".strip()

def build_verdict_repair_prompt(
    row: pd.Series,
    rubric_text: str,
    config: dict,
    previous_verdict: JudgeVerdict,
    validation_error: str,
) -> str:
    """
    Ask the judge to repair a structured verdict that failed deterministic
    validation.

    The repair request contains the model's previous verdict and the specific
    validation failure, but never exposes human_score or human_reason.
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

    return f"""
You are repairing a structured semantic-relevance verdict that failed
deterministic validation.

Do NOT use external knowledge. Use only the supplied query, book information,
rubric, and description.

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

PREVIOUS VERDICT

has_relevant_evidence:
{previous_verdict.has_relevant_evidence}

evidence_text:
{previous_verdict.evidence_text}

matched_concept:
{previous_verdict.matched_concept}

match_level:
{previous_verdict.match_level}

score:
{previous_verdict.score}

reason:
{previous_verdict.reason}

VALIDATION FAILURE

{validation_error}

REPAIR REQUIREMENTS

Return a complete corrected verdict.

The corrected verdict must satisfy ALL of these rules:

1. none       -> score 0
2. incidental -> score 1
3. partial    -> score 2
4. clear      -> score 3
5. strong     -> score 4

If has_relevant_evidence=False:

- evidence_text must be null
- matched_concept must be null
- match_level must be "none"
- score must be 0

If has_relevant_evidence=True:

- evidence_text must be a short EXACT contiguous phrase copied directly
  from the supplied Book description
- do not paraphrase evidence_text
- matched_concept must be non-null and identify the user-requested concept
  or close semantic equivalent supported by evidence_text
- match_level must not be "none"

If the previous semantic judgment cannot be supported by exact textual
evidence, change the verdict appropriately rather than inventing evidence.

Return the full structured verdict again.
""".strip()

def build_evidence_extraction_prompt(
    row: pd.Series,
    verdict: JudgeVerdict,
) -> str:
    """
    Ask only for an exact supporting quote.

    The semantic classification is already frozen. This call is not allowed
    to change matched_concept, match_level, score, or reason.
    """

    return f"""
Extract textual evidence for an already-completed semantic relevance judgment.

User query:
{row["query"]}

Book description:
{row["description"]}

Matched concept:
{verdict.matched_concept}

Semantic match level:
{verdict.match_level}

Your ONLY task is to extract evidence_text.

Rules:

1. evidence_text must be copied VERBATIM from the Book description.
2. It must be one contiguous substring.
3. Do not paraphrase or summarize.
4. Prefer the shortest phrase or sentence that supports the matched concept.
5. Do not change or reconsider the semantic judgment.
6. If no exact supporting text exists, return null.

Return only the structured EvidenceExtraction result.
""".strip()

# =============================================================================
# Text normalization used for evidence verification
# =============================================================================

def normalize_text(
    value: str,
) -> str:
    """
    Normalize text before checking whether evidence_text occurs in the
    supplied description.

    We intentionally perform only mechanical normalization:

    - Unicode normalization
    - lowercase
    - normalize curly quotes/dashes
    - collapse repeated whitespace

    We do NOT perform semantic matching here. Python should only verify
    that the LLM actually copied evidence from the description.
    """

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    # Normalize common punctuation variants that may differ between
    # model output and CSV text.
    value = (
        value
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )

    value = value.lower()

    # Convert any repeated whitespace/newlines/tabs to one space.
    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# =============================================================================
# Judge verdict validation
# =============================================================================

def validate_verdict(
    verdict: JudgeVerdict,
    description: str,
    config: dict,
) -> int:
    """
    Validate that the LLM's structured verdict is internally consistent.

    Returns
    -------
    int
        The deterministic numeric score derived from match_level.

    Why does Python derive the score again?
    ---------------------------------------
    The LLM should perform the semantic classification.

    Python should perform mechanical rules.

    This lets us detect contradictions such as:

        match_level = "none"
        score = 1

    instead of silently accepting them.
    """

    score_mapping = config[
        "score_mapping"
    ]

    derived_score = score_mapping[
        verdict.match_level
    ]

    # -------------------------------------------------------------------------
    # Rule 1:
    # match_level -> numeric score must always be exact.
    # -------------------------------------------------------------------------
    if verdict.score != derived_score:

        raise JudgeVerdictValidationError(
            "Judge returned inconsistent "
            "match_level/score:\n"
            f"match_level={verdict.match_level!r}\n"
            f"score={verdict.score}\n"
            f"expected_score={derived_score}"
        )

    # -------------------------------------------------------------------------
    # Rule 2:
    # No relevant evidence means none/0.
    # -------------------------------------------------------------------------
    if not verdict.has_relevant_evidence:

        if verdict.match_level != "none":

            raise JudgeVerdictValidationError(
                "Judge returned "
                "has_relevant_evidence=False "
                "but match_level is not 'none':\n"
                f"{verdict.match_level!r}"
            )

        if verdict.score != 0:

            raise JudgeVerdictValidationError(
                "Judge returned "
                "has_relevant_evidence=False "
                f"but score={verdict.score}."
            )

        if verdict.evidence_text is not None:

            raise JudgeVerdictValidationError(
                "evidence_text must be null when "
                "has_relevant_evidence=False."
            )

        if verdict.matched_concept is not None:

            raise JudgeVerdictValidationError(
                "matched_concept must be null when "
                "has_relevant_evidence=False."
            )

        return derived_score

    # -------------------------------------------------------------------------
    # Rule 3:
    # Positive evidence cannot map to match_level=none.
    # -------------------------------------------------------------------------
    if verdict.match_level == "none":

        raise JudgeVerdictValidationError(
            "Judge returned "
            "has_relevant_evidence=True "
            "but match_level='none'."
        )

    # -------------------------------------------------------------------------
    # Rule 4:
    # Positive evidence requires an evidence phrase.
    # -------------------------------------------------------------------------
    if (
        verdict.evidence_text is None
        or not verdict.evidence_text.strip()
    ):

        raise JudgeVerdictValidationError(
            "Positive evidence verdict requires "
            "a non-empty evidence_text."
        )

    # -------------------------------------------------------------------------
    # Rule 5:
    # Positive evidence requires identification of the matched concept.
    # -------------------------------------------------------------------------
    if (
        verdict.matched_concept is None
        or not verdict.matched_concept.strip()
    ):

        raise JudgeVerdictValidationError(
            "Positive evidence verdict requires "
            "a non-empty matched_concept."
        )

    return derived_score


# =============================================================================
# Git/run metadata helpers
# =============================================================================

def git_commit_sha() -> str | None:
    """
    Return the current Git commit SHA when available.

    The SHA lets us associate an evaluation run with the exact code state.
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
        # Missing Git metadata should not prevent a local calibration run.
        return None


def create_run_dir(
    resume: str | None,
) -> Path:
    """
    Create a new timestamped run directory or resolve an existing run
    directory when --resume is supplied.
    """

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if resume:

        run_dir = Path(
            resume
        )

        # Support both absolute and repository-relative paths.
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

    # Use UTC so run IDs are unambiguous across machines/time zones.
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

    This metadata is rewritten while the run proceeds so even a partial
    or interrupted evaluation has reproducibility information.
    """

    metadata = {
        "run_id": run_dir.name,
        "status": status,

        "updated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        # -------------------------------------------------------------
        # Code provenance
        # -------------------------------------------------------------
        "git_commit_sha": (
            git_commit_sha()
        ),

        # -------------------------------------------------------------
        # Versioned evaluation artifacts
        # -------------------------------------------------------------
        "rubric_version": (
            RUBRIC_VERSION
        ),

        "calibration_dataset_version": (
            DATASET_VERSION
        ),

        "judge_config_version": (
            JUDGE_CONFIG_VERSION
        ),

        # -------------------------------------------------------------
        # Judge runtime configuration
        # -------------------------------------------------------------
        "judge_provider": (
            config["provider"]
        ),

        "judge_model": (
            config["model"]
        ),

        "judge_base_url": (
            config["base_url"]
        ),

        "temperature": (
            config["temperature"]
        ),

        # -------------------------------------------------------------
        # Runtime/library versions
        # -------------------------------------------------------------
        "python_version": (
            platform.python_version()
        ),

        "deepeval_version": (
            package_version("deepeval")
        ),

        # -------------------------------------------------------------
        # Run progress
        # -------------------------------------------------------------
        "completed_cases": (
            completed_cases
        ),

        "total_cases": (
            total_cases
        ),
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
# DeepEval / Ollama output normalization
# =============================================================================

MAX_VERDICT_REPAIR_ATTEMPTS = 2


def generate_validated_verdict(
    judge_model: OllamaModel,
    row: pd.Series,
    rubric_text: str,
    config: dict,
) -> tuple[JudgeVerdict, int]:
    """
    Generate and validate one semantic-relevance verdict.

    Any deterministic judge-output validation failure gets a limited number
    of repair attempts.

    Examples of repairable failures:
    - paraphrased evidence_text
    - missing matched_concept
    - missing evidence_text
    - match_level / score mismatch
    - evidence boolean / match_level inconsistency

    Configuration/programming errors remain normal exceptions and are not
    automatically retried.
    """

    description = str(
        row["description"]
    )

    # ---------------------------------------------------------------------
    # Initial judgment
    # ---------------------------------------------------------------------
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

    try:
        score = validate_verdict(
            verdict=verdict,
            description=description,
            config=config,
        )

        return verdict, score

    except JudgeVerdictValidationError as error:
        validation_error = str(error)

    # ---------------------------------------------------------------------
    # Limited repair loop
    # ---------------------------------------------------------------------
    for repair_attempt in range(
        1,
        MAX_VERDICT_REPAIR_ATTEMPTS + 1,
    ):

        print(
            f"  Invalid verdict for {row['case_id']}. "
            f"Repair attempt "
            f"{repair_attempt}/{MAX_VERDICT_REPAIR_ATTEMPTS}..."
        )

        print(
            f"  Validation error: {validation_error}"
        )

        repair_prompt = build_verdict_repair_prompt(
            row=row,
            rubric_text=rubric_text,
            config=config,
            previous_verdict=verdict,
            validation_error=validation_error,
        )

        repaired_generation = judge_model.generate(
            prompt=repair_prompt,
            schema=JudgeVerdict,
        )

        verdict = unpack_generation(
            repaired_generation
        )

        try:
            score = validate_verdict(
                verdict=verdict,
                description=description,
                config=config,
            )

            return verdict, score

        except JudgeVerdictValidationError as error:

            validation_error = str(
                error
            )

            # If this was the final allowed repair, surface the real
            # validation failure rather than silently accepting bad output.
            if (
                repair_attempt
                == MAX_VERDICT_REPAIR_ATTEMPTS
            ):
                raise

MAX_EVIDENCE_EXTRACTION_ATTEMPTS = 2


def extract_grounded_evidence(
    judge_model: OllamaModel,
    row: pd.Series,
    verdict: JudgeVerdict,
) -> str | None:
    """
    Obtain a verbatim evidence span without allowing the model to change
    the semantic verdict.
    """

    description = str(row["description"])

    # First, accept the original evidence if it was already quoted correctly.
    if verdict.evidence_text:

        if (
            normalize_text(verdict.evidence_text)
            in normalize_text(description)
        ):
            return verdict.evidence_text

    prompt = build_evidence_extraction_prompt(
        row=row,
        verdict=verdict,
    )

    for attempt in range(
        1,
        MAX_EVIDENCE_EXTRACTION_ATTEMPTS + 1,
    ):

        print(
            f"  Extracting grounded evidence for {row['case_id']} "
            f"({attempt}/{MAX_EVIDENCE_EXTRACTION_ATTEMPTS})..."
        )

        generated = judge_model.generate(
            prompt=prompt,
            schema=EvidenceExtraction,
        )

        if isinstance(generated, tuple):
            extraction = generated[0]
        else:
            extraction = generated

        if isinstance(extraction, dict):
            extraction = EvidenceExtraction(
                **extraction
            )

        evidence = extraction.evidence_text

        if evidence is None:
            return None

        if (
            normalize_text(evidence)
            in normalize_text(description)
        ):
            return evidence

        print(
            "  Extracted text was still not verbatim."
        )

    return None            

def unpack_generation(
    generated,
) -> JudgeVerdict:
    """
    Normalize DeepEval structured model output into JudgeVerdict.

    Depending on DeepEval/model integration details, structured generation
    may return:

    - the Pydantic object directly,
    - a dictionary,
    - or a tuple whose first element contains the result.
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
            "Unexpected judge response type: "
            f"{type(verdict)!r}"
        )

    return verdict


# =============================================================================
# Main calibration loop
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Blindly score semantic relevance calibration cases "
            "with Judge Config v0.5.0 using a local Ollama model."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Score only the first N remaining cases. "
            "Useful for development smoke tests."
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

    # =========================================================================
    # 1. Verify required versioned artifacts exist.
    # =========================================================================

    for required_file in [
        DATASET_FILE,
        RUBRIC_FILE,
        JUDGE_CONFIG_FILE,
    ]:

        if not required_file.exists():

            raise FileNotFoundError(
                "Required file not found: "
                f"{required_file}"
            )

    # =========================================================================
    # 2. Load calibration dataset, rubric, and judge config.
    # =========================================================================

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

    # =========================================================================
    # 3. Create a new run or resume an existing one.
    # =========================================================================

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

    # Select only cases that have not already been scored.
    remaining_cases = dataset[
        ~dataset[
            "case_id"
        ]
        .astype(str)
        .isin(
            completed_case_ids
        )
    ].copy()

    # --limit applies to currently unscored cases only.
    if args.limit is not None:

        remaining_cases = (
            remaining_cases
            .head(
                args.limit
            )
        )

    # =========================================================================
    # 4. Initialize the local Ollama judge.
    # =========================================================================

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

    # =========================================================================
    # 5. Score each remaining query-book pair.
    # =========================================================================

    for _, row in remaining_cases.iterrows():

        case_id = str(
            row["case_id"]
        )

        # ---------------------------------------------------------------------
        # Ask the local model for structured output.
        # Validate all deterministic relationships in Python.
        #
        # The LLM decides semantic relevance.
        # Python verifies mechanical consistency and evidence grounding.
        # ---------------------------------------------------------------------

        verdict, judge_score = generate_validated_verdict(
            judge_model=judge_model,
            row=row,
            rubric_text=rubric_text,
            config=config,
        )

        if verdict.has_relevant_evidence:

            grounded_evidence = extract_grounded_evidence(
            judge_model=judge_model,
            row=row,
            verdict=verdict,
            )

            if grounded_evidence is None:
                raise EvidenceGroundingError(
                 f"Unable to obtain grounded evidence for "
                 f"{row['case_id']} after "
                 f"{MAX_EVIDENCE_EXTRACTION_ATTEMPTS} attempts."
                )

        else:
            grounded_evidence = None

        # ---------------------------------------------------------------------
        # Store only judge-produced evaluation information.
        #
        # Human labels remain absent from judge_results.csv.
        # ---------------------------------------------------------------------

        results.append(
            {
                "case_id": case_id,

                "has_relevant_evidence": (
                    verdict.has_relevant_evidence
                ),

                "evidence_text": (
                    verdict.evidence_text
                ),

                "matched_concept": (
                    verdict.matched_concept
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

        # ---------------------------------------------------------------------
        # Persist after every case.
        #
        # Local LLM calibration runs can take time. If case 47 fails, we do
        # not want to lose cases 1-46.
        # ---------------------------------------------------------------------

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
            f"{case_id}: "
            f"evidence="
            f"{verdict.has_relevant_evidence}, "
            f"level="
            f"{verdict.match_level}, "
            f"score="
            f"{judge_score}, "
            f"matched_concept="
            f"{verdict.matched_concept!r}"
        )

    # =========================================================================
    # 6. Finalize run status.
    # =========================================================================

    unique_completed_cases = len(
        {
            str(
                result["case_id"]
            )
            for result
            in results
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