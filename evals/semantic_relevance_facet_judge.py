"""
Deterministic-cue + isolated semantic facet judge for v0.15.0.

v0.15.0 preserves the v0.13 execution architecture. The v0.14 behavioral
changes live in the frozen Q03 facet definitions and deterministic aggregation;
this module keeps the v0.13 cue/LLM stage boundaries.

Execution stages:

    0. deterministic high-precision DIRECT cue scan
    A. priority-ranked top-3 evidence selection when no cue fires
    B. isolated evidence verification
    C. deterministic best-candidate resolution
    D. core-only prominence assessment
    E. deterministic support derivation and 0-4 aggregation

A deterministic cue is positive-only. It may establish relation=DIRECT for an
exact description span, but it never decides core prominence or the final score.
If no cue matches, the v0.13 LLM pipeline runs unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from semantic_relevance_facet_scoring import (
    CandidateVerificationRecord,
    FacetJudgeVerdict,
    FacetPipelineAssessment,
    FacetProminence,
    FacetSupport,
    QueryFacet,
    QueryFacetSpec,
    VerificationRelation,
    derive_facet_support,
    validate_facet_assessments,
)


NO_EVIDENCE_SPAN = "NONE"
MAX_EVIDENCE_CANDIDATES = 3
MAX_STAGE_ATTEMPTS = 2


DETERMINISTIC_CUE_REASON_PREFIX = "deterministic_direct_cue:"


class DeterministicDirectCueMatch(BaseModel):
    """One high-confidence positive DIRECT cue grounded in an exact source span."""

    cue_id: str = Field(min_length=1)
    evidence_span_id: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1)
    matched_expression: str = Field(min_length=1)


def _regex_matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def find_deterministic_direct_cue(
    spec_query_id: str,
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
) -> DeterministicDirectCueMatch | None:
    """
    Return the first configured high-precision DIRECT cue match.

    Rule order is configuration order; within a rule, the earliest exact source
    span wins. No match has no negative semantic meaning and simply falls through
    to Stage A.
    """

    stage = config.get("deterministic_direct_cue_stage", {})
    rules = stage.get("rules", [])

    for rule in rules:
        if rule.get("query_id") != spec_query_id:
            continue
        if rule.get("facet_id") != facet.facet_id:
            continue

        any_patterns = list(rule.get("any_regex", []))
        all_groups = list(rule.get("all_regex_groups", []))

        for span_id, text in spans.items():
            for pattern in any_patterns:
                if _regex_matches(pattern, text):
                    return DeterministicDirectCueMatch(
                        cue_id=str(rule["cue_id"]),
                        evidence_span_id=span_id,
                        evidence_text=text,
                        matched_expression=pattern,
                    )

            for group in all_groups:
                if group and all(_regex_matches(pattern, text) for pattern in group):
                    return DeterministicDirectCueMatch(
                        cue_id=str(rule["cue_id"]),
                        evidence_span_id=span_id,
                        evidence_text=text,
                        matched_expression=" ALL ".join(group),
                    )

    return None


class JudgeOutputValidationError(ValueError):
    """Mechanically detectable structured-output/source-grounding failure."""


class EvidenceSelection(BaseModel):
    """
    Ranked source-span candidates from Stage A.

    The model should normally return 1-3 real S# IDs.
    ["NONE"] is allowed only when there is no plausible candidate.
    """

    evidence_span_ids: list[str] = Field(
        min_length=1,
        max_length=MAX_EVIDENCE_CANDIDATES,
    )

    reason: str = Field(min_length=1)


class EvidenceVerification(BaseModel):
    """Structured output for one isolated verification."""

    verification_relation: VerificationRelation
    reason: str = Field(min_length=1)


class ProminenceAssessment(BaseModel):
    """Structured output for one core-prominence assessment."""

    prominence: FacetProminence
    reason: str = Field(min_length=1)


class SemanticGenerationResult(BaseModel):
    """Complete per-case semantic result consumed by the runner."""

    verdict: FacetJudgeVerdict

    # Winning/representative exact evidence used by legacy compatibility fields.
    evidence_by_id: dict[str, str]

    # Every Stage-A candidate, retained for audit in facet_evidence_json.
    candidate_evidence_by_id: dict[str, dict[str, str]]

    generation_mode: str = "deterministic_cue_plus_isolated_precedence_pipeline"

    stage_retry_count: int = 0

    deterministic_direct_cue_count: int = 0


# =============================================================================
# Exact source span construction
# =============================================================================

def build_description_spans(
    description: str,
) -> dict[str, str]:
    """Split the exact description into conservative sentence-like spans."""

    text = str(description).strip()

    if not text:
        return {}

    raw_spans = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    spans = [
        span.strip()
        for span in raw_spans
        if span.strip()
    ]

    if not spans:
        spans = [text]

    return {
        f"S{index}": span
        for index, span
        in enumerate(spans, start=1)
    }


def format_description_spans(
    spans: dict[str, str],
) -> str:
    """Render exact source spans for Stage A / prominence prompts."""

    return "\n".join(
        f"{span_id}: {span_text}"
        for span_id, span_text
        in spans.items()
    )


# =============================================================================
# DeepEval/Ollama structured-output normalization
# =============================================================================

def unpack_generated_model(
    generated: Any,
    schema_type: type[BaseModel],
) -> BaseModel:
    """Normalize common DeepEval structured-output return shapes."""

    value = (
        generated[0]
        if isinstance(generated, tuple)
        else generated
    )

    if isinstance(value, schema_type):
        return value

    if isinstance(value, dict):
        return schema_type(**value)

    raise JudgeOutputValidationError(
        f"Unexpected model output type: {type(value)!r}"
    )


# =============================================================================
# Stage A: multi-candidate evidence retrieval
# =============================================================================

def build_evidence_selection_prompt(
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
) -> str:
    """
    Build the recall-oriented Stage-A prompt.

    Original query and other facets remain hidden.
    """

    stage = config["evidence_selection_stage"]

    priorities = "\n".join(
        (
            f"{item['priority']}. {item['name']}: "
            f"{item['rule']}"
        )
        for item in stage.get("candidate_priority", [])
    )

    instructions = "\n".join(
        f"{index}. {instruction}"
        for index, instruction
        in enumerate(stage["instructions"], start=1)
    )

    max_candidates = int(
        stage.get(
            "max_candidates",
            MAX_EVIDENCE_CANDIDATES,
        )
    )

    return f"""
Find up to {max_candidates} candidate evidence spans for ONE semantic facet.

FACET
{facet.text}

FROZEN SEMANTIC DEFINITION
{facet.semantic_definition}

NUMBERED BOOK-DESCRIPTION SPANS
{format_description_spans(spans)}

CANDIDATE RANKING PRIORITY
{priorities}

INSTRUCTIONS
{instructions}

OUTPUT
Return one EvidenceSelection with:

evidence_span_ids:
- 1 to {max_candidates} UNIQUE supplied S# IDs, ranked best-first; OR
- exactly ["NONE"] if no supplied span is even a plausible candidate

reason:
- briefly explain why these were chosen

Do not decide final support, verification relation, prominence, or score.
""".strip()


def normalize_candidate_span_ids(
    selection: EvidenceSelection,
    spans: dict[str, str],
) -> list[str]:
    """
    Validate and normalize Stage-A output.

    Returns an empty list for the explicit ["NONE"] sentinel.
    """

    selected_ids = [
        str(span_id).strip()
        for span_id
        in selection.evidence_span_ids
    ]

    if selected_ids == [NO_EVIDENCE_SPAN]:
        return []

    if NO_EVIDENCE_SPAN in selected_ids:
        raise JudgeOutputValidationError(
            "NONE cannot be mixed with real source span IDs."
        )

    if len(selected_ids) != len(set(selected_ids)):
        raise JudgeOutputValidationError(
            "Evidence candidate IDs must be unique."
        )

    if len(selected_ids) > MAX_EVIDENCE_CANDIDATES:
        raise JudgeOutputValidationError(
            f"At most {MAX_EVIDENCE_CANDIDATES} evidence candidates are allowed."
        )

    unknown = [
        span_id
        for span_id
        in selected_ids
        if span_id not in spans
    ]

    if unknown:
        raise JudgeOutputValidationError(
            f"Unknown evidence span IDs: {unknown}. "
            f"Valid IDs: {list(spans)}."
        )

    return selected_ids


def select_candidate_evidence(
    judge_model: Any,
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
) -> tuple[EvidenceSelection, list[str], int]:
    """
    Select up to three valid candidate source spans.
    """

    validation_error: str | None = None

    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):

        prompt = build_evidence_selection_prompt(
            facet=facet,
            spans=spans,
            config=config,
        )

        if validation_error:
            prompt += (
                "\n\nPREVIOUS VALIDATION FAILURE\n"
                f"{validation_error}\n"
                "Return a corrected EvidenceSelection only."
            )

        generated = judge_model.generate(
            prompt=prompt,
            schema=EvidenceSelection,
        )

        selection = unpack_generated_model(
            generated,
            EvidenceSelection,
        )

        assert isinstance(selection, EvidenceSelection)

        try:
            candidate_ids = normalize_candidate_span_ids(
                selection=selection,
                spans=spans,
            )

            return selection, candidate_ids, attempt - 1

        except JudgeOutputValidationError as error:
            validation_error = str(error)

    raise JudgeOutputValidationError(
        validation_error
        or "Unable to obtain valid multi-candidate evidence selection."
    )


# =============================================================================
# Stage B: isolated evidence verification
# v0.12.0 preserves the isolated stage boundary and adds explicit decision precedence.
# =============================================================================

def build_verification_prompt(
    facet: QueryFacet,
    evidence_text: str,
    config: dict[str, Any],
) -> str:
    """
    Build the isolated verifier prompt.

    Only the target facet and exact candidate evidence are visible.
    """

    stage = config["verification_stage"]

    scale = "\n".join(
        f"- {name}: {definition}"
        for name, definition
        in stage["relation_scale"].items()
    )

    precedence = "\n".join(
        (
            f"{item['step']}. {item['name']}: "
            f"{item['rule']}"
        )
        for item in stage.get("decision_precedence", [])
    )

    instructions = "\n".join(
        f"{index}. {instruction}"
        for index, instruction
        in enumerate(stage["instructions"], start=1)
    )

    return f"""
Decide whether ONE exact evidence span establishes ONE semantic facet.

FACET
{facet.text}

FROZEN SEMANTIC DEFINITION
{facet.semantic_definition}

EXACT CANDIDATE EVIDENCE
{evidence_text}

VERIFICATION SCALE
{scale}

DECISION PRECEDENCE
{precedence}

INSTRUCTIONS
{instructions}

Return one EvidenceVerification.

Do not use any information other than the facet, frozen semantic definition, and evidence shown above.
""".strip()


def verify_candidate_evidence(
    judge_model: Any,
    facet: QueryFacet,
    evidence_text: str,
    config: dict[str, Any],
) -> tuple[EvidenceVerification, int]:
    """Verify one candidate independently from all other context."""

    last_error: Exception | None = None

    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):

        try:
            generated = judge_model.generate(
                prompt=build_verification_prompt(
                    facet=facet,
                    evidence_text=evidence_text,
                    config=config,
                ),
                schema=EvidenceVerification,
            )

            result = unpack_generated_model(
                generated,
                EvidenceVerification,
            )

            assert isinstance(result, EvidenceVerification)

            return result, attempt - 1

        except Exception as error:
            last_error = error

    raise JudgeOutputValidationError(
        "Unable to obtain valid isolated evidence verification."
    ) from last_error


# =============================================================================
# Deterministic candidate resolution
# =============================================================================

VERIFICATION_STRENGTH = {
    VerificationRelation.UNSUPPORTED: 0,
    VerificationRelation.ADJACENT: 1,
    VerificationRelation.ENTAILED: 2,
    VerificationRelation.DIRECT: 3,
}


def choose_best_verified_candidate(
    records: list[CandidateVerificationRecord],
) -> CandidateVerificationRecord:
    """
    Choose the strongest verified candidate deterministically.

    Priority:
        direct > entailed > adjacent > unsupported

    Tie-break:
        lower selector rank wins.

    This function is deliberately pure and unit-testable.
    """

    if not records:
        raise ValueError(
            "At least one candidate verification record is required."
        )

    return max(
        records,
        key=lambda record: (
            VERIFICATION_STRENGTH[
                record.verification_relation
            ],
            -record.candidate_rank,
        ),
    )


def verify_ranked_candidates(
    judge_model: Any,
    facet: QueryFacet,
    candidate_ids: list[str],
    spans: dict[str, str],
    config: dict[str, Any],
) -> tuple[
    CandidateVerificationRecord,
    list[CandidateVerificationRecord],
    int,
]:
    """
    Verify candidate spans in selector order and select the strongest result.

    DIRECT short-circuits later verification because no later relation can
    outrank it. ENTAILED and ADJACENT do not short-circuit because a later
    stronger candidate should win.
    """

    records: list[CandidateVerificationRecord] = []
    total_retries = 0

    for rank, span_id in enumerate(
        candidate_ids,
        start=1,
    ):

        verification, retries = verify_candidate_evidence(
            judge_model=judge_model,
            facet=facet,
            evidence_text=spans[span_id],
            config=config,
        )

        total_retries += retries

        record = CandidateVerificationRecord(
            candidate_rank=rank,
            evidence_span_id=span_id,
            verification_relation=(
                verification.verification_relation
            ),
            verification_reason=verification.reason,
        )

        records.append(record)

        if (
            verification.verification_relation
            == VerificationRelation.DIRECT
        ):
            break

    return (
        choose_best_verified_candidate(records),
        records,
        total_retries,
    )


# =============================================================================
# Stage C: core prominence
# v0.12.0 preserves the isolated stage boundary and adds explicit decision precedence.
# =============================================================================

def build_prominence_prompt(
    facet: QueryFacet,
    evidence_text: str,
    spans: dict[str, str],
    config: dict[str, Any],
) -> str:
    """
    Judge prominence only after a core facet has passed verification.
    """

    stage = config["prominence_stage"]

    scale = "\n".join(
        f"- {name}: {definition}"
        for name, definition
        in stage["prominence_scale"].items()
    )

    instructions = "\n".join(
        f"{index}. {instruction}"
        for index, instruction
        in enumerate(stage["instructions"], start=1)
    )

    return f"""
Assess the prominence of ONE already-verified semantic facet in a book description.

FACET
{facet.text}

FROZEN SEMANTIC DEFINITION
{facet.semantic_definition}

VERIFIED SUPPORTING EVIDENCE
{evidence_text}

FULL NUMBERED BOOK-DESCRIPTION SPANS
{format_description_spans(spans)}

PROMINENCE SCALE
{scale}

INSTRUCTIONS
{instructions}

Return one ProminenceAssessment.
Use only incidental, substantive, or central.
""".strip()


def assess_core_prominence(
    judge_model: Any,
    facet: QueryFacet,
    evidence_text: str,
    spans: dict[str, str],
    config: dict[str, Any],
) -> tuple[ProminenceAssessment, int]:
    """Get one valid core prominence classification."""

    last_error: Exception | None = None

    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):

        try:
            generated = judge_model.generate(
                prompt=build_prominence_prompt(
                    facet=facet,
                    evidence_text=evidence_text,
                    spans=spans,
                    config=config,
                ),
                schema=ProminenceAssessment,
            )

            result = unpack_generated_model(
                generated,
                ProminenceAssessment,
            )

            assert isinstance(result, ProminenceAssessment)

            if result.prominence == FacetProminence.NOT_APPLICABLE:
                raise JudgeOutputValidationError(
                    "Core prominence stage returned not_applicable."
                )

            return result, attempt - 1

        except Exception as error:
            last_error = error

    raise JudgeOutputValidationError(
        "Unable to obtain valid core prominence assessment."
    ) from last_error


# =============================================================================
# Complete per-facet v0.12.0 pipeline
# =============================================================================

def evaluate_one_facet(
    judge_model: Any,
    query_id: str,
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
) -> tuple[
    FacetPipelineAssessment,
    str | None,
    dict[str, str],
    int,
]:
    """
    Run:
        multi-candidate selection
        -> isolated verification(s)
        -> deterministic winner
        -> optional prominence

    Returns:
        final assessment
        winning/representative evidence text
        every selected candidate's exact source text
        total structured-output retry count
    """

    total_retries = 0

    cue_match = find_deterministic_direct_cue(
        spec_query_id=query_id,
        facet=facet,
        spans=spans,
        config=config,
    )

    if cue_match is not None:
        span_id = cue_match.evidence_span_id
        reason = (
            f"{DETERMINISTIC_CUE_REASON_PREFIX}{cue_match.cue_id}; "
            f"matched={cue_match.matched_expression}"
        )
        candidate_texts = {span_id: cue_match.evidence_text}
        records = [
            CandidateVerificationRecord(
                candidate_rank=1,
                evidence_span_id=span_id,
                verification_relation=VerificationRelation.DIRECT,
                verification_reason=reason,
            )
        ]

        if facet.facet_type == "qualifier":
            return (
                FacetPipelineAssessment(
                    facet_id=facet.facet_id,
                    candidate_evidence_span_ids=[span_id],
                    candidate_verifications=records,
                    candidate_evidence_span_id=span_id,
                    verification_relation=VerificationRelation.DIRECT,
                    prominence=FacetProminence.NOT_APPLICABLE,
                    evidence_selection_reason=reason,
                    verification_reason=reason,
                    prominence_reason=None,
                ),
                cue_match.evidence_text,
                candidate_texts,
                total_retries,
            )

        prominence_result, retries = assess_core_prominence(
            judge_model=judge_model,
            facet=facet,
            evidence_text=cue_match.evidence_text,
            spans=spans,
            config=config,
        )
        total_retries += retries

        return (
            FacetPipelineAssessment(
                facet_id=facet.facet_id,
                candidate_evidence_span_ids=[span_id],
                candidate_verifications=records,
                candidate_evidence_span_id=span_id,
                verification_relation=VerificationRelation.DIRECT,
                prominence=prominence_result.prominence,
                evidence_selection_reason=reason,
                verification_reason=reason,
                prominence_reason=prominence_result.reason,
            ),
            cue_match.evidence_text,
            candidate_texts,
            total_retries,
        )

    selection, candidate_ids, retries = select_candidate_evidence(
        judge_model=judge_model,
        facet=facet,
        spans=spans,
        config=config,
    )

    total_retries += retries

    candidate_texts = {
        span_id: spans[span_id]
        for span_id
        in candidate_ids
    }

    # No plausible candidate from Stage A.
    if not candidate_ids:

        return (
            FacetPipelineAssessment(
                facet_id=facet.facet_id,
                candidate_evidence_span_ids=[],
                candidate_verifications=[],
                candidate_evidence_span_id=NO_EVIDENCE_SPAN,
                verification_relation=VerificationRelation.UNSUPPORTED,
                prominence=FacetProminence.NOT_APPLICABLE,
                evidence_selection_reason=selection.reason,
                verification_reason=(
                    "Evidence selector returned NONE; verification skipped."
                ),
                prominence_reason=None,
            ),
            None,
            candidate_texts,
            total_retries,
        )

    best, verification_records, retries = verify_ranked_candidates(
        judge_model=judge_model,
        facet=facet,
        candidate_ids=candidate_ids,
        spans=spans,
        config=config,
    )

    total_retries += retries

    winning_span_id = best.evidence_span_id
    winning_evidence_text = spans[winning_span_id]
    relation = best.verification_relation

    # Candidate(s) existed but none established the facet.
    if relation == VerificationRelation.UNSUPPORTED:

        return (
            FacetPipelineAssessment(
                facet_id=facet.facet_id,
                candidate_evidence_span_ids=candidate_ids,
                candidate_verifications=verification_records,
                candidate_evidence_span_id=winning_span_id,
                verification_relation=relation,
                prominence=FacetProminence.NOT_APPLICABLE,
                evidence_selection_reason=selection.reason,
                verification_reason=best.verification_reason,
                prominence_reason=None,
            ),
            winning_evidence_text,
            candidate_texts,
            total_retries,
        )

    # Adjacent core evidence is already the weak/incidental semantic region.
    # Do not ask prominence to turn an incomplete semantic match into a stronger one.
    if (
        relation == VerificationRelation.ADJACENT
        and facet.facet_type == "core"
    ):
        return (
            FacetPipelineAssessment(
                facet_id=facet.facet_id,
                candidate_evidence_span_ids=candidate_ids,
                candidate_verifications=verification_records,
                candidate_evidence_span_id=winning_span_id,
                verification_relation=relation,
                prominence=FacetProminence.NOT_APPLICABLE,
                evidence_selection_reason=selection.reason,
                verification_reason=best.verification_reason,
                prominence_reason=None,
            ),
            winning_evidence_text,
            candidate_texts,
            total_retries,
        )

    # Qualifiers remain relation-only; prominence is intentionally irrelevant.
    if facet.facet_type == "qualifier":

        return (
            FacetPipelineAssessment(
                facet_id=facet.facet_id,
                candidate_evidence_span_ids=candidate_ids,
                candidate_verifications=verification_records,
                candidate_evidence_span_id=winning_span_id,
                verification_relation=relation,
                prominence=FacetProminence.NOT_APPLICABLE,
                evidence_selection_reason=selection.reason,
                verification_reason=best.verification_reason,
                prominence_reason=None,
            ),
            winning_evidence_text,
            candidate_texts,
            total_retries,
        )

    prominence_result, retries = assess_core_prominence(
        judge_model=judge_model,
        facet=facet,
        evidence_text=winning_evidence_text,
        spans=spans,
        config=config,
    )

    total_retries += retries

    return (
        FacetPipelineAssessment(
            facet_id=facet.facet_id,
            candidate_evidence_span_ids=candidate_ids,
            candidate_verifications=verification_records,
            candidate_evidence_span_id=winning_span_id,
            verification_relation=relation,
            prominence=prominence_result.prominence,
            evidence_selection_reason=selection.reason,
            verification_reason=best.verification_reason,
            prominence_reason=prominence_result.reason,
        ),
        winning_evidence_text,
        candidate_texts,
        total_retries,
    )


def generate_validated_semantic_verdict(
    judge_model: Any,
    row: pd.Series,
    spec: QueryFacetSpec,
    rubric: dict[str, Any],
    config: dict[str, Any],
) -> SemanticGenerationResult:
    """
    Run v0.15.0 for every frozen facet.

    The rubric argument remains for runner compatibility but is intentionally
    NOT shown to Stage A or the isolated verifier. The frozen semantic definition
    is supplied instead of the full query context.
    """

    del rubric

    spans = build_description_spans(
        str(row["description"])
    )

    if not spans:
        raise JudgeOutputValidationError(
            f"Description is empty for case {row['case_id']}."
        )

    assessments: list[FacetPipelineAssessment] = []

    evidence_by_id: dict[str, str] = {}

    candidate_evidence_by_id: dict[
        str,
        dict[str, str],
    ] = {}

    total_retries = 0

    for facet in spec.facets:

        (
            assessment,
            winning_evidence_text,
            candidate_texts,
            retries,
        ) = evaluate_one_facet(
            judge_model=judge_model,
            query_id=spec.query_id,
            facet=facet,
            spans=spans,
            config=config,
        )

        assessments.append(assessment)

        total_retries += retries

        candidate_evidence_by_id[
            facet.facet_id
        ] = candidate_texts

        if winning_evidence_text is not None:
            evidence_by_id[
                facet.facet_id
            ] = winning_evidence_text

    temporary_verdict = FacetJudgeVerdict(
        assessments=assessments,
        overall_reason="pending deterministic support summary",
    )

    try:
        validate_facet_assessments(
            spec=spec,
            verdict=temporary_verdict,
        )
    except ValueError as error:
        raise JudgeOutputValidationError(str(error)) from error

    facet_by_id = {
        facet.facet_id: facet
        for facet in spec.facets
    }

    support_fragments = []

    for assessment in assessments:

        support = derive_facet_support(
            facet=facet_by_id[assessment.facet_id],
            assessment=assessment,
        )

        support_fragments.append(
            (
                f"{assessment.facet_id}={support.value}: "
                f"verification={assessment.verification_relation.value}, "
                f"prominence={assessment.prominence.value}, "
                f"candidates={len(assessment.candidate_evidence_span_ids)}"
            )
        )

    verdict = FacetJudgeVerdict(
        assessments=assessments,
        overall_reason=" | ".join(support_fragments),
    )

    deterministic_direct_cue_count = sum(
        assessment.verification_reason.startswith(DETERMINISTIC_CUE_REASON_PREFIX)
        for assessment in assessments
    )

    return SemanticGenerationResult(
        verdict=verdict,
        evidence_by_id=evidence_by_id,
        candidate_evidence_by_id=candidate_evidence_by_id,
        stage_retry_count=total_retries,
        deterministic_direct_cue_count=deterministic_direct_cue_count,
    )


# =============================================================================
# Serialization / backward-compatible summary helpers
# =============================================================================

def serialize_facet_assessments(
    spec: QueryFacetSpec,
    verdict: FacetJudgeVerdict,
) -> str:
    """
    Persist selector candidates, every executed verification, winner, prominence,
    and Python-derived support.
    """

    facet_by_id = {
        facet.facet_id: facet
        for facet in spec.facets
    }

    payload = []

    for assessment in verdict.assessments:

        facet = facet_by_id[assessment.facet_id]

        payload.append(
            {
                "facet_id": assessment.facet_id,
                "facet_text": facet.text,
                "facet_type": facet.facet_type,
                "facet_semantic_definition": facet.semantic_definition,
                "candidate_evidence_span_ids": (
                    assessment.candidate_evidence_span_ids
                ),
                "candidate_verifications": [
                    record.model_dump(mode="json")
                    for record
                    in assessment.candidate_verifications
                ],
                "winning_evidence_span_id": (
                    assessment.candidate_evidence_span_id
                ),
                "verification_relation": (
                    assessment.verification_relation.value
                ),
                "prominence": assessment.prominence.value,
                "derived_support": derive_facet_support(
                    facet=facet,
                    assessment=assessment,
                ).value,
                "evidence_selection_reason": (
                    assessment.evidence_selection_reason
                ),
                "verification_reason": (
                    assessment.verification_reason
                ),
                "prominence_reason": (
                    assessment.prominence_reason
                ),
            }
        )

    return json.dumps(
        payload,
        ensure_ascii=False,
    )


def serialize_facet_evidence(
    candidate_evidence_by_id: dict[str, dict[str, str]],
) -> str:
    """
    Persist exact text for ALL Stage-A candidates, not only the winning span.

    This makes evidence-selection recall failures auditable after the run.
    """

    return json.dumps(
        candidate_evidence_by_id,
        ensure_ascii=False,
    )


def supported_core_facet_texts(
    spec: QueryFacetSpec,
    verdict: FacetJudgeVerdict,
) -> list[str]:
    """Return core facets whose derived support is positive."""

    assessment_by_id = {
        assessment.facet_id: assessment
        for assessment in verdict.assessments
    }

    matched: list[str] = []

    for facet in spec.facets:

        if facet.facet_type != "core":
            continue

        support = derive_facet_support(
            facet=facet,
            assessment=assessment_by_id[facet.facet_id],
        )

        if support != FacetSupport.ABSENT:
            matched.append(facet.text)

    return matched


def supported_evidence_by_id(
    spec: QueryFacetSpec,
    verdict: FacetJudgeVerdict,
    evidence_by_id: dict[str, str],
) -> dict[str, str]:
    """
    Keep winning exact evidence only for genuinely supported facets.

    Unsupported candidate evidence remains available in facet_evidence_json.
    """

    facet_by_id = {
        facet.facet_id: facet
        for facet in spec.facets
    }

    supported: dict[str, str] = {}

    for assessment in verdict.assessments:

        facet = facet_by_id[assessment.facet_id]

        support = derive_facet_support(
            facet=facet,
            assessment=assessment,
        )

        if (
            support != FacetSupport.ABSENT
            and assessment.facet_id in evidence_by_id
        ):
            supported[
                assessment.facet_id
            ] = evidence_by_id[
                assessment.facet_id
            ]

    return supported
