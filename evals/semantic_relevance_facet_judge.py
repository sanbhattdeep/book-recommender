"""
Semantic facet judge v0.22.0 with facet-independent book-subject extraction.

The key v0.22.0 change is architectural: the model analyzes each book description
ONCE without seeing the user query or any facet, producing a frozen book-level
primary-subject/premise summary. Every later core-facet role decision receives
that exact frozen analysis. This prevents the current facet from redefining the
book's primary subject differently from one facet call to another.

Candidate selection, isolated verification, composite recovery, full-description
hard-exclusion prechecks, deterministic role resolution, and deterministic 0-4
scoring remain otherwise unchanged from v0.21.2.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from semantic_relevance_facet_scoring import (
    CandidateVerificationRecord,
    EvidenceInferenceKind,
    FacetContextRole,
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
MAX_COMPOSITE_CANDIDATES = 4
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


def _polarity_guard_blocks_match(
    text: str,
    match: re.Match[str],
    config: dict[str, Any],
) -> bool:
    """Return True only for explicit local negation/absence/contrast."""

    stage = config.get("deterministic_direct_cue_stage", {})
    guard = stage.get("polarity_guard", {})

    if not guard.get("enabled", False):
        return False

    pre_window = int(guard.get("pre_window_chars", 96))
    post_window = int(guard.get("post_window_chars", 64))

    prefix = text[max(0, match.start() - pre_window):match.start()]
    suffix = text[match.end():match.end() + post_window]

    for pattern in guard.get("negative_prefix_regex", []):
        if re.search(pattern, prefix, flags=re.IGNORECASE):
            return True

    for pattern in guard.get("negative_suffix_regex", []):
        if re.search(pattern, suffix, flags=re.IGNORECASE):
            return True

    return False


def _first_positive_regex_match(
    pattern: str,
    text: str,
    config: dict[str, Any],
) -> tuple[re.Match[str] | None, int]:
    """Return first polarity-safe match plus number of blocked occurrences."""

    blocked = 0

    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        if _polarity_guard_blocks_match(text, match, config):
            blocked += 1
            continue
        return match, blocked

    return None, blocked


def _scan_deterministic_direct_cue(
    spec_query_id: str,
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
) -> tuple[DeterministicDirectCueMatch | None, int]:
    """
    Return the first configured polarity-safe DIRECT cue match plus a count of
    suppressed lexical occurrences.

    A blocked cue is not negative evidence; it simply falls through to Stage A.
    """

    stage = config.get("deterministic_direct_cue_stage", {})
    rules = stage.get("rules", [])
    blocked_count = 0

    for rule in rules:
        if rule.get("query_id") != spec_query_id:
            continue
        if rule.get("facet_id") != facet.facet_id:
            continue

        any_patterns = list(rule.get("any_regex", []))
        all_groups = list(rule.get("all_regex_groups", []))

        for span_id, text in spans.items():
            for pattern in any_patterns:
                match, blocked = _first_positive_regex_match(
                    pattern=pattern,
                    text=text,
                    config=config,
                )
                blocked_count += blocked
                if match is not None:
                    return (
                        DeterministicDirectCueMatch(
                            cue_id=str(rule["cue_id"]),
                            evidence_span_id=span_id,
                            evidence_text=text,
                            matched_expression=pattern,
                        ),
                        blocked_count,
                    )

            for group in all_groups:
                if not group:
                    continue

                group_matches: list[re.Match[str]] = []
                group_blocked = 0

                for pattern in group:
                    match, blocked = _first_positive_regex_match(
                        pattern=pattern,
                        text=text,
                        config=config,
                    )
                    group_blocked += blocked
                    if match is None:
                        group_matches = []
                        break
                    group_matches.append(match)

                blocked_count += group_blocked

                if len(group_matches) == len(group):
                    return (
                        DeterministicDirectCueMatch(
                            cue_id=str(rule["cue_id"]),
                            evidence_span_id=span_id,
                            evidence_text=text,
                            matched_expression=" ALL ".join(group),
                        ),
                        blocked_count,
                    )

    return None, blocked_count


def find_deterministic_direct_cue(
    spec_query_id: str,
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
) -> DeterministicDirectCueMatch | None:
    """Backward-compatible public helper returning only the positive cue match."""

    match, _ = _scan_deterministic_direct_cue(
        spec_query_id=spec_query_id,
        facet=facet,
        spans=spans,
        config=config,
    )
    return match


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
    inference_kind: EvidenceInferenceKind
    hard_exclusion_triggered: bool = False
    hard_exclusion_id: str | None = None
    reason: str = Field(min_length=1)


class CompositeEvidenceVerification(BaseModel):
    """Self-selecting full-context multi-span verification output."""

    supporting_span_ids: list[str] = Field(
        default_factory=list,
        max_length=MAX_COMPOSITE_CANDIDATES,
    )
    verification_relation: VerificationRelation
    inference_kind: EvidenceInferenceKind
    hard_exclusion_triggered: bool = False
    hard_exclusion_id: str | None = None
    combined_evidence_summary: str = Field(min_length=1)
    missing_semantic_component: str | None = None
    reason: str = Field(min_length=1)


class BookSubjectAnalysis(BaseModel):
    """Facet-independent book-level subject/premise analysis, frozen once per case."""

    primary_subject_summary: str = Field(min_length=1)
    primary_subject_span_ids: list[str] = Field(min_length=1, max_length=6)
    reason: str = Field(min_length=1)


class ProminenceAssessment(BaseModel):
    """Facet role signals evaluated against the already-frozen book subject."""

    is_primary_subject: bool
    is_background_cause_or_factor: bool
    is_example_or_illustration: bool
    is_meta_discussion: bool
    is_substantively_examined: bool
    supporting_span_ids: list[str] = Field(default_factory=list, max_length=6)
    reason: str = Field(min_length=1)


class HardExclusionPrecheck(BaseModel):
    """Full-description negative-boundary precheck before positive evidence search."""

    hard_exclusion_triggered: bool
    hard_exclusion_id: str | None = None
    supporting_span_ids: list[str] = Field(default_factory=list, max_length=4)
    reason: str = Field(min_length=1)


class SemanticGenerationResult(BaseModel):
    """Complete per-case semantic result consumed by the runner."""

    verdict: FacetJudgeVerdict

    # Winning/representative exact evidence used by legacy compatibility fields.
    evidence_by_id: dict[str, str]

    # Every Stage-A candidate plus any full-context composite supporting spans, retained for audit in facet_evidence_json.
    candidate_evidence_by_id: dict[str, dict[str, str]]

    book_subject_analysis: BookSubjectAnalysis

    subject_analysis_retry_count: int = 0

    generation_mode: str = "facet_independent_book_subject_plus_full_description_hard_exclusion_precheck_plus_inference_kind_plus_decomposed_role_pipeline"

    stage_retry_count: int = 0

    deterministic_direct_cue_count: int = 0

    deterministic_cue_polarity_blocked_count: int = 0

    composite_verification_attempt_count: int = 0

    composite_verification_count: int = 0

    hard_exclusion_precheck_attempt_count: int = 0
    hard_exclusion_precheck_trigger_count: int = 0


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




def _speculative_bridge_match(text: str, config: dict[str, Any]) -> str | None:
    """Return the first configured possibility/association phrase in text."""

    for pattern in config.get("inference_contract", {}).get(
        "entailed_forbidden_reason_regex", []
    ):
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return None


def format_hard_exclusions(facet: QueryFacet) -> str:
    """Render frozen hard exclusions for verifier prompts."""

    if not facet.hard_exclusions:
        return "- NONE"
    return "\n".join(
        f"- {item.exclusion_id}: {item.rule}"
        for item in facet.hard_exclusions
    )


def build_hard_exclusion_precheck_prompt(
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
) -> str:
    """Judge frozen negative boundaries against the full description first."""

    stage = config.get("hard_exclusion_precheck_stage", {})
    instructions = "\n".join(
        f"{i}. {x}" for i, x in enumerate(stage.get("instructions", []), start=1)
    )
    return f"""
Decide whether ONE frozen hard exclusion conclusively blocks ONE semantic facet.

FACET
{facet.text}

FROZEN SEMANTIC DEFINITION
{facet.semantic_definition}

FROZEN HARD EXCLUSIONS
{format_hard_exclusions(facet)}

FULL NUMBERED BOOK-DESCRIPTION SPANS
{format_description_spans(spans)}

INSTRUCTIONS
{instructions}

Return one HardExclusionPrecheck with hard_exclusion_triggered, hard_exclusion_id,
supporting_span_ids, and reason.
""".strip()


def assess_hard_exclusion_precheck(
    judge_model: Any,
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
) -> tuple[HardExclusionPrecheck, int]:
    """Run the full-description hard-exclusion guard before positive search."""

    last_error: Exception | None = None
    valid_ids = {item.exclusion_id for item in facet.hard_exclusions}
    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):
        try:
            generated = judge_model.generate(
                prompt=build_hard_exclusion_precheck_prompt(facet, spans, config),
                schema=HardExclusionPrecheck,
            )
            result = unpack_generated_model(generated, HardExclusionPrecheck)
            assert isinstance(result, HardExclusionPrecheck)
            unknown = set(result.supporting_span_ids) - set(spans)
            if unknown:
                raise JudgeOutputValidationError(
                    f"Hard-exclusion precheck returned unknown spans: {sorted(unknown)}."
                )
            if result.hard_exclusion_triggered:
                if result.hard_exclusion_id not in valid_ids:
                    raise JudgeOutputValidationError(
                        "Triggered precheck must name a frozen exclusion ID."
                    )
                if not result.supporting_span_ids:
                    raise JudgeOutputValidationError(
                        "Triggered precheck must cite at least one supporting span."
                    )
            elif result.hard_exclusion_id is not None:
                raise JudgeOutputValidationError(
                    "Non-triggered precheck must return hard_exclusion_id=null."
                )
            return result, attempt - 1
        except Exception as error:
            last_error = error
    raise JudgeOutputValidationError(
        "Unable to obtain valid hard-exclusion precheck assessment."
    ) from last_error


def _validate_hard_exclusion_contract(
    *,
    relation: VerificationRelation,
    inference_kind: EvidenceInferenceKind,
    triggered: bool,
    exclusion_id: str | None,
    facet: QueryFacet,
    stage_name: str,
) -> None:
    """Mechanically enforce structured hard-exclusion consistency."""

    valid_ids = {item.exclusion_id for item in facet.hard_exclusions}
    if triggered:
        if not exclusion_id or exclusion_id not in valid_ids:
            raise JudgeOutputValidationError(
                f"{stage_name}: triggered hard exclusion must name one of {sorted(valid_ids)}."
            )
        if relation != VerificationRelation.UNSUPPORTED:
            raise JudgeOutputValidationError(
                f"{stage_name}: hard exclusion {exclusion_id} requires UNSUPPORTED."
            )
        if inference_kind != EvidenceInferenceKind.NONE:
            raise JudgeOutputValidationError(
                f"{stage_name}: triggered hard exclusion requires inference_kind=none."
            )
    elif exclusion_id is not None:
        raise JudgeOutputValidationError(
            f"{stage_name}: hard_exclusion_id must be null when not triggered."
        )


def _context_role_from_decomposed_signals(
    result: ProminenceAssessment,
) -> FacetContextRole:
    """Resolve decomposed role signals deterministically.

    v0.22.0 preserves the v0.21.1 resolver precedence: meta/example use stays
    incidental, while a genuinely primary or substantively examined facet is
    not suppressed by background-cause. v0.22.0 tightens the upstream LLM role
    classification rather than changing this deterministic resolver.
    """

    if result.is_meta_discussion:
        return FacetContextRole.META_DISCUSSION
    if result.is_example_or_illustration:
        return FacetContextRole.EXAMPLE_OR_ILLUSTRATION
    if result.is_primary_subject:
        return FacetContextRole.CENTRAL_SUBJECT
    if result.is_substantively_examined:
        return FacetContextRole.SUBSTANTIVE_SUBJECT
    if result.is_background_cause_or_factor:
        return FacetContextRole.BACKGROUND_CAUSE
    return FacetContextRole.INCIDENTAL_MENTION


def _prominence_from_context_role(
    context_role: FacetContextRole,
    config: dict[str, Any],
) -> FacetProminence:
    """Derive prominence only in Python from the frozen context-role mapping."""

    mapping = config.get("prominence_stage", {}).get("context_role_to_prominence", {})
    expected = mapping.get(context_role.value)
    if expected is None:
        raise JudgeOutputValidationError(
            f"Unknown context_role mapping: {context_role.value}."
        )
    return FacetProminence(expected)


def _validate_inference_contract(
    relation: VerificationRelation,
    inference_kind: EvidenceInferenceKind,
    reason: str,
    config: dict[str, Any],
    *,
    stage_name: str,
) -> None:
    """Mechanically enforce relation <-> inference-kind consistency."""

    raw = config.get("inference_contract", {}).get("relation_mapping", {})
    allowed = set(raw.get(relation.value, []))
    if allowed and inference_kind.value not in allowed:
        raise JudgeOutputValidationError(
            f"{stage_name}: relation={relation.value} is incompatible with "
            f"inference_kind={inference_kind.value}; allowed={sorted(allowed)}."
        )

    if relation == VerificationRelation.ENTAILED:
        pattern = _speculative_bridge_match(reason, config)
        if pattern is not None:
            raise JudgeOutputValidationError(
                f"{stage_name}: ENTAILED reason uses speculative bridge language "
                f"matching {pattern!r}."
            )


def _validate_verification_result(
    result: EvidenceVerification,
    facet: QueryFacet,
    config: dict[str, Any],
) -> None:
    _validate_hard_exclusion_contract(
        relation=result.verification_relation,
        inference_kind=result.inference_kind,
        triggered=result.hard_exclusion_triggered,
        exclusion_id=result.hard_exclusion_id,
        facet=facet,
        stage_name="single-span verification",
    )
    _validate_inference_contract(
        relation=result.verification_relation,
        inference_kind=result.inference_kind,
        reason=result.reason,
        config=config,
        stage_name="single-span verification",
    )


def _validate_book_subject_result(
    result: BookSubjectAnalysis,
    spans: dict[str, str],
) -> None:
    """Mechanically validate the frozen subject analysis against source spans."""

    ids = result.primary_subject_span_ids
    if len(ids) != len(set(ids)):
        raise JudgeOutputValidationError(
            "Book subject analysis returned duplicate primary_subject_span_ids."
        )
    unknown = set(ids) - set(spans)
    if unknown:
        raise JudgeOutputValidationError(
            f"Book subject analysis returned unknown span IDs: {sorted(unknown)}."
        )


def _validate_prominence_result(
    result: ProminenceAssessment,
    spans: dict[str, str],
    config: dict[str, Any],
) -> None:
    """Validate decomposed role output; Python alone resolves final role."""

    unknown = set(result.supporting_span_ids) - set(spans)
    if unknown:
        raise JudgeOutputValidationError(
            f"Role assessment returned unknown supporting span IDs: {sorted(unknown)}."
        )
    role = _context_role_from_decomposed_signals(result)
    _prominence_from_context_role(role, config)


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

HARD EXCLUSIONS — APPLY BEFORE ANY POSITIVE RELATION
{format_hard_exclusions(facet)}

EXACT CANDIDATE EVIDENCE
{evidence_text}

VERIFICATION SCALE
{scale}

DECISION PRECEDENCE
{precedence}

INSTRUCTIONS
{instructions}

Return one EvidenceVerification with verification_relation, inference_kind, hard_exclusion_triggered, hard_exclusion_id, and reason.

The inference_kind must obey the configured relation mapping.
Do not use any information other than the facet, frozen semantic definition, and evidence shown above.
""".strip()


def verify_candidate_evidence(
    judge_model: Any,
    facet: QueryFacet,
    evidence_text: str,
    config: dict[str, Any],
) -> tuple[EvidenceVerification, int]:
    """Verify one candidate independently and enforce inference-kind consistency."""

    validation_error: str | None = None

    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):
        prompt = build_verification_prompt(
            facet=facet,
            evidence_text=evidence_text,
            config=config,
        )
        if validation_error:
            prompt += (
                "\n\nPREVIOUS VALIDATION FAILURE\n"
                f"{validation_error}\n"
                "Return a corrected EvidenceVerification only. ENTAILED requires "
                "inference_kind=necessary_semantic_inference and a necessary, "
                "non-speculative justification."
            )

        generated = judge_model.generate(
            prompt=prompt,
            schema=EvidenceVerification,
        )

        try:
            result = unpack_generated_model(generated, EvidenceVerification)
            assert isinstance(result, EvidenceVerification)
            _validate_verification_result(result=result, facet=facet, config=config)
            return result, attempt - 1
        except (JudgeOutputValidationError, ValueError, TypeError) as error:
            validation_error = str(error)

    raise JudgeOutputValidationError(
        validation_error or "Unable to obtain valid isolated evidence verification."
    )


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
            inference_kind=verification.inference_kind,
            hard_exclusion_triggered=verification.hard_exclusion_triggered,
            hard_exclusion_id=verification.hard_exclusion_id,
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
# Stage C: self-selecting full-context composite verification
# =============================================================================

def build_composite_verification_prompt(
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
) -> str:
    """Build the full-context self-selecting multi-span composition prompt."""

    stage = config["composite_verification_stage"]

    scale = "\n".join(
        f"- {name}: {definition}"
        for name, definition
        in stage["relation_scale"].items()
    )

    instructions = "\n".join(
        f"{index}. {instruction}"
        for index, instruction
        in enumerate(stage["instructions"], start=1)
    )

    min_spans = int(stage.get("min_spans", 2))
    max_spans = int(stage.get("max_spans", MAX_COMPOSITE_CANDIDATES))

    return f"""
Evaluate ONE core semantic facet using the FULL numbered book description.

Your task has TWO inseparable parts:
1. identify the smallest coherent set of exact source spans that jointly bears
   on the facet; and
2. decide whether those spans make the facet UNSUPPORTED, ADJACENT, or ENTAILED.

FACET
{facet.text}

FROZEN SEMANTIC DEFINITION
{facet.semantic_definition}

HARD EXCLUSIONS — APPLY BEFORE ANY POSITIVE RELATION
{format_hard_exclusions(facet)}

FULL NUMBERED BOOK-DESCRIPTION SPANS
{format_description_spans(spans)}

COMPOSITE VERIFICATION SCALE
{scale}

INSTRUCTIONS
{instructions}

OUTPUT
Return one CompositeEvidenceVerification with:

supporting_span_ids:
- [] for UNSUPPORTED
- {min_spans}-{max_spans} UNIQUE supplied S# IDs for ADJACENT or ENTAILED

verification_relation:
- unsupported, adjacent, or entailed only

inference_kind:
- hard-exclusion-triggered unsupported => none
- other unsupported => none, symbolic_possibility, or associative_world_knowledge
- adjacent => incomplete_connection
- entailed => necessary_semantic_inference

hard_exclusion_triggered / hard_exclusion_id:
- evaluate the frozen hard exclusions first
- if one applies, return true plus its exact exclusion_id and UNSUPPORTED
- otherwise return false and null

combined_evidence_summary:
- summarize only what the cited spans jointly establish
- do not add facts not present in those spans

missing_semantic_component:
- null for ENTAILED
- for ADJACENT, name the specific required facet component that remains
  unstated or merely plausible
- for UNSUPPORTED, null is preferred

reason:
- explain why the cited spans and missing-component field justify the relation

DIRECT is forbidden.
Do not require exact facet wording for ENTAILED.
Do not use any information outside the facet, frozen definition, and supplied
numbered description.
""".strip()


def _validate_composite_result(
    result: CompositeEvidenceVerification,
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
) -> None:
    """Deterministically validate v0.20 composite structure/consistency."""

    stage = config["composite_verification_stage"]
    min_spans = int(stage.get("min_spans", 2))
    max_spans = int(stage.get("max_spans", MAX_COMPOSITE_CANDIDATES))
    relation = result.verification_relation
    ids = list(result.supporting_span_ids)

    if relation == VerificationRelation.DIRECT:
        raise JudgeOutputValidationError(
            "Composite verification may not return DIRECT."
        )

    _validate_hard_exclusion_contract(
        relation=relation,
        inference_kind=result.inference_kind,
        triggered=result.hard_exclusion_triggered,
        exclusion_id=result.hard_exclusion_id,
        facet=facet,
        stage_name="composite verification",
    )

    _validate_inference_contract(
        relation=relation,
        inference_kind=result.inference_kind,
        reason=(
            result.combined_evidence_summary + "\n" + result.reason
        ),
        config=config,
        stage_name="composite verification",
    )

    if len(ids) != len(set(ids)):
        raise JudgeOutputValidationError(
            "Composite supporting span IDs must be unique."
        )

    unknown = [span_id for span_id in ids if span_id not in spans]
    if unknown:
        raise JudgeOutputValidationError(
            f"Composite verification returned unknown span IDs: {unknown}."
        )

    if relation in {
        VerificationRelation.ADJACENT,
        VerificationRelation.ENTAILED,
    }:
        if not (min_spans <= len(ids) <= max_spans):
            raise JudgeOutputValidationError(
                f"Positive composite verification requires {min_spans}-{max_spans} "
                "supporting spans."
            )
    else:
        if ids:
            raise JudgeOutputValidationError(
                "Unsupported composite verification must return no supporting span IDs."
            )

    missing = (
        result.missing_semantic_component.strip()
        if result.missing_semantic_component
        else ""
    )

    if relation == VerificationRelation.ADJACENT and not missing:
        raise JudgeOutputValidationError(
            "ADJACENT composite verification must identify the missing semantic component."
        )

    if relation == VerificationRelation.ENTAILED and missing:
        raise JudgeOutputValidationError(
            "ENTAILED composite verification cannot declare a missing semantic component."
        )


def verify_composite_evidence(
    judge_model: Any,
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
) -> tuple[CompositeEvidenceVerification, int]:
    """
    Scan the full description, self-select 2-4 supporting spans, and determine
    the composite relation in one isolated call.
    """

    validation_error: str | None = None

    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):
        prompt = build_composite_verification_prompt(
            facet=facet,
            spans=spans,
            config=config,
        )

        if validation_error:
            prompt += (
                "\n\nPREVIOUS VALIDATION FAILURE\n"
                f"{validation_error}\n"
                "Return a corrected CompositeEvidenceVerification only. "
                "Remember: ENTAILED => missing_semantic_component=null; "
                "ADJACENT => explicitly identify the missing required component; "
                "positive relations require 2-4 unique real span IDs; relation and inference_kind must match."
            )

        # Transport/model-call failures intentionally propagate. Only generated
        # structured-output validation failures are repaired/fallback-handled.
        generated = judge_model.generate(
            prompt=prompt,
            schema=CompositeEvidenceVerification,
        )

        try:
            result = unpack_generated_model(
                generated,
                CompositeEvidenceVerification,
            )
            assert isinstance(result, CompositeEvidenceVerification)
            _validate_composite_result(
                result=result,
                facet=facet,
                spans=spans,
                config=config,
            )
            return result, attempt - 1

        except (JudgeOutputValidationError, ValueError, TypeError) as error:
            validation_error = str(error)

    # Composition is an optional recovery path. A malformed response after
    # bounded retries must not create positive evidence or abort the case.
    fallback = CompositeEvidenceVerification(
        supporting_span_ids=[],
        verification_relation=VerificationRelation.UNSUPPORTED,
        inference_kind=EvidenceInferenceKind.NONE,
        hard_exclusion_triggered=False,
        hard_exclusion_id=None,
        combined_evidence_summary=(
            "No valid composite evidence judgment was available after bounded "
            "structured-output repair attempts."
        ),
        missing_semantic_component=None,
        reason=(
            "structured_output_recovery_fallback: treating optional composite "
            "recovery as unsupported. Last validation error: "
            f"{validation_error or 'unknown'}"
        ),
    )

    return fallback, MAX_STAGE_ATTEMPTS



def format_composite_evidence(
    span_ids: list[str],
    spans: dict[str, str],
) -> str:
    """Render exact composed evidence while preserving source-span IDs."""

    return "\n".join(
        f"{span_id}: {spans[span_id]}"
        for span_id in span_ids
    )


# =============================================================================
# Stage C0: facet-independent book subject extraction
# =============================================================================

def build_book_subject_prompt(
    spans: dict[str, str],
    config: dict[str, Any],
) -> str:
    """Build a book-level subject prompt that contains no query or facet."""

    stage = config["book_subject_stage"]
    instructions = "\n".join(
        f"{index}. {instruction}"
        for index, instruction in enumerate(stage["instructions"], start=1)
    )

    return f"""
Analyze ONE supplied book description to identify what the BOOK ITSELF is primarily about.

IMPORTANT ISOLATION RULE
No user query or query facet is available at this stage. Base the answer only on the supplied description.

FULL NUMBERED BOOK-DESCRIPTION SPANS
{format_description_spans(spans)}

INSTRUCTIONS
{instructions}

Return one BookSubjectAnalysis with primary_subject_summary,
primary_subject_span_ids, and reason.
""".strip()


def assess_book_subject(
    judge_model: Any,
    spans: dict[str, str],
    config: dict[str, Any],
) -> tuple[BookSubjectAnalysis, int]:
    """Analyze/freeze the book-level subject exactly once for this case."""

    last_error: Exception | None = None

    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):
        try:
            generated = judge_model.generate(
                prompt=build_book_subject_prompt(spans=spans, config=config),
                schema=BookSubjectAnalysis,
            )
            result = unpack_generated_model(generated, BookSubjectAnalysis)
            assert isinstance(result, BookSubjectAnalysis)
            _validate_book_subject_result(result=result, spans=spans)
            return result, attempt - 1
        except Exception as error:
            last_error = error

    raise JudgeOutputValidationError(
        "Unable to obtain valid facet-independent book subject analysis."
    ) from last_error


# =============================================================================
# Stage C1: core facet role against the frozen book subject
# =============================================================================

def build_prominence_prompt(
    facet: QueryFacet,
    evidence_text: str,
    book_subject: BookSubjectAnalysis,
    spans: dict[str, str],
    config: dict[str, Any],
) -> str:
    """Judge a verified core facet relative to the frozen case-level subject."""

    stage = config["prominence_stage"]

    context_scale = "\n".join(
        f"- {name}: {definition}"
        for name, definition in stage["context_role_scale"].items()
    )

    instructions = "\n".join(
        f"{index}. {instruction}"
        for index, instruction in enumerate(stage["instructions"], start=1)
    )

    frozen_subject_evidence = format_composite_evidence(
        book_subject.primary_subject_span_ids,
        spans,
    )

    return f"""
Assess the role/prominence of ONE already-verified semantic facet in a book description.

FACET
{facet.text}

FROZEN SEMANTIC DEFINITION
{facet.semantic_definition}

VERIFIED FACET EVIDENCE
{evidence_text}

FROZEN FACET-INDEPENDENT BOOK SUBJECT
{book_subject.primary_subject_summary}

BOOK-SUBJECT SUPPORTING SPANS
{frozen_subject_evidence}

FULL NUMBERED BOOK-DESCRIPTION SPANS
{format_description_spans(spans)}

CONTEXT ROLE SCALE
{context_scale}

INSTRUCTIONS
{instructions}

Return one ProminenceAssessment with the five boolean role signals,
supporting_span_ids, and reason.
Do NOT return or redefine the book subject. Python derives final context_role
and prominence after this stage.
""".strip()


def assess_core_prominence(
    judge_model: Any,
    facet: QueryFacet,
    evidence_text: str,
    book_subject: BookSubjectAnalysis,
    spans: dict[str, str],
    config: dict[str, Any],
) -> tuple[ProminenceAssessment, int]:
    """Get one valid core-facet role classification against frozen subject."""

    last_error: Exception | None = None

    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):
        try:
            generated = judge_model.generate(
                prompt=build_prominence_prompt(
                    facet=facet,
                    evidence_text=evidence_text,
                    book_subject=book_subject,
                    spans=spans,
                    config=config,
                ),
                schema=ProminenceAssessment,
            )
            result = unpack_generated_model(generated, ProminenceAssessment)
            assert isinstance(result, ProminenceAssessment)
            _validate_prominence_result(result=result, spans=spans, config=config)
            return result, attempt - 1
        except Exception as error:
            last_error = error

    raise JudgeOutputValidationError(
        "Unable to obtain valid core prominence assessment against frozen book subject."
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
    book_subject: BookSubjectAnalysis | None = None,
) -> tuple[
    FacetPipelineAssessment,
    str | None,
    dict[str, str],
    int,
]:
    """
    Run v0.22.0 facet evaluation against one frozen book subject:

        full-description hard-exclusion precheck
        -> polarity-safe deterministic cue
        -> Stage-A standalone candidate selection
        -> isolated single-span verification(s)
        -> self-selecting full-context composite verification when eligible
        -> optional prominence

    Returns:
        final assessment
        winning/representative evidence text
        exact text for every Stage-A OR composition-selected candidate
        total structured-output retry count
    """

    total_retries = 0

    precheck_attempted = bool(facet.hard_exclusions)
    precheck_triggered = False
    precheck_id: str | None = None
    precheck_supporting_span_ids: list[str] = []
    precheck_reason: str | None = None

    if precheck_attempted:
        precheck, retries = assess_hard_exclusion_precheck(
            judge_model=judge_model,
            facet=facet,
            spans=spans,
            config=config,
        )
        total_retries += retries
        precheck_triggered = precheck.hard_exclusion_triggered
        precheck_id = precheck.hard_exclusion_id
        precheck_supporting_span_ids = list(precheck.supporting_span_ids)
        precheck_reason = precheck.reason

        if precheck_triggered:
            candidate_texts = {
                span_id: spans[span_id]
                for span_id in precheck_supporting_span_ids
            }
            return (
                FacetPipelineAssessment(
                    facet_id=facet.facet_id,
                    candidate_evidence_span_ids=[],
                    candidate_verifications=[],
                    candidate_evidence_span_id=NO_EVIDENCE_SPAN,
                    verification_relation=VerificationRelation.UNSUPPORTED,
                    inference_kind=EvidenceInferenceKind.NONE,
                    hard_exclusion_triggered=True,
                    hard_exclusion_id=precheck_id,
                    hard_exclusion_precheck_attempted=True,
                    hard_exclusion_precheck_triggered=True,
                    hard_exclusion_precheck_id=precheck_id,
                    hard_exclusion_precheck_supporting_span_ids=precheck_supporting_span_ids,
                    hard_exclusion_precheck_reason=precheck_reason,
                    prominence=FacetProminence.NOT_APPLICABLE,
                    context_role=FacetContextRole.NOT_APPLICABLE,
                    evidence_selection_reason="skipped: full-description hard-exclusion precheck triggered",
                    verification_reason=precheck_reason or "full-description hard-exclusion precheck triggered",
                    prominence_reason=None,
                ),
                None,
                candidate_texts,
                total_retries,
            )

    cue_match, polarity_blocked_count = _scan_deterministic_direct_cue(
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
                inference_kind=EvidenceInferenceKind.EXPLICIT_COMPONENTS,
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
                    inference_kind=EvidenceInferenceKind.EXPLICIT_COMPONENTS,
                    hard_exclusion_precheck_attempted=precheck_attempted,
                    hard_exclusion_precheck_triggered=precheck_triggered,
                    hard_exclusion_precheck_id=precheck_id,
                    hard_exclusion_precheck_supporting_span_ids=precheck_supporting_span_ids,
                    hard_exclusion_precheck_reason=precheck_reason,
                    prominence=FacetProminence.NOT_APPLICABLE,
                    context_role=FacetContextRole.NOT_APPLICABLE,
                    evidence_selection_reason=reason,
                    verification_reason=reason,
                    prominence_reason=None,
                    deterministic_cue_polarity_blocked_count=polarity_blocked_count,
                ),
                cue_match.evidence_text,
                candidate_texts,
                total_retries,
            )

        if book_subject is None:
            raise JudgeOutputValidationError(
                "Verified core facet requires frozen book subject analysis."
            )
        prominence_result, retries = assess_core_prominence(
            judge_model=judge_model,
            facet=facet,
            evidence_text=cue_match.evidence_text,
            book_subject=book_subject,
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
                inference_kind=EvidenceInferenceKind.EXPLICIT_COMPONENTS,
                prominence=_prominence_from_context_role(_context_role_from_decomposed_signals(prominence_result), config),
                context_role=_context_role_from_decomposed_signals(prominence_result),
                primary_subject_summary=book_subject.primary_subject_summary,
                primary_subject_span_ids=book_subject.primary_subject_span_ids,
                is_primary_subject=prominence_result.is_primary_subject,
                is_background_cause_or_factor=prominence_result.is_background_cause_or_factor,
                is_example_or_illustration=prominence_result.is_example_or_illustration,
                is_meta_discussion=prominence_result.is_meta_discussion,
                is_substantively_examined=prominence_result.is_substantively_examined,
                role_supporting_span_ids=prominence_result.supporting_span_ids,
                role_reason=prominence_result.reason,
                evidence_selection_reason=reason,
                verification_reason=reason,
                prominence_reason=prominence_result.reason,
                deterministic_cue_polarity_blocked_count=polarity_blocked_count,
            ),
            cue_match.evidence_text,
            candidate_texts,
            total_retries,
        )

    # ------------------------------------------------------------------
    # Stage A/B: standalone evidence path.
    # ------------------------------------------------------------------
    selection, candidate_ids, retries = select_candidate_evidence(
        judge_model=judge_model,
        facet=facet,
        spans=spans,
        config=config,
    )
    total_retries += retries

    candidate_texts = {
        span_id: spans[span_id]
        for span_id in candidate_ids
    }

    verification_records: list[CandidateVerificationRecord] = []
    winning_span_id = NO_EVIDENCE_SPAN
    winning_evidence_text: str | None = None
    relation = VerificationRelation.UNSUPPORTED
    inference_kind = EvidenceInferenceKind.NONE
    hard_exclusion_triggered = False
    hard_exclusion_id: str | None = None
    best_single_reason = (
        "Evidence selector returned NONE; isolated single-span verification skipped."
    )

    if candidate_ids:
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
        inference_kind = best.inference_kind
        hard_exclusion_triggered = best.hard_exclusion_triggered
        hard_exclusion_id = best.hard_exclusion_id
        best_single_reason = best.verification_reason

    # ------------------------------------------------------------------
    # Stage C: self-selecting full-context composition recovery.
    # ------------------------------------------------------------------
    composite_verification_attempted = False
    composite_span_ids: list[str] = []
    composite_relation: VerificationRelation | None = None
    composite_inference_kind: EvidenceInferenceKind | None = None
    composite_hard_exclusion_triggered = False
    composite_hard_exclusion_id: str | None = None
    composite_summary: str | None = None
    composite_missing_component: str | None = None
    composite_reason: str | None = None

    verifier_stage = config.get("composite_verification_stage", {})

    eligible_relations = {
        VerificationRelation(value)
        for value in verifier_stage.get(
            "eligible_when_best_single_relation",
            ["unsupported", "adjacent"],
        )
    }

    if (
        verifier_stage
        and facet.facet_type == "core"
        and relation in eligible_relations
    ):
        composite_verification_attempted = True

        composite_result, retries = verify_composite_evidence(
            judge_model=judge_model,
            facet=facet,
            spans=spans,
            config=config,
        )
        total_retries += retries

        composite_relation = composite_result.verification_relation
        composite_inference_kind = composite_result.inference_kind
        composite_hard_exclusion_triggered = composite_result.hard_exclusion_triggered
        composite_hard_exclusion_id = composite_result.hard_exclusion_id
        composite_summary = composite_result.combined_evidence_summary
        composite_missing_component = (
            composite_result.missing_semantic_component
        )
        composite_reason = composite_result.reason
        composite_span_ids = list(
            composite_result.supporting_span_ids
        )

        # Persist exact full-context evidence chosen by the composite verifier,
        # including spans Stage A never selected.
        for span_id in composite_span_ids:
            candidate_texts[span_id] = spans[span_id]

        # Composition is recovery-only. It replaces the standalone relation
        # only when strictly stronger. DIRECT remains impossible here.
        if (
            VERIFICATION_STRENGTH[composite_relation]
            > VERIFICATION_STRENGTH[relation]
        ):
            relation = composite_relation
            inference_kind = composite_inference_kind or EvidenceInferenceKind.NONE
            hard_exclusion_triggered = composite_hard_exclusion_triggered
            hard_exclusion_id = composite_hard_exclusion_id
            winning_evidence_text = format_composite_evidence(
                composite_span_ids,
                spans,
            )

    final_verification_reason = best_single_reason

    if (
        composite_relation is not None
        and VERIFICATION_STRENGTH[composite_relation]
        > VERIFICATION_STRENGTH[
            (
                verification_records
                and choose_best_verified_candidate(verification_records).verification_relation
            )
            or VerificationRelation.UNSUPPORTED
        ]
    ):
        final_verification_reason = composite_reason or best_single_reason

    common_kwargs = dict(
        facet_id=facet.facet_id,
        candidate_evidence_span_ids=candidate_ids,
        candidate_verifications=verification_records,
        candidate_evidence_span_id=winning_span_id,
        verification_relation=relation,
        inference_kind=inference_kind,
        hard_exclusion_triggered=hard_exclusion_triggered,
        hard_exclusion_id=hard_exclusion_id,
        hard_exclusion_precheck_attempted=precheck_attempted,
        hard_exclusion_precheck_triggered=precheck_triggered,
        hard_exclusion_precheck_id=precheck_id,
        hard_exclusion_precheck_supporting_span_ids=precheck_supporting_span_ids,
        hard_exclusion_precheck_reason=precheck_reason,
        evidence_selection_reason=selection.reason,
        verification_reason=final_verification_reason,
        deterministic_cue_polarity_blocked_count=polarity_blocked_count,
        composite_verification_attempted=composite_verification_attempted,
        composite_evidence_span_ids=composite_span_ids,
        composite_verification_relation=composite_relation,
        composite_inference_kind=composite_inference_kind,
        composite_hard_exclusion_triggered=composite_hard_exclusion_triggered,
        composite_hard_exclusion_id=composite_hard_exclusion_id,
        composite_combined_evidence_summary=composite_summary,
        composite_missing_semantic_component=composite_missing_component,
        composite_verification_reason=composite_reason,
    )

    if relation == VerificationRelation.UNSUPPORTED:
        return (
            FacetPipelineAssessment(
                **common_kwargs,
                prominence=FacetProminence.NOT_APPLICABLE,
                context_role=FacetContextRole.NOT_APPLICABLE,
                prominence_reason=None,
            ),
            winning_evidence_text,
            candidate_texts,
            total_retries,
        )

    if (
        relation == VerificationRelation.ADJACENT
        and facet.facet_type == "core"
    ):
        return (
            FacetPipelineAssessment(
                **common_kwargs,
                prominence=FacetProminence.NOT_APPLICABLE,
                context_role=FacetContextRole.NOT_APPLICABLE,
                prominence_reason=None,
            ),
            winning_evidence_text,
            candidate_texts,
            total_retries,
        )

    if facet.facet_type == "qualifier":
        return (
            FacetPipelineAssessment(
                **common_kwargs,
                prominence=FacetProminence.NOT_APPLICABLE,
                context_role=FacetContextRole.NOT_APPLICABLE,
                prominence_reason=None,
            ),
            winning_evidence_text,
            candidate_texts,
            total_retries,
        )

    assert winning_evidence_text is not None

    if book_subject is None:
        raise JudgeOutputValidationError(
            "Verified core facet requires frozen book subject analysis."
        )
    prominence_result, retries = assess_core_prominence(
        judge_model=judge_model,
        facet=facet,
        evidence_text=winning_evidence_text,
        book_subject=book_subject,
        spans=spans,
        config=config,
    )
    total_retries += retries

    return (
        FacetPipelineAssessment(
            **common_kwargs,
            prominence=_prominence_from_context_role(_context_role_from_decomposed_signals(prominence_result), config),
            context_role=_context_role_from_decomposed_signals(prominence_result),
            primary_subject_summary=book_subject.primary_subject_summary,
                primary_subject_span_ids=book_subject.primary_subject_span_ids,
            is_primary_subject=prominence_result.is_primary_subject,
            is_background_cause_or_factor=prominence_result.is_background_cause_or_factor,
            is_example_or_illustration=prominence_result.is_example_or_illustration,
            is_meta_discussion=prominence_result.is_meta_discussion,
            is_substantively_examined=prominence_result.is_substantively_examined,
            role_supporting_span_ids=prominence_result.supporting_span_ids,
            role_reason=prominence_result.reason,
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
    Run v0.22.0 with one frozen book-subject analysis followed by every frozen facet.

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

    # v0.22.0: freeze one facet-independent subject analysis before ANY facet
    # role classification. No query/facet is visible to this call.
    book_subject, subject_retries = assess_book_subject(
        judge_model=judge_model,
        spans=spans,
        config=config,
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
            book_subject=book_subject,
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

    deterministic_cue_polarity_blocked_count = sum(
        assessment.deterministic_cue_polarity_blocked_count
        for assessment in assessments
    )

    composite_verification_attempt_count = sum(
        assessment.composite_verification_attempted
        for assessment in assessments
    )

    composite_verification_count = sum(
        bool(assessment.composite_evidence_span_ids)
        for assessment in assessments
    )

    hard_exclusion_precheck_attempt_count = sum(
        assessment.hard_exclusion_precheck_attempted
        for assessment in assessments
    )

    hard_exclusion_precheck_trigger_count = sum(
        assessment.hard_exclusion_precheck_triggered
        for assessment in assessments
    )

    return SemanticGenerationResult(
        verdict=verdict,
        evidence_by_id=evidence_by_id,
        candidate_evidence_by_id=candidate_evidence_by_id,
        book_subject_analysis=book_subject,
        subject_analysis_retry_count=subject_retries,
        stage_retry_count=total_retries,
        deterministic_direct_cue_count=deterministic_direct_cue_count,
        deterministic_cue_polarity_blocked_count=(
            deterministic_cue_polarity_blocked_count
        ),
        composite_verification_attempt_count=composite_verification_attempt_count,
        composite_verification_count=composite_verification_count,
        hard_exclusion_precheck_attempt_count=hard_exclusion_precheck_attempt_count,
        hard_exclusion_precheck_trigger_count=hard_exclusion_precheck_trigger_count,
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
                "inference_kind": assessment.inference_kind.value,
                "hard_exclusion_triggered": assessment.hard_exclusion_triggered,
                "hard_exclusion_id": assessment.hard_exclusion_id,
                "context_role": assessment.context_role.value,
                "resolved_context_role": assessment.context_role.value,
                "prominence": assessment.prominence.value,
                "primary_subject_summary": assessment.primary_subject_summary,
                "primary_subject_span_ids": assessment.primary_subject_span_ids,
                "is_primary_subject": assessment.is_primary_subject,
                "is_background_cause_or_factor": assessment.is_background_cause_or_factor,
                "is_example_or_illustration": assessment.is_example_or_illustration,
                "is_meta_discussion": assessment.is_meta_discussion,
                "is_substantively_examined": assessment.is_substantively_examined,
                "role_supporting_span_ids": assessment.role_supporting_span_ids,
                "role_reason": assessment.role_reason,
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
                "deterministic_cue_polarity_blocked_count": (
                    assessment.deterministic_cue_polarity_blocked_count
                ),
                "composite_verification_attempted": (
                    assessment.composite_verification_attempted
                ),
                "composite_evidence_span_ids": (
                    assessment.composite_evidence_span_ids
                ),
                "composite_verification_relation": (
                    assessment.composite_verification_relation.value
                    if assessment.composite_verification_relation is not None
                    else None
                ),
                "composite_inference_kind": (
                    assessment.composite_inference_kind.value
                    if assessment.composite_inference_kind is not None
                    else None
                ),
                "composite_hard_exclusion_triggered": (
                    assessment.composite_hard_exclusion_triggered
                ),
                "composite_hard_exclusion_id": (
                    assessment.composite_hard_exclusion_id
                ),
                "composite_combined_evidence_summary": (
                    assessment.composite_combined_evidence_summary
                ),
                "composite_missing_semantic_component": (
                    assessment.composite_missing_semantic_component
                ),
                "composite_verification_reason": (
                    assessment.composite_verification_reason
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
    Persist exact text for ALL Stage-A candidates and composite supporting spans, not only the winning span.

    This makes standalone and composition evidence-selection failures auditable after the run.
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
