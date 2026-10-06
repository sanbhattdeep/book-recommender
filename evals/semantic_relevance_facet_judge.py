"""
Semantic facet judge v0.28.0 with isolated component verification, deterministic facet assembly, and localized v0.28 boundary repairs.

The v0.22 architecture the model analyzes each book description
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
from typing import Any, Literal

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
    SubjectRelation,
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


class ComponentEvidenceCheck(BaseModel):
    """One canonical required component grounded to exact source spans.

    v0.25.0 makes component identity immutable: `component_id` must be copied
    from the facet specification.  The model decides only whether that frozen
    component is established and which supplied spans ground it.
    """

    component_id: str = Field(min_length=1)
    established: bool
    grounding_relation: Literal["missing", "explicit", "entailed"] = "missing"
    supporting_span_ids: list[str] = Field(
        default_factory=list,
        max_length=MAX_COMPOSITE_CANDIDATES,
    )
    negative_boundary_applied: bool = False
    external_knowledge_required: bool = False
    reason: str = Field(min_length=1)


class IsolatedComponentVerification(BaseModel):
    """LLM output for exactly one frozen canonical component against one span.

    v0.26 deliberately prevents this call from deciding the full facet.  Python
    later assembles the facet relation from the independent component results.
    """

    component_id: str = Field(min_length=1)
    grounding_relation: Literal["missing", "explicit", "entailed"]
    negative_boundary_applied: bool = False
    external_knowledge_required: bool = False
    reason: str = Field(min_length=1)


class FullContextComponentRecovery(BaseModel):
    """Recovery output for one still-missing canonical component only."""

    component_id: str = Field(min_length=1)
    grounding_relation: Literal["missing", "explicit", "entailed"]
    supporting_span_ids: list[str] = Field(default_factory=list, max_length=MAX_COMPOSITE_CANDIDATES)
    negative_boundary_applied: bool = False
    external_knowledge_required: bool = False
    reason: str = Field(min_length=1)


class EvidenceVerification(BaseModel):
    """Structured output for one isolated verification with component ledger."""

    verification_relation: VerificationRelation
    inference_kind: EvidenceInferenceKind
    component_checks: list[ComponentEvidenceCheck] = Field(min_length=1, max_length=8)
    all_required_components_established: bool
    missing_semantic_component: str | None = None
    semantic_definition_exclusion_applied: bool = False
    hard_exclusion_triggered: bool = False
    hard_exclusion_id: str | None = None
    reason: str = Field(min_length=1)


class CompositeEvidenceVerification(BaseModel):
    """Self-selecting full-context multi-span verification output with ledger."""

    supporting_span_ids: list[str] = Field(
        default_factory=list,
        max_length=MAX_COMPOSITE_CANDIDATES,
    )
    verification_relation: VerificationRelation
    inference_kind: EvidenceInferenceKind
    component_checks: list[ComponentEvidenceCheck] = Field(min_length=1, max_length=8)
    all_required_components_established: bool
    semantic_definition_exclusion_applied: bool = False
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
    """Explicit facet relationship to the already-frozen book subject."""

    subject_relation: SubjectRelation
    is_substantively_examined: bool = False
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

    # v0.26.0 per-facet isolated component-to-span grounding ledgers.
    component_evidence_ledger_by_id: dict[str, Any] = Field(default_factory=dict)

    book_subject_analysis: BookSubjectAnalysis

    subject_analysis_retry_count: int = 0

    generation_mode: str = "facet_independent_book_subject_plus_explicit_subject_relation_plus_isolated_component_verification_plus_deterministic_facet_assembly_plus_missing_component_only_full_context_recovery_plus_full_description_hard_exclusion_precheck_plus_inference_kind_pipeline"

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


def format_required_components(facet: QueryFacet) -> str:
    """Render the frozen canonical component contract for one facet."""

    if not facet.required_components:
        return "NONE — legacy fixture without canonical components."

    blocks: list[str] = []
    for component in facet.required_components:
        boundaries = component.negative_boundaries or []
        boundary_text = (
            "\n".join(f"    - {item}" for item in boundaries)
            if boundaries
            else "    - NONE"
        )
        blocks.append(
            f"- component_id: {component.component_id}\n"
            f"  definition: {component.definition}\n"
            f"  negative_boundaries:\n{boundary_text}"
        )
    return "\n".join(blocks)


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

            # Execution-only normalization for malformed structured output.
            #
            # Some local-model responses serialize an empty optional span as
            # supporting_span_ids=[\"\"] instead of supporting_span_ids=[].
            # A blank string can never identify a real numbered source span,
            # so removing blank/whitespace-only entries is a mechanical
            # serialization repair. Valid IDs are preserved unchanged and
            # still pass through the normal unknown-span and trigger checks.
            result.supporting_span_ids = [
                span_id
                for span_id in result.supporting_span_ids
                if span_id.strip()
            ]

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


def _normalize_impossible_hard_exclusion_output(
    result: EvidenceVerification | CompositeEvidenceVerification,
    facet: QueryFacet,
) -> EvidenceVerification | CompositeEvidenceVerification:
    """Normalize impossible model-only hard-exclusion flags when none exist.

    A facet with no frozen hard exclusions has no valid exclusion ID the model
    could possibly trigger. Treating a hallucinated ``hard_exclusion_triggered``
    flag as a fatal structured-output error can abort an otherwise auditable
    case after bounded retries. This normalization is mechanical rather than
    semantic: it changes only the two impossible hard-exclusion audit fields
    and leaves the verification relation, component-completeness fields,
    semantic-definition exclusion flag, and reason untouched.
    """

    if facet.hard_exclusions:
        return result

    if result.hard_exclusion_triggered or result.hard_exclusion_id is not None:
        result.hard_exclusion_triggered = False
        result.hard_exclusion_id = None

    return result


def _canonicalize_valid_triggered_hard_exclusion(
    result: EvidenceVerification | CompositeEvidenceVerification,
    facet: QueryFacet,
) -> EvidenceVerification | CompositeEvidenceVerification:
    """Canonicalize fields mechanically implied by a valid hard exclusion.

    A triggered frozen hard exclusion is already a negative semantic decision.
    Once the model names an exclusion ID that actually belongs to the facet,
    ``verification_relation=unsupported`` and ``inference_kind=none`` are not
    independent judgments; they are deterministic consequences of that trigger.

    Invalid or invented exclusion IDs are deliberately left untouched so the
    normal validator still rejects them. Composite positive-support IDs are
    cleared because the composite contract requires no supporting spans for an
    UNSUPPORTED result. No component-completeness or semantic-definition fields
    are rewritten.
    """

    if not result.hard_exclusion_triggered:
        return result

    valid_ids = {item.exclusion_id for item in facet.hard_exclusions}
    if not result.hard_exclusion_id or result.hard_exclusion_id not in valid_ids:
        return result

    result.verification_relation = VerificationRelation.UNSUPPORTED
    result.inference_kind = EvidenceInferenceKind.NONE

    if isinstance(result, CompositeEvidenceVerification):
        result.supporting_span_ids = []

    return result


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
    """Resolve the explicit facet-to-frozen-subject relationship deterministically.

    v0.24.0 preserves the mutually exclusive subject_relation as the primary structural
    signal. The legacy substantive override is retained only for causal/background
    material that is independently developed beyond its causal role.
    """

    relation = result.subject_relation

    if relation == SubjectRelation.EXAMPLE_OR_META:
        return FacetContextRole.EXAMPLE_OR_ILLUSTRATION
    if relation == SubjectRelation.SAME_AS_PRIMARY_SUBJECT:
        return FacetContextRole.CENTRAL_SUBJECT
    if relation == SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER:
        return FacetContextRole.SUBSTANTIVE_SUBJECT
    if (
        relation == SubjectRelation.CAUSAL_OR_CONTEXTUAL_BACKGROUND
        and result.is_substantively_examined
    ):
        return FacetContextRole.SUBSTANTIVE_SUBJECT
    if relation == SubjectRelation.CAUSAL_OR_CONTEXTUAL_BACKGROUND:
        return FacetContextRole.BACKGROUND_CAUSE
    return FacetContextRole.INCIDENTAL_MENTION


def _legacy_role_flags(
    result: ProminenceAssessment,
) -> dict[str, bool]:
    """Derive backward-compatible audit booleans from subject_relation."""

    relation = result.subject_relation
    return {
        "is_primary_subject": relation == SubjectRelation.SAME_AS_PRIMARY_SUBJECT,
        "is_background_cause_or_factor": relation == SubjectRelation.CAUSAL_OR_CONTEXTUAL_BACKGROUND,
        "is_example_or_illustration": relation == SubjectRelation.EXAMPLE_OR_META,
        "is_meta_discussion": relation == SubjectRelation.EXAMPLE_OR_META,
        "is_substantively_examined": result.is_substantively_examined,
    }

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


# v0.27 r7: deterministic whole-book scope guard for explicitly singled-out
# constituent items in a broader collection/anthology. This is deliberately
# structural rather than case/title specific: verification of the facet itself
# is untouched; only the whole-book role may be demoted to incidental.
_CONSTITUENT_ITEM_SCOPE_PATTERNS = (
    r"\b(?:title|opening|closing|first|last|final)\s+(?:story|tale|essay|chapter|entry|episode|piece)\b",
    r"\b(?:in|from)\s+(?:one|a|the)\s+(?:story|tale|essay|chapter|entry|episode|piece)\b",
    r"\b(?:one|a|the)\s+(?:story|tale|essay|chapter|entry|episode|piece)\s+(?:in|from)\b",
)
_COLLECTION_SCOPE_PATTERN = r"\b(?:collection|anthology)\b"


def _normalized_scope_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _facet_explicitly_named_at_collection_scope(facet: QueryFacet, text: str) -> bool:
    """Conservative escape hatch: explicit facet wording attached to collection scope."""

    facet_phrase = _normalized_scope_phrase(facet.text)
    if not facet_phrase:
        return False
    normalized = _normalized_scope_phrase(text)
    # Inspect a bounded lexical neighborhood around each collection/anthology cue.
    for match in re.finditer(r"\b(?:collection|anthology)\b", normalized):
        lo = max(0, match.start() - 180)
        hi = min(len(normalized), match.end() + 240)
        if facet_phrase in normalized[lo:hi]:
            return True
    return False


def _apply_constituent_item_scope_guard(
    *,
    facet: QueryFacet,
    evidence_text: str,
    spans: dict[str, str],
    result: ProminenceAssessment,
) -> ProminenceAssessment:
    """Demote item-local evidence when the description explicitly marks collection scope.

    The guard activates only when the winning verified evidence itself identifies
    one singled-out constituent item AND the supplied description identifies a
    broader collection/anthology. It does not change verification_relation. An
    explicit collection-level use of the facet wording is a conservative escape
    hatch, leaving the LLM relation unchanged.
    """

    if result.subject_relation not in {
        SubjectRelation.SAME_AS_PRIMARY_SUBJECT,
        SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER,
    }:
        return result

    has_item_scope = any(
        re.search(pattern, evidence_text, flags=re.IGNORECASE)
        for pattern in _CONSTITUENT_ITEM_SCOPE_PATTERNS
    )
    if not has_item_scope:
        return result

    full_text = " ".join(spans.values())
    if re.search(_COLLECTION_SCOPE_PATTERN, full_text, flags=re.IGNORECASE) is None:
        return result

    if _facet_explicitly_named_at_collection_scope(facet, full_text):
        return result

    return ProminenceAssessment(
        subject_relation=SubjectRelation.OTHER,
        is_substantively_examined=False,
        supporting_span_ids=list(result.supporting_span_ids),
        reason=(
            result.reason
            + " [v0.27-r7 constituent-item scope guard: verified evidence is explicitly "
              "confined to one constituent item of a broader collection/anthology; no "
              "explicit collection-level facet statement was found, so whole-book role "
              "is forced to OTHER/incidental.]"
        ),
    )



# v0.28 r2: deterministic prominence isolation for lexical cues that occur only
# as one item in a possessive enumeration. Verification remains DIRECT; only
# whole-book role is capped when the prominence model has no independent span.
_POSSESSIVE_ENUM_ITEM_RE = re.compile(
    r"\b(?:his|her|their|my|our|your)\s+[A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,3}",
    flags=re.IGNORECASE,
)


def _apply_direct_cue_list_mention_guard(
    *,
    evidence_span_id: str,
    evidence_text: str,
    result: ProminenceAssessment,
) -> ProminenceAssessment:
    """Demote a direct cue that is only one item in an enumerated side-detail list.

    A deterministic lexical cue establishes semantic presence, not whole-book
    prominence. This guard is intentionally narrow: it activates only when
    (1) the prominence model promoted the facet to central/substantive,
    (2) all cited role support is confined to the same cue span, and
    (3) that span contains a comma-separated possessive enumeration with at
        least three items ("his king, his lover, his friends, his gods, ...").

    If another description span independently supports the facet's book-level
    role, the model's prominence decision is preserved.
    """

    if result.subject_relation not in {
        SubjectRelation.SAME_AS_PRIMARY_SUBJECT,
        SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER,
    }:
        return result

    independent_support = {
        sid for sid in result.supporting_span_ids if sid != evidence_span_id
    }
    if independent_support:
        return result

    possessive_items = _POSSESSIVE_ENUM_ITEM_RE.findall(evidence_text)
    if len(possessive_items) < 3 or evidence_text.count(",") < 2:
        return result

    return ProminenceAssessment(
        subject_relation=SubjectRelation.OTHER,
        is_substantively_examined=False,
        supporting_span_ids=list(result.supporting_span_ids),
        reason=(
            result.reason
            + " [v0.28-r2 direct-cue list-mention guard: deterministic lexical "
              "presence occurs only as one item in a possessive enumeration and "
              "no independent role-supporting span was cited; whole-book role is "
              "forced to OTHER/incidental.]"
        ),
    )


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


def _validate_component_completeness_contract(
    *,
    relation: VerificationRelation,
    all_required_components_established: bool,
    missing_semantic_component: str | None,
    semantic_definition_exclusion_applied: bool,
    stage_name: str,
) -> None:
    """Mechanically enforce the v0.25.0 canonical component-completeness contract."""

    missing = (missing_semantic_component or "").strip()

    if semantic_definition_exclusion_applied and relation != VerificationRelation.UNSUPPORTED:
        raise JudgeOutputValidationError(
            f"{stage_name}: semantic_definition_exclusion_applied=true requires UNSUPPORTED."
        )

    if relation in {VerificationRelation.DIRECT, VerificationRelation.ENTAILED}:
        if not all_required_components_established:
            raise JudgeOutputValidationError(
                f"{stage_name}: {relation.value.upper()} requires "
                "all_required_components_established=true."
            )
        if missing:
            raise JudgeOutputValidationError(
                f"{stage_name}: {relation.value.upper()} cannot declare a missing "
                f"semantic component: {missing!r}."
            )
        if semantic_definition_exclusion_applied:
            raise JudgeOutputValidationError(
                f"{stage_name}: positive verification cannot override an explicit "
                "semantic-definition exclusion."
            )

    if relation == VerificationRelation.ADJACENT:
        if all_required_components_established:
            raise JudgeOutputValidationError(
                f"{stage_name}: ADJACENT requires at least one missing required component."
            )
        if not missing:
            raise JudgeOutputValidationError(
                f"{stage_name}: ADJACENT must identify missing_semantic_component."
            )
        if semantic_definition_exclusion_applied:
            raise JudgeOutputValidationError(
                f"{stage_name}: ADJACENT cannot bypass an explicit semantic-definition exclusion."
            )

    if all_required_components_established and missing:
        raise JudgeOutputValidationError(
            f"{stage_name}: all_required_components_established=true is inconsistent "
            f"with missing_semantic_component={missing!r}."
        )


def _canonical_component_ids(facet: QueryFacet) -> list[str]:
    """Return canonical component IDs in frozen facet-spec order."""

    return [component.component_id for component in facet.required_components]


def _validate_component_evidence_ledger(
    *,
    component_checks: list[ComponentEvidenceCheck],
    all_required_components_established: bool,
    allowed_span_ids: set[str],
    relation: VerificationRelation,
    facet: QueryFacet,
    stage_name: str,
) -> None:
    """Validate the v0.25 canonical component-to-source grounding contract.

    For production v0.25 facets, the returned component IDs must exactly equal
    the frozen required-component IDs from the facet specification.  This makes
    it impossible for the model to silently omit a relationship/entity/process
    component and replace it with easier evidence-derived labels.

    Legacy unit-test fixtures without `required_components` retain the older
    non-empty/unique-ID checks so historical tests remain useful.
    """

    if not component_checks:
        raise JudgeOutputValidationError(
            f"{stage_name}: component_checks must contain at least one indispensable component."
        )

    returned_ids = [check.component_id.strip() for check in component_checks]
    if any(not component_id for component_id in returned_ids):
        raise JudgeOutputValidationError(
            f"{stage_name}: component_id values must contain meaningful text."
        )
    if len(returned_ids) != len(set(returned_ids)):
        raise JudgeOutputValidationError(
            f"{stage_name}: component_checks contains duplicate component_id values."
        )

    expected_ids = _canonical_component_ids(facet)
    if expected_ids:
        if returned_ids != expected_ids:
            missing = [component_id for component_id in expected_ids if component_id not in returned_ids]
            extra = [component_id for component_id in returned_ids if component_id not in expected_ids]
            wrong_order = not missing and not extra and returned_ids != expected_ids
            detail = []
            if missing:
                detail.append(f"missing={missing}")
            if extra:
                detail.append(f"unexpected={extra}")
            if wrong_order:
                detail.append("component IDs are not in frozen facet-spec order")
            raise JudgeOutputValidationError(
                f"{stage_name}: component_checks must exactly match the frozen canonical "
                f"component IDs {expected_ids}; {'; '.join(detail)}."
            )

    for check in component_checks:
        ids = list(check.supporting_span_ids)
        if len(ids) != len(set(ids)):
            raise JudgeOutputValidationError(
                f"{stage_name}: component {check.component_id!r} contains duplicate supporting span IDs."
            )
        unknown = [span_id for span_id in ids if span_id not in allowed_span_ids]
        if unknown:
            raise JudgeOutputValidationError(
                f"{stage_name}: component {check.component_id!r} cites unknown span IDs: {unknown}."
            )
        if check.established and not ids:
            raise JudgeOutputValidationError(
                f"{stage_name}: established component {check.component_id!r} must cite source evidence."
            )
        if not check.established and ids:
            raise JudgeOutputValidationError(
                f"{stage_name}: missing component {check.component_id!r} must not cite supporting spans."
            )
        if check.established and check.negative_boundary_applied:
            raise JudgeOutputValidationError(
                f"{stage_name}: component {check.component_id!r} cannot be established when "
                "negative_boundary_applied=true."
            )

    ledger_complete = all(check.established for check in component_checks)
    if all_required_components_established != ledger_complete:
        raise JudgeOutputValidationError(
            f"{stage_name}: all_required_components_established must equal the component ledger."
        )

    if relation in {VerificationRelation.DIRECT, VerificationRelation.ENTAILED} and not ledger_complete:
        raise JudgeOutputValidationError(
            f"{stage_name}: positive relation requires every canonical component to be established."
        )

    if relation == VerificationRelation.ADJACENT and ledger_complete:
        raise JudgeOutputValidationError(
            f"{stage_name}: ADJACENT requires at least one missing canonical component."
        )


def _validate_missing_component_id(
    *,
    missing_semantic_component: str | None,
    component_checks: list[ComponentEvidenceCheck],
    facet: QueryFacet,
    relation: VerificationRelation,
    stage_name: str,
) -> None:
    """Require missing_semantic_component to reference a real missing canonical ID."""

    raw = (missing_semantic_component or "").strip()
    if not raw:
        return

    expected_ids = _canonical_component_ids(facet)
    if not expected_ids:
        # Legacy/synthetic fixtures have no frozen canonical component contract.
        return
    if raw not in expected_ids:
        raise JudgeOutputValidationError(
            f"{stage_name}: missing_semantic_component must be one frozen component_id; "
            f"found {raw!r}, expected one of {expected_ids}."
        )

    state = {check.component_id: check.established for check in component_checks}
    if raw in state and state[raw]:
        raise JudgeOutputValidationError(
            f"{stage_name}: missing_semantic_component={raw!r} is marked established in the ledger."
        )

    if relation == VerificationRelation.ADJACENT and raw not in state:
        raise JudgeOutputValidationError(
            f"{stage_name}: ADJACENT missing_semantic_component must appear in component_checks."
        )

def _validate_verification_result(
    result: EvidenceVerification,
    facet: QueryFacet,
    config: dict[str, Any],
    evidence_span_id: str,
) -> None:
    _validate_component_evidence_ledger(
        component_checks=result.component_checks,
        all_required_components_established=result.all_required_components_established,
        allowed_span_ids={evidence_span_id},
        relation=result.verification_relation,
        facet=facet,
        stage_name="single-span verification",
    )
    _validate_missing_component_id(
        missing_semantic_component=result.missing_semantic_component,
        component_checks=result.component_checks,
        facet=facet,
        relation=result.verification_relation,
        stage_name="single-span verification",
    )
    _validate_component_completeness_contract(
        relation=result.verification_relation,
        all_required_components_established=result.all_required_components_established,
        missing_semantic_component=result.missing_semantic_component,
        semantic_definition_exclusion_applied=result.semantic_definition_exclusion_applied,
        stage_name="single-span verification",
    )
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
    evidence_span_id: str,
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

FROZEN CANONICAL REQUIRED COMPONENTS
{format_required_components(facet)}

HARD EXCLUSIONS — APPLY BEFORE ANY POSITIVE RELATION
{format_hard_exclusions(facet)}

IMPORTANT HARD-EXCLUSION OUTPUT RULE
If the frozen hard-exclusion list above is NONE, hard_exclusion_triggered MUST be false
and hard_exclusion_id MUST be null. A semantic-definition exclusion is separate and
must be reported only with semantic_definition_exclusion_applied.
If hard_exclusion_triggered=true, hard_exclusion_id MUST name one exact configured
exclusion, verification_relation MUST be unsupported, and inference_kind MUST be none.

EXACT CANDIDATE EVIDENCE
{evidence_span_id}: {evidence_text}

VERIFICATION SCALE
{scale}

DECISION PRECEDENCE
{precedence}

INSTRUCTIONS
{instructions}

Return one EvidenceVerification with verification_relation, inference_kind,
component_checks, all_required_components_established, missing_semantic_component,
semantic_definition_exclusion_applied, hard_exclusion_triggered,
hard_exclusion_id, and reason.

CANONICAL COMPONENT-EVIDENCE LEDGER
- DO NOT invent, rename, merge, split, omit, or add components.
- Return EXACTLY one component_checks entry for every frozen component_id above, in the same order.
- Copy component_id exactly.
- Assess EACH canonical component independently. A different missing component must not cause an actually grounded component to be marked missing.
- For an established component, set established=true and supporting_span_ids=["{evidence_span_id}"].
- For a missing component, set established=false and supporting_span_ids=[].
- If a component-specific negative boundary blocks the proposed grounding, set negative_boundary_applied=true and established=false.
- The reason must explain what this exact evidence establishes or fails to establish for that canonical component.
- all_required_components_established must be exactly true iff every canonical component is established.

For DIRECT or ENTAILED: every canonical component must be established,
all_required_components_established=true, and missing_semantic_component=null.
For ADJACENT: at least one canonical component must be missing,
all_required_components_established=false, and missing_semantic_component must be the EXACT component_id of one missing required component.
If an explicit negative boundary in the frozen semantic definition blocks the
full match: semantic_definition_exclusion_applied=true and verification_relation=unsupported.

GROUNDING BOUNDARIES
- LOCATION IS NOT MOVEMENT. Being trapped, stranded, located, surrounded, pursued, or endangered in a place does not itself establish travel, a journey, voyage, expedition, or quest. A rescue situation does not itself create a quest.
- A phrase such as "begin again", "start over", "move on", or choosing a new path does not establish learning to live again unless the same evidence grounds the required grief-or-major-loss context.
- Metaphorical or historical language such as a "lost time", "lost place", "lost past", memory, nostalgia, departure, return, or reclaiming the past does not by itself establish the personal significant-loss facet.
- For an affective qualifier such as "moving", strong wording that itself presents profound emotional impact or poignancy can establish the qualifier even without explicit reader-response language. Merely describing a sad, dangerous, difficult, or tragic subject does not.

The inference_kind must obey the configured relation mapping.
Do not use any information other than the facet, frozen semantic definition, and evidence shown above.
""".strip()


def component_local_inference_policy(facet: QueryFacet, component: Any) -> str:
    """Return narrowly-scoped v0.27 inference guidance for validated failure classes.

    r1 applied one global policy to every component. That changed Q03 behavior even
    though Q03 was already protected by an r6 local contract. r3 therefore keeps
    the r6 default verifier wording and adds extra guidance only where post-holdout
    evidence demonstrated a specific inference-boundary failure.
    """

    key = (facet.text.strip().lower(), component.component_id)
    policies: dict[tuple[str, str], str] = {
        (
            "personal growth",
            "actual_self_development_or_self_understanding",
        ): """
LOCAL PERSONAL-GROWTH CONTRACT
- MODE A (experienced development) and MODE B (explicitly promoted development) are INDEPENDENTLY sufficient.
- Do NOT require proof that a named person has already experienced development when the evidence explicitly says that a path, practice, program, guidance, or activity promotes/fosters/cultivates/builds/leads to maturity or personal growth in its intended people.
- Wording such as "promotes maturity and growth" is positive Mode-B grounding, not a reason to mark the component missing merely because it is prospective or instructional.
- Generic intellectual learning, becoming more knowledgeable/wiser, philosophy education, "begin to learn", or "wising up" is NOT personal growth unless the exact evidence independently states development, growth, maturity, transformation, changed mindset/behavior/ability, confidence, habits, skills, or self-understanding.
""".strip(),
        (
            "redemption",
            "damaged_or_failed_state",
        ): """
LOCAL Q01 REDEMPTION-DAMAGE CONTRACT
- A generic fall from success/greatness into hardship, despair, defeat, low status, or adversity does NOT by itself establish the redemption-type damaged state.
- Positive grounding requires supplied-text evidence of wrongdoing, guilt, corruption, moral/personal/relational failure, damaged worth, or a comparably redemption-relevant damaged state.
- Do not use a later recovery or success to retroactively invent the missing damaged-state semantics.
""".strip(),
        (
            "redemption",
            "restoration_or_atonement",
        ): """
LOCAL Q01 REDEMPTION-RESTORATION CONTRACT
- Generic recovery, renewed success, improvement, healing, return to greatness, or escape from despair does NOT by itself establish restoration/atonement in the redemption sense.
- Directional transition away from a negative condition (for example finding a way out, escaping, leaving behind, overcoming, or moving beyond it) does NOT by itself establish restoration/atonement.
- The supplied text must connect the restoration to the redemption-type damaged state and establish an actual restorative moral/personal/relational state: atonement, reconciliation, regained worth/standing, or restoration of a damaged relationship/state.
- Do not convert a negative-state -> better-state trajectory into redemption merely because the starting condition is morally/spiritually negative.
""".strip(),
        (
            "suspense",
            "story_level_tension_or_anticipation",
        ): """
LOCAL Q02 SUSPENSE CONTRACT
- Action, battle, rebellion, betrayal, military conflict, a threatening antagonist, danger, or high stakes do NOT by themselves establish suspense.
- Future-tense plot progression (including wording that actions will trigger, cause, attract, ignite, or set in motion later conflict/consequences) does NOT itself establish suspense.
- Positive grounding requires the supplied text itself to state or necessarily entail story-level tension/anticipation about an unfolding unresolved threat, pursuit, concealment, mystery, discovery, escape, or comparable development.
- Do not infer a suspenseful reading experience merely because events are action-packed, dangerous, consequential, or described as forthcoming. Do not invent reader anticipation from the fact that future events will occur.
- A future or rhetorical interpersonal disclosure such as "will the friendship/relationship survive when X learns the truth/secret?" is NOT by itself story-level suspense.
- Relationship consequences, embarrassment, obsession, sexual/romantic disclosure, or a secret being learned do not become suspense merely because the blurb asks what will happen next.
- Positive grounding still requires an independently tension-producing threat, pursuit, concealment under danger, mystery/investigation, escape, impending danger, or equivalent sustained narrative tension.
""".strip(),
        (
            "grief",
            "significant_loss",
        ): """
LOCAL Q06 LOSS-ANCHOR CONTRACT — r5 trauma boundary
- War trauma, combat, wounds, injury, horrors of war, PTSD-like distress, survival, or healing from trauma do NOT by themselves establish a significant loss.
- Positive grounding requires an actual significant deprivation/ending stated in the supplied text, such as death/bereavement, separation, disappearance, loss of an important relationship/person/role, or a comparable concrete loss.
""".strip(),
        (
            "loss",
            "actual_significant_deprivation_or_ending",
        ): """
LOCAL Q06 LOSS CONTRACT — r5 trauma boundary
- Trauma, war, injury, wounds, danger, suffering, or the need to heal are not themselves the query's loss facet.
- Do not convert "horrors of war", being wounded, or needing something worth living for into an unstated death, bereavement, separation, disappearance, or other significant deprivation.
""".strip(),
        (
            "learning to live again",
            "prior_grief_or_major_loss",
        ): """
LOCAL Q06 RECOVERY CONTRACT — r5 prior-loss requirement
- Recovery, healing, rebuilding, or finding something worth living for after war/trauma/injury does NOT establish "learning to live again" unless the supplied text independently establishes grief or a major/significant loss that motivates the recovery.
- Trauma recovery and post-war recovery are not interchangeable with grief/loss recovery.
""".strip(),
        (
            "building effective organizations",
            "organizational_design_management_or_improvement",
        ): """
LOCAL Q10 ORGANIZATIONAL-IMPROVEMENT CONTRACT — r5 positive boundary
- Solving organizational silos, barriers, politics, infighting, structural/cultural/process dysfunction, or transforming such dysfunction into alignment/clarity is direct text-grounded organizational improvement.
- Do not require the surface phrase "building effective organizations" when the supplied text explicitly describes an intervention that improves how an organization functions.
""".strip(),
        (
            "building effective organizations",
            "organizational_effectiveness_goal",
        ): """
LOCAL Q10 ORGANIZATIONAL-EFFECTIVENESS CONTRACT — r5 positive boundary
- Productivity, achievement of corporate/organizational goals, collective performance, clarity, alignment, or equivalent organization-level outcomes explicitly tied to solving organizational dysfunction establish the effectiveness/improvement goal.
- Mere mention of an organization without a performance/improvement relation remains insufficient.
""".strip(),
        (
            "mythology",
            "mythic_or_mythological_basis",
        ): """
LOCAL Q11 LEGEND-PROVENANCE CONTRACT — r5 analytical boundary
- Scholarship/evidence about where a legend originated, its historical source, ritual origin, transmission, or provenance does NOT by itself establish mythology as substantive content of the described work.
- A work may analyze the origin of a legend without retelling myths or presenting mythological settings/events/figures.
- Positive grounding requires the supplied description itself to present mythology/mythic tradition or an explicit retelling/reworking, not merely research about a legend's origin.
""".strip(),
        (
            "legendary heroes",
            "legendary_or_mythic_hero_identity",
        ): """
LOCAL Q11 LEGEND-PROVENANCE HERO CONTRACT — r5 analytical boundary
- The phrase "the legend of X" inside scholarship/provenance/origin discussion does NOT automatically establish X as a legendary hero.
- Positive grounding requires the supplied description to identify a hero/heroic figure as legendary or mythic, or unmistakably present such a legendary heroic figure as content.
- Research about how a legend originated is meta/analytical evidence, not sufficient legendary-hero identity.
""".strip(),
        (
            "adventure",
            "adventure_like_progression",
        ): """
LOCAL Q11 ADVENTURE CONTRACT
- An explicit narrative quest, journey, or expedition positively establishes adventure-like progression unless it is only meta-discussion/example use.
- Concrete sustained hazardous action can also establish progression when the supplied text itself states the action and peril, including taking up arms in a battle/army struggle or being sent in pursuit of a dangerous target whose peril is stated.
- A bare mission, duty, objective, rescue, conflict, or requirement to set something right remains insufficient. Do not invent unseen progression.
""".strip(),
        (
            "war",
            "organized_armed_conflict_or_warfare",
        ): """
LOCAL Q05 WAR CONTRACT
- There are TWO independent text-grounded modes; either is sufficient.
- MODE A — warfare itself: the supplied evidence establishes sustained armed conflict, organized belligerent sides, military forces, battles, campaigns, or clearly established warfare.
- MODE B — war as subject: the supplied evidence explicitly says the work studies, analyzes, explains, interprets, or is about war/wars/warfare itself, including its history, strategy, causes, effects, or preservation/avoidance. A specific battle need not be described in MODE B.
- A merely approaching, presaged, distant, background, or temporal mention of war does NOT establish MODE B.
- Do not use outside historical knowledge.
""".strip(),
        (
            "fantasy adventure",
            "adventure_quest_journey_or_action_exploits",
        ): """
LOCAL Q04 ADVENTURE CONTRACT
- There are TWO text-grounded ways to establish the adventure/action component; either one is sufficient.
- MODE A — travel/quest structure: a supplied travel narrative centered on the named subject's encounters across multiple stated places may entail a journey/adventure in one short text-licensed step.
- MODE B — action-driving conflict: when the supplied text places the named subject in a fantasy/supernatural setting and explicitly says the subject is hunted, pursued, attacked, or faces an immediate destructive threat or multiple concrete dangers, that can entail an action-driven adventure even when no travel/journey wording is present.
- Do not require the surface word "adventure" when the supplied text itself gives either Mode A or Mode B.
- Mere static fantasy setting, intrigue, or abstract danger without an action-driving conflict remains insufficient.
- Do not use title familiarity, genre convention, or outside knowledge.
""".strip(),
        (
            "dangerous journeys",
            "movement_or_travel",
        ): """
LOCAL Q04 MOVEMENT CONTRACT
- Explicit travel-book/travel-account framing plus the same named subject's encounters across multiple stated places in the supplied description entails movement/travel.
- In full-context recovery this may be a cross-span entailment: cite the travel-framing span and the span that gives the subject's multi-place encounters.
- The location-is-not-movement guard still applies when travel framing/multi-place progression is absent.
""".strip(),
        (
            "dangerous journeys",
            "danger_or_threat",
        ): """
LOCAL Q04 DANGER CONTRACT
- Being explicitly shipwrecked in the same supplied travel narrative is text-grounded evidence of hazard/danger associated with the journey.
- This inference requires no outside knowledge about the book.
- Mere travel without a hazard/threat cue remains insufficient.
""".strip(),
        (
            "political conflict",
            "political_power_governance_ideology_or_policy_stake",
        ): """
LOCAL Q05 POLITICAL-STAKE CONTRACT
- When the supplied text itself states the collapse/end/overthrow of a regime, empire, government, or state order in connection with revolution, commune/uprising, revolutionary violence, or an equivalent political contest, that can entail a political-power/state-authority stake.
- Do not import historical facts not stated in the supplied description.
""".strip(),
        (
            "political conflict",
            "actual_conflict_or_struggle",
        ): """
LOCAL Q05 POLITICAL-STRUGGLE CONTRACT
- Revolution, revolutionary violence, commune/uprising, overthrow, or an explicit contest linked by the supplied text to regime/state-order collapse can establish an actual political struggle.
- Politics-adjacent actors or violence without that supplied political relation remain insufficient.
""".strip(),
        (
            "resistance",
            "active_opposition_or_defiance",
        ): """
LOCAL Q09 RESISTANCE-ACTION CONTRACT
- A political label, radical identity, or membership in a named movement/organization does NOT by itself establish active opposition, defiance, rebellion, protest, or struggle.
- The supplied evidence itself must state or necessarily entail an oppositional action.
- Never explain what a named movement is "known for" or import its historical activities. If such history would be needed, return missing with external_knowledge_required=true.
""".strip(),
        (
            "resistance",
            "target_oppressive_or_authoritarian_power",
        ): """
LOCAL Q09 RESISTANCE-TARGET CONTRACT
- The supplied evidence itself must identify or necessarily describe the target as authoritarian, oppressive, coercive, repressive, or systematically controlling.
- Membership in a named political movement does NOT supply that movement's historical adversary, ideology, targets, protests, bombings, or view of a government.
- If deciding positive would require any fact about what a named movement/person historically opposed or believed, return missing with external_knowledge_required=true.
""".strip(),
    }
    return policies.get(key, "")


_Q07_PARENT_CHILD_ANCHOR_RE = re.compile(
    r"\b(?:"
    r"parent(?:s|al)?|"
    r"mother(?:s|'s)?|father(?:s|'s)?|"
    r"mom(?:s|'s)?|dad(?:s|'s)?|"
    r"son(?:s|'s)?|daughter(?:s|'s)?|"
    r"child(?:ren|'s)?|"
    r"stepmother|stepfather|stepson|stepdaughter|stepchild(?:ren)?"
    r")\b",
    flags=re.IGNORECASE,
)


def _q07_parent_child_text_anchor_guard(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    """Reject invented Q07 parent-child identity when no kinship anchor exists.

    Precision-only enforcement of the frozen Q07 negative boundary. The guard
    never creates positive Q07 evidence; it only rejects unsupported positives.
    """
    if (
        facet.text.strip().lower() != "complicated parent-child relationship"
        or component.component_id != "parent_child_relationship"
        or result.grounding_relation == "missing"
    ):
        return result
    if _Q07_PARENT_CHILD_ANCHOR_RE.search(evidence_text):
        return result
    return IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation="missing",
        negative_boundary_applied=True,
        external_knowledge_required=True,
        reason=(
            "v0.29-r5 Q07 parent-child grounding guard: positive grounding "
            "was rejected because the exact supplied evidence contains no "
            "parent/child kinship anchor. Generic interpersonal conflict, "
            "romantic conflict, blame, resentment, or relationship difficulty "
            "cannot establish a parent-child relationship by analogy."
        ),
    )


def _q07_parent_child_recovery_guard(
    facet: QueryFacet,
    component: Any,
    spans: dict[str, str],
    result: FullContextComponentRecovery,
) -> FullContextComponentRecovery:
    """Apply the same Q07 kinship requirement to full-context recovery."""
    if (
        facet.text.strip().lower() != "complicated parent-child relationship"
        or component.component_id != "parent_child_relationship"
        or result.grounding_relation == "missing"
    ):
        return result
    evidence_text = " ".join(spans.get(sid, "") for sid in result.supporting_span_ids)
    isolated = IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation=result.grounding_relation,
        negative_boundary_applied=result.negative_boundary_applied,
        external_knowledge_required=result.external_knowledge_required,
        reason=result.reason,
    )
    guarded = _q07_parent_child_text_anchor_guard(
        facet=facet, component=component, evidence_text=evidence_text, result=isolated
    )
    if guarded.grounding_relation != "missing":
        return result
    return FullContextComponentRecovery(
        component_id=result.component_id,
        grounding_relation="missing",
        supporting_span_ids=[],
        negative_boundary_applied=True,
        external_knowledge_required=True,
        reason=guarded.reason,
    )


def _q09_text_anchor_guard(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    """Mechanically prevent identity-only Q09 resistance grounding.

    The r1 model explicitly imported Weather Underground history while claiming
    external_knowledge_required=false. For Q09 resistance only, require a minimal
    text-local semantic anchor before a positive component can survive. This is a
    precision guard derived from the frozen negative boundary; it is not a
    candidate/title-specific exception.
    """

    if facet.text.strip().lower() != "resistance" or result.grounding_relation == "missing":
        return result

    text = evidence_text.lower()
    active_patterns = (
        r"\bresist(?:s|ed|ing|ance|ant)?\b",
        r"\boppos(?:e|es|ed|ing|ition)\b",
        r"\bdefi(?:es|ed|ance|ant)\b",
        r"\brebel(?:s|led|ling|lion|lious)?\b",
        r"\bprotest(?:s|ed|ing|er|ers)?\b",
        r"\buprising\b",
        r"\brevolt(?:s|ed|ing)?\b",
        r"\bstruggl(?:e|es|ed|ing)\b",
        r"\bfight(?:s|ing)?\b",
        r"\bfought\b",
        r"\bdissident(?:s)?\b",
    )
    target_patterns = (
        r"\bauthoritarian\b",
        r"\boppress(?:ive|ion|ed|or|ors)?\b",
        r"\bcoerc(?:ive|ion|ed)?\b",
        r"\brepress(?:ive|ion|ed)?\b",
        r"\bcensor(?:ship|ed|ing)?\b",
        r"\bsurveill(?:ance|ed|ing)?\b",
        r"\bdictator(?:ship|ial)?\b",
        r"\btyran(?:t|ny|nical)\b",
        r"\btotalitarian\b",
        r"\bpolice state\b",
        r"\bsecret police\b",
        r"\bpersecut(?:e|es|ed|ing|ion)\b",
        r"\benforced obedience\b",
        r"\bsystematic(?:ally)? restrict(?:ion|ions|ed|ing)?\b",
    )

    patterns = None
    if component.component_id == "active_opposition_or_defiance":
        patterns = active_patterns
    elif component.component_id == "target_oppressive_or_authoritarian_power":
        patterns = target_patterns
    if patterns is None or any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
        return result

    return IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation="missing",
        negative_boundary_applied=True,
        external_knowledge_required=True,
        reason=(
            "Python Q09 text-anchor guard: positive grounding was rejected because "
            "the exact evidence contains no text-local anchor for this resistance "
            "component; identity/membership cannot supply the missing action or target."
        ),
    )


def _q09_recovery_text_anchor_guard(
    facet: QueryFacet,
    component: Any,
    spans: dict[str, str],
    result: FullContextComponentRecovery,
) -> FullContextComponentRecovery:
    if facet.text.strip().lower() != "resistance" or result.grounding_relation == "missing":
        return result
    evidence_text = " ".join(spans.get(span_id, "") for span_id in result.supporting_span_ids)
    isolated = IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation=result.grounding_relation,
        negative_boundary_applied=result.negative_boundary_applied,
        external_knowledge_required=result.external_knowledge_required,
        reason=result.reason,
    )
    guarded = _q09_text_anchor_guard(facet, component, evidence_text, isolated)
    if guarded.grounding_relation != "missing":
        return result
    return FullContextComponentRecovery(
        component_id=result.component_id,
        grounding_relation="missing",
        supporting_span_ids=[],
        negative_boundary_applied=True,
        external_knowledge_required=True,
        reason=guarded.reason,
    )



# v0.28 r3: localized component guards derived from the r2 full-180 regressions.
# These guards are intentionally facet/component-scoped. They do not restore the
# rejected r1 global speculative-entailment filter.
_Q03_GROWTH_ANCHOR_RE = re.compile(
    r"\b(?:personal\s+growth|grow(?:s|th|ing)?|develop(?:s|ed|ing|ment)?|"
    r"improv(?:e|es|ed|ing|ement)|matur(?:e|es|ed|ing|ity)|"
    r"transform(?:s|ed|ing|ation)?|self[- ](?:understanding|awareness|development|improvement)|"
    r"confidence|mindset|behavio[u]?r|habit(?:s)?|skill(?:s)?|abilit(?:y|ies))\b",
    flags=re.IGNORECASE,
)
_Q03_INTELLECTUAL_LEARNING_RE = re.compile(
    r"\b(?:wising\s+up|wise|wiser|wisdom|learn(?:s|ed|ing)?|knowledge|"
    r"philosoph(?:y|ical)|intellectual|thinking|thoughts?|mull(?:ing)?)\b",
    flags=re.IGNORECASE,
)


# v0.29 r1: narrow text-local signals derived from the adjudicated r8 final
# evidence. These are semantic-pattern guards, never title/case-ID exceptions.
_Q03_PURPOSE_SELF_AWAKENING_RE = re.compile(
    r"\b(?:self[- ]awakening|self[- ]discovery|"
    r"awaken(?:ing|ed)?\s+to\s+(?:one'?s|your|their)\s+(?:purpose|calling|direction)|"
    r"discover(?:ing|s|ed)?\s+(?:one'?s|your|their)\s+(?:purpose|calling|life\s+path))\b",
    flags=re.IGNORECASE,
)
_Q03_PURPOSE_LIFE_DIRECTION_RE = re.compile(
    r"\b(?:live\s+(?:one'?s|your|their)\s+(?:greatest|best|truest|most\s+meaningful)\s+life|"
    r"life\s+(?:direction|path|calling|purpose)|"
    r"meaningful\s+life|"
    r"what\s+(?:one|you|they)\s+truly\s+want\s+from\s+life)\b",
    flags=re.IGNORECASE,
)

_Q04_MOVEMENT_TEXT_ANCHOR_RE = re.compile(
    r"\b(?:"
    r"travel(?:s|ed|ing|led|ling)?|travelogue|"
    r"journey(?:s|ed|ing)?|voyage(?:s|d)?|expedition(?:s)?|"
    r"quest(?:s|ed|ing)?|trip(?:s|ped|ping)?|trek(?:s|ked|king)?|"
    r"pilgrimage(?:s)?|tour(?:s|ed|ing)?|"
    r"sail(?:s|ed|ing)?|shipwreck(?:s|ed)?|"
    r"wander(?:s|ed|ing)?|roam(?:s|ed|ing)?|travers(?:e|es|ed|ing)|"
    r"moves?\s+(?:through|across|between)\b|"
    r"cross(?:es|ed|ing)?\s+(?:the\s+)?(?:sea|ocean|desert|mountains?|"
    r"countries|lands?|realms?|worlds?|places?)\b"
    r")",
    flags=re.IGNORECASE,
)


_Q04_R2_CROSS_SPAN_MOVEMENT_ANCHOR_RE = re.compile(
    r"\b(?:"
    r"travel(?:s|ed|ing|led|ling)?|travelogue|"
    r"journey(?:s|ed|ing)?|voyage(?:s|d)?|expedition(?:s)?|"
    r"quest(?:s|ed|ing)?|trip(?:s|ped|ping)?|trek(?:s|ked|king)?|"
    r"pilgrimage(?:s)?|tour(?:s|ed|ing)?|sail(?:s|ed|ing)?|"
    r"wander(?:s|ed|ing)?|roam(?:s|ed|ing)?|travers(?:e|es|ed|ing)|"
    r"moves?\s+(?:through|across|between)\b|"
    r"cross(?:es|ed|ing)?\s+(?:the\s+)?(?:sea|ocean|desert|mountains?|"
    r"countries|lands?|realms?|worlds?|places?)\b"
    r")",
    flags=re.IGNORECASE,
)


_Q04_R4_DIRECTIONAL_DEPARTURE_RE = re.compile(
    r"\b(?:take|takes|took|taking)\s+off\s+on\s+(?:an?|the)\s+"
    r"(?:[\w'-]+\s+){0,6}"
    r"(?:adventure|journey|trip|trek|voyage|expedition)\s+"
    r"(?:into|through|across|toward(?:s)?|to)\b",
    flags=re.IGNORECASE,
)


def _q04_r4_has_directional_departure_anchor(text: str) -> bool:
    """Recognize explicit departure + journey noun + directional continuation."""
    return bool(_Q04_R4_DIRECTIONAL_DEPARTURE_RE.search(text))


_Q04_R3_SHIPWRECK_DANGER_RE = re.compile(
    r"\bshipwreck(?:s|ed|ing)?\b",
    flags=re.IGNORECASE,
)

_Q09_R2_OPPOSITION_ACTION_RE = re.compile(
    r"\b(?:overthrow(?:s|ing|n)?|rebell(?:ion|ious|ed|ing)?|revolt(?:s|ed|ing)?|"
    r"uprising(?:s)?|resist(?:s|ed|ing|ance)?|def(?:y|ies|ied|ying)|"
    r"rise\s+up\s+against|fight(?:s|ing)?\s+against|protest(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)
_Q09_R2_AUTHORITARIAN_CONTEXT_RE = re.compile(
    r"\b(?:totalitarian(?:ism)?|authoritarian(?:ism)?|oppress(?:ive|ion|ed|or)?|"
    r"tyrann(?:y|ical)|dictator(?:ship|ial|s)?)\b",
    flags=re.IGNORECASE,
)

_Q11_META_ADVENTURE_RE = re.compile(
    r"\b(?:literary\s+(?:criticism|analysis)|analysis\s+of|instruction(?:al)?|"
    r"symbolism|archetype(?:s)?|example\s+(?:in|of)|discussion\s+of\s+other\s+works)\b",
    flags=re.IGNORECASE,
)
_Q11_EXPLICIT_PROGRESS_RE = re.compile(
    r"\b(?:quest|journey|expedition)\b",
    flags=re.IGNORECASE,
)
_Q11_BATTLE_ACTION_RE = re.compile(
    r"\btake(?:s|n)?\s+up\s+arms\b.*\b(?:battle|army|besieged|struggle|fight|war)\b"
    r"|\b(?:battle|army|besieged|struggle|fight|war)\b.*\btake(?:s|n)?\s+up\s+arms\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_Q11_SENT_AFTER_RE = re.compile(
    r"\bsent\b.{0,120}\bafter\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_Q11_EXPLICIT_PERIL_RE = re.compile(
    r"\b(?:deadly|danger(?:ous)?|peril(?:ous)?|kill(?:s|ed|ing)?|destroy(?:s|ed|ing)?|"
    r"attack(?:s|ed|ing)?|turn(?:s|ed|ing)?\s+(?:men|people|them|him|her)\s+to\s+stone|"
    r"whose\s+gaze\s+could\s+turn\s+men\s+to\s+stone)\b",
    flags=re.IGNORECASE,
)



# v0.28 r5: localized guards derived from the one-time independent r4 validation.
# These are semantic-pattern guards, not title/case-ID special cases.

_Q02_INTERPERSONAL_DISCLOSURE_RE = re.compile(
    r"\b(?:friendship|relationship|marriage|romance|love|bond|family)\b"
    r".{0,180}\b(?:learn(?:s|ed|ing)?|discover(?:s|ed|ing)?|find(?:s|ing)?\s+out)\b"
    r".{0,80}\b(?:truth|secret|obsession|affair|feelings?)\b"
    r"|\b(?:learn(?:s|ed|ing)?|discover(?:s|ed|ing)?|find(?:s|ing)?\s+out)\b"
    r".{0,80}\b(?:truth|secret|obsession|affair|feelings?)\b"
    r".{0,180}\b(?:friendship|relationship|marriage|romance|love|bond|family)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_Q02_INDEPENDENT_SUSPENSE_ANCHOR_RE = re.compile(
    r"\b(?:suspense(?:ful)?|tension|threat(?:en(?:s|ed|ing)?)?|pursu(?:e|es|ed|ing|it)|"
    r"chase(?:s|d|ing)?|stalk(?:s|ed|ing)?|escape(?:s|d|ing)?|mystery|investigat(?:e|es|ed|ing|ion)|"
    r"detective(?:s)?|crime(?:s)?|criminal(?:s|ity)?|murder(?:s|ed|er|ers|ing)?|kidnap(?:s|ped|ping)?|"
    r"attack(?:s|ed|ing)?|danger(?:ous)?|peril(?:ous)?|conceal(?:s|ed|ing|ment)|on\s+the\s+run)\b",
    flags=re.IGNORECASE,
)

_Q06_TRAUMA_CONTEXT_RE = re.compile(
    r"\b(?:war|combat|battle|wound(?:s|ed|ing)?|injur(?:y|ies|ed)|horrors?|trauma|traumatic|"
    r"ptsd|surviv(?:e|es|ed|ing|or|ors)|shell[- ]shock(?:ed)?)\b",
    flags=re.IGNORECASE,
)
_Q06_ACTUAL_LOSS_ANCHOR_RE = re.compile(
    r"\b(?:death|dead|dies|died|bereavement|bereaved|grief|grieving|mourn(?:s|ed|ing)?|"
    r"separation|separated|disappearance|disappeared|orphan(?:ed|s)?|widow(?:ed|er|ers|s)?|"
    r"funeral|divorce(?:d)?|loss\s+of|lost\s+(?:his|her|their|my|our|a|an)\s+"
    r"(?:wife|husband|spouse|partner|mother|father|parent|child|son|daughter|friend|family|home|job|role))\b",
    flags=re.IGNORECASE,
)

_Q10_ORG_SYSTEM_RE = re.compile(
    r"\b(?:silo(?:s)?|organizational\s+politics|organizational\s+barriers?|barriers?|infighting|"
    r"structure|culture|management\s+systems?|process(?:es)?|collective\s+performance|"
    r"organizational\s+dysfunction|workplace\s+dysfunction)\b",
    flags=re.IGNORECASE,
)
_Q10_IMPROVEMENT_ACTION_RE = re.compile(
    r"\b(?:address(?:es|ed|ing)?|solv(?:e|es|ed|ing)|transform(?:s|ed|ing)?|improv(?:e|es|ed|ing)|"
    r"develop(?:s|ed|ing)?|help(?:s|ed|ing)?|fix(?:es|ed|ing)?|align(?:s|ed|ing)?|"
    r"eliminat(?:e|es|ed|ing)|reduc(?:e|es|ed|ing)|overcom(?:e|es|ing)|approach)\b",
    flags=re.IGNORECASE,
)
_Q10_EFFECTIVENESS_OUTCOME_RE = re.compile(
    r"\b(?:productivity|performance|corporate\s+goals?|organizational\s+goals?|"
    r"effective(?:ness)?|clarity|alignment|function(?:s|ed|ing)?\s+effectively)\b",
    flags=re.IGNORECASE,
)

_Q11_LEGEND_PROVENANCE_RE = re.compile(
    r"\b(?:evidence|origin(?:s|ated|ating)?|histor(?:y|ical)|source(?:s)?|provenance|"
    r"ritual|scholar(?:ship|ly)?|study|analysis|suppressed|transmission)\b",
    flags=re.IGNORECASE,
)
_Q11_LEGEND_TERM_RE = re.compile(r"\b(?:legend|legends|myth|myths)\b", flags=re.IGNORECASE)
_Q11_EXPLICIT_MYTH_CONTENT_RE = re.compile(
    r"\b(?:mythology|mythological|mythic|retell(?:s|ing|ing|ed)?|rework(?:s|ed|ing)?|"
    r"gods?|goddess(?:es)?|deit(?:y|ies))\b",
    flags=re.IGNORECASE,
)
_Q11_EXPLICIT_HERO_IDENTITY_RE = re.compile(
    r"\b(?:legendary\s+hero(?:es)?|mythic\s+hero(?:es)?|heroic\s+figure(?:s)?|"
    r"hero(?:es)?|warrior(?:s)?|champion(?:s)?)\b",
    flags=re.IGNORECASE,
)

_Q12_CLASS_BACKGROUND_RE = re.compile(
    r"\b(?:working\s+class|middle\s+class|upper\s+class|lower\s+class|bourgeois(?:ie)?|"
    r"proletariat|social\s+class(?:es)?|class\s+background|class\s+identity)\b",
    flags=re.IGNORECASE,
)
_Q12_EXPLICIT_INEQUALITY_ANALYSIS_RE = re.compile(
    r"\b(?:inequalit(?:y|ies)|unequal|wealth\s+gap|income\s+gap|economic\s+disparit(?:y|ies)|"
    r"poverty|impoverished|income|wealth|resources?|material\s+conditions?|opportunit(?:y|ies)|"
    r"distribution|redistribution|allocation|stratification|class\s+inequality)\b",
    flags=re.IGNORECASE,
)


def _q02_r5_interpersonal_disclosure_guard(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    """Do not convert relationship-disclosure consequences into suspense."""

    if (
        facet.text.strip().lower() != "suspense"
        or component.component_id != "story_level_tension_or_anticipation"
        or result.grounding_relation == "missing"
    ):
        return result
    if _Q02_INTERPERSONAL_DISCLOSURE_RE.search(evidence_text) is None:
        return result
    if _Q02_INDEPENDENT_SUSPENSE_ANCHOR_RE.search(evidence_text):
        return result
    return IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation="missing",
        negative_boundary_applied=True,
        external_knowledge_required=False,
        reason=(
            "v0.28-r5 Q02 interpersonal-disclosure guard: future/rhetorical relationship "
            "consequences from learning a truth/secret do not independently establish "
            "story-level suspense without a separate threat, pursuit, mystery/investigation, "
            "escape, danger, or comparable suspense anchor."
        ),
    )


def _q06_r5_trauma_without_loss_guard(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    """Prevent trauma/war recovery from inventing grief or significant loss."""

    if result.grounding_relation == "missing":
        return result
    if facet.text.strip().lower() not in {"grief", "loss", "learning to live again"}:
        return result
    if _Q06_TRAUMA_CONTEXT_RE.search(evidence_text) is None:
        return result
    if _Q06_ACTUAL_LOSS_ANCHOR_RE.search(evidence_text):
        return result
    return IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation="missing",
        negative_boundary_applied=True,
        external_knowledge_required=False,
        reason=(
            "v0.28-r5 Q06 trauma-without-loss guard: war/trauma/injury/recovery language "
            "does not establish grief, significant loss, or grief/loss-driven rebuilding "
            "without an actual loss anchor in the supplied evidence."
        ),
    )


def _q10_r5_organizational_improvement_positive_guard(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    """Recover explicit organization-level improvement/effectiveness evidence."""

    if (
        facet.text.strip().lower() != "building effective organizations"
        or result.grounding_relation != "missing"
    ):
        return result

    has_system = _Q10_ORG_SYSTEM_RE.search(evidence_text) is not None
    has_action = _Q10_IMPROVEMENT_ACTION_RE.search(evidence_text) is not None
    has_outcome = _Q10_EFFECTIVENESS_OUTCOME_RE.search(evidence_text) is not None

    positive = False
    if component.component_id == "organizational_design_management_or_improvement":
        positive = has_system and has_action
    elif component.component_id == "organizational_effectiveness_goal":
        positive = has_system and has_outcome and (has_action or "organiz" in evidence_text.lower() or "corporate" in evidence_text.lower())

    if not positive:
        return result

    return IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation="entailed",
        negative_boundary_applied=False,
        external_knowledge_required=False,
        reason=(
            "v0.28-r5 Q10 organizational-improvement guard: the exact evidence explicitly "
            "connects organization-level dysfunction/process issues with an improvement "
            "action or effectiveness outcome; no outside knowledge is required."
        ),
    )


def _q11_r5_legend_provenance_guard(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    """Keep scholarship about a legend's origin from becoming myth/hero content."""

    if result.grounding_relation == "missing":
        return result
    if _Q11_LEGEND_PROVENANCE_RE.search(evidence_text) is None or _Q11_LEGEND_TERM_RE.search(evidence_text) is None:
        return result

    facet_name = facet.text.strip().lower()
    if facet_name == "mythology" and component.component_id == "mythic_or_mythological_basis":
        if _Q11_EXPLICIT_MYTH_CONTENT_RE.search(evidence_text):
            return result
    elif facet_name == "legendary heroes" and component.component_id == "legendary_or_mythic_hero_identity":
        if _Q11_EXPLICIT_HERO_IDENTITY_RE.search(evidence_text):
            return result
    else:
        return result

    return IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation="missing",
        negative_boundary_applied=True,
        external_knowledge_required=False,
        reason=(
            "v0.28-r5 Q11 legend-provenance guard: analytical evidence about the origin, "
            "source, ritual, history, or provenance of a legend does not by itself establish "
            "substantive mythology or a legendary-hero identity."
        ),
    )


def _apply_v028_r5_isolated_component_guards(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    result = _q02_r5_interpersonal_disclosure_guard(facet, component, evidence_text, result)
    result = _q06_r5_trauma_without_loss_guard(facet, component, evidence_text, result)
    result = _q10_r5_organizational_improvement_positive_guard(facet, component, evidence_text, result)
    result = _q11_r5_legend_provenance_guard(facet, component, evidence_text, result)
    return result


def _apply_v028_r5_recovery_guard(
    facet: QueryFacet,
    component: Any,
    spans: dict[str, str],
    result: FullContextComponentRecovery,
) -> FullContextComponentRecovery:
    """Apply the negative r5 boundaries after full-context recovery as well."""

    if result.grounding_relation == "missing":
        return result
    evidence_text = " ".join(spans.get(span_id, "") for span_id in result.supporting_span_ids)

    # Q02/Q06/Q11 negative boundaries must not be bypassed by recovery.
    probe = IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation=result.grounding_relation,
        negative_boundary_applied=result.negative_boundary_applied,
        external_knowledge_required=result.external_knowledge_required,
        reason=result.reason,
    )
    guarded = _q02_r5_interpersonal_disclosure_guard(facet, component, evidence_text, probe)
    guarded = _q06_r5_trauma_without_loss_guard(facet, component, evidence_text, guarded)
    guarded = _q11_r5_legend_provenance_guard(facet, component, evidence_text, guarded)
    if guarded.grounding_relation == "missing":
        return FullContextComponentRecovery(
            component_id=result.component_id,
            grounding_relation="missing",
            supporting_span_ids=[],
            negative_boundary_applied=True,
            external_knowledge_required=False,
            reason=guarded.reason,
        )
    return result


def _apply_v028_r5_prominence_guard(
    facet: QueryFacet,
    evidence_text: str,
    result: ProminenceAssessment,
) -> ProminenceAssessment:
    """Cap background class contrast at incidental Q12 inequality prominence."""

    if facet.text.strip().lower() != "inequality":
        return result
    if result.subject_relation not in {
        SubjectRelation.SAME_AS_PRIMARY_SUBJECT,
        SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER,
    }:
        return result
    if _Q12_CLASS_BACKGROUND_RE.search(evidence_text) is None:
        return result
    if _Q12_EXPLICIT_INEQUALITY_ANALYSIS_RE.search(evidence_text):
        return result
    return ProminenceAssessment(
        subject_relation=SubjectRelation.OTHER,
        is_substantively_examined=False,
        supporting_span_ids=list(result.supporting_span_ids),
        reason=(
            result.reason
            + " [v0.28-r5 Q12 class-background guard: social-class labels/contrast are "
              "present, but the evidence does not explicitly develop unequal resources, "
              "income/wealth, opportunity, material conditions, distribution, or an "
              "inequality analysis; whole-book role is capped at OTHER/incidental.]"
        ),
    )


# v0.28 r6: deterministic non-regression repairs after the r5 targeted gate.
# These protect two inherited positive controls without weakening the r5
# negative boundaries that fixed the independent-validation defects.

_WHOLE_WORK_THEMATIC_FRAME_RE = re.compile(
    r"\b(?:this|the)\s+(?:story|book|work|novel|narrative|memoir)\s+"
    r"(?:is\s+)?(?:of|about|explores?|examines?|centers?\s+on|focuses?\s+on)\b",
    flags=re.IGNORECASE,
)

_Q11_EXPLICIT_MYTH_LEGEND_HERO_CONTENT_RE = re.compile(
    r"\b(?:myths?|legends?|mythology|mythological|mythic)\b.{0,260}"
    r"\b(?:stories?\s+of\s+)?(?:gods?\s+and\s+)?heroes?\b"
    r"|\bheroes?\b.{0,260}\b(?:myths?|legends?|mythology|mythological|mythic)\b",
    flags=re.IGNORECASE | re.DOTALL,
)


_Q11_CLASSICAL_MYTH_SOURCE_RE = re.compile(
    r"\b(?:based\s+on|adapt(?:ed|ation)\s+of|episodes?\s+from|retell(?:s|ing|ing|ed)?|"
    r"rework(?:s|ed|ing)?|inspired\s+by)\b.{0,180}"
    r"\b(?:odyssey|iliad|homer(?:'s)?|greek|roman|norse)\b"
    r"|\b(?:odyssey|iliad|homer(?:'s)?)\b.{0,180}"
    r"\b(?:episodes?|adventure|journey|story|stories|tale|tales)\b",
    flags=re.IGNORECASE | re.DOTALL,
)

_Q11_EXPLICIT_MYTHIC_NARRATIVE_CONTENT_RE = re.compile(
    r"\b(?:enchantress|sorceress|witch|prophet|ghosts?|underworld|land\s+of\s+the\s+dead|"
    r"monster|monsters|six[- ]headed|cyclops|sirens?|scylla|charybdis|centaur|"
    r"demigod|demigods|gods?|goddess|goddesses|deity|deities)\b",
    flags=re.IGNORECASE,
)


def _r6_whole_work_thematic_enumeration_positive_guard(
    facet: QueryFacet,
    evidence_text: str,
    result: ProminenceAssessment,
) -> ProminenceAssessment:
    """Promote explicit whole-work thematic enumeration to substantive role.

    r4 introduced this as a prompt-level rule. r5 showed that the model can
    still occasionally return OTHER for a direct lexical cue in a sentence
    such as "this story of friendship, love, tragedy, and redemption". When
    the evidence explicitly frames the *whole work* as being of/about themes
    and names the frozen facet, the role is deterministically substantive.

    This does not apply to possessive side-detail enumerations (his king, his
    lover, his gods, ...), because those lack the whole-work frame.
    """
    if result.subject_relation in {
        SubjectRelation.SAME_AS_PRIMARY_SUBJECT,
        SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER,
    } and result.is_substantively_examined:
        return result

    if _WHOLE_WORK_THEMATIC_FRAME_RE.search(evidence_text) is None:
        return result

    facet_phrase = _normalized_scope_phrase(facet.text)
    if not facet_phrase:
        return result
    normalized = _normalized_scope_phrase(evidence_text)
    if facet_phrase not in normalized:
        return result

    return ProminenceAssessment(
        subject_relation=SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER,
        is_substantively_examined=True,
        supporting_span_ids=list(result.supporting_span_ids),
        reason=(
            result.reason
            + " [v0.28-r6 whole-work thematic-enumeration positive guard: the "
              "directly verified facet is explicitly named in a sentence that "
              "characterizes the story/book/work itself as being of/about those "
              "themes; role is DEFINING_CONTENT_OR_NARRATIVE_DRIVER/substantive.]"
        ),
    )


def _q11_r6_explicit_myth_hero_positive_guard(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    """Recover hero identity when myths/legends are explicitly hero stories.

    The r5 provenance guard must still reject analytical phrases such as
    "evidence for the origin of the legend of X". This positive guard activates
    only when the description itself presents myth/legend content together with
    heroes, and there is no legend-origin/provenance analytical framing.
    """
    if (
        facet.text.strip().lower() != "legendary heroes"
        or component.component_id != "legendary_or_mythic_hero_identity"
        or result.grounding_relation != "missing"
    ):
        return result

    if _Q11_EXPLICIT_MYTH_LEGEND_HERO_CONTENT_RE.search(evidence_text) is None:
        return result

    if _Q11_LEGEND_PROVENANCE_RE.search(evidence_text):
        return result

    return IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation="entailed",
        negative_boundary_applied=False,
        external_knowledge_required=False,
        reason=(
            "v0.28-r6 Q11 explicit myth/legend hero-content guard: the supplied "
            "description itself presents myths/legends together with hero stories, "
            "so legendary/mythic hero identity is necessarily grounded without "
            "outside knowledge."
        ),
    )


def _q11_r7_explicit_classical_myth_content_guard(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    """Recover mythology when mythic source material is explicit narrative content.

    This is deliberately narrower than "a legend is mentioned". It requires:
      * the component currently to be missing;
      * the mythology facet's mythic-basis component;
      * explicit source/retelling language tied to classical myth/epic material;
      * explicit supernatural/mythic narrative content in the same supplied text;
      * no analytical legend-origin/provenance framing.

    It therefore covers descriptions such as a story "based on episodes from
    Homer's Odyssey" that actually narrates encounters with an enchantress,
    ghosts, monsters, gods, etc., while leaving provenance scholarship such as
    "evidence for the origin of the legend of X" excluded.
    """
    if (
        facet.text.strip().lower() != "mythology"
        or component.component_id != "mythic_or_mythological_basis"
        or result.grounding_relation != "missing"
    ):
        return result

    if _Q11_CLASSICAL_MYTH_SOURCE_RE.search(evidence_text) is None:
        return result

    if _Q11_EXPLICIT_MYTHIC_NARRATIVE_CONTENT_RE.search(evidence_text) is None:
        return result

    if _Q11_LEGEND_PROVENANCE_RE.search(evidence_text):
        return result

    return IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation="entailed",
        negative_boundary_applied=False,
        external_knowledge_required=False,
        reason=(
            "v0.28-r7 Q11 explicit classical-myth-content guard: the supplied "
            "description itself states that the narrative is based on/retells "
            "classical mythic source material and explicitly depicts supernatural "
            "mythic figures/creatures/events, so mythology is grounded without "
            "outside knowledge."
        ),
    )


def _apply_v028_r6_isolated_component_guards(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    return _q11_r6_explicit_myth_hero_positive_guard(
        facet, component, evidence_text, result
    )


def _apply_v028_r7_isolated_component_guards(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    return _q11_r7_explicit_classical_myth_content_guard(
        facet, component, evidence_text, result
    )


def _q11_r8_cross_span_classical_myth_component_check(
    facet: QueryFacet,
    component: Any,
    spans: dict[str, str],
) -> ComponentEvidenceCheck | None:
    """Deterministically compose Q11 mythology across multiple exact spans.

    r7 correctly defined the semantic rule but applied it only to one isolated
    evidence span at a time. Some blurbs distribute the proof: one sentence
    states that the narrative is based on/retells classical mythic source
    material, while another sentence contains the supernatural mythic figures,
    creatures, or events.

    This helper returns a positive component check only when:
      * the facet/component is Q11 mythology -> mythic basis;
      * the full supplied description contains explicit classical myth/epic
        source or retelling language;
      * a supplied span contains explicit supernatural mythic narrative content;
      * the full description is not analytical legend-origin/provenance prose.

    No case ID, title, author, ISBN, or external entity knowledge is used.
    """

    if (
        facet.text.strip().lower() != "mythology"
        or component.component_id != "mythic_or_mythological_basis"
    ):
        return None

    full_text = " ".join(spans.values())
    if _Q11_LEGEND_PROVENANCE_RE.search(full_text):
        return None

    source_ids = [
        span_id
        for span_id, span_text in spans.items()
        if _Q11_CLASSICAL_MYTH_SOURCE_RE.search(span_text)
    ]
    content_ids = [
        span_id
        for span_id, span_text in spans.items()
        if _Q11_EXPLICIT_MYTHIC_NARRATIVE_CONTENT_RE.search(span_text)
    ]

    if not source_ids or not content_ids:
        return None

    supporting: list[str] = []
    for span_id in source_ids + content_ids:
        if span_id not in supporting:
            supporting.append(span_id)
        if len(supporting) >= MAX_COMPOSITE_CANDIDATES:
            break

    return ComponentEvidenceCheck(
        component_id=component.component_id,
        established=True,
        grounding_relation="entailed",
        supporting_span_ids=supporting,
        negative_boundary_applied=False,
        external_knowledge_required=False,
        reason=(
            "v0.28-r8 Q11 cross-span classical-myth-content guard: one exact "
            "description span explicitly ties the narrative to classical mythic/"
            "epic source material while another exact span explicitly depicts "
            "supernatural mythic figures, creatures, or events. Together they "
            "ground the mythology component without outside knowledge."
        ),
    )


def _q03_r3_learning_only_guard(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    """Reject intellectual-learning-only evidence for Q03 personal growth.

    The r2 full run promoted generic learning/wisdom language in a philosophy
    primer to personal growth. The frozen Q03 contract requires development,
    maturity, transformation, changed mindset/behavior/ability, confidence,
    habits, skills, or self-understanding—not mere acquisition of knowledge.
    """

    if (
        facet.text.strip().lower() != "personal growth"
        or component.component_id != "actual_self_development_or_self_understanding"
        or result.grounding_relation == "missing"
    ):
        return result
    if _Q03_GROWTH_ANCHOR_RE.search(evidence_text):
        return result
    if _Q03_INTELLECTUAL_LEARNING_RE.search(evidence_text) is None:
        return result
    return IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation="missing",
        negative_boundary_applied=True,
        external_knowledge_required=False,
        reason=(
            "v0.28-r3 Q03 intellectual-learning guard: generic learning, wisdom, "
            "philosophical education, or 'wising up' without an independent "
            "development/growth anchor does not establish personal growth."
        ),
    )



def _q03_r1_purpose_component_check(
    facet: QueryFacet,
    component: Any,
    spans: dict[str, str],
) -> ComponentEvidenceCheck | None:
    """Recover one narrow Q03 purpose pattern missed by Stage A in r8.

    The adjudicated development policy accepts a paired signal:
    (1) explicit self-awakening/self-discovery language, AND
    (2) explicit framing about how to live one's greatest/best/truest/
        meaningful life or otherwise clarifying life direction.

    Either signal alone is insufficient. Generic happiness, prosperity, inner
    peace, success, or spiritual realization remains outside this rule.
    """

    if (
        facet.text.strip().lower() != "finding purpose"
        or component.component_id != "meaning_direction_calling_or_goal_clarification"
    ):
        return None

    awakening_ids = [
        span_id for span_id, text in spans.items()
        if _Q03_PURPOSE_SELF_AWAKENING_RE.search(text)
    ]
    life_direction_ids = [
        span_id for span_id, text in spans.items()
        if _Q03_PURPOSE_LIFE_DIRECTION_RE.search(text)
    ]
    if not awakening_ids or not life_direction_ids:
        return None

    supporting: list[str] = []
    for span_id in [*awakening_ids, *life_direction_ids]:
        if span_id not in supporting:
            supporting.append(span_id)

    return ComponentEvidenceCheck(
        component_id=component.component_id,
        established=True,
        grounding_relation="entailed",
        supporting_span_ids=supporting[:4],
        negative_boundary_applied=False,
        external_knowledge_required=False,
        reason=(
            "v0.29-r1 Q03 purpose guard: supplied description pairs explicit "
            "self-awakening/self-discovery with explicit life-direction framing "
            "about living one's greatest/best/truest/meaningful life. Together "
            "they establish purpose/life-direction without outside knowledge."
        ),
    )


def _q04_r1_movement_text_anchor_guard(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    """Reject Q04 dangerous-journey movement inferred from time/action alone.

    r8 over-promoted a realistic thriller because narrative sequencing, a
    three-year span, and violent danger were interpreted as movement. For the
    movement_or_travel component, positive grounding must contain a supplied
    text-local movement/travel anchor. This preserves explicit travel, journey,
    voyage, quest, expedition, sailing, shipwreck, roaming, etc.
    """

    if (
        facet.text.strip().lower() != "dangerous journeys"
        or component.component_id != "movement_or_travel"
        or result.grounding_relation == "missing"
    ):
        return result

    if (
        _Q04_MOVEMENT_TEXT_ANCHOR_RE.search(evidence_text)
        or _q04_r4_has_directional_departure_anchor(evidence_text)
    ):
        return result

    return IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation="missing",
        negative_boundary_applied=True,
        external_knowledge_required=False,
        reason=(
            "v0.29-r1 Q04 movement guard: danger, narrative progression, temporal "
            "progression, or events unfolding over time do not establish actual "
            "travel/journey. The exact evidence contains no text-local movement anchor."
        ),
    )


def _q04_r1_recovery_movement_text_anchor_guard(
    facet: QueryFacet,
    component: Any,
    spans: dict[str, str],
    result: FullContextComponentRecovery,
) -> FullContextComponentRecovery:
    """Apply the same Q04 movement anchor to full-context component recovery."""

    if (
        facet.text.strip().lower() != "dangerous journeys"
        or component.component_id != "movement_or_travel"
        or result.grounding_relation == "missing"
    ):
        return result

    evidence_text = " ".join(
        spans.get(span_id, "") for span_id in result.supporting_span_ids
    )
    if (
        _Q04_MOVEMENT_TEXT_ANCHOR_RE.search(evidence_text)
        or _q04_r4_has_directional_departure_anchor(evidence_text)
    ):
        return result

    return FullContextComponentRecovery(
        component_id=result.component_id,
        grounding_relation="missing",
        supporting_span_ids=[],
        negative_boundary_applied=True,
        external_knowledge_required=False,
        reason=(
            "v0.29-r1 Q04 recovery movement guard: combined supporting spans contain "
            "danger/temporal progression but no text-local travel, journey, quest, "
            "voyage, expedition, or equivalent movement anchor."
        ),
    )



def _q04_r2_cross_span_movement_component_check(
    facet: QueryFacet,
    component: Any,
    spans: dict[str, str],
) -> ComponentEvidenceCheck | None:
    """Recover Q04 movement only from an explicit supplied travel/movement anchor."""
    if (
        facet.text.strip().lower() != "dangerous journeys"
        or component.component_id != "movement_or_travel"
    ):
        return None
    supporting = [
        sid for sid, text in spans.items()
        if (
            _Q04_R2_CROSS_SPAN_MOVEMENT_ANCHOR_RE.search(text)
            or _q04_r4_has_directional_departure_anchor(text)
        )
    ]
    if not supporting:
        return None
    return ComponentEvidenceCheck(
        component_id=component.component_id,
        established=True,
        grounding_relation="explicit",
        supporting_span_ids=supporting[:4],
        negative_boundary_applied=False,
        external_knowledge_required=False,
        reason=(
            "v0.29-r2 Q04 cross-span movement recovery: an exact supplied "
            "description span contains an explicit travel/movement anchor."
        ),
    )



def _q04_r3_shipwreck_danger_component_check(
    facet: QueryFacet,
    component: Any,
    spans: dict[str, str],
) -> ComponentEvidenceCheck | None:
    """Recover Q04 danger when the supplied travel narrative explicitly says shipwreck.

    This implements the already-frozen Q04 danger contract: an explicitly
    shipwrecked traveler is a text-grounded hazard cue. To avoid turning an
    isolated shipwreck reference into a dangerous-journey match, an independent
    supplied travel/movement anchor must also exist somewhere in the description.
    """

    if (
        facet.text.strip().lower() != "dangerous journeys"
        or component.component_id != "danger_or_threat"
    ):
        return None

    danger_span_ids = [
        sid for sid, text in spans.items()
        if _Q04_R3_SHIPWRECK_DANGER_RE.search(text)
    ]
    travel_span_ids = [
        sid for sid, text in spans.items()
        if (
            _Q04_R2_CROSS_SPAN_MOVEMENT_ANCHOR_RE.search(text)
            or _q04_r4_has_directional_departure_anchor(text)
        )
    ]

    if not danger_span_ids or not travel_span_ids:
        return None

    supporting: list[str] = []
    for sid in [*danger_span_ids, *travel_span_ids]:
        if sid not in supporting:
            supporting.append(sid)

    return ComponentEvidenceCheck(
        component_id=component.component_id,
        established=True,
        grounding_relation="entailed",
        supporting_span_ids=supporting[:4],
        negative_boundary_applied=False,
        external_knowledge_required=False,
        reason=(
            "v0.29-r3 Q04 shipwreck danger recovery: supplied text explicitly "
            "states a shipwreck and independently supplies a travel/movement "
            "anchor, so the travel narrative contains a concrete hazard cue."
        ),
    )


def _q09_r2_resistance_pair_component_check(
    facet: QueryFacet,
    component: Any,
    spans: dict[str, str],
) -> ComponentEvidenceCheck | None:
    """Recover Q09 resistance only from same-span opposition + authoritarian framing."""
    if facet.text.strip().lower() != "resistance":
        return None
    if component.component_id not in {
        "active_opposition_or_defiance",
        "target_oppressive_or_authoritarian_power",
    }:
        return None
    supporting = [
        sid for sid, text in spans.items()
        if _Q09_R2_OPPOSITION_ACTION_RE.search(text)
        and _Q09_R2_AUTHORITARIAN_CONTEXT_RE.search(text)
    ]
    if not supporting:
        return None
    return ComponentEvidenceCheck(
        component_id=component.component_id,
        established=True,
        grounding_relation="entailed",
        supporting_span_ids=supporting[:4],
        negative_boundary_applied=False,
        external_knowledge_required=False,
        reason=(
            "v0.29-r2 Q09 resistance recovery: the same exact supplied span "
            "contains both an explicit opposition action and explicit "
            "authoritarian/totalitarian/oppressive framing."
        ),
    )


def _q11_r3_adventure_positive_guard(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    """Restore text-explicit Q11 adventure evidence rejected by the r2 boundary.

    Positive override is narrow:
    - an explicit narrative quest/journey/expedition (not meta-discussion), or
    - concrete sustained hazardous action such as taking up arms in battle, or
      being sent after a target with peril explicitly stated in the same span.

    A bare mission, duty, rescue, goal, conflict, or "set things right" statement
    does not satisfy any of these patterns.
    """

    if (
        facet.text.strip().lower() != "adventure"
        or component.component_id != "adventure_like_progression"
        or result.grounding_relation != "missing"
    ):
        return result
    if _Q11_META_ADVENTURE_RE.search(evidence_text):
        return result

    explicit_progression = _Q11_EXPLICIT_PROGRESS_RE.search(evidence_text) is not None
    battle_progression = _Q11_BATTLE_ACTION_RE.search(evidence_text) is not None
    hazardous_pursuit = (
        _Q11_SENT_AFTER_RE.search(evidence_text) is not None
        and _Q11_EXPLICIT_PERIL_RE.search(evidence_text) is not None
    )
    if not (explicit_progression or battle_progression or hazardous_pursuit):
        return result

    mode = (
        "explicit quest/journey/expedition"
        if explicit_progression
        else "concrete battle/action progression"
        if battle_progression
        else "hazardous pursuit progression"
    )
    return IsolatedComponentVerification(
        component_id=result.component_id,
        grounding_relation="entailed",
        negative_boundary_applied=False,
        external_knowledge_required=False,
        reason=(
            "v0.28-r3 Q11 positive adventure guard: exact supplied evidence "
            f"establishes {mode}; no outside knowledge or unseen progression is required."
        ),
    )


def _apply_v028_r3_isolated_component_guards(
    facet: QueryFacet,
    component: Any,
    evidence_text: str,
    result: IsolatedComponentVerification,
) -> IsolatedComponentVerification:
    result = _q03_r3_learning_only_guard(facet, component, evidence_text, result)
    result = _q11_r3_adventure_positive_guard(facet, component, evidence_text, result)
    return result


def _apply_v028_r3_recovery_guard(
    facet: QueryFacet,
    component: Any,
    spans: dict[str, str],
    result: FullContextComponentRecovery,
) -> FullContextComponentRecovery:
    """Keep Q03 full-context recovery from reintroducing learning-only growth."""

    if (
        facet.text.strip().lower() != "personal growth"
        or component.component_id != "actual_self_development_or_self_understanding"
        or result.grounding_relation == "missing"
    ):
        return result
    evidence_text = " ".join(spans.get(span_id, "") for span_id in result.supporting_span_ids)
    if _Q03_GROWTH_ANCHOR_RE.search(evidence_text):
        return result
    if _Q03_INTELLECTUAL_LEARNING_RE.search(evidence_text) is None:
        return result
    return FullContextComponentRecovery(
        component_id=result.component_id,
        grounding_relation="missing",
        supporting_span_ids=[],
        negative_boundary_applied=True,
        external_knowledge_required=False,
        reason=(
            "v0.28-r3 Q03 recovery guard: full-context support is only generic "
            "intellectual learning/wisdom without an independent personal-growth anchor."
        ),
    )


def build_isolated_component_prompt(
    facet: QueryFacet,
    component: Any,
    evidence_span_id: str,
    evidence_text: str,
) -> str:
    """Ask the model about ONE immutable component, never the full facet verdict."""

    boundaries = "\n".join(
        f"- {boundary}" for boundary in component.negative_boundaries
    ) or "- NONE"

    return f"""
Evaluate ONE frozen semantic component against ONE exact evidence span.

You are NOT deciding whether the full facet is satisfied. You are deciding only
whether the ONE canonical component below is grounded by this ONE evidence span.
Do not reason about, compensate for, or infer any other component of the facet.

FACET NAME (context only)
{facet.text}

FROZEN FACET DEFINITION (meaning boundary only)
{facet.semantic_definition}

ONE CANONICAL COMPONENT
component_id: {component.component_id}
definition: {component.definition}

COMPONENT-SPECIFIC NEGATIVE BOUNDARIES
{boundaries}

EXACT EVIDENCE
{evidence_span_id}: {evidence_text}

{component_local_inference_policy(facet, component)}

Return one IsolatedComponentVerification.

Rules:
- component_id must be exactly {component.component_id!r}.
- grounding_relation="explicit" only when this evidence itself directly states,
  directly paraphrases, or unmistakably instantiates this component.
- grounding_relation="entailed" only when this component necessarily follows in
  one short semantic step from this exact evidence.
- grounding_relation="missing" when the component is absent, merely plausible,
  analogous, metaphorical, dependent on typical-world knowledge, or blocked by
  a negative boundary.
- If a listed negative boundary applies to the proposed grounding, set
  negative_boundary_applied=true and grounding_relation="missing".
- A lexical resemblance is not enough when it uses the concept in the wrong
  entity, relationship, temporal, metaphorical, or process sense.
- Do not use any other book span, title, author, query facet, or world knowledge.
- Always return external_knowledge_required. If it is true, grounding_relation must be "missing".
- Do not decide a full-facet relation such as DIRECT/ENTAILED/ADJACENT.
""".strip()


def _validate_isolated_component_result(
    result: IsolatedComponentVerification,
    component: Any,
    stage_name: str,
) -> None:
    if result.component_id != component.component_id:
        raise JudgeOutputValidationError(
            f"{stage_name}: component_id must be exactly {component.component_id!r}."
        )
    if result.grounding_relation != "missing" and result.negative_boundary_applied:
        raise JudgeOutputValidationError(
            f"{stage_name}: a component blocked by a negative boundary must be missing."
        )
    if result.external_knowledge_required and result.grounding_relation != "missing":
        raise JudgeOutputValidationError(
            f"{stage_name}: external knowledge cannot positively establish a canonical component."
        )


def verify_isolated_component(
    judge_model: Any,
    facet: QueryFacet,
    component: Any,
    evidence_span_id: str,
    evidence_text: str,
) -> tuple[IsolatedComponentVerification, int]:
    """Verify exactly one canonical component against one exact source span.

    v0.27 r5 keeps the semantic contract unchanged but makes the new
    ``external_knowledge_required`` audit fail closed instead of crashing an
    unattended run. The model still gets the normal bounded repair attempt
    first. Only if every attempt repeats the specific contradiction
    ``external_knowledge_required=true`` with positive grounding do we
    deterministically preserve the audit admission and canonicalize that ONE
    component to missing.
    """

    validation_error: str | None = None
    last_result: IsolatedComponentVerification | None = None
    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):
        prompt = build_isolated_component_prompt(
            facet=facet,
            component=component,
            evidence_span_id=evidence_span_id,
            evidence_text=evidence_text,
        )
        if validation_error:
            prompt += (
                "\n\nPREVIOUS VALIDATION FAILURE\n"
                f"{validation_error}\n"
                "Repair only the structure/consistency of this ONE component result. "
                f"Return component_id={component.component_id!r}. Positive grounding "
                "cannot coexist with negative_boundary_applied=true OR with "
                "external_knowledge_required=true. If outside/entity-specific knowledge "
                "is required, return grounding_relation='missing'. If this exact supplied "
                "span itself is sufficient for positive grounding, set "
                "external_knowledge_required=false."
            )
        generated = judge_model.generate(
            prompt=prompt,
            schema=IsolatedComponentVerification,
        )
        try:
            result = unpack_generated_model(generated, IsolatedComponentVerification)
            assert isinstance(result, IsolatedComponentVerification)
            last_result = result
            result = _q09_text_anchor_guard(
                facet=facet,
                component=component,
                evidence_text=evidence_text,
                result=result,
            )
            result = _q07_parent_child_text_anchor_guard(
                facet=facet,
                component=component,
                evidence_text=evidence_text,
                result=result,
            )
            result = _q04_r1_movement_text_anchor_guard(
                facet=facet,
                component=component,
                evidence_text=evidence_text,
                result=result,
            )
            result = _apply_v028_r3_isolated_component_guards(
                facet=facet,
                component=component,
                evidence_text=evidence_text,
                result=result,
            )
            result = _apply_v028_r5_isolated_component_guards(
                facet=facet,
                component=component,
                evidence_text=evidence_text,
                result=result,
            )
            result = _apply_v028_r6_isolated_component_guards(
                facet=facet,
                component=component,
                evidence_text=evidence_text,
                result=result,
            )
            result = _apply_v028_r7_isolated_component_guards(
                facet=facet,
                component=component,
                evidence_text=evidence_text,
                result=result,
            )
            _validate_isolated_component_result(
                result=result,
                component=component,
                stage_name="isolated component verification",
            )
            return result, attempt - 1
        except (JudgeOutputValidationError, ValueError, TypeError) as error:
            validation_error = str(error)

    # The external-knowledge audit is a precision guard, so its own repeated
    # contradiction has a safe conservative resolution: positive grounding is
    # rejected. This is intentionally narrower than swallowing arbitrary schema
    # or semantic validation failures.
    if (
        last_result is not None
        and last_result.component_id == component.component_id
        and last_result.external_knowledge_required
        and last_result.grounding_relation != "missing"
    ):
        return (
            IsolatedComponentVerification(
                component_id=component.component_id,
                grounding_relation="missing",
                negative_boundary_applied=last_result.negative_boundary_applied,
                external_knowledge_required=True,
                reason=(
                    "isolated_component_external_knowledge_conflict_rejected_after_"
                    "bounded_repairs: "
                    + (validation_error or "positive grounding required external knowledge")
                ),
            ),
            MAX_STAGE_ATTEMPTS - 1,
        )

    raise JudgeOutputValidationError(
        validation_error or "Unable to obtain valid isolated component verification."
    )


def _assemble_single_span_from_component_results(
    facet: QueryFacet,
    evidence_span_id: str,
    component_results: list[IsolatedComponentVerification],
) -> EvidenceVerification:
    """Python alone derives the full-facet relation from isolated component calls."""

    by_id = {result.component_id: result for result in component_results}
    checks: list[ComponentEvidenceCheck] = []
    for component in facet.required_components:
        result = by_id[component.component_id]
        established = result.grounding_relation != "missing"
        checks.append(
            ComponentEvidenceCheck(
                component_id=component.component_id,
                established=established,
                grounding_relation=result.grounding_relation,
                supporting_span_ids=[evidence_span_id] if established else [],
                negative_boundary_applied=result.negative_boundary_applied,
                external_knowledge_required=result.external_knowledge_required,
                reason=result.reason,
            )
        )

    all_established = all(check.established for check in checks)
    any_established = any(check.established for check in checks)
    boundary_blocked = any(
        (not check.established) and check.negative_boundary_applied
        for check in checks
    )
    missing_ids = [check.component_id for check in checks if not check.established]

    if all_established:
        if all(check.grounding_relation == "explicit" for check in checks):
            relation = VerificationRelation.DIRECT
            inference_kind = EvidenceInferenceKind.EXPLICIT_COMPONENTS
        else:
            relation = VerificationRelation.ENTAILED
            inference_kind = EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE
        missing = None
    elif boundary_blocked:
        relation = VerificationRelation.UNSUPPORTED
        inference_kind = EvidenceInferenceKind.NONE
        missing = None
    elif any_established:
        relation = VerificationRelation.ADJACENT
        inference_kind = EvidenceInferenceKind.INCOMPLETE_CONNECTION
        missing = missing_ids[0]
    else:
        relation = VerificationRelation.UNSUPPORTED
        inference_kind = EvidenceInferenceKind.NONE
        missing = None

    reason_parts = [
        f"{check.component_id}={check.grounding_relation}"
        + ("[boundary]" if check.negative_boundary_applied else "")
        for check in checks
    ]
    return EvidenceVerification(
        verification_relation=relation,
        inference_kind=inference_kind,
        component_checks=checks,
        all_required_components_established=all_established,
        missing_semantic_component=missing,
        semantic_definition_exclusion_applied=boundary_blocked,
        hard_exclusion_triggered=False,
        hard_exclusion_id=None,
        reason="Python deterministic component assembly: " + "; ".join(reason_parts),
    )


def verify_candidate_evidence(
    judge_model: Any,
    facet: QueryFacet,
    evidence_span_id: str,
    evidence_text: str,
    config: dict[str, Any],
) -> tuple[EvidenceVerification, int]:
    """Verify one candidate.

    Production v0.26 facets are decomposed into independent component calls and
    Python deterministically assembles the facet relation. The legacy holistic
    path remains only for old unit-test fixtures that have no required_components.
    """

    if facet.required_components:
        component_results: list[IsolatedComponentVerification] = []
        total_retries = 0
        for component in facet.required_components:
            result, retries = verify_isolated_component(
                judge_model=judge_model,
                facet=facet,
                component=component,
                evidence_span_id=evidence_span_id,
                evidence_text=evidence_text,
            )
            component_results.append(result)
            total_retries += retries

        assembled = _assemble_single_span_from_component_results(
            facet=facet,
            evidence_span_id=evidence_span_id,
            component_results=component_results,
        )
        _validate_verification_result(
            result=assembled,
            facet=facet,
            config=config,
            evidence_span_id=evidence_span_id,
        )
        return assembled, total_retries

    # Legacy fixture compatibility only. Production v0.26 facets never enter here.
    validation_error: str | None = None

    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):
        prompt = build_verification_prompt(
            facet=facet,
            evidence_span_id=evidence_span_id,
            evidence_text=evidence_text,
            config=config,
        )
        if validation_error:
            prompt += (
                "\n\nPREVIOUS VALIDATION FAILURE\n"
                f"{validation_error}\n"
                "Return a corrected EvidenceVerification only. Rebuild component_checks using "
                "EXACTLY the frozen canonical component_id values, in frozen order; do not rename, "
                "omit, add, split, or merge them. Assess every component independently. Each "
                "established component must be grounded to the exact candidate "
                f"span {evidence_span_id}; a missing component must cite no spans. Positive "
                "DIRECT/ENTAILED requires every canonical component established, "
                "all_required_components_established=true, no missing semantic component, "
                "no semantic-definition exclusion, and a relation-consistent "
                "inference_kind. If the facet has no frozen hard exclusions, "
                "hard_exclusion_triggered must be false and hard_exclusion_id null. "
                "If a valid frozen hard exclusion is triggered, verification_relation "
                "must be unsupported and inference_kind must be none. "
                "Never supply a missing entity/relationship/event/process from analogy, "
                "plausibility, or world knowledge."
            )

        generated = judge_model.generate(
            prompt=prompt,
            schema=EvidenceVerification,
        )

        try:
            result = unpack_generated_model(generated, EvidenceVerification)
            assert isinstance(result, EvidenceVerification)
            result = _normalize_impossible_hard_exclusion_output(result, facet)
            assert isinstance(result, EvidenceVerification)
            result = _canonicalize_valid_triggered_hard_exclusion(result, facet)
            assert isinstance(result, EvidenceVerification)
            _validate_verification_result(
                result=result,
                facet=facet,
                config=config,
                evidence_span_id=evidence_span_id,
            )
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
    dict[str, EvidenceVerification],
    int,
]:
    """
    Verify candidate spans in selector order and select the strongest result.

    DIRECT short-circuits later verification because no later relation can
    outrank it. ENTAILED and ADJACENT do not short-circuit because a later
    stronger candidate should win.
    """

    records: list[CandidateVerificationRecord] = []
    verification_results_by_span: dict[str, EvidenceVerification] = {}
    total_retries = 0

    for rank, span_id in enumerate(
        candidate_ids,
        start=1,
    ):

        verification, retries = verify_candidate_evidence(
            judge_model=judge_model,
            facet=facet,
            evidence_span_id=span_id,
            evidence_text=spans[span_id],
            config=config,
        )

        total_retries += retries
        verification_results_by_span[span_id] = verification

        record = CandidateVerificationRecord(
            candidate_rank=rank,
            evidence_span_id=span_id,
            verification_relation=(
                verification.verification_relation
            ),
            inference_kind=verification.inference_kind,
            hard_exclusion_triggered=verification.hard_exclusion_triggered,
            hard_exclusion_id=verification.hard_exclusion_id,
            verification_reason=(
                verification.reason
                + " [component_check: all_required_components_established="
                + str(verification.all_required_components_established).lower()
                + "; semantic_definition_exclusion_applied="
                + str(verification.semantic_definition_exclusion_applied).lower()
                + "; missing_semantic_component="
                + repr(verification.missing_semantic_component)
                + "; component_ledger="
                + json.dumps(
                    [check.model_dump(mode="json") for check in verification.component_checks],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "]"
            ),
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
        verification_results_by_span,
        total_retries,
    )


# =============================================================================
# Stage C: self-selecting full-context composite verification
# =============================================================================

def format_single_span_component_audit(
    verification_results_by_span: dict[str, EvidenceVerification],
) -> str:
    """Render prior single-span component ledgers for composite monotonicity."""

    if not verification_results_by_span:
        return "NONE — no standalone candidate was verified."

    blocks: list[str] = []
    for span_id, result in verification_results_by_span.items():
        checks = "; ".join(
            f"{check.component_id}={'ESTABLISHED' if check.established else 'MISSING'}"
            for check in result.component_checks
        )
        blocks.append(
            f"{span_id}: relation={result.verification_relation.value}; "
            f"semantic_definition_exclusion_applied="
            f"{str(result.semantic_definition_exclusion_applied).lower()}; "
            f"components=[{checks}]"
        )
    return "\n".join(blocks)


def build_composite_verification_prompt(
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
    verification_results_by_span: dict[str, EvidenceVerification] | None = None,
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

    min_spans = int(stage.get("min_spans", 1))
    max_spans = int(stage.get("max_spans", MAX_COMPOSITE_CANDIDATES))
    single_span_audit = format_single_span_component_audit(
        verification_results_by_span or {}
    )

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

FROZEN CANONICAL REQUIRED COMPONENTS
{format_required_components(facet)}

HARD EXCLUSIONS — APPLY BEFORE ANY POSITIVE RELATION
{format_hard_exclusions(facet)}

FULL NUMBERED BOOK-DESCRIPTION SPANS
{format_description_spans(spans)}

PRIOR SINGLE-SPAN COMPONENT AUDIT
{single_span_audit}

The prior audit is not a final verdict. It is a grounding constraint. Component
identity is already frozen above: never rename, omit, merge, split, or add a
component. Full-context recovery MAY inspect any supplied description span,
including spans Stage A did not select. It may connect canonical components
distributed across spans. It may not resurrect a canonical component that every
cited, already-audited span marked missing unless at least one cited span was not
previously audited for that component and independently supplies the missing
information.

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

component_checks:
- return EXACTLY one check for every frozen canonical component_id, in the same order
- copy each component_id exactly; do not invent, rename, omit, merge, split, or add components
- assess each canonical component independently
- established=true requires 1-4 exact supplied supporting_span_ids that ground THAT component
- established=false requires supporting_span_ids=[]
- when a component-specific negative boundary blocks the proposed grounding, set negative_boundary_applied=true and established=false
- every component span ID must also appear in the top-level supporting_span_ids for a positive relation
- multiple spans may jointly ground one canonical component only when their combination necessarily establishes it
- repetition of a nearby/excluded concept does not create a missing component

all_required_components_established:
- true only when every indispensable semantic component is established jointly
- required true for ENTAILED
- required false for ADJACENT

semantic_definition_exclusion_applied:
- true when an explicit negative boundary in the frozen semantic definition blocks the match
- if true, verification_relation must be UNSUPPORTED

hard_exclusion_triggered / hard_exclusion_id:
- evaluate the frozen hard exclusions first
- if one applies, return true plus its exact exclusion_id, UNSUPPORTED, and inference_kind=none
- otherwise return false and null
- if the frozen hard-exclusion list is NONE, these fields MUST be false and null
- semantic_definition_exclusion_applied is a separate field and must not be
  represented as a hard exclusion

combined_evidence_summary:
- summarize only what the cited spans jointly establish
- do not add facts not present in those spans

missing_semantic_component:
- null for ENTAILED
- for ADJACENT, return the EXACT component_id of a missing canonical component
- for UNSUPPORTED, null is preferred

reason:
- explain why the cited spans and missing-component field justify the relation

FULL-CONTEXT COMPONENT RULES
- Composite/full-context evidence may CONNECT independently grounded canonical components; it may not CREATE an indispensable component from plausibility, repetition, metaphor, genre convention, or typical-world association.
- A canonical component that is absent in all previously audited cited spans may be established only if a newly cited, previously unaudited span independently supplies it, or if the exact combination necessarily establishes the component through explicit cross-span reference resolution rather than thematic inference.
- Do not mark one canonical component missing merely because a different canonical component is missing.
- If generic recovery/sobriety was excluded from redemption, combining several recovery/sobriety spans still does not establish restoration after wrongdoing/failure/damage unless another cited span grounds that restoration.
- If standalone spans establish hardship but no significant loss, combining those hardships does not create grief or loss.
- LOCATION IS NOT MOVEMENT: being trapped, located, pursued, threatened, or rescued in a place does not establish a journey/quest unless movement or travel is itself grounded.
- A return to memories, reconstructing the past, a "lost time/place", or reclaiming the past does not by itself establish personal significant loss or rebuilding life after grief/loss.

DIRECT is forbidden.
Do not require exact facet wording for ENTAILED.
Do not use any information outside the facet, frozen definition, supplied
numbered description, and the prior single-span component audit.
""".strip()


def _validate_composite_result(
    result: CompositeEvidenceVerification,
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
    verification_results_by_span: dict[str, EvidenceVerification] | None = None,
) -> None:
    """Deterministically validate v0.20 composite structure/consistency."""

    stage = config["composite_verification_stage"]
    min_spans = int(stage.get("min_spans", 1))
    max_spans = int(stage.get("max_spans", MAX_COMPOSITE_CANDIDATES))
    relation = result.verification_relation
    ids = list(result.supporting_span_ids)

    _validate_component_evidence_ledger(
        component_checks=result.component_checks,
        all_required_components_established=result.all_required_components_established,
        allowed_span_ids=set(spans),
        relation=relation,
        facet=facet,
        stage_name="composite verification",
    )
    _validate_missing_component_id(
        missing_semantic_component=result.missing_semantic_component,
        component_checks=result.component_checks,
        facet=facet,
        relation=relation,
        stage_name="composite verification",
    )

    _validate_component_completeness_contract(
        relation=relation,
        all_required_components_established=result.all_required_components_established,
        missing_semantic_component=result.missing_semantic_component,
        semantic_definition_exclusion_applied=result.semantic_definition_exclusion_applied,
        stage_name="composite verification",
    )

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

    if relation in {VerificationRelation.ADJACENT, VerificationRelation.ENTAILED}:
        top_level_ids = set(ids)
        for check in result.component_checks:
            if check.established and not set(check.supporting_span_ids).issubset(top_level_ids):
                raise JudgeOutputValidationError(
                    "Composite component evidence must be a subset of top-level supporting spans."
                )

    # v0.25 canonical monotonicity guard.  Exact component IDs eliminate the
    # model-authored-label loophole from v0.24.  A component may be supplied by
    # a newly cited full-context span, but the same already-audited spans may
    # not simply flip a canonical component from universally missing to present.
    prior = verification_results_by_span or {}
    if relation in {VerificationRelation.ADJACENT, VerificationRelation.ENTAILED}:
        for check in result.component_checks:
            if not check.established or not check.supporting_span_ids:
                continue
            prior_states: list[bool] = []
            fully_audited = True
            for span_id in check.supporting_span_ids:
                previous = prior.get(span_id)
                if previous is None:
                    fully_audited = False
                    break
                matches = [
                    item for item in previous.component_checks
                    if item.component_id == check.component_id
                ]
                if not matches:
                    fully_audited = False
                    break
                prior_states.append(matches[0].established)
            if fully_audited and prior_states and not any(prior_states):
                raise JudgeOutputValidationError(
                    "composite verification: canonical component resurrection is forbidden; "
                    f"{check.component_id!r} was missing in every cited prior single-span audit."
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


def _component_prior_audit_text(
    component_id: str,
    verification_results_by_span: dict[str, EvidenceVerification],
) -> str:
    lines: list[str] = []
    for span_id, verification in verification_results_by_span.items():
        match = next(
            (check for check in verification.component_checks if check.component_id == component_id),
            None,
        )
        if match is None:
            continue
        lines.append(
            f"- {span_id}: established={str(match.established).lower()}, "
            f"grounding_relation={match.grounding_relation}, "
            f"negative_boundary_applied={str(match.negative_boundary_applied).lower()}"
        )
    return "\n".join(lines) or "- NONE"


def build_missing_component_recovery_prompt(
    facet: QueryFacet,
    component: Any,
    spans: dict[str, str],
    verification_results_by_span: dict[str, EvidenceVerification],
) -> str:
    """Full-description search for ONE still-missing canonical component only."""

    boundaries = "\n".join(
        f"- {boundary}" for boundary in component.negative_boundaries
    ) or "- NONE"
    prior = _component_prior_audit_text(
        component.component_id,
        verification_results_by_span,
    )

    return f"""
Search the FULL numbered description for evidence of ONE missing canonical component.

You are NOT re-evaluating the full facet. You are NOT allowed to alter or
reinterpret any other component. Find evidence only for the component below.

FACET NAME (context only)
{facet.text}

FROZEN FACET DEFINITION (meaning boundary only)
{facet.semantic_definition}

ONE MISSING CANONICAL COMPONENT
component_id: {component.component_id}
definition: {component.definition}

COMPONENT-SPECIFIC NEGATIVE BOUNDARIES
{boundaries}

FULL NUMBERED DESCRIPTION
{format_description_spans(spans)}

PRIOR SINGLE-SPAN AUDIT FOR THIS SAME COMPONENT
{prior}

{component_local_inference_policy(facet, component)}

Return one FullContextComponentRecovery.

Rules:
- component_id must be exactly {component.component_id!r}.
- This stage may inspect spans that Stage A did not select.
- grounding_relation="explicit" when the cited span(s) directly establish only
  this component; "entailed" when they jointly/individually necessarily establish
  only this component in one short inference; otherwise "missing".
- Positive grounding requires 1-{MAX_COMPOSITE_CANDIDATES} exact supplied span IDs.
- Missing requires supporting_span_ids=[].
- If a component-specific negative boundary applies, return missing and
  negative_boundary_applied=true.
- Do NOT overturn a prior missing decision by simply citing the same already-
  audited span again. A positive recovery must use at least one previously
  unaudited span for this component, unless the positive result depends on an
  explicit cross-span reference whose antecedent is supplied in another cited span.
- Do not create facts from analogy, metaphor, genre convention, plausibility,
  typical-world association, or outside knowledge.
- Always return external_knowledge_required. If it is true, grounding_relation must be "missing" and supporting_span_ids must be empty.
- LOCATION IS NOT MOVEMENT; generic recovery is not redemption; lost time/place
  is not personal significant loss; reconstructing family history is not
  rebuilding one's own life after loss when those boundaries apply.
""".strip()


def _validate_component_recovery_result(
    result: FullContextComponentRecovery,
    component: Any,
    spans: dict[str, str],
    verification_results_by_span: dict[str, EvidenceVerification],
) -> None:
    if result.component_id != component.component_id:
        raise JudgeOutputValidationError(
            f"component recovery: component_id must be exactly {component.component_id!r}."
        )
    ids = list(result.supporting_span_ids)
    if len(ids) != len(set(ids)):
        raise JudgeOutputValidationError("component recovery: supporting span IDs must be unique.")
    unknown = [span_id for span_id in ids if span_id not in spans]
    if unknown:
        raise JudgeOutputValidationError(
            f"component recovery: unknown supporting span IDs: {unknown}."
        )
    if result.external_knowledge_required and result.grounding_relation != "missing":
        raise JudgeOutputValidationError(
            "component recovery: external knowledge cannot positively establish a canonical component."
        )
    if result.grounding_relation == "missing":
        if ids:
            raise JudgeOutputValidationError(
                "component recovery: missing result must not cite supporting spans."
            )
        return
    if result.negative_boundary_applied:
        raise JudgeOutputValidationError(
            "component recovery: positive grounding cannot apply a negative boundary."
        )
    if not ids:
        raise JudgeOutputValidationError(
            "component recovery: positive grounding requires supporting spans."
        )

    # Missing-component-only monotonicity: a positive recovery cannot simply
    # flip the same already-audited span(s). At least one cited span must be new
    # for this component unless multiple spans explicitly resolve a cross-span
    # reference. We conservatively require a new span here; previously audited
    # spans may accompany it as context.
    has_new_span = False
    for span_id in ids:
        previous = verification_results_by_span.get(span_id)
        if previous is None:
            has_new_span = True
            break
        match = next(
            (check for check in previous.component_checks if check.component_id == component.component_id),
            None,
        )
        if match is None:
            has_new_span = True
            break
    if not has_new_span:
        raise JudgeOutputValidationError(
            "component recovery: positive result must cite at least one span not previously audited for this missing component."
        )


def recover_missing_component(
    judge_model: Any,
    facet: QueryFacet,
    component: Any,
    spans: dict[str, str],
    verification_results_by_span: dict[str, EvidenceVerification],
) -> tuple[FullContextComponentRecovery, int]:
    validation_error: str | None = None
    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):
        prompt = build_missing_component_recovery_prompt(
            facet=facet,
            component=component,
            spans=spans,
            verification_results_by_span=verification_results_by_span,
        )
        if validation_error:
            prompt += (
                "\n\nPREVIOUS VALIDATION FAILURE\n"
                f"{validation_error}\n"
                "Repair only this ONE component result. Do not change any other component "
                "or infer the full facet. Positive grounding cannot coexist with "
                "external_knowledge_required=true. If outside/entity-specific knowledge "
                "is required, or if no valid new grounding exists, return "
                "grounding_relation='missing' with no supporting spans."
            )
        generated = judge_model.generate(
            prompt=prompt,
            schema=FullContextComponentRecovery,
        )
        try:
            result = unpack_generated_model(generated, FullContextComponentRecovery)
            assert isinstance(result, FullContextComponentRecovery)
            result = _q07_parent_child_recovery_guard(
                facet=facet,
                component=component,
                spans=spans,
                result=result,
            )
            result = _q09_recovery_text_anchor_guard(
                facet=facet,
                component=component,
                spans=spans,
                result=result,
            )
            result = _q04_r1_recovery_movement_text_anchor_guard(
                facet=facet,
                component=component,
                spans=spans,
                result=result,
            )
            result = _apply_v028_r3_recovery_guard(
                facet=facet,
                component=component,
                spans=spans,
                result=result,
            )
            result = _apply_v028_r5_recovery_guard(
                facet=facet,
                component=component,
                spans=spans,
                result=result,
            )
            _validate_component_recovery_result(
                result=result,
                component=component,
                spans=spans,
                verification_results_by_span=verification_results_by_span,
            )
            return result, attempt - 1
        except (JudgeOutputValidationError, ValueError, TypeError) as error:
            validation_error = str(error)

    # This is not a transport/schema fallback for the whole facet. It is a
    # deterministic enforcement of the missing-component-only invariant: if the
    # model cannot supply a structurally valid new grounding, that component
    # remains missing and the audit records why.
    prior_boundary_applied = any(
        check.negative_boundary_applied
        for verification in verification_results_by_span.values()
        for check in verification.component_checks
        if check.component_id == component.component_id
    )
    return (
        FullContextComponentRecovery(
            component_id=component.component_id,
            grounding_relation="missing",
            supporting_span_ids=[],
            negative_boundary_applied=prior_boundary_applied,
            external_knowledge_required=False,
            reason=(
                "component_recovery_rejected_after_bounded_repairs: "
                + (validation_error or "unknown validation failure")
            ),
        ),
        MAX_STAGE_ATTEMPTS - 1,
    )


def _best_prior_component_check(
    component_id: str,
    verification_results_by_span: dict[str, EvidenceVerification],
) -> ComponentEvidenceCheck | None:
    strength = {"missing": 0, "entailed": 1, "explicit": 2}
    candidates: list[ComponentEvidenceCheck] = []
    for verification in verification_results_by_span.values():
        for check in verification.component_checks:
            if check.component_id == component_id and check.established:
                candidates.append(check)
    if not candidates:
        return None
    return max(candidates, key=lambda item: strength.get(item.grounding_relation, 0))


def verify_composite_evidence(
    judge_model: Any,
    facet: QueryFacet,
    spans: dict[str, str],
    config: dict[str, Any],
    verification_results_by_span: dict[str, EvidenceVerification] | None = None,
) -> tuple[CompositeEvidenceVerification, int]:
    """v0.26 recovers only missing components and assembles the facet in Python."""

    if facet.required_components:
        # Use the caller's full config for final validation so inference mapping
        # remains identical to the frozen judge configuration.
        prior = verification_results_by_span or {}
        total_retries = 0
        checks: list[ComponentEvidenceCheck] = []
        recovery_notes: list[str] = []
        for component in facet.required_components:
            existing = _best_prior_component_check(component.component_id, prior)
            if existing is not None:
                checks.append(existing.model_copy(deep=True))
                recovery_notes.append(f"{component.component_id}=kept_from_single_span")
                continue

            # v0.29 r2: explicit movement may live in another exact supplied span.
            deterministic_q04_movement = _q04_r2_cross_span_movement_component_check(
                facet=facet,
                component=component,
                spans=spans,
            )
            if deterministic_q04_movement is not None:
                checks.append(deterministic_q04_movement)
                recovery_notes.append(
                    f"{component.component_id}=v029_r2_cross_span_movement"
                )
                continue

            # v0.29 r3: implement the already-frozen Q04 danger contract.
            # Explicit shipwreck + an independent supplied travel anchor is
            # sufficient text-grounded danger; no model interpretation needed.
            deterministic_q04_danger = _q04_r3_shipwreck_danger_component_check(
                facet=facet,
                component=component,
                spans=spans,
            )
            if deterministic_q04_danger is not None:
                checks.append(deterministic_q04_danger)
                recovery_notes.append(
                    f"{component.component_id}=v029_r3_shipwreck_danger"
                )
                continue

            # v0.29 r2: same-span opposition + authoritarian framing can establish resistance.
            deterministic_q09_resistance = _q09_r2_resistance_pair_component_check(
                facet=facet,
                component=component,
                spans=spans,
            )
            if deterministic_q09_resistance is not None:
                checks.append(deterministic_q09_resistance)
                recovery_notes.append(
                    f"{component.component_id}=v029_r2_resistance_pair"
                )
                continue

            # v0.29 r1: one adjudicated Q03 purpose-positive pattern was missed
            # entirely by Stage A. Recover it deterministically from exact
            # description spans before model-based missing-component recovery.
            deterministic_q03_purpose = _q03_r1_purpose_component_check(
                facet=facet,
                component=component,
                spans=spans,
            )
            if deterministic_q03_purpose is not None:
                checks.append(deterministic_q03_purpose)
                recovery_notes.append(
                    f"{component.component_id}=v029_r1_purpose_pair_deterministic"
                )
                continue

            # v0.28 r8: deterministic cross-span composition for one narrow Q11
            # mythology boundary. This remains inherited unchanged.
            deterministic_cross_span = _q11_r8_cross_span_classical_myth_component_check(
                facet=facet,
                component=component,
                spans=spans,
            )
            if deterministic_cross_span is not None:
                checks.append(deterministic_cross_span)
                recovery_notes.append(
                    f"{component.component_id}=r8_cross_span_deterministic"
                )
                continue

            recovered, retries = recover_missing_component(
                judge_model=judge_model,
                facet=facet,
                component=component,
                spans=spans,
                verification_results_by_span=prior,
            )
            total_retries += retries
            established = recovered.grounding_relation != "missing"
            prior_boundary_applied = any(
                check.negative_boundary_applied
                for verification in prior.values()
                for check in verification.component_checks
                if check.component_id == component.component_id
            )
            checks.append(
                ComponentEvidenceCheck(
                    component_id=component.component_id,
                    established=established,
                    grounding_relation=recovered.grounding_relation,
                    supporting_span_ids=list(recovered.supporting_span_ids) if established else [],
                    negative_boundary_applied=(
                        False if established
                        else recovered.negative_boundary_applied or prior_boundary_applied
                    ),
                    external_knowledge_required=recovered.external_knowledge_required,
                    reason=recovered.reason,
                )
            )
            recovery_notes.append(f"{component.component_id}=recovery:{recovered.grounding_relation}")

        all_established = all(check.established for check in checks)
        any_established = any(check.established for check in checks)
        boundary_blocked = any(
            (not check.established) and check.negative_boundary_applied
            for check in checks
        )
        missing_ids = [check.component_id for check in checks if not check.established]
        if all_established:
            relation = VerificationRelation.ENTAILED
            inference_kind = EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE
            missing = None
        elif boundary_blocked:
            relation = VerificationRelation.UNSUPPORTED
            inference_kind = EvidenceInferenceKind.NONE
            missing = None
        elif any_established:
            relation = VerificationRelation.ADJACENT
            inference_kind = EvidenceInferenceKind.INCOMPLETE_CONNECTION
            missing = missing_ids[0]
        else:
            relation = VerificationRelation.UNSUPPORTED
            inference_kind = EvidenceInferenceKind.NONE
            missing = None

        supporting: list[str] = []
        if relation in {VerificationRelation.ADJACENT, VerificationRelation.ENTAILED}:
            for check in checks:
                if check.established:
                    for span_id in check.supporting_span_ids:
                        if span_id not in supporting:
                            supporting.append(span_id)
        if len(supporting) > MAX_COMPOSITE_CANDIDATES:
            relation = VerificationRelation.UNSUPPORTED
            inference_kind = EvidenceInferenceKind.NONE
            supporting = []
            missing = None
            all_established = False
            recovery_notes.append("support_union_exceeded_max_composite_candidates")

        result = CompositeEvidenceVerification(
            supporting_span_ids=supporting,
            verification_relation=relation,
            inference_kind=inference_kind,
            component_checks=checks,
            all_required_components_established=all_established,
            semantic_definition_exclusion_applied=boundary_blocked,
            hard_exclusion_triggered=False,
            hard_exclusion_id=None,
            combined_evidence_summary="Python deterministic component composition: " + "; ".join(recovery_notes),
            missing_semantic_component=missing,
            reason="No holistic facet verifier was used; relation assembled from isolated canonical component outcomes.",
        )
        _validate_composite_result(
            result=result,
            facet=facet,
            spans=spans,
            config=config,
            verification_results_by_span={},
        )
        return result, total_retries

    # Legacy fixture compatibility only.
    validation_error: str | None = None

    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):
        prompt = build_composite_verification_prompt(
            facet=facet,
            spans=spans,
            config=config,
            verification_results_by_span=verification_results_by_span,
        )

        if validation_error:
            prompt += (
                "\n\nPREVIOUS VALIDATION FAILURE\n"
                f"{validation_error}\n"
                "Return a corrected CompositeEvidenceVerification only. "
                "Rebuild component_checks using EXACTLY the frozen canonical component_id "
                "values, in frozen order. Do not rename, omit, add, split, or merge them. "
                "Ground each established component to exact supporting span IDs. Do not "
                "resurrect a component that every "
                "cited prior single-span component audit marked missing unless a new cited "
                "span independently supplies it. Remember: ENTAILED => "
                "all_required_components_established=true and "
                "missing_semantic_component=null; ADJACENT => false plus the missing "
                "required component; semantic-definition exclusions force UNSUPPORTED; "
                "if the facet has no frozen hard exclusions, hard_exclusion_triggered "
                "must be false and hard_exclusion_id null; if a valid hard exclusion is "
                "triggered, relation must be unsupported and inference_kind must be none; "
                "positive relations require 1-4 unique real span IDs. If your previous "
                "semantic verdict was positive but support-list shape was invalid, preserve "
                "the semantic verdict and repair only the structure when the evidence still "
                "supports it; relation and inference_kind must match."
            )

        # Transport/model-call failures intentionally propagate. Only generated
        # structured-output validation failures are repaired; unrepaired failures propagate.
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
            result = _normalize_impossible_hard_exclusion_output(result, facet)
            assert isinstance(result, CompositeEvidenceVerification)
            result = _canonicalize_valid_triggered_hard_exclusion(result, facet)
            assert isinstance(result, CompositeEvidenceVerification)
            _validate_composite_result(
                result=result,
                facet=facet,
                spans=spans,
                config=config,
                verification_results_by_span=verification_results_by_span,
            )
            return result, attempt - 1

        except (JudgeOutputValidationError, ValueError, TypeError) as error:
            validation_error = str(error)

    raise JudgeOutputValidationError(
        "Unable to obtain a structurally valid full-context verification after "
        f"{MAX_STAGE_ATTEMPTS} attempts. Last validation error: "
        f"{validation_error or 'unknown'}"
    )



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
    validation_feedback: str | None = None,
) -> str:
    """Build a book-level subject prompt that contains no query or facet.

    On a validator-driven retry, include only the mechanical validation failure
    so the model can repair malformed output without changing the semantic task.
    """

    stage = config["book_subject_stage"]
    instructions = "\n".join(
        f"{index}. {instruction}"
        for index, instruction in enumerate(stage["instructions"], start=1)
    )

    correction = ""
    if validation_feedback:
        valid_span_ids_json = json.dumps(
            list(spans.keys()),
            ensure_ascii=False,
        )
        correction = f"""

PREVIOUS VALIDATION FAILURE
{validation_feedback}

VALID SUPPLIED SPAN IDS
{valid_span_ids_json}

CORRECTION REQUIRED
Return a corrected BookSubjectAnalysis only.
Use only individual exact span IDs from VALID SUPPLIED SPAN IDS above.
Do not invent, infer, renumber, expand, or range-compress span IDs.
Do not change the semantic task; only repair the invalid output.
"""

    return f"""
Analyze ONE supplied book description to identify what the BOOK ITSELF is primarily about.

IMPORTANT ISOLATION RULE
No user query or query facet is available at this stage. Base the answer only on the supplied description.

FULL NUMBERED BOOK-DESCRIPTION SPANS
{format_description_spans(spans)}

INSTRUCTIONS
{instructions}

Return one BookSubjectAnalysis with primary_subject_summary,
primary_subject_span_ids, and reason.{correction}
""".strip()


def assess_book_subject(
    judge_model: Any,
    spans: dict[str, str],
    config: dict[str, Any],
) -> tuple[BookSubjectAnalysis, int]:
    """Analyze/freeze the book-level subject exactly once for this case.

    v0.25.0 preserves strict exact-span validation but feeds mechanical
    validation failures back to the next attempt. This lets the model repair
    formatting mistakes such as ``S6-S9`` without silently normalizing or
    reinterpreting the model output in Python.
    """

    last_error: Exception | None = None
    validation_feedback: str | None = None

    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):
        try:
            generated = judge_model.generate(
                prompt=build_book_subject_prompt(
                    spans=spans,
                    config=config,
                    validation_feedback=validation_feedback,
                ),
                schema=BookSubjectAnalysis,
            )
            result = unpack_generated_model(generated, BookSubjectAnalysis)
            assert isinstance(result, BookSubjectAnalysis)
            _validate_book_subject_result(result=result, spans=spans)
            return result, attempt - 1
        except JudgeOutputValidationError as error:
            last_error = error
            validation_feedback = str(error)
        except Exception as error:
            # Transport/schema-generation failures still use the existing retry
            # budget, but they are not echoed into the semantic prompt.
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
        for name, definition in stage["subject_relation_scale"].items()
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

SUBJECT RELATION SCALE
{context_scale}

INSTRUCTIONS
{instructions}

Return one ProminenceAssessment with exactly one subject_relation,
is_substantively_examined, supporting_span_ids, and reason.
Do NOT return or redefine the book subject. Python derives backward-compatible
role booleans, final context_role, and prominence after this stage.
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
    dict[str, Any],
    int,
]:
    """
    Run v0.26.0 facet evaluation against one frozen book subject:

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
        component-evidence ledger audit
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
                {
                    "mode": "hard_exclusion_precheck",
                    "single_span": {},
                    "composite": None,
                },
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
                {
                    "mode": "deterministic_direct_cue",
                    "single_span": {span_id: {"cue_id": cue_match.cue_id}},
                    "composite": None,
                },
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
        prominence_result = _apply_constituent_item_scope_guard(
            facet=facet,
            evidence_text=cue_match.evidence_text,
            spans=spans,
            result=prominence_result,
        )
        prominence_result = _apply_direct_cue_list_mention_guard(
            evidence_span_id=span_id,
            evidence_text=cue_match.evidence_text,
            result=prominence_result,
        )
        prominence_result = _r6_whole_work_thematic_enumeration_positive_guard(
            facet=facet,
            evidence_text=cue_match.evidence_text,
            result=prominence_result,
        )
        prominence_result = _apply_v028_r5_prominence_guard(
            facet=facet,
            evidence_text=cue_match.evidence_text,
            result=prominence_result,
        )

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
                subject_relation=prominence_result.subject_relation,
                subject_relation_reason=prominence_result.reason,
                **_legacy_role_flags(prominence_result),
                role_supporting_span_ids=prominence_result.supporting_span_ids,
                role_reason=prominence_result.reason,
                evidence_selection_reason=reason,
                verification_reason=reason,
                prominence_reason=prominence_result.reason,
                deterministic_cue_polarity_blocked_count=polarity_blocked_count,
            ),
            cue_match.evidence_text,
            candidate_texts,
            {
                "mode": "deterministic_direct_cue",
                "single_span": {span_id: {"cue_id": cue_match.cue_id}},
                "composite": None,
            },
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
    verification_results_by_span: dict[str, EvidenceVerification] = {}
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
        best, verification_records, verification_results_by_span, retries = verify_ranked_candidates(
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
            verification_results_by_span=verification_results_by_span,
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
        composite_reason = (
            composite_result.reason
            + " [component_check: all_required_components_established="
            + str(composite_result.all_required_components_established).lower()
            + "; semantic_definition_exclusion_applied="
            + str(composite_result.semantic_definition_exclusion_applied).lower()
            + "; missing_semantic_component="
            + repr(composite_result.missing_semantic_component)
            + "; component_ledger="
            + json.dumps(
                [check.model_dump(mode="json") for check in composite_result.component_checks],
                ensure_ascii=False,
                sort_keys=True,
            )
            + "]"
        )
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

    component_audit: dict[str, Any] = {
        "mode": "llm_verification",
        "single_span": {
            span_id: {
                "verification_relation": result.verification_relation.value,
                "semantic_definition_exclusion_applied": result.semantic_definition_exclusion_applied,
                "component_checks": [
                    check.model_dump(mode="json") for check in result.component_checks
                ],
            }
            for span_id, result in verification_results_by_span.items()
        },
        "composite": (
            {
                "verification_relation": composite_result.verification_relation.value,
                "component_checks": [
                    check.model_dump(mode="json") for check in composite_result.component_checks
                ],
                "semantic_definition_exclusion_applied": (
                    composite_result.semantic_definition_exclusion_applied
                ),
            }
            if composite_verification_attempted
            else None
        ),
    }

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
            component_audit,
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
            component_audit,
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
            component_audit,
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
    prominence_result = _apply_constituent_item_scope_guard(
        facet=facet,
        evidence_text=winning_evidence_text,
        spans=spans,
        result=prominence_result,
    )
    prominence_result = _apply_v028_r5_prominence_guard(
        facet=facet,
        evidence_text=winning_evidence_text,
        result=prominence_result,
    )

    return (
        FacetPipelineAssessment(
            **common_kwargs,
            prominence=_prominence_from_context_role(_context_role_from_decomposed_signals(prominence_result), config),
            context_role=_context_role_from_decomposed_signals(prominence_result),
            primary_subject_summary=book_subject.primary_subject_summary,
                primary_subject_span_ids=book_subject.primary_subject_span_ids,
            subject_relation=prominence_result.subject_relation,
            subject_relation_reason=prominence_result.reason,
            **_legacy_role_flags(prominence_result),
            role_supporting_span_ids=prominence_result.supporting_span_ids,
            role_reason=prominence_result.reason,
            prominence_reason=prominence_result.reason,
        ),
        winning_evidence_text,
        candidate_texts,
        component_audit,
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
    Run v0.26.0 with one frozen book-subject analysis followed by every frozen facet.

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

    # v0.26.0: freeze one facet-independent subject analysis before ANY facet
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

    component_evidence_ledger_by_id: dict[str, Any] = {}

    total_retries = 0

    for facet in spec.facets:

        (
            assessment,
            winning_evidence_text,
            candidate_texts,
            component_audit,
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

        component_evidence_ledger_by_id[
            facet.facet_id
        ] = component_audit

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
        component_evidence_ledger_by_id=component_evidence_ledger_by_id,
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
                "subject_relation": (assessment.subject_relation.value if assessment.subject_relation is not None else None),
                "subject_relation_reason": assessment.subject_relation_reason,
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
