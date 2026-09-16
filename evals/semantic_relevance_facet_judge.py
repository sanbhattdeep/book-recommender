"""
Polarity-guarded deterministic cue + self-selecting full-context composition semantic facet judge for v0.18.0.

v0.18.0 removes the separate composition selector. When a core facet's best
standalone relation is UNSUPPORTED or ADJACENT, one isolated composite call sees
the full numbered description, selects its own 2-4 exact supporting spans, and
returns UNSUPPORTED / ADJACENT / ENTAILED.

ENTAILED is explicitly multi-span semantic entailment: exact wording is not
required when the cited spans jointly supply every semantic component through a
short necessary inference. ADJACENT requires an explicit missing semantic
component. DIRECT remains forbidden for composite recovery.
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
    reason: str = Field(min_length=1)


class CompositeEvidenceVerification(BaseModel):
    """Self-selecting full-context multi-span verification output."""

    supporting_span_ids: list[str] = Field(
        default_factory=list,
        max_length=MAX_COMPOSITE_CANDIDATES,
    )
    verification_relation: VerificationRelation
    combined_evidence_summary: str = Field(min_length=1)
    missing_semantic_component: str | None = None
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

    # Every Stage-A candidate plus any full-context composite supporting spans, retained for audit in facet_evidence_json.
    candidate_evidence_by_id: dict[str, dict[str, str]]

    generation_mode: str = "polarity_guarded_cue_plus_isolated_self_selecting_full_context_composite_pipeline"

    stage_retry_count: int = 0

    deterministic_direct_cue_count: int = 0

    deterministic_cue_polarity_blocked_count: int = 0

    composite_verification_attempt_count: int = 0

    composite_verification_count: int = 0


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
    spans: dict[str, str],
    config: dict[str, Any],
) -> None:
    """Deterministically validate v0.18 composite structure/consistency."""

    stage = config["composite_verification_stage"]
    min_spans = int(stage.get("min_spans", 2))
    max_spans = int(stage.get("max_spans", MAX_COMPOSITE_CANDIDATES))
    relation = result.verification_relation
    ids = list(result.supporting_span_ids)

    if relation == VerificationRelation.DIRECT:
        raise JudgeOutputValidationError(
            "Composite verification may not return DIRECT."
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
                "positive relations require 2-4 unique real span IDs."
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
    Run v0.18 facet evaluation:

        polarity-safe deterministic cue
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
                    deterministic_cue_polarity_blocked_count=polarity_blocked_count,
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
        best_single_reason = best.verification_reason

    # ------------------------------------------------------------------
    # Stage C: self-selecting full-context composition recovery.
    # ------------------------------------------------------------------
    composite_verification_attempted = False
    composite_span_ids: list[str] = []
    composite_relation: VerificationRelation | None = None
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
        evidence_selection_reason=selection.reason,
        verification_reason=final_verification_reason,
        deterministic_cue_polarity_blocked_count=polarity_blocked_count,
        composite_verification_attempted=composite_verification_attempted,
        composite_evidence_span_ids=composite_span_ids,
        composite_verification_relation=composite_relation,
        composite_combined_evidence_summary=composite_summary,
        composite_missing_semantic_component=composite_missing_component,
        composite_verification_reason=composite_reason,
    )

    if relation == VerificationRelation.UNSUPPORTED:
        return (
            FacetPipelineAssessment(
                **common_kwargs,
                prominence=FacetProminence.NOT_APPLICABLE,
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
                prominence_reason=None,
            ),
            winning_evidence_text,
            candidate_texts,
            total_retries,
        )

    assert winning_evidence_text is not None

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
            **common_kwargs,
            prominence=prominence_result.prominence,
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
    Run v0.18.0 for every frozen facet.

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

    return SemanticGenerationResult(
        verdict=verdict,
        evidence_by_id=evidence_by_id,
        candidate_evidence_by_id=candidate_evidence_by_id,
        stage_retry_count=total_retries,
        deterministic_direct_cue_count=deterministic_direct_cue_count,
        deterministic_cue_polarity_blocked_count=(
            deterministic_cue_polarity_blocked_count
        ),
        composite_verification_attempt_count=composite_verification_attempt_count,
        composite_verification_count=composite_verification_count,
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
