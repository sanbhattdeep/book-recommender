"""
Deterministic scoring models for Semantic Recommendation Relevance v0.26.0.

v0.20.0 preserves the deterministic support derivation and final 0-4 aggregation unchanged. It adds auditable inference-kind and context-role fields used by the semantic stages before deterministic support derivation.

v0.11.0 added an ADJACENT verification relation so weak but genuine facet-specific
connections can map to score-1 incidental relevance without being promoted to
full semantic support. It also makes required qualifiers capable of capping a
clear/strong score when the modifier is absent.

Core support derivation
-----------------------
unsupported                         -> absent
adjacent                            -> incidental
entailed/direct + incidental        -> incidental
entailed + substantive/central      -> meaningful
direct + substantive                -> meaningful
direct + central                    -> strong

Qualifier support derivation
----------------------------
unsupported/adjacent -> absent
entailed             -> meaningful
direct               -> strong

Overall aggregation
-------------------
0  all core facets absent
1  at least one incidental core facet and no meaningful/strong core facet
2  substantive core support exists but no score-3 rule is met
3  at least two-thirds of core facets are meaningful/strong, OR exactly two
   core facets exist, at least one is strong, and neither core facet is absent
4  all core facets strong

If any required qualifier is absent, provisional 3/4 is capped to 2.
If all qualifiers are meaningful but not all are strong, provisional 4 is capped to 3.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


FacetType = Literal["core", "qualifier"]


class HardExclusion(BaseModel):
    """Frozen facet-specific negative boundary evaluated before positive support."""

    exclusion_id: str = Field(min_length=1)
    rule: str = Field(min_length=1)


class FacetRequiredComponent(BaseModel):
    """One canonical indispensable component of a frozen semantic facet.

    v0.25.0+ moves component identity out of model generation and into the
    versioned facet specification.  `negative_boundaries` are generic semantic
    constraints for this component; they must never encode candidate-specific
    facts, labels, or observed judge outputs.
    """

    component_id: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    negative_boundaries: list[str] = Field(default_factory=list, max_length=12)


class QueryFacet(BaseModel):
    """
    One frozen semantic facet from the query-facet specification.

    `semantic_definition` and `required_components` are part of the versioned
    evaluation contract.  Components are canonical: the verifier may determine
    whether each one is established, but may not rename, omit, add, or substitute
    components at generation time.
    """

    facet_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    facet_type: FacetType
    semantic_definition: str = Field(min_length=1)
    hard_exclusions: list[HardExclusion] = Field(default_factory=list, max_length=12)
    # Default keeps legacy unit-test fixtures parseable; the v0.25 facet-spec
    # contract itself requires at least one canonical component for every facet.
    required_components: list[FacetRequiredComponent] = Field(default_factory=list, max_length=8)


class QueryFacetSpec(BaseModel):
    """Frozen facet decomposition for one query."""

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    facets: list[QueryFacet] = Field(min_length=1, max_length=8)


class VerificationRelation(str, Enum):
    """Result of isolated facet-vs-evidence verification."""

    UNSUPPORTED = "unsupported"
    ADJACENT = "adjacent"
    ENTAILED = "entailed"
    DIRECT = "direct"


class EvidenceInferenceKind(str, Enum):
    """Auditable semantic bridge used to justify a verification relation."""

    NONE = "none"
    EXPLICIT_COMPONENTS = "explicit_components"
    NECESSARY_SEMANTIC_INFERENCE = "necessary_semantic_inference"
    INCOMPLETE_CONNECTION = "incomplete_connection"
    SYMBOLIC_POSSIBILITY = "symbolic_possibility"
    ASSOCIATIVE_WORLD_KNOWLEDGE = "associative_world_knowledge"


class SubjectRelation(str, Enum):
    """Facet relationship to the frozen facet-independent book subject."""

    SAME_AS_PRIMARY_SUBJECT = "same_as_primary_subject"
    DEFINING_CONTENT_OR_NARRATIVE_DRIVER = "defining_content_or_narrative_driver"
    CAUSAL_OR_CONTEXTUAL_BACKGROUND = "causal_or_contextual_background"
    EXAMPLE_OR_META = "example_or_meta"
    OTHER = "other"


class FacetContextRole(str, Enum):
    """Role played by an already-verified core facet in the described work."""

    NOT_APPLICABLE = "not_applicable"
    CENTRAL_SUBJECT = "central_subject"
    SUBSTANTIVE_SUBJECT = "substantive_subject"
    BACKGROUND_CAUSE = "background_cause"
    EXAMPLE_OR_ILLUSTRATION = "example_or_illustration"
    META_DISCUSSION = "meta_discussion"
    INCIDENTAL_MENTION = "incidental_mention"


class FacetProminence(str, Enum):
    """Prominence of an already-verified CORE facet."""

    NOT_APPLICABLE = "not_applicable"
    INCIDENTAL = "incidental"
    SUBSTANTIVE = "substantive"
    CENTRAL = "central"


class FacetSupport(str, Enum):
    """Legacy support scale, derived only by Python."""

    ABSENT = "absent"
    INCIDENTAL = "incidental"
    MEANINGFUL = "meaningful"
    STRONG = "strong"


SUPPORT_VALUE = {
    FacetSupport.ABSENT: 0,
    FacetSupport.INCIDENTAL: 1,
    FacetSupport.MEANINGFUL: 2,
    FacetSupport.STRONG: 3,
}


class CandidateVerificationRecord(BaseModel):
    """
    Audit record for one candidate source span sent through the isolated
    verifier.

    Candidate rank is 1-based and follows the evidence selector's order.
    """

    candidate_rank: int = Field(ge=1, le=3)
    evidence_span_id: str = Field(min_length=1)
    verification_relation: VerificationRelation
    inference_kind: EvidenceInferenceKind = EvidenceInferenceKind.NONE
    hard_exclusion_triggered: bool = False
    hard_exclusion_id: str | None = None
    verification_reason: str = Field(min_length=1)


class FacetPipelineAssessment(BaseModel):
    """
    Final v0.22.0 result for one frozen facet.

    `candidate_evidence_span_ids` contains the selector's ranked candidates.

    `candidate_verifications` contains every candidate that was actually
    verified. Verification may stop early when DIRECT is found because no later
    candidate can outrank DIRECT.

    `candidate_evidence_span_id` is retained as the winning/representative span
    for compatibility with v0.9.0/v0.9.1 audit output. It is "NONE" when the
    selector found no candidate.
    """

    facet_id: str = Field(min_length=1)

    candidate_evidence_span_ids: list[str] = Field(
        default_factory=list,
        max_length=3,
    )

    candidate_verifications: list[CandidateVerificationRecord] = Field(
        default_factory=list,
        max_length=3,
    )

    candidate_evidence_span_id: str = Field(min_length=1)

    verification_relation: VerificationRelation

    inference_kind: EvidenceInferenceKind = EvidenceInferenceKind.NONE

    hard_exclusion_triggered: bool = False
    hard_exclusion_id: str | None = None

    # v0.21+ full-description negative-boundary precheck audit.
    hard_exclusion_precheck_attempted: bool = False
    hard_exclusion_precheck_triggered: bool = False
    hard_exclusion_precheck_id: str | None = None
    hard_exclusion_precheck_supporting_span_ids: list[str] = Field(default_factory=list, max_length=4)
    hard_exclusion_precheck_reason: str | None = None

    prominence: FacetProminence

    # v0.22.1 explicit relation to the frozen case-level book subject.
    subject_relation: SubjectRelation | None = None
    subject_relation_reason: str | None = None

    # Context role remains an audit field, but in v0.21+ it is derived
    # deterministically in Python from decomposed role signals.
    context_role: FacetContextRole = FacetContextRole.NOT_APPLICABLE
    primary_subject_summary: str | None = None
    primary_subject_span_ids: list[str] = Field(default_factory=list, max_length=6)
    is_primary_subject: bool = False
    is_background_cause_or_factor: bool = False
    is_example_or_illustration: bool = False
    is_meta_discussion: bool = False
    is_substantively_examined: bool = False
    role_supporting_span_ids: list[str] = Field(default_factory=list, max_length=6)
    role_reason: str | None = None

    evidence_selection_reason: str = Field(min_length=1)

    verification_reason: str = Field(min_length=1)

    prominence_reason: str | None = None

    # v0.16+ audit: deterministic lexical cue matches suppressed because the
    # local context explicitly negated/contrasted the matched concept.
    deterministic_cue_polarity_blocked_count: int = Field(default=0, ge=0)

    # v0.19 audit: the full-context composite verifier scans all numbered
    # description spans and selects its own 1-4 supporting span IDs.
    composite_verification_attempted: bool = False
    composite_evidence_span_ids: list[str] = Field(
        default_factory=list,
        max_length=4,
    )
    composite_verification_relation: VerificationRelation | None = None
    composite_inference_kind: EvidenceInferenceKind | None = None
    composite_hard_exclusion_triggered: bool = False
    composite_hard_exclusion_id: str | None = None
    composite_combined_evidence_summary: str | None = None
    composite_missing_semantic_component: str | None = None
    composite_verification_reason: str | None = None


class FacetJudgeVerdict(BaseModel):
    """Complete frozen-facet verdict for one query-book pair."""

    assessments: list[FacetPipelineAssessment] = Field(min_length=1)
    overall_reason: str = Field(min_length=1)


ScoreLabel = Literal[
    "none",
    "incidental",
    "partial",
    "clear",
    "strong",
]


class FacetScoreResult(BaseModel):
    """Deterministic 0-4 score plus audit diagnostics."""

    score: Literal[0, 1, 2, 3, 4]
    label: ScoreLabel

    core_facet_count: int
    incidental_core_count: int
    meaningful_core_count: int
    meaningful_core_coverage: float
    strong_core_count: int

    direct_core_count: int
    entailed_core_count: int
    adjacent_core_count: int
    unsupported_core_count: int

    qualifier_count: int
    satisfied_qualifier_count: int
    qualifier_cap_applied: bool

    clear_rule_applied: str | None

    explanation: str


SCORE_LABELS: dict[int, ScoreLabel] = {
    0: "none",
    1: "incidental",
    2: "partial",
    3: "clear",
    4: "strong",
}


# =============================================================================
# Mechanical validation
# =============================================================================

def validate_query_facet_spec(spec: QueryFacetSpec) -> None:
    """Validate the frozen query-facet decomposition."""

    facet_ids = [facet.facet_id for facet in spec.facets]

    if len(facet_ids) != len(set(facet_ids)):
        raise ValueError(
            f"Duplicate facet_id values in query {spec.query_id}: {facet_ids}"
        )

    if not any(facet.facet_type == "core" for facet in spec.facets):
        raise ValueError(
            f"Query {spec.query_id} must contain at least one core facet."
        )


def validate_facet_assessments(
    spec: QueryFacetSpec,
    verdict: FacetJudgeVerdict,
) -> dict[str, FacetPipelineAssessment]:
    """
    Validate only mechanically checkable contracts.

    Semantic correctness is intentionally not validated here.
    """

    validate_query_facet_spec(spec)

    expected_ids = {facet.facet_id for facet in spec.facets}
    returned_ids = [a.facet_id for a in verdict.assessments]

    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError(
            f"Judge returned duplicate facet assessments: {returned_ids}"
        )

    returned_set = set(returned_ids)

    missing = expected_ids - returned_set
    unexpected = returned_set - expected_ids

    if missing:
        raise ValueError(
            f"Judge omitted required facets: {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"Judge invented unexpected facets: {sorted(unexpected)}"
        )

    facet_by_id = {facet.facet_id: facet for facet in spec.facets}
    assessment_by_id = {
        assessment.facet_id: assessment
        for assessment in verdict.assessments
    }

    for facet_id, assessment in assessment_by_id.items():

        facet = facet_by_id[facet_id]

        candidates = assessment.candidate_evidence_span_ids

        if len(candidates) != len(set(candidates)):
            raise ValueError(
                f"{facet_id}: candidate evidence span IDs must be unique."
            )

        if len(candidates) > 3:
            raise ValueError(
                f"{facet_id}: at most 3 candidate spans are allowed."
            )

        composite_ids = assessment.composite_evidence_span_ids
        composite_relation = assessment.composite_verification_relation
        missing_component = assessment.composite_missing_semantic_component

        # Stage-A may legitimately return NONE while the independent v0.19
        # full-context composite verifier later recovers the facet.
        if assessment.candidate_evidence_span_id == "NONE":

            if candidates:
                raise ValueError(
                    f"{facet_id}: winning span NONE is incompatible with "
                    "non-empty candidate_evidence_span_ids."
                )

            if (
                assessment.verification_relation
                != VerificationRelation.UNSUPPORTED
                and not composite_ids
            ):
                raise ValueError(
                    f"{facet_id}: non-unsupported final relation with Stage-A NONE "
                    "requires positive composite evidence."
                )

        else:
            if assessment.candidate_evidence_span_id not in candidates:
                raise ValueError(
                    f"{facet_id}: winning Stage-A span must appear in the ranked "
                    "candidate list."
                )

        if len(composite_ids) != len(set(composite_ids)):
            raise ValueError(
                f"{facet_id}: composite evidence span IDs must be unique."
            )

        if len(composite_ids) > 4:
            raise ValueError(
                f"{facet_id}: at most 4 composite supporting spans are allowed."
            )

        if composite_ids and not assessment.composite_verification_attempted:
            raise ValueError(
                f"{facet_id}: composite evidence requires "
                "composite_verification_attempted=True."
            )

        if assessment.composite_verification_attempted and composite_relation is None:
            raise ValueError(
                f"{facet_id}: attempted composite verification requires a relation."
            )

        if composite_relation == VerificationRelation.DIRECT:
            raise ValueError(
                f"{facet_id}: composite verification may never be DIRECT."
            )

        if composite_relation in {
            VerificationRelation.ADJACENT,
            VerificationRelation.ENTAILED,
        }:
            if len(composite_ids) < 1:
                raise ValueError(
                    f"{facet_id}: positive full-context verification requires "
                    "at least one supporting span."
                )
        elif composite_relation == VerificationRelation.UNSUPPORTED:
            if composite_ids:
                raise ValueError(
                    f"{facet_id}: unsupported composite verification must not "
                    "contain supporting span IDs."
                )

        if composite_relation == VerificationRelation.ADJACENT:
            if not (missing_component and missing_component.strip()):
                raise ValueError(
                    f"{facet_id}: ADJACENT composite verification must identify "
                    "a missing semantic component."
                )

        if composite_relation == VerificationRelation.ENTAILED:
            if missing_component and missing_component.strip():
                raise ValueError(
                    f"{facet_id}: ENTAILED composite verification cannot declare "
                    "a missing semantic component."
                )

        if assessment.verification_relation in {
            VerificationRelation.UNSUPPORTED,
            VerificationRelation.ADJACENT,
        }:
            if assessment.prominence != FacetProminence.NOT_APPLICABLE:
                raise ValueError(
                    f"{facet_id}: unsupported/adjacent verification requires "
                    "prominence='not_applicable'."
                )

        elif facet.facet_type == "qualifier":
            if assessment.prominence != FacetProminence.NOT_APPLICABLE:
                raise ValueError(
                    f"{facet_id}: qualifiers must use "
                    "prominence='not_applicable'."
                )

        else:
            if assessment.prominence == FacetProminence.NOT_APPLICABLE:
                raise ValueError(
                    f"{facet_id}: supported core facets require prominence."
                )

    return assessment_by_id


# =============================================================================
# Deterministic support derivation
# =============================================================================

def derive_facet_support(
    facet: QueryFacet,
    assessment: FacetPipelineAssessment,
) -> FacetSupport:
    """
    Convert winning verification relation + prominence into legacy support.

    Behavior unchanged from v0.9.2/v0.9.3.
    """

    relation = assessment.verification_relation
    prominence = assessment.prominence

    if relation == VerificationRelation.UNSUPPORTED:
        return FacetSupport.ABSENT

    if facet.facet_type == "qualifier":

        if relation == VerificationRelation.DIRECT:
            return FacetSupport.STRONG

        if relation == VerificationRelation.ENTAILED:
            return FacetSupport.MEANINGFUL

        # Adjacent qualifier evidence is too weak to satisfy a requested modifier.
        return FacetSupport.ABSENT

    # Adjacent is a genuine weak facet-specific connection. It intentionally maps
    # to incidental without a separate prominence call.
    if relation == VerificationRelation.ADJACENT:
        return FacetSupport.INCIDENTAL

    if prominence == FacetProminence.INCIDENTAL:
        return FacetSupport.INCIDENTAL

    if (
        relation == VerificationRelation.DIRECT
        and prominence == FacetProminence.CENTRAL
    ):
        return FacetSupport.STRONG

    return FacetSupport.MEANINGFUL


def derived_support_by_id(
    spec: QueryFacetSpec,
    verdict: FacetJudgeVerdict,
) -> dict[str, FacetSupport]:
    """Return Python-derived support for every frozen facet."""

    assessment_by_id = validate_facet_assessments(
        spec=spec,
        verdict=verdict,
    )

    return {
        facet.facet_id: derive_facet_support(
            facet=facet,
            assessment=assessment_by_id[facet.facet_id],
        )
        for facet in spec.facets
    }


# =============================================================================
# Overall deterministic 0-4 aggregation
# =============================================================================

def compute_facet_score(
    spec: QueryFacetSpec,
    verdict: FacetJudgeVerdict,
) -> FacetScoreResult:
    """
    Aggregate Python-derived support using the v0.14 two-core absence guard.
    """

    assessment_by_id = validate_facet_assessments(
        spec=spec,
        verdict=verdict,
    )

    support_by_id = derived_support_by_id(
        spec=spec,
        verdict=verdict,
    )

    core_facets = [
        facet
        for facet in spec.facets
        if facet.facet_type == "core"
    ]

    qualifier_facets = [
        facet
        for facet in spec.facets
        if facet.facet_type == "qualifier"
    ]

    core_supports = [
        support_by_id[facet.facet_id]
        for facet in core_facets
    ]

    qualifier_supports = [
        support_by_id[facet.facet_id]
        for facet in qualifier_facets
    ]

    core_count = len(core_facets)

    incidental_core_count = sum(
        support == FacetSupport.INCIDENTAL
        for support in core_supports
    )

    meaningful_core_count = sum(
        SUPPORT_VALUE[support]
        >= SUPPORT_VALUE[FacetSupport.MEANINGFUL]
        for support in core_supports
    )

    strong_core_count = sum(
        support == FacetSupport.STRONG
        for support in core_supports
    )

    meaningful_core_coverage = (
        meaningful_core_count / core_count
    )

    direct_core_count = sum(
        assessment_by_id[facet.facet_id].verification_relation
        == VerificationRelation.DIRECT
        for facet in core_facets
    )

    entailed_core_count = sum(
        assessment_by_id[facet.facet_id].verification_relation
        == VerificationRelation.ENTAILED
        for facet in core_facets
    )

    adjacent_core_count = sum(
        assessment_by_id[facet.facet_id].verification_relation
        == VerificationRelation.ADJACENT
        for facet in core_facets
    )

    unsupported_core_count = sum(
        assessment_by_id[facet.facet_id].verification_relation
        == VerificationRelation.UNSUPPORTED
        for facet in core_facets
    )

    satisfied_qualifier_count = sum(
        SUPPORT_VALUE[support]
        >= SUPPORT_VALUE[FacetSupport.MEANINGFUL]
        for support in qualifier_supports
    )

    qualifier_cap_applied = False
    clear_rule_applied: str | None = None

    max_core_value = max(
        SUPPORT_VALUE[support]
        for support in core_supports
    )

    if max_core_value == SUPPORT_VALUE[FacetSupport.ABSENT]:
        score = 0

    elif max_core_value <= SUPPORT_VALUE[FacetSupport.INCIDENTAL]:
        score = 1

    else:

        all_core_strong = (
            strong_core_count == core_count
        )

        if all_core_strong:
            score = 4

        else:

            meaningful_required_for_clear = math.ceil(
                (2 * core_count) / 3
            )

            coverage_rule_met = (
                meaningful_core_count
                >= meaningful_required_for_clear
            )

            # v0.14: a strong match on one half of an exactly two-core
            # request cannot by itself make the recommendation "clear" when
            # the other half is completely absent. Weak-but-genuine incidental
            # support on the second facet is enough to retain the historical
            # two-core exception.
            two_facet_strong_exception_met = (
                core_count == 2
                and strong_core_count >= 1
                and all(
                    support != FacetSupport.ABSENT
                    for support in core_supports
                )
            )

            if coverage_rule_met:
                score = 3
                clear_rule_applied = (
                    "two_thirds_core_coverage"
                )

            elif two_facet_strong_exception_met:
                score = 3
                clear_rule_applied = (
                    "two_core_one_strong_no_absent"
                )

            else:
                score = 2

    # Qualifiers cannot promote relevance.
    #
    # v0.11 distinguishes:
    # - absent qualifier: a requested modifier is missing, so a clear/strong
    #   match is capped to partial (2);
    # - meaningful qualifier: enough for a clear match, but not enough for a
    #   perfect/strong 4 unless every qualifier is strong.
    if qualifier_facets:
        absent_qualifier_count = sum(
            support == FacetSupport.ABSENT
            for support in qualifier_supports
        )
        strong_qualifier_count = sum(
            support == FacetSupport.STRONG
            for support in qualifier_supports
        )

        if absent_qualifier_count and score >= 3:
            score = 2
            qualifier_cap_applied = True
            clear_rule_applied = "qualifier_absent_cap_to_partial"

        elif (
            score == 4
            and strong_qualifier_count < len(qualifier_facets)
        ):
            score = 3
            qualifier_cap_applied = True
            clear_rule_applied = "qualifier_meaningful_cap_from_strong"

    explanation = (
        f"{meaningful_core_count}/{core_count} core facets derive to "
        f"meaningful/strong; {strong_core_count}/{core_count} derive to strong. "
        f"Verification: direct={direct_core_count}, "
        f"entailed={entailed_core_count}, "
        f"adjacent={adjacent_core_count}, "
        f"unsupported={unsupported_core_count}."
    )

    if incidental_core_count:
        explanation += (
            f" Incidental core facets={incidental_core_count}."
        )

    if qualifier_facets:
        explanation += (
            f" Satisfied qualifiers={satisfied_qualifier_count}/"
            f"{len(qualifier_facets)}."
        )

    if clear_rule_applied:
        explanation += (
            f" Clear-score rule applied: {clear_rule_applied}."
        )

    return FacetScoreResult(
        score=score,
        label=SCORE_LABELS[score],
        core_facet_count=core_count,
        incidental_core_count=incidental_core_count,
        meaningful_core_count=meaningful_core_count,
        meaningful_core_coverage=meaningful_core_coverage,
        strong_core_count=strong_core_count,
        direct_core_count=direct_core_count,
        entailed_core_count=entailed_core_count,
        adjacent_core_count=adjacent_core_count,
        unsupported_core_count=unsupported_core_count,
        qualifier_count=len(qualifier_facets),
        satisfied_qualifier_count=satisfied_qualifier_count,
        qualifier_cap_applied=qualifier_cap_applied,
        clear_rule_applied=clear_rule_applied,
        explanation=explanation,
    )
