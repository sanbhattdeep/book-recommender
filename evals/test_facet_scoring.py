"""
Deterministic scoring tests for Judge Config v0.21.0.

No LLM is called.
"""

from semantic_relevance_facet_scoring import (
    CandidateVerificationRecord,
    FacetJudgeVerdict,
    FacetPipelineAssessment,
    FacetProminence,
    FacetSupport,
    QueryFacet,
    QueryFacetSpec,
    VerificationRelation,
    compute_facet_score,
    derive_facet_support,
)


def make_spec(
    query_id: str,
    facets: list[tuple[str, str]],
) -> QueryFacetSpec:
    return QueryFacetSpec(
        query_id=query_id,
        query="synthetic query",
        facets=[
            QueryFacet(
                facet_id=f"F{index}",
                text=text,
                facet_type=facet_type,
                semantic_definition=f"Synthetic definition for {text}.",
            )
            for index, (text, facet_type)
            in enumerate(facets, start=1)
        ],
    )


def assessment(
    facet_id: str,
    relation: VerificationRelation,
    prominence: FacetProminence,
) -> FacetPipelineAssessment:

    if relation == VerificationRelation.UNSUPPORTED:
        candidates: list[str] = []
        records: list[CandidateVerificationRecord] = []
        winning = "NONE"
    else:
        candidates = ["S1"]
        records = [
            CandidateVerificationRecord(
                candidate_rank=1,
                evidence_span_id="S1",
                verification_relation=relation,
                verification_reason="synthetic",
            )
        ]
        winning = "S1"

    return FacetPipelineAssessment(
        facet_id=facet_id,
        candidate_evidence_span_ids=candidates,
        candidate_verifications=records,
        candidate_evidence_span_id=winning,
        verification_relation=relation,
        prominence=prominence,
        evidence_selection_reason="synthetic",
        verification_reason="synthetic",
        prominence_reason=(
            "synthetic"
            if prominence != FacetProminence.NOT_APPLICABLE
            else None
        ),
        primary_subject_summary=(
            "synthetic subject"
            if prominence != FacetProminence.NOT_APPLICABLE
            else None
        ),
        role_supporting_span_ids=(
            ["S1"]
            if prominence != FacetProminence.NOT_APPLICABLE
            else []
        ),
    )


def verdict(
    *items: FacetPipelineAssessment,
) -> FacetJudgeVerdict:
    return FacetJudgeVerdict(
        assessments=list(items),
        overall_reason="synthetic",
    )


# Core support mapping.
core = QueryFacet(
    facet_id="F1",
    text="crime",
    facet_type="core",
    semantic_definition="Criminal acts or conduct.",
)

qualifier = QueryFacet(
    facet_id="F1",
    text="beginner-friendly",
    facet_type="qualifier",
    semantic_definition="Accessible to a beginner or general reader.",
)

assert derive_facet_support(
    core,
    assessment(
        "F1",
        VerificationRelation.UNSUPPORTED,
        FacetProminence.NOT_APPLICABLE,
    ),
) == FacetSupport.ABSENT

assert derive_facet_support(
    core,
    assessment(
        "F1",
        VerificationRelation.ADJACENT,
        FacetProminence.NOT_APPLICABLE,
    ),
) == FacetSupport.INCIDENTAL

assert derive_facet_support(
    core,
    assessment(
        "F1",
        VerificationRelation.ENTAILED,
        FacetProminence.INCIDENTAL,
    ),
) == FacetSupport.INCIDENTAL

assert derive_facet_support(
    core,
    assessment(
        "F1",
        VerificationRelation.DIRECT,
        FacetProminence.INCIDENTAL,
    ),
) == FacetSupport.INCIDENTAL

assert derive_facet_support(
    core,
    assessment(
        "F1",
        VerificationRelation.ENTAILED,
        FacetProminence.CENTRAL,
    ),
) == FacetSupport.MEANINGFUL

assert derive_facet_support(
    core,
    assessment(
        "F1",
        VerificationRelation.DIRECT,
        FacetProminence.SUBSTANTIVE,
    ),
) == FacetSupport.MEANINGFUL

assert derive_facet_support(
    core,
    assessment(
        "F1",
        VerificationRelation.DIRECT,
        FacetProminence.CENTRAL,
    ),
) == FacetSupport.STRONG

assert derive_facet_support(
    qualifier,
    assessment(
        "F1",
        VerificationRelation.ADJACENT,
        FacetProminence.NOT_APPLICABLE,
    ),
) == FacetSupport.ABSENT

assert derive_facet_support(
    qualifier,
    assessment(
        "F1",
        VerificationRelation.ENTAILED,
        FacetProminence.NOT_APPLICABLE,
    ),
) == FacetSupport.MEANINGFUL

assert derive_facet_support(
    qualifier,
    assessment(
        "F1",
        VerificationRelation.DIRECT,
        FacetProminence.NOT_APPLICABLE,
    ),
) == FacetSupport.STRONG


# Score 0.
s = make_spec(
    "T0",
    [
        ("leadership", "core"),
        ("teamwork", "core"),
    ],
)

r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.UNSUPPORTED,
            FacetProminence.NOT_APPLICABLE,
        ),
        assessment(
            "F2",
            VerificationRelation.UNSUPPORTED,
            FacetProminence.NOT_APPLICABLE,
        ),
    ),
)
assert r.score == 0


# Score 1.
s = make_spec(
    "T1",
    [
        ("inequality", "core"),
        ("poverty", "core"),
        ("wealth distribution", "core"),
    ],
)

r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.ADJACENT,
            FacetProminence.NOT_APPLICABLE,
        ),
        assessment(
            "F2",
            VerificationRelation.UNSUPPORTED,
            FacetProminence.NOT_APPLICABLE,
        ),
        assessment(
            "F3",
            VerificationRelation.UNSUPPORTED,
            FacetProminence.NOT_APPLICABLE,
        ),
    ),
)
assert r.score == 1


# Score 2.
r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.DIRECT,
            FacetProminence.SUBSTANTIVE,
        ),
        assessment(
            "F2",
            VerificationRelation.UNSUPPORTED,
            FacetProminence.NOT_APPLICABLE,
        ),
        assessment(
            "F3",
            VerificationRelation.UNSUPPORTED,
            FacetProminence.NOT_APPLICABLE,
        ),
    ),
)
assert r.score == 2


# Score 3 via 2/3 coverage.
s = make_spec(
    "T3",
    [
        ("suspense", "core"),
        ("crime", "core"),
        ("investigation", "core"),
    ],
)

r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.DIRECT,
            FacetProminence.CENTRAL,
        ),
        assessment(
            "F2",
            VerificationRelation.DIRECT,
            FacetProminence.SUBSTANTIVE,
        ),
        assessment(
            "F3",
            VerificationRelation.UNSUPPORTED,
            FacetProminence.NOT_APPLICABLE,
        ),
    ),
)
assert r.score == 3
assert r.clear_rule_applied == "two_thirds_core_coverage"


# Score 4.
r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.DIRECT,
            FacetProminence.CENTRAL,
        ),
        assessment(
            "F2",
            VerificationRelation.DIRECT,
            FacetProminence.CENTRAL,
        ),
        assessment(
            "F3",
            VerificationRelation.DIRECT,
            FacetProminence.CENTRAL,
        ),
    ),
)
assert r.score == 4


# Entailed + central remains meaningful, never strong.
r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.DIRECT,
            FacetProminence.CENTRAL,
        ),
        assessment(
            "F2",
            VerificationRelation.ENTAILED,
            FacetProminence.CENTRAL,
        ),
        assessment(
            "F3",
            VerificationRelation.DIRECT,
            FacetProminence.CENTRAL,
        ),
    ),
)
assert r.score == 3
assert r.strong_core_count == 2


# v0.16: strong + absent on a two-core query remains partial.
s = make_spec(
    "T_TWO",
    [
        ("forgiveness", "core"),
        ("redemption", "core"),
    ],
)

r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.UNSUPPORTED,
            FacetProminence.NOT_APPLICABLE,
        ),
        assessment(
            "F2",
            VerificationRelation.DIRECT,
            FacetProminence.CENTRAL,
        ),
    ),
)
assert r.score == 2
assert r.clear_rule_applied is None


# v0.16: strong + incidental preserves the guarded two-core clear exception.
r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.ADJACENT,
            FacetProminence.NOT_APPLICABLE,
        ),
        assessment(
            "F2",
            VerificationRelation.DIRECT,
            FacetProminence.CENTRAL,
        ),
    ),
)
assert r.score == 3
assert r.clear_rule_applied == "two_core_one_strong_no_absent"


# One strong facet of three remains partial.
s = make_spec(
    "T_THREE",
    [
        ("leadership", "core"),
        ("teamwork", "core"),
        ("building effective organizations", "core"),
    ],
)

r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.UNSUPPORTED,
            FacetProminence.NOT_APPLICABLE,
        ),
        assessment(
            "F2",
            VerificationRelation.UNSUPPORTED,
            FacetProminence.NOT_APPLICABLE,
        ),
        assessment(
            "F3",
            VerificationRelation.DIRECT,
            FacetProminence.CENTRAL,
        ),
    ),
)
assert r.score == 2


# Meaningful qualifier permits clear relevance but prevents a perfect 4.
s = make_spec(
    "T_QUAL",
    [
        ("astronomy and the universe", "core"),
        ("beginner-friendly explanation", "qualifier"),
    ],
)

r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.DIRECT,
            FacetProminence.CENTRAL,
        ),
        assessment(
            "F2",
            VerificationRelation.ENTAILED,
            FacetProminence.NOT_APPLICABLE,
        ),
    ),
)
assert r.score == 3
assert r.qualifier_cap_applied
assert r.clear_rule_applied == "qualifier_meaningful_cap_from_strong"


# Unsupported qualifier caps a clear/strong result to partial.
r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.DIRECT,
            FacetProminence.CENTRAL,
        ),
        assessment(
            "F2",
            VerificationRelation.UNSUPPORTED,
            FacetProminence.NOT_APPLICABLE,
        ),
    ),
)
assert r.score == 2
assert r.qualifier_cap_applied
assert r.clear_rule_applied == "qualifier_absent_cap_to_partial"


# Single composite facet.
s = make_spec(
    "T_COMPOSITE",
    [
        ("complicated parent-child relationship", "core"),
    ],
)

r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.ENTAILED,
            FacetProminence.CENTRAL,
        ),
    ),
)
assert r.score == 3

r = compute_facet_score(
    s,
    verdict(
        assessment(
            "F1",
            VerificationRelation.DIRECT,
            FacetProminence.CENTRAL,
        ),
    ),
)
assert r.score == 4



# v0.11 qualifier behavior: absent required qualifier caps clear -> partial.
s = QueryFacetSpec(
    query_id="T_QUAL_ABSENT",
    query="beginner astronomy",
    facets=[
        QueryFacet(
            facet_id="F1",
            text="astronomy",
            facet_type="core",
            semantic_definition="Astronomy subject matter.",
        ),
        QueryFacet(
            facet_id="F2",
            text="beginner-friendly",
            facet_type="qualifier",
            semantic_definition="Accessible to beginners.",
        ),
    ],
)
r = compute_facet_score(
    s,
    verdict(
        assessment("F1", VerificationRelation.DIRECT, FacetProminence.CENTRAL),
        assessment("F2", VerificationRelation.UNSUPPORTED, FacetProminence.NOT_APPLICABLE),
    ),
)
assert r.score == 2
assert r.qualifier_cap_applied
assert r.clear_rule_applied == "qualifier_absent_cap_to_partial"

# Meaningful qualifier permits clear but caps a perfect core match from 4 -> 3.
r = compute_facet_score(
    s,
    verdict(
        assessment("F1", VerificationRelation.DIRECT, FacetProminence.CENTRAL),
        assessment("F2", VerificationRelation.ENTAILED, FacetProminence.NOT_APPLICABLE),
    ),
)
assert r.score == 3
assert r.qualifier_cap_applied
assert r.clear_rule_applied == "qualifier_meaningful_cap_from_strong"

# Direct qualifier + direct/central core allows 4.
r = compute_facet_score(
    s,
    verdict(
        assessment("F1", VerificationRelation.DIRECT, FacetProminence.CENTRAL),
        assessment("F2", VerificationRelation.DIRECT, FacetProminence.NOT_APPLICABLE),
    ),
)
assert r.score == 4


print("All v0.21.0 facet scoring tests passed.")
