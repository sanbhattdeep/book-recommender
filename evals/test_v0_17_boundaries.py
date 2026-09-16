"""
Deterministic contract tests for v0.17.0.

No LLM is called.
"""

from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    build_verification_prompt,
)
from semantic_relevance_facet_scoring import (
    CandidateVerificationRecord,
    FacetJudgeVerdict,
    FacetPipelineAssessment,
    FacetProminence,
    QueryFacet,
    QueryFacetSpec,
    VerificationRelation,
    compute_facet_score,
)


EVALS_DIR = Path(__file__).resolve().parent

FACET_FILE = (
    EVALS_DIR
    / "facets"
    / "semantic_relevance"
    / "semantic_relevance_query_facets.v0.6.0.json"
)

CONFIG_FILE = (
    EVALS_DIR
    / "judge_configs"
    / "semantic_relevance_judge.v0.17.0.json"
)


def load_json(
    path: Path,
) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


facets_payload = load_json(
    FACET_FILE
)

config = load_json(
    CONFIG_FILE
)

assert (
    facets_payload[
        "version"
    ]
    == "0.6.0"
)

assert (
    config[
        "version"
    ]
    == "0.17.0"
)

assert (
    config[
        "facet_spec_version"
    ]
    == "0.6.0"
)

q03 = next(
    query
    for query
    in facets_payload[
        "queries"
    ]
    if query[
        "query_id"
    ] == "Q03"
)

f1_raw = next(
    facet
    for facet
    in q03[
        "facets"
    ]
    if facet[
        "facet_id"
    ] == "F1"
)

f2_raw = next(
    facet
    for facet
    in q03[
        "facets"
    ]
    if facet[
        "facet_id"
    ] == "F2"
)

f1_definition = (
    f1_raw[
        "semantic_definition"
    ]
)

f2_definition = (
    f2_raw[
        "semantic_definition"
    ]
)

assert (
    "does not establish personal growth"
    in f1_definition
)

assert (
    "not self-understanding for this facet"
    in f1_definition
)

assert (
    "exact evidence independently states or necessarily entails"
    in f1_definition
)

assert (
    "could plausibly lead to it"
    in f1_definition
)

assert (
    "does not by itself establish finding purpose"
    in f2_definition
)

assert (
    "exact evidence must independently connect"
    in f2_definition
)

assert (
    "vocation or calling"
    in f2_definition
)

assert (
    "could plausibly lead to meaning or direction"
    in f2_definition
)

# Candidate-specific leakage is prohibited.
serialized_artifacts = json.dumps(
    {
        "facets": facets_payload,
        "config": config,
    },
    ensure_ascii=False,
).lower()

for forbidden in [
    "u_q03_t70",
    "courageous faith through the year",
    "learned optimism",
]:
    assert (
        forbidden
        not in serialized_artifacts
    )

verification_instructions = "\n".join(
    config[
        "verification_stage"
    ][
        "instructions"
    ]
)

assert (
    "hard gates"
    in verification_instructions
)

assert (
    "plausible downstream benefit"
    in verification_instructions
)

assert (
    "definition supplies meaning and boundaries; it is not evidence"
    in verification_instructions
)

assert (
    "NECESSARY inference from PLAUSIBLE inference"
    in verification_instructions
)

entailed_definition = (
    config[
        "verification_stage"
    ][
        "relation_scale"
    ][
        "entailed"
    ]
)

assert (
    "plausible consequence or association is not entailment"
    in entailed_definition
)

# Prompt contract for the semantic pattern that exposed the problem.
f2 = QueryFacet(
    **f2_raw
)

prompt = build_verification_prompt(
    facet=f2,
    evidence_text=(
        "This devotional shows believers how to know God more deeply "
        "and discover how he works in them."
    ),
    config=config,
)

assert (
    "does not by itself establish finding purpose"
    in prompt
)

assert (
    "exact evidence must independently connect"
    in prompt
)

assert (
    "hard gates"
    in prompt
)

assert (
    "plausible downstream benefit"
    in prompt
)

assert (
    "The semantic definition supplies meaning and boundaries; it is not evidence."
    in prompt
)


def assessment(
    facet_id: str,
    relation: VerificationRelation,
    prominence: FacetProminence,
) -> FacetPipelineAssessment:

    if (
        relation
        == VerificationRelation.UNSUPPORTED
    ):
        candidates = []
        records = []
        winning = "NONE"

    else:
        candidates = [
            "S1"
        ]

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
            if (
                prominence
                != FacetProminence.NOT_APPLICABLE
            )
            else None
        ),
    )


spec = QueryFacetSpec(
    query_id="SYN",
    query="synthetic",
    facets=[
        QueryFacet(
            facet_id="F1",
            text="first",
            facet_type="core",
            semantic_definition="first",
        ),
        QueryFacet(
            facet_id="F2",
            text="second",
            facet_type="core",
            semantic_definition="second",
        ),
    ],
)

# v0.14 aggregation invariant: strong + absent -> 2.
result = compute_facet_score(
    spec,
    FacetJudgeVerdict(
        assessments=[
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
        ],
        overall_reason="synthetic",
    ),
)

assert (
    result.score
    == 2
)

# v0.14 aggregation invariant: strong + incidental -> 3.
result = compute_facet_score(
    spec,
    FacetJudgeVerdict(
        assessments=[
            assessment(
                "F1",
                VerificationRelation.DIRECT,
                FacetProminence.CENTRAL,
            ),
            assessment(
                "F2",
                VerificationRelation.ENTAILED,
                FacetProminence.INCIDENTAL,
            ),
        ],
        overall_reason="synthetic",
    ),
)

assert (
    result.score
    == 3
)

assert (
    result.clear_rule_applied
    == "two_core_one_strong_no_absent"
)

# v0.17 preserves the v0.16 Q02 suspense boundary: ordinary dilemmas/high stakes are insufficient.
q02 = next(q for q in facets_payload["queries"] if q["query_id"] == "Q02")
q02_f1 = next(f for f in q02["facets"] if f["facet_id"] == "F1")
q02_definition = q02_f1["semantic_definition"]
assert "difficult personal choice" in q02_definition
assert "generic uncertainty does not by itself establish suspense" in q02_definition
assert "story-level tension or anticipation" in q02_definition

print("All v0.17.0 boundary/verifier/aggregation contract tests passed.")
