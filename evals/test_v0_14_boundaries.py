"""v0.14-specific boundary and aggregation contract tests. No LLM is called."""

from __future__ import annotations

import json
from pathlib import Path

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
    / "semantic_relevance_query_facets.v0.4.0.json"
)


def a(fid: str, relation: VerificationRelation, prominence: FacetProminence):
    if relation == VerificationRelation.UNSUPPORTED:
        candidates = []
        records = []
        winner = "NONE"
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
        winner = "S1"

    return FacetPipelineAssessment(
        facet_id=fid,
        candidate_evidence_span_ids=candidates,
        candidate_verifications=records,
        candidate_evidence_span_id=winner,
        verification_relation=relation,
        prominence=prominence,
        evidence_selection_reason="synthetic",
        verification_reason="synthetic",
        prominence_reason=(
            "synthetic"
            if prominence != FacetProminence.NOT_APPLICABLE
            else None
        ),
    )


spec = QueryFacetSpec(
    query_id="SYN_TWO",
    query="two-part request",
    facets=[
        QueryFacet(
            facet_id="F1",
            text="first",
            facet_type="core",
            semantic_definition="first concept",
        ),
        QueryFacet(
            facet_id="F2",
            text="second",
            facet_type="core",
            semantic_definition="second concept",
        ),
    ],
)

# Regression that motivated the aggregation fix: strong + absent is only partial.
r = compute_facet_score(
    spec,
    FacetJudgeVerdict(
        assessments=[
            a("F1", VerificationRelation.DIRECT, FacetProminence.CENTRAL),
            a("F2", VerificationRelation.UNSUPPORTED, FacetProminence.NOT_APPLICABLE),
        ],
        overall_reason="synthetic",
    ),
)
assert r.score == 2
assert r.clear_rule_applied is None

# Weak-but-real second-facet support keeps the historical two-core exception.
r = compute_facet_score(
    spec,
    FacetJudgeVerdict(
        assessments=[
            a("F1", VerificationRelation.DIRECT, FacetProminence.CENTRAL),
            a("F2", VerificationRelation.ADJACENT, FacetProminence.NOT_APPLICABLE),
        ],
        overall_reason="synthetic",
    ),
)
assert r.score == 3
assert r.clear_rule_applied == "two_core_one_strong_no_absent"

payload = json.loads(FACET_FILE.read_text(encoding="utf-8"))
q03 = next(q for q in payload["queries"] if q["query_id"] == "Q03")
defs = {f["facet_id"]: f["semantic_definition"].lower() for f in q03["facets"]}

assert "does not by itself establish personal growth" in defs["F1"]
assert "actual development" in defs["F1"]
assert "deity" in defs["F2"]
assert "does not by itself establish finding purpose" in defs["F2"]
assert "meaning, direction, vocation, values, goals, or life direction" in defs["F2"]

print("All v0.14.0 boundary/aggregation contract tests passed.")
