"""
Unit tests for v0.20.0 deterministic multi-candidate resolution.

No LLM is called.
"""

from semantic_relevance_facet_judge import (
    choose_best_verified_candidate,
)
from semantic_relevance_facet_scoring import (
    CandidateVerificationRecord,
    VerificationRelation,
)


def record(
    rank: int,
    span_id: str,
    relation: VerificationRelation,
) -> CandidateVerificationRecord:
    return CandidateVerificationRecord(
        candidate_rank=rank,
        evidence_span_id=span_id,
        verification_relation=relation,
        verification_reason="synthetic",
    )


# Later DIRECT beats earlier ENTAILED.
best = choose_best_verified_candidate(
    [
        record(1, "S1", VerificationRelation.ENTAILED),
        record(2, "S2", VerificationRelation.DIRECT),
        record(3, "S3", VerificationRelation.UNSUPPORTED),
    ]
)
assert best.evidence_span_id == "S2"


# Earlier rank wins when relation strength ties.
best = choose_best_verified_candidate(
    [
        record(1, "S4", VerificationRelation.ENTAILED),
        record(2, "S5", VerificationRelation.ENTAILED),
    ]
)
assert best.evidence_span_id == "S4"


# ENTAILED beats any number of unsupported candidates.
best = choose_best_verified_candidate(
    [
        record(1, "S1", VerificationRelation.UNSUPPORTED),
        record(2, "S2", VerificationRelation.ENTAILED),
        record(3, "S3", VerificationRelation.UNSUPPORTED),
    ]
)
assert best.evidence_span_id == "S2"


# When all candidates are unsupported, selector rank is the tie-breaker.
best = choose_best_verified_candidate(
    [
        record(1, "S7", VerificationRelation.UNSUPPORTED),
        record(2, "S8", VerificationRelation.UNSUPPORTED),
    ]
)
assert best.evidence_span_id == "S7"


print("All v0.20.0 candidate-resolution tests passed.")


# ENTAILED beats ADJACENT.
best = choose_best_verified_candidate(
    [
        record(1, "S1", VerificationRelation.ADJACENT),
        record(2, "S2", VerificationRelation.ENTAILED),
    ]
)
assert best.evidence_span_id == "S2"

# ADJACENT beats UNSUPPORTED.
best = choose_best_verified_candidate(
    [
        record(1, "S1", VerificationRelation.UNSUPPORTED),
        record(2, "S2", VerificationRelation.ADJACENT),
    ]
)
assert best.evidence_span_id == "S2"
