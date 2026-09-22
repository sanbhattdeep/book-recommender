import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    JudgeOutputValidationError,
    ProminenceAssessment,
    _context_role_from_signals,
    _prominence_from_role_signals,
    _validate_prominence_result,
)
from semantic_relevance_facet_scoring import FacetContextRole, FacetProminence


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "evals/judge_configs/semantic_relevance_judge.v0.21.0.json").read_text(encoding="utf-8")
)
assert CONFIG["prominence_stage"]["role_precedence"]


def result(**signals: bool) -> ProminenceAssessment:
    defaults = {
        "is_primary_subject": False,
        "is_background_cause_or_factor": False,
        "is_example_or_illustration": False,
        "is_meta_discussion": False,
        "is_substantively_examined": False,
    }
    defaults.update(signals)
    return ProminenceAssessment(
        primary_subject_summary="synthetic primary subject",
        supporting_span_ids=["S1"],
        reason="synthetic grounded role judgment",
        **defaults,
    )


cases = [
    (
        result(is_meta_discussion=True, is_primary_subject=True),
        FacetContextRole.META_DISCUSSION,
        FacetProminence.INCIDENTAL,
    ),
    (
        result(is_example_or_illustration=True, is_primary_subject=True),
        FacetContextRole.EXAMPLE_OR_ILLUSTRATION,
        FacetProminence.INCIDENTAL,
    ),
    (
        result(
            is_background_cause_or_factor=True,
            is_primary_subject=True,
            is_substantively_examined=True,
        ),
        FacetContextRole.BACKGROUND_CAUSE,
        FacetProminence.INCIDENTAL,
    ),
    (
        result(is_primary_subject=True),
        FacetContextRole.CENTRAL_SUBJECT,
        FacetProminence.CENTRAL,
    ),
    (
        result(is_substantively_examined=True),
        FacetContextRole.SUBSTANTIVE_SUBJECT,
        FacetProminence.SUBSTANTIVE,
    ),
    (
        result(),
        FacetContextRole.INCIDENTAL_MENTION,
        FacetProminence.INCIDENTAL,
    ),
]

for item, expected_role, expected_prominence in cases:
    _validate_prominence_result(item, {"S1": "source"})
    assert _context_role_from_signals(item) == expected_role
    assert _prominence_from_role_signals(item) == expected_prominence

bad = result()
bad.supporting_span_ids = ["UNKNOWN"]
try:
    _validate_prominence_result(bad, {"S1": "source"})
    raise AssertionError("expected unknown-span validation failure")
except JudgeOutputValidationError:
    pass

print("v0.21 decomposed role-signal contract tests passed.")
