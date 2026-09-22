import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    FullDescriptionHardExclusionAssessment,
    JudgeOutputValidationError,
    _validate_full_description_hard_exclusion_result,
    evaluate_one_facet,
)
from semantic_relevance_facet_scoring import (
    HardExclusion,
    QueryFacet,
    VerificationRelation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "evals/judge_configs/semantic_relevance_judge.v0.21.0.json").read_text(encoding="utf-8")
)
FACET = QueryFacet(
    facet_id="F1",
    text="adventure",
    facet_type="core",
    semantic_definition="A journey or quest as story content.",
    hard_exclusions=[
        HardExclusion(
            exclusion_id="HX_META",
            rule="A quest used only as a literary-analysis example is not adventure content.",
        )
    ],
)
SPANS = {"S1": "The guide uses a quest as an example of literary symbolism."}

triggered = FullDescriptionHardExclusionAssessment(
    hard_exclusion_triggered=True,
    hard_exclusion_id="HX_META",
    supporting_span_ids=["S1"],
    reason="The only connection is a meta-literary example.",
)
_validate_full_description_hard_exclusion_result(triggered, FACET, SPANS)

for bad in [
    FullDescriptionHardExclusionAssessment(
        hard_exclusion_triggered=True,
        hard_exclusion_id="UNKNOWN",
        supporting_span_ids=["S1"],
        reason="bad id",
    ),
    FullDescriptionHardExclusionAssessment(
        hard_exclusion_triggered=False,
        hard_exclusion_id=None,
        supporting_span_ids=["S1"],
        reason="bad non-triggered evidence",
    ),
]:
    try:
        _validate_full_description_hard_exclusion_result(bad, FACET, SPANS)
        raise AssertionError("expected precheck validation failure")
    except JudgeOutputValidationError:
        pass


class TriggeringModel:
    def __init__(self) -> None:
        self.schemas = []

    def generate(self, prompt, schema):
        self.schemas.append(schema.__name__)
        assert "FULL NUMBERED BOOK-DESCRIPTION SPANS" in prompt
        return triggered


model = TriggeringModel()
assessment, evidence, candidate_texts, retries = evaluate_one_facet(
    judge_model=model,
    query_id="QX",
    facet=FACET,
    spans=SPANS,
    config=CONFIG,
)
assert model.schemas == ["FullDescriptionHardExclusionAssessment"]
assert assessment.hard_exclusion_precheck_triggered is True
assert assessment.verification_relation == VerificationRelation.UNSUPPORTED
assert assessment.candidate_evidence_span_ids == []
assert evidence is None
assert candidate_texts == SPANS
assert retries == 0

print("v0.21 full-description hard-exclusion precheck tests passed.")
