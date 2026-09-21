"""Regression: impossible hard-exclusion flags must not abort facets with no exclusions."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    ComponentEvidenceCheck,
    CompositeEvidenceVerification,
    EvidenceVerification,
    verify_candidate_evidence,
    verify_composite_evidence,
)
from semantic_relevance_facet_scoring import EvidenceInferenceKind, QueryFacet, VerificationRelation

E = Path(__file__).resolve().parent
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.26.0.json").read_text(encoding="utf-8"))

facet = QueryFacet(
    facet_id="F1",
    text="personal growth",
    facet_type="core",
    semantic_definition="Meaningful self-development; generic relaxation alone does not establish it.",
)
assert facet.hard_exclusions == []

class SingleModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, schema, **kwargs):
        self.calls += 1
        assert "hard_exclusion_triggered MUST be false" in prompt
        return EvidenceVerification(
            verification_relation=VerificationRelation.UNSUPPORTED,
            inference_kind=EvidenceInferenceKind.NONE,
            component_checks=[ComponentEvidenceCheck(component_id="actual development or change", established=False, supporting_span_ids=[], reason="missing")],
            all_required_components_established=False,
            missing_semantic_component="actual development or change",
            semantic_definition_exclusion_applied=True,
            hard_exclusion_triggered=True,
            hard_exclusion_id="HX_HALLUCINATED",
            reason="Relaxation is explicitly insufficient for personal growth.",
        )

single_model = SingleModel()
single, retries = verify_candidate_evidence(
    judge_model=single_model,
    facet=facet,
    evidence_span_id="S1",
    evidence_text="A meditation guide offering relaxation and inner peace.",
    config=CONFIG,
)
assert single_model.calls == 1
assert retries == 0
assert single.verification_relation == VerificationRelation.UNSUPPORTED
assert single.semantic_definition_exclusion_applied is True
assert single.hard_exclusion_triggered is False
assert single.hard_exclusion_id is None

class CompositeModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, schema, **kwargs):
        self.calls += 1
        return CompositeEvidenceVerification(
            supporting_span_ids=[],
            verification_relation=VerificationRelation.UNSUPPORTED,
            inference_kind=EvidenceInferenceKind.NONE,
            component_checks=[ComponentEvidenceCheck(component_id="actual development or change", established=False, supporting_span_ids=[], reason="missing")],
            all_required_components_established=False,
            semantic_definition_exclusion_applied=True,
            hard_exclusion_triggered=True,
            hard_exclusion_id="HX_HALLUCINATED",
            combined_evidence_summary="The spans establish relaxation but not actual development.",
            missing_semantic_component="actual development or change",
            reason="The frozen definition excludes generic relaxation as sufficient evidence.",
        )

composite_model = CompositeModel()
composite, retries = verify_composite_evidence(
    judge_model=composite_model,
    facet=facet,
    spans={"S1": "A meditation guide.", "S2": "It offers relaxation and inner peace."},
    config=CONFIG,
)
assert composite_model.calls == 1
assert retries == 0
assert composite.verification_relation == VerificationRelation.UNSUPPORTED
assert composite.semantic_definition_exclusion_applied is True
assert composite.hard_exclusion_triggered is False
assert composite.hard_exclusion_id is None
print("v0.24.0 no-hard-exclusion normalization passed")
