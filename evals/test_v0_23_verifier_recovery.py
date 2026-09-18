"""v0.23.0 retries a structurally inconsistent positive verifier output conservatively."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import EvidenceVerification, verify_candidate_evidence
from semantic_relevance_facet_scoring import EvidenceInferenceKind, QueryFacet, VerificationRelation

E = Path(__file__).resolve().parent
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.23.0.json").read_text(encoding="utf-8"))

facet = QueryFacet(
    facet_id="F1",
    text="dangerous journeys",
    facet_type="core",
    semantic_definition="Dangerous travel; danger without movement does not establish the facet.",
)

class Model:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, schema, **kwargs):
        self.calls += 1
        if self.calls == 1:
            # Invalid positive: the model itself admits the movement component is missing.
            return EvidenceVerification(
                verification_relation=VerificationRelation.ENTAILED,
                inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
                all_required_components_established=False,
                missing_semantic_component="travel or movement",
                semantic_definition_exclusion_applied=False,
                reason="Danger is present but movement is absent.",
            )
        assert "PREVIOUS VALIDATION FAILURE" in prompt
        return EvidenceVerification(
            verification_relation=VerificationRelation.UNSUPPORTED,
            inference_kind=EvidenceInferenceKind.NONE,
            all_required_components_established=False,
            missing_semantic_component="travel or movement",
            semantic_definition_exclusion_applied=True,
            reason="The frozen definition explicitly says danger without travel is insufficient.",
        )

model = Model()
result, retries = verify_candidate_evidence(
    judge_model=model,
    facet=facet,
    evidence_text="Trapped in the backwoods with hit-men on her trail.",
    config=CONFIG,
)
assert model.calls == 2
assert retries == 1
assert result.verification_relation == VerificationRelation.UNSUPPORTED
assert result.semantic_definition_exclusion_applied is True
print("v0.23.0 verifier component-repair retry passed")
