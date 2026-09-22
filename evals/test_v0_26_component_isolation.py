"""v0.26 verifies each canonical component in its own model call."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    IsolatedComponentVerification,
    verify_candidate_evidence,
)
from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec, VerificationRelation

E = Path(__file__).resolve().parent
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.26.0.json").read_text())
FACETS = json.loads((E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.1.json").read_text())


def facet(query_id: str, facet_id: str) -> QueryFacet:
    raw = next(q for q in FACETS["queries"] if q["query_id"] == query_id)
    spec = QueryFacetSpec(query_id=raw["query_id"], query=raw["query"], facets=[QueryFacet(**f) for f in raw["facets"]])
    return next(f for f in spec.facets if f.facet_id == facet_id)


class QueueModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.schemas: list[str] = []

    def generate(self, prompt, schema):
        self.prompts.append(prompt)
        self.schemas.append(schema.__name__)
        assert schema is IsolatedComponentVerification
        return self.outputs.pop(0)


# Q07: complex marriage must not become parent-child by analogy.
q07 = facet("Q07", "F1")
model = QueueModel([
    IsolatedComponentVerification(
        component_id="parent_child_relationship",
        grounding_relation="missing",
        negative_boundary_applied=True,
        reason="The span explicitly says forty-year marriage, not parent-child.",
    ),
    IsolatedComponentVerification(
        component_id="difficult_or_complex_dynamic",
        grounding_relation="explicit",
        reason="Turbulence and emotional fallout directly establish complexity.",
    ),
])
result, retries = verify_candidate_evidence(
    model,
    q07,
    "S1",
    "The story of a forty-year marriage follows the turbulence and joys shared by Henry and Beatrice.",
    CONFIG,
)
assert retries == 0
assert result.verification_relation == VerificationRelation.UNSUPPORTED
assert result.semantic_definition_exclusion_applied is True
assert len(model.prompts) == 2
assert "parent_child_relationship" in model.prompts[0]
assert "difficult_or_complex_dynamic" not in model.prompts[0]
assert "difficult_or_complex_dynamic" in model.prompts[1]
assert "parent_child_relationship" not in model.prompts[1]

# Q04/F3: danger may be explicit while movement stays blocked.
q04 = facet("Q04", "F3")
model = QueueModel([
    IsolatedComponentVerification(
        component_id="movement_or_travel",
        grounding_relation="missing",
        negative_boundary_applied=True,
        reason="Being trapped while hit-men are on her trail does not establish her travel.",
    ),
    IsolatedComponentVerification(
        component_id="danger_or_threat",
        grounding_relation="explicit",
        reason="Two hit-men on her trail directly establish threat.",
    ),
])
result, _ = verify_candidate_evidence(
    model,
    q04,
    "S2",
    "Trapped in the Canadian backwoods, with two hit-men on her trail, only one man can save her.",
    CONFIG,
)
assert result.verification_relation == VerificationRelation.UNSUPPORTED
assert result.component_checks[0].component_id == "movement_or_travel"
assert result.component_checks[0].established is False
assert result.component_checks[1].established is True

# Q01/F2: generic recovery may ground a damaged state, but not restoration/atonement.
q01 = facet("Q01", "F2")
model = QueueModel([
    IsolatedComponentVerification(
        component_id="damaged_or_failed_state",
        grounding_relation="entailed",
        reason="Alcoholism is a damaged personal state in this context.",
    ),
    IsolatedComponentVerification(
        component_id="restoration_or_atonement",
        grounding_relation="missing",
        negative_boundary_applied=True,
        reason="Sobriety/recovery alone is explicitly excluded from this component.",
    ),
])
result, _ = verify_candidate_evidence(
    model,
    q01,
    "S4",
    "The program helps alcoholics get sober today.",
    CONFIG,
)
assert result.verification_relation == VerificationRelation.UNSUPPORTED
assert result.semantic_definition_exclusion_applied is True

print("v0.26.0 isolated component verification contract passed")
