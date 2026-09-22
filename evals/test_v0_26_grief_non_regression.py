"""v0.26.0 r3: local negative boundaries must not globally suppress valid grief entailment."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    IsolatedComponentVerification,
    build_isolated_component_prompt,
    verify_candidate_evidence,
)
from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec, VerificationRelation

E = Path(__file__).resolve().parent
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.26.0.json").read_text())
FACETS = json.loads((E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.1.json").read_text())


def facet(query_id: str, facet_id: str) -> QueryFacet:
    raw = next(q for q in FACETS["queries"] if q["query_id"] == query_id)
    spec = QueryFacetSpec(
        query_id=raw["query_id"],
        query=raw["query"],
        facets=[QueryFacet(**f) for f in raw["facets"]],
    )
    return next(f for f in spec.facets if f.facet_id == facet_id)


class QueueModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt, schema):
        self.prompts.append(prompt)
        assert schema is IsolatedComponentVerification
        return self.outputs.pop(0)


grief = facet("Q06", "F1")
assert [c.component_id for c in grief.required_components] == [
    "significant_loss",
    "mourning_or_deep_sorrow_response",
]

# r3 removes the r2 generic precedence wording from unrelated component calls.
for component in grief.required_components:
    prompt = build_isolated_component_prompt(
        grief,
        component,
        "S8",
        "Suddenly devastation tears at the heart of their family, and the depth of their existence.",
    )
    assert "NEGATIVE-BOUNDARY PRIORITY" not in prompt
    assert "important or consequential personal decision" not in prompt
    # The component's own frozen negative boundary is still present.
    for boundary in component.negative_boundaries:
        assert boundary in prompt

# The deterministic facet assembler must continue to accept independently grounded
# significant-loss + deep-sorrow components from the same evidence span.
model = QueueModel([
    IsolatedComponentVerification(
        component_id="significant_loss",
        grounding_relation="entailed",
        negative_boundary_applied=False,
        reason="The described devastation to the heart of the family entails a significant personal loss.",
    ),
    IsolatedComponentVerification(
        component_id="mourning_or_deep_sorrow_response",
        grounding_relation="entailed",
        negative_boundary_applied=False,
        reason="The language about devastation tearing at the heart entails profound sorrow connected to that loss.",
    ),
])
result, retries = verify_candidate_evidence(
    model,
    grief,
    "S8",
    "Suddenly devastation tears at the heart of their family, and the depth of their existence.",
    CONFIG,
)
assert retries == 0
assert result.verification_relation == VerificationRelation.ENTAILED
assert result.semantic_definition_exclusion_applied is False
assert all(check.established for check in result.component_checks)

print("v0.26.0 r3 grief non-regression contract passed")
