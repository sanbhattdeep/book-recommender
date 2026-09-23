"""v0.26.0 r6: Q02/F1 locally separates decision uncertainty from suspense."""
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
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.26.0.json").read_text(encoding="utf-8"))
FACETS = json.loads((E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.3.json").read_text(encoding="utf-8"))


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



# r6 preserves all canonical IDs from v0.9.1. Q02/F1 remains identical; only Q03/F1 and Q07/F1 change semantically.
OLD_FACETS = json.loads((E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.1.json").read_text(encoding="utf-8"))

def facet_map(payload):
    return {
        (q["query_id"], f["facet_id"]): f
        for q in payload["queries"]
        for f in q["facets"]
    }

old_map = facet_map(OLD_FACETS)
new_map = facet_map(FACETS)
assert old_map.keys() == new_map.keys()
for key in old_map:
    old_ids = [c["component_id"] for c in old_map[key]["required_components"]]
    new_ids = [c["component_id"] for c in new_map[key]["required_components"]]
    assert old_ids == new_ids, key
    if key not in {("Q03", "F1"), ("Q07", "F1")}:
        assert old_map[key] == new_map[key], key
assert old_map[("Q02", "F1")] == new_map[("Q02", "F1")]

suspense = facet("Q02", "F1")
component = suspense.required_components[0]
assert component.component_id == "story_level_tension_or_anticipation"

joined_boundaries = " ".join(component.negative_boundaries).lower()
for required_phrase in [
    "important, consequential",
    "caregiving",
    "decision uncertainty",
    "decision driving the plot",
    "independently tension-producing",
]:
    assert required_phrase in joined_boundaries, required_phrase

# The production prompt must surface the frozen Q02/F1 local boundary, without
# imposing the r2 suspense example as a global verifier instruction.
prompt = build_isolated_component_prompt(
    suspense,
    component,
    "S1",
    "She must decide whether to attend the ceremony or stay with her sick sister.",
)
assert "decision uncertainty" in prompt
assert "independently tension-producing" in prompt
assert "NEGATIVE-BOUNDARY PRIORITY" not in prompt
assert "For example, an important or consequential personal decision" not in prompt

# Negative fixture 1: important caregiving decision is not suspense by itself.
model = QueueModel([
    IsolatedComponentVerification(
        component_id="story_level_tension_or_anticipation",
        grounding_relation="missing",
        negative_boundary_applied=True,
        reason="The evidence is only a consequential caregiving choice; no independent narrative suspense is established.",
    )
])
result, retries = verify_candidate_evidence(
    model,
    suspense,
    "S1",
    "She must decide whether to attend the ceremony or stay with her sick sister.",
    CONFIG,
)
assert retries == 0
assert result.verification_relation == VerificationRelation.UNSUPPORTED
assert result.semantic_definition_exclusion_applied is True

# Negative fixture 2: even severe consequences do not turn a choice into suspense.
model = QueueModel([
    IsolatedComponentVerification(
        component_id="story_level_tension_or_anticipation",
        grounding_relation="missing",
        negative_boundary_applied=True,
        reason="A consequential confession dilemma is still decision uncertainty without an independent tension-producing event.",
    )
])
result, _ = verify_candidate_evidence(
    model,
    suspense,
    "S1",
    "He must choose whether to confess, knowing the decision could destroy his family.",
    CONFIG,
)
assert result.verification_relation == VerificationRelation.UNSUPPORTED

# Positive fixture: the independent threat supplies narrative anticipation beyond the choice.
model = QueueModel([
    IsolatedComponentVerification(
        component_id="story_level_tension_or_anticipation",
        grounding_relation="entailed",
        negative_boundary_applied=False,
        reason="Someone silently turning the handle from the other side creates an unfolding threat independent of the decision to open the door.",
    )
])
result, _ = verify_candidate_evidence(
    model,
    suspense,
    "S1",
    "She must decide whether to open the door while someone silently turns the handle from the other side.",
    CONFIG,
)
assert result.verification_relation == VerificationRelation.ENTAILED
assert result.semantic_definition_exclusion_applied is False

print("v0.26.0 r6 local suspense decision-boundary contract passed")
