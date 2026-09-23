"""Deterministic contracts for the v0.27 text-licensed entailment boundary."""
from pathlib import Path
import sys

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))

from semantic_relevance_facet_judge import (
    FullContextComponentRecovery,
    IsolatedComponentVerification,
    _validate_component_recovery_result,
    _validate_isolated_component_result,
    build_isolated_component_prompt,
    build_missing_component_recovery_prompt,
    JudgeOutputValidationError,
)
from semantic_relevance_facet_scoring import QueryFacet, FacetRequiredComponent

component = FacetRequiredComponent(
    component_id="target",
    definition="The target component is established.",
    negative_boundaries=[],
)
facet = QueryFacet(
    facet_id="F1",
    text="test facet",
    facet_type="core",
    semantic_definition="Test definition.",
    hard_exclusions=[],
    required_components=[component],
)

prompt = build_isolated_component_prompt(facet, component, "S1", "A travel account describes a subject across several places.")
for phrase in (
    "TEXT-LICENSED INFERENCE POLICY",
    "external_knowledge_required",
    "Named entities do NOT import",
    "travel account",
):
    assert phrase in prompt, phrase

bad = IsolatedComponentVerification(
    component_id="target",
    grounding_relation="entailed",
    external_knowledge_required=True,
    reason="Would require outside entity history.",
)
try:
    _validate_isolated_component_result(bad, component, "test")
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("positive isolated grounding using external knowledge must be rejected")

ok = IsolatedComponentVerification(
    component_id="target",
    grounding_relation="missing",
    external_knowledge_required=True,
    reason="Text does not state the named entity's target.",
)
_validate_isolated_component_result(ok, component, "test")

recovery_prompt = build_missing_component_recovery_prompt(facet, component, {"S1":"Named movement membership is stated."}, {})
assert "external_knowledge_required" in recovery_prompt
assert "outside/entity-specific knowledge" in recovery_prompt

bad_recovery = FullContextComponentRecovery(
    component_id="target",
    grounding_relation="entailed",
    supporting_span_ids=["S1"],
    external_knowledge_required=True,
    reason="Would import outside history.",
)
try:
    _validate_component_recovery_result(bad_recovery, component, {"S1":"x"}, {})
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("positive recovery using external knowledge must be rejected")

print("v0.27 text-licensed inference and external-knowledge contracts passed")
