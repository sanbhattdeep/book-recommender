"""Deterministic contracts for v0.27 r3 localized inference boundaries."""
from pathlib import Path
import json
import sys

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))

from semantic_relevance_facet_judge import (
    FullContextComponentRecovery,
    IsolatedComponentVerification,
    _q09_recovery_text_anchor_guard,
    _q09_text_anchor_guard,
    _validate_component_recovery_result,
    _validate_isolated_component_result,
    build_isolated_component_prompt,
    build_missing_component_recovery_prompt,
    component_local_inference_policy,
    JudgeOutputValidationError,
)
from semantic_relevance_facet_scoring import QueryFacet, FacetRequiredComponent


def make_facet(text: str, component_id: str, definition: str = "component"):
    component = FacetRequiredComponent(
        component_id=component_id,
        definition=definition,
        negative_boundaries=[],
    )
    facet = QueryFacet(
        facet_id="F1",
        text=text,
        facet_type="core",
        semantic_definition="Test definition.",
        hard_exclusions=[],
        required_components=[component],
    )
    return facet, component


# r1's global policy is intentionally gone: unrelated facets keep the r6 prompt surface.
default_facet, default_component = make_facet("unrelated facet", "target")
default_prompt = build_isolated_component_prompt(
    default_facet,
    default_component,
    "S1",
    "Some evidence.",
)
assert "LOCAL Q04" not in default_prompt
assert "LOCAL Q05" not in default_prompt
assert "LOCAL Q09" not in default_prompt
assert "travel account centered on a named subject" not in default_prompt
assert "Always return external_knowledge_required" in default_prompt

# Q03 Mode B is explicit and local rather than a global inference rule.
q03, q03_component = make_facet(
    "personal growth",
    "actual_self_development_or_self_understanding",
)
q03_policy = component_local_inference_policy(q03, q03_component)
assert "MODE A" in q03_policy and "MODE B" in q03_policy
assert "promotes maturity and growth" in q03_policy

# Q04 movement/danger rules are local to their canonical components.
q04_move, move_component = make_facet("dangerous journeys", "movement_or_travel")
move_prompt = build_missing_component_recovery_prompt(
    q04_move,
    move_component,
    {"S1": "This is a travel book.", "S2": "The named traveler encounters four places."},
    {},
)
assert "LOCAL Q04 MOVEMENT CONTRACT" in move_prompt
assert "cross-span entailment" in move_prompt

q04_danger, danger_component = make_facet("dangerous journeys", "danger_or_threat")
danger_prompt = build_isolated_component_prompt(
    q04_danger,
    danger_component,
    "S2",
    "The shipwrecked traveler encounters several places.",
)
assert "LOCAL Q04 DANGER CONTRACT" in danger_prompt
assert "shipwrecked" in danger_prompt

# Q09 explicitly rejects identity/membership-only positive grounding.
q09_active, active_component = make_facet("resistance", "active_opposition_or_defiance")
q09_target, target_component = make_facet("resistance", "target_oppressive_or_authoritarian_power")
identity_evidence = "Hannah is a political radical and member of the Weather Underground."
for facet, component in ((q09_active, active_component), (q09_target, target_component)):
    policy = component_local_inference_policy(facet, component)
    assert "named movement" in policy or "named political movement" in policy
    positive = IsolatedComponentVerification(
        component_id=component.component_id,
        grounding_relation="entailed",
        external_knowledge_required=False,
        reason="Imported group history.",
    )
    guarded = _q09_text_anchor_guard(facet, component, identity_evidence, positive)
    assert guarded.grounding_relation == "missing"
    assert guarded.external_knowledge_required is True
    assert guarded.supporting_span_ids if hasattr(guarded, "supporting_span_ids") else True

# A genuine text-local Q09 resistance statement survives the guard.
action = IsolatedComponentVerification(
    component_id="active_opposition_or_defiance",
    grounding_relation="entailed",
    external_knowledge_required=False,
    reason="The text explicitly describes protest.",
)
action_guarded = _q09_text_anchor_guard(
    q09_active,
    active_component,
    "She protests and resists the regime.",
    action,
)
assert action_guarded.grounding_relation == "entailed"

target = IsolatedComponentVerification(
    component_id="target_oppressive_or_authoritarian_power",
    grounding_relation="entailed",
    external_knowledge_required=False,
    reason="The target is described as authoritarian.",
)
target_guarded = _q09_text_anchor_guard(
    q09_target,
    target_component,
    "She resists an authoritarian regime.",
    target,
)
assert target_guarded.grounding_relation == "entailed"

# Recovery has the same identity-only precision guard.
recovery = FullContextComponentRecovery(
    component_id="target_oppressive_or_authoritarian_power",
    grounding_relation="entailed",
    supporting_span_ids=["S1"],
    external_knowledge_required=False,
    reason="Imported group target.",
)
recovery = _q09_recovery_text_anchor_guard(
    q09_target,
    target_component,
    {"S1": identity_evidence},
    recovery,
)
assert recovery.grounding_relation == "missing"
assert recovery.supporting_span_ids == []
assert recovery.external_knowledge_required is True

# Structural external-knowledge invariant remains enforced everywhere.
bad = IsolatedComponentVerification(
    component_id="target",
    grounding_relation="entailed",
    external_knowledge_required=True,
    reason="Would require outside entity history.",
)
try:
    _validate_isolated_component_result(bad, default_component, "test")
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("positive isolated grounding using external knowledge must be rejected")

bad_recovery = FullContextComponentRecovery(
    component_id="target",
    grounding_relation="entailed",
    supporting_span_ids=["S1"],
    external_knowledge_required=True,
    reason="Would import outside history.",
)
try:
    _validate_component_recovery_result(bad_recovery, default_component, {"S1": "x"}, {})
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("positive recovery using external knowledge must be rejected")

# Q03's high-precision deterministic Mode-B cue is present in config.
cfg = json.loads((EVALS_DIR / "judge_configs/semantic_relevance_judge.v0.27.0.json").read_text(encoding="utf-8"))
rules = cfg["deterministic_direct_cue_stage"]["rules"]
mode_b = next(rule for rule in rules if rule.get("cue_id") == "Q03_F1_promoted_growth_mode_b")
assert mode_b["query_id"] == "Q03" and mode_b["facet_id"] == "F1"
assert mode_b["all_regex_groups"]

print("v0.27 r3 localized inference and external-knowledge contracts passed")
