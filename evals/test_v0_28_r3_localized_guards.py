from types import SimpleNamespace
from semantic_relevance_facet_judge import (
    IsolatedComponentVerification,
    FullContextComponentRecovery,
    _apply_v028_r3_isolated_component_guards,
    _apply_v028_r3_recovery_guard,
)

def res(rel="missing"):
    return IsolatedComponentVerification(
        component_id="adventure_like_progression",
        grounding_relation=rel,
        negative_boundary_applied=False,
        external_knowledge_required=False,
        reason="model result",
    )

adventure = SimpleNamespace(text="adventure")
adventure_component = SimpleNamespace(component_id="adventure_like_progression")

# Explicit narrative expedition is positive; this repairs U_Q11_T70.
r = _apply_v028_r3_isolated_component_guards(
    adventure,
    adventure_component,
    "A fictionalized account based on the Lewis and Clark expedition to the Pacific Ocean.",
    res(),
)
assert r.grounding_relation == "entailed"

# A concrete hazardous pursuit is positive; this repairs U2_Q11_T70.
r = _apply_v028_r3_isolated_component_guards(
    adventure,
    adventure_component,
    "But this King sent Garth after a basilisk whose gaze could turn men to stone.",
    res(),
)
assert r.grounding_relation == "entailed"

# Explicit take-up-arms / battle / army action is positive; repairs U3_Q11_T03.
r = _apply_v028_r3_isolated_component_guards(
    adventure,
    adventure_component,
    "Two young heroes must take up arms in the struggle to protect their besieged world and stand at the fore of the battle against an army.",
    res(),
)
assert r.grounding_relation == "entailed"

# r2 holdout control: a bare mission to set things right remains missing.
r = _apply_v028_r3_isolated_component_guards(
    adventure,
    adventure_component,
    "Able must set his world right, restoring the proper order among the denizens of all the seven worlds.",
    res(),
)
assert r.grounding_relation == "missing"

# Adjudication-only control: rescue/world-saving conflict alone remains missing.
r = _apply_v028_r3_isolated_component_guards(
    adventure,
    adventure_component,
    "He is rescued by a shapeshifter and helps her save herself and Earth from the wrath of her enemy.",
    res(),
)
assert r.grounding_relation == "missing"

# Meta use of quest/expedition remains blocked.
r = _apply_v028_r3_isolated_component_guards(
    adventure,
    adventure_component,
    "A literary analysis uses the hero's quest as an example of symbolism.",
    res(),
)
assert r.grounding_relation == "missing"

growth = SimpleNamespace(text="personal growth")
growth_component = SimpleNamespace(component_id="actual_self_development_or_self_understanding")
positive_growth = IsolatedComponentVerification(
    component_id="actual_self_development_or_self_understanding",
    grounding_relation="explicit",
    negative_boundary_applied=False,
    external_knowledge_required=False,
    reason="model result",
)

# Generic intellectual learning/wisdom is not personal growth.
for text in (
    "Philosophy For Dummies will put you on the path to wising up as you steer through life.",
    "If we are aware of how little we know, then we can really begin to learn.",
):
    r = _apply_v028_r3_isolated_component_guards(
        growth, growth_component, text, positive_growth
    )
    assert r.grounding_relation == "missing"
    assert r.negative_boundary_applied

# Explicit growth/development wording remains positive.
r = _apply_v028_r3_isolated_component_guards(
    growth,
    growth_component,
    "The program promotes maturity and personal growth and builds confidence.",
    positive_growth,
)
assert r.grounding_relation == "explicit"

# Recovery cannot reintroduce learning-only evidence.
recovery = FullContextComponentRecovery(
    component_id="actual_self_development_or_self_understanding",
    grounding_relation="entailed",
    supporting_span_ids=["S1"],
    negative_boundary_applied=False,
    external_knowledge_required=False,
    reason="model recovery",
)
r = _apply_v028_r3_recovery_guard(
    growth,
    growth_component,
    {"S1": "The book encourages philosophical thinking and learning."},
    recovery,
)
assert r.grounding_relation == "missing"
assert r.supporting_span_ids == []

print("v0.28 r3 localized component guards passed")
