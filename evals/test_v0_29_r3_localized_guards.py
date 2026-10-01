from __future__ import annotations
from types import SimpleNamespace

from semantic_relevance_facet_judge import (
    _q04_r3_shipwreck_danger_component_check,
)

def facet(text):
    return SimpleNamespace(text=text)

def comp(cid):
    return SimpleNamespace(component_id=cid)

def main():
    # Positive: exact consumed boundary — travel framing + explicit shipwreck.
    gulliver = {
        "S1": "Gulliver's travels purports to be a travel book.",
        "S2": "It describes the shipwrecked Gulliver's encounters with inhabitants of four places.",
    }
    r = _q04_r3_shipwreck_danger_component_check(
        facet("dangerous journeys"), comp("danger_or_threat"), gulliver
    )
    assert r is not None and r.established
    assert r.grounding_relation == "entailed"
    assert set(r.supporting_span_ids) == {"S1", "S2"}

    # Negative: travel with no danger cue is not dangerous travel.
    safe_trip = {
        "S1": "A travel book following a long journey across several towns.",
        "S2": "The traveler meets local families and records their customs.",
    }
    assert _q04_r3_shipwreck_danger_component_check(
        facet("dangerous journeys"), comp("danger_or_threat"), safe_trip
    ) is None

    # Negative: an isolated shipwreck reference without an independent travel
    # anchor does not trigger this deterministic dangerous-journey rule.
    isolated_shipwreck = {
        "S1": "A museum catalogue discusses a famous shipwreck.",
    }
    assert _q04_r3_shipwreck_danger_component_check(
        facet("dangerous journeys"), comp("danger_or_threat"), isolated_shipwreck
    ) is None

    # Negative: Banker-like danger/action remains outside this rule.
    banker = {
        "S1": "Events unfold over three years.",
        "S2": "Violent explosive action creates a threat of death.",
    }
    assert _q04_r3_shipwreck_danger_component_check(
        facet("dangerous journeys"), comp("danger_or_threat"), banker
    ) is None

    print("v0.29 r3 localized Q04 shipwreck-danger tests passed")

if __name__ == "__main__":
    main()
