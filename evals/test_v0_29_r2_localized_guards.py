from types import SimpleNamespace
from semantic_relevance_facet_judge import _q04_r2_cross_span_movement_component_check,_q09_r2_resistance_pair_component_check
def facet(t): return SimpleNamespace(text=t)
def comp(c): return SimpleNamespace(component_id=c)
def main():
    gulliver={"S1":"Gulliver's travels purports to be a travel book.","S2":"It describes the shipwrecked Gulliver's encounters with inhabitants of four places."}
    r=_q04_r2_cross_span_movement_component_check(facet("dangerous journeys"),comp("movement_or_travel"),gulliver)
    assert r is not None and r.established and r.supporting_span_ids==["S1"]
    banker={"S1":"Events unfold over three years.","S2":"Violent explosive action creates a threat of death."}
    assert _q04_r2_cross_span_movement_component_check(facet("dangerous journeys"),comp("movement_or_travel"),banker) is None
    animal={"S1":"A classic satire on totalitarianism in which farm animals overthrow their human owner."}
    for cid in ["active_opposition_or_defiance","target_oppressive_or_authoritarian_power"]:
        rr=_q09_r2_resistance_pair_component_check(facet("resistance"),comp(cid),animal); assert rr is not None and rr.established
    phil={"S1":"A tiny nation falls under tyrannical rule."}
    for cid in ["active_opposition_or_defiance","target_oppressive_or_authoritarian_power"]:
        assert _q09_r2_resistance_pair_component_check(facet("resistance"),comp(cid),phil) is None
    neutral={"S1":"The workers overthrow their manager and elect a new committee."}
    assert _q09_r2_resistance_pair_component_check(facet("resistance"),comp("active_opposition_or_defiance"),neutral) is None
    print("v0.29 r2 localized Q04/Q09 regression tests passed")
if __name__=="__main__": main()
