from __future__ import annotations
from types import SimpleNamespace
from semantic_relevance_facet_judge import (
    IsolatedComponentVerification, FullContextComponentRecovery,
    _q07_parent_child_text_anchor_guard, _q07_parent_child_recovery_guard,
)
def facet(text): return SimpleNamespace(text=text)
def component(cid): return SimpleNamespace(component_id=cid)
def pos():
    return IsolatedComponentVerification(component_id="parent_child_relationship",grounding_relation="entailed",negative_boundary_applied=False,external_knowledge_required=False,reason="fixture")
def main():
    f=facet("complicated parent-child relationship"); c=component("parent_child_relationship")
    for text in [
        "They reunite after eleven years and blame and resentment resurface.",
        "Their relationship is strained by years of anger and misunderstanding.",
        "Two former lovers struggle with a difficult relationship.",
    ]:
        r=_q07_parent_child_text_anchor_guard(f,c,text,pos())
        assert r.grounding_relation=="missing" and r.negative_boundary_applied
    for text in [
        "Her father rejects her decision and refuses to speak to her.",
        "The mother and daughter remain estranged for years.",
        "His parents oppose the relationship.",
    ]:
        r=_q07_parent_child_text_anchor_guard(f,c,text,pos())
        assert r.grounding_relation=="entailed"
    spans={"S1":"Two young lovers reunite after eleven years.","S2":"Blame and resentment resurface between them."}
    rec=FullContextComponentRecovery(component_id="parent_child_relationship",grounding_relation="entailed",supporting_span_ids=["S1","S2"],negative_boundary_applied=False,external_knowledge_required=False,reason="fixture")
    rr=_q07_parent_child_recovery_guard(f,c,spans,rec)
    assert rr.grounding_relation=="missing" and rr.supporting_span_ids==[]
    spans2={"S1":"Her parents oppose the relationship.","S2":"She refuses to accept their decision."}
    rec2=FullContextComponentRecovery(component_id="parent_child_relationship",grounding_relation="entailed",supporting_span_ids=["S1","S2"],negative_boundary_applied=False,external_knowledge_required=False,reason="fixture")
    assert _q07_parent_child_recovery_guard(f,c,spans2,rec2).grounding_relation=="entailed"
    print("v0.29 r5 localized Q07 parent-child grounding tests passed")
if __name__=="__main__": main()
