from __future__ import annotations
from types import SimpleNamespace
from semantic_relevance_facet_judge import (
    IsolatedComponentVerification, ProminenceAssessment, SubjectRelation,
    _q02_r5_interpersonal_disclosure_guard,
    _q06_r5_trauma_without_loss_guard,
    _q10_r5_organizational_improvement_positive_guard,
    _q11_r5_legend_provenance_guard,
    _apply_v028_r5_prominence_guard,
)

def facet(text): return SimpleNamespace(text=text)
def comp(cid): return SimpleNamespace(component_id=cid)
def positive(cid): return IsolatedComponentVerification(component_id=cid,grounding_relation="entailed",reason="test")
def missing(cid): return IsolatedComponentVerification(component_id=cid,grounding_relation="missing",reason="test")

def main():
    r=_q02_r5_interpersonal_disclosure_guard(
        facet("suspense"),comp("story_level_tension_or_anticipation"),
        "Will their friendship be shattered when Tohno learns the truth about the obsession?",positive("story_level_tension_or_anticipation"))
    assert r.grounding_relation=="missing"
    r=_q02_r5_interpersonal_disclosure_guard(
        facet("suspense"),comp("story_level_tension_or_anticipation"),
        "A detective pursues a murderer while a friendship is threatened when she learns the truth.",positive("story_level_tension_or_anticipation"))
    assert r.grounding_relation!="missing"

    r=_q06_r5_trauma_without_loss_guard(
        facet("learning to live again"),comp("prior_grief_or_major_loss"),
        "He is wounded and tries to heal from the horrors of war and create something worth living for.",positive("prior_grief_or_major_loss"))
    assert r.grounding_relation=="missing"
    r=_q06_r5_trauma_without_loss_guard(
        facet("learning to live again"),comp("prior_grief_or_major_loss"),
        "After the death of his wife, the wounded veteran tries to rebuild his life.",positive("prior_grief_or_major_loss"))
    assert r.grounding_relation!="missing"

    r=_q10_r5_organizational_improvement_positive_guard(
        facet("building effective organizations"),comp("organizational_design_management_or_improvement"),
        "An approach helps transform confusion and infighting into clarity and alignment.",missing("organizational_design_management_or_improvement"))
    assert r.grounding_relation=="entailed"
    r=_q10_r5_organizational_improvement_positive_guard(
        facet("building effective organizations"),comp("organizational_effectiveness_goal"),
        "An approach helps transform confusion and infighting into clarity and alignment.",missing("organizational_effectiveness_goal"))
    assert r.grounding_relation=="entailed"

    r=_q11_r5_legend_provenance_guard(
        facet("mythology"),comp("mythic_or_mythological_basis"),
        "The author presents evidence that the legend originated in an ancient ritual.",positive("mythic_or_mythological_basis"))
    assert r.grounding_relation=="missing"
    r=_q11_r5_legend_provenance_guard(
        facet("legendary heroes"),comp("legendary_or_mythic_hero_identity"),
        "The study examines the historical origin of the legend of Beowulf.",positive("legendary_or_mythic_hero_identity"))
    assert r.grounding_relation=="missing"

    p=ProminenceAssessment(
        subject_relation=SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER,
        is_substantively_examined=True,supporting_span_ids=["S1"],reason="test")
    p=_apply_v028_r5_prominence_guard(
        facet("inequality"),
        "The characters are the dispossessed working class challenged by the neorural bourgeoisie.",
        p)
    assert p.subject_relation==SubjectRelation.OTHER and not p.is_substantively_examined
    print("v0.28 r5 localized guard tests passed")
if __name__=="__main__":
    main()
