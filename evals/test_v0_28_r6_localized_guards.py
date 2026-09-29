from __future__ import annotations
from types import SimpleNamespace
from semantic_relevance_facet_judge import (
    IsolatedComponentVerification, ProminenceAssessment, SubjectRelation,
    _q11_r6_explicit_myth_hero_positive_guard,
    _r6_whole_work_thematic_enumeration_positive_guard,
)

def facet(text): return SimpleNamespace(text=text)
def comp(cid): return SimpleNamespace(component_id=cid)
def missing(cid): return IsolatedComponentVerification(component_id=cid,grounding_relation="missing",reason="test")

def main():
    # r6 Q01: explicit whole-work thematic enumeration must be substantive.
    p=ProminenceAssessment(subject_relation=SubjectRelation.OTHER,is_substantively_examined=False,supporting_span_ids=["S1"],reason="test")
    p=_r6_whole_work_thematic_enumeration_positive_guard(
        facet("redemption"),
        "This story of friendship, love, envy, tragedy, and redemption gives breath to philosophy.",
        p,
    )
    assert p.subject_relation==SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER
    assert p.is_substantively_examined is True

    # Side-detail possessive lists are not whole-work thematic framing.
    p2=ProminenceAssessment(subject_relation=SubjectRelation.OTHER,is_substantively_examined=False,supporting_span_ids=["S1"],reason="test")
    p2=_r6_whole_work_thematic_enumeration_positive_guard(
        facet("gods"), "His king, his lover, his friends, his gods, and his duty all compete.", p2
    )
    assert p2.subject_relation==SubjectRelation.OTHER

    # r6 Q11: explicit myths/legends + hero stories is positive content.
    r=_q11_r6_explicit_myth_hero_positive_guard(
        facet("legendary heroes"),comp("legendary_or_mythic_hero_identity"),
        "The book brings to life Greek, Roman, and Norse myths and legends - the stories of gods and heroes.",
        missing("legendary_or_mythic_hero_identity"),
    )
    assert r.grounding_relation=="entailed"

    # Provenance scholarship remains blocked from this positive escape hatch.
    r2=_q11_r6_explicit_myth_hero_positive_guard(
        facet("legendary heroes"),comp("legendary_or_mythic_hero_identity"),
        "A scholarly study presents evidence for the historical origin of the legend of a hero.",
        missing("legendary_or_mythic_hero_identity"),
    )
    assert r2.grounding_relation=="missing"
    print("v0.28 r6 localized non-regression guard tests passed")
if __name__=="__main__": main()
