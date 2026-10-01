from __future__ import annotations
from types import SimpleNamespace

from semantic_relevance_facet_judge import (
    IsolatedComponentVerification,
    FullContextComponentRecovery,
    _q03_r1_purpose_component_check,
    _q04_r1_movement_text_anchor_guard,
    _q04_r1_recovery_movement_text_anchor_guard,
)
from semantic_relevance_facet_scoring import (
    QueryFacet,
    QueryFacetSpec,
    FacetJudgeVerdict,
    FacetPipelineAssessment,
    FacetProminence,
    VerificationRelation,
    compute_facet_score,
)

def facet(text):
    return SimpleNamespace(text=text)

def comp(cid):
    return SimpleNamespace(component_id=cid)

def isolated(cid, relation="entailed"):
    return IsolatedComponentVerification(
        component_id=cid,
        grounding_relation=relation,
        negative_boundary_applied=False,
        external_knowledge_required=False,
        reason="fixture",
    )

def recovery(cid, relation="entailed", spans=None):
    return FullContextComponentRecovery(
        component_id=cid,
        grounding_relation=relation,
        supporting_span_ids=list(spans or ["S1"]),
        negative_boundary_applied=False,
        external_knowledge_required=False,
        reason="fixture",
    )

def assessment(fid, supported):
    if supported:
        return FacetPipelineAssessment(
            facet_id=fid,
            candidate_evidence_span_ids=["S1"],
            candidate_evidence_span_id="S1",
            verification_relation=VerificationRelation.ENTAILED,
            prominence=FacetProminence.SUBSTANTIVE,
            evidence_selection_reason="fixture",
            verification_reason="fixture",
        )
    return FacetPipelineAssessment(
        facet_id=fid,
        candidate_evidence_span_ids=[],
        candidate_evidence_span_id="NONE",
        verification_relation=VerificationRelation.UNSUPPORTED,
        prominence=FacetProminence.NOT_APPLICABLE,
        evidence_selection_reason="fixture",
        verification_reason="fixture",
    )

def q09_spec(qid="Q09"):
    return QueryFacetSpec(
        query_id=qid,
        query="fixture",
        facets=[
            QueryFacet(facet_id="F1", text="authoritarian control", facet_type="core", semantic_definition="x"),
            QueryFacet(facet_id="F2", text="resistance", facet_type="core", semantic_definition="x"),
            QueryFacet(facet_id="F3", text="dystopian", facet_type="core", semantic_definition="x"),
        ],
    )

def main():
    # Q03 positive: the pair is sufficient.
    spans = {
        "S1": "A potent pathway to self-awakening that will help you to live your greatest life.",
        "S2": "It also promises happiness, prosperity and inner peace.",
    }
    r = _q03_r1_purpose_component_check(
        facet("finding purpose"),
        comp("meaning_direction_calling_or_goal_clarification"),
        spans,
    )
    assert r is not None and r.established and r.grounding_relation == "entailed"
    assert "S1" in r.supporting_span_ids

    # Q03 negatives: either half by itself must remain insufficient.
    assert _q03_r1_purpose_component_check(
        facet("finding purpose"),
        comp("meaning_direction_calling_or_goal_clarification"),
        {"S1": "A pathway to self-awakening and inner peace."},
    ) is None
    assert _q03_r1_purpose_component_check(
        facet("finding purpose"),
        comp("meaning_direction_calling_or_goal_clarification"),
        {"S1": "Advice to live your greatest life with happiness and prosperity."},
    ) is None

    # Q04 negative: temporal narrative progression + danger is not movement.
    banker = (
        "Onwards from there, events lead towards violent explosive action and the "
        "threat of death. The story covers a span of three years."
    )
    guarded = _q04_r1_movement_text_anchor_guard(
        facet("dangerous journeys"),
        comp("movement_or_travel"),
        banker,
        isolated("movement_or_travel"),
    )
    assert guarded.grounding_relation == "missing"
    assert guarded.negative_boundary_applied is True

    # Q04 positive: explicit travel/journey anchor survives.
    travel = "She travels across dangerous mountain passes on a long journey."
    kept = _q04_r1_movement_text_anchor_guard(
        facet("dangerous journeys"),
        comp("movement_or_travel"),
        travel,
        isolated("movement_or_travel"),
    )
    assert kept.grounding_relation == "entailed"

    # Q04 full-context recovery uses the same boundary.
    rec = _q04_r1_recovery_movement_text_anchor_guard(
        facet("dangerous journeys"),
        comp("movement_or_travel"),
        {"S1": "Events unfold over three years.", "S2": "Violence threatens death."},
        recovery("movement_or_travel", spans=["S1","S2"]),
    )
    assert rec.grounding_relation == "missing"

    # Q09: two-thirds generic coverage no longer produces clear when resistance
    # is fully absent.
    verdict = FacetJudgeVerdict(
        assessments=[
            assessment("F1", True),
            assessment("F2", False),
            assessment("F3", True),
        ],
        overall_reason="fixture",
    )
    score = compute_facet_score(q09_spec("Q09"), verdict)
    assert score.score == 2
    assert score.clear_rule_applied == "q09_missing_resistance_cap_to_partial"

    # The cap is local to Q09; generic three-core scoring remains unchanged.
    generic = compute_facet_score(q09_spec("Q99"), verdict)
    assert generic.score == 3

    # Q09 with resistance present remains clear.
    all_supported = FacetJudgeVerdict(
        assessments=[assessment("F1", True), assessment("F2", True), assessment("F3", True)],
        overall_reason="fixture",
    )
    positive = compute_facet_score(q09_spec("Q09"), all_supported)
    assert positive.score >= 3

    print("v0.29 r1 localized Q03/Q04/Q09 guard tests passed")

if __name__ == "__main__":
    main()
