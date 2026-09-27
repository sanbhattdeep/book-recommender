from semantic_relevance_facet_judge import (
    ProminenceAssessment,
    SubjectRelation,
    _apply_direct_cue_list_mention_guard,
)

promoted = ProminenceAssessment(
    subject_relation=SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER,
    is_substantively_examined=False,
    supporting_span_ids=["S3"],
    reason="The listed gods are central.",
)

listed = (
    "His knighthood now beyond question, Able works to fulfill his vows "
    "to his king, his lover, his friends, his gods, and even his enemies."
)
guarded = _apply_direct_cue_list_mention_guard(
    evidence_span_id="S3",
    evidence_text=listed,
    result=promoted,
)
assert guarded.subject_relation == SubjectRelation.OTHER
assert "direct-cue list-mention guard" in guarded.reason

# Independent role support from another span defeats the cap.
independent = promoted.model_copy(update={"supporting_span_ids": ["S3", "S4"]})
preserved = _apply_direct_cue_list_mention_guard(
    evidence_span_id="S3",
    evidence_text=listed,
    result=independent,
)
assert preserved.subject_relation == SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER

# A direct cue in ordinary prose is not treated as a list-side-detail.
ordinary = _apply_direct_cue_list_mention_guard(
    evidence_span_id="S1",
    evidence_text="The gods govern the world and repeatedly intervene in the hero's quest.",
    result=promoted.model_copy(update={"supporting_span_ids": ["S1"]}),
)
assert ordinary.subject_relation == SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER

print("v0.28 r2 direct-cue prominence isolation contract passed")
