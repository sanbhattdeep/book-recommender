from semantic_relevance_facet_judge import ProminenceAssessment, _context_role_from_decomposed_signals
from semantic_relevance_facet_scoring import FacetContextRole, SubjectRelation

def role(relation, substantive=False):
    return _context_role_from_decomposed_signals(ProminenceAssessment(
        subject_relation=relation,
        is_substantively_examined=substantive,
        supporting_span_ids=["S1"],
        reason="synthetic",
    ))

assert role(SubjectRelation.SAME_AS_PRIMARY_SUBJECT) == FacetContextRole.CENTRAL_SUBJECT
assert role(SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER) == FacetContextRole.SUBSTANTIVE_SUBJECT
assert role(SubjectRelation.CAUSAL_OR_CONTEXTUAL_BACKGROUND) == FacetContextRole.BACKGROUND_CAUSE
assert role(SubjectRelation.CAUSAL_OR_CONTEXTUAL_BACKGROUND, True) == FacetContextRole.SUBSTANTIVE_SUBJECT
assert role(SubjectRelation.EXAMPLE_OR_META) == FacetContextRole.EXAMPLE_OR_ILLUSTRATION
assert role(SubjectRelation.OTHER) == FacetContextRole.INCIDENTAL_MENTION
print("v0.22.2 explicit subject-relation resolver contract passed")
