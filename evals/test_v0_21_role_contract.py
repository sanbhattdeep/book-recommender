from semantic_relevance_facet_judge import ProminenceAssessment, _context_role_from_decomposed_signals
from semantic_relevance_facet_scoring import FacetContextRole


def role(**kw):
    base=dict(
        primary_subject_summary="x",
        is_primary_subject=False,
        is_background_cause_or_factor=False,
        is_example_or_illustration=False,
        is_meta_discussion=False,
        is_substantively_examined=False,
        supporting_span_ids=["S1"],
        reason="x",
    )
    base.update(kw)
    return _context_role_from_decomposed_signals(ProminenceAssessment(**base))

assert role(is_background_cause_or_factor=True, is_substantively_examined=True) == FacetContextRole.SUBSTANTIVE_SUBJECT
assert role(is_primary_subject=True, is_background_cause_or_factor=True) == FacetContextRole.CENTRAL_SUBJECT
assert role(is_meta_discussion=True, is_primary_subject=True) == FacetContextRole.META_DISCUSSION
assert role(is_example_or_illustration=True, is_substantively_examined=True) == FacetContextRole.EXAMPLE_OR_ILLUSTRATION
assert role(is_background_cause_or_factor=True) == FacetContextRole.BACKGROUND_CAUSE
assert role() == FacetContextRole.INCIDENTAL_MENTION
print("v0.21.2 decomposed role precedence contract passed")
