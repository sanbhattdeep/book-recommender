"""Python, not the LLM, owns full-facet relation assembly in v0.26."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    IsolatedComponentVerification,
    _assemble_single_span_from_component_results,
)
from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec, VerificationRelation

E = Path(__file__).resolve().parent
FACETS = json.loads((E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.3.json").read_text(encoding="utf-8"))
raw = next(q for q in FACETS["queries"] if q["query_id"] == "Q07")
spec = QueryFacetSpec(query_id=raw["query_id"], query=raw["query"], facets=[QueryFacet(**f) for f in raw["facets"]])
facet = spec.facets[0]


def c(cid: str, rel: str, boundary: bool = False):
    return IsolatedComponentVerification(
        component_id=cid,
        grounding_relation=rel,
        negative_boundary_applied=boundary,
        reason=f"{cid}:{rel}",
    )

r = _assemble_single_span_from_component_results(
    facet, "S1", [c("parent_child_relationship", "explicit"), c("difficult_or_complex_dynamic", "explicit")]
)
assert r.verification_relation == VerificationRelation.DIRECT

r = _assemble_single_span_from_component_results(
    facet, "S1", [c("parent_child_relationship", "entailed"), c("difficult_or_complex_dynamic", "explicit")]
)
assert r.verification_relation == VerificationRelation.ENTAILED

r = _assemble_single_span_from_component_results(
    facet, "S1", [c("parent_child_relationship", "missing"), c("difficult_or_complex_dynamic", "explicit")]
)
assert r.verification_relation == VerificationRelation.ADJACENT
assert r.missing_semantic_component == "parent_child_relationship"

r = _assemble_single_span_from_component_results(
    facet, "S1", [c("parent_child_relationship", "missing", True), c("difficult_or_complex_dynamic", "explicit")]
)
assert r.verification_relation == VerificationRelation.UNSUPPORTED
assert r.semantic_definition_exclusion_applied is True

print("v0.26.0 deterministic facet assembly contract passed")
