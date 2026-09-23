"""v0.26.0 r6: facet-specific sufficiency semantics must stay local."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    build_isolated_component_prompt,
    build_missing_component_recovery_prompt,
)
from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec

E = Path(__file__).resolve().parent
FACETS = json.loads(
    (E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.3.json")
    .read_text(encoding="utf-8")
)


def facet(query_id: str, facet_id: str) -> QueryFacet:
    raw = next(q for q in FACETS["queries"] if q["query_id"] == query_id)
    spec = QueryFacetSpec(
        query_id=raw["query_id"],
        query=raw["query"],
        facets=[QueryFacet(**f) for f in raw["facets"]],
    )
    return next(f for f in spec.facets if f.facet_id == facet_id)


# Q03/F1 keeps its two-mode contract because it is stored in the local component.
q03 = facet("Q03", "F1")
q03_component = q03.required_components[0]
q03_prompt = build_isolated_component_prompt(
    q03,
    q03_component,
    "S1",
    "Dating can be a spiritual path that promotes maturity and growth in young people.",
).lower()
assert "two independently sufficient positive modes" in q03_prompt
assert "mode a" in q03_prompt and "mode b" in q03_prompt
assert "promotes maturity and growth" in q03_prompt
assert "alternative positive grounding modes" not in q03_prompt
assert "do not silently replace it with a stricter criterion" not in q03_prompt

q03_recovery = build_missing_component_recovery_prompt(
    q03,
    q03_component,
    {"S1": "Dating can be a spiritual path that promotes maturity and growth in young people."},
    {},
).lower()
assert "two independently sufficient positive modes" in q03_recovery
assert "promotes maturity and growth" in q03_recovery
assert "alternative positive grounding modes" not in q03_recovery

# Q07 sees only its own referent-binding contract, never Q03's Mode A/Mode B language.
q07 = facet("Q07", "F1")
q07_component = next(c for c in q07.required_components if c.component_id == "parent_child_relationship")
q07_prompt = build_isolated_component_prompt(
    q07,
    q07_component,
    "S1",
    "A young couple marry and face bitter parental opposition.",
).lower()
assert "parental opposition" in q07_prompt
assert "same person may separately be a spouse/partner and a child" in q07_prompt
assert "two independently sufficient positive modes" not in q07_prompt
assert "alternative positive grounding modes" not in q07_prompt
assert "do not silently replace it with a stricter criterion" not in q07_prompt

q07_recovery = build_missing_component_recovery_prompt(
    q07,
    q07_component,
    {"S1": "A young couple marry and face bitter parental opposition."},
    {},
).lower()
assert "parental opposition" in q07_recovery
assert "two independently sufficient positive modes" not in q07_recovery
assert "alternative positive grounding modes" not in q07_recovery

print("v0.26.0 r6 local-contract isolation passed")
