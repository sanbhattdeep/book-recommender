"""Generic component boundaries added in facet spec v0.9.0 cover the v0.24 failures."""
from __future__ import annotations

import json
from pathlib import Path

E = Path(__file__).resolve().parent
payload = json.loads((E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.0.json").read_text())

by_key = {
    (q["query_id"], f["facet_id"]): f
    for q in payload["queries"]
    for f in q["facets"]
}

def component(query_id: str, facet_id: str, component_id: str) -> dict:
    facet = by_key[(query_id, facet_id)]
    return next(c for c in facet["required_components"] if c["component_id"] == component_id)

q07 = component("Q07", "F1", "parent_child_relationship")
assert "marriage" in " ".join(q07["negative_boundaries"]).lower()

q04 = component("Q04", "F3", "movement_or_travel")
movement_boundaries = " ".join(q04["negative_boundaries"]).lower()
assert "trapped" in movement_boundaries
assert "trail" in movement_boundaries
assert "pursued" in movement_boundaries

q06_loss = component("Q06", "F2", "actual_significant_deprivation_or_ending")
loss_boundaries = " ".join(q06_loss["negative_boundaries"]).lower()
assert "lost time" in loss_boundaries
assert "lost place" in loss_boundaries

q06_rebuild = component("Q06", "F3", "rebuilding_or_resuming_own_life_after_loss")
rebuild_boundaries = " ".join(q06_rebuild["negative_boundaries"]).lower()
assert "reconstructing history" in rebuild_boundaries
assert "another person's former life" in rebuild_boundaries

q06_moving = component("Q06", "F4", "emotionally_affecting_or_poignant_presentation")
moving_boundaries = " ".join(q06_moving["negative_boundaries"]).lower()
assert "reader-response wording is not required" in moving_boundaries

print("v0.25.0 component-specific boundary contract passed")
