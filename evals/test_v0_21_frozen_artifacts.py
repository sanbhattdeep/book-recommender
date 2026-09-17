"""Verify required v0.21 artifacts are execution-ready."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
FACET=ROOT/"evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json"
CONFIG=ROOT/"evals/judge_configs/semantic_relevance_judge.v0.21.0.json"
RUBRIC=ROOT/"evals/rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
facet=json.loads(FACET.read_text(encoding="utf-8"))
config=json.loads(CONFIG.read_text(encoding="utf-8"))
rubric=json.loads(RUBRIC.read_text(encoding="utf-8"))
assert facet.get("version")=="0.8.0"
assert facet.get("status")=="frozen", "Facet specification must be frozen."
assert config.get("version")=="0.21.0"
assert rubric.get("version")=="0.1.0"
assert set(rubric.get("scores",{}))=={"0","1","2","3","4"}
print("v0.21 frozen-artifact contract passed.")
