from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rubric=json.loads((ROOT/"rubrics"/"semantic_relevance"/"semantic_relevance_rubric.v0.1.0.json").read_text(encoding="utf-8"))
config=json.loads((ROOT/"judge_configs"/"semantic_relevance_judge.v0.19.0.json").read_text(encoding="utf-8"))
facets=json.loads((ROOT/"facets"/"semantic_relevance"/"semantic_relevance_query_facets.v0.7.0.json").read_text(encoding="utf-8"))
assert rubric["version"]=="0.1.0" and set(rubric["scores"])=={"0","1","2","3","4"}
assert all(rubric["scores"][str(i)]["definition"].strip() for i in range(5))
assert len((ROOT/"rubrics"/"semantic_relevance"/"semantic_relevance_rubric.v0.1.0.json").read_bytes())>1000
assert config["version"]=="0.19.0" and config["facet_spec_version"]=="0.7.0" and config["rubric_version"]=="0.1.0"
assert facets["version"]=="0.7.0" and facets["rubric_version"]=="0.1.0"
assert "inference_contract" in config
assert "context_role_scale" in config["prominence_stage"]
print("v0.19 package semantic/config integrity: PASS")
print(f"Rubric bytes: {(ROOT/'rubrics'/'semantic_relevance'/'semantic_relevance_rubric.v0.1.0.json').stat().st_size}")
print("Rubric scores present: 0,1,2,3,4")
