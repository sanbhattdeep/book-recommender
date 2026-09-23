"""Ensure v0.9.5 only tightens the validated r1 residual Q04/Q09 boundaries."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
p94 = ROOT / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.4.json"
p95 = ROOT / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.5.json"
a = json.loads(p94.read_text(encoding="utf-8"))
b = json.loads(p95.read_text(encoding="utf-8"))
assert b["version"] == "0.9.5"
assert b["development_dataset_version"] == "3.0.0"

qa={q["query_id"]:q for q in a["queries"]}
qb={q["query_id"]:q for q in b["queries"]}
assert set(qa)==set(qb)
for qid in qa:
    if qid not in {"Q04","Q09"}:
        assert qa[qid] == qb[qid], f"unexpected semantic change in {qid}"

# Component IDs remain immutable everywhere.
for qid in qa:
    for fa, fb in zip(qa[qid]["facets"], qb[qid]["facets"], strict=True):
        assert fa["facet_id"] == fb["facet_id"]
        assert [x["component_id"] for x in fa["required_components"]] == [x["component_id"] for x in fb["required_components"]]

q04 = qb["Q04"]
text = json.dumps(q04, ensure_ascii=False)
assert "shipwrecked" in text and "danger/hazard cue" in text
assert "travel account" in text and "multiple stated places" in text

q09 = qb["Q09"]
text = json.dumps(q09, ensure_ascii=False)
assert "political label" in text
assert "membership in a named movement" in text
assert "outside knowledge" in text

print("v0.27 v0.9.5 localized facet clarification contract passed")
