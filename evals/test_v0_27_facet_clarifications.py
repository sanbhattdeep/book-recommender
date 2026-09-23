"""Ensure v0.9.4 is a narrow Q04/Q05/Q09 clarification over frozen v0.9.3."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
p93 = ROOT / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.3.json"
p94 = ROOT / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.4.json"
a = json.loads(p93.read_text(encoding="utf-8"))
b = json.loads(p94.read_text(encoding="utf-8"))
assert b["version"] == "0.9.4"
assert b["development_dataset_version"] == "3.0.0"

qa={q["query_id"]:q for q in a["queries"]}
qb={q["query_id"]:q for q in b["queries"]}
assert set(qa)==set(qb)
for qid in qa:
    if qid not in {"Q04","Q05","Q09"}:
        assert qa[qid] == qb[qid], f"unexpected semantic change in {qid}"

# Component IDs must remain immutable everywhere.
for qid in qa:
    for fa, fb in zip(qa[qid]["facets"], qb[qid]["facets"], strict=True):
        assert fa["facet_id"] == fb["facet_id"]
        assert [x["component_id"] for x in fa["required_components"]] == [x["component_id"] for x in fb["required_components"]]

q04 = qb["Q04"]
text = json.dumps(q04, ensure_ascii=False)
assert "travel account" in text and "multiple stated places" in text
q05 = qb["Q05"]
text = json.dumps(q05, ensure_ascii=False)
assert "collapse" in text and "revolutionary violence" in text and "political-power/state-authority" in text
q09 = qb["Q09"]
text = json.dumps(q09, ensure_ascii=False)
assert "named political movement" in text and "outside knowledge" in text

print("v0.27 v0.9.4 narrow facet clarification contract passed")
