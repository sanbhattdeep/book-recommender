from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / "facets" / "semantic_relevance" / "semantic_relevance_query_facets.v0.7.0.json"
payload = json.loads(P.read_text(encoding="utf-8"))
assert payload["version"] == "0.7.0"
q = {x["query_id"]: x for x in payload["queries"]}

def definition(qid, fid):
    return next(f["semantic_definition"] for f in q[qid]["facets"] if f["facet_id"] == fid)

assert "terrorist cell" in definition("Q05","F1")
assert "do not by themselves establish war" in definition("Q05","F1").lower()
assert "literary criticism" in definition("Q11","F1")
assert "do not invent gods" in definition("Q11","F3").lower()
assert "movement" in definition("Q04","F1").lower() and "threat" in definition("Q04","F1").lower()
assert "different clauses" in definition("Q04","F3").lower()
print("v0.19 facet boundary specification tests passed.")
