from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / "rubrics" / "semantic_relevance" / "semantic_relevance_rubric.v0.1.0.json"
r = json.loads(P.read_text(encoding="utf-8"))
assert r["rubric_id"] == "semantic-recommendation-relevance"
assert r["version"] == "0.1.0"
assert r["status"] == "frozen"
assert str(r.get("purpose", "")).strip()
assert len(r.get("evidence_rules", [])) >= 4
assert str(r.get("tie_break_rule", "")).strip()
assert set(r.get("scores", {})) == {"0","1","2","3","4"}
for score in ["0","1","2","3","4"]:
    item = r["scores"][score]
    assert str(item.get("label", "")).strip(), score
    assert str(item.get("definition", "")).strip(), score
# Regression guard against the accidental package placeholder {"version":"0.1.0"}.
assert len(P.read_bytes()) > 1000
print("Rubric v0.1.0 integrity test passed.")
