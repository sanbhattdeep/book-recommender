from __future__ import annotations
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def j(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def t(p): return p.read_text(encoding="utf-8-sig")
def main():
    spec=j(R/"evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.10.1.json")
    cfg=j(R/"evals/judge_configs/semantic_relevance_judge.v0.29.0-r5.json")
    reg=j(R/"evals/datasets/semantic_relevance_v0.29_regression_manifest.v2.4.0.json")
    judge=t(R/"evals/semantic_relevance_facet_judge.py"); runner=t(R/"evals/run_judge_v0_29_r5_development.py")
    assert spec["version"]=="0.10.1" and cfg["version"]=="0.29.0-r5" and reg["version"]=="2.4.0"
    e=reg["targeted_expectations"]; assert len(e)==60
    assert e["U_Q04_T10"]=={"judge_min":3,"judge_max":4}
    assert e["U4_Q07_T03"]=={"judge_max":1}
    assert e["U2_Q04_T70"]["judge_min"]==3 and e["U2_Q09_T02"]["judge_min"]==3
    assert e["U4_Q03_T02"]=={"judge_min":2,"judge_max":2}
    assert e["U4_Q04_T02"]=={"judge_max":0} and e["U4_Q09_T04"]=={"judge_min":2,"judge_max":2}
    assert "_Q07_PARENT_CHILD_ANCHOR_RE" in judge and "_q07_parent_child_text_anchor_guard" in judge and "_q07_parent_child_recovery_guard" in judge
    assert "_q04_r4_has_directional_departure_anchor" in judge and "_q04_r3_shipwreck_danger_component_check" in judge and "_q09_r2_resistance_pair_component_check" in judge
    assert 'JUDGE_CONFIG_VERSION = "0.29.0-r5"' in runner and 'FACET_SPEC_VERSION = "0.10.1"' in runner
    q07=next(q for q in spec["queries"] if q["query_id"]=="Q07"); pc=next(c for c in q07["facets"][0]["required_components"] if c["component_id"]=="parent_child_relationship")
    b=" ".join(pc["negative_boundaries"]).lower(); assert "romantic relationship" in b and "cannot be substituted for a parent-child relationship by analogy" in b
    print("v0.29 r5 semantic/provenance contract passed")
if __name__=="__main__": main()
