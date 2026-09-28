"""Static restart-safe r4 stability overlay contract. No LLM calls."""
from pathlib import Path
import json
E=Path(__file__).resolve().parent
m=json.loads((E/"datasets/semantic_relevance_v0.28_r4_stability_manifest.v1.1.0.json").read_text(encoding="utf-8"))
assert m["required_fresh_runs"]==3
assert len(m["expectations"])==13
assert m["expectations"]["U_Q01_T02"]["judge_min"]==2
assert m["expectations"]["U_Q01_T02"]["judge_max"]==3
for cid in ["U3_Q03_T01","U3_Q11_T02","U_Q11_T70","U2_Q11_T70","U3_Q11_T03","U3_Q03_T04"]:
    assert cid in m["expectations"]
runner=(E.parent/"run_v028_r4_stability.ps1").read_text(encoding="utf-8-sig")
for token in ["v0.28_r4_stability_state.v1.1.0.json","--resume","STABILITY_PASS","Prepared fresh stability run directory before judge execution"]:
    assert token in runner, token
print("v0.28 r4 stability contract passed")
