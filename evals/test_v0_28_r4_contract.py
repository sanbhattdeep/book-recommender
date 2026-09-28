"""Static r4 contract: whole-work thematic lists are not side-detail lists. No LLM calls."""
from pathlib import Path
import json
E=Path(__file__).resolve().parent
cfg=json.loads((E/"judge_configs/semantic_relevance_judge.v0.28.0.json").read_text(encoding="utf-8"))
text="\n".join(cfg["prominence_stage"]["instructions"])
assert "WHOLE-WORK THEMATIC ENUMERATION EXCEPTION" in text
assert "this story of friendship, love, tragedy, and redemption" in text
assert "his king, his lover, his friends, his gods" in text
reg=json.loads((E/"datasets/semantic_relevance_v0.28_regression_manifest.v1.3.0.json").read_text(encoding="utf-8"))
assert len(reg["targeted_expectations"])==41
assert reg["targeted_expectations"]["U_Q01_T02"]=={"judge_min":2,"judge_max":3}
assert set(reg["diagnostic_only_cases"])=={"U3_Q10_T01","U3_Q11_T04"}
print("v0.28 r4 whole-work thematic-enumeration prominence contract passed")
