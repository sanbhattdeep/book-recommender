from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.11.json"
CONFIG = ROOT / "evals/judge_configs/semantic_relevance_judge.v0.28.0-r6.json"
REG = ROOT / "evals/datasets/semantic_relevance_v0.28_regression_manifest.v1.5.0.json"
ADJ = ROOT / "evals/datasets/semantic_relevance_v0.28_post_validation_label_adjudications.v1.0.0.json"

def main():
    spec=json.loads(SPEC.read_text(encoding="utf-8"))
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"))
    reg=json.loads(REG.read_text(encoding="utf-8"))
    adj=json.loads(ADJ.read_text(encoding="utf-8"))
    assert spec["version"]=="0.9.11"
    assert cfg["version"]=="0.28.0-r6"
    assert cfg["facet_spec_version"]=="0.9.11"
    assert len(reg["targeted_expectations"])==51
    assert set(reg["r5_new_targets"])=={"U4_Q02_T05","U4_Q06_T02","U4_Q10_T02","U4_Q11_T03","U4_Q12_T05"}
    assert set(reg["diagnostic_only_cases"])=={"U3_Q10_T01","U3_Q11_T04"}
    a=adj["adjudications"][0]
    assert a["case_id"]=="U4_Q12_T05" and a["original_human_score"]==0 and a["adjudicated_human_score"]==1
    assert adj["r4_validation_result_remains_immutable"] is True
    assert set(reg["r6_repair_targets"])=={"U_Q01_T02","U_Q11_T02"}
    print("v0.28 r6 semantic/provenance contract passed")
if __name__=="__main__":
    main()
