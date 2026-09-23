"""Static contract checks for v0.27 post-holdout label revisions."""
from __future__ import annotations

from pathlib import Path
import json

EVALS = Path(__file__).resolve().parent
DATASETS = EVALS / "datasets"
REV = DATASETS / "semantic_relevance_post_holdout_label_revisions.v1.0.0.json"
REG = DATASETS / "semantic_relevance_v0.27_regression_manifest.v1.0.0.json"

revisions = json.loads(REV.read_text(encoding="utf-8"))
expected = {
    "U2_Q07_T10": (4, 0),
    "U2_Q06_T10": (3, 0),
    "U2_Q01_T10": (2, 0),
    "U2_Q01_T70": (2, 0),
}
actual = {
    str(r["case_id"]): (int(r["previous_human_score"]), int(r["new_human_score"]))
    for r in revisions["revisions"]
}
assert actual == expected, (actual, expected)
assert revisions["review_method"] == "post_holdout_human_readjudication"
assert "MUST NOT" in revisions["methodology_note"]

reg = json.loads(REG.read_text(encoding="utf-8"))
targets = reg["targeted_expectations"]
assert len(targets) == 24, len(targets)
assert targets["U2_Q04_T70"]["judge_min"] == 3
assert targets["U2_Q05_T30"]["judge_min"] == 3
assert targets["U2_Q09_T30"]["judge_max"] == 1
for cid in expected:
    assert cid not in reg["new_v0_27_targets"]

print("v0.27 post-holdout label-revision and regression-target contracts passed.")
