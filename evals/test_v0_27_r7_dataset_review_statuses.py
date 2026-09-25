"""Unit contract for r7 consumed-development review provenance. No judge calls."""
from pathlib import Path
import importlib.util
import pandas as pd

EVALS=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("r7_builder",EVALS/"build_v0_27_r7_development_dataset.py")
mod=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)

audit=mod._revision_audit_map(mod.BLIND_REVISIONS,mod.POST_HOLDOUT_REVISIONS)
assert audit["U2_Q03_T30"]=={"review_status":"ADJUDICATED_BLIND_REVIEW","new_human_score":0}
assert audit["U2_Q07_T10"]=={"review_status":"POST_HOLDOUT_READJUDICATED","new_human_score":0}

# Exact audit statuses + revised scores are valid.
rows=[{"case_id":cid,"review_status":item["review_status"],"human_score":item["new_human_score"]} for cid,item in audit.items()]
rows.append({"case_id":"U_Q99_TEST","review_status":"LABELLED","human_score":2})
frame=pd.DataFrame(rows)
mod.validate_development_review_statuses(frame)

# Normalized LABELLED is also valid for audited cases when the revised score is present.
normalized=frame.copy()
normalized.loc[normalized.case_id.isin(audit),"review_status"]="LABELLED"
mod.validate_development_review_statuses(normalized)

# Mixed persistence is valid too (matches the real v0.27 source-history pattern).
mixed=frame.copy()
for cid in ["U2_Q01_T02","U2_Q03_T30","U2_Q05_T10"]:
    mixed.loc[mixed.case_id.eq(cid),"review_status"]="LABELLED"
mod.validate_development_review_statuses(mixed)

# An unaudited non-LABELLED status must fail.
bad=frame.copy(); bad.loc[bad.case_id.eq("U_Q99_TEST"),"review_status"]="POST_HOLDOUT_READJUDICATED"
try:
    mod.validate_development_review_statuses(bad)
except ValueError:
    pass
else:
    raise AssertionError("Unaudited non-LABELLED development status must fail")

# An audited case cannot carry some other audit-status family.
bad2=frame.copy(); bad2.loc[bad2.case_id.eq("U2_Q03_T30"),"review_status"]="POST_HOLDOUT_READJUDICATED"
try:
    mod.validate_development_review_statuses(bad2)
except ValueError:
    pass
else:
    raise AssertionError("Audited case with the wrong audit status must fail")

# Most importantly, LABELLED must not mask a stale pre-revision score.
bad3=normalized.copy(); bad3.loc[bad3.case_id.eq("U2_Q03_T30"),"human_score"]=2
try:
    mod.validate_development_review_statuses(bad3)
except ValueError:
    pass
else:
    raise AssertionError("Audited case with stale human_score must fail even when status is LABELLED")

mod.validate_validation_review_statuses(pd.DataFrame([{"case_id":"U3_X","review_status":"LABELLED"}]))
print("v0.27 r7 dataset review-provenance contract passed")
