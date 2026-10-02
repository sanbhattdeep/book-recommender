"""Static contract for the corrected v0.29-r3 full-240 analyzer."""
from pathlib import Path

R = Path(__file__).resolve().parents[1]
P = R / "evals/analyze_v0_29_r3_development.py"
T = P.read_text(encoding="utf-8")

assert "semantic_relevance_v0.29_development.v6.1.0.csv" in T
assert "semantic_relevance_v0.29_regression_manifest.v2.2.0.json" in T
assert '"judge_config_version": "0.29.0-r3"' in T
assert '"facet_spec_version": "0.10.1"' in T
assert "Expected 58 gated regression expectations" in T

# Correct r8 closeout schema.
assert 'closeout.get("closeout_status")' in T
assert 'EVALUATED_NOT_FINAL_QUALIFIED' in T
assert 'closeout.get("final_holdout_decision")' in T
assert 'FINAL_HOLDOUT_REVIEW' in T
assert 'evidence_boundary_for_next_lineage' in T
assert 'r8_final_holdout_may_be_reused_as_independent_evidence' in T

# Correct v0.29 development provenance partitions.
assert '"consumed_v0_26_final_holdout_30"' in T
assert '"consumed_v0_27_final_holdout_30"' not in T
assert "original_development_150" in T
assert "consumed_v0_26_final_30" in T
assert "consumed_r4_validation_30" in T
assert "consumed_v0_28_final_30" in T

assert "development_regression_gate_pass" in T
assert "new_independent_evidence_required" in T
assert "FINAL_HOLDOUT_COMPLETED" in T

compile(T, str(P), "exec")
print("v0.29 r3 full-240 analyzer contract passed")
