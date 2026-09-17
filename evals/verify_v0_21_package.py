from pathlib import Path
import hashlib
import json
import py_compile
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
required = [
    EVALS / "semantic_relevance_facet_judge.py",
    EVALS / "semantic_relevance_facet_scoring.py",
    EVALS / "judge_configs/semantic_relevance_judge.v0.21.0.json",
    EVALS / "facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json",
    EVALS / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json",
    EVALS / "datasets/semantic_relevance_v0.21_regression_manifest.v1.0.0.json",
]
for path in required:
    assert path.exists(), path

config = json.loads(required[2].read_text(encoding="utf-8"))
facet = json.loads(required[3].read_text(encoding="utf-8"))
rubric = json.loads(required[4].read_text(encoding="utf-8"))
assert config.get("version") == "0.21.0"
assert facet.get("version") == "0.8.0" and facet.get("status") == "frozen"
assert rubric.get("version") == "0.1.0"
assert set(rubric.get("scores", {})) == {"0", "1", "2", "3", "4"}

for path in EVALS.glob("*.py"):
    py_compile.compile(str(path), doraise=True)

tests = [
    "test_v0_21_frozen_artifacts.py",
    "test_rubric_integrity.py",
    "test_v0_21_boundary_spec.py",
    "test_v0_21_hard_exclusion_contract.py",
    "test_v0_21_full_description_precheck.py",
    "test_v0_21_role_signal_contract.py",
    "test_v0_21_audit_serialization.py",
    "test_v0_21_positive_control_manifest.py",
    "test_v0_21_no_case_specific_behavior.py",
    "test_v0_21_dataset_builder_contract.py",
    "test_facet_scoring.py",
    "test_deterministic_direct_cues.py",
    "test_polarity_aware_direct_cues.py",
    "test_evidence_candidate_selection.py",
    "test_composite_relation_consistency.py",
    "test_composite_verifier_recovery.py",
    "test_semantic_definition_prompts.py",
    "test_multi_span_composition.py",
]
for name in tests:
    subprocess.run([sys.executable, str(EVALS / name)], cwd=EVALS, check=True)

print("v0.21 package verification passed.")
print("Rubric bytes:", required[4].stat().st_size)
print("Rubric SHA-256:", hashlib.sha256(required[4].read_bytes()).hexdigest())
