from pathlib import Path
import hashlib
import json
import py_compile
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evals"
required = [
    E / "semantic_relevance_facet_judge.py",
    E / "semantic_relevance_facet_scoring.py",
    E / "judge_configs/semantic_relevance_judge.v0.25.0.json",
    E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.0.json",
    E / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json",
    E / "datasets/semantic_relevance_v0.25_regression_manifest.v4.0.0.json",
    E / "datasets/semantic_relevance_label_revisions.v1.0.0.json",
    E / "analyze_v0_25_targeted.py",
    E / "analyze_v0_25_development.py",
    E / "run_judge_development.py",
    E / "run_judge_development_direct_transport.py",
    E / "build_v0_25_development_dataset.py",
]
for path in required:
    assert path.exists(), path

cfg = json.loads((E / "judge_configs/semantic_relevance_judge.v0.25.0.json").read_text(encoding="utf-8"))
facet_payload = json.loads((E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.0.json").read_text(encoding="utf-8"))
assert cfg["version"] == "0.25.0"
assert cfg["dataset_version"] == "0.5.0"  # calibration provenance remains pinned
assert cfg["facet_spec_version"] == "0.9.0"
assert cfg["rubric_version"] == "0.1.0"
assert "canonical_component_contract" in cfg["pipeline"]
assert cfg["composite_verification_stage"]["min_spans"] == 1
assert cfg["composite_verification_stage"]["max_spans"] == 4
assert "book_subject_stage" in cfg

assert facet_payload["version"] == "0.9.0"
assert facet_payload["status"] == "frozen"
component_count = 0
for query in facet_payload["queries"]:
    for facet in query["facets"]:
        components = facet.get("required_components", [])
        assert components, (query["query_id"], facet["facet_id"])
        ids = [component["component_id"] for component in components]
        assert len(ids) == len(set(ids)), (query["query_id"], facet["facet_id"], ids)
        component_count += len(components)
assert component_count >= 34

runner_text = (E / "run_judge_development.py").read_text(encoding="utf-8")
assert 'JUDGE_CONFIG_VERSION = "0.25.0"' in runner_text
assert 'EVALUATION_DATASET_VERSION = "2.1.0"' in runner_text
assert 'JUDGE_CALIBRATION_DATASET_VERSION = "0.5.0"' in runner_text
assert 'FACET_SPEC_VERSION = "0.9.0"' in runner_text
assert '"component_evidence_ledger_json"' in runner_text
assert "v0.25 facet is missing required_components" in runner_text

judge_text = (E / "semantic_relevance_facet_judge.py").read_text(encoding="utf-8")
for required_text in [
    "class ComponentEvidenceCheck",
    "component_id: str",
    "def format_required_components",
    "def _validate_component_evidence_ledger",
    "component_checks must exactly match the frozen canonical",
    "canonical component resurrection is forbidden",
    "FROZEN CANONICAL REQUIRED COMPONENTS",
    "Full-context recovery MAY inspect any supplied description span",
    "Unable to obtain a structurally valid full-context verification",
    "component_evidence_ledger_by_id",
]:
    assert required_text in judge_text, required_text

# Direct transport remains bounded and uses JSON mode for the historically
# problematic composite schema.
direct_text = (E / "run_judge_development_direct_transport.py").read_text(encoding="utf-8")
assert 'schema_name == "CompositeEvidenceVerification"' in direct_text
assert '"num_predict": self.num_predict' in direct_text
assert 'format_mode = "json" if composite_json_mode else "json_schema"' in direct_text

# Rubric remains byte-identical; deterministic scoring behavior is covered by
# test_facet_scoring.py even though the scoring module now carries the new
# FacetRequiredComponent schema and a 1-span full-context audit allowance.
rubric_path = E / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
assert hashlib.sha256(rubric_path.read_bytes()).hexdigest() == "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e"

for path in E.glob("*.py"):
    py_compile.compile(str(path), doraise=True)

for name in [
    "test_v0_22_frozen_artifacts.py",
    "test_v0_22_subject_contract.py",
    "test_v0_22_role_contract.py",
    "test_v0_24_component_completeness.py",
    "test_v0_24_composite_monotonicity.py",
    "test_v0_24_valid_hard_exclusion_canonicalization.py",
    "test_v0_25_canonical_component_contract.py",
    "test_v0_25_full_context_component_recovery.py",
    "test_v0_25_component_boundaries.py",
    "test_facet_scoring.py",
    "test_evidence_candidate_selection.py",
]:
    subprocess.run([sys.executable, str(E / name)], cwd=E, check=True)


print("v0.25.0 package verification passed.")
print("Scoring SHA-256:", hashlib.sha256((E / "semantic_relevance_facet_scoring.py").read_bytes()).hexdigest())
print("Facet-spec SHA-256:", hashlib.sha256((E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.0.json").read_bytes()).hexdigest())
print("Rubric SHA-256:", hashlib.sha256(rubric_path.read_bytes()).hexdigest())
