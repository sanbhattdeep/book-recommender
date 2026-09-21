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
    E / "judge_configs/semantic_relevance_judge.v0.26.0.json",
    E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.0.json",
    E / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json",
    E / "datasets/semantic_relevance_v0.26_regression_manifest.v5.0.0.json",
    E / "datasets/semantic_relevance_label_revisions.v1.0.0.json",
    E / "analyze_v0_26_targeted.py",
    E / "analyze_v0_26_development.py",
    E / "run_judge_development.py",
    E / "run_judge_development_direct_transport.py",
    E / "build_v0_26_development_dataset.py",
    E / "test_v0_26_component_isolation.py",
    E / "test_v0_26_deterministic_facet_assembly.py",
    E / "test_v0_26_missing_component_recovery.py",
]
for path in required:
    assert path.exists(), path

cfg = json.loads((E / "judge_configs/semantic_relevance_judge.v0.26.0.json").read_text(encoding="utf-8"))
facet_path = E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.0.json"
facet_payload = json.loads(facet_path.read_text(encoding="utf-8"))
rubric_path = E / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"

assert cfg["version"] == "0.26.0"
assert cfg["dataset_version"] == "0.5.0"  # calibration provenance remains pinned
assert cfg["facet_spec_version"] == "0.9.0"
assert cfg["rubric_version"] == "0.1.0"
assert "isolated_component_verification" in cfg["pipeline"]
assert "deterministic_facet_assembly" in cfg["pipeline"]
assert "missing_component_only_full_context_recovery" in cfg["pipeline"]
assert cfg["verification_stage"]["v0_26_component_contract"]["full_facet_relation_owner"] == "python_deterministic"
assert cfg["composite_verification_stage"]["v0_26_recovery_contract"]["recover_missing_components_only"] is True
assert cfg["composite_verification_stage"]["v0_26_recovery_contract"]["may_override_established_components"] is False

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

# Frozen semantic artifacts remain unchanged from v0.25.
assert hashlib.sha256(facet_path.read_bytes()).hexdigest() == "26739c6eb818c7fb7ddbbf3ae1e9e391111b70307b8f3f44f564d1d969c11c1f"
assert hashlib.sha256(rubric_path.read_bytes()).hexdigest() == "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e"

runner_text = (E / "run_judge_development.py").read_text(encoding="utf-8")
assert 'JUDGE_CONFIG_VERSION = "0.26.0"' in runner_text
assert 'EVALUATION_DATASET_VERSION = "2.1.0"' in runner_text
assert 'JUDGE_CALIBRATION_DATASET_VERSION = "0.5.0"' in runner_text
assert 'FACET_SPEC_VERSION = "0.9.0"' in runner_text
assert 'semantic_relevance_v0.26_development' in runner_text
assert 'v0.26 facet is missing required_components' in runner_text
assert '"component_evidence_ledger_json"' in runner_text

judge_text = (E / "semantic_relevance_facet_judge.py").read_text(encoding="utf-8")
for required_text in [
    "class IsolatedComponentVerification",
    "class FullContextComponentRecovery",
    "def build_isolated_component_prompt",
    "def verify_isolated_component",
    "def _assemble_single_span_from_component_results",
    "def build_missing_component_recovery_prompt",
    "def recover_missing_component",
    "Production v0.26 facets are decomposed into independent component calls",
    "No holistic facet verifier was used",
    "positive result must cite at least one span not previously audited for this missing component",
    "grounding_relation",
]:
    assert required_text in judge_text, required_text

# The production branch must execute before the retained legacy holistic fixture path.
component_branch = judge_text.index("if facet.required_components:", judge_text.index("def verify_candidate_evidence"))
legacy_marker = judge_text.index("Legacy fixture compatibility only", component_branch)
assert component_branch < legacy_marker

# Direct recovery transport remains bounded; use JSON mode for the schemas most
# likely to trigger expensive constrained decoding.
direct_text = (E / "run_judge_development_direct_transport.py").read_text(encoding="utf-8")
assert '"CompositeEvidenceVerification"' in direct_text
assert '"FullContextComponentRecovery"' in direct_text
assert '"num_predict": self.num_predict' in direct_text
assert 'format_mode = "json" if json_mode_schema else "json_schema"' in direct_text

# Compile every Python file before behavioral contract tests.
for path in E.glob("*.py"):
    py_compile.compile(str(path), doraise=True)

# Focused architecture tests + frozen regression contracts that do not require Ollama.
for name in [
    "test_v0_26_component_isolation.py",
    "test_v0_26_deterministic_facet_assembly.py",
    "test_v0_26_missing_component_recovery.py",
    "test_v0_25_component_boundaries.py",
    "test_v0_24_component_completeness.py",
    "test_v0_24_component_evidence_ledger.py",
    "test_v0_24_valid_hard_exclusion_canonicalization.py",
    "test_v0_24_direct_transport_contract.py",
    "test_v0_24_label_revision_manifest.py",
    "test_v0_22_subject_contract.py",
    "test_v0_22_subject_reuse.py",
    "test_v0_22_role_contract.py",
    "test_deterministic_direct_cues.py",
    "test_polarity_aware_direct_cues.py",
    "test_evidence_candidate_selection.py",
    "test_facet_scoring.py",
]:
    subprocess.run([sys.executable, str(E / name)], cwd=E, check=True)

print("v0.26.0 package verification passed.")
print("Judge SHA-256:", hashlib.sha256((E / "semantic_relevance_facet_judge.py").read_bytes()).hexdigest())
print("Facet-spec SHA-256:", hashlib.sha256(facet_path.read_bytes()).hexdigest())
print("Rubric SHA-256:", hashlib.sha256(rubric_path.read_bytes()).hexdigest())
