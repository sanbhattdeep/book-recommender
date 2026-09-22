from pathlib import Path
import hashlib
import json
import py_compile
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evals"

required = [
    E / "semantic_relevance_facet_judge.py",
    E / "semantic_relevance_facet_scoring.py",
    E / "judge_configs/semantic_relevance_judge.v0.26.0.json",
    E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.1.json",
    E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.2.json",
    E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.3.json",
    E / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json",
    E / "datasets/semantic_relevance_v0.26_regression_manifest.v6.0.0.json",
    E / "datasets/semantic_relevance_v0.26_regression_manifest.v6.1.0.json",
    E / "datasets/semantic_relevance_label_revisions.v1.1.0.json",
    E / "analyze_v0_26_targeted.py",
    E / "analyze_v0_26_development.py",
    E / "run_judge_development.py",
    E / "run_judge_development_direct_transport.py",
    E / "build_v0_26_development_dataset.py",
    E / "test_v0_26_component_isolation.py",
    E / "test_v0_26_deterministic_facet_assembly.py",
    E / "test_v0_26_missing_component_recovery.py",
    E / "test_v0_26_suspense_boundary.py",
    E / "test_v0_26_grief_non_regression.py",
    E / "test_v0_26_parent_child_referent_binding.py",
    E / "test_v0_26_personal_growth_scope.py",
    E / "test_v0_26_r4_label_revisions.py",
]
for path in required:
    assert path.exists(), path

cfg = json.loads((E / "judge_configs/semantic_relevance_judge.v0.26.0.json").read_text(encoding="utf-8"))
old_facet_path = E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.2.json"
facet_path = E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.3.json"
old_facet_payload = json.loads(old_facet_path.read_text(encoding="utf-8"))
facet_payload = json.loads(facet_path.read_text(encoding="utf-8"))
rubric_path = E / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
label_path = E / "datasets/semantic_relevance_label_revisions.v1.1.0.json"
regression_path = E / "datasets/semantic_relevance_v0.26_regression_manifest.v6.1.0.json"

assert cfg["version"] == "0.26.0"
assert cfg["dataset_version"] == "0.5.0"
assert cfg["facet_spec_version"] == "0.9.3"
assert cfg["rubric_version"] == "0.1.0"
assert "isolated_component_verification" in cfg["pipeline"]
assert "deterministic_facet_assembly" in cfg["pipeline"]
assert "missing_component_only_full_context_recovery" in cfg["pipeline"]
assert cfg["verification_stage"]["v0_26_component_contract"]["full_facet_relation_owner"] == "python_deterministic"
assert cfg["composite_verification_stage"]["v0_26_recovery_contract"]["recover_missing_components_only"] is True
assert cfg["composite_verification_stage"]["v0_26_recovery_contract"]["may_override_established_components"] is False

assert facet_payload["version"] == "0.9.3"
assert facet_payload["development_dataset_version"] == "2.2.0"
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


def facet_map(payload):
    return {
        (q["query_id"], f["facet_id"]): f
        for q in payload["queries"]
        for f in q["facets"]
    }

# r5 preserves canonical IDs everywhere and changes semantic content only for
# Q03/F1 relative to v0.9.2. Q02 suspense and Q07 referent binding stay frozen.
old_map = facet_map(old_facet_payload)
new_map = facet_map(facet_payload)
assert old_map.keys() == new_map.keys()
changed = set()
for key in old_map:
    old_ids = [c["component_id"] for c in old_map[key]["required_components"]]
    new_ids = [c["component_id"] for c in new_map[key]["required_components"]]
    assert old_ids == new_ids, key
    if old_map[key] != new_map[key]:
        changed.add(key)
assert changed == {("Q03", "F1")}, changed
assert old_map[("Q02", "F1")] == new_map[("Q02", "F1")]
assert old_map[("Q07", "F1")] == new_map[("Q07", "F1")]

q03 = new_map[("Q03", "F1")]
q03_text = q03["semantic_definition"].lower()
assert "two independently sufficient" in q03_text
assert "mode a" in q03_text and "mode b" in q03_text
assert "promotes maturity and growth" in q03_text
assert "does not require evidence that a named individual has already completed" in q03_text
q03_growth = next(c for c in q03["required_components"] if c["component_id"] == "actual_self_development_or_self_understanding")
component_text = q03_growth["definition"].lower()
assert "two independently sufficient positive modes" in component_text
assert "either mode establishes this component" in component_text
assert "promotes maturity and growth" in component_text
assert "do not require proof that a named individual has already completed" in component_text
assert any("prospective" in x.lower() and "explicit promotion" in x.lower() for x in q03_growth["negative_boundaries"])

# Byte-level contracts. Scoring/rubric remain frozen; judge changes only for
# explicit spec-fidelity wording; facet spec advances to v0.9.3.
assert hashlib.sha256((E / "semantic_relevance_facet_judge.py").read_bytes()).hexdigest() == "af005258748c825bc8d66638169de41ab61d1bb3171b67b5d1a770f1400c1fab"
assert hashlib.sha256((E / "semantic_relevance_facet_scoring.py").read_bytes()).hexdigest() == "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808"
assert hashlib.sha256(facet_path.read_bytes()).hexdigest() == "1821180bdbd9284db67d9bea661e69049713c29051b1ba5ea297d41690735c01"
assert hashlib.sha256(rubric_path.read_bytes()).hexdigest() == "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e"

labels = json.loads(label_path.read_text(encoding="utf-8"))
assert labels["version"] == "1.1.0"
label_map = {x["case_id"]: x for x in labels["revisions"]}
assert set(label_map) == {"U2_Q03_T30", "U2_Q01_T02", "U2_Q05_T10"}
assert (label_map["U2_Q01_T02"]["previous_human_score"], label_map["U2_Q01_T02"]["new_human_score"]) == (2, 1)
assert (label_map["U2_Q05_T10"]["previous_human_score"], label_map["U2_Q05_T10"]["new_human_score"]) == (4, 2)

reg = json.loads(regression_path.read_text(encoding="utf-8"))
assert reg["version"] == "6.1.0"
assert len(reg["targeted_expectations"]) == 21
assert reg["targeted_expectations"]["U_Q03_T02"] == {"judge_min": 2, "judge_max": 2}
assert reg["targeted_expectations"]["U_Q07_T02"]["judge_min"] == 3
assert reg["targeted_expectations"]["U2_Q05_T10"] == {"judge_min": 2, "judge_max": 2}

runner_text = (E / "run_judge_development.py").read_text(encoding="utf-8")
assert 'JUDGE_CONFIG_VERSION = "0.26.0"' in runner_text
assert 'EVALUATION_DATASET_VERSION = "2.2.0"' in runner_text
assert 'JUDGE_CALIBRATION_DATASET_VERSION = "0.5.0"' in runner_text
assert 'FACET_SPEC_VERSION = "0.9.3"' in runner_text
assert 'semantic_relevance_v0.26_development' in runner_text
assert 'v0.26 facet is missing required_components' in runner_text
assert '"component_evidence_ledger_json"' in runner_text

builder_text = (E / "build_v0_26_development_dataset.py").read_text(encoding="utf-8")
assert 'semantic_relevance_v0.26_development.v2.2.0.csv' in builder_text
assert 'semantic_relevance_label_revisions.v1.1.0.json' in builder_text
assert 'combined["dataset_version"] = "2.2.0"' in builder_text

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
    "The frozen component definition is binding",
    "Do not silently replace it with a stricter criterion",
]:
    assert required_text in judge_text, required_text

assert judge_text.count("The frozen component definition is binding") == 2
# Preserve r3/r4 removal of the over-broad r2 global negative-boundary wording.
assert "NEGATIVE-BOUNDARY PRIORITY" not in judge_text
assert "For example, an important or consequential personal decision" not in judge_text

component_branch = judge_text.index("if facet.required_components:", judge_text.index("def verify_candidate_evidence"))
legacy_marker = judge_text.index("Legacy fixture compatibility only", component_branch)
assert component_branch < legacy_marker

direct_text = (E / "run_judge_development_direct_transport.py").read_text(encoding="utf-8")
assert '"CompositeEvidenceVerification"' in direct_text
assert '"FullContextComponentRecovery"' in direct_text
assert '"num_predict": self.num_predict' in direct_text
assert 'format_mode = "json" if json_mode_schema else "json_schema"' in direct_text

# Windows portability contract: package sources must not rely on the platform
# default encoding when reading text artifacts. Windows commonly defaults to
# cp1252, while the frozen JSON/config artifacts are UTF-8.
for path in E.glob("*.py"):
    source_text = path.read_text(encoding="utf-8")
    assert re.search(r"\.read_text\(\s*\)", source_text) is None, f'bare read_text() without UTF-8 encoding: {path.name}'
    py_compile.compile(str(path), doraise=True)

for name in [
    "test_v0_26_component_isolation.py",
    "test_v0_26_deterministic_facet_assembly.py",
    "test_v0_26_missing_component_recovery.py",
    "test_v0_26_suspense_boundary.py",
    "test_v0_26_grief_non_regression.py",
    "test_v0_26_parent_child_referent_binding.py",
    "test_v0_26_personal_growth_scope.py",
    "test_v0_26_r4_label_revisions.py",
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

print("v0.26.0 r5 winfix2 package verification passed.")
print("Judge SHA-256:", hashlib.sha256((E / "semantic_relevance_facet_judge.py").read_bytes()).hexdigest())
print("Scoring SHA-256:", hashlib.sha256((E / "semantic_relevance_facet_scoring.py").read_bytes()).hexdigest())
print("Facet-spec SHA-256:", hashlib.sha256(facet_path.read_bytes()).hexdigest())
print("Rubric SHA-256:", hashlib.sha256(rubric_path.read_bytes()).hexdigest())
