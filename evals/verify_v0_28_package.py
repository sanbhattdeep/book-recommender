"""Static/unit verifier for v0.28.0 r3. No judge-model calls."""
from pathlib import Path
import hashlib, json, subprocess, sys, re

E = Path(__file__).resolve().parent
ROOT = E.parent

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

required = [
    E / "semantic_relevance_facet_judge.py",
    E / "semantic_relevance_facet_scoring.py",
    E / "run_judge_development.py",
    E / "build_v0_28_r3_development_dataset.py",
    E / "analyze_v0_28_r3_targeted.py",
    E / "analyze_v0_28_r3_development.py",
    ROOT / "prepare_v028_r3.ps1",
    ROOT / "run_v028_r3_targeted.ps1",
    ROOT / "run_v028_r3_full_180.ps1",
    E / "datasets/semantic_relevance_v0.28_regression_manifest.v1.2.0.json",
    E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.9.json",
    E / "judge_configs/semantic_relevance_judge.v0.28.0.json",
    ROOT / "v0.28.0_r3_package_manifest.json",
]
for path in required:
    assert path.exists(), path

cfg = json.loads(
    (E / "judge_configs/semantic_relevance_judge.v0.28.0.json")
    .read_text(encoding="utf-8")
)
assert cfg["version"] == "0.28.0"
assert cfg["facet_spec_version"] == "0.9.9"

reg = json.loads(
    (E / "datasets/semantic_relevance_v0.28_regression_manifest.v1.2.0.json")
    .read_text(encoding="utf-8")
)
assert len(reg["targeted_expectations"]) == 40
assert set(reg["diagnostic_only_cases"]) == {"U3_Q10_T01", "U3_Q11_T04"}

runner = (E / "run_judge_development.py").read_text(encoding="utf-8")
for token in [
    'JUDGE_CONFIG_VERSION = "0.28.0"',
    'EVALUATION_DATASET_VERSION = "4.0.0"',
    'FACET_SPEC_VERSION = "0.9.9"',
    'len(dataset) != 180',
    'startswith("U3_").sum()) != 60',
]:
    assert token in runner, token

builder = (E / "build_v0_28_r3_development_dataset.py").read_text(encoding="utf-8")
for token in [
    "FINAL_HOLDOUT_COMPLETED",
    "judge_run_count",
    "EXPECTED_DEV150_SHA='72810fa61cc71e28f05beaec9664f2ccdee55d5834272c44e2f1cc97cafeee7d'",
    "EXPECTED_FINAL_SHA='cce944bd48b102452c024134560b7b87866b85f5fa1bc1194775138b8976eb2e'",
]:
    assert token in builder, token

# Scoring and rubric stay byte-identical to r7/r1/r2.
assert sha(E / "semantic_relevance_facet_scoring.py") == "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808"
assert sha(E / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json") == "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e"

for path in E.glob("*.py"):
    source = path.read_text(encoding="utf-8-sig")
    assert re.search(r"\.read_text\(\s*\)", source) is None, f"bare read_text(): {path.name}"
    compile(source, str(path), "exec")

for name in [
    "test_v0_28_entailment_discipline.py",
    "test_v0_28_direct_cue_prominence_guard.py",
    "test_v0_28_r3_contract.py",
    "test_v0_28_r3_localized_guards.py",
    "test_v0_28_r3_full_180_contract.py",
    "test_v0_27_r7_validation_boundaries.py",
    "test_v0_27_r6_validation_boundaries.py",
    "test_v0_27_external_knowledge_repair.py",
    "test_v0_26_suspense_boundary.py",
    "test_v0_26_grief_non_regression.py",
    "test_v0_26_parent_child_referent_binding.py",
    "test_v0_26_personal_growth_scope.py",
]:
    subprocess.run([sys.executable, str(E / name)], cwd=ROOT, check=True)

print("v0.28.0 r3 package verification passed.")
print("Judge SHA-256:", sha(E / "semantic_relevance_facet_judge.py"))
print("Scoring SHA-256:", sha(E / "semantic_relevance_facet_scoring.py"))
print("Facet-spec SHA-256:", sha(E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.9.json"))
print("Judge-config SHA-256:", sha(E / "judge_configs/semantic_relevance_judge.v0.28.0.json"))
print("Regression-manifest SHA-256:", sha(E / "datasets/semantic_relevance_v0.28_regression_manifest.v1.2.0.json"))
print("Rubric SHA-256:", sha(E / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"))
