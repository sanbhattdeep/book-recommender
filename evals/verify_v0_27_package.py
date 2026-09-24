"""Static/unit verifier for the v0.27.0 r4 overlay package."""
from pathlib import Path
import hashlib, json, subprocess, sys, re

EVALS = Path(__file__).resolve().parent
ROOT = EVALS.parent

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

required = [
    EVALS / "semantic_relevance_facet_judge.py",
    EVALS / "semantic_relevance_facet_scoring.py",
    EVALS / "run_judge_development.py",
    EVALS / "run_judge_development_direct_transport.py",
    EVALS / "analyze_v0_27_targeted.py",
    EVALS / "analyze_v0_27_development.py",
    EVALS / "datasets/semantic_relevance_v0.27_regression_manifest.v1.1.0.json",
    EVALS / "datasets/semantic_relevance_post_holdout_label_revisions.v1.0.0.json",
    EVALS / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.5.json",
    EVALS / "judge_configs/semantic_relevance_judge.v0.27.0.json",
]
for path in required:
    assert path.exists(), path

config=json.loads((EVALS/"judge_configs/semantic_relevance_judge.v0.27.0.json").read_text(encoding="utf-8"))
assert config["facet_spec_version"] == "0.9.5"
assert config["version"] == "0.27.0"
assert config["text_licensed_inference_contract"]["audit_field"] == "external_knowledge_required"
assert config["text_licensed_inference_contract"]["scope"] == "component_local_after_r1_targeted_regression"
assert config["text_licensed_inference_contract"]["local_precision_guard"] == "Q09/F2 resistance identity-only text-anchor guard"
assert any(rule.get("cue_id") == "Q03_F1_promoted_growth_mode_b" for rule in config["deterministic_direct_cue_stage"]["rules"])
assert any(rule.get("cue_id") == "Q05_F1_named_world_war_lexical" for rule in config["deterministic_direct_cue_stage"]["rules"])


reg=json.loads((EVALS/"datasets/semantic_relevance_v0.27_regression_manifest.v1.1.0.json").read_text(encoding="utf-8"))
assert len(reg["targeted_expectations"]) == 26
assert reg["targeted_expectations"]["U2_Q04_T70"] == {"judge_min": 3}
assert reg["targeted_expectations"]["U2_Q05_T30"] == {"judge_min": 3}
assert reg["targeted_expectations"]["U2_Q09_T30"] == {"judge_max": 1}
assert reg["targeted_expectations"]["U2_Q04_T02"] == {"judge_min": 3}
assert reg["targeted_expectations"]["U_Q05_T30"] == {"judge_min": 1, "judge_max": 3}

targeted=(EVALS/"analyze_v0_27_targeted.py").read_text(encoding="utf-8")
assert "semantic_relevance_v0.27_regression_manifest.v1.1.0.json" in targeted
assert "semantic_relevance_v0.27_development.v3.0.0.csv" in targeted
dev_an=(EVALS/"analyze_v0_27_development.py").read_text(encoding="utf-8")
assert 'EXPECTED_JUDGE_CONFIG_VERSION = "0.27.0"' in dev_an
assert 'EXPECTED_EVALUATION_DATASET_VERSION = "3.0.0"' in dev_an
assert 'len(gold) != 120' in dev_an

runner=(EVALS/"run_judge_development.py").read_text(encoding="utf-8")
for token in ('JUDGE_CONFIG_VERSION = "0.27.0"','EVALUATION_DATASET_VERSION = "3.0.0"','FACET_SPEC_VERSION = "0.9.5"','semantic_relevance_v0_27_development'):
    assert token in runner, token
assert "len(dataset) != 120" in runner
assert 'startswith("U2_").sum()) != 60' in runner

# Guard Windows decoding regression: eval Python source may not use bare read_text().
for p in EVALS.glob("*.py"):
    source_text=p.read_text(encoding="utf-8-sig")
    assert re.search(r"\.read_text\(\s*\)", source_text) is None, f"bare read_text() without UTF-8 encoding: {p.name}"

# Compile and run deterministic contract tests. No Ollama call is made.
for p in EVALS.glob("*.py"):
    compile(p.read_text(encoding="utf-8-sig"), str(p), "exec")

for name in [
    "test_v0_27_text_licensed_inference.py",
    "test_v0_27_facet_clarifications.py",
    "test_v0_27_post_holdout_label_revisions.py",
    "test_v0_26_local_contract_isolation.py",
    "test_v0_26_suspense_boundary.py",
    "test_v0_26_grief_non_regression.py",
    "test_v0_26_parent_child_referent_binding.py",
    "test_v0_26_personal_growth_scope.py",
    "test_v0_26_r4_label_revisions.py",
]:
    subprocess.run([sys.executable, str(EVALS/name)], cwd=ROOT, check=True)

print("v0.27.0 r4 package verification passed.")
print("Judge SHA-256:", sha(EVALS/"semantic_relevance_facet_judge.py"))
print("Scoring SHA-256:", sha(EVALS/"semantic_relevance_facet_scoring.py"))
print("Facet-spec SHA-256:", sha(EVALS/"facets/semantic_relevance/semantic_relevance_query_facets.v0.9.5.json"))
print("Rubric SHA-256:", sha(EVALS/"rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"))
