"""Static verification for the v0.28.0 r4 freeze overlay. No judge calls."""
from pathlib import Path
import hashlib
import py_compile
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evals"

subprocess.run([sys.executable, str(E / "verify_v0_28_package.py")], cwd=ROOT, check=True)

expected = {
    "semantic_relevance_facet_judge.py": "2ad6fb76876c3d892de3299a7867d6d7bff8b8456975796b310e3969b9b09c1b",
    "semantic_relevance_facet_scoring.py": "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808",
    "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.9.json": "378fe3d381703a07b22256b4b5e26e578d2336391cc3e4d861591bdbe314aafe",
    "judge_configs/semantic_relevance_judge.v0.28.0.json": "82e75d61e0b1c02bc126edd40749b480a5643a966c05725802688b98dc78c64b",
    "datasets/semantic_relevance_v0.28_regression_manifest.v1.3.0.json": "28aa38338e3130a23e6b891b707e0a3ba886805d62845f5e91cea87744bd6973",
    "datasets/semantic_relevance_v0.28_r4_stability_manifest.v1.1.0.json": "9a87ed55667a6dbcc25df286aea983eaeb2c57edcf53c2f7b2c0b248d0d83bab",
    "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json": "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e",
}
for rel, expected_hash in expected.items():
    path = E / rel
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected_hash, (rel, actual, expected_hash)

for name in ["freeze_v0_28_r4_release_candidate.py", "verify_v0_28_r4_release_candidate_package.py"]:
    path = E / name
    assert path.exists(), path
    py_compile.compile(str(path), doraise=True)

freeze = (E / "freeze_v0_28_r4_release_candidate.py").read_text(encoding="utf-8")
for token in [
    "Exactly three --stability-run directories are required",
    "complete 180-case development run",
    "all 41 r4 targeted checks",
    "all eight frozen 150-case reference checks",
    "semantic_relevance_v0.28_r4_stability_manifest.v1.1.0.json",
    "FINAL_HOLDOUT_COMPLETED",
    "judge_run_count must remain exactly 1",
    "FROZEN_BEFORE_NEW_UNSEEN_V0_28_EVIDENCE",
    "NO JUDGE CALLS WERE MADE BY THIS FREEZE COMMAND",
    "identical_across_three_runs",
]:
    assert token in freeze, token

wrapper = (ROOT / "freeze_v028_r4.ps1").read_text(encoding="utf-8")
for token in [
    "20260928T150657Z",
    "20260929T045644Z_stability_r4_run1",
    "20260929T052257Z_stability_r4_run2",
    "20260929T054353Z_stability_r4_run3",
    "verify_v0_28_r4_release_candidate_package.py",
    "freeze_v0_28_r4_release_candidate.py",
]:
    assert token in wrapper, token
assert "run_judge" not in wrapper

print("v0.28.0 r4 freeze-overlay verification passed.")
print("No judge calls performed.")
