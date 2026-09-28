"""Static contract for the v0.28.0 r3 targeted and full 180-case gates."""
from pathlib import Path

R = Path(__file__).resolve().parents[1]
E = R / "evals"
targeted = (R / "run_v028_r3_targeted.ps1").read_text(encoding="utf-8-sig")
full = (R / "run_v028_r3_full_180.ps1").read_text(encoding="utf-8-sig")
analyzer = (E / "analyze_v0_28_r3_development.py").read_text(encoding="utf-8")

for token in [
    "semantic_relevance_v0.28_regression_manifest.v1.2.0.json",
    "Expected 42 executed cases (40 gated + 2 diagnostic)",
    "$ids.Count -ne 42",
    "analyze_v0_28_r3_targeted.py",
]:
    assert token in targeted, token

for token in [
    "$Expected = 180",
    "verify_v0_28_package.py",
    "build_v0_28_r3_development_dataset.py",
    "run_judge_development.py",
    "analyze_v0_28_r3_development.py",
    "No independent v0.28 holdout exists yet.",
]:
    assert token in full, token

for token in [
    '"judge_config_version": "0.28.0"',
    '"evaluation_dataset_version": "4.0.0"',
    '"facet_spec_version": "0.9.9"',
    "KNOWN_V027_FINAL_SEVERE",
    "len(prior150) != 150",
    "len(consumed30) != 30",
    "reference_checks(prior_m)",
    "new_severe_consumed",
    "all(targeted_checks.values())",
    "no independent v0.28 holdout exists yet",
]:
    assert token in analyzer, token

compile(analyzer, str(E / "analyze_v0_28_r3_development.py"), "exec")
print("v0.28 r3 targeted/full-180 gate contract passed")
