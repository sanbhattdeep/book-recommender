"""Static contract for the v0.28.0 r2 full 180-case development gate."""
from pathlib import Path

R = Path(__file__).resolve().parents[1]
E = R / "evals"
runner = (R / "run_v028_r2_full_180.ps1").read_text(encoding="utf-8")
analyzer = (E / "analyze_v0_28_r2_development.py").read_text(encoding="utf-8")

for token in [
    "$Expected = 180",
    "verify_v0_28_package.py",
    "build_v0_28_r2_development_dataset.py",
    "run_judge_development.py",
    "analyze_v0_28_r2_development.py",
    "No independent v0.28 holdout exists yet.",
]:
    assert token in runner, token

for token in [
    '"judge_config_version": "0.28.0"',
    '"evaluation_dataset_version": "4.0.0"',
    "KNOWN_V027_FINAL_SEVERE",
    "U3_Q11_T02",
    "U3_Q03_T01",
    "U3_Q10_T01",
    "U3_Q11_T04",
    "len(prior150) != 150",
    "len(consumed30) != 30",
    "reference_checks(prior_m)",
    "new_severe_consumed",
    "all(targeted_checks.values())",
    "no independent v0.28 holdout exists yet",
]:
    assert token in analyzer, token

compile(analyzer, str(E / "analyze_v0_28_r2_development.py"), "exec")
print("v0.28 r2 full-180 development contract passed")
