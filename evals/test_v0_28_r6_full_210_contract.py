"""Static contract for v0.28.0 r6 canonical full-210 execution."""
from pathlib import Path

R = Path(__file__).resolve().parents[1]
full = (R / "run_v028_r6_full_210.ps1").read_text(encoding="utf-8-sig")
analyzer = (R / "evals/analyze_v0_28_r6_development.py").read_text(encoding="utf-8")

for token in [
    "$Expected = 210",
    "verify_v0_28_r6_package.py",
    "build_v0_28_r6_development_dataset.py",
    "analyze_v0_28_r6_targeted.py",
    "run_judge_v0_28_r6_development.py",
    "analyze_v0_28_r6_development.py",
    "semantic_relevance_v0.28_r6_full210_state.v1.0.0.json",
    "Final holdout remains LOCKED_DO_NOT_RUN",
]:
    assert token in full, token

for token in [
    '"judge_config_version": "0.28.0-r6"',
    '"evaluation_dataset_version": "5.0.0"',
    '"facet_spec_version": "0.9.11"',
    "len(prior150) != 150",
    "len(consumed_v027_final30) != 30",
    "len(consumed_r4_validation30) != 30",
    "len(expectations) != 51",
    "KNOWN_V027_FINAL_SEVERE",
    "KNOWN_R4_VALIDATION_SEVERE",
    "reference_checks(prior_m)",
    "new_severe_old_final",
    "new_severe_r4_validation",
    "assert_final_holdout_untouched()",
]:
    assert token in analyzer, token

compile(analyzer, str(R / "evals/analyze_v0_28_r6_development.py"), "exec")
print("v0.28 r6 full-210 execution contract passed")
