"""Static contract for v0.28.0 r8 stability overlay."""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
M = R / "evals/datasets/semantic_relevance_v0.28_r8_stability_manifest.v1.0.0.json"
P = R / "run_v028_r8_stability.ps1"
A = R / "evals/analyze_v0_28_r8_stability.py"

def main():
    manifest = json.loads(M.read_text(encoding="utf-8"))
    assert manifest["judge_version"] == "0.28.0-r8"
    assert manifest["facet_spec_version"] == "0.9.13"
    assert manifest["dataset_version"] == "5.0.0"
    assert manifest["required_fresh_runs"] == 3
    assert manifest["max_observed_score_span"] == 1
    assert len(manifest["expectations"]) == 20
    assert manifest["expectations"]["U2_Q11_T02"] == {
        "judge_min": 3,
        "judge_max": 4,
        "role": "r8_cross_span_classical_myth_positive",
    }
    assert manifest["expectations"]["U4_Q11_T03"]["judge_max"] == 1
    assert manifest["expectations"]["U4_Q12_T05"]["judge_min"] == 1
    assert manifest["expectations"]["U4_Q12_T05"]["judge_max"] == 1

    ps = P.read_text(encoding="utf-8-sig")
    for token in [
        "verify_v0_28_r8_package.py",
        "build_v0_28_r8_development_dataset.py",
        "analyze_v0_28_r8_development.py",
        "run_judge_v0_28_r8_development.py",
        "analyze_v0_28_r8_stability.py",
        "semantic_relevance_v0.28_r8_full210_state.v1.0.0.json",
        "semantic_relevance_v0.28_r8_stability_state.v1.0.0.json",
        "Expected exactly 20 stability cases",
        "Final holdout remains LOCKED_DO_NOT_RUN",
    ]:
        assert token in ps, token

    analyzer = A.read_text(encoding="utf-8")
    for token in [
        'EXPECTED_JUDGE_CONFIG_VERSION = "0.28.0-r8"',
        'EXPECTED_FACET_SPEC_VERSION = "0.9.13"',
        'EXPECTED_DATASET_VERSION = "5.0.0"',
        "max_observed_score_span",
        "span <= max_span",
    ]:
        assert token in analyzer, token

    compile(analyzer, str(A), "exec")
    print("v0.28 r8 stability overlay contract passed")

if __name__ == "__main__":
    main()
