"""Static contract checks for the v0.28.0 r3 stability overlay.

No judge calls are made. These checks pin the selected cases, thresholds, and
restart-safe runner behavior before the stability evidence is generated.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MANIFEST = HERE / "datasets" / "semantic_relevance_v0.28_r3_stability_manifest.v1.0.0.json"
RUNNER = REPO / "run_v028_r3_stability.ps1"
ANALYZER = HERE / "analyze_v0_28_r3_stability.py"

EXPECTED = {
    "U3_Q01_T03": (0, 0),
    "U_Q01_T02": (2, 3),
    "U3_Q02_T05": (0, 0),
    "U_Q02_T02": (3, 4),
    "U3_Q07_T01": (1, 1),
    "U_Q07_T02": (3, 4),
    "U3_Q05_T01": (2, 3),
    "U3_Q03_T01": (0, 0),
    "U3_Q11_T02": (0, 1),
    "U_Q11_T70": (1, 3),
    "U2_Q11_T70": (1, 3),
    "U3_Q11_T03": (1, 3),
    "U3_Q03_T04": (0, 1),
}


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["judge_version"] == "0.28.0"
    assert payload["facet_spec_version"] == "0.9.9"
    assert payload["dataset_version"] == "4.0.0"
    assert payload["required_fresh_runs"] == 3
    assert payload["require_identical_scores"] is False

    got = {
        case_id: (int(spec["judge_min"]), int(spec["judge_max"]))
        for case_id, spec in payload["expectations"].items()
    }
    assert got == EXPECTED, (got, EXPECTED)

    runner = RUNNER.read_text(encoding="utf-8")
    assert "build_v0_28_r3_development_dataset.py" in runner
    assert "test_v0_28_r3_stability_contract.py" in runner
    assert "analyze_v0_28_r3_stability.py" in runner
    assert "semantic_relevance_v0.28_r3_stability_state.v1.0.0.json" in runner
    assert "--resume" in runner
    assert "New-Item" in runner  # pre-creates run dirs before judge calls for reboot safety
    assert "build_v0_27_r7_development_dataset.py" not in runner
    assert "LOCKED_DO_NOT_RUN" not in runner

    compile(ANALYZER.read_text(encoding="utf-8"), str(ANALYZER), "exec")
    print("v0.28 r3 stability overlay contract passed")


if __name__ == "__main__":
    main()
