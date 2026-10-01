"""Static semantic/provenance contract for v0.29.0-r2.

Windows-safe UTF-8 reads and semantic validation of inherited expectation
bounds. Do not require fields that the inherited manifest never defined.
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def main() -> None:
    spec = read_json(
        R / "evals/facets/semantic_relevance/"
            "semantic_relevance_query_facets.v0.10.1.json"
    )
    cfg = read_json(
        R / "evals/judge_configs/semantic_relevance_judge.v0.29.0-r2.json"
    )
    reg = read_json(
        R / "evals/datasets/"
            "semantic_relevance_v0.29_regression_manifest.v2.1.0.json"
    )
    judge = read_utf8(R / "evals/semantic_relevance_facet_judge.py")
    runner = read_utf8(R / "evals/run_judge_v0_29_r2_development.py")

    assert spec["version"] == "0.10.1"
    assert cfg["version"] == "0.29.0-r2"
    assert cfg["facet_spec_version"] == "0.10.1"
    assert reg["version"] == "2.1.0"
    assert len(reg["targeted_expectations"]) == 58

    # Inherited r8 expectation is lower-bound only; do not invent judge_max.
    q04 = reg["targeted_expectations"]["U2_Q04_T70"]
    assert int(q04["judge_min"]) == 3
    assert "judge_max" not in q04 or int(q04["judge_max"]) >= 3

    q09 = reg["targeted_expectations"]["U2_Q09_T02"]
    assert int(q09["judge_min"]) == 3
    if "judge_max" in q09:
        assert int(q09["judge_max"]) >= 3

    # New r1 repair protections remain exact where intentionally specified.
    assert reg["targeted_expectations"]["U4_Q04_T02"] == {
        "judge_max": 0,
    }
    assert reg["targeted_expectations"]["U4_Q09_T04"] == {
        "judge_min": 2,
        "judge_max": 2,
    }

    assert "_q04_r2_cross_span_movement_component_check" in judge
    assert "_q09_r2_resistance_pair_component_check" in judge
    assert 'JUDGE_CONFIG_VERSION = "0.29.0-r2"' in runner
    assert 'FACET_SPEC_VERSION = "0.10.1"' in runner

    print("v0.29 r2 semantic/provenance contract passed")


if __name__ == "__main__":
    main()
