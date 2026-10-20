from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
V4A_RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
V3_HOLDOUT_RUN = REPO_ROOT / "evals/runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"
V3_POSTHOC_RUN = REPO_ROOT / "evals/runs/semantic_relevance_v3_posthoc_diagnostic/v2_deep_candidate_prepare"
BUILDER = REPO_ROOT / "evals/system_evaluation/prepare_semantic_relevance_v4a_dev_score_reuse.py"

CHECKS = [
    ("v4a displayed union", V4A_RUN / "v3b_v4a_dev_display_union.csv", "caa34848a25b4f6d245a398a8860325f9d2ed3898951b665fa387ef6f4c7b85a"),
    ("v4a displayed-union lock", V4A_RUN / "v3b_v4a_dev_display_union.lock.json", "607fe469087ce78dd04dc4574c7f0f889f2f04939b6f73abac003243a9be6028"),
    ("v3 holdout judge scores", V3_HOLDOUT_RUN / "holdout_judge_scores.csv", "d1c9ba1848fb4875c1e925b6c5bdafc3483db37a613bad49286a634d96803fa3"),
    ("v3 holdout scoring lock", V3_HOLDOUT_RUN / "holdout_judge_scoring_lock.json", "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777"),
    ("v3 posthoc complete score map", V3_POSTHOC_RUN / "target_top50_complete_score_map.csv", "723fa27ecb8947334d25b3b7423a6597da39d594eb7ec2f979056b1c3cc516e5"),
    ("v3 posthoc scoring lock", V3_POSTHOC_RUN / "deep_candidate_scoring_lock.json", "af67cc8090d0613973da4c927b573d29bb956be34d4c44255594cb12bde6fd1f"),
    ("score-reuse builder", BUILDER, "2f3a569a57a0dd09de9e5eae7321cb68d66d59a4c4a1fad56a7e9a3e3f70c0de"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    print("Semantic relevance v4a - development score-reuse package preflight")
    print("-----------------------------------------------------------------")
    print("Score inventory creation: DISABLED")
    print("Judge execution:          DISABLED")
    print("Human labels:             DISABLED")
    print("Release analysis:         DISABLED")
    print()

    for label, path, expected in CHECKS:
        req(path.exists(), f"Missing {label}: {path}")
        actual = sha256(path)
        req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
        print(f"PASS  {label} SHA-256 verified")

    union = pd.read_csv(CHECKS[0][1], dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(union) == 347, f"Expected 347 display-union rows, got {len(union)}")
    req(union["query_id"].nunique() == 24, "Display union is not H01-H24")
    req(not union.duplicated(["query_id", "isbn13"]).any(), "Display union contains duplicate pairs")

    holdout = pd.read_csv(CHECKS[2][1], dtype=str, keep_default_na=False, encoding="utf-8")
    posthoc = pd.read_csv(CHECKS[4][1], dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(holdout) == 322, f"Expected 322 holdout scores, got {len(holdout)}")
    req(len(posthoc) == 850, f"Expected 850 posthoc score-map rows, got {len(posthoc)}")
    req(not holdout.duplicated(["query_id", "isbn13"]).any(), "Holdout score source contains duplicate pairs")
    req(not posthoc.duplicated(["query_id", "isbn13"]).any(), "Posthoc score source contains duplicate pairs")

    display_lock = load_json(CHECKS[1][1])
    req(display_lock.get("status") == "V4A_DEV_DISPLAY_UNION_FROZEN_BEFORE_JUDGING", "Unexpected display-union lock status")
    holdout_lock = load_json(CHECKS[3][1])
    req(holdout_lock.get("status") == "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS", "Unexpected holdout scoring-lock status")
    posthoc_lock = load_json(CHECKS[5][1])
    req(posthoc_lock.get("status") == "V3_POSTHOC_DEEP_JUDGE_SCORES_FROZEN_BEFORE_CLASSIFICATION", "Unexpected posthoc scoring-lock status")

    print("PASS  frozen score-source structures and lock states verified")
    print()
    print("V4A DEVELOPMENT SCORE REUSE PACKAGE PREFLIGHT: PASS")
    print("Score inventory created: NO")
    print("Judge calls:             0")
    print("Human labels used:       NO")
    print("Release analysis:        NO")
    print("Next: build/freeze reuse-vs-novel score inventory before judging.")


if __name__ == "__main__":
    main()
