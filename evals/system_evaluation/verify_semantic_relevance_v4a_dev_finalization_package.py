from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
FINALIZER = REPO_ROOT / "evals/system_evaluation/finalize_semantic_relevance_v4a_dev_scores.py"

FILES = {
    "score inventory": (RUN / "v4a_dev_display_score_inventory.csv", "363ef15e6e8cfa07afcab6e73b20d0d2e21d778235f1800ab388a22f6ac8f818"),
    "novel judging input": (RUN / "v4a_dev_novel_display_pairs_to_judge.csv", "3d33db6b52314b1a022705c0545798ded6cf17a5e3e4355aad19b5cd43dcdd56"),
    "judging-input lock": (RUN / "v4a_dev_judging_input.lock.json", "551d22ffa5c8e3d04aa5457675d47c40abed19d6162efc66dde2a8c329a95c97"),
    "novel judge scores": (RUN / "v4a_dev_novel_judge_scores.csv", "87b3a425e793e402f313268b653e76b8e162baec6ddb5d9167b7835a7de19044"),
    "novel scoring lock": (RUN / "v4a_dev_novel_judge_scoring.lock.json", "3be5b00e48d057093f8b5c8e98ca29f66f7c88e70441043d601487960c26e8bf"),
}
EXPECTED_FINALIZER_SHA256 = "472cbbdf6e3db87bf5a15e587238fcfe08eb25805d995cb74f3256f4a42f83da"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    print("Semantic relevance v4a - development score-finalization package preflight")
    print("------------------------------------------------------------------------")
    print("Complete-map creation: DISABLED")
    print("Judge execution:       DISABLED")
    print("Model execution:       DISABLED")
    print("Development comparison: DISABLED")
    print("Development selection:  DISABLED")
    print("Release analysis:        DISABLED")
    print()

    for label, (path, expected) in FILES.items():
        req(path.exists(), f"Missing frozen input: {path}")
        actual = sha256(path)
        req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
        print(f"PASS  {label} SHA-256 verified")

    req(FINALIZER.exists(), f"Missing finalizer: {FINALIZER}")
    actual = sha256(FINALIZER)
    req(actual == EXPECTED_FINALIZER_SHA256, f"finalizer hash mismatch: expected={EXPECTED_FINALIZER_SHA256} actual={actual}")
    print("PASS  finalizer SHA-256 verified")

    jlock = load(FILES["judging-input lock"][0])
    req(jlock.get("status") == "V4A_DEV_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION", "Unexpected judging-input lock status")
    req(int(jlock.get("reused_frozen_scores", -1)) == 268, "Expected 268 reused scores")
    req(int(jlock.get("novel_pairs_to_judge", -1)) == 79, "Expected 79 novel pairs")

    slock = load(FILES["novel scoring lock"][0])
    req(slock.get("status") == "V4A_DEV_NOVEL_JUDGE_SCORES_FROZEN_BEFORE_FINALIZATION", "Unexpected novel scoring-lock status")
    req(int(slock.get("novel_pairs", -1)) == 79, "Novel scoring-lock count changed")
    req(slock.get("novel_judge_scores_sha256") == FILES["novel judge scores"][1], "Novel scoring lock hash mismatch")
    req(slock.get("human_relevance_labels_used") is False, "Human labels unexpectedly used")
    req(slock.get("development_selection_decision_produced") is False, "Development selection already produced")

    inventory = pd.read_csv(FILES["score inventory"][0], dtype=str, keep_default_na=False, encoding="utf-8")
    scores = pd.read_csv(FILES["novel judge scores"][0], dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(inventory) == 347, "Score inventory is not 347 rows")
    req(int((inventory["score_status"] == "reused").sum()) == 268, "Inventory reused count changed")
    req(int((inventory["score_status"] == "novel").sum()) == 79, "Inventory novel count changed")
    req(len(scores) == 79, "Novel score artifact is not 79 rows")
    req(not scores["case_id"].duplicated().any(), "Duplicate novel score case IDs")
    vals = pd.to_numeric(scores["judge_score"], errors="raise").astype(int)
    req(vals.between(0, 4).all(), "Novel score outside 0..4")
    print("PASS  frozen finalization structure verified: 268 reused + 79 judged = 347")

    print()
    print("V4A DEVELOPMENT SCORE FINALIZATION PACKAGE PREFLIGHT: PASS")
    print("Complete map created:       NO")
    print("Judge calls:                0")
    print("Model calls:                0")
    print("Human labels used:          NO")
    print("Development comparison:     NO")
    print("Development selection:      NO")
    print("Release analysis:           NO")
    print("Next: deterministically finalize and freeze the complete 347-pair score map.")


if __name__ == "__main__":
    main()
