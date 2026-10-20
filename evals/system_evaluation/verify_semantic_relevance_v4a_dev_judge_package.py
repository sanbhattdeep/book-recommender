from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
HOLDOUT_RUN = EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"

NOVEL_INPUT = RUN / "v4a_dev_novel_display_pairs_to_judge.csv"
SCORE_INVENTORY = RUN / "v4a_dev_display_score_inventory.csv"
JUDGING_INPUT_LOCK = RUN / "v4a_dev_judging_input.lock.json"
HOLDOUT_SCORING_LOCK = HOLDOUT_RUN / "holdout_judge_scoring_lock.json"
HOLDOUT_FACET_FILE = EVALS / "facets/semantic_relevance/semantic_relevance_v3_holdout_query_facets.v0.1.0.json"
JUDGE_FILE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS / "semantic_relevance_facet_scoring.py"
JUDGE_CONFIG_FILE = EVALS / "judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
RUBRIC_FILE = EVALS / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
SPAN_ECHO_PATCH_FILE = EVALS / "system_evaluation/semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"
RUNNER = EVALS / "system_evaluation/score_semantic_relevance_v4a_dev_novel_pairs.py"

EXPECTED = {
    "novel_input": "3d33db6b52314b1a022705c0545798ded6cf17a5e3e4355aad19b5cd43dcdd56",
    "score_inventory": "363ef15e6e8cfa07afcab6e73b20d0d2e21d778235f1800ab388a22f6ac8f818",
    "judging_input_lock": "551d22ffa5c8e3d04aa5457675d47c40abed19d6162efc66dde2a8c329a95c97",
    "holdout_scoring_lock": "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777",
    "span_echo_patch": "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266",
    "runner": "f52fee369e958f57f2b850ea6373f8708d791426b6e535aa13704bad93934f07",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_sha256(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bool_value(value: Any) -> bool:
    s = str(value).strip().lower()
    if s in {"true", "1", "yes"}: return True
    if s in {"false", "0", "no", ""}: return False
    raise ValueError(f"Invalid bool {value!r}")


def verify_text(label: str, path: Path, expected: str) -> None:
    raw = sha256(path)
    norm = lf_sha256(path)
    req(raw == expected or norm == expected, f"{label} hash mismatch: expected={expected} raw={raw} lf={norm}")
    mode = "raw" if raw == expected else "canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")


def main() -> None:
    print("Semantic relevance v4a - development novel-pair judge package preflight")
    print("--------------------------------------------------------------------")
    print("Judge execution:          DISABLED")
    print("Human labels:             DISABLED")
    print("Release analysis:         DISABLED")
    print("Development selection:    DISABLED")
    print()

    for label, path, key in [
        ("novel judging input", NOVEL_INPUT, "novel_input"),
        ("score inventory", SCORE_INVENTORY, "score_inventory"),
        ("judging-input lock", JUDGING_INPUT_LOCK, "judging_input_lock"),
        ("v3 holdout scoring lock", HOLDOUT_SCORING_LOCK, "holdout_scoring_lock"),
        ("span-echo execution patch", SPAN_ECHO_PATCH_FILE, "span_echo_patch"),
        ("v4a novel judge runner", RUNNER, "runner"),
    ]:
        req(path.exists(), f"Missing {label}: {path}")
        actual = sha256(path)
        req(actual == EXPECTED[key], f"{label} hash mismatch: expected={EXPECTED[key]} actual={actual}")
        print(f"PASS  {label} SHA-256 verified")

    lock = load_json(JUDGING_INPUT_LOCK)
    req(lock.get("status") == "V4A_DEV_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION", "Unexpected v4a judging-input lock status")
    req(int(lock.get("display_union_rows", -1)) == 347, "Display union row count changed")
    req(int(lock.get("reused_frozen_scores", -1)) == 268, "Reused score count changed")
    req(int(lock.get("novel_pairs_to_judge", -1)) == 79, "Novel score count changed")
    req(lock.get("judge_execution_performed") is False, "Input lock says judging already occurred")
    req(lock.get("human_relevance_labels_used") is False, "Input lock says human labels were used")
    req(lock.get("release_metrics_calculated") is False, "Input lock says release metrics were calculated")

    hlock = load_json(HOLDOUT_SCORING_LOCK)
    req(hlock.get("status") == "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS", "Unexpected v3 holdout scoring lock status")
    req(hlock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5", "Judge candidate changed")
    req(hlock.get("human_relevance_labels_used") is False, "Frozen v3 judge runtime used human labels")

    for label, path, key in [
        ("frozen judge source", JUDGE_FILE, "judge_source_sha256"),
        ("frozen deterministic scoring", SCORING_FILE, "scoring_source_sha256"),
        ("frozen judge config", JUDGE_CONFIG_FILE, "judge_config_sha256"),
        ("frozen rubric", RUBRIC_FILE, "rubric_sha256"),
    ]:
        expected = str(hlock.get(key, ""))
        req(expected, f"Holdout scoring lock missing {key}")
        verify_text(label, path, expected)

    expected_facet_sha = str(hlock.get("holdout_facet_spec_sha256", ""))
    req(expected_facet_sha, "Holdout scoring lock missing holdout_facet_spec_sha256")
    verify_text("frozen H01-H24 facet spec", HOLDOUT_FACET_FILE, expected_facet_sha)

    patch = load_json(SPAN_ECHO_PATCH_FILE)
    req(patch.get("status") == "APPLIED_NON_SEMANTIC_EXECUTION_PATCH", "Unexpected span-echo patch status")
    req(patch.get("patch_id") == "v3-posthoc-prominence-span-echo-normalization1", "Unexpected span-echo patch ID")
    req(patch.get("semantic_rules_changed") is False, "Span-echo patch changed semantic rules")
    req(patch.get("deterministic_scoring_rules_changed") is False, "Span-echo patch changed scoring rules")

    frame = pd.read_csv(NOVEL_INPUT, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(frame) == 79, f"Expected 79 frozen novel pairs, got {len(frame)}")
    req(not frame["case_id"].duplicated().any(), "Duplicate novel case IDs")
    req(not frame.duplicated(["query_id", "isbn13"]).any(), "Duplicate novel query-book pairs")
    req(frame["pair_source"].eq("v4a_only").all(), "Novel input includes a non-v4a-only pair")
    req((~frame["in_v3b"].map(bool_value)).all(), "Novel input includes a v3b pair")
    req(frame["in_v4a"].map(bool_value).all(), "Novel input includes a pair not displayed by v4a")
    ranks = pd.to_numeric(frame["v4a_rank"], errors="raise").astype(int)
    req(ranks.between(1, 10).all(), "Novel input v4a rank outside 1..10")
    req(frame["query_id"].nunique() <= 24 and frame["query_id"].str.match(r"H(?:0[1-9]|1[0-9]|2[0-4])$").all(), "Unexpected query IDs in novel input")
    print(f"PASS  frozen novel judging structure verified: {len(frame)} pairs across {frame['query_id'].nunique()} queries")

    print()
    print("V4A DEVELOPMENT NOVEL-JUDGE PACKAGE PREFLIGHT: PASS")
    print("Novel pairs frozen:       79")
    print("Judge execution:          NO")
    print("Human labels used:        NO")
    print("Release analysis:         NO")
    print("Development selection:    NO")
    print("Next: execute frozen judge only for the 79 novel v4a displayed pairs.")


if __name__ == "__main__":
    main()
