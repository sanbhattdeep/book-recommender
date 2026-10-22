from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

SCORE_INVENTORY = RUN / "v4b_dev_display_score_inventory.csv"
NOVEL_INPUT = RUN / "v4b_dev_novel_display_pairs_to_judge.csv"
JUDGING_INPUT_LOCK = RUN / "v4b_dev_judging_input.lock.json"
NOVEL_SCORES = RUN / "v4b_dev_novel_judge_scores.csv"
NOVEL_SCORING_LOCK = RUN / "v4b_dev_novel_judge_scoring.lock.json"

COMPLETE_MAP = RUN / "v4b_dev_complete_display_score_map.csv"
FINALIZATION_METADATA = RUN / "v4b_dev_score_finalization_metadata.json"
FINALIZATION_LOCK = RUN / "v4b_dev_complete_score_map.lock.json"

EXPECTED = {
    "score_inventory": "f2c3d8905e582cc9df6b3a9417f3647d71496eed45ea5240d95c1cf1ece85cc1",
    "novel_input": "5eeb07e860ba70248a3c8f6df238f347535d7fe0a6d3c76314815b493ef753d0",
    "judging_input_lock": "9099bae135a7a97459633243a0659543b631fcc9bb3987aee0a86b6ee5fedf20",
    "novel_scores": "42dcbe7f8fb39427775c9a5a4d9251c677dc7de844aecaf91f1738a2e79a68c8",
    "novel_scoring_lock": "87555c2703582f0d8fd9c3ccdee1b7a3e8e6df850ceea48ee85275e0e888194f"
}
EXPECTED_TOTAL = 367
EXPECTED_REUSED = 345
EXPECTED_NOVEL = 22

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def bool_value(value: Any) -> bool:
    t = str(value).strip().lower()
    if t in {"true","1","yes"}:
        return True
    if t in {"false","0","no",""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")

def main() -> None:
    print("Semantic relevance V4B - score finalization preflight")
    print("----------------------------------------------------")
    print("Complete-map generation: DISABLED")
    print("Judge/model execution: DISABLED")
    print("Development comparison: NOT CALCULATED")
    print("Development selection: NOT PRODUCED")
    print("Release analysis: DISABLED")
    print()

    for label, path, expected_sha in [
        ("score inventory", SCORE_INVENTORY, EXPECTED["score_inventory"]),
        ("novel judging input", NOVEL_INPUT, EXPECTED["novel_input"]),
        ("judging-input lock", JUDGING_INPUT_LOCK, EXPECTED["judging_input_lock"]),
        ("novel judge scores", NOVEL_SCORES, EXPECTED["novel_scores"]),
        ("novel scoring lock", NOVEL_SCORING_LOCK, EXPECTED["novel_scoring_lock"]),
    ]:
        req(path.exists(), f"Missing frozen input: {path}")
        actual = sha256(path)
        req(actual == expected_sha, f"{label} hash mismatch: expected={expected_sha} actual={actual}")
        print(f"PASS  {label} raw SHA-256 verified")

    for out in [COMPLETE_MAP, FINALIZATION_METADATA, FINALIZATION_LOCK]:
        req(not out.exists(), f"Refusing preflight because finalization output already exists: {out}")

    ilock = load(JUDGING_INPUT_LOCK)
    req(ilock.get("status") == "V4B_DEV_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "Unexpected judging-input status")
    req(int(ilock.get("display_union_rows",-1)) == EXPECTED_TOTAL, "Display-union total changed")
    req(int(ilock.get("reused_frozen_scores",-1)) == EXPECTED_REUSED, "Reused count changed")
    req(int(ilock.get("novel_pairs_to_judge",-1)) == EXPECTED_NOVEL, "Novel count changed")
    req(ilock.get("known_h14_quarantine_pair_present") is False,
        "Judging-input lock says unresolved H14 pair is present")

    slock = load(NOVEL_SCORING_LOCK)
    req(slock.get("status") == "V4B_DEV_NOVEL_JUDGE_SCORES_FROZEN_BEFORE_FINALIZATION",
        "Unexpected novel-scoring status")
    req(int(slock.get("novel_pairs",-1)) == EXPECTED_NOVEL, "Frozen judged count changed")
    req(slock.get("novel_judge_scores_sha256") == EXPECTED["novel_scores"],
        "Novel score hash changed inside scoring lock")
    req(slock.get("human_relevance_labels_used") is False,
        "Novel scoring lock says human labels were used")
    req(slock.get("development_selection_decision_produced") is False,
        "Novel scoring lock says selection was produced")

    inventory = pd.read_csv(SCORE_INVENTORY, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(inventory) == EXPECTED_TOTAL, "Inventory row count changed")
    req(not inventory.duplicated(["query_id","isbn13"]).any(), "Duplicate inventory pair")
    req(int((inventory["score_status"]=="reused").sum()) == EXPECTED_REUSED, "Reused inventory count changed")
    req(int((inventory["score_status"]=="novel").sum()) == EXPECTED_NOVEL, "Novel inventory count changed")
    req(int(inventory["in_v3b"].map(bool_value).sum()) == 240, "V3B membership is not 240")
    req(int(inventory["in_v4b"].map(bool_value).sum()) == 240, "V4B membership is not 240")

    novel = pd.read_csv(NOVEL_INPUT, dtype=str, keep_default_na=False, encoding="utf-8")
    scores = pd.read_csv(NOVEL_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(novel) == EXPECTED_NOVEL and len(scores) == EXPECTED_NOVEL,
        "Novel input/score row counts changed")
    req(set(novel["case_id"]) == set(scores["case_id"]),
        "Frozen novel input and judged case sets differ")
    req(not ((novel["query_id"]=="H14") & (novel["isbn13"]=="9781582972947")).any(),
        "Unresolved H14 quarantine pair entered finalization")

    vals = pd.to_numeric(scores["judge_score"], errors="raise").astype(int)
    req(vals.between(0,4).all(), "Novel score outside 0..4")

    print("PASS  exact 345-reused + 22-judged boundary verified")
    print("PASS  all 22 frozen novel case IDs have frozen judge scores")
    print("PASS  unresolved H14 quarantine pair absent")
    print()
    print("V4B SCORE FINALIZATION PREFLIGHT: PASS")
    print("Displayed pairs: 367")
    print("Reused frozen scores: 345")
    print("New frozen scores: 22")
    print("Complete score map generated: NO")
    print("Judge calls performed: 0")
    print("Model calls performed: 0")
    print("Development comparison calculated: NO")
    print("Development selection produced: NO")
    print("Release analysis: NO")
    print("Next: deterministically finalize and freeze the 367-row V4B complete score map.")

if __name__ == "__main__":
    main()
