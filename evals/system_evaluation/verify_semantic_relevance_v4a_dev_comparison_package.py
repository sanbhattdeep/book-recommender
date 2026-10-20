from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

FILES = {
    "v4a spec": (REPO_ROOT / "evals/system_evaluation/semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json", "6aba9c8354f201ad41044dd82d4476612f81ebd27a5ce4f8bf4f327a8b3281b9"),
    "v4a spec lock": (REPO_ROOT / "evals/system_evaluation/semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.lock.json", "c9fee869cda6ccfd26c1ad0b2624cbcf693b9c896366b6baaa5c0b7b98371db2"),
    "complete score map": (RUN / "v4a_dev_complete_display_score_map.csv", "60fb8b44f6eb13d486eaa6ffae483dca374e1415af8708d833912a18ec4d2000"),
    "finalization lock": (RUN / "v4a_dev_complete_score_map.lock.json", "fe5b4865df05c4086247360fc7693e71d3cfe63f8877e944eaea61016d418f23"),
    "comparison analyzer": (REPO_ROOT / "evals/system_evaluation/analyze_semantic_relevance_v4a_dev_comparison.py", "df2be8c787fb212a2a2e0c1dfbcd341f9742f4dc4e4d5c16f992a6c5689d94d6"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def b(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def main() -> None:
    print("Semantic relevance v4a - development comparison package preflight")
    print("---------------------------------------------------------------")
    print("Development metrics:    DISABLED")
    print("Development selection:  DISABLED")
    print("Judge execution:        DISABLED")
    print("Model execution:        DISABLED")
    print("Release gates/decision: DISABLED")
    print()

    for label, (path, expected) in FILES.items():
        req(path.exists(), f"Missing frozen input/source: {path}")
        actual = sha256(path)
        req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
        print(f"PASS  {label} SHA-256 verified")

    spec = load(FILES["v4a spec"][0])
    proto = spec.get("development_protocol", {})
    req(proto.get("query_set") == "H01-H24", "Unexpected development query set")
    req(int(proto.get("query_count", -1)) == 24, "Unexpected development query count")
    bs = proto.get("bootstrap", {})
    req(int(bs.get("replicates", -1)) == 5000 and int(bs.get("seed", -1)) == 20261002, "Bootstrap contract changed")
    selection = spec.get("development_selection_rule", {})
    req(selection.get("ordered_criteria") == [
        "lower catastrophic_query_rate",
        "lower irrelevant_rate_at_10",
        "higher clear_or_strong_rate_at_10",
        "higher macro_mean_relevance_at_10",
        "higher ndcg_at_10",
    ], "Selection order changed")
    req(selection.get("final_tie_break") == "retain v3b", "Final tie-break changed")

    flock = load(FILES["finalization lock"][0])
    req(flock.get("status") == "V4A_DEV_COMPLETE_DISPLAY_SCORE_MAP_FROZEN_BEFORE_COMPARISON", "Unexpected finalization lock status")
    req(flock.get("complete_score_map_sha256") == FILES["complete score map"][1], "Map hash differs inside finalization lock")
    req(flock.get("development_comparison_calculated") is False, "Comparison already calculated during finalization")
    req(flock.get("development_selection_decision_produced") is False, "Selection already produced during finalization")

    frame = pd.read_csv(FILES["complete score map"][0], dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(frame) == 347, "Complete score map is not 347 rows")
    req(frame["query_id"].nunique() == 24, "Complete map is not H01-H24")
    req(int(frame["in_v3b"].map(b).sum()) == 240, "v3b membership is not 240")
    req(int(frame["in_v4a"].map(b).sum()) == 240, "v4a membership is not 240")
    vals = pd.to_numeric(frame["final_judge_score"], errors="raise").astype(int)
    req(vals.between(0, 4).all(), "Complete score map has score outside 0..4")
    print("PASS  frozen development comparison structure verified: 347 pairs / 24 queries / 240 rows per system")

    print()
    print("V4A DEVELOPMENT COMPARISON PACKAGE PREFLIGHT: PASS")
    print("Development metrics calculated: NO")
    print("Development selection produced: NO")
    print("Judge calls:                    0")
    print("Model calls:                    0")
    print("Human labels used:              NO")
    print("Release gates/decision:         NO")
    print("Next: calculate frozen v3b-vs-v4a development metrics and apply the pre-registered selection rule.")


if __name__ == "__main__":
    main()
