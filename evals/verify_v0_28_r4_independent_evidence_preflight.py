from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = REPO_ROOT / "evals" / "datasets"
RELEASES = REPO_ROOT / "evals" / "releases"

EXPECTED_RELEASE_MANIFEST_SHA = "6d1f390a21d804351ff75def5bfd74e7e5eafbd48685aa4ee50058b6af670faa"
EXPECTED_LABELLED_SHA = "df742aa351a292c8a84fe330887f745d87c761dba40ba346c1ac9aa21a5368f0"
EXPECTED_SPLIT_MANIFEST_SHA = "889673f565cc7c377f6f0c11b690c709c10e93b574e9ad404c852cb2f2d742df"
EXPECTED_VALIDATION_SHA = "41b548363a984fd8f1a923e23fbb8b1db2c9b6bd3e00c9d43102ecca0c47672f"
EXPECTED_HOLDOUT_SHA = "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85"
EXPECTED_REVIEW_MANIFEST_SHA = "be76c29361f0c2bba3eedeadd1f6658a217e5dc2602543f58346216e3ad9cbb7"

RELEASE_MANIFEST = RELEASES / "semantic_relevance_v0.28.0_r4_release_candidate.json"
LABELLED = DATASETS / "semantic_relevance_unseen_pool.v4.0.0.labelled.final.csv"
SPLIT_MANIFEST = DATASETS / "semantic_relevance_unseen_split_manifest.v4.0.0.json"
VALIDATION = DATASETS / "semantic_relevance_validation.v4.0.0.DO_NOT_RUN_YET.csv"
HOLDOUT = DATASETS / "semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv"
VALIDATION_LOCK = DATASETS / "semantic_relevance_validation_lock.v4.0.0.json"
HOLDOUT_LOCK = DATASETS / "semantic_relevance_final_holdout_lock.v4.0.0.json"
REVIEW_MANIFEST = DATASETS / "semantic_relevance_unseen_label_review_manifest.v4.0.0.json"

CONSUMED_SOURCES = [
    DATASETS / "semantic_relevance_v0.28_development.v4.0.0.csv",
    DATASETS / "semantic_relevance_calibration.v0.5.0.csv",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def canonical_isbn(value: str) -> str:
    raw = str(value or "").strip()
    if re.fullmatch(r"\d+\.0", raw):
        raw = raw[:-2]
    return raw


def norm_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def title_author_key(row: dict[str, str]) -> str:
    return norm_text(row.get("title", "")) + "|" + norm_text(row.get("authors", ""))


def pos(case_id: str) -> str:
    return case_id.rsplit("_", 1)[-1]


def require(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    print("v0.28.0 r4 independent-evidence PRE-FLIGHT")
    print("------------------------------------------")
    print("JUDGE EXECUTION: DISABLED")

    for p in [
        RELEASE_MANIFEST, LABELLED, SPLIT_MANIFEST, VALIDATION, HOLDOUT,
        VALIDATION_LOCK, HOLDOUT_LOCK, REVIEW_MANIFEST, *CONSUMED_SOURCES,
    ]:
        require(p.exists(), f"required file exists: {p.relative_to(REPO_ROOT)}")

    require(sha256(RELEASE_MANIFEST) == EXPECTED_RELEASE_MANIFEST_SHA,
            "frozen r4 release-candidate manifest hash pinned")
    require(sha256(LABELLED) == EXPECTED_LABELLED_SHA,
            "final 60-case human-labelled pool hash pinned")
    require(sha256(SPLIT_MANIFEST) == EXPECTED_SPLIT_MANIFEST_SHA,
            "blind deterministic split manifest hash pinned")
    require(sha256(VALIDATION) == EXPECTED_VALIDATION_SHA,
            "30-case validation dataset hash pinned")
    require(sha256(HOLDOUT) == EXPECTED_HOLDOUT_SHA,
            "30-case final holdout dataset hash pinned")
    require(sha256(REVIEW_MANIFEST) == EXPECTED_REVIEW_MANIFEST_SHA,
            "human-label review manifest hash pinned")

    source = load_csv(LABELLED)
    val = load_csv(VALIDATION)
    hold = load_csv(HOLDOUT)

    require(len(source) == 60, "labelled source contains 60 cases")
    require(len(val) == 30, "validation contains 30 cases")
    require(len(hold) == 30, "final holdout contains 30 cases")

    source_ids = {r["case_id"] for r in source}
    val_ids = {r["case_id"] for r in val}
    hold_ids = {r["case_id"] for r in hold}
    require(len(source_ids) == 60, "source case IDs unique")
    require(val_ids.isdisjoint(hold_ids), "validation/final-holdout case IDs do not overlap")
    require(val_ids | hold_ids == source_ids, "30/30 split exactly partitions the 60-case source")

    source_isbns = [canonical_isbn(r["isbn13"]) for r in source]
    source_ta = [title_author_key(r) for r in source]
    require(len(set(source_isbns)) == 60, "new pool ISBN identities unique")
    require(len(set(source_ta)) == 60, "new pool title+author identities unique")

    consumed_isbns = set()
    consumed_ta = set()
    for path in CONSUMED_SOURCES:
        for r in load_csv(path):
            consumed_isbns.add(canonical_isbn(r.get("isbn13", "")))
            consumed_ta.add(title_author_key(r))
    require(not (set(source_isbns) & consumed_isbns),
            "no ISBN overlap with consumed development/calibration")
    require(not (set(source_ta) & consumed_ta),
            "no title+author overlap with consumed development/calibration")

    for rows, label in [(val, "validation"), (hold, "final holdout")]:
        q = Counter(r["query_id"] for r in rows)
        p = Counter(pos(r["case_id"]) for r in rows)
        s = Counter(int(r["human_score"]) for r in rows)
        c = Counter(r["candidate_source"] for r in rows)

        require(set(q) == {f"Q{i:02d}" for i in range(1, 13)},
                f"{label}: Q01-Q12 all represented")
        require(all(v in (2, 3) for v in q.values()),
                f"{label}: each query contributes 2 or 3 cases")
        require(p == Counter({"T01": 6, "T02": 6, "T03": 6, "T04": 6, "T05": 6}),
                f"{label}: six cases from every sampling position")
        require(s == Counter({0: 19, 1: 3, 2: 4, 3: 3, 4: 1}),
                f"{label}: frozen human-score distribution")
        require(c == Counter({"semantic_retrieval": 24, "random_negative_candidate": 6}),
                f"{label}: 24 retrieval + 6 random candidates")
        require(all(r["review_status"] == "LABELLED" for r in rows),
                f"{label}: all human labels frozen as LABELLED")

    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    require(split["judge_outputs_used_for_split"] is False,
            "split did not use judge output")
    require(split["human_labels_frozen_before_judge"] is True,
            "human labels frozen before judge")
    require(split["validation"]["sha256"] == EXPECTED_VALIDATION_SHA,
            "split manifest pins validation SHA")
    require(split["final_holdout"]["sha256"] == EXPECTED_HOLDOUT_SHA,
            "split manifest pins final-holdout SHA")

    for path, role in [
        (VALIDATION_LOCK, "independent_validation"),
        (HOLDOUT_LOCK, "independent_final_holdout"),
    ]:
        lock = json.loads(path.read_text(encoding="utf-8"))
        require(lock["evidence_role"] == role, f"{role}: lock role correct")
        require(lock["status"] == "LOCKED_DO_NOT_RUN", f"{role}: status LOCKED_DO_NOT_RUN")
        require(lock["judge_run_count"] == 0, f"{role}: judge_run_count = 0")
        require(lock["judge_outputs_used_for_split"] is False, f"{role}: split remained blind")
        require(lock["human_labels_frozen"] is True, f"{role}: human labels frozen")
        require(lock["finality_marker_present"] is False, f"{role}: no finality marker exists")

    print()
    print("PRE-FLIGHT: PASS")
    print("NO JUDGE CALLS WERE MADE.")
    print("Validation and final holdout remain LOCKED_DO_NOT_RUN.")


if __name__ == "__main__":
    main()
