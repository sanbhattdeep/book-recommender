"""Non-judging preflight for the frozen v0.28.0 r8 one-time final holdout."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS = REPO_ROOT / "evals"
DATASETS = EVALS / "datasets"
RELEASES = EVALS / "releases"
RUN_ROOT = EVALS / "runs" / "semantic_relevance_v0_28_final_holdout"

RELEASE_MANIFEST = RELEASES / "semantic_relevance_v0.28.0_r8_release_candidate.json"
PROTOCOL = RELEASES / "semantic_relevance_v0.28.0_r8_final_holdout_protocol.v1.0.0.json"
FINAL_HOLDOUT = DATASETS / "semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv"
FINAL_LOCK = DATASETS / "semantic_relevance_final_holdout_lock.v4.0.0.json"
VALIDATION_LOCK = DATASETS / "semantic_relevance_validation_lock.v4.0.0.json"
SPLIT_MANIFEST = DATASETS / "semantic_relevance_unseen_split_manifest.v4.0.0.json"
LABELLED_POOL = DATASETS / "semantic_relevance_unseen_pool.v4.0.0.labelled.final.csv"
REVIEW_MANIFEST = DATASETS / "semantic_relevance_unseen_label_review_manifest.v4.0.0.json"
DEVELOPMENT = DATASETS / "semantic_relevance_v0.28_development.v5.0.0.csv"
FINALITY_MARKER = RUN_ROOT / "FINAL_HOLDOUT_STARTED.json"

EXPECTED_RELEASE_SHA = "c25759ccaba3a3258552a5c138170821b3a9afd48acb968859a5943b4fc91cab"
EXPECTED_PROTOCOL_SHA = "7aa066815dd7c084be20773daec70c09981feb6c646061b658ed4fea1fca87f8"
EXPECTED_HOLDOUT_SHA = "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85"
EXPECTED_SPLIT_SHA = "889673f565cc7c377f6f0c11b690c709c10e93b574e9ad404c852cb2f2d742df"
EXPECTED_LABELLED_SHA = "df742aa351a292c8a84fe330887f745d87c761dba40ba346c1ac9aa21a5368f0"
EXPECTED_REVIEW_SHA = "be76c29361f0c2bba3eedeadd1f6658a217e5dc2602543f58346216e3ad9cbb7"
EXPECTED_DEV_SHA = "21e53ee575fab0f99e69ffabee705f8afcc843878a33d640d6ada795be1ea520"

EXPECTED_SEMANTIC_HASHES = {
    "evals/semantic_relevance_facet_judge.py": "e4efec54d03601e3016b376741c7f14fc1259ab3d63d1fdff75925cc2bfb43c1",
    "evals/semantic_relevance_facet_scoring.py": "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808",
    "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.13.json": "0952d80ebed8fbdebfb3b625f5648be9ccc08148d87d6753adc83d010953e289",
    "evals/judge_configs/semantic_relevance_judge.v0.28.0-r8.json": "bdc4157b457e3453ff94b860cb45022a7c7c83f311432ef9841a1c1fe5a9bb8d",
    "evals/datasets/semantic_relevance_v0.28_regression_manifest.v1.7.0.json": "da8d7ffe0f9131dd5ab976a0af72d7c478ee592b4cf1b495e105e483d908c585",
    "evals/rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json": "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e",
}

EXPECTED_CHECK_NAMES = {
    "within_one_ge_0_95",
    "exact_ge_0_50",
    "quadratic_kappa_ge_0_80",
    "linear_kappa_ge_0_60",
    "absolute_difference_ge_2_le_1_case",
    "binary_precision_ge_0_90",
    "binary_recall_ge_0_80",
    "deterministic_cue_severe_false_positives_eq_0",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    print("v0.28.0 r8 final-holdout PRE-FLIGHT")
    print("-----------------------------------")
    print("JUDGE EXECUTION: DISABLED")

    required = [
        RELEASE_MANIFEST, PROTOCOL, FINAL_HOLDOUT, FINAL_LOCK, VALIDATION_LOCK,
        SPLIT_MANIFEST, LABELLED_POOL, REVIEW_MANIFEST, DEVELOPMENT,
    ]
    for path in required:
        require(path.exists(), f"required file exists: {path.relative_to(REPO_ROOT)}")

    require(sha256(RELEASE_MANIFEST) == EXPECTED_RELEASE_SHA,
            "frozen r8 release-candidate manifest hash pinned")
    require(sha256(PROTOCOL) == EXPECTED_PROTOCOL_SHA,
            "final-holdout protocol hash pinned before execution")
    require(sha256(FINAL_HOLDOUT) == EXPECTED_HOLDOUT_SHA,
            "30-case final-holdout dataset hash pinned")
    require(sha256(SPLIT_MANIFEST) == EXPECTED_SPLIT_SHA,
            "blind split manifest hash pinned")
    require(sha256(LABELLED_POOL) == EXPECTED_LABELLED_SHA,
            "60-case labelled source pool hash pinned")
    require(sha256(REVIEW_MANIFEST) == EXPECTED_REVIEW_SHA,
            "human-label review manifest hash pinned")
    require(sha256(DEVELOPMENT) == EXPECTED_DEV_SHA,
            "210-case consumed-development dataset hash pinned")

    for rel, expected in EXPECTED_SEMANTIC_HASHES.items():
        path = REPO_ROOT / rel
        require(path.exists(), f"frozen semantic artifact exists: {rel}")
        require(sha256(path) == expected, f"frozen semantic hash pinned: {rel}")

    release = load_json(RELEASE_MANIFEST)
    require(release.get("release_candidate") == "semantic_relevance_v0.28.0-r8",
            "release manifest candidate identity = semantic_relevance_v0.28.0-r8")
    require(release.get("release_status") == "FROZEN_BEFORE_FINAL_HOLDOUT",
            "release status = FROZEN_BEFORE_FINAL_HOLDOUT")
    require(release.get("decision") == "READY_FOR_FINAL_HOLDOUT_EXECUTION_REVIEW",
            "release decision = READY_FOR_FINAL_HOLDOUT_EXECUTION_REVIEW")

    evidence = release.get("evidence", {})
    require("targeted_regression" in evidence, "release manifest contains targeted evidence")
    require("full_210_consumed_development" in evidence, "release manifest contains full-210 evidence")
    require("stability" in evidence, "release manifest contains stability evidence")
    require(evidence["full_210_consumed_development"].get("case_count") == 210,
            "release manifest pins complete 210-case full run")
    require(evidence["stability"].get("required_runs") == 3,
            "release manifest pins three fresh stability runs")
    require(evidence["stability"].get("identical_cases") == 20,
            "release manifest pins 20/20 identical stability cases")

    frozen = release.get("frozen_hashes", {})
    for rel, expected in EXPECTED_SEMANTIC_HASHES.items():
        require(frozen.get(rel) == expected, f"release manifest pins semantic hash: {rel}")
    require(frozen.get("evals/datasets/semantic_relevance_v0.28_development.v5.0.0.csv") == EXPECTED_DEV_SHA,
            "release manifest pins 210-case development SHA")
    require(frozen.get("evals/datasets/semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv") == EXPECTED_HOLDOUT_SHA,
            "release manifest pins final-holdout SHA")

    protocol = load_json(PROTOCOL)
    require(protocol.get("candidate_release") == "semantic_relevance_v0.28.0-r8",
            "protocol candidate = semantic_relevance_v0.28.0-r8")
    require(protocol.get("evidence_role") == "independent_final_holdout",
            "protocol evidence role = independent_final_holdout")
    policy = protocol.get("execution_policy", {})
    require(policy.get("fresh_runs_permitted") == 1, "protocol permits exactly one fresh final-holdout run")
    require(policy.get("case_count") == 30, "protocol case count = 30")
    require(policy.get("recovery_policy") == "resume_recorded_run_only",
            "protocol recovery = resume recorded run only")
    require(policy.get("rerun_after_completed_result_permitted") is False,
            "protocol forbids rerun after completed result")
    require(policy.get("retune_after_result_permitted") is False,
            "protocol forbids retuning same candidate after result")
    require(policy.get("relabel_after_result_permitted") is False,
            "protocol forbids relabeling after result")
    require(policy.get("confirmation_token") == "FINAL_V0_28_R8",
            "execution confirmation token frozen")
    checks = protocol.get("preregistered_checks", {})
    require(set(checks) == EXPECTED_CHECK_NAMES,
            "exactly eight final-holdout quality checks preregistered")

    hold = load_csv(FINAL_HOLDOUT)
    dev = load_csv(DEVELOPMENT)
    require(len(hold) == 30, "final holdout contains exactly 30 cases")
    require(len({r["case_id"] for r in hold}) == 30, "final-holdout case IDs unique")
    require(all(r.get("review_status") == "LABELLED" for r in hold),
            "all final-holdout human labels remain frozen as LABELLED")

    q = Counter(r["query_id"] for r in hold)
    p = Counter(pos(r["case_id"]) for r in hold)
    s = Counter(int(r["human_score"]) for r in hold)
    c = Counter(r["candidate_source"] for r in hold)
    require(set(q) == {f"Q{i:02d}" for i in range(1, 13)},
            "final holdout: Q01-Q12 all represented")
    require(all(v in (2, 3) for v in q.values()),
            "final holdout: each query contributes 2 or 3 cases")
    require(p == Counter({"T01": 6, "T02": 6, "T03": 6, "T04": 6, "T05": 6}),
            "final holdout: six cases from every sampling position")
    require(s == Counter({0: 19, 1: 3, 2: 4, 3: 3, 4: 1}),
            "final holdout: frozen human-score distribution")
    require(c == Counter({"semantic_retrieval": 24, "random_negative_candidate": 6}),
            "final holdout: 24 retrieval + 6 random candidates")

    hold_ids = {r["case_id"] for r in hold}
    dev_ids = {r["case_id"] for r in dev}
    require(hold_ids.isdisjoint(dev_ids), "final-holdout case IDs do not overlap consumed development")

    hold_isbn = {canonical_isbn(r.get("isbn13", "")) for r in hold}
    dev_isbn = {canonical_isbn(r.get("isbn13", "")) for r in dev}
    hold_ta = {title_author_key(r) for r in hold}
    dev_ta = {title_author_key(r) for r in dev}
    require(not (hold_isbn & dev_isbn), "final-holdout ISBN identities do not overlap consumed development")
    require(not (hold_ta & dev_ta), "final-holdout title+author identities do not overlap consumed development")

    split = load_json(SPLIT_MANIFEST)
    require(split.get("judge_outputs_used_for_split") is False, "split remained blind to judge output")
    require(split.get("human_labels_frozen_before_judge") is True, "human labels were frozen before judge")
    require(split["final_holdout"]["sha256"] == EXPECTED_HOLDOUT_SHA,
            "split manifest pins final-holdout SHA")

    validation_lock = load_json(VALIDATION_LOCK)
    require(validation_lock.get("evidence_role") == "independent_validation",
            "validation lock role remains independent_validation")
    require(validation_lock.get("status") == "INDEPENDENT_VALIDATION_COMPLETED",
            "validation status = INDEPENDENT_VALIDATION_COMPLETED")
    require(int(validation_lock.get("judge_run_count", -1)) == 1,
            "validation judge_run_count = 1")
    require(validation_lock.get("validation_decision") == "REVIEW_STOP_FINAL_HOLDOUT",
            "historical r4 validation failure decision preserved")

    final_lock = load_json(FINAL_LOCK)
    require(final_lock.get("evidence_role") == "independent_final_holdout",
            "final lock role = independent_final_holdout")
    require(final_lock.get("status") == "LOCKED_DO_NOT_RUN",
            "final holdout status = LOCKED_DO_NOT_RUN")
    require(int(final_lock.get("judge_run_count", -1)) == 0,
            "final holdout judge_run_count = 0")
    require(final_lock.get("judge_outputs_used_for_split") is False,
            "final holdout split remained blind")
    require(final_lock.get("human_labels_frozen") is True,
            "final holdout human labels remain frozen")
    require(final_lock.get("finality_marker_present") is False,
            "lock records no finality marker")

    require(not FINALITY_MARKER.exists(),
            "repository final-holdout finality marker does not exist")

    print()
    print("FINAL-HOLDOUT PRE-FLIGHT: PASS")
    print("NO JUDGE CALLS WERE MADE.")
    print("Final holdout remains LOCKED_DO_NOT_RUN / judge_run_count=0.")
    print("Protocol confirmation token for separately reviewed execution: FINAL_V0_28_R8")


if __name__ == "__main__":
    main()
