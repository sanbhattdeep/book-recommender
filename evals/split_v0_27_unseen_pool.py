"""Create the frozen 30/30 v0.27 validation/final-holdout split.

The split is LABEL-AWARE but JUDGE-BLIND:
- all 60 human labels must already be final;
- no judge output may have been generated;
- exact score-distribution and sampling-slot balance are pre-registered;
- each query contributes 2 or 3 cases to validation and the complement to holdout;
- the final holdout is written LOCKED_DO_NOT_RUN.

r5 implementation note
----------------------
r4 used ``scipy.optimize.milp`` only as a constraint solver. On the target
Windows machine, Application Control blocks one of SciPy's native DLLs.
r5 preserves the SAME split constraints but replaces SciPy with a deterministic
pure-Python constraint search. The search uses only case identity, human_score,
query_id and fixed sampling slot. It never reads or invokes judge predictions.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
import subprocess
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS = REPO_ROOT / "evals"
DATASETS = EVALS / "datasets"
POOL = DATASETS / "semantic_relevance_unseen_pool.v3.0.0.csv"
VALIDATION = DATASETS / "semantic_relevance_validation.v3.0.0.csv"
HOLDOUT = DATASETS / "semantic_relevance_final_holdout.v3.0.0.DO_NOT_RUN_YET.csv"
LOCK = DATASETS / "semantic_relevance_final_holdout_lock.v3.0.0.json"
SPLIT_MANIFEST = DATASETS / "semantic_relevance_unseen_pool_manifest.v3.0.0.json"
SPEC = DATASETS / "semantic_relevance_unseen_pool_split_spec.v3.0.0.json"
FREEZE = EVALS / "releases" / "semantic_relevance_v0.27.0_r5_development_candidate.json"
EXPECTED_FREEZE_SHA = "3de6ee3a6274a5d066f17ff24bc965b05dfb8d60b785c0210d45d743538d84e5"
SEED = "semantic-relevance-v0.27-unseen-v3-label-aware-seed-20260925"
SPLIT_METHOD = "label_aware_balanced_deterministic_backtracking_30_30_before_any_judge_output"

TARGET_VALIDATION_SCORE_COUNTS = {0: 18, 1: 2, 2: 2, 3: 8, 4: 0}
TARGET_HOLDOUT_SCORE_COUNTS = {0: 18, 1: 2, 2: 2, 3: 7, 4: 1}
TARGET_SLOT_COUNT = 6
SLOTS = ("T01", "T02", "T03", "T04", "T05")
MAX_SEARCH_NODES = 2_000_000

# Constraint vector layout used by the deterministic search:
# total, score0..score4, T01..T05
TARGET_VECTOR = (
    30,
    TARGET_VALIDATION_SCORE_COUNTS[0],
    TARGET_VALIDATION_SCORE_COUNTS[1],
    TARGET_VALIDATION_SCORE_COUNTS[2],
    TARGET_VALIDATION_SCORE_COUNTS[3],
    TARGET_VALIDATION_SCORE_COUNTS[4],
    TARGET_SLOT_COUNT,
    TARGET_SLOT_COUNT,
    TARGET_SLOT_COUNT,
    TARGET_SLOT_COUNT,
    TARGET_SLOT_COUNT,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash_int(value: str) -> int:
    """Return a stable SHA-256 integer used only for deterministic ordering."""
    return int(
        hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).hexdigest(),
        16,
    )


def sampling_slot(case_id: str) -> str:
    slot = case_id.rsplit("_", 1)[-1]
    if slot not in SLOTS:
        raise ValueError(f"Unexpected sampling slot in case_id {case_id}")
    return slot


def counts(series: pd.Series) -> dict[int, int]:
    numeric = pd.to_numeric(series, errors="raise").astype(int)
    return {score: int((numeric == score).sum()) for score in range(5)}


def option_vector(records: list[dict[str, Any]], indices: tuple[int, ...]) -> tuple[int, ...]:
    vector = [len(indices)] + [0] * 10
    for index in indices:
        record = records[index]
        score = int(record["human_score"])
        slot_index = SLOTS.index(sampling_slot(str(record["case_id"])))
        vector[1 + score] += 1
        vector[6 + slot_index] += 1
    return tuple(vector)


def add_vectors(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(left, right))


def exceeds_target(vector: tuple[int, ...]) -> bool:
    return any(value > limit for value, limit in zip(vector, TARGET_VECTOR))


def build_query_options(df: pd.DataFrame) -> list[tuple[str, list[dict[str, Any]]]]:
    """Build all legal 2/3-case choices per query, deterministically ordered."""
    groups: list[tuple[str, list[dict[str, Any]]]] = []

    for query_id, group in df.groupby("query_id", sort=True):
        group = group.sort_values("case_id").reset_index(drop=True)
        if len(group) != 5:
            raise ValueError(f"{query_id} must contain exactly 5 pool cases, got {len(group)}")

        records = group.to_dict("records")
        slots = [sampling_slot(str(record["case_id"])) for record in records]
        if set(slots) != set(SLOTS):
            raise ValueError(f"{query_id} must contain exactly one case from every sampling slot: {slots}")

        options: list[dict[str, Any]] = []
        for size in (2, 3):
            for indices in combinations(range(5), size):
                vector = option_vector(records, indices)
                if exceeds_target(vector):
                    # E.g. TARGET score-4 count is zero, so any option containing
                    # the sole score-4 case is impossible for validation.
                    continue
                case_ids = tuple(str(records[i]["case_id"]) for i in indices)
                option_key = f"{query_id}|{'|'.join(case_ids)}"
                options.append(
                    {
                        "case_ids": case_ids,
                        "vector": vector,
                        "order_key": stable_hash_int(option_key),
                    }
                )

        if not options:
            raise RuntimeError(f"No legal validation choices remain for {query_id}")

        options.sort(key=lambda item: (item["order_key"], item["case_ids"]))
        groups.append((str(query_id), options))

    if len(groups) != 12:
        raise ValueError(f"Expected 12 query groups, got {len(groups)}")

    # Search the most constrained query first. The secondary score-rarity
    # heuristic depends only on final human labels and is therefore still
    # label-aware/judge-blind. query_id makes the order fully deterministic.
    pool_score_counts = counts(df["human_score"])
    validation_targets = TARGET_VALIDATION_SCORE_COUNTS

    def rarity_score(query_id: str) -> int:
        group = df.loc[df["query_id"] == query_id]
        score = 0
        for raw in group["human_score"]:
            value = int(raw)
            if validation_targets[value] < pool_score_counts[value]:
                score += pool_score_counts[value] - validation_targets[value]
        return score

    groups.sort(
        key=lambda pair: (
            len(pair[1]),
            -rarity_score(pair[0]),
            pair[0],
        )
    )
    return groups


def suffix_bounds(groups: list[tuple[str, list[dict[str, Any]]]]) -> tuple[list[list[int]], list[list[int]]]:
    dimensions = len(TARGET_VECTOR)
    mins = [[0] * dimensions for _ in range(len(groups) + 1)]
    maxs = [[0] * dimensions for _ in range(len(groups) + 1)]

    for pos in range(len(groups) - 1, -1, -1):
        options = groups[pos][1]
        for dim in range(dimensions):
            values = [int(option["vector"][dim]) for option in options]
            mins[pos][dim] = mins[pos + 1][dim] + min(values)
            maxs[pos][dim] = maxs[pos + 1][dim] + max(values)

    return mins, maxs


def choose_validation_case_ids(df: pd.DataFrame) -> tuple[set[str], dict[str, Any]]:
    """Return one deterministic feasible assignment satisfying every frozen constraint."""
    required = {"case_id", "query_id", "human_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pool missing split columns: {sorted(missing)}")
    if len(df) != 60:
        raise ValueError(f"Expected 60 pool cases, got {len(df)}")
    if df["case_id"].duplicated().any():
        raise ValueError("Duplicate case_id values in pool.")

    groups = build_query_options(df)
    suffix_min, suffix_max = suffix_bounds(groups)
    zero = tuple(0 for _ in TARGET_VECTOR)
    dead_states: set[tuple[int, tuple[int, ...]]] = set()
    nodes_visited = 0

    def remaining_can_hit_target(pos: int, vector: tuple[int, ...]) -> bool:
        for dim, target in enumerate(TARGET_VECTOR):
            minimum = vector[dim] + suffix_min[pos][dim]
            maximum = vector[dim] + suffix_max[pos][dim]
            if minimum > target or maximum < target:
                return False
        return True

    def search(
        pos: int,
        vector: tuple[int, ...],
        path: list[tuple[str, tuple[str, ...]]],
    ) -> list[tuple[str, tuple[str, ...]]] | None:
        nonlocal nodes_visited
        nodes_visited += 1
        if nodes_visited > MAX_SEARCH_NODES:
            raise RuntimeError(
                "Deterministic split search exceeded its safety bound. "
                "No split artifacts were written."
            )

        if pos == len(groups):
            return path if vector == TARGET_VECTOR else None

        state = (pos, vector)
        if state in dead_states:
            return None
        if not remaining_can_hit_target(pos, vector):
            dead_states.add(state)
            return None

        query_id, options = groups[pos]
        for option in options:
            candidate = add_vectors(vector, option["vector"])
            if exceeds_target(candidate):
                continue
            if not remaining_can_hit_target(pos + 1, candidate):
                continue

            result = search(
                pos + 1,
                candidate,
                path + [(query_id, option["case_ids"])],
            )
            if result is not None:
                return result

        dead_states.add(state)
        return None

    selected_path = search(0, zero, [])
    if selected_path is None:
        raise RuntimeError(
            "Unable to construct the pre-registered balanced split with the "
            "deterministic pure-Python constraint search."
        )

    selected = {
        case_id
        for _query_id, case_ids in selected_path
        for case_id in case_ids
    }
    if len(selected) != 30:
        raise RuntimeError(f"Deterministic search returned {len(selected)} validation cases, expected 30")

    metadata = {
        "solver": "pure_python_deterministic_backtracking_with_suffix_bounds",
        "query_search_order": [query_id for query_id, _options in groups],
        "nodes_visited": nodes_visited,
        "dead_states": len(dead_states),
        "max_search_nodes": MAX_SEARCH_NODES,
    }
    return selected, metadata


def main() -> None:
    for path in (POOL, SPEC, FREEZE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    for path in (VALIDATION, HOLDOUT, LOCK, SPLIT_MANIFEST):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing split artifact: {path}")
    if sha256(FREEZE) != EXPECTED_FREEZE_SHA:
        raise ValueError("Frozen r5 candidate manifest hash mismatch.")

    spec = json.loads(SPEC.read_text(encoding="utf-8-sig"))
    if spec.get("split_method") != SPLIT_METHOD:
        raise ValueError("Unexpected split method in split specification.")
    if spec.get("split_seed") != SEED:
        raise ValueError("Unexpected split seed in split specification.")

    completed = subprocess.run(
        [sys.executable, str(EVALS / "validate_v0_27_unseen_pool.py")],
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        raise RuntimeError("Unseen-pool authoring validation failed; split not created.")

    df = pd.read_csv(POOL, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = df.sort_values("case_id").reset_index(drop=True)
    pool_score_counts = counts(df["human_score"])
    expected_pool = {
        int(key): int(value)
        for key, value in spec["expected_pool_human_score_distribution"].items()
    }
    if pool_score_counts != expected_pool:
        raise ValueError(
            "Human score distribution differs from the pre-registered split spec. "
            f"Expected={expected_pool} Actual={pool_score_counts}"
        )

    validation_case_ids, solver_metadata = choose_validation_case_ids(df)
    validation_mask = df["case_id"].isin(validation_case_ids)
    val = df.loc[validation_mask].copy()
    hol = df.loc[~validation_mask].copy()
    val = val.sort_values(["query_id", "case_id"]).reset_index(drop=True)
    hol = hol.sort_values(["query_id", "case_id"]).reset_index(drop=True)

    if len(val) != 30 or len(hol) != 30:
        raise ValueError(f"Split must be 30/30, got {len(val)}/{len(hol)}")
    if set(val.case_id) & set(hol.case_id):
        raise ValueError("Validation/final-holdout overlap.")
    if set(val.case_id) | set(hol.case_id) != set(df.case_id):
        raise ValueError("Split does not reconstruct full pool.")

    val_scores = counts(val["human_score"])
    hol_scores = counts(hol["human_score"])
    if val_scores != TARGET_VALIDATION_SCORE_COUNTS:
        raise ValueError(f"Validation score distribution mismatch: {val_scores}")
    if hol_scores != TARGET_HOLDOUT_SCORE_COUNTS:
        raise ValueError(f"Final holdout score distribution mismatch: {hol_scores}")

    val_slots = pd.Series([sampling_slot(x) for x in val["case_id"]]).value_counts().to_dict()
    hol_slots = pd.Series([sampling_slot(x) for x in hol["case_id"]]).value_counts().to_dict()
    expected_slots = {f"T{i:02d}": 6 for i in range(1, 6)}
    if val_slots != expected_slots or hol_slots != expected_slots:
        raise ValueError(
            f"Sampling-slot balance mismatch. validation={val_slots} holdout={hol_slots}"
        )

    val_query_counts = val["query_id"].value_counts().to_dict()
    hol_query_counts = hol["query_id"].value_counts().to_dict()
    if len(val_query_counts) != 12 or any(v not in (2, 3) for v in val_query_counts.values()):
        raise ValueError(f"Bad validation query counts: {val_query_counts}")
    if len(hol_query_counts) != 12 or any(v not in (2, 3) for v in hol_query_counts.values()):
        raise ValueError(f"Bad holdout query counts: {hol_query_counts}")

    if int((pd.to_numeric(val["human_score"]) == 4).sum()) != 0:
        raise ValueError("Validation unexpectedly contains a score-4 case.")
    if int((pd.to_numeric(hol["human_score"]) == 4).sum()) != 1:
        raise ValueError("Final holdout must contain the sole score-4 case.")

    val.to_csv(VALIDATION, index=False, encoding="utf-8")
    hol.to_csv(HOLDOUT, index=False, encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat()
    assignment = {
        case_id: "validation" for case_id in val["case_id"]
    } | {
        case_id: "final_holdout" for case_id in hol["case_id"]
    }

    manifest = {
        "schema_version": "1.2.0",
        "pool_version": "3.0.0",
        "split_seed": SEED,
        "split_method": SPLIT_METHOD,
        "created_at_utc": now,
        "judge_outputs_used_for_split": False,
        "human_labels_completed_before_split": True,
        "solver_metadata": solver_metadata,
        "frozen_candidate_manifest": str(FREEZE.relative_to(REPO_ROOT)),
        "frozen_candidate_manifest_sha256": sha256(FREEZE),
        "pool": {
            "path": str(POOL.relative_to(REPO_ROOT)),
            "cases": 60,
            "sha256": sha256(POOL),
            "human_score_distribution": pool_score_counts,
        },
        "validation": {
            "path": str(VALIDATION.relative_to(REPO_ROOT)),
            "cases": 30,
            "sha256": sha256(VALIDATION),
            "human_score_distribution": val_scores,
            "sampling_slot_distribution": val_slots,
            "query_distribution": val_query_counts,
            "case_ids": val.case_id.tolist(),
        },
        "final_holdout": {
            "path": str(HOLDOUT.relative_to(REPO_ROOT)),
            "cases": 30,
            "sha256": sha256(HOLDOUT),
            "human_score_distribution": hol_scores,
            "sampling_slot_distribution": hol_slots,
            "query_distribution": hol_query_counts,
            "case_ids": hol.case_id.tolist(),
            "status": "LOCKED_DO_NOT_RUN",
        },
        "assignment": assignment,
    }
    SPLIT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lock = {
        "schema_version": "1.0.0",
        "status": "LOCKED_DO_NOT_RUN",
        "reason": (
            "Final holdout must remain untouched until validation is complete "
            "and a candidate is frozen for one-time holdout evaluation."
        ),
        "created_at_utc": now,
        "final_holdout_path": str(HOLDOUT.relative_to(REPO_ROOT)),
        "final_holdout_sha256": sha256(HOLDOUT),
        "split_manifest_path": str(SPLIT_MANIFEST.relative_to(REPO_ROOT)),
        "split_manifest_sha256": sha256(SPLIT_MANIFEST),
        "frozen_candidate_manifest_sha256": sha256(FREEZE),
        "judge_run_count": 0,
    }
    LOCK.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("v0.27 unseen pool split complete")
    print("-------------------------------")
    print("PASS  split is human-label-aware and judge-blind")
    print("PASS  deterministic solver: pure Python; no SciPy/native MILP dependency")
    print(f"PASS  constraint-search nodes visited: {solver_metadata['nodes_visited']}")
    print("PASS  source pool:       60 labelled cases")
    print("PASS  validation:        30 cases")
    print("PASS  final holdout:     30 cases")
    print("PASS  each sampling slot T01-T05: 6 validation / 6 holdout")
    print(f"PASS  validation score distribution: {val_scores}")
    print(f"PASS  holdout score distribution:    {hol_scores}")
    print("PASS  each query contributes 2 or 3 cases to each half")
    print("PASS  validation/holdout disjoint")
    print("PASS  validation + holdout reconstruct pool")
    print("PASS  final holdout status: LOCKED_DO_NOT_RUN")
    print(f"Pool SHA-256:            {sha256(POOL)}")
    print(f"Validation SHA-256:      {sha256(VALIDATION)}")
    print(f"Final holdout SHA-256:   {sha256(HOLDOUT)}")
    print(f"Split manifest SHA-256:  {sha256(SPLIT_MANIFEST)}")
    print(f"Lock manifest SHA-256:   {sha256(LOCK)}")


if __name__ == "__main__":
    main()
