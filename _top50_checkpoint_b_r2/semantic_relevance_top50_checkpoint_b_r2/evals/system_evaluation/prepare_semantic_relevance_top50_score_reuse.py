from __future__ import annotations

"""
Prepare score reuse for the frozen semantic-relevance top-50 diagnostic.

CHECKPOINT B ONLY:
- verify the Checkpoint A 600-row candidate artifact and lock;
- verify frozen v1/v2 judge inputs and judge-score artifacts;
- verify shared historical scores are consistent;
- reuse scores only when the current diagnostic payload exactly matches the
  frozen historical judge input for the same (query_id, isbn13);
- produce a novel-only judge input for Checkpoint C;
- make ZERO judge calls.

This script intentionally contains no LLM/judge invocation code.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

TOP50_RUN_DIR = (
    EVALS
    / "runs/semantic_relevance_top50_diagnostic_v1"
    / "20261005T113402Z_candidates"
)
TOP50_CANDIDATES_FILE = TOP50_RUN_DIR / "top50_candidates.csv"
TOP50_COLLECTION_LOCK_FILE = TOP50_RUN_DIR / "top50_collection_lock.json"

V1_RUN_DIR = (
    EVALS
    / "runs/semantic_relevance_system_eval_v1"
    / "20261002T152856Z_recommendations"
)
V2_RUN_DIR = (
    EVALS
    / "runs/semantic_relevance_system_eval_v2"
    / "20261004T140218Z_recommendations"
)

V1_JUDGE_INPUT_FILE = V1_RUN_DIR / "judge_input.csv"
V2_JUDGE_INPUT_FILE = V2_RUN_DIR / "judge_input.csv"
V1_JUDGE_SCORES_FILE = V1_RUN_DIR / "judge_scores.csv"
V2_JUDGE_SCORES_FILE = V2_RUN_DIR / "judge_scores.csv"

EXPECTED_TOP50_SHA256 = (
    "679dc2a8a3607db8d98e4ee868b5d7fc6566d11cb235eebed24e4676240abbad"
)
EXPECTED_V1_JUDGE_INPUT_SHA256 = (
    "4faa63f38015b6eef1e9bbad8bb6f868df780d05151abe90809544dfc50ff02f"
)
EXPECTED_V2_JUDGE_INPUT_SHA256 = (
    "8c5a9b58f8da0ba2adbb02ee75231cec8cbc3e30769b69e589db15ff88224134"
)
EXPECTED_V1_JUDGE_SCORES_SHA256 = (
    "d079a74c7f0473e46d28f19e37e4241a780804e972f3fa5f25b3a67e62a7bd11"
)
EXPECTED_V2_JUDGE_SCORES_SHA256 = (
    "4770986e868b26821131d2b36f6cc648f0cab68ae50acda0d1ba4df6d0bd0da1"
)

EXPECTED_TOP50_ROWS = 600
EXPECTED_QUERY_COUNT = 12
EXPECTED_CANDIDATES_PER_QUERY = 50

# These counts are deterministic for the frozen Checkpoint A artifact and the
# frozen v1/v2 score artifacts. They are recomputed below and then asserted.
EXPECTED_HISTORICAL_SHARED_PAIRS = 24
EXPECTED_HISTORICAL_SCORE_CONFLICTS = 0
EXPECTED_REUSE_V1_ONLY = 93
EXPECTED_REUSE_V2_ONLY = 96
EXPECTED_REUSE_V1_AND_V2 = 24
EXPECTED_REUSABLE_PAIRS = 213
EXPECTED_NOVEL_PAIRS = 387

PAYLOAD_FIELDS = ["query", "isbn13", "title", "description", "authors"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_hash(label: str, path: Path, expected: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash changed: expected={expected} actual={actual} path={path}"
        )
    print(f"PASS  {label} hash unchanged")
    return actual


def canonical_isbn(value: Any) -> str:
    """Normalize ISBN identity without changing its digits."""

    if pd.isna(value):
        raise ValueError("ISBN13 is missing.")

    text = str(value).strip().strip('"').strip("'")
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if not text:
        raise ValueError("ISBN13 is empty after normalization.")
    return text


def text_value(value: Any) -> str:
    """Return a stable textual value for exact frozen-payload comparison."""

    if pd.isna(value):
        return ""
    return str(value)


def canonical_optional_authors(value: Any) -> str:
    """Normalize only missing-value serialization for optional authors.

    The historical scorer persisted a few missing author values as the literal
    string ``nan`` while the frozen judge input represented the same value as
    blank. Treat those representations as the same missing optional metadata.

    Non-missing author text is otherwise preserved exactly; this function does
    not weaken query/title/description provenance checks.
    """

    if pd.isna(value):
        return ""

    text = str(value)
    normalized = text.strip().casefold()
    if normalized in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def payload_hash(row: pd.Series) -> str:
    payload = {
        field: (
            canonical_optional_authors(row.get(field, ""))
            if field == "authors"
            else text_value(row.get(field, ""))
        )
        for field in PAYLOAD_FIELDS
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def diagnostic_case_id(query_id: str, vector_rank: int) -> str:
    return f"TOP50_{query_id}_VR{vector_rank:02d}"


def load_top50() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and verify the immutable Checkpoint A candidate pool."""

    lock = load_json(TOP50_COLLECTION_LOCK_FILE)

    if lock.get("status") != "TOP50_CANDIDATES_FROZEN_BEFORE_SCORE_REUSE":
        raise ValueError(
            "Checkpoint A lock is not in TOP50_CANDIDATES_FROZEN_BEFORE_SCORE_REUSE state."
        )

    if int(lock.get("judge_calls_performed", -1)) != 0:
        raise ValueError("Checkpoint A lock unexpectedly reports judge calls.")

    if int(lock.get("query_book_pairs", -1)) != EXPECTED_TOP50_ROWS:
        raise ValueError("Checkpoint A lock does not describe 600 query-book pairs.")

    lock_hash = str(lock.get("top50_candidates_sha256", ""))
    if lock_hash != EXPECTED_TOP50_SHA256:
        raise ValueError(
            "Checkpoint A lock points to an unexpected candidate artifact: "
            f"expected={EXPECTED_TOP50_SHA256} lock={lock_hash}"
        )

    check_hash(
        "Checkpoint A top-50 candidates",
        TOP50_CANDIDATES_FILE,
        EXPECTED_TOP50_SHA256,
    )

    frame = pd.read_csv(
        TOP50_CANDIDATES_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    required = {
        "query_id",
        "query",
        "vector_rank",
        "isbn13",
        "title",
        "description",
        "authors",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"top50_candidates.csv missing required columns: {sorted(missing)}"
        )

    if len(frame) != EXPECTED_TOP50_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TOP50_ROWS} top-50 rows; got {len(frame)}"
        )

    frame["query_id"] = frame["query_id"].astype(str)
    frame["isbn13"] = frame["isbn13"].map(canonical_isbn)
    frame["vector_rank"] = frame["vector_rank"].astype(int)
    frame["authors"] = frame["authors"].map(canonical_optional_authors)

    if frame.duplicated(["query_id", "isbn13"]).any():
        dupes = frame.loc[
            frame.duplicated(["query_id", "isbn13"], keep=False),
            ["query_id", "isbn13", "vector_rank"],
        ]
        raise ValueError(
            "Duplicate query-book pairs in frozen top-50 artifact:\n"
            + dupes.to_string(index=False)
        )

    counts = frame.groupby("query_id").size()
    if len(counts) != EXPECTED_QUERY_COUNT or not (
        counts == EXPECTED_CANDIDATES_PER_QUERY
    ).all():
        raise ValueError(
            "Expected exactly 50 candidates for each of 12 queries; got "
            f"{counts.to_dict()}"
        )

    for query_id, group in frame.groupby("query_id"):
        ranks = sorted(group["vector_rank"].tolist())
        if ranks != list(range(1, 51)):
            raise ValueError(
                f"{query_id}: vector_rank must be exactly 1..50; got {ranks}"
            )

    frame["diagnostic_case_id"] = frame.apply(
        lambda row: diagnostic_case_id(
            str(row["query_id"]),
            int(row["vector_rank"]),
        ),
        axis=1,
    )

    if frame["diagnostic_case_id"].duplicated().any():
        raise ValueError("Duplicate diagnostic_case_id values found.")

    return frame, lock


def load_historical_source(
    *,
    label: str,
    judge_input_file: Path,
    judge_scores_file: Path,
    expected_input_hash: str,
    expected_scores_hash: str,
) -> pd.DataFrame:
    """Load one frozen system-evaluation judge source and validate its linkage."""

    check_hash(
        f"{label} frozen judge input",
        judge_input_file,
        expected_input_hash,
    )
    check_hash(
        f"{label} frozen judge scores",
        judge_scores_file,
        expected_scores_hash,
    )

    judge_input = pd.read_csv(
        judge_input_file,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    scores = pd.read_csv(
        judge_scores_file,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    required_input = {
        "query_id",
        "query",
        "rank",
        "isbn13",
        "title",
        "description",
    }
    missing_input = required_input - set(judge_input.columns)
    if missing_input:
        raise ValueError(
            f"{label} judge_input.csv missing columns: {sorted(missing_input)}"
        )

    required_scores = {
        "case_id",
        "query_id",
        "query",
        "rank",
        "isbn13",
        "title",
        "judge_score",
        "match_level",
        "has_relevant_evidence",
    }
    missing_scores = required_scores - set(scores.columns)
    if missing_scores:
        raise ValueError(
            f"{label} judge_scores.csv missing columns: {sorted(missing_scores)}"
        )

    if "authors" not in judge_input.columns:
        judge_input["authors"] = ""
    if "authors" not in scores.columns:
        scores["authors"] = ""

    # Some historical scorer rows serialized a missing optional author as the
    # literal text "nan", while judge_input.csv retained it as blank. Normalize
    # only that optional field before provenance comparison.
    judge_input["authors"] = judge_input["authors"].map(
        canonical_optional_authors
    )
    scores["authors"] = scores["authors"].map(
        canonical_optional_authors
    )

    if len(judge_input) != 120 or len(scores) != 120:
        raise ValueError(
            f"{label}: expected 120 judge-input rows and 120 score rows; "
            f"got input={len(judge_input)} scores={len(scores)}"
        )

    for frame in [judge_input, scores]:
        frame["query_id"] = frame["query_id"].astype(str)
        frame["isbn13"] = frame["isbn13"].map(canonical_isbn)
        frame["rank"] = frame["rank"].astype(int)

    if judge_input.duplicated(["query_id", "isbn13"]).any():
        raise ValueError(f"{label}: duplicate query-book pairs in judge input.")
    if scores.duplicated(["query_id", "isbn13"]).any():
        raise ValueError(f"{label}: duplicate query-book pairs in judge scores.")

    # Validate that every persisted score still points to the exact frozen input
    # identity for its query/rank pair before that score is considered reusable.
    input_by_rank = judge_input.set_index(["query_id", "rank"], drop=False)
    score_by_rank = scores.set_index(["query_id", "rank"], drop=False)

    if set(input_by_rank.index) != set(score_by_rank.index):
        raise ValueError(
            f"{label}: score rows and frozen judge-input rows do not share the same "
            "(query_id, rank) identities."
        )

    rows: list[dict[str, Any]] = []
    for key in sorted(input_by_rank.index):
        inp = input_by_rank.loc[key]
        score = score_by_rank.loc[key]

        for field in ["query_id", "query", "isbn13", "title", "authors"]:
            if field == "authors":
                input_value = canonical_optional_authors(inp.get(field, ""))
                score_value_text = canonical_optional_authors(
                    score.get(field, "")
                )
            else:
                input_value = text_value(inp.get(field, ""))
                score_value_text = text_value(score.get(field, ""))

            if input_value != score_value_text:
                raise ValueError(
                    f"{label}: score/input mismatch at {key} field={field}: "
                    f"input={inp.get(field)!r} score={score.get(field)!r}"
                )

        score_value = int(score["judge_score"])
        if score_value < 0 or score_value > 4:
            raise ValueError(
                f"{label}: invalid judge_score={score_value} at {key}"
            )

        source_case_id = str(score["case_id"])
        expected_case_id = f"SYS_{key[0]}_R{int(key[1]):02d}"
        if source_case_id != expected_case_id:
            raise ValueError(
                f"{label}: unexpected case_id at {key}: "
                f"expected={expected_case_id} actual={source_case_id}"
            )

        rows.append(
            {
                "source": label,
                "source_case_id": source_case_id,
                "query_id": str(inp["query_id"]),
                "query": text_value(inp["query"]),
                "source_rank": int(inp["rank"]),
                "isbn13": canonical_isbn(inp["isbn13"]),
                "title": text_value(inp["title"]),
                "description": text_value(inp["description"]),
                "authors": canonical_optional_authors(
                    inp.get("authors", "")
                ),
                "judge_score": score_value,
                "match_level": text_value(score["match_level"]),
                "has_relevant_evidence": text_value(
                    score["has_relevant_evidence"]
                ),
                "historical_payload_sha256": payload_hash(inp),
            }
        )

    result = pd.DataFrame(rows)
    if len(result) != 120:
        raise AssertionError(f"{label}: internal source normalization failed.")
    return result


def build_historical_index(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], pd.DataFrame]:
    """Index historical scores and fail if shared semantic outcomes conflict."""

    history: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for frame in [v1, v2]:
        for row in frame.to_dict("records"):
            key = (str(row["query_id"]), str(row["isbn13"]))
            history.setdefault(key, []).append(row)

    conflict_rows: list[dict[str, Any]] = []
    shared = 0

    for (query_id, isbn13), records in sorted(history.items()):
        if len(records) == 2:
            shared += 1
            a, b = records

            # The same query-book pair must have received the same semantic
            # classification in both frozen system runs before reuse is allowed.
            conflict_fields = []
            for field in [
                "judge_score",
                "match_level",
                "has_relevant_evidence",
            ]:
                if text_value(a[field]) != text_value(b[field]):
                    conflict_fields.append(field)

            # The frozen judge payload itself must also be identical.
            if a["historical_payload_sha256"] != b["historical_payload_sha256"]:
                conflict_fields.append("historical_payload")

            if conflict_fields:
                conflict_rows.append(
                    {
                        "query_id": query_id,
                        "isbn13": isbn13,
                        "v1_case_id": a["source_case_id"]
                        if a["source"] == "v1"
                        else b["source_case_id"],
                        "v2_case_id": a["source_case_id"]
                        if a["source"] == "v2"
                        else b["source_case_id"],
                        "conflict_fields": ",".join(conflict_fields),
                        "v1_judge_score": next(
                            r["judge_score"]
                            for r in records
                            if r["source"] == "v1"
                        ),
                        "v2_judge_score": next(
                            r["judge_score"]
                            for r in records
                            if r["source"] == "v2"
                        ),
                    }
                )

    conflicts = pd.DataFrame(
        conflict_rows,
        columns=[
            "query_id",
            "isbn13",
            "v1_case_id",
            "v2_case_id",
            "conflict_fields",
            "v1_judge_score",
            "v2_judge_score",
        ],
    )

    if shared != EXPECTED_HISTORICAL_SHARED_PAIRS:
        raise ValueError(
            "Unexpected v1/v2 historical score overlap: "
            f"expected={EXPECTED_HISTORICAL_SHARED_PAIRS} actual={shared}"
        )

    if len(conflicts) != EXPECTED_HISTORICAL_SCORE_CONFLICTS:
        raise ValueError(
            "Conflicting historical semantic outcomes found. "
            f"Expected {EXPECTED_HISTORICAL_SCORE_CONFLICTS}; got {len(conflicts)}."
        )

    return history, conflicts


def prepare_reuse(
    top50: pd.DataFrame,
    history: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the 600-row reuse map and novel-only judge input."""

    reuse_rows: list[dict[str, Any]] = []
    payload_mismatches: list[dict[str, Any]] = []

    for candidate in top50.sort_values(
        ["query_id", "vector_rank"],
        kind="stable",
    ).to_dict("records"):
        key = (str(candidate["query_id"]), str(candidate["isbn13"]))
        records = history.get(key, [])

        candidate_series = pd.Series(candidate)
        current_payload_sha = payload_hash(candidate_series)

        source_names = sorted(record["source"] for record in records)
        if source_names == ["v1", "v2"]:
            score_source = "v1_and_v2"
        elif source_names == ["v1"]:
            score_source = "v1"
        elif source_names == ["v2"]:
            score_source = "v2"
        elif not source_names:
            score_source = "UNSCORED"
        else:
            raise ValueError(
                f"Unexpected historical source combination for {key}: {source_names}"
            )

        v1_record = next(
            (record for record in records if record["source"] == "v1"),
            None,
        )
        v2_record = next(
            (record for record in records if record["source"] == "v2"),
            None,
        )

        # Existing scores are reusable only if the full judge payload matches
        # the current diagnostic candidate. A mismatch is a provenance failure,
        # not something to silently rescore.
        for record in records:
            if record["historical_payload_sha256"] != current_payload_sha:
                mismatch_fields = []
                for field in PAYLOAD_FIELDS:
                    if field == "authors":
                        current_value = canonical_optional_authors(
                            candidate.get(field, "")
                        )
                        historical_value = canonical_optional_authors(
                            record.get(field, "")
                        )
                    else:
                        current_value = text_value(candidate.get(field, ""))
                        historical_value = text_value(record.get(field, ""))

                    if current_value != historical_value:
                        mismatch_fields.append(field)

                payload_mismatches.append(
                    {
                        "diagnostic_case_id": candidate["diagnostic_case_id"],
                        "query_id": candidate["query_id"],
                        "vector_rank": candidate["vector_rank"],
                        "isbn13": candidate["isbn13"],
                        "source": record["source"],
                        "source_case_id": record["source_case_id"],
                        "mismatch_fields": ",".join(mismatch_fields),
                        "current_payload_sha256": current_payload_sha,
                        "historical_payload_sha256": record[
                            "historical_payload_sha256"
                        ],
                    }
                )

        selected = v2_record or v1_record

        row = {
            "diagnostic_case_id": candidate["diagnostic_case_id"],
            "query_id": candidate["query_id"],
            "query": candidate["query"],
            "vector_rank": int(candidate["vector_rank"]),
            "isbn13": candidate["isbn13"],
            "title": candidate["title"],
            "authors": candidate["authors"],
            "score_source": score_source,
            "reused_judge_score": (
                int(selected["judge_score"]) if selected is not None else ""
            ),
            "reused_match_level": (
                selected["match_level"] if selected is not None else ""
            ),
            "reused_has_relevant_evidence": (
                selected["has_relevant_evidence"]
                if selected is not None
                else ""
            ),
            "current_payload_sha256": current_payload_sha,
            "v1_case_id": v1_record["source_case_id"]
            if v1_record is not None
            else "",
            "v1_source_rank": v1_record["source_rank"]
            if v1_record is not None
            else "",
            "v1_judge_score": v1_record["judge_score"]
            if v1_record is not None
            else "",
            "v2_case_id": v2_record["source_case_id"]
            if v2_record is not None
            else "",
            "v2_source_rank": v2_record["source_rank"]
            if v2_record is not None
            else "",
            "v2_judge_score": v2_record["judge_score"]
            if v2_record is not None
            else "",
        }
        reuse_rows.append(row)

    reuse_map = pd.DataFrame(reuse_rows)

    mismatches = pd.DataFrame(
        payload_mismatches,
        columns=[
            "diagnostic_case_id",
            "query_id",
            "vector_rank",
            "isbn13",
            "source",
            "source_case_id",
            "mismatch_fields",
            "current_payload_sha256",
            "historical_payload_sha256",
        ],
    )

    if not mismatches.empty:
        raise ValueError(
            "Existing judge scores cannot be safely reused because current "
            "top-50 payloads differ from frozen historical judge inputs. "
            f"Mismatch rows={len(mismatches)}. Inspect top50_payload_mismatches.csv."
        )

    novel = top50.merge(
        reuse_map[
            [
                "diagnostic_case_id",
                "score_source",
            ]
        ],
        on="diagnostic_case_id",
        how="left",
        validate="one_to_one",
    )
    novel = novel.loc[novel["score_source"] == "UNSCORED"].copy()

    novel = novel[
        [
            "diagnostic_case_id",
            "query_id",
            "query",
            "vector_rank",
            "isbn13",
            "title",
            "description",
            "authors",
        ]
    ].sort_values(["query_id", "vector_rank"], kind="stable")

    return reuse_map, novel, mismatches


def validate_expected_counts(
    *,
    reuse_map: pd.DataFrame,
    novel: pd.DataFrame,
) -> dict[str, int]:
    counts = reuse_map["score_source"].value_counts().to_dict()

    actual = {
        "v1_only": int(counts.get("v1", 0)),
        "v2_only": int(counts.get("v2", 0)),
        "v1_and_v2": int(counts.get("v1_and_v2", 0)),
        "reusable": int(
            (reuse_map["score_source"] != "UNSCORED").sum()
        ),
        "novel": int((reuse_map["score_source"] == "UNSCORED").sum()),
    }

    expected = {
        "v1_only": EXPECTED_REUSE_V1_ONLY,
        "v2_only": EXPECTED_REUSE_V2_ONLY,
        "v1_and_v2": EXPECTED_REUSE_V1_AND_V2,
        "reusable": EXPECTED_REUSABLE_PAIRS,
        "novel": EXPECTED_NOVEL_PAIRS,
    }

    if actual != expected:
        raise ValueError(
            "Checkpoint B reuse counts differ from the deterministic expectation "
            f"for the frozen Checkpoint A artifact. expected={expected} actual={actual}"
        )

    if len(novel) != EXPECTED_NOVEL_PAIRS:
        raise ValueError(
            f"Expected {EXPECTED_NOVEL_PAIRS} novel judge-input rows; got {len(novel)}"
        )

    if reuse_map["diagnostic_case_id"].duplicated().any():
        raise ValueError("Reuse map contains duplicate diagnostic case IDs.")

    if novel["diagnostic_case_id"].duplicated().any():
        raise ValueError("Novel judge input contains duplicate diagnostic case IDs.")

    if novel.duplicated(["query_id", "isbn13"]).any():
        raise ValueError("Novel judge input contains duplicate query-book pairs.")

    # The novel set remains a subset of the same 12 frozen queries, but each
    # query can have a different count because score reuse differs by query.
    if set(novel["query_id"]) - set(reuse_map["query_id"]):
        raise ValueError("Novel judge input contains an unknown query_id.")

    return actual


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint B: prepare frozen-score reuse and novel-only judge input "
            "for the semantic-relevance top-50 diagnostic."
        )
    )
    parser.parse_args()

    print("Semantic relevance top-50 diagnostic - Checkpoint B")
    print("---------------------------------------------------")
    print("Purpose: reuse frozen v1/v2 judge evidence + prepare novel-only input")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Top-50 candidate mutation: NONE")
    print()

    top50, top50_lock = load_top50()

    v1 = load_historical_source(
        label="v1",
        judge_input_file=V1_JUDGE_INPUT_FILE,
        judge_scores_file=V1_JUDGE_SCORES_FILE,
        expected_input_hash=EXPECTED_V1_JUDGE_INPUT_SHA256,
        expected_scores_hash=EXPECTED_V1_JUDGE_SCORES_SHA256,
    )
    v2 = load_historical_source(
        label="v2",
        judge_input_file=V2_JUDGE_INPUT_FILE,
        judge_scores_file=V2_JUDGE_SCORES_FILE,
        expected_input_hash=EXPECTED_V2_JUDGE_INPUT_SHA256,
        expected_scores_hash=EXPECTED_V2_JUDGE_SCORES_SHA256,
    )

    history, conflicts = build_historical_index(v1, v2)
    print(
        "PASS  historical v1/v2 shared-score integrity: "
        f"{EXPECTED_HISTORICAL_SHARED_PAIRS} shared pairs, "
        f"{len(conflicts)} conflicts"
    )

    # Create the mismatch artifact before failing, so a provenance failure is
    # inspectable rather than represented only by a traceback.
    reuse_map, novel, mismatches = prepare_reuse(top50, history)

    conflicts_file = TOP50_RUN_DIR / "top50_existing_score_conflicts.csv"
    mismatches_file = TOP50_RUN_DIR / "top50_payload_mismatches.csv"
    conflicts.to_csv(conflicts_file, index=False, encoding="utf-8")
    mismatches.to_csv(mismatches_file, index=False, encoding="utf-8")

    counts = validate_expected_counts(reuse_map=reuse_map, novel=novel)

    reuse_map_file = TOP50_RUN_DIR / "top50_score_reuse_map.csv"
    novel_file = TOP50_RUN_DIR / "top50_novel_judge_input.csv"

    reuse_map.to_csv(reuse_map_file, index=False, encoding="utf-8")
    novel.to_csv(novel_file, index=False, encoding="utf-8")

    metadata = {
        "diagnostic_id": "semantic_relevance_top50_diagnostic_v1",
        "stage": "CHECKPOINT_B_SCORE_REUSE_PREPARED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(TOP50_RUN_DIR),
        "top50_candidates_sha256": sha256(TOP50_CANDIDATES_FILE),
        "top50_collection_lock_sha256": sha256(TOP50_COLLECTION_LOCK_FILE),
        "v1_judge_input_sha256": sha256(V1_JUDGE_INPUT_FILE),
        "v1_judge_scores_sha256": sha256(V1_JUDGE_SCORES_FILE),
        "v2_judge_input_sha256": sha256(V2_JUDGE_INPUT_FILE),
        "v2_judge_scores_sha256": sha256(V2_JUDGE_SCORES_FILE),
        "historical_shared_pairs": EXPECTED_HISTORICAL_SHARED_PAIRS,
        "historical_score_conflicts": len(conflicts),
        "payload_mismatches": len(mismatches),
        "reused_v1_only": counts["v1_only"],
        "reused_v2_only": counts["v2_only"],
        "reused_v1_and_v2": counts["v1_and_v2"],
        "reusable_pairs": counts["reusable"],
        "novel_pairs": counts["novel"],
        "total_pairs": len(reuse_map),
        "score_reuse_map_file": reuse_map_file.name,
        "score_reuse_map_sha256": sha256(reuse_map_file),
        "novel_judge_input_file": novel_file.name,
        "novel_judge_input_sha256": sha256(novel_file),
        "score_conflicts_file": conflicts_file.name,
        "score_conflicts_sha256": sha256(conflicts_file),
        "payload_mismatches_file": mismatches_file.name,
        "payload_mismatches_sha256": sha256(mismatches_file),
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "reuse_policy": (
            "Reuse only exact (query_id,isbn13) historical judgments whose "
            "query/title/description/authors payload is byte-for-byte text-equivalent "
            "to the frozen Checkpoint A candidate payload."
        ),
        "next_stage": "CHECKPOINT_C_SCORE_NOVEL_PAIRS_ONLY",
    }

    metadata_file = TOP50_RUN_DIR / "top50_score_reuse_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    lock = {
        "status": "TOP50_SCORE_REUSE_FROZEN_BEFORE_NOVEL_JUDGING",
        "diagnostic_id": metadata["diagnostic_id"],
        "run_directory": str(TOP50_RUN_DIR),
        "top50_candidates_sha256": metadata["top50_candidates_sha256"],
        "score_reuse_map_sha256": metadata["score_reuse_map_sha256"],
        "novel_judge_input_sha256": metadata["novel_judge_input_sha256"],
        "v1_judge_scores_sha256": metadata["v1_judge_scores_sha256"],
        "v2_judge_scores_sha256": metadata["v2_judge_scores_sha256"],
        "historical_score_conflicts": 0,
        "payload_mismatches": 0,
        "reusable_pairs": counts["reusable"],
        "novel_pairs": counts["novel"],
        "total_pairs": len(reuse_map),
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "next_stage": "CHECKPOINT_C_SCORE_NOVEL_PAIRS_ONLY",
    }

    lock_file = TOP50_RUN_DIR / "top50_score_reuse_lock.json"
    lock_file.write_text(
        json.dumps(lock, indent=2) + "\n",
        encoding="utf-8",
    )

    # Compact per-query counts make it easy to review score reuse before any
    # expensive judge execution.
    per_query = (
        reuse_map.assign(
            reusable=reuse_map["score_source"].ne("UNSCORED").astype(int),
            novel=reuse_map["score_source"].eq("UNSCORED").astype(int),
        )
        .groupby("query_id", as_index=False)[["reusable", "novel"]]
        .sum()
        .sort_values("query_id")
    )

    print()
    print("CHECKPOINT B INTEGRITY")
    print("----------------------")
    print(
        f"Historical shared v1/v2 pairs: {EXPECTED_HISTORICAL_SHARED_PAIRS}/"
        f"{EXPECTED_HISTORICAL_SHARED_PAIRS}"
    )
    print("Historical score conflicts:    0")
    print("Historical payload mismatches: 0")
    print(f"Reusable v1-only pairs:        {counts['v1_only']}")
    print(f"Reusable v2-only pairs:        {counts['v2_only']}")
    print(f"Reusable v1+v2 pairs:          {counts['v1_and_v2']}")
    print(f"Reusable pairs total:          {counts['reusable']}/600")
    print(f"Novel judge-input pairs:       {counts['novel']}/600")
    print()
    print("Per-query reuse / novel counts")
    print("------------------------------")
    for row in per_query.itertuples(index=False):
        print(
            f"{row.query_id}: reusable={int(row.reusable):2d}  "
            f"novel={int(row.novel):2d}"
        )

    print()
    print("TOP-50 DIAGNOSTIC CHECKPOINT B: PASS")
    print(f"Run directory: {TOP50_RUN_DIR}")
    print(f"Score reuse map SHA-256: {sha256(reuse_map_file)}")
    print(f"Novel judge input SHA-256: {sha256(novel_file)}")
    print("Judge calls performed: 0")
    print("Human relevance labels used: NO")
    print(
        "Next: review Checkpoint B before running the novel-only calibrated judge."
    )


if __name__ == "__main__":
    main()
