from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_V3_RUN = REPO_ROOT / "evals/runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"
DEFAULT_V4A_RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

EXPECTED_V3_RECOMMENDATIONS_SHA256 = "e5e136a6d611a527635d202ec5edb928e9de77a8a0db46fa0fcab295c69a4ba2"
EXPECTED_V4A_RECOMMENDATIONS_SHA256 = "5328ab30132b4d117bd2bacd9c54cb6ed460968c0bdaf10b2720c2f9256481b6"
EXPECTED_V4A_RERANKER_LOCK_SHA256 = "97a8cc18cf2e8c2cf5abec679f41de81546a945b6ae9b36ab06abd97d2508a22"
EXPECTED_V4A_RETRIEVAL_LOCK_SHA256 = "7587aa140c1cb26f8ac5addaf6873f3bc7ec05b6ec34775a6cbfe98d7ffa40d1"

EXPECTED_QUERY_IDS = [f"H{i:02d}" for i in range(1, 25)]
EXPECTED_QUERY_COUNT = 24
EXPECTED_DISPLAY_DEPTH = 10
EXPECTED_ROWS_PER_SYSTEM = 240


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def canonical_isbn(value: Any) -> str:
    text = safe_text(value).strip('"').strip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text:
        raise ValueError("ISBN is empty.")
    return text


def read_recommendations(path: Path, system: str) -> pd.DataFrame:
    require(path.exists(), f"Missing {system} recommendations: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")

    required = {"query_id", "query", "stratum", "rank", "isbn13", "title", "description"}
    missing = required - set(frame.columns)
    require(not missing, f"{system} recommendations missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
    frame["isbn13"] = frame["isbn13"].map(canonical_isbn)
    if "authors" not in frame.columns:
        frame["authors"] = ""

    require(len(frame) == EXPECTED_ROWS_PER_SYSTEM, f"{system}: expected 240 rows, got {len(frame)}")
    require(not frame.duplicated(["query_id", "isbn13"]).any(), f"{system}: duplicate query-book pair")

    counts = frame.groupby("query_id").size()
    require(len(counts) == EXPECTED_QUERY_COUNT, f"{system}: expected 24 queries, got {len(counts)}")
    require((counts == EXPECTED_DISPLAY_DEPTH).all(), f"{system}: expected exactly 10 rows/query")
    require(sorted(frame["query_id"].unique().tolist()) == EXPECTED_QUERY_IDS, f"{system}: query IDs changed")

    for qid, group in frame.groupby("query_id", sort=True):
        require(sorted(group["rank"].tolist()) == list(range(1, 11)), f"{system} {qid}: ranks not 1..10")
        require(group["query"].nunique() == 1, f"{system} {qid}: query text not unique")
        require(group["stratum"].nunique() == 1, f"{system} {qid}: stratum not unique")

    return frame


def validate_inputs(v3_run: Path, v4a_run: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path, Path]:
    v3_file = v3_run / "v3_holdout_recommendations.csv"
    v4a_file = v4a_run / "v4a_dev_recommendations.csv"
    v4a_reranker_lock = v4a_run / "v4a_dev_reranker.lock.json"
    v4a_retrieval_lock = v4a_run / "v4a_dev_retrieval.lock.json"

    require(sha256(v3_file) == EXPECTED_V3_RECOMMENDATIONS_SHA256, "Frozen v3b recommendations hash changed.")
    require(sha256(v4a_file) == EXPECTED_V4A_RECOMMENDATIONS_SHA256, "Frozen v4a recommendations hash changed.")
    require(sha256(v4a_reranker_lock) == EXPECTED_V4A_RERANKER_LOCK_SHA256, "Frozen v4a reranker lock changed.")
    require(sha256(v4a_retrieval_lock) == EXPECTED_V4A_RETRIEVAL_LOCK_SHA256, "Frozen v4a retrieval lock changed.")

    v3 = read_recommendations(v3_file, "v3b")
    v4a = read_recommendations(v4a_file, "v4a")

    v3_q = v3[["query_id", "query", "stratum"]].drop_duplicates().sort_values("query_id").reset_index(drop=True)
    v4_q = v4a[["query_id", "query", "stratum"]].drop_duplicates().sort_values("query_id").reset_index(drop=True)
    require(v3_q.equals(v4_q), "v3b/v4a query text or stratum mismatch.")

    return v3, v4a, v3_file, v4a_file, v4a_reranker_lock, v4a_retrieval_lock


def build_union(v3: pd.DataFrame, v4a: pd.DataFrame) -> pd.DataFrame:
    v3_cols = ["query_id", "query", "stratum", "isbn13", "title", "description", "authors", "rank"]
    v4_cols = ["query_id", "query", "stratum", "isbn13", "title", "description", "authors", "rank"]
    a = v3[v3_cols].copy().rename(columns={"rank": "v3b_rank"})
    b = v4a[v4_cols].copy().rename(columns={"rank": "v4a_rank"})

    merged = a.merge(
        b,
        on=["query_id", "isbn13"],
        how="outer",
        suffixes=("_v3b", "_v4a"),
        validate="one_to_one",
        indicator="merge_source",
    )

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        source = {"both": "both", "left_only": "v3b_only", "right_only": "v4a_only"}[row.merge_source]

        q3, q4 = safe_text(row.query_v3b), safe_text(row.query_v4a)
        s3, s4 = safe_text(row.stratum_v3b), safe_text(row.stratum_v4a)
        t3, t4 = safe_text(row.title_v3b), safe_text(row.title_v4a)
        d3, d4 = safe_text(row.description_v3b), safe_text(row.description_v4a)
        a3, a4 = safe_text(row.authors_v3b), safe_text(row.authors_v4a)

        if source == "both":
            require(q3 == q4, f"{row.query_id}/{row.isbn13}: shared query payload mismatch")
            require(s3 == s4, f"{row.query_id}/{row.isbn13}: shared stratum mismatch")
            require(t3 == t4, f"{row.query_id}/{row.isbn13}: shared title mismatch")
            require(d3 == d4, f"{row.query_id}/{row.isbn13}: shared description mismatch")
            require(a3 == a4, f"{row.query_id}/{row.isbn13}: shared authors mismatch")

        rows.append({
            "query_id": row.query_id,
            "query": q3 or q4,
            "stratum": s3 or s4,
            "isbn13": canonical_isbn(row.isbn13),
            "title": t3 or t4,
            "description": d3 or d4,
            "authors": a3 or a4,
            "pair_source": source,
            "in_v3b": source in {"both", "v3b_only"},
            "v3b_rank": "" if pd.isna(row.v3b_rank) else int(row.v3b_rank),
            "in_v4a": source in {"both", "v4a_only"},
            "v4a_rank": "" if pd.isna(row.v4a_rank) else int(row.v4a_rank),
        })

    union = pd.DataFrame(rows)
    require(not union.duplicated(["query_id", "isbn13"]).any(), "Union contains duplicate query-book pairs.")
    union = union.sort_values(["query_id", "isbn13"], kind="stable").reset_index(drop=True)
    return union


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-run", default=str(DEFAULT_V3_RUN))
    parser.add_argument("--v4a-run", default=str(DEFAULT_V4A_RUN))
    args = parser.parse_args()

    v3_run = Path(args.v3_run)
    v4a_run = Path(args.v4a_run)
    if not v3_run.is_absolute():
        v3_run = REPO_ROOT / v3_run
    if not v4a_run.is_absolute():
        v4a_run = REPO_ROOT / v4a_run

    print("Semantic relevance v4a - freeze v3b/v4a development displayed-pair union")
    print("------------------------------------------------------------------------")
    print("Judge execution: DISABLED")
    print("Human relevance labels loaded: NO")
    print("Release metrics: DISABLED")
    print()

    v3, v4a, v3_file, v4a_file, v4a_reranker_lock, v4a_retrieval_lock = validate_inputs(v3_run, v4a_run)
    print("PASS  frozen v3b/v4a recommendation and v4a lock hashes verified")
    print("PASS  v3b/v4a displayed structures verified: 24x10 each")
    print("PASS  H01-H24 query text/stratum agreement verified")

    union = build_union(v3, v4a)
    counts = union["pair_source"].value_counts().to_dict()
    query_counts = union.groupby("query_id").size().to_dict()

    require(all(10 <= n <= 20 for n in query_counts.values()), f"Unexpected per-query union sizes: {query_counts}")
    require(sum(counts.values()) == len(union), "Union source counts do not sum to row count.")
    require(int(counts.get("both", 0)) + int(counts.get("v3b_only", 0)) == 240, "v3b coverage mismatch in union")
    require(int(counts.get("both", 0)) + int(counts.get("v4a_only", 0)) == 240, "v4a coverage mismatch in union")

    union_file = v4a_run / "v3b_v4a_dev_display_union.csv"
    metadata_file = v4a_run / "v3b_v4a_dev_display_union_metadata.json"
    lock_file = v4a_run / "v3b_v4a_dev_display_union.lock.json"
    for path in [union_file, metadata_file, lock_file]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing union artifact: {path}")

    union.to_csv(union_file, index=False, encoding="utf-8")

    metadata = {
        "status": "V4A_DEV_DISPLAY_UNION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_query_set": "H01-H24",
        "baseline": "frozen_v3b",
        "candidate": "v4a",
        "v3b_recommendations_sha256": sha256(v3_file),
        "v4a_recommendations_sha256": sha256(v4a_file),
        "v4a_reranker_lock_sha256": sha256(v4a_reranker_lock),
        "v4a_retrieval_lock_sha256": sha256(v4a_retrieval_lock),
        "query_count": 24,
        "display_depth_per_system": 10,
        "union_rows": len(union),
        "pair_source_counts": {k: int(v) for k, v in counts.items()},
        "min_union_rows_per_query": int(min(query_counts.values())),
        "max_union_rows_per_query": int(max(query_counts.values())),
        "judge_labels_loaded": False,
        "human_relevance_labels_loaded": False,
        "judge_calls_performed": 0,
        "release_metrics_calculated": False,
        "outputs": {"v3b_v4a_dev_display_union.csv": sha256(union_file)},
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V4A_DEV_DISPLAY_UNION_FROZEN_BEFORE_JUDGING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_query_set": "H01-H24",
        "v3b_recommendations_sha256": sha256(v3_file),
        "v4a_recommendations_sha256": sha256(v4a_file),
        "v4a_reranker_lock_sha256": sha256(v4a_reranker_lock),
        "v4a_retrieval_lock_sha256": sha256(v4a_retrieval_lock),
        "display_union_sha256": sha256(union_file),
        "display_union_metadata_sha256": sha256(metadata_file),
        "union_rows": len(union),
        "pair_source_counts": {k: int(v) for k, v in counts.items()},
        "judge_execution_performed": False,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "next_stage": "prepare_reuse_and_novel_score_map_for_v4a_development_judging",
    }
    lock_file.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print(f"PASS  unique displayed-pair union built: {len(union)} rows")
    print(f"INFO  pair sources: {counts}")
    print(f"INFO  per-query union size: min={min(query_counts.values())} max={max(query_counts.values())}")
    print("PASS  displayed-pair union frozen before judging")
    print()
    print("V4A DEVELOPMENT DISPLAY UNION: PASS")
    print(f"Run directory: {v4a_run}")
    print(f"Display union SHA-256: {sha256(union_file)}")
    print(f"Display union lock SHA-256: {sha256(lock_file)}")
    print("Judge calls performed: 0")
    print("Release metrics calculated: NO")
    print("Next: prepare reuse/novel score map; judge only previously-unjudged displayed pairs.")


if __name__ == "__main__":
    main()
