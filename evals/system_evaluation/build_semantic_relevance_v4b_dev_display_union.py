from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
SYS = EVALS / "system_evaluation"
V3_RUN = EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"
V4_RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

V3_RECS = V3_RUN / "v3_holdout_recommendations.csv"
V4B_RECS = V4_RUN / "v4b_dev_recommendations.csv"
V4B_RANKING_LOCK = V4_RUN / "v4b_dev_ranking.lock.json"
V4B_DESIGN_LOCK = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.lock.json"
V4B_IMPLEMENTATION_LOCK = SYS / "semantic_relevance_v4b_implementation.lock.json"

OUT_UNION = V4_RUN / "v3b_v4b_dev_display_union.csv"
OUT_METADATA = V4_RUN / "v3b_v4b_dev_display_union_metadata.json"
OUT_LOCK = V4_RUN / "v3b_v4b_dev_display_union.lock.json"

EXPECTED = {
    "v3b_recommendations": "e5e136a6d611a527635d202ec5edb928e9de77a8a0db46fa0fcab295c69a4ba2",
    "v4b_recommendations": "d8e13df73e8c2df4a28c0d1a64ed8ebb2c6cf7a4356d2407d8a96169b8fe387d",
    "v4b_ranking_lock": "1ad4fe462d9f41bbba349016502117fe02fce27a72615dbba656368dd2ba7230",
    "v4b_design_lock": "e60ca1032f7fa733f3e7a51ec87a229d7454645aa6109b5e557588af50383c6d",
    "v4b_implementation_lock": "f30a61e7ca7a5e07766c3f2baa3d4a65abcedb48cdedf23ee6d5c0e05b3171b0"
}
EXPECTED_QUERY_IDS = [f"H{i:02d}" for i in range(1,25)]
KNOWN_QUARANTINE_QUERY = "H14"
KNOWN_QUARANTINE_ISBN = "9781582972947"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def req(condition: bool, message: str) -> None:
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

def read_recs(path: Path, system: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"query_id","query","stratum","rank","isbn13","title","description"}
    req(not (required - set(frame.columns)), f"{system} missing columns: {sorted(required - set(frame.columns))}")
    if "authors" not in frame.columns:
        frame["authors"] = ""
    frame = frame.copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
    frame["isbn13"] = frame["isbn13"].map(canonical_isbn)
    req(len(frame) == 240, f"{system} expected 240 rows")
    req(not frame.duplicated(["query_id","isbn13"]).any(), f"{system} duplicate pair")
    req(sorted(frame["query_id"].unique().tolist()) == EXPECTED_QUERY_IDS, f"{system} query IDs changed")
    req((frame.groupby("query_id").size() == 10).all(), f"{system} not 10 recommendations/query")
    return frame

def main() -> None:
    print("Semantic relevance V4B - freeze V3B/V4B displayed-pair union")
    print("------------------------------------------------------------")
    print("Score reuse: DISABLED")
    print("Judge execution: DISABLED")
    print("Human relevance labels loaded: NO")
    print("Development selection: NOT CALCULATED")
    print("Release analysis: DISABLED")
    print()

    for out in [OUT_UNION, OUT_METADATA, OUT_LOCK]:
        req(not out.exists(), f"Refusing to overwrite existing union artifact: {out}")

    for path, expected, label in [
        (V3_RECS, EXPECTED["v3b_recommendations"], "V3B recommendations"),
        (V4B_RECS, EXPECTED["v4b_recommendations"], "V4B recommendations"),
        (V4B_RANKING_LOCK, EXPECTED["v4b_ranking_lock"], "V4B ranking lock"),
        (V4B_DESIGN_LOCK, EXPECTED["v4b_design_lock"], "V4B design lock"),
        (V4B_IMPLEMENTATION_LOCK, EXPECTED["v4b_implementation_lock"], "V4B implementation lock"),
    ]:
        req(sha256(path) == expected, f"{label} hash mismatch")

    v3 = read_recs(V3_RECS, "V3B")
    v4 = read_recs(V4B_RECS, "V4B")

    q3 = v3[["query_id","query","stratum"]].drop_duplicates().sort_values("query_id").reset_index(drop=True)
    q4 = v4[["query_id","query","stratum"]].drop_duplicates().sort_values("query_id").reset_index(drop=True)
    req(q3.equals(q4), "V3B/V4B query payload mismatch")

    cols = ["query_id","query","stratum","isbn13","title","description","authors","rank"]
    a = v3[cols].copy().rename(columns={"rank":"v3b_rank"})
    b = v4[cols].copy().rename(columns={"rank":"v4b_rank"})

    merged = a.merge(
        b,
        on=["query_id","isbn13"],
        how="outer",
        suffixes=("_v3b","_v4b"),
        validate="one_to_one",
        indicator="merge_source",
    )

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        source = {"both":"both","left_only":"v3b_only","right_only":"v4b_only"}[row.merge_source]

        q3t, q4t = safe_text(row.query_v3b), safe_text(row.query_v4b)
        s3, s4 = safe_text(row.stratum_v3b), safe_text(row.stratum_v4b)
        t3, t4 = safe_text(row.title_v3b), safe_text(row.title_v4b)
        d3, d4 = safe_text(row.description_v3b), safe_text(row.description_v4b)
        a3, a4 = safe_text(row.authors_v3b), safe_text(row.authors_v4b)

        if source == "both":
            req(q3t == q4t, f"{row.query_id}/{row.isbn13} shared query mismatch")
            req(s3 == s4, f"{row.query_id}/{row.isbn13} shared stratum mismatch")
            req(t3 == t4, f"{row.query_id}/{row.isbn13} shared title mismatch")
            req(d3 == d4, f"{row.query_id}/{row.isbn13} shared description mismatch")
            req(a3 == a4, f"{row.query_id}/{row.isbn13} shared authors mismatch")

        rows.append({
            "query_id": str(row.query_id),
            "query": q3t or q4t,
            "stratum": s3 or s4,
            "isbn13": canonical_isbn(row.isbn13),
            "title": t3 or t4,
            "description": d3 or d4,
            "authors": a3 or a4,
            "pair_source": source,
            "in_v3b": source in {"both","v3b_only"},
            "v3b_rank": "" if pd.isna(row.v3b_rank) else int(row.v3b_rank),
            "in_v4b": source in {"both","v4b_only"},
            "v4b_rank": "" if pd.isna(row.v4b_rank) else int(row.v4b_rank),
        })

    union = pd.DataFrame(rows)
    req(not union.duplicated(["query_id","isbn13"]).any(), "Display union contains duplicate pairs")
    union = union.sort_values(["query_id","isbn13"], kind="stable").reset_index(drop=True)

    counts = union["pair_source"].value_counts().to_dict()
    per_query = union.groupby("query_id").size()
    req(len(per_query) == 24, "Display union query count changed")
    req(per_query.between(10,20).all(), f"Unexpected per-query union sizes: {per_query.to_dict()}")
    req(int(counts.get("both",0)) + int(counts.get("v3b_only",0)) == 240, "V3B coverage mismatch")
    req(int(counts.get("both",0)) + int(counts.get("v4b_only",0)) == 240, "V4B coverage mismatch")

    quarantine_present = bool(
        (
            union["query_id"].eq(KNOWN_QUARANTINE_QUERY)
            & union["isbn13"].eq(KNOWN_QUARANTINE_ISBN)
        ).any()
    )
    req(not quarantine_present, "Unresolved H14 quarantine pair entered displayed comparison union")

    OUT_UNION.parent.mkdir(parents=True, exist_ok=True)
    union.to_csv(OUT_UNION, index=False, encoding="utf-8")

    metadata = {
        "status": "V4B_DEV_DISPLAY_UNION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_query_set": "H01-H24",
        "baseline": "frozen_v3b",
        "candidate": "v4b",
        "v3b_recommendations_sha256": EXPECTED["v3b_recommendations"],
        "v4b_recommendations_sha256": EXPECTED["v4b_recommendations"],
        "v4b_ranking_lock_sha256": EXPECTED["v4b_ranking_lock"],
        "query_count": 24,
        "display_depth_per_system": 10,
        "union_rows": int(len(union)),
        "pair_source_counts": {str(k): int(v) for k,v in counts.items()},
        "min_union_rows_per_query": int(per_query.min()),
        "max_union_rows_per_query": int(per_query.max()),
        "known_h14_quarantine_pair_present": False,
        "score_reuse_performed": False,
        "judge_labels_loaded": False,
        "human_relevance_labels_loaded": False,
        "judge_calls_performed": 0,
        "development_selection_calculated": False,
        "release_metrics_calculated": False,
        "outputs": {OUT_UNION.name: sha256(OUT_UNION)},
    }
    OUT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V4B_DEV_DISPLAY_UNION_FROZEN_BEFORE_SCORE_REUSE_OR_JUDGING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_query_set": "H01-H24",
        "v3b_recommendations_sha256": EXPECTED["v3b_recommendations"],
        "v4b_recommendations_sha256": EXPECTED["v4b_recommendations"],
        "v4b_ranking_lock_sha256": EXPECTED["v4b_ranking_lock"],
        "display_union_sha256": sha256(OUT_UNION),
        "display_union_metadata_sha256": sha256(OUT_METADATA),
        "union_rows": int(len(union)),
        "pair_source_counts": {str(k): int(v) for k,v in counts.items()},
        "known_h14_quarantine_pair_present": False,
        "score_reuse_performed": False,
        "judge_execution_performed": False,
        "human_relevance_labels_used": False,
        "development_selection_calculated": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "next_stage": "PREPARE_V4B_DEVELOPMENT_REUSE_AND_NOVEL_SCORE_MAP",
    }
    OUT_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print(f"PASS  unique V3B/V4B displayed-pair union built: {len(union)} rows")
    print(f"INFO  pair sources: {counts}")
    print(f"INFO  per-query union size: min={int(per_query.min())} max={int(per_query.max())}")
    print("PASS  unresolved H14 quarantine pair absent from frozen displayed union")
    print("PASS  displayed-pair union frozen before score reuse or judging")
    print()
    print("V4B DEVELOPMENT DISPLAY UNION: PASS")
    print(f"Display union SHA-256: {sha256(OUT_UNION)}")
    print(f"Display union lock SHA-256: {sha256(OUT_LOCK)}")
    print("Score reuse performed: NO")
    print("Judge calls performed: 0")
    print("Development selection calculated: NO")
    print("Release analysis: NO")
    print("Next: prepare score reuse/novel inventory; judge only previously-unjudged displayed pairs.")

if __name__ == "__main__":
    main()
