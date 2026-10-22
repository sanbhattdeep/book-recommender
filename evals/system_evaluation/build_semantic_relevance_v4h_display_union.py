from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SYS = REPO_ROOT / "evals/system_evaluation"
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"

RERANKING_CONTRACT = SYS / "semantic_relevance_v4h_reranking_contract.v0.1.0.json"
RERANKING_LOCK = RUN / "v4h_reranking.lock.json"
V3B_RECS = RUN / "v4h_v3b_recommendations.csv"
V4B_RECS = RUN / "v4h_v4b_recommendations.csv"
RETRIEVAL_LOCK = RUN / "v4h_retrieval.lock.json"
AUTHORING_LOCK = SYS / "semantic_relevance_v4_holdout_authoring.lock.json"
DISPLAY_UNION_CONTRACT = SYS / "semantic_relevance_v4h_display_union_contract.v0.1.0.json"

OUT_UNION = RUN / "v4h_v3b_v4b_display_union.csv"
OUT_METADATA = RUN / "v4h_v3b_v4b_display_union_metadata.json"
OUT_LOCK = RUN / "v4h_v3b_v4b_display_union.lock.json"


EXPECTED = {
    "reranking_contract": "0cf17eda345d7abb906a8f5f4730fb5d356d1514401d603271ab7cf4707df471",
    "reranking_lock": "6d3a7014f5995d978a90df627c720c0b81aa2e8b28e8bba99cfa2c64d27811b5",
    "v3b_recommendations": "2f1aec34740f35127822a634e00f772db716e35e2d8aeaeaf729d54ecc95b505",
    "v4b_recommendations": "27c8fb4502909f0a3f89d6b6dfbcec55ac82ae489eab800d51334f312d16fec2",
    "retrieval_lock": "e2902e1cc537e74a39b6a0c1d421c8db0d00637596c89a7889841ae242089884",
    "authoring_lock": "14cd3a8a9f6bf22666dd4c74a150e85f82a53789ccaa5bd2870d606b66ef83e7"
}
EXPECTED_CONTRACT_SHA256 = "ec4c8a2d196a1f619111e311cc9126b6ca7906173edec9bd215cc34d922afd92"
EXPECTED_QUERY_IDS = [f"V4H{i:02d}" for i in range(1,25)]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))

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
    missing = required - set(frame.columns)
    req(not missing, f"{system} recommendations missing columns: {sorted(missing)}")
    if "authors" not in frame.columns:
        frame["authors"] = ""

    frame = frame.copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
    frame["isbn13"] = frame["isbn13"].map(canonical_isbn)

    req(len(frame) == 240, f"{system} expected 240 rows, got {len(frame)}")
    req(not frame.duplicated(["query_id","isbn13"]).any(), f"{system} duplicate query-book pair")
    req(sorted(frame["query_id"].unique().tolist()) == EXPECTED_QUERY_IDS,
        f"{system} query IDs changed")
    counts = frame.groupby("query_id").size()
    req(len(counts) == 24 and (counts == 10).all(), f"{system} is not exactly 24x10")
    for qid, group in frame.groupby("query_id", sort=True):
        req(sorted(group["rank"].tolist()) == list(range(1,11)),
            f"{system} {qid} ranks are not exactly 1..10")
        req(group["query"].nunique() == 1, f"{system} {qid} query text not unique")
        req(group["stratum"].nunique() == 1, f"{system} {qid} stratum not unique")
    return frame

def verify_frozen_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path, expected, label in [
        (RERANKING_CONTRACT, EXPECTED["reranking_contract"], "V4H reranking contract"),
        (RERANKING_LOCK, EXPECTED["reranking_lock"], "V4H reranking lock"),
        (V3B_RECS, EXPECTED["v3b_recommendations"], "V3B recommendations"),
        (V4B_RECS, EXPECTED["v4b_recommendations"], "V4B recommendations"),
        (RETRIEVAL_LOCK, EXPECTED["retrieval_lock"], "V4H retrieval lock"),
        (AUTHORING_LOCK, EXPECTED["authoring_lock"], "V4H authoring lock"),
        (DISPLAY_UNION_CONTRACT, EXPECTED_CONTRACT_SHA256, "V4H display-union contract"),
    ]:
        req(path.exists(), f"Missing {label}: {path}")
        actual = sha256(path)
        req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
        print(f"PASS  {label} raw SHA-256 verified")

    rlock = load_json(RERANKING_LOCK)
    req(rlock.get("status") == "V4H_V3B_V4B_RERANKING_FROZEN_BEFORE_DISPLAY_UNION_OR_JUDGING",
        "Unexpected V4H reranking-lock status")
    req(rlock.get("query_set") == "V4H01-V4H24", "Reranking lock query set changed")
    req(rlock.get("baseline") == "v3b" and rlock.get("candidate") == "v4b",
        "Reranking lock system identities changed")
    req(rlock.get("v3b_recommendations_sha256") == EXPECTED["v3b_recommendations"],
        "Reranking lock V3B recommendation hash mismatch")
    req(rlock.get("v4b_recommendations_sha256") == EXPECTED["v4b_recommendations"],
        "Reranking lock V4B recommendation hash mismatch")
    req(rlock.get("judge_calls_performed") == 0, "Reranking lock says judge already ran")
    req(rlock.get("human_relevance_labels_used") is False, "Reranking lock says human labels were used")
    req(rlock.get("release_metrics_calculated") is False, "Reranking lock says release metrics already calculated")
    req(rlock.get("next_stage") == "FREEZE_UNIQUE_V3B_V4B_DISPLAYED_PAIR_UNION_BEFORE_JUDGING",
        "Reranking lock next-stage contract changed")

    contract = load_json(DISPLAY_UNION_CONTRACT)
    req(contract.get("status") == "FROZEN_BY_PACKAGE_BEFORE_V4H_DISPLAY_UNION_BUILD",
        "Unexpected display-union contract status")
    reuse = contract.get("score_reuse_policy", {})
    req(reuse.get("historical_query_book_score_reuse_allowed") is False,
        "Display-union contract unexpectedly allows historical score reuse")
    req(reuse.get("all_unique_displayed_pairs_require_frozen_judge_scoring") is True,
        "Display-union contract does not require scoring the full unique union")

    v3 = read_recs(V3B_RECS, "V3B")
    v4 = read_recs(V4B_RECS, "V4B")
    q3 = v3[["query_id","query","stratum"]].drop_duplicates().sort_values("query_id").reset_index(drop=True)
    q4 = v4[["query_id","query","stratum"]].drop_duplicates().sort_values("query_id").reset_index(drop=True)
    req(q3.equals(q4), "V3B/V4B query text or stratum mismatch")
    print("PASS  frozen V3B/V4B displayed structures verified: exact 24x10 each")
    print("PASS  V4H query text/stratum agreement verified")
    return v3, v4


def main() -> None:
    print("Semantic relevance V4H - freeze V3B/V4B displayed-pair union")
    print("------------------------------------------------------------")
    print("Score reuse: DISABLED")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release metrics/decision: NOT CALCULATED")
    print()

    for out in [OUT_UNION, OUT_METADATA, OUT_LOCK]:
        req(not out.exists(), f"Refusing to overwrite existing display-union artifact: {out}")

    v3, v4 = verify_frozen_sources()

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

    rows = []
    for row in merged.itertuples(index=False):
        source = {"both":"both","left_only":"v3b_only","right_only":"v4b_only"}[row.merge_source]

        q3, q4 = safe_text(row.query_v3b), safe_text(row.query_v4b)
        s3, s4 = safe_text(row.stratum_v3b), safe_text(row.stratum_v4b)
        t3, t4 = safe_text(row.title_v3b), safe_text(row.title_v4b)
        d3, d4 = safe_text(row.description_v3b), safe_text(row.description_v4b)
        a3, a4 = safe_text(row.authors_v3b), safe_text(row.authors_v4b)

        if source == "both":
            req(q3 == q4, f"{row.query_id}/{row.isbn13} shared query mismatch")
            req(s3 == s4, f"{row.query_id}/{row.isbn13} shared stratum mismatch")
            req(t3 == t4, f"{row.query_id}/{row.isbn13} shared title mismatch")
            req(d3 == d4, f"{row.query_id}/{row.isbn13} shared description mismatch")
            req(a3 == a4, f"{row.query_id}/{row.isbn13} shared authors mismatch")

        rows.append({
            "query_id": str(row.query_id),
            "query": q3 or q4,
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

    req(len(per_query) == 24, "Display union does not contain 24 queries")
    req(per_query.between(10,20).all(), f"Unexpected per-query union sizes: {per_query.to_dict()}")
    req(int(counts.get("both",0)) + int(counts.get("v3b_only",0)) == 240, "V3B union coverage mismatch")
    req(int(counts.get("both",0)) + int(counts.get("v4b_only",0)) == 240, "V4B union coverage mismatch")
    req(len(union) == 480 - int(counts.get("both",0)), "Union cardinality identity failed")

    OUT_UNION.parent.mkdir(parents=True, exist_ok=True)
    union.to_csv(OUT_UNION, index=False, encoding="utf-8")

    metadata = {
        "status":"V4H_FINAL_HOLDOUT_DISPLAY_UNION_COMPLETE",
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "query_set":"V4H01-V4H24",
        "query_count":24,
        "baseline":"v3b",
        "candidate":"v4b",
        "display_depth_per_system":10,
        "v3b_recommendations_sha256":EXPECTED["v3b_recommendations"],
        "v4b_recommendations_sha256":EXPECTED["v4b_recommendations"],
        "reranking_lock_sha256":EXPECTED["reranking_lock"],
        "display_union_contract_sha256":EXPECTED_CONTRACT_SHA256,
        "union_rows":int(len(union)),
        "pair_source_counts":{str(k):int(v) for k,v in counts.items()},
        "min_union_rows_per_query":int(per_query.min()),
        "max_union_rows_per_query":int(per_query.max()),
        "historical_score_reuse_allowed":False,
        "all_unique_pairs_require_judging":True,
        "score_reuse_performed":False,
        "judge_calls_performed":0,
        "human_relevance_labels_used":False,
        "release_metrics_calculated":False,
        "release_decision_produced":False,
        "outputs":{OUT_UNION.name:sha256(OUT_UNION)},
    }
    OUT_METADATA.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")

    lock = {
        "status":"V4H_DISPLAY_UNION_FROZEN_BEFORE_JUDGING",
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "query_set":"V4H01-V4H24",
        "query_count":24,
        "baseline":"v3b",
        "candidate":"v4b",
        "v3b_recommendations_sha256":EXPECTED["v3b_recommendations"],
        "v4b_recommendations_sha256":EXPECTED["v4b_recommendations"],
        "reranking_lock_sha256":EXPECTED["reranking_lock"],
        "display_union_contract_sha256":EXPECTED_CONTRACT_SHA256,
        "display_union_sha256":sha256(OUT_UNION),
        "display_union_metadata_sha256":sha256(OUT_METADATA),
        "union_rows":int(len(union)),
        "pair_source_counts":{str(k):int(v) for k,v in counts.items()},
        "historical_score_reuse_allowed":False,
        "all_unique_pairs_require_judging":True,
        "score_reuse_performed":False,
        "judge_execution_performed":False,
        "human_relevance_labels_used":False,
        "release_metrics_calculated":False,
        "release_decision_produced":False,
        "next_stage":"JUDGE_ALL_UNIQUE_FROZEN_V3B_V4B_DISPLAYED_PAIRS",
    }
    OUT_LOCK.write_text(json.dumps(lock,indent=2)+"\n",encoding="utf-8")

    print(f"PASS  unique V3B/V4B displayed-pair union built: {len(union)} rows")
    print(f"INFO  pair sources: {counts}")
    print(f"INFO  per-query union size: min={int(per_query.min())} max={int(per_query.max())}")
    print("PASS  all unique V4H displayed pairs designated for fresh frozen-judge scoring")
    print("PASS  displayed-pair union frozen before judging")
    print()
    print("V4H DISPLAY UNION: PASS")
    print(f"Display union SHA-256: {sha256(OUT_UNION)}")
    print(f"Display union lock SHA-256: {sha256(OUT_LOCK)}")
    print("Historical score reuse performed: NO")
    print("Judge calls performed: 0")
    print("Release metrics calculated: NO")
    print("Next: prepare frozen-judge scoring for every unique displayed pair.")

if __name__ == "__main__":
    main()
