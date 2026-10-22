from __future__ import annotations

import hashlib
import json
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

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def canonical_isbn(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text:
        raise ValueError("ISBN is empty.")
    return text

def read_recs(path: Path, system: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"query_id","query","stratum","rank","isbn13","title","description"}
    req(not (required - set(frame.columns)), f"{system} recommendations missing columns: {sorted(required - set(frame.columns))}")
    if "authors" not in frame.columns:
        frame["authors"] = ""

    frame = frame.copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
    frame["isbn13"] = frame["isbn13"].map(canonical_isbn)

    req(len(frame) == 240, f"{system} expected 240 rows, got {len(frame)}")
    req(not frame.duplicated(["query_id","isbn13"]).any(), f"{system} duplicate query-book pair")
    req(sorted(frame["query_id"].unique().tolist()) == EXPECTED_QUERY_IDS, f"{system} query IDs changed")
    counts = frame.groupby("query_id").size()
    req(len(counts) == 24 and (counts == 10).all(), f"{system} is not exactly 24x10")
    for qid, group in frame.groupby("query_id", sort=True):
        req(sorted(group["rank"].tolist()) == list(range(1,11)), f"{system} {qid} ranks are not 1..10")
        req(group["query"].nunique() == 1, f"{system} {qid} query text is not unique")
        req(group["stratum"].nunique() == 1, f"{system} {qid} stratum is not unique")
    return frame

def main() -> None:
    print("Semantic relevance V4B - displayed-pair union preflight")
    print("------------------------------------------------------")
    print("Union generation: DISABLED")
    print("Score reuse: DISABLED")
    print("Judge execution: DISABLED")
    print("Relevance labels loaded: NO")
    print("Development selection: NOT CALCULATED")
    print("Release analysis: DISABLED")
    print()

    for out in [OUT_UNION, OUT_METADATA, OUT_LOCK]:
        req(not out.exists(), f"Refusing to overwrite existing V3B/V4B display-union artifact: {out}")

    for path, expected, label in [
        (V3_RECS, EXPECTED["v3b_recommendations"], "frozen V3B recommendations"),
        (V4B_RECS, EXPECTED["v4b_recommendations"], "frozen V4B recommendations"),
        (V4B_RANKING_LOCK, EXPECTED["v4b_ranking_lock"], "V4B ranking lock"),
        (V4B_DESIGN_LOCK, EXPECTED["v4b_design_lock"], "V4B design lock"),
        (V4B_IMPLEMENTATION_LOCK, EXPECTED["v4b_implementation_lock"], "V4B implementation lock"),
    ]:
        req(path.exists(), f"Missing {label}: {path}")
        actual = sha256(path)
        req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
        print(f"PASS  {label} raw SHA-256 verified")

    lock = load(V4B_RANKING_LOCK)
    req(lock.get("status") == "V4B_DEVELOPMENT_RANKING_FROZEN_BEFORE_DISPLAY_UNION_OR_JUDGING", "Unexpected V4B ranking-lock status")
    req(lock.get("recommendation_rows") == 240, "V4B ranking lock recommendation count changed")
    req(lock.get("known_h14_quarantine_candidate_in_v4b_top10") is False, "V4B ranking lock says quarantined H14 pair entered top10")
    req(lock.get("judge_calls_performed") == 0, "V4B ranking lock judge-call boundary changed")
    req(lock.get("development_selection_calculated") is False, "V4B ranking lock says selection calculated")

    v3 = read_recs(V3_RECS, "V3B")
    v4 = read_recs(V4B_RECS, "V4B")
    v3q = v3[["query_id","query","stratum"]].drop_duplicates().sort_values("query_id").reset_index(drop=True)
    v4q = v4[["query_id","query","stratum"]].drop_duplicates().sort_values("query_id").reset_index(drop=True)
    req(v3q.equals(v4q), "V3B/V4B query text or stratum mismatch")

    quarantine_in_v3 = (
        v3["query_id"].eq(KNOWN_QUARANTINE_QUERY)
        & v3["isbn13"].eq(KNOWN_QUARANTINE_ISBN)
    ).any()
    quarantine_in_v4 = (
        v4["query_id"].eq(KNOWN_QUARANTINE_QUERY)
        & v4["isbn13"].eq(KNOWN_QUARANTINE_ISBN)
    ).any()
    req(not quarantine_in_v3 and not quarantine_in_v4, "Known H14 quarantined pair is present in a displayed system output")

    print("PASS  V3B/V4B displayed structures verified: 24x10 each")
    print("PASS  H01-H24 query text/stratum agreement verified")
    print("PASS  unresolved H14 quarantined pair absent from both displayed systems")
    print()
    print("V4B DISPLAY UNION PREFLIGHT: PASS")
    print("V3B rows: 240")
    print("V4B rows: 240")
    print("Relevance labels loaded: NO")
    print("Score reuse performed: NO")
    print("Judge calls: 0")
    print("Development selection calculated: NO")
    print("Release analysis: NO")
    print("Next: build and freeze the unique V3B/V4B displayed-pair union.")

if __name__ == "__main__":
    main()
