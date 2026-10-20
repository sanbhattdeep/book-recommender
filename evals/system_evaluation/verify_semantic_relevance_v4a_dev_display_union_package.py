from __future__ import annotations

import hashlib
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
V3_RUN = REPO_ROOT / "evals/runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"
V4A_RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
BUILDER = REPO_ROOT / "evals/system_evaluation/build_semantic_relevance_v4a_dev_display_union.py"

EXPECTED = {
    "v3_recommendations": "e5e136a6d611a527635d202ec5edb928e9de77a8a0db46fa0fcab295c69a4ba2",
    "v4a_recommendations": "5328ab30132b4d117bd2bacd9c54cb6ed460968c0bdaf10b2720c2f9256481b6",
    "v4a_reranker_lock": "97a8cc18cf2e8c2cf5abec679f41de81546a945b6ae9b36ab06abd97d2508a22",
    "v4a_retrieval_lock": "7587aa140c1cb26f8ac5addaf6873f3bc7ec05b6ec34775a6cbfe98d7ffa40d1",
    "builder": "ae762213bc6cf5535452df5c86fec2ff5f173c0401c45a9f29390a52f5f53194",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_24x10(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"query_id", "query", "stratum", "rank", "isbn13", "title", "description"}
    require(not (required - set(frame.columns)), f"{label}: missing required columns")
    require(len(frame) == 240, f"{label}: expected 240 rows")
    require(not frame.duplicated(["query_id", "isbn13"]).any(), f"{label}: duplicate pairs")
    counts = frame.groupby("query_id").size()
    require(len(counts) == 24 and (counts == 10).all(), f"{label}: not 24x10")
    return frame


def main() -> None:
    print("Semantic relevance v4a - development displayed-union package preflight")
    print("--------------------------------------------------------------------")
    print("Union creation:     DISABLED")
    print("Judge execution:    DISABLED")
    print("Human labels:       DISABLED")
    print("Release analysis:   DISABLED")
    print()

    files = {
        "v3_recommendations": V3_RUN / "v3_holdout_recommendations.csv",
        "v4a_recommendations": V4A_RUN / "v4a_dev_recommendations.csv",
        "v4a_reranker_lock": V4A_RUN / "v4a_dev_reranker.lock.json",
        "v4a_retrieval_lock": V4A_RUN / "v4a_dev_retrieval.lock.json",
        "builder": BUILDER,
    }
    for name, path in files.items():
        require(path.exists(), f"Missing {name}: {path}")
        actual = sha256(path)
        require(actual == EXPECTED[name], f"{name} hash mismatch: expected={EXPECTED[name]} actual={actual}")
        print(f"PASS  {name} SHA-256 verified")

    v3 = verify_24x10(files["v3_recommendations"], "v3b")
    v4a = verify_24x10(files["v4a_recommendations"], "v4a")
    q3 = v3[["query_id", "query", "stratum"]].drop_duplicates().sort_values("query_id").reset_index(drop=True)
    q4 = v4a[["query_id", "query", "stratum"]].drop_duplicates().sort_values("query_id").reset_index(drop=True)
    require(q3.equals(q4), "v3b/v4a query text or stratum mismatch")

    print("PASS  v3b/v4a displayed structures verified: 24x10 each")
    print("PASS  H01-H24 query text/stratum agreement verified")
    print()
    print("V4A DEVELOPMENT DISPLAY UNION PACKAGE PREFLIGHT: PASS")
    print("Union created:       NO")
    print("Judge calls:         0")
    print("Human labels used:   NO")
    print("Release analysis:    NO")
    print("Next: build and freeze the unique v3b/v4a displayed-pair union.")


if __name__ == "__main__":
    main()
