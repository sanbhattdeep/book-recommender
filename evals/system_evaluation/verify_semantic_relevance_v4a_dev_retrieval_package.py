from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
BASE = EVALS / "system_evaluation"

FILES = [
    ("v4a JSON spec", BASE / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json", "6aba9c8354f201ad41044dd82d4476612f81ebd27a5ce4f8bf4f327a8b3281b9"),
    ("v4a Markdown spec", BASE / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.md", "5af31f1b986187b668b0d486d4209da198e88eb6ea03e3b12818ea8c04e33b90"),
    ("v4a spec lock", BASE / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.lock.json", "c9fee869cda6ccfd26c1ad0b2624cbcf693b9c896366b6baaa5c0b7b98371db2"),
    ("v4a implementation", BASE / "semantic_relevance_v4a_hybrid.py", "6f31504e29078fffbab86ddb86ef26b02aaa22082d3bfdb6944792bf689f6db7"),
    ("v4a implementation lock", BASE / "semantic_relevance_v4a_implementation.lock.json", "b67d305ca5c5da7df82997f43129ad596c7c79eb085571bb4a83b8638f016775"),
    ("H01-H24 query file", EVALS / "datasets/semantic_relevance_v3_holdout_queries.v0.1.0.csv", "75c06bb2c41155cee216339ab7c0421e214f7b3324fac5ec7414f3e2c5bb7b45"),
    ("frozen H01-H24 dense top50", EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval/holdout_top50_candidates.csv", "5587b0400b1fcabb86bbbfcbc6deee741e418cc277e2b9a18ab0efa29e8d4acf"),
    ("frozen dense retrieval lock", EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval/holdout_retrieval_lock.json", "4c0553065f94314ded8cd4b5d8edd90cef3a812c910c69c0a5907950a2b503af"),
    ("v4a dev retrieval collector", BASE / "collect_semantic_relevance_v4a_dev_retrieval.py", "088341cc95021eccad5bf8e708c9de3e7e5f7e11ed180d269e766b7b67967838"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    print("Semantic relevance v4a - development retrieval package preflight")
    print("---------------------------------------------------------------")
    print("Runtime catalog read:  DISABLED")
    print("BM25 catalog indexing: DISABLED")
    print("BM25 retrieval:        DISABLED")
    print("Reranker inference:    DISABLED")
    print("Judge execution:       DISABLED")
    print()

    for label, path, expected in FILES:
        require(path.exists(), f"Missing {label}: {path}")
        actual = sha256(path)
        require(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
        print(f"PASS  {label} SHA-256 verified")

    spec = json.loads((BASE / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json").read_text(encoding="utf-8-sig"))
    impl_lock = json.loads((BASE / "semantic_relevance_v4a_implementation.lock.json").read_text(encoding="utf-8-sig"))
    require(spec.get("status") == "FROZEN_BEFORE_V4A_IMPLEMENTATION", "Spec is not frozen.")
    require(impl_lock.get("status") == "V4A_IMPLEMENTATION_FROZEN_BEFORE_DEVELOPMENT_RETRIEVAL", "Implementation is not frozen.")
    require(spec["development_protocol"]["query_set"] == "H01-H24", "Development query-set contract changed.")
    require(spec["candidate_generation"]["dense_branch"]["candidate_depth"] == 50, "Dense depth changed.")
    require(spec["candidate_generation"]["lexical_branch"]["candidate_depth"] == 50, "BM25 depth changed.")
    require(spec["candidate_generation"]["union"]["truncate_before_reranker"] is False, "Union truncation contract changed.")

    print("PASS  frozen v4a development-retrieval contract verified")
    print()
    print("V4A DEVELOPMENT RETRIEVAL PACKAGE PREFLIGHT: PASS")
    print("Runtime catalog read:  NO")
    print("BM25 index built:      NO")
    print("Development retrieval: NO")
    print("Reranker inference:    NO")
    print("Judge calls:           0")
    print("Next: execute the frozen v4a development BM25/hybrid retrieval collector.")


if __name__ == "__main__":
    main()
