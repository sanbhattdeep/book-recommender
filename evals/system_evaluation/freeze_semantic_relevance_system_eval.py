from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

CONTRACT_FILE = (
    EVALS / "system_evaluation"
    / "semantic_relevance_system_eval.v1.0.0.json"
)
QUERY_FILE = (
    EVALS / "datasets"
    / "semantic_relevance_system_eval_queries.v1.0.0.csv"
)
FACET_SPEC_FILE = (
    EVALS / "facets" / "semantic_relevance"
    / "semantic_relevance_query_facets.v0.10.1.json"
)
JUDGE_FILE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS / "semantic_relevance_facet_scoring.py"
CONFIG_FILE = (
    EVALS / "judge_configs"
    / "semantic_relevance_judge.v0.29.0-r5.json"
)
RECOMMENDER_FILE = REPO_ROOT / "gradio-dashboard.py"

LOCK_FILE = (
    EVALS / "system_evaluation"
    / "semantic_relevance_system_eval_input_lock.v1.0.0.json"
)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: expected={expected} actual={actual}"
        )
    print(f"PASS  pinned {label} hash")

def main() -> None:
    contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8-sig"))
    pinned = contract["measurement_instrument"]["pinned_hashes"]

    require_hash(JUDGE_FILE, pinned["judge"], "r5 judge")
    require_hash(SCORING_FILE, pinned["scoring"], "r5 scoring")
    require_hash(FACET_SPEC_FILE, pinned["facet_spec"], "facet spec 0.10.1")
    require_hash(CONFIG_FILE, pinned["judge_config"], "r5 judge config")

    if not RECOMMENDER_FILE.exists():
        raise FileNotFoundError(RECOMMENDER_FILE)
    recommender_text = RECOMMENDER_FILE.read_text(encoding="utf-8")
    required_fragments = [
        "def retrieve_semantic_recommedations(",
        "initial_top_k: int = 50",
        "final_top_k: int = 16",
        "db_books.similarity_search(query, k=initial_top_k)",
    ]
    for fragment in required_fragments:
        if fragment not in recommender_text:
            raise ValueError(
                "Recommender interface changed; missing expected fragment: "
                f"{fragment!r}"
            )
    print("PASS  expected semantic recommender interface present")

    queries = pd.read_csv(QUERY_FILE, dtype=str, encoding="utf-8")
    if len(queries) != 12 or queries["query_id"].duplicated().any():
        raise ValueError("Expected exactly 12 unique system-eval queries.")

    facet_payload = json.loads(
        FACET_SPEC_FILE.read_text(encoding="utf-8-sig")
    )
    expected = {
        str(q["query_id"]): str(q["query"])
        for q in facet_payload["queries"]
    }
    actual = dict(zip(queries["query_id"], queries["query"]))
    if actual != expected:
        raise ValueError(
            "System-eval queries no longer exactly match frozen facet spec."
        )
    print("PASS  12 benchmark queries exactly match frozen facet spec")

    if contract["measurement_instrument"]["human_scores_used_in_system_eval"]:
        raise ValueError("System evaluation must not use human relevance scores.")
    if contract["statistical_uncertainty"]["resampling_unit"] != "query":
        raise ValueError("Bootstrap resampling unit must remain query.")
    if len(contract["release_gates"]) != 8:
        raise ValueError("Expected exactly 8 preregistered release gates.")

    lock = {
        "evaluation_id": contract["evaluation_id"],
        "evaluation_version": contract["version"],
        "status": "FROZEN_BEFORE_SYSTEM_EVAL_EXECUTION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_sha(),
        "recommender_file": "gradio-dashboard.py",
        "recommender_sha256": sha256(RECOMMENDER_FILE),
        "query_file": str(QUERY_FILE.relative_to(REPO_ROOT)),
        "query_file_sha256": sha256(QUERY_FILE),
        "contract_file": str(CONTRACT_FILE.relative_to(REPO_ROOT)),
        "contract_file_sha256": sha256(CONTRACT_FILE),
        "judge_sha256": sha256(JUDGE_FILE),
        "scoring_sha256": sha256(SCORING_FILE),
        "facet_spec_sha256": sha256(FACET_SPEC_FILE),
        "judge_config_sha256": sha256(CONFIG_FILE),
        "query_count": 12,
        "recommendations_per_query": 10,
        "expected_query_book_pairs": 120,
        "human_scores_used": False,
        "recommender_calls_performed_during_freeze": 0,
        "judge_calls_performed_during_freeze": 0,
        "bootstrap": contract["statistical_uncertainty"],
        "release_gates": contract["release_gates"],
    }
    LOCK_FILE.write_text(
        json.dumps(lock, indent=2) + "\n",
        encoding="utf-8",
    )

    print("PASS  8 system release gates preregistered")
    print("PASS  query-level bootstrap CI method preregistered")
    print("PASS  human relevance scores excluded from system evaluation")
    print(f"Frozen recommender SHA-256: {lock['recommender_sha256']}")
    print(f"Frozen git commit: {lock['git_commit_sha']}")
    print(f"Input lock: {LOCK_FILE}")
    print("SYSTEM EVALUATION CONTRACT FREEZE: PASS")
    print("No recommender calls performed.")
    print("No judge calls performed.")

if __name__ == "__main__":
    main()
