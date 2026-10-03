from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

CONTRACT_FILE = (
    EVALS / "system_evaluation"
    / "semantic_relevance_system_eval.v1.0.0.json"
)
LOCK_FILE = (
    EVALS / "system_evaluation"
    / "semantic_relevance_system_eval_input_lock.v1.0.0.json"
)
QUERY_FILE = (
    EVALS / "datasets"
    / "semantic_relevance_system_eval_queries.v1.0.0.csv"
)
RECOMMENDER_FILE = REPO_ROOT / "gradio-dashboard.py"
RUN_ROOT = (
    EVALS / "runs"
    / "semantic_relevance_system_eval_v1"
)

EXPECTED_RECOMMENDER_FUNCTION = "retrieve_semantic_recommedations"

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

def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def load_frozen_contract_and_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8-sig"))
    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8-sig"))

    if lock.get("status") != "FROZEN_BEFORE_SYSTEM_EVAL_EXECUTION":
        raise ValueError("System-evaluation input lock is not frozen.")

    if contract.get("version") != lock.get("evaluation_version"):
        raise ValueError("Contract version differs from frozen input lock.")

    checks = {
        "recommender": (RECOMMENDER_FILE, lock["recommender_sha256"]),
        "queries": (QUERY_FILE, lock["query_file_sha256"]),
        "contract": (CONTRACT_FILE, lock["contract_file_sha256"]),
    }
    for label, (path, expected) in checks.items():
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"{label} changed after freeze: expected={expected} actual={actual}"
            )
        print(f"PASS  frozen {label} hash unchanged")

    current_git = git_sha()
    frozen_git = lock.get("git_commit_sha")
    if frozen_git is not None and current_git != frozen_git:
        raise ValueError(
            "Git commit changed after freeze: "
            f"frozen={frozen_git} current={current_git}"
        )
    print("PASS  frozen Git commit unchanged")

    return contract, lock

def load_recommender_namespace_without_ui() -> dict[str, Any]:
    """Execute only the backend prefix of gradio-dashboard.py.

    The dashboard file is an application entrypoint rather than an import-safe
    library module. Importing it directly could construct or launch the Gradio
    UI. To evaluate the real recommender without exercising the UI, parse the
    source, execute top-level statements only through the recommender function
    definition, and stop before UI construction.

    This preserves the actual backend initialization and the exact frozen
    retrieve_semantic_recommedations implementation from the locked file.
    """

    source = RECOMMENDER_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RECOMMENDER_FILE))

    function_index = None
    for index, node in enumerate(tree.body):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == EXPECTED_RECOMMENDER_FUNCTION
        ):
            function_index = index
            break

    if function_index is None:
        raise ValueError(
            f"Could not find {EXPECTED_RECOMMENDER_FUNCTION} in "
            f"{RECOMMENDER_FILE.name}."
        )

    # Execute all top-level setup through the target function definition.
    # Nothing after the function definition is executed, which keeps Gradio
    # dashboard construction/launch outside this system-evaluation harness.
    backend_tree = ast.Module(
        body=tree.body[: function_index + 1],
        type_ignores=[],
    )
    ast.fix_missing_locations(backend_tree)

    namespace: dict[str, Any] = {
        "__name__": "_semantic_relevance_system_eval_backend",
        "__file__": str(RECOMMENDER_FILE),
    }
    exec(
        compile(
            backend_tree,
            filename=str(RECOMMENDER_FILE),
            mode="exec",
        ),
        namespace,
        namespace,
    )

    func = namespace.get(EXPECTED_RECOMMENDER_FUNCTION)
    if not callable(func):
        raise TypeError(
            f"{EXPECTED_RECOMMENDER_FUNCTION} did not resolve to a callable."
        )

    return namespace

def normalize_result(
    result: Any,
    *,
    query_id: str,
    query: str,
) -> pd.DataFrame:
    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            "Expected recommender to return a pandas DataFrame; "
            f"got {type(result).__name__}."
        )

    if len(result) != 10:
        raise ValueError(
            f"{query_id}: expected exactly 10 recommendations; got {len(result)}."
        )

    frame = result.copy().reset_index(drop=True)

    # Insert evaluation identity before the recommender's own columns so the
    # artifact remains easy to inspect while preserving every returned field.
    frame.insert(0, "rank", range(1, len(frame) + 1))
    frame.insert(0, "query", query)
    frame.insert(0, "query_id", query_id)

    # Require the fields needed by the later judge-scoring stage. Do not guess
    # descriptions or titles from unrelated columns.
    required = ["isbn13", "title", "description"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{query_id}: recommender output is missing required judge-input "
            f"columns: {missing}. Returned columns={list(frame.columns)}"
        )

    if frame["isbn13"].astype(str).duplicated().any():
        dupes = (
            frame.loc[
                frame["isbn13"].astype(str).duplicated(keep=False),
                "isbn13",
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"{query_id}: duplicate ISBN13 values in top 10: {dupes}"
        )

    return frame

def main() -> None:
    print("Semantic relevance system evaluation - recommendation collection")
    print("---------------------------------------------------------------")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")

    contract, lock = load_frozen_contract_and_lock()

    queries = pd.read_csv(QUERY_FILE, dtype=str, encoding="utf-8")
    if len(queries) != 12 or queries["query_id"].duplicated().any():
        raise ValueError("Expected exactly 12 unique frozen queries.")

    if int(contract["benchmark"]["recommendations_per_query"]) != 10:
        raise ValueError("Frozen contract no longer specifies top 10.")

    namespace = load_recommender_namespace_without_ui()
    recommender = namespace[EXPECTED_RECOMMENDER_FUNCTION]

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_recommendations"
    )
    run_dir = RUN_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)

    frames: list[pd.DataFrame] = []

    for index, row in queries.iterrows():
        query_id = str(row["query_id"])
        query = str(row["query"])

        print(f"{index + 1:2d}/12 {query_id}: {query}")

        result = recommender(
            query=query,
            category="All",
            tone="All",
            initial_top_k=50,
            final_top_k=10,
        )
        frame = normalize_result(
            result,
            query_id=query_id,
            query=query,
        )
        frames.append(frame)

        # Persist incrementally so a partial system run is inspectable. A
        # partial run is never promoted to the immutable collection artifact.
        partial = pd.concat(frames, ignore_index=True)
        partial.to_csv(
            run_dir / "recommendations.partial.csv",
            index=False,
            encoding="utf-8",
        )

    recommendations = pd.concat(frames, ignore_index=True)

    if len(recommendations) != 120:
        raise ValueError(
            f"Expected 120 query-book pairs; got {len(recommendations)}."
        )

    counts = recommendations.groupby("query_id").size().to_dict()
    if any(int(counts.get(qid, 0)) != 10 for qid in queries["query_id"]):
        raise ValueError(
            f"Expected 10 recommendations for every query; got {counts}"
        )

    # The immutable system-output artifact for the later judge stage.
    recommendations_file = run_dir / "recommendations.csv"
    recommendations.to_csv(
        recommendations_file,
        index=False,
        encoding="utf-8",
    )

    # Minimal judge-input projection. The full recommendations.csv remains the
    # source artifact; this projection is convenience only.
    judge_input_columns = [
        "query_id",
        "query",
        "rank",
        "isbn13",
        "title",
        "description",
    ]
    if "authors" in recommendations.columns:
        judge_input_columns.append("authors")
    judge_input = recommendations[judge_input_columns].copy()
    judge_input_file = run_dir / "judge_input.csv"
    judge_input.to_csv(
        judge_input_file,
        index=False,
        encoding="utf-8",
    )

    partial_file = run_dir / "recommendations.partial.csv"
    if partial_file.exists():
        partial_file.unlink()

    metadata = {
        "evaluation_id": contract["evaluation_id"],
        "evaluation_version": contract["version"],
        "stage": "RECOMMENDATION_COLLECTION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir),
        "git_commit_sha": git_sha(),
        "recommender_file": str(RECOMMENDER_FILE.relative_to(REPO_ROOT)),
        "recommender_sha256": sha256(RECOMMENDER_FILE),
        "contract_sha256": sha256(CONTRACT_FILE),
        "input_lock_sha256": sha256(LOCK_FILE),
        "query_file_sha256": sha256(QUERY_FILE),
        "query_count": 12,
        "recommendations_per_query": 10,
        "query_book_pairs": 120,
        "category": "All",
        "tone": "All",
        "initial_top_k": 50,
        "final_top_k": 10,
        "recommendations_file": recommendations_file.name,
        "recommendations_sha256": sha256(recommendations_file),
        "judge_input_file": judge_input_file.name,
        "judge_input_sha256": sha256(judge_input_file),
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "python": sys.version,
        "platform": platform.platform(),
        "selected_package_versions": {
            name: package_version(name)
            for name in [
                "pandas",
                "langchain",
                "langchain-community",
                "langchain-openai",
                "chromadb",
                "openai",
                "gradio",
            ]
        },
    }
    metadata_file = run_dir / "run_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    collection_lock = {
        "status": "RECOMMENDATIONS_FROZEN_BEFORE_JUDGE_SCORING",
        "run_directory": str(run_dir),
        "recommendations_sha256": sha256(recommendations_file),
        "judge_input_sha256": sha256(judge_input_file),
        "query_book_pairs": 120,
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
    }
    collection_lock_file = run_dir / "recommendation_collection_lock.json"
    collection_lock_file.write_text(
        json.dumps(collection_lock, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("RECOMMENDATION COLLECTION: PASS")
    print(f"Run directory: {run_dir}")
    print(f"Queries: 12")
    print(f"Recommendations/query: 10")
    print(f"Total query-book pairs: 120")
    print(f"Recommendations SHA-256: {sha256(recommendations_file)}")
    print(f"Judge-input SHA-256: {sha256(judge_input_file)}")
    print("Judge calls performed: 0")
    print("Human relevance labels used: NO")
    print(
        "Next: score this frozen judge_input.csv with the pinned "
        "semantic_relevance_v0.29.0-r5 judge."
    )

if __name__ == "__main__":
    main()
