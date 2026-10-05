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

# The benchmark definition remains the original frozen v1 contract/query set.
CONTRACT_FILE = (
    EVALS / "system_evaluation"
    / "semantic_relevance_system_eval.v1.0.0.json"
)
BASELINE_INPUT_LOCK_FILE = (
    EVALS / "system_evaluation"
    / "semantic_relevance_system_eval_input_lock.v1.0.0.json"
)
QUERY_FILE = (
    EVALS / "datasets"
    / "semantic_relevance_system_eval_queries.v1.0.0.csv"
)

# The system under test is now the corrected v2 recommender.
RECOMMENDER_V2_LOCK_FILE = (
    EVALS / "system_evaluation"
    / "semantic_recommender_v2_lock.json"
)
RECOMMENDER_FILE = REPO_ROOT / "gradio-dashboard.py"

# Never write into the v1 baseline namespace.
RUN_ROOT = (
    EVALS / "runs"
    / "semantic_relevance_system_eval_v2"
)

EXPECTED_RECOMMENDER_FUNCTION = "retrieve_semantic_recommedations"
EXPECTED_RECOMMENDER_SHA256 = "cdd0458ff23bd669185a8c882ceb132438d037af1a29f4dd60bd7fef9c405b60"
EXPECTED_EXECUTION_GIT_COMMIT = "91fd9d2a0ebde9f66e7eb6b826631db70b63a165"
BASELINE_V1_RECOMMENDER_SHA256 = "b8f7c1223a525345a7e278fd29e0cd32a5e0c644a9a0b88e5cc9ac319f749974"
BASELINE_V1_RUN = (
    "evals/runs/semantic_relevance_system_eval_v1/"
    "20261002T152856Z_recommendations"
)
SYSTEM_VARIANT = "v2-orderfix"


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


def load_frozen_benchmark_and_v2_recommender_lock() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Verify the controlled v1 -> v2 comparison boundary.

    The original v1 input lock remains authoritative for the benchmark
    contract and query set. The v2 recommender lock is authoritative for the
    corrected system-under-test source.

    This intentionally does not require the entire Git worktree to be clean;
    unrelated historical artifacts may remain present. Instead it pins the
    exact execution commit plus the exact recommender source hash.
    """
    contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8-sig"))
    baseline_lock = json.loads(
        BASELINE_INPUT_LOCK_FILE.read_text(encoding="utf-8-sig")
    )
    v2_lock = json.loads(
        RECOMMENDER_V2_LOCK_FILE.read_text(encoding="utf-8-sig")
    )

    if baseline_lock.get("status") != "FROZEN_BEFORE_SYSTEM_EVAL_EXECUTION":
        raise ValueError("Original system-evaluation input lock is not frozen.")

    if contract.get("version") != baseline_lock.get("evaluation_version"):
        raise ValueError(
            "Contract version differs from original frozen input lock."
        )

    # Keep the benchmark itself byte-identical to v1.
    benchmark_checks = {
        "queries": (
            QUERY_FILE,
            baseline_lock["query_file_sha256"],
        ),
        "contract": (
            CONTRACT_FILE,
            baseline_lock["contract_file_sha256"],
        ),
    }

    for label, (path, expected) in benchmark_checks.items():
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"{label} changed after v1 benchmark freeze: "
                f"expected={expected} actual={actual}"
            )
        print(f"PASS  frozen v1 benchmark {label} hash unchanged")

    # Pin the corrected recommender independently from the v1 system lock.
    if v2_lock.get("status") != "FROZEN_FOR_SYSTEM_EVALUATION":
        raise ValueError("Semantic recommender v2 lock is not frozen.")

    if v2_lock.get("candidate") != "semantic-recommender-v2-orderfix":
        raise ValueError(
            "Unexpected v2 recommender candidate in freeze lock."
        )

    locked_source_sha = v2_lock.get("source_sha256")
    if locked_source_sha != EXPECTED_RECOMMENDER_SHA256:
        raise ValueError(
            "v2 lock source hash differs from expected corrected recommender: "
            f"lock={locked_source_sha} expected={EXPECTED_RECOMMENDER_SHA256}"
        )

    actual_recommender_sha = sha256(RECOMMENDER_FILE)
    if actual_recommender_sha != EXPECTED_RECOMMENDER_SHA256:
        raise ValueError(
            "Corrected recommender changed before v2 collection: "
            f"expected={EXPECTED_RECOMMENDER_SHA256} "
            f"actual={actual_recommender_sha}"
        )
    print("PASS  frozen v2 recommender hash unchanged")

    baseline_sha = v2_lock.get("baseline_v1_source_sha256")
    if baseline_sha != BASELINE_V1_RECOMMENDER_SHA256:
        raise ValueError(
            "v2 lock does not point back to the expected v1 recommender."
        )
    print("PASS  v2 lock links to frozen v1 recommender")

    policy = v2_lock.get("controlled_comparison_policy", {})
    required_policy = {
        "same_queries": True,
        "same_judge": True,
        "same_rubric": True,
        "same_scoring_logic": True,
        "same_initial_top_k": 50,
        "same_final_top_k": 10,
        "same_bootstrap_method": True,
        "same_release_gates": True,
    }
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            raise ValueError(
                f"v2 controlled-comparison policy mismatch for {key}: "
                f"expected={expected!r} actual={policy.get(key)!r}"
            )
    print("PASS  v2 controlled-comparison policy unchanged")

    current_git = git_sha()
    if current_git != EXPECTED_EXECUTION_GIT_COMMIT:
        raise ValueError(
            "Git commit changed before v2 recommendation collection: "
            f"expected={EXPECTED_EXECUTION_GIT_COMMIT} current={current_git}"
        )
    print("PASS  v2 execution Git commit pinned")

    return contract, baseline_lock, v2_lock


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

    backend_tree = ast.Module(
        body=tree.body[: function_index + 1],
        type_ignores=[],
    )
    ast.fix_missing_locations(backend_tree)

    namespace: dict[str, Any] = {
        "__name__": "_semantic_relevance_system_eval_v2_backend",
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
            f"{query_id}: expected exactly 10 recommendations; "
            f"got {len(result)}."
        )

    frame = result.copy().reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    frame.insert(0, "query", query)
    frame.insert(0, "query_id", query_id)

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
    print(
        "Semantic relevance system evaluation v2 - "
        "recommendation collection"
    )
    print("----------------------------------------------------------")
    print("System variant: v2-orderfix")
    print("Comparison baseline: frozen v1 system evaluation")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")

    contract, baseline_lock, v2_lock = (
        load_frozen_benchmark_and_v2_recommender_lock()
    )

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

    recommendations_file = run_dir / "recommendations.csv"
    recommendations.to_csv(
        recommendations_file,
        index=False,
        encoding="utf-8",
    )

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
        "system_eval_variant": SYSTEM_VARIANT,
        "stage": "RECOMMENDATION_COLLECTION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir),
        "git_commit_sha": git_sha(),
        "expected_execution_git_commit": EXPECTED_EXECUTION_GIT_COMMIT,
        "recommender_file": str(
            RECOMMENDER_FILE.relative_to(REPO_ROOT)
        ),
        "recommender_sha256": sha256(RECOMMENDER_FILE),
        "baseline_v1_recommender_sha256": BASELINE_V1_RECOMMENDER_SHA256,
        "contract_sha256": sha256(CONTRACT_FILE),
        "baseline_input_lock_sha256": sha256(BASELINE_INPUT_LOCK_FILE),
        "recommender_v2_lock_sha256": sha256(RECOMMENDER_V2_LOCK_FILE),
        "query_file_sha256": sha256(QUERY_FILE),
        "baseline_v1_run": BASELINE_V1_RUN,
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
        "controlled_comparison": {
            "same_queries": True,
            "same_judge": True,
            "same_rubric": True,
            "same_scoring_logic": True,
            "same_initial_top_k": 50,
            "same_final_top_k": 10,
            "same_bootstrap_method": True,
            "same_release_gates": True,
            "intended_system_change": (
                "preserve vector-store semantic rank through metadata join"
            ),
        },
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
        "system_eval_variant": SYSTEM_VARIANT,
        "run_directory": str(run_dir),
        "recommender_sha256": sha256(RECOMMENDER_FILE),
        "expected_execution_git_commit": EXPECTED_EXECUTION_GIT_COMMIT,
        "baseline_v1_run": BASELINE_V1_RUN,
        "recommendations_sha256": sha256(recommendations_file),
        "judge_input_sha256": sha256(judge_input_file),
        "query_book_pairs": 120,
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
    }

    collection_lock_file = (
        run_dir / "recommendation_collection_lock.json"
    )
    collection_lock_file.write_text(
        json.dumps(collection_lock, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("V2 RECOMMENDATION COLLECTION: PASS")
    print(f"System variant: {SYSTEM_VARIANT}")
    print(f"Run directory: {run_dir}")
    print("Queries: 12")
    print("Recommendations/query: 10")
    print("Total query-book pairs: 120")
    print(f"Recommender SHA-256: {sha256(RECOMMENDER_FILE)}")
    print(
        f"Recommendations SHA-256: "
        f"{sha256(recommendations_file)}"
    )
    print(f"Judge-input SHA-256: {sha256(judge_input_file)}")
    print("Judge calls performed: 0")
    print("Human relevance labels used: NO")
    print(f"Baseline v1 run: {BASELINE_V1_RUN}")
    print(
        "Next: score this frozen v2 judge_input.csv with the same "
        "pinned semantic_relevance_v0.29.0-r5 judge."
    )


if __name__ == "__main__":
    main()
