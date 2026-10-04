"""
Regression contract for semantic candidate ordering.

Purpose
-------
Prove that retrieve_semantic_recommedations() preserves the order returned by
the vector store when joining candidate ISBNs back to the metadata DataFrame.

This test intentionally extracts only the production function from
gradio-dashboard.py. It does not launch Gradio, load embeddings, access the real
vector database, or call an LLM.

Expected lifecycle
------------------
1. BEFORE the fix: this test MUST FAIL. That reproduces the confirmed defect.
2. Apply the minimal production fix.
3. AFTER the fix: this same test MUST PASS unchanged.

Do not weaken or rewrite the assertion to make the test pass.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = REPO_ROOT / "gradio-dashboard.py"
TARGET_FUNCTION = "retrieve_semantic_recommedations"


@dataclass
class FakeDocument:
    page_content: str


class FakeVectorStore:
    """Returns a fixed semantic ranking independent of metadata row order."""

    def __init__(self, ranked_isbns: list[int]) -> None:
        self.ranked_isbns = ranked_isbns

    def similarity_search(self, query: str, k: int):
        del query
        return [
            FakeDocument(page_content=f"{isbn} synthetic description")
            for isbn in self.ranked_isbns[:k]
        ]


def load_production_function():
    """Compile only the target function from gradio-dashboard.py.

    This keeps the test tightly coupled to the production implementation while
    avoiding top-level app initialization and Gradio launch behavior.
    """
    source = SOURCE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_FILE))

    target = None
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == TARGET_FUNCTION
        ):
            target = node
            break

    if target is None:
        raise AssertionError(
            f"Could not find {TARGET_FUNCTION} in {SOURCE_FILE}"
        )

    module = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {
        "pd": pd,
    }

    exec(
        compile(module, str(SOURCE_FILE), "exec"),
        namespace,
    )

    return namespace[TARGET_FUNCTION], namespace


def main() -> None:
    retrieve, namespace = load_production_function()

    # Deliberately make metadata order differ from vector-similarity order.
    #
    # Vector ranking:
    #   300 -> 100 -> 400 -> 200
    #
    # Metadata DataFrame order:
    #   100 -> 200 -> 300 -> 400
    #
    # The production function must preserve the VECTOR ranking.
    vector_ranked_isbns = [300, 100, 400, 200]

    metadata = pd.DataFrame(
        [
            {
                "isbn13": 100,
                "title": "Book 100",
                "simple_categories": "Synthetic",
                "joy": 0.1,
                "surprise": 0.1,
                "anger": 0.1,
                "fear": 0.1,
                "sadness": 0.1,
            },
            {
                "isbn13": 200,
                "title": "Book 200",
                "simple_categories": "Synthetic",
                "joy": 0.2,
                "surprise": 0.2,
                "anger": 0.2,
                "fear": 0.2,
                "sadness": 0.2,
            },
            {
                "isbn13": 300,
                "title": "Book 300",
                "simple_categories": "Synthetic",
                "joy": 0.3,
                "surprise": 0.3,
                "anger": 0.3,
                "fear": 0.3,
                "sadness": 0.3,
            },
            {
                "isbn13": 400,
                "title": "Book 400",
                "simple_categories": "Synthetic",
                "joy": 0.4,
                "surprise": 0.4,
                "anger": 0.4,
                "fear": 0.4,
                "sadness": 0.4,
            },
        ]
    )

    namespace["db_books"] = FakeVectorStore(vector_ranked_isbns)
    namespace["books"] = metadata

    result = retrieve(
        query="synthetic semantic query",
        category="All",
        tone="All",
        initial_top_k=4,
        final_top_k=3,
    )

    actual = result["isbn13"].astype(int).tolist()
    expected = vector_ranked_isbns[:3]

    print("SEMANTIC RETRIEVAL ORDER REGRESSION")
    print("-----------------------------------")
    print("Vector-store ranking: ", vector_ranked_isbns)
    print("Metadata row order:   ", metadata["isbn13"].astype(int).tolist())
    print("Expected final top-3: ", expected)
    print("Actual final top-3:   ", actual)
    print()

    if actual != expected:
        print("REGRESSION TEST: FAIL (EXPECTED BEFORE FIX)")
        print(
            "Confirmed: production function does not preserve "
            "vector-store semantic order."
        )
        raise AssertionError(
            "Semantic order was lost while joining vector candidates "
            f"to metadata: expected={expected}, actual={actual}"
        )

    print("REGRESSION TEST: PASS")
    print("Production function preserves vector-store semantic order.")


if __name__ == "__main__":
    main()
