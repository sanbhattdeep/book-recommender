"""Static/runtime verifier for the v0.28.0 r4 one-time validation package.

This performs no judge calls. It pins the preregistered validation protocol,
frozen candidate manifest, independent evidence hashes, and confirms that the
final holdout is still untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
E = REPO_ROOT / "evals"
D = E / "datasets"
R = E / "releases"

EXPECTED = {
    R / "semantic_relevance_v0.28.0_r4_release_candidate.json": "6d1f390a21d804351ff75def5bfd74e7e5eafbd48685aa4ee50058b6af670faa",
    R / "semantic_relevance_v0.28.0_r4_validation_protocol.v1.0.0.json": "6b2e9153256dd33e01a3b35733b89dbeb45033f12210c0e3ec8aab9592eddb03",
    D / "semantic_relevance_unseen_split_manifest.v4.0.0.json": "889673f565cc7c377f6f0c11b690c709c10e93b574e9ad404c852cb2f2d742df",
    D / "semantic_relevance_validation.v4.0.0.DO_NOT_RUN_YET.csv": "41b548363a984fd8f1a923e23fbb8b1db2c9b6bd3e00c9d43102ecca0c47672f",
    D / "semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv": "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85",
}

SEMANTIC_HASHES = {
    "evals/semantic_relevance_facet_judge.py": "2ad6fb76876c3d892de3299a7867d6d7bff8b8456975796b310e3969b9b09c1b",
    "evals/semantic_relevance_facet_scoring.py": "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808",
    "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.9.json": "378fe3d381703a07b22256b4b5e26e578d2336391cc3e4d861591bdbe314aafe",
    "evals/judge_configs/semantic_relevance_judge.v0.28.0.json": "82e75d61e0b1c02bc126edd40749b480a5643a966c05725802688b98dc78c64b",
    "evals/rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json": "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)
    print("PASS  " + message)


def main() -> None:
    print("v0.28.0 r4 validation-execution package verification")
    print("---------------------------------------------------")
    print("JUDGE EXECUTION: DISABLED")

    for path, expected in EXPECTED.items():
        require(path.exists(), f"required frozen file exists: {path.relative_to(REPO_ROOT)}")
        require(sha256(path) == expected, f"hash pinned: {path.relative_to(REPO_ROOT)}")

    for rel, expected in SEMANTIC_HASHES.items():
        path = REPO_ROOT / rel
        require(path.exists(), f"semantic artifact exists: {rel}")
        require(sha256(path) == expected, f"semantic artifact hash pinned: {rel}")

    protocol = load_json(R / "semantic_relevance_v0.28.0_r4_validation_protocol.v1.0.0.json")
    require(len(protocol.get("pre_registered_checks", {})) == 8, "exactly eight validation checks preregistered")
    require(protocol.get("one_time_validation_run") is True, "validation is one-time")
    require(protocol.get("final_holdout_must_remain_locked") is True, "protocol requires final holdout to remain locked")

    validation_lock = load_json(D / "semantic_relevance_validation_lock.v4.0.0.json")
    final_lock = load_json(D / "semantic_relevance_final_holdout_lock.v4.0.0.json")

    vstatus = validation_lock.get("status")
    require(vstatus in {"LOCKED_DO_NOT_RUN", "INDEPENDENT_VALIDATION_STARTED", "INDEPENDENT_VALIDATION_COMPLETED"},
            "validation lock has an allowed lifecycle state")
    require(int(validation_lock.get("judge_run_count", -1)) in {0, 1}, "validation judge_run_count is 0 or 1")

    require(final_lock.get("status") == "LOCKED_DO_NOT_RUN", "final holdout remains LOCKED_DO_NOT_RUN")
    require(int(final_lock.get("judge_run_count", -1)) == 0, "final holdout judge_run_count remains 0")
    require(final_lock.get("evidence_role") == "independent_final_holdout", "final holdout lock role correct")

    print("v0.28.0 r4 validation-execution package verification passed.")
    print("NO JUDGE CALLS WERE MADE.")


if __name__ == "__main__":
    main()
