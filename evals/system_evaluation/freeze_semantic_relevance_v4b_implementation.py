from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
SYS = EVALS / "system_evaluation"

V4B_SPEC_JSON = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.json"
V4B_SPEC_MD = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.md"
V4B_DESIGN_LOCK = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.lock.json"
RANKING_SOURCE = SYS / "semantic_relevance_v4b_ranking.py"
RANKING_TESTS = SYS / "test_semantic_relevance_v4b_ranking.py"
IMPLEMENTATION_LOCK = SYS / "semantic_relevance_v4b_implementation.lock.json"

EXPECTED = {
    "spec_json": "c72b5edbc330feae0620314fa5d5c6f90a55b356174d70a075ae77b47a8d0fca",
    "spec_md": "dae818b2519be70d49880e28642d76c414fa2a6ccda139242af5174984d29850",
    "design_lock": "e60ca1032f7fa733f3e7a51ec87a229d7454645aa6109b5e557588af50383c6d",
    "ranking_source": "5729b078cee1a2706a8c3d9bcaa9e8449292bc1f9f1a2816053b84758db7d677",
    "ranking_tests": "76593ff7471ed8e4beb61bcd349e3788bf97081d405decda0821f403b6c730f2",
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def lf_sha256(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def verify_text(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    raw = sha256(path)
    if raw == expected:
        mode = "raw"
    else:
        normalized = lf_sha256(path)
        req(normalized == expected, f"{label} hash mismatch: expected={expected} raw={raw} lf={normalized}")
        mode = "canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")

def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

def main() -> None:
    print("Semantic relevance V4B - implementation freeze")
    print("----------------------------------------------")
    print("V4B recommendation generation: DISABLED")
    print("Retrieval execution: DISABLED")
    print("Model execution: DISABLED")
    print("Judge execution: DISABLED")
    print("Development selection: NOT CALCULATED")
    print("Release analysis: DISABLED")
    print()

    req(not IMPLEMENTATION_LOCK.exists(), f"Refusing to overwrite existing implementation lock: {IMPLEMENTATION_LOCK}")

    verify_text(V4B_SPEC_JSON, EXPECTED["spec_json"], "V4B JSON spec")
    verify_text(V4B_SPEC_MD, EXPECTED["spec_md"], "V4B Markdown spec")
    verify_raw(V4B_DESIGN_LOCK, EXPECTED["design_lock"], "V4B design lock")
    verify_text(RANKING_SOURCE, EXPECTED["ranking_source"], "V4B ranking source")
    verify_text(RANKING_TESTS, EXPECTED["ranking_tests"], "V4B ranking tests")

    design = load(V4B_DESIGN_LOCK)
    req(design.get("status") == "V4B_DESIGN_FROZEN_BEFORE_IMPLEMENTATION", "Unexpected V4B design-lock status")
    req(design.get("ranking_method") == "reranker_score_only", "V4B design ranking method changed")
    req(design.get("non_tie_retrieval_rank_weight") is False, "V4B design permits non-tie retrieval weight")
    req(design.get("reuse_frozen_union") is True, "V4B design no longer reuses frozen union")
    req(design.get("reuse_frozen_reranker_scores") is True, "V4B design no longer reuses frozen reranker scores")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(RANKING_TESTS), "-q"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip())
    req(result.returncode == 0, "V4B ranking unit tests failed")
    print("PASS  V4B ranking unit tests passed")

    payload = {
        "status": "V4B_IMPLEMENTATION_FROZEN_BEFORE_DEVELOPMENT_RANKING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "semantic-relevance-hybrid-retrieval-v4b",
        "version": "0.1.0",
        "v4b_spec_json_sha256": EXPECTED["spec_json"],
        "v4b_spec_md_sha256": EXPECTED["spec_md"],
        "v4b_design_lock_sha256": EXPECTED["design_lock"],
        "v4b_ranking_source_sha256": EXPECTED["ranking_source"],
        "v4b_ranking_tests_sha256": EXPECTED["ranking_tests"],
        "ranking_contract": [
            "reranker_score descending",
            "hybrid_retrieval_rank ascending on exact reranker_score tie only",
            "isbn13 ascending on remaining exact tie"
        ],
        "score_rounding": False,
        "epsilon_tie_bucket": False,
        "score_calibration": False,
        "retrieval_rank_non_tie_weight": False,
        "top_k": 10,
        "recommendations_generated": False,
        "retrieval_calls_performed": 0,
        "model_calls_performed": 0,
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "development_selection_calculated": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "next_stage": "DERIVE_V4B_DEVELOPMENT_TOP10_FROM_FROZEN_V4A_RERANKER_SCORES"
    }
    IMPLEMENTATION_LOCK.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()
    print("V4B IMPLEMENTATION FREEZE: PASS")
    print(f"Ranking source SHA-256: {sha256(RANKING_SOURCE)}")
    print(f"Ranking tests SHA-256: {sha256(RANKING_TESTS)}")
    print(f"Implementation lock SHA-256: {sha256(IMPLEMENTATION_LOCK)}")
    print("Recommendations generated: NO")
    print("Retrieval calls: 0")
    print("Model calls: 0")
    print("Judge calls: 0")
    print("Development selection calculated: NO")
    print("Release analysis: NO")
    print("Next: deterministically derive V4B 24x10 recommendations from frozen V4A reranker scores.")

if __name__ == "__main__":
    main()
