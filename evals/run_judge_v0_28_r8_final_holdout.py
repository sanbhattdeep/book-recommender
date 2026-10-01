"""One-time independent final-holdout runner for frozen v0.28.0 r8.

This module makes no semantic/scoring changes. It imports the frozen r8
development runner and substitutes only:
- the frozen 30-case independent final-holdout dataset;
- final-holdout provenance metadata;
- one-time finality marker / lock lifecycle.

A fresh run is permitted exactly once. Any interruption must resume the same
recorded run directory. A completed result must never be rerun for the same
candidate, regardless of whether the final gates PASS or REVIEW.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
sys.path.insert(0, str(EVALS_DIR))

import run_judge_v0_28_r8_development as base  # noqa: E402

FINAL_RUNS_DIR = EVALS_DIR / "runs" / "semantic_relevance_v0_28_final_holdout"
FINALITY_MARKER = FINAL_RUNS_DIR / "FINAL_HOLDOUT_STARTED.json"

DEFAULT_RELEASE_MANIFEST = EVALS_DIR / "releases" / "semantic_relevance_v0.28.0_r8_release_candidate.json"
DEFAULT_PROTOCOL = EVALS_DIR / "releases" / "semantic_relevance_v0.28.0_r8_final_holdout_protocol.v1.0.0.json"
DEFAULT_DATASET = EVALS_DIR / "datasets" / "semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv"

FINAL_LOCK = EVALS_DIR / "datasets" / "semantic_relevance_final_holdout_lock.v4.0.0.json"
VALIDATION_LOCK = EVALS_DIR / "datasets" / "semantic_relevance_validation_lock.v4.0.0.json"
SPLIT_MANIFEST = EVALS_DIR / "datasets" / "semantic_relevance_unseen_split_manifest.v4.0.0.json"

CONFIRM_TOKEN = "FINAL_V0_28_R8"
EXPECTED_CASES = 30
EXPECTED_DATASET_VERSION = "4.0.0"
EXPECTED_HOLDOUT_SHA256 = "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85"
EXPECTED_SPLIT_SHA256 = "889673f565cc7c377f6f0c11b690c709c10e93b574e9ad404c852cb2f2d742df"
EXPECTED_RELEASE_MANIFEST_SHA256 = "c25759ccaba3a3258552a5c138170821b3a9afd48acb968859a5943b4fc91cab"
EXPECTED_PROTOCOL_SHA256 = "7aa066815dd7c084be20773daec70c09981feb6c646061b658ed4fea1fca87f8"
EXPECTED_RELEASE_CANDIDATE_ID = "semantic_relevance_v0.28.0-r8"

ROLE = "independent_final_holdout"
SCOPE = "one_time_blind_final_holdout_of_frozen_v0_28_0_r8"

_original_write_metadata = base.write_metadata
_RUNTIME: dict[str, Any] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_release_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != EXPECTED_RELEASE_MANIFEST_SHA256:
        raise ValueError(
            f"Frozen r8 release manifest changed: {actual} != {EXPECTED_RELEASE_MANIFEST_SHA256}"
        )
    manifest = load_json(path)
    if manifest.get("release_candidate") != EXPECTED_RELEASE_CANDIDATE_ID:
        raise ValueError("Unexpected frozen release candidate identity.")
    if manifest.get("release_status") != "FROZEN_BEFORE_FINAL_HOLDOUT":
        raise ValueError("r8 release candidate is not frozen before final holdout.")
    if manifest.get("decision") != "READY_FOR_FINAL_HOLDOUT_EXECUTION_REVIEW":
        raise ValueError("r8 release candidate is not authorized for final-holdout review.")
    return manifest


def validate_protocol(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    if sha256(path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Final-holdout protocol changed after preregistration.")
    protocol = load_json(path)
    if protocol.get("candidate_release") != EXPECTED_RELEASE_CANDIDATE_ID:
        raise ValueError("Unexpected final-holdout protocol candidate.")
    if protocol.get("evidence_role") != ROLE:
        raise ValueError("Unexpected final-holdout protocol evidence role.")
    if protocol.get("dataset_sha256") != EXPECTED_HOLDOUT_SHA256:
        raise ValueError("Protocol final-holdout SHA mismatch.")
    policy = protocol.get("execution_policy", {})
    if int(policy.get("fresh_runs_permitted", -1)) != 1:
        raise ValueError("Protocol must permit exactly one fresh final-holdout run.")
    if int(policy.get("case_count", -1)) != EXPECTED_CASES:
        raise ValueError("Protocol final-holdout case count mismatch.")
    if policy.get("recovery_policy") != "resume_recorded_run_only":
        raise ValueError("Protocol recovery policy changed.")
    if policy.get("rerun_after_completed_result_permitted") is not False:
        raise ValueError("Protocol must forbid completed final-holdout reruns.")
    if policy.get("retune_after_result_permitted") is not False:
        raise ValueError("Protocol must forbid same-candidate retuning after final result.")
    if policy.get("relabel_after_result_permitted") is not False:
        raise ValueError("Protocol must forbid relabeling after final result.")
    if policy.get("confirmation_token") != CONFIRM_TOKEN:
        raise ValueError("Protocol confirmation token mismatch.")
    if len(protocol.get("preregistered_checks", {})) != 8:
        raise ValueError("Protocol must contain exactly eight preregistered checks.")
    return protocol


def validate_historical_validation_lock() -> dict:
    if not VALIDATION_LOCK.exists():
        raise FileNotFoundError(VALIDATION_LOCK)
    lock = load_json(VALIDATION_LOCK)
    if lock.get("evidence_role") != "independent_validation":
        raise ValueError("Historical validation lock role changed.")
    if lock.get("status") != "INDEPENDENT_VALIDATION_COMPLETED":
        raise ValueError("Historical validation is not in completed state.")
    if int(lock.get("judge_run_count", -1)) != 1:
        raise ValueError("Historical validation judge_run_count must remain 1.")
    if lock.get("validation_decision") != "REVIEW_STOP_FINAL_HOLDOUT":
        raise ValueError("Historical r4 validation decision changed.")
    return lock


def validate_final_lock_pre_start(dataset: Path) -> dict:
    if not FINAL_LOCK.exists():
        raise FileNotFoundError(FINAL_LOCK)
    lock = load_json(FINAL_LOCK)
    if lock.get("evidence_role") != ROLE:
        raise ValueError("Unexpected final-holdout lock role.")
    if lock.get("status") != "LOCKED_DO_NOT_RUN":
        raise ValueError(
            f"Final holdout lock status is {lock.get('status')!r}, expected LOCKED_DO_NOT_RUN."
        )
    if int(lock.get("judge_run_count", -1)) != 0:
        raise ValueError("Final holdout judge_run_count must be 0 before first start.")
    if bool(lock.get("finality_marker_present")):
        raise ValueError("Final holdout lock unexpectedly records an existing finality marker.")
    if sha256(dataset) != EXPECTED_HOLDOUT_SHA256:
        raise ValueError("Final holdout dataset hash changed before first start.")
    return lock


def transition_final_lock_started(
    dataset: Path,
    release_manifest: Path,
    protocol: Path,
    run_dir: Path,
) -> None:
    lock = load_json(FINAL_LOCK)

    if lock.get("status") == "LOCKED_DO_NOT_RUN" and int(lock.get("judge_run_count", -1)) == 0:
        lock.update({
            "status": "FINAL_HOLDOUT_STARTED",
            "judge_run_count": 1,
            "finality_marker_present": True,
            "release_candidate_id": EXPECTED_RELEASE_CANDIDATE_ID,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_directory": str(run_dir.resolve()),
            "dataset_sha256": sha256(dataset),
            "release_manifest_sha256": sha256(release_manifest),
            "final_holdout_protocol_sha256": sha256(protocol),
            "instruction": (
                "Resume this same final-holdout run after interruption. "
                "Never start a second fresh final-holdout run for this candidate."
            ),
        })
        write_json_atomic(FINAL_LOCK, lock)
        return

    if (
        lock.get("status") in {"FINAL_HOLDOUT_STARTED", "FINAL_HOLDOUT_COMPLETED"}
        and int(lock.get("judge_run_count", -1)) == 1
    ):
        if Path(str(lock.get("run_directory"))).resolve() != run_dir.resolve():
            raise RuntimeError("Final-holdout lock points to a different run directory.")
        if lock.get("dataset_sha256") != sha256(dataset):
            raise RuntimeError("Final-holdout lock dataset SHA mismatch.")
        if lock.get("release_manifest_sha256") != sha256(release_manifest):
            raise RuntimeError("Final-holdout lock release-manifest SHA mismatch.")
        if lock.get("final_holdout_protocol_sha256") != sha256(protocol):
            raise RuntimeError("Final-holdout lock protocol SHA mismatch.")
        return

    raise RuntimeError(
        "Final-holdout lock cannot transition/resume: "
        f"status={lock.get('status')!r}, judge_run_count={lock.get('judge_run_count')!r}"
    )


def validate_inputs(dataset, rubric, config, facet_payload, facet_specs) -> None:
    required = {
        "case_id", "query_id", "query", "title", "authors", "description",
        "human_score", "human_reason", "review_status", "dataset_version",
        "rubric_version", "isbn13",
    }
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Final-holdout dataset missing required columns: {sorted(missing)}")
    if len(dataset) != EXPECTED_CASES:
        raise ValueError(
            f"Final holdout must contain exactly {EXPECTED_CASES} cases; found {len(dataset)}."
        )
    if dataset["case_id"].astype(str).duplicated().any():
        raise ValueError("Final holdout has duplicate case_id values.")
    if not dataset["case_id"].astype(str).str.match(r"^U4_Q\d{2}_T0[1-5]$").all():
        raise ValueError("Final-holdout case IDs must use the frozen U4_Qxx_T0x convention.")

    scores = pd.to_numeric(dataset["human_score"], errors="coerce")
    if scores.isna().any() or not scores.isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Final-holdout human_score must be complete and in 0..4.")
    expected_scores = {0: 19, 1: 3, 2: 4, 3: 3, 4: 1}
    actual_scores = scores.astype(int).value_counts().to_dict()
    if actual_scores != expected_scores:
        raise ValueError(
            f"Final-holdout human-score distribution changed: {actual_scores} != {expected_scores}"
        )
    if not dataset["human_reason"].fillna("").astype(str).str.strip().ne("").all():
        raise ValueError("Final-holdout human_reason must be complete.")
    if not dataset["review_status"].astype(str).str.strip().eq("LABELLED").all():
        raise ValueError("Every final-holdout row must remain LABELLED before judge execution.")
    if set(dataset["dataset_version"].astype(str)) != {EXPECTED_DATASET_VERSION}:
        raise ValueError("Final-holdout dataset_version mismatch.")
    if set(dataset["rubric_version"].astype(str)) != {base.RUBRIC_VERSION}:
        raise ValueError("Final-holdout rubric version mismatch.")

    if str(rubric.get("version")) != base.RUBRIC_VERSION:
        raise ValueError("Rubric file version mismatch.")
    if str(config.get("version")) != base.JUDGE_CONFIG_VERSION:
        raise ValueError("Judge config version mismatch.")
    if str(config.get("dataset_version")) != base.JUDGE_CALIBRATION_DATASET_VERSION:
        raise ValueError("Judge calibration provenance mismatch.")
    if str(config.get("rubric_version")) != base.RUBRIC_VERSION:
        raise ValueError("Judge-config rubric version mismatch.")
    if str(config.get("facet_spec_version")) != base.FACET_SPEC_VERSION:
        raise ValueError("Judge-config facet-spec version mismatch.")
    if config.get("provider") != "ollama":
        raise ValueError("Judge provider must be ollama.")
    if str(facet_payload.get("version")) != base.FACET_SPEC_VERSION:
        raise ValueError("Frozen facet-spec version mismatch.")
    if facet_payload.get("status") != "frozen":
        raise ValueError("Facet spec must remain frozen.")

    qids = set(dataset["query_id"].astype(str))
    expected_qids = {f"Q{i:02d}" for i in range(1, 13)}
    if qids != expected_qids:
        raise ValueError(f"Final holdout must cover Q01-Q12; found {sorted(qids)}")
    for qid, group in dataset.groupby("query_id"):
        if len(group) not in (2, 3):
            raise ValueError(f"Final-holdout query {qid} must contribute 2 or 3 cases.")
        texts = set(group["query"].astype(str))
        if len(texts) != 1:
            raise ValueError(f"Final-holdout query text is inconsistent inside {qid}.")
        if next(iter(texts)) != facet_specs[str(qid)].query:
            raise ValueError(f"Final-holdout query text mismatch for {qid}.")

    if sha256(SPLIT_MANIFEST) != EXPECTED_SPLIT_SHA256:
        raise ValueError("Blind split manifest changed before final holdout.")
    if sha256(DATASET_FILE_RUNTIME()) != EXPECTED_HOLDOUT_SHA256:
        raise ValueError("Final-holdout dataset SHA mismatch.")

    validate_historical_validation_lock()


def DATASET_FILE_RUNTIME() -> Path:
    return Path(str(_RUNTIME["dataset"]))


def validate_resume_metadata(run_dir: Path) -> None:
    p = run_dir / "run_metadata.json"
    if not p.exists():
        return
    m = load_json(p)
    expected = {
        "judge_config_version": "0.28.0-r8",
        "evaluation_dataset_version": EXPECTED_DATASET_VERSION,
        "evaluation_dataset_role": ROLE,
        "evaluation_scope": SCOPE,
        "judge_behavior_frozen": True,
        "independent_final_holdout_one_time": True,
        "rubric_version": "0.1.0",
        "facet_spec_version": "0.9.13",
    }
    bad = [
        f"{k}: run={m.get(k)!r}, expected={v!r}"
        for k, v in expected.items()
        if str(m.get(k)) != str(v)
    ]
    if bad:
        raise ValueError(
            "Cannot resume final holdout due provenance mismatch:\n- " + "\n- ".join(bad)
        )


def write_final_metadata(
    run_dir: Path,
    config: dict[str, Any],
    status: str,
    completed_cases: int,
    total_cases: int,
) -> None:
    _original_write_metadata(run_dir, config, status, completed_cases, total_cases)
    p = run_dir / "run_metadata.json"
    m = load_json(p)

    dataset_path = Path(str(_RUNTIME["dataset"]))
    release_path = Path(str(_RUNTIME["release_manifest"]))
    protocol_path = Path(str(_RUNTIME["protocol"]))

    m.update({
        "evaluation_dataset_version": EXPECTED_DATASET_VERSION,
        "evaluation_dataset_role": ROLE,
        "evaluation_scope": SCOPE,
        "source_splits_consumed": False,
        "development_dataset_post_holdout": False,
        "judge_behavior_frozen": True,
        "independent_final_holdout_one_time": True,
        "evaluation_dataset_file": (
            str(dataset_path.relative_to(REPO_ROOT))
            if dataset_path.is_relative_to(REPO_ROOT)
            else str(dataset_path)
        ),
        "evaluation_dataset_sha256": sha256(dataset_path),
        "release_manifest_file": (
            str(release_path.relative_to(REPO_ROOT))
            if release_path.is_relative_to(REPO_ROOT)
            else str(release_path)
        ),
        "release_manifest_sha256": sha256(release_path),
        "final_holdout_protocol_file": (
            str(protocol_path.relative_to(REPO_ROOT))
            if protocol_path.is_relative_to(REPO_ROOT)
            else str(protocol_path)
        ),
        "final_holdout_protocol_sha256": sha256(protocol_path),
        "finality_marker_file": str(FINALITY_MARKER.relative_to(REPO_ROOT)),
        "final_holdout_lock_file": str(FINAL_LOCK.relative_to(REPO_ROOT)),
        "historical_validation_lock_file": str(VALIDATION_LOCK.relative_to(REPO_ROOT)),
    })
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_fresh_run(
    dataset: Path,
    release_manifest: Path,
    protocol: Path,
    confirm: str | None,
) -> Path:
    if confirm != CONFIRM_TOKEN:
        raise ValueError(f"Fresh final holdout requires --confirm-final-holdout {CONFIRM_TOKEN}")
    if FINALITY_MARKER.exists():
        marker = load_json(FINALITY_MARKER)
        raise RuntimeError(
            "Final holdout was already started. Resume only: "
            + str(marker.get("run_directory"))
        )

    validate_final_lock_pre_start(dataset)
    validate_historical_validation_lock()

    FINAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_final_r8"
    run_dir = FINAL_RUNS_DIR / run_id
    run_dir.mkdir(parents=False, exist_ok=False)

    marker = {
        "release_candidate_id": EXPECTED_RELEASE_CANDIDATE_ID,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir.resolve()),
        "dataset_file": str(dataset.resolve()),
        "dataset_sha256": sha256(dataset),
        "release_manifest_file": str(release_manifest.resolve()),
        "release_manifest_sha256": sha256(release_manifest),
        "final_holdout_protocol_file": str(protocol.resolve()),
        "final_holdout_protocol_sha256": sha256(protocol),
        "one_time_only": True,
        "instruction": (
            "Resume this same final-holdout run after interruption. "
            "Do not start another final-holdout run, even if the final gates fail."
        ),
    }

    # Finality is established BEFORE the first possible judge call.
    write_json_atomic(FINALITY_MARKER, marker)
    transition_final_lock_started(dataset, release_manifest, protocol, run_dir)
    return run_dir


def validate_resume_finality(
    run_dir: Path,
    dataset: Path,
    release_manifest: Path,
    protocol: Path,
) -> None:
    if not FINALITY_MARKER.exists():
        raise RuntimeError("Final-holdout finality marker missing; refusing resume.")

    marker = load_json(FINALITY_MARKER)
    if marker.get("release_candidate_id") != EXPECTED_RELEASE_CANDIDATE_ID:
        raise RuntimeError("Finality marker belongs to a different release candidate.")
    if Path(str(marker.get("run_directory"))).resolve() != run_dir.resolve():
        raise RuntimeError("--resume does not match the recorded one-time final-holdout run.")
    if Path(str(marker.get("dataset_file"))).resolve() != dataset.resolve():
        raise RuntimeError("Finality marker points to a different final-holdout dataset.")
    if marker.get("dataset_sha256") != sha256(dataset):
        raise RuntimeError("Final-holdout dataset changed after finality marker creation.")
    if marker.get("release_manifest_sha256") != sha256(release_manifest):
        raise RuntimeError("Release manifest changed after final-holdout start.")
    if marker.get("final_holdout_protocol_sha256") != sha256(protocol):
        raise RuntimeError("Final-holdout protocol changed after final-holdout start.")

    transition_final_lock_started(dataset, release_manifest, protocol, run_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run frozen v0.28.0 r8 exactly once on the independent 30-case final holdout."
    )
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--release-manifest", default=str(DEFAULT_RELEASE_MANIFEST))
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    ap.add_argument("--confirm-final-holdout", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Operational recovery only; allowed only with --resume.",
    )
    args = ap.parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.case_id and not args.resume:
        raise ValueError(
            "Fresh targeted final-holdout runs are forbidden. --case-id is recovery-only with --resume."
        )

    dataset = resolve(args.dataset)
    release_manifest = resolve(args.release_manifest)
    protocol = resolve(args.protocol)

    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if sha256(dataset) != EXPECTED_HOLDOUT_SHA256:
        raise ValueError("Final-holdout dataset hash mismatch.")

    validate_release_manifest(release_manifest)
    validate_protocol(protocol)
    validate_historical_validation_lock()

    base.DATASET_FILE = dataset
    base.RUNS_DIR = FINAL_RUNS_DIR
    base.EVALUATION_DATASET_VERSION = EXPECTED_DATASET_VERSION
    base.EVALUATION_DATASET_ROLE = ROLE
    base.EVALUATION_SCOPE = SCOPE
    base.validate_inputs = validate_inputs
    base.validate_resume_metadata = validate_resume_metadata
    base.write_metadata = write_final_metadata

    _RUNTIME.update({
        "dataset": dataset,
        "release_manifest": release_manifest,
        "protocol": protocol,
    })

    if args.resume:
        run_dir = resolve(args.resume)
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        validate_resume_finality(run_dir, dataset, release_manifest, protocol)
    else:
        run_dir = prepare_fresh_run(
            dataset, release_manifest, protocol, args.confirm_final_holdout
        )

    argv = ["run_judge_v0_28_r8_development.py", "--resume", str(run_dir)]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    for cid in args.case_id:
        argv += ["--case-id", str(cid)]

    old = sys.argv[:]
    try:
        sys.argv = argv
        base.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
