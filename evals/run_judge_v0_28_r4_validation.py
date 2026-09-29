"""One-time independent-validation runner for frozen Semantic Relevance v0.28.0 r4.

This module changes only dataset/provenance/finality plumbing. The judge,
facet definitions, prompts, and deterministic scoring are imported unchanged
from the already-frozen v0.28.0 r4 development runner.

A fresh validation run may be started exactly once. Any interruption must
resume the same recorded run directory. The independent final holdout is
never executed by this module and must remain LOCKED_DO_NOT_RUN.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import unicodedata

import pandas as pd

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
sys.path.insert(0, str(EVALS_DIR))
import run_judge_development as base  # noqa: E402

VALIDATION_RUNS_DIR = EVALS_DIR / "runs" / "semantic_relevance_v0_28_validation"
FINALITY_MARKER = VALIDATION_RUNS_DIR / "INDEPENDENT_VALIDATION_STARTED.json"
DEFAULT_RELEASE_MANIFEST = EVALS_DIR / "releases" / "semantic_relevance_v0.28.0_r4_release_candidate.json"
DEFAULT_PROTOCOL = EVALS_DIR / "releases" / "semantic_relevance_v0.28.0_r4_validation_protocol.v1.0.0.json"
DEFAULT_DATASET = EVALS_DIR / "datasets" / "semantic_relevance_validation.v4.0.0.DO_NOT_RUN_YET.csv"
VALIDATION_LOCK = EVALS_DIR / "datasets" / "semantic_relevance_validation_lock.v4.0.0.json"
FINAL_HOLDOUT = EVALS_DIR / "datasets" / "semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv"
FINAL_HOLDOUT_LOCK = EVALS_DIR / "datasets" / "semantic_relevance_final_holdout_lock.v4.0.0.json"
SPLIT_MANIFEST = EVALS_DIR / "datasets" / "semantic_relevance_unseen_split_manifest.v4.0.0.json"
DEVELOPMENT_DATASET = EVALS_DIR / "datasets" / "semantic_relevance_v0.28_development.v4.0.0.csv"

CONFIRM_TOKEN = "VALIDATE_V0_28_R4"
EXPECTED_CASES = 30
EXPECTED_DATASET_VERSION = "4.0.0"
EXPECTED_VALIDATION_SHA256 = "41b548363a984fd8f1a923e23fbb8b1db2c9b6bd3e00c9d43102ecca0c47672f"
EXPECTED_HOLDOUT_SHA256 = "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85"
EXPECTED_SPLIT_SHA256 = "889673f565cc7c377f6f0c11b690c709c10e93b574e9ad404c852cb2f2d742df"
EXPECTED_RELEASE_MANIFEST_SHA256 = "6d1f390a21d804351ff75def5bfd74e7e5eafbd48685aa4ee50058b6af670faa"
EXPECTED_PROTOCOL_SHA256 = "6b2e9153256dd33e01a3b35733b89dbeb45033f12210c0e3ec8aab9592eddb03"
EXPECTED_RELEASE_CANDIDATE_ID = "semantic_relevance_v0.28.0-r4"
ROLE = "independent_validation"
SCOPE = "one_time_blind_validation_of_frozen_v0_28_0_r4"

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


def normalize_text(value: object) -> str:
    s = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", s)


def validate_release_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != EXPECTED_RELEASE_MANIFEST_SHA256:
        raise ValueError(f"Frozen r4 release manifest changed: {actual} != {EXPECTED_RELEASE_MANIFEST_SHA256}")
    manifest = load_json(path)
    if manifest.get("release_candidate_id") != EXPECTED_RELEASE_CANDIDATE_ID:
        raise ValueError("Unexpected frozen release candidate ID.")
    if manifest.get("status") != "FROZEN_BEFORE_NEW_UNSEEN_V0_28_EVIDENCE":
        raise ValueError("r4 release candidate is not in the expected frozen state.")
    for rel, info in manifest.get("artifacts", {}).items():
        p = REPO_ROOT / rel
        if not p.exists():
            raise FileNotFoundError(p)
        if sha256(p) != info.get("sha256"):
            raise ValueError(f"Frozen semantic artifact changed after release freeze: {rel}")
    return manifest


def validate_protocol(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    if sha256(path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Validation protocol changed after preregistration.")
    protocol = load_json(path)
    if protocol.get("candidate_release") != "v0.28.0_r4":
        raise ValueError("Unexpected validation protocol candidate.")
    if int(protocol.get("validation_cases", -1)) != EXPECTED_CASES:
        raise ValueError("Validation protocol case count mismatch.")
    if protocol.get("one_time_validation_run") is not True:
        raise ValueError("Validation protocol must require one-time execution.")
    if protocol.get("final_holdout_must_remain_locked") is not True:
        raise ValueError("Validation protocol must keep final holdout locked.")
    if len(protocol.get("pre_registered_checks", {})) != 8:
        raise ValueError("Validation protocol must contain exactly eight preregistered checks.")
    return protocol


def validate_final_holdout_untouched() -> dict:
    if not FINAL_HOLDOUT.exists() or not FINAL_HOLDOUT_LOCK.exists():
        raise FileNotFoundError("Independent final-holdout dataset/lock missing.")
    if sha256(FINAL_HOLDOUT) != EXPECTED_HOLDOUT_SHA256:
        raise ValueError("Independent final-holdout dataset hash changed.")
    lock = load_json(FINAL_HOLDOUT_LOCK)
    if lock.get("evidence_role") != "independent_final_holdout":
        raise ValueError("Unexpected final-holdout lock role.")
    if lock.get("status") != "LOCKED_DO_NOT_RUN":
        raise ValueError(f"Final holdout is not untouched: status={lock.get('status')!r}")
    if int(lock.get("judge_run_count", -1)) != 0:
        raise ValueError("Final holdout judge_run_count must remain 0.")
    if bool(lock.get("finality_marker_present")):
        raise ValueError("Final holdout unexpectedly records a finality marker.")
    return lock


def validate_validation_lock_pre_start(dataset: Path) -> dict:
    if not VALIDATION_LOCK.exists():
        raise FileNotFoundError(VALIDATION_LOCK)
    lock = load_json(VALIDATION_LOCK)
    if lock.get("evidence_role") != ROLE:
        raise ValueError("Unexpected validation lock role.")
    if lock.get("status") != "LOCKED_DO_NOT_RUN":
        raise ValueError(f"Validation lock status is {lock.get('status')!r}, expected LOCKED_DO_NOT_RUN.")
    if int(lock.get("judge_run_count", -1)) != 0:
        raise ValueError("Validation judge_run_count must be 0 before first start.")
    if sha256(dataset) != EXPECTED_VALIDATION_SHA256:
        raise ValueError("Validation dataset hash changed before first start.")
    return lock


def transition_validation_lock_started(dataset: Path, release_manifest: Path, protocol: Path, run_dir: Path) -> None:
    lock = load_json(VALIDATION_LOCK)
    if lock.get("status") == "LOCKED_DO_NOT_RUN" and int(lock.get("judge_run_count", -1)) == 0:
        lock.update({
            "status": "INDEPENDENT_VALIDATION_STARTED",
            "judge_run_count": 1,
            "finality_marker_present": True,
            "release_candidate_id": EXPECTED_RELEASE_CANDIDATE_ID,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_directory": str(run_dir.resolve()),
            "dataset_sha256": sha256(dataset),
            "release_manifest_sha256": sha256(release_manifest),
            "validation_protocol_sha256": sha256(protocol),
            "instruction": "Resume this same validation run after interruption. Never start a second validation run for this candidate.",
        })
        write_json_atomic(VALIDATION_LOCK, lock)
        return
    if lock.get("status") in {"INDEPENDENT_VALIDATION_STARTED", "INDEPENDENT_VALIDATION_COMPLETED"} and int(lock.get("judge_run_count", -1)) == 1:
        if Path(str(lock.get("run_directory"))).resolve() != run_dir.resolve():
            raise RuntimeError("Validation lock points to a different run directory.")
        if lock.get("dataset_sha256") != sha256(dataset):
            raise RuntimeError("Validation lock dataset hash mismatch.")
        return
    raise RuntimeError(
        f"Validation lock cannot transition/resume: status={lock.get('status')!r}, "
        f"judge_run_count={lock.get('judge_run_count')!r}"
    )


def validate_inputs(dataset, rubric, config, facet_payload, facet_specs) -> None:
    required = {
        "case_id", "query_id", "query", "title", "authors", "description",
        "human_score", "human_reason", "review_status", "dataset_version",
        "rubric_version", "isbn13",
    }
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Validation dataset missing required columns: {sorted(missing)}")
    if len(dataset) != EXPECTED_CASES:
        raise ValueError(f"Validation must contain exactly {EXPECTED_CASES} cases; found {len(dataset)}.")
    if dataset["case_id"].astype(str).duplicated().any():
        raise ValueError("Validation has duplicate case_id values.")
    if not dataset["case_id"].astype(str).str.match(r"^U4_Q\d{2}_T0[1-5]$").all():
        raise ValueError("Validation case IDs must use the frozen U4_Qxx_T0x convention.")
    scores = pd.to_numeric(dataset["human_score"], errors="coerce")
    if scores.isna().any() or not scores.isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Validation human_score must be complete and in 0..4.")
    expected_scores = {0: 19, 1: 3, 2: 4, 3: 3, 4: 1}
    actual_scores = scores.astype(int).value_counts().to_dict()
    if actual_scores != expected_scores:
        raise ValueError(f"Validation human-score distribution changed: {actual_scores} != {expected_scores}")
    if not dataset["human_reason"].fillna("").astype(str).str.strip().ne("").all():
        raise ValueError("Validation human_reason must be complete.")
    if not dataset["review_status"].astype(str).str.strip().eq("LABELLED").all():
        raise ValueError("Every validation row must be LABELLED before judge execution.")
    if set(dataset["dataset_version"].astype(str)) != {EXPECTED_DATASET_VERSION}:
        raise ValueError("Validation dataset_version mismatch.")
    if set(dataset["rubric_version"].astype(str)) != {base.RUBRIC_VERSION}:
        raise ValueError("Validation rubric version mismatch.")
    if str(rubric.get("version")) != base.RUBRIC_VERSION:
        raise ValueError("Rubric file version mismatch.")
    if str(config.get("version")) != base.JUDGE_CONFIG_VERSION:
        raise ValueError("Judge config version mismatch.")
    if str(config.get("dataset_version")) != base.JUDGE_CALIBRATION_DATASET_VERSION:
        raise ValueError("Judge calibration provenance mismatch.")
    if str(config.get("rubric_version")) != base.RUBRIC_VERSION or str(config.get("facet_spec_version")) != base.FACET_SPEC_VERSION:
        raise ValueError("Judge config artifact version mismatch.")
    if config.get("provider") != "ollama":
        raise ValueError("Judge provider must be ollama.")
    if str(facet_payload.get("version")) != base.FACET_SPEC_VERSION or facet_payload.get("status") != "frozen":
        raise ValueError("Frozen facet spec mismatch.")

    qids = set(dataset["query_id"].astype(str))
    expected_qids = {f"Q{i:02d}" for i in range(1, 13)}
    if qids != expected_qids:
        raise ValueError(f"Validation must cover Q01-Q12; found {sorted(qids)}")
    for qid, group in dataset.groupby("query_id"):
        if len(group) not in (2, 3):
            raise ValueError(f"Validation query {qid} must contribute 2 or 3 cases.")
        texts = set(group["query"].astype(str))
        if len(texts) != 1:
            raise ValueError(f"Validation query text is inconsistent inside {qid}.")
        frozen = facet_specs[qid].query
        if next(iter(texts)) != frozen:
            raise ValueError(f"Validation query text mismatch for {qid}.")

    if sha256(SPLIT_MANIFEST) != EXPECTED_SPLIT_SHA256:
        raise ValueError("Split manifest changed before validation.")
    if sha256(DATASET_FILE_RUNTIME()) != EXPECTED_VALIDATION_SHA256:
        raise ValueError("Validation dataset SHA mismatch.")
    validate_final_holdout_untouched()


def DATASET_FILE_RUNTIME() -> Path:
    return Path(str(_RUNTIME["dataset"]))


def validate_resume_metadata(run_dir: Path) -> None:
    p = run_dir / "run_metadata.json"
    if not p.exists():
        return
    m = load_json(p)
    expected = {
        "judge_config_version": "0.28.0",
        "evaluation_dataset_version": EXPECTED_DATASET_VERSION,
        "evaluation_dataset_role": ROLE,
        "evaluation_scope": SCOPE,
        "judge_behavior_frozen": True,
        "independent_validation_one_time": True,
        "rubric_version": "0.1.0",
        "facet_spec_version": "0.9.9",
    }
    bad = [
        f"{k}: run={m.get(k)!r}, expected={v!r}"
        for k, v in expected.items()
        if str(m.get(k)) != str(v)
    ]
    if bad:
        raise ValueError("Cannot resume validation due provenance mismatch:\n- " + "\n- ".join(bad))


def write_validation_metadata(run_dir: Path, config: dict[str, Any], status: str, completed_cases: int, total_cases: int) -> None:
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
        "independent_validation_one_time": True,
        "evaluation_dataset_file": str(dataset_path.relative_to(REPO_ROOT)) if dataset_path.is_relative_to(REPO_ROOT) else str(dataset_path),
        "evaluation_dataset_sha256": sha256(dataset_path),
        "release_manifest_file": str(release_path.relative_to(REPO_ROOT)) if release_path.is_relative_to(REPO_ROOT) else str(release_path),
        "release_manifest_sha256": sha256(release_path),
        "validation_protocol_file": str(protocol_path.relative_to(REPO_ROOT)) if protocol_path.is_relative_to(REPO_ROOT) else str(protocol_path),
        "validation_protocol_sha256": sha256(protocol_path),
        "finality_marker_file": str(FINALITY_MARKER.relative_to(REPO_ROOT)),
        "validation_lock_file": str(VALIDATION_LOCK.relative_to(REPO_ROOT)),
        "final_holdout_lock_file": str(FINAL_HOLDOUT_LOCK.relative_to(REPO_ROOT)),
    })
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_fresh_run(dataset: Path, release_manifest: Path, protocol: Path, confirm: str | None) -> Path:
    if confirm != CONFIRM_TOKEN:
        raise ValueError(f"Fresh validation requires --confirm-validation {CONFIRM_TOKEN}")
    if FINALITY_MARKER.exists():
        marker = load_json(FINALITY_MARKER)
        raise RuntimeError("Validation was already started. Resume only: " + str(marker.get("run_directory")))
    validate_validation_lock_pre_start(dataset)
    validate_final_holdout_untouched()
    VALIDATION_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = VALIDATION_RUNS_DIR / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    marker = {
        "release_candidate_id": EXPECTED_RELEASE_CANDIDATE_ID,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir.resolve()),
        "dataset_file": str(dataset.resolve()),
        "dataset_sha256": sha256(dataset),
        "release_manifest_file": str(release_manifest.resolve()),
        "release_manifest_sha256": sha256(release_manifest),
        "validation_protocol_file": str(protocol.resolve()),
        "validation_protocol_sha256": sha256(protocol),
        "one_time_only": True,
        "instruction": "Resume this same validation run after interruption. Do not start another validation run.",
    }
    write_json_atomic(FINALITY_MARKER, marker)
    transition_validation_lock_started(dataset, release_manifest, protocol, run_dir)
    return run_dir


def validate_resume_finality(run_dir: Path, dataset: Path, release_manifest: Path, protocol: Path) -> None:
    if not FINALITY_MARKER.exists():
        raise RuntimeError("Validation finality marker missing; refusing resume.")
    marker = load_json(FINALITY_MARKER)
    if marker.get("release_candidate_id") != EXPECTED_RELEASE_CANDIDATE_ID:
        raise RuntimeError("Validation marker belongs to a different release candidate.")
    if Path(str(marker.get("run_directory"))).resolve() != run_dir.resolve():
        raise RuntimeError("--resume does not match the recorded one-time validation run.")
    if Path(str(marker.get("dataset_file"))).resolve() != dataset.resolve() or marker.get("dataset_sha256") != sha256(dataset):
        raise RuntimeError("Validation dataset changed or path differs from start marker.")
    if marker.get("release_manifest_sha256") != sha256(release_manifest):
        raise RuntimeError("Release manifest changed after validation start.")
    if marker.get("validation_protocol_sha256") != sha256(protocol):
        raise RuntimeError("Validation protocol changed after validation start.")
    validate_final_holdout_untouched()
    transition_validation_lock_started(dataset, release_manifest, protocol, run_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run frozen v0.28.0 r4 on the one-time 30-case independent validation set.")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--release-manifest", default=str(DEFAULT_RELEASE_MANIFEST))
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    ap.add_argument("--confirm-validation", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--case-id", action="append", default=[], help="Operational recovery only; allowed only with --resume.")
    args = ap.parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.case_id and not args.resume:
        raise ValueError("Fresh targeted validation runs are forbidden. --case-id is recovery-only with --resume.")

    dataset = resolve(args.dataset)
    release_manifest = resolve(args.release_manifest)
    protocol = resolve(args.protocol)

    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if sha256(dataset) != EXPECTED_VALIDATION_SHA256:
        raise ValueError("Validation dataset hash mismatch.")
    validate_release_manifest(release_manifest)
    validate_protocol(protocol)
    validate_final_holdout_untouched()

    base.DATASET_FILE = dataset
    base.RUNS_DIR = VALIDATION_RUNS_DIR
    base.EVALUATION_DATASET_VERSION = EXPECTED_DATASET_VERSION
    base.EVALUATION_DATASET_ROLE = ROLE
    base.EVALUATION_SCOPE = SCOPE
    base.validate_inputs = validate_inputs
    base.validate_resume_metadata = validate_resume_metadata
    base.write_metadata = write_validation_metadata
    _RUNTIME.update({"dataset": dataset, "release_manifest": release_manifest, "protocol": protocol})

    if args.resume:
        run_dir = resolve(args.resume)
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        validate_resume_finality(run_dir, dataset, release_manifest, protocol)
    else:
        run_dir = prepare_fresh_run(dataset, release_manifest, protocol, args.confirm_validation)

    argv = ["run_judge_development.py", "--resume", str(run_dir)]
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
