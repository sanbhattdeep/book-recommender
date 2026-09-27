"""One-time final-holdout runner for frozen Semantic Relevance Judge v0.27.0 r7.

This wrapper changes only evaluation provenance, dataset validation, and finality
controls. Judge semantics, prompts, facets, and scoring remain the frozen r7
implementation imported from run_judge_development.py.

A fresh final-holdout run can be started exactly once. Any interruption must be
resumed in the same recorded run directory.
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

FINAL_RUNS_DIR = EVALS_DIR / "runs" / "semantic_relevance_v0_27_final_holdout"
FINALITY_MARKER = FINAL_RUNS_DIR / "FINAL_HOLDOUT_STARTED.json"
DEFAULT_RELEASE_MANIFEST = EVALS_DIR / "releases" / "semantic_relevance_v0.27.0_r7_release_candidate.json"
DEFAULT_DATASET = EVALS_DIR / "datasets" / "semantic_relevance_final_holdout.v3.0.0.DO_NOT_RUN_YET.csv"
FINAL_LOCK = EVALS_DIR / "datasets" / "semantic_relevance_final_holdout_lock.v3.0.0.json"
DEVELOPMENT_DATASET = EVALS_DIR / "datasets" / "semantic_relevance_v0.27_development.v3.1.0.csv"

CONFIRM_TOKEN = "FINAL_V0_27_R7"
EXPECTED_HOLDOUT_CASES = 30
EXPECTED_HOLDOUT_DATASET_VERSION = "3.0.0"
EXPECTED_HOLDOUT_SHA256 = "cce944bd48b102452c024134560b7b87866b85f5fa1bc1194775138b8976eb2e"
EXPECTED_RELEASE_MANIFEST_SHA256 = "5822551371fe21bd48cb5eb423f4c516b12770a75d7d11d5bb754ae5ca0dfbfc"
EXPECTED_RELEASE_CANDIDATE_ID = "semantic_relevance_v0.27.0-r7"
ROLE = "unseen_final_holdout"
SCOPE = "one_time_frozen_release_validation"

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
    return json.loads(path.read_text(encoding="utf-8"))


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
        raise FileNotFoundError(f"Frozen release manifest not found: {path}")
    actual_manifest_sha = sha256(path)
    if actual_manifest_sha != EXPECTED_RELEASE_MANIFEST_SHA256:
        raise ValueError(
            "Frozen release manifest changed after reviewed freeze: "
            f"{actual_manifest_sha} != {EXPECTED_RELEASE_MANIFEST_SHA256}"
        )
    m = load_json(path)
    if m.get("status") != "FROZEN_FOR_ONE_TIME_FINAL_HOLDOUT":
        raise ValueError("Release manifest is not frozen for final holdout.")
    if m.get("release_candidate_id") != EXPECTED_RELEASE_CANDIDATE_ID:
        raise ValueError("Unexpected release candidate ID.")
    holdout = m.get("independent_final_holdout", {})
    if holdout.get("sha256") != EXPECTED_HOLDOUT_SHA256:
        raise ValueError("Release manifest does not pin the expected v3.0.0 holdout hash.")
    if int(holdout.get("judge_run_count", -1)) != 0 or holdout.get("status") != "LOCKED_DO_NOT_RUN":
        raise ValueError("Frozen release manifest does not record an untouched final holdout.")
    protocol = m.get("final_holdout_protocol", {})
    if int(protocol.get("expected_cases", -1)) != EXPECTED_HOLDOUT_CASES:
        raise ValueError("Release manifest final-holdout case count mismatch.")
    if protocol.get("one_time_only") is not True or protocol.get("targeted_fresh_runs_forbidden") is not True:
        raise ValueError("Release manifest does not enforce one-time finality.")
    for rel, info in m.get("artifacts", {}).items():
        p = REPO_ROOT / rel
        if not p.exists():
            raise FileNotFoundError(p)
        actual = sha256(p)
        if actual != info.get("sha256"):
            raise ValueError(f"Frozen artifact changed after release freeze: {rel}")
    return m


def validate_lock_pre_start(dataset: Path) -> dict:
    if not FINAL_LOCK.exists():
        raise FileNotFoundError(FINAL_LOCK)
    lock = load_json(FINAL_LOCK)
    if lock.get("status") != "LOCKED_DO_NOT_RUN":
        raise ValueError(f"Final holdout lock status is {lock.get('status')!r}, expected LOCKED_DO_NOT_RUN")
    if int(lock.get("judge_run_count", -1)) != 0:
        raise ValueError(f"Final holdout judge_run_count={lock.get('judge_run_count')!r}, expected 0")
    if sha256(dataset) != EXPECTED_HOLDOUT_SHA256:
        raise ValueError("Final holdout dataset hash changed before start.")
    return lock


def transition_lock_started(dataset: Path, release_manifest: Path, run_dir: Path) -> None:
    lock = load_json(FINAL_LOCK)
    # Recovery path: a finality marker may have been written immediately before
    # an interrupted lock transition. Only permit that exact 0 -> 1 transition.
    if lock.get("status") == "LOCKED_DO_NOT_RUN" and int(lock.get("judge_run_count", -1)) == 0:
        lock.update({
            "status": "FINAL_HOLDOUT_STARTED",
            "judge_run_count": 1,
            "release_candidate_id": EXPECTED_RELEASE_CANDIDATE_ID,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_directory": str(run_dir.resolve()),
            "dataset_sha256": sha256(dataset),
            "release_manifest_sha256": sha256(release_manifest),
            "instruction": "Resume this same run after interruptions. Never start a second final-holdout run.",
        })
        write_json_atomic(FINAL_LOCK, lock)
        return
    if lock.get("status") == "FINAL_HOLDOUT_STARTED" and int(lock.get("judge_run_count", -1)) == 1:
        if Path(str(lock.get("run_directory"))).resolve() != run_dir.resolve():
            raise RuntimeError("Final holdout lock points to a different run directory.")
        if lock.get("dataset_sha256") != sha256(dataset):
            raise RuntimeError("Final holdout lock dataset hash mismatch.")
        return
    raise RuntimeError(
        f"Final holdout lock cannot transition/resume: status={lock.get('status')!r}, "
        f"judge_run_count={lock.get('judge_run_count')!r}"
    )


def validate_holdout_inputs(
    dataset: pd.DataFrame,
    rubric: dict[str, Any],
    config: dict[str, Any],
    facet_payload: dict[str, Any],
    facet_specs: dict[str, Any],
) -> None:
    required = {
        "case_id", "query_id", "query", "title", "authors", "description",
        "human_score", "human_reason", "review_status", "dataset_version", "rubric_version",
    }
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Final holdout missing required columns: {sorted(missing)}")
    if len(dataset) != EXPECTED_HOLDOUT_CASES:
        raise ValueError(f"Final holdout must contain exactly {EXPECTED_HOLDOUT_CASES} cases; found {len(dataset)}.")
    if dataset["case_id"].astype(str).duplicated().any():
        raise ValueError("Final holdout has duplicate case_id values.")
    if "isbn13" in dataset.columns:
        isbn = dataset["isbn13"].fillna("").astype(str).str.strip()
        nonblank = isbn[isbn.ne("") & isbn.ne("nan")]
        if nonblank.duplicated().any():
            raise ValueError("Final holdout has duplicate nonblank ISBNs.")
    scores = pd.to_numeric(dataset["human_score"], errors="coerce")
    if scores.isna().any() or not scores.isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Final holdout human_score must be complete and in 0..4.")
    if not dataset["human_reason"].fillna("").astype(str).str.strip().ne("").all():
        raise ValueError("Final holdout human_reason must be complete.")
    if not dataset["review_status"].astype(str).str.strip().eq("LABELLED").all():
        raise ValueError("Every final-holdout row must be LABELLED before judge execution.")
    if set(dataset["dataset_version"].astype(str)) != {EXPECTED_HOLDOUT_DATASET_VERSION}:
        raise ValueError(f"Final holdout rows must declare dataset_version={EXPECTED_HOLDOUT_DATASET_VERSION}.")
    if set(dataset["rubric_version"].astype(str)) != {base.RUBRIC_VERSION}:
        raise ValueError("Final holdout rubric version mismatch.")
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

    dataset_query_ids = set(dataset["query_id"].astype(str))
    if dataset_query_ids != {f"Q{i:02d}" for i in range(1, 13)}:
        raise ValueError(f"Final holdout must cover Q01-Q12; found {sorted(dataset_query_ids)}")
    missing_facets = dataset_query_ids - set(facet_specs)
    if missing_facets:
        raise ValueError(f"No frozen facet spec for: {sorted(missing_facets)}")
    for query_id, group in dataset.groupby("query_id"):
        texts = set(group["query"].astype(str))
        if len(texts) != 1:
            raise ValueError(f"Multiple query texts for {query_id}")
        text = next(iter(texts))
        if text != facet_specs[str(query_id)].query:
            raise ValueError(f"Query text mismatch for {query_id}.")

    if not DEVELOPMENT_DATASET.exists():
        raise FileNotFoundError("Frozen 150-case development dataset required for overlap guard.")
    dev = pd.read_csv(
        DEVELOPMENT_DATASET,
        dtype={"case_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    if set(dataset["case_id"].astype(str)) & set(dev["case_id"].astype(str)):
        raise ValueError("Final holdout overlaps development case IDs.")
    if "isbn13" in dataset.columns and "isbn13" in dev.columns:
        holdout_isbn = {x for x in dataset["isbn13"].fillna("").astype(str).str.strip() if x and x != "nan"}
        dev_isbn = {x for x in dev["isbn13"].fillna("").astype(str).str.strip() if x and x != "nan"}
        overlap = holdout_isbn & dev_isbn
        if overlap:
            raise ValueError(f"Final holdout overlaps development ISBNs: {sorted(overlap)[:5]}")
    holdout_title_author = {
        normalize_text(t) + "||" + normalize_text(a)
        for t, a in zip(dataset["title"], dataset["authors"])
        if normalize_text(t) or normalize_text(a)
    }
    dev_title_author = {
        normalize_text(t) + "||" + normalize_text(a)
        for t, a in zip(dev["title"], dev["authors"])
        if normalize_text(t) or normalize_text(a)
    }
    if holdout_title_author & dev_title_author:
        raise ValueError("Final holdout overlaps development title+author identities.")


def validate_resume_metadata(run_dir: Path) -> None:
    p = run_dir / "run_metadata.json"
    if not p.exists():
        return
    m = load_json(p)
    expected = {
        "judge_config_version": "0.27.0",
        "evaluation_dataset_version": EXPECTED_HOLDOUT_DATASET_VERSION,
        "evaluation_dataset_role": ROLE,
        "evaluation_scope": SCOPE,
        "judge_behavior_frozen": True,
        "final_holdout_one_time": True,
        "rubric_version": "0.1.0",
        "facet_spec_version": "0.9.7",
    }
    mismatches = [
        f"{k}: run={m.get(k)!r}, expected={v!r}"
        for k, v in expected.items()
        if str(m.get(k)) != str(v)
    ]
    if mismatches:
        raise ValueError("Cannot resume final holdout due provenance mismatch:\n- " + "\n- ".join(mismatches))


def write_final_metadata(run_dir: Path, config: dict[str, Any], status: str, completed_cases: int, total_cases: int) -> None:
    _original_write_metadata(run_dir, config, status, completed_cases, total_cases)
    p = run_dir / "run_metadata.json"
    m = load_json(p)
    release_path = Path(str(_RUNTIME["release_manifest"]))
    dataset_path = Path(str(_RUNTIME["dataset"]))
    m.update({
        "evaluation_dataset_version": EXPECTED_HOLDOUT_DATASET_VERSION,
        "evaluation_dataset_role": ROLE,
        "evaluation_scope": SCOPE,
        "source_splits_consumed": False,
        "development_dataset_post_holdout": False,
        "judge_behavior_frozen": True,
        "final_holdout_one_time": True,
        "evaluation_dataset_file": str(dataset_path.relative_to(REPO_ROOT)) if dataset_path.is_relative_to(REPO_ROOT) else str(dataset_path),
        "evaluation_dataset_sha256": sha256(dataset_path),
        "release_manifest_file": str(release_path.relative_to(REPO_ROOT)) if release_path.is_relative_to(REPO_ROOT) else str(release_path),
        "release_manifest_sha256": sha256(release_path),
        "finality_marker_file": str(FINALITY_MARKER.relative_to(REPO_ROOT)),
        "final_holdout_lock_file": str(FINAL_LOCK.relative_to(REPO_ROOT)),
    })
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_fresh_run(dataset: Path, release_manifest: Path, confirm: str | None) -> Path:
    if confirm != CONFIRM_TOKEN:
        raise ValueError(f"Fresh final holdout requires --confirm-final-holdout {CONFIRM_TOKEN}")
    if FINALITY_MARKER.exists():
        marker = load_json(FINALITY_MARKER)
        raise RuntimeError(
            "Final holdout was already started. Resume the recorded run only: "
            f"{marker.get('run_directory')}"
        )
    validate_lock_pre_start(dataset)
    FINAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
        "one_time_only": True,
        "instruction": "Resume this same run after interruptions. Do not start a second final-holdout run.",
    }
    # Marker is deliberately persisted before the first judge call.
    write_json_atomic(FINALITY_MARKER, marker)
    transition_lock_started(dataset, release_manifest, run_dir)
    return run_dir


def validate_resume_finality(run_dir: Path, dataset: Path, release_manifest: Path) -> None:
    if not FINALITY_MARKER.exists():
        raise RuntimeError("Finality marker missing; refusing resume.")
    m = load_json(FINALITY_MARKER)
    if m.get("release_candidate_id") != EXPECTED_RELEASE_CANDIDATE_ID:
        raise RuntimeError("Finality marker belongs to a different release candidate.")
    if Path(str(m.get("run_directory"))).resolve() != run_dir.resolve():
        raise RuntimeError("--resume does not match the one recorded final-holdout run.")
    if Path(str(m.get("dataset_file"))).resolve() != dataset.resolve() or m.get("dataset_sha256") != sha256(dataset):
        raise RuntimeError("Final-holdout dataset changed or path differs from the frozen start marker.")
    if m.get("release_manifest_sha256") != sha256(release_manifest):
        raise RuntimeError("Release manifest changed after final-holdout start.")
    transition_lock_started(dataset, release_manifest, run_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run frozen v0.27.0 r7 on the one-time 30-case final holdout.")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to the untouched 30-case v3.0.0 final-holdout CSV.")
    ap.add_argument("--release-manifest", default=str(DEFAULT_RELEASE_MANIFEST))
    ap.add_argument("--confirm-final-holdout", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--case-id", action="append", default=[], help="Operational recovery only; allowed only with --resume.")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.case_id and not args.resume:
        raise ValueError(
            "Targeted fresh holdout runs are forbidden. --case-id is allowed only while resuming "
            "the single recorded run for operational recovery."
        )

    dataset = resolve(args.dataset)
    release_manifest = resolve(args.release_manifest)
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if sha256(dataset) != EXPECTED_HOLDOUT_SHA256:
        raise ValueError(f"Final holdout hash mismatch: {sha256(dataset)} != {EXPECTED_HOLDOUT_SHA256}")
    validate_release_manifest(release_manifest)

    # Patch only provenance/dataset plumbing. Judge semantics/scoring remain frozen r7.
    base.DATASET_FILE = dataset
    base.RUNS_DIR = FINAL_RUNS_DIR
    base.EVALUATION_DATASET_VERSION = EXPECTED_HOLDOUT_DATASET_VERSION
    base.EVALUATION_DATASET_ROLE = ROLE
    base.EVALUATION_SCOPE = SCOPE
    base.validate_inputs = validate_holdout_inputs
    base.validate_resume_metadata = validate_resume_metadata
    base.write_metadata = write_final_metadata
    _RUNTIME.update({"dataset": dataset, "release_manifest": release_manifest})

    if args.resume:
        run_dir = resolve(args.resume)
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        validate_resume_finality(run_dir, dataset, release_manifest)
    else:
        run_dir = prepare_fresh_run(dataset, release_manifest, args.confirm_final_holdout)

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
