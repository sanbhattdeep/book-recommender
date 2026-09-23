"""One-time final-holdout runner for frozen Semantic Relevance Judge v0.26.0 r6.

The semantic pipeline is imported unchanged from run_judge_development.py. This
wrapper changes only evaluation provenance, dataset validation, and finality
controls. A fresh final-holdout run can be started once; interruptions must be
resumed in the same recorded run directory.
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
import run_judge_development as base  # noqa: E402

FINAL_RUNS_DIR = EVALS_DIR / "runs" / "semantic_relevance_v0_26_final_holdout"
FINALITY_MARKER = FINAL_RUNS_DIR / "FINAL_HOLDOUT_STARTED.json"
DEFAULT_RELEASE_MANIFEST = EVALS_DIR / "releases" / "semantic_relevance_v0.26.0_r6_release_candidate.json"
CONFIRM_TOKEN = "FINAL_V0_26_R6"
EXPECTED_HOLDOUT_CASES = 30
EXPECTED_HOLDOUT_DATASET_VERSION = "2.0.0"
ROLE = "unseen_final_holdout"
SCOPE = "one_time_frozen_release_validation"

_original_write_metadata = base.write_metadata

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def resolve(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute(): p = REPO_ROOT / p
    return p.resolve()

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def validate_release_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Frozen release manifest not found: {path}")
    m = load_json(path)
    if m.get("status") != "FROZEN_FOR_ONE_TIME_FINAL_HOLDOUT":
        raise ValueError("Release manifest is not frozen for final holdout.")
    if m.get("release_candidate_id") != "semantic_relevance_v0.26.0-r6":
        raise ValueError("Unexpected release candidate ID.")
    for rel, info in m.get("artifacts", {}).items():
        p = REPO_ROOT / rel
        if not p.exists(): raise FileNotFoundError(p)
        actual = sha256(p)
        if actual != info.get("sha256"):
            raise ValueError(f"Frozen artifact changed after release freeze: {rel}")
    return m

def validate_holdout_inputs(dataset: pd.DataFrame, rubric: dict[str, Any], config: dict[str, Any], facet_payload: dict[str, Any], facet_specs: dict[str, Any]) -> None:
    required = {"case_id","query_id","query","title","authors","description","human_score","human_reason","review_status","dataset_version","rubric_version"}
    missing = required - set(dataset.columns)
    if missing: raise ValueError(f"Final holdout missing required columns: {sorted(missing)}")
    if len(dataset) != EXPECTED_HOLDOUT_CASES:
        raise ValueError(f"Final holdout must contain exactly {EXPECTED_HOLDOUT_CASES} cases; found {len(dataset)}.")
    if dataset["case_id"].astype(str).duplicated().any(): raise ValueError("Final holdout has duplicate case_id values.")
    if "isbn13" in dataset.columns and dataset["isbn13"].astype(str).duplicated().any(): raise ValueError("Final holdout has duplicate ISBNs.")
    if not dataset["human_score"].notna().all() or not dataset["human_score"].isin([0,1,2,3,4]).all(): raise ValueError("Final holdout human_score must be complete and in 0..4.")
    if not dataset["review_status"].astype(str).eq("LABELLED").all(): raise ValueError("Every final-holdout row must be LABELLED before judge execution.")
    if set(dataset["dataset_version"].astype(str)) != {EXPECTED_HOLDOUT_DATASET_VERSION}: raise ValueError(f"Final holdout rows must declare dataset_version={EXPECTED_HOLDOUT_DATASET_VERSION}.")
    if set(dataset["rubric_version"].astype(str)) != {base.RUBRIC_VERSION}: raise ValueError("Final holdout rubric version mismatch.")
    if str(rubric.get("version")) != base.RUBRIC_VERSION: raise ValueError("Rubric file version mismatch.")
    if str(config.get("version")) != base.JUDGE_CONFIG_VERSION: raise ValueError("Judge config version mismatch.")
    if str(config.get("dataset_version")) != base.JUDGE_CALIBRATION_DATASET_VERSION: raise ValueError("Judge calibration provenance mismatch.")
    if str(config.get("rubric_version")) != base.RUBRIC_VERSION or str(config.get("facet_spec_version")) != base.FACET_SPEC_VERSION: raise ValueError("Judge config artifact version mismatch.")
    if config.get("provider") != "ollama": raise ValueError("Judge provider must be ollama.")
    if str(facet_payload.get("version")) != base.FACET_SPEC_VERSION or facet_payload.get("status") != "frozen": raise ValueError("Frozen facet spec mismatch.")

    dataset_query_ids = set(dataset["query_id"].astype(str))
    if dataset_query_ids != {f"Q{i:02d}" for i in range(1,13)}:
        raise ValueError(f"Final holdout must cover Q01-Q12; found {sorted(dataset_query_ids)}")
    missing_facets = dataset_query_ids - set(facet_specs)
    if missing_facets: raise ValueError(f"No frozen facet spec for: {sorted(missing_facets)}")
    for query_id, group in dataset.groupby("query_id"):
        texts = set(group["query"].astype(str))
        if len(texts) != 1: raise ValueError(f"Multiple query texts for {query_id}")
        text = next(iter(texts))
        if text != facet_specs[str(query_id)].query:
            raise ValueError(f"Query text mismatch for {query_id}.")

    dev_file = EVALS_DIR / "datasets" / "semantic_relevance_v0.26_development.v2.2.0.csv"
    if not dev_file.exists(): raise FileNotFoundError("Development dataset required for final-holdout overlap guard.")
    dev = pd.read_csv(dev_file, dtype={"case_id":str,"isbn13":str}, encoding="utf-8")
    if set(dataset["case_id"].astype(str)) & set(dev["case_id"].astype(str)):
        raise ValueError("Final holdout overlaps development case IDs.")
    if "isbn13" in dataset.columns and "isbn13" in dev.columns:
        overlap = set(dataset["isbn13"].astype(str)) & set(dev["isbn13"].astype(str))
        overlap.discard("")
        overlap.discard("nan")
        if overlap: raise ValueError(f"Final holdout overlaps development ISBNs: {sorted(overlap)[:5]}")

def validate_resume_metadata(run_dir: Path) -> None:
    p = run_dir / "run_metadata.json"
    if not p.exists(): return
    m = load_json(p)
    expected = {
        "judge_config_version":"0.26.0",
        "evaluation_dataset_version":EXPECTED_HOLDOUT_DATASET_VERSION,
        "evaluation_dataset_role":ROLE,
        "evaluation_scope":SCOPE,
        "judge_behavior_frozen":True,
        "final_holdout_one_time":True,
        "rubric_version":"0.1.0",
        "facet_spec_version":"0.9.3",
    }
    mismatches=[f"{k}: run={m.get(k)!r}, expected={v!r}" for k,v in expected.items() if str(m.get(k)) != str(v)]
    if mismatches: raise ValueError("Cannot resume final holdout due provenance mismatch:\n- " + "\n- ".join(mismatches))

def write_final_metadata(run_dir: Path, config: dict[str, Any], status: str, completed_cases: int, total_cases: int) -> None:
    _original_write_metadata(run_dir, config, status, completed_cases, total_cases)
    p = run_dir / "run_metadata.json"
    m = load_json(p)
    release_path = Path(str(_RUNTIME["release_manifest"]))
    m.update({
        "evaluation_dataset_version": EXPECTED_HOLDOUT_DATASET_VERSION,
        "evaluation_dataset_role": ROLE,
        "evaluation_scope": SCOPE,
        "source_splits_consumed": False,
        "development_dataset_post_holdout": False,
        "judge_behavior_frozen": True,
        "final_holdout_one_time": True,
        "evaluation_dataset_file": str(Path(str(_RUNTIME["dataset"])).relative_to(REPO_ROOT)) if Path(str(_RUNTIME["dataset"])).is_relative_to(REPO_ROOT) else str(_RUNTIME["dataset"]),
        "evaluation_dataset_sha256": sha256(Path(str(_RUNTIME["dataset"]))),
        "release_manifest_file": str(release_path.relative_to(REPO_ROOT)) if release_path.is_relative_to(REPO_ROOT) else str(release_path),
        "release_manifest_sha256": sha256(release_path),
        "finality_marker_file": str(FINALITY_MARKER.relative_to(REPO_ROOT)),
    })
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

_RUNTIME: dict[str, Any] = {}

def prepare_fresh_run(dataset: Path, release_manifest: Path, confirm: str) -> Path:
    if confirm != CONFIRM_TOKEN:
        raise ValueError(f"Fresh final holdout requires --confirm-final-holdout {CONFIRM_TOKEN}")
    if FINALITY_MARKER.exists():
        marker=load_json(FINALITY_MARKER)
        raise RuntimeError(f"Final holdout was already started. Resume the recorded run only: {marker.get('run_directory')}")
    FINAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir=FINAL_RUNS_DIR/run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    marker={
        "release_candidate_id":"semantic_relevance_v0.26.0-r6",
        "started_at_utc":datetime.now(timezone.utc).isoformat(),
        "run_directory":str(run_dir.resolve()),
        "dataset_file":str(dataset.resolve()),
        "dataset_sha256":sha256(dataset),
        "release_manifest_file":str(release_manifest.resolve()),
        "release_manifest_sha256":sha256(release_manifest),
        "one_time_only":True,
        "instruction":"Resume this same run after interruptions. Do not start a second final-holdout run.",
    }
    FINALITY_MARKER.write_text(json.dumps(marker,indent=2)+"\n",encoding="utf-8")
    return run_dir

def validate_resume_finality(run_dir: Path, dataset: Path, release_manifest: Path) -> None:
    if not FINALITY_MARKER.exists(): raise RuntimeError("Finality marker missing; refusing resume.")
    m=load_json(FINALITY_MARKER)
    if Path(str(m.get("run_directory"))).resolve() != run_dir.resolve(): raise RuntimeError("--resume does not match the one recorded final-holdout run.")
    if Path(str(m.get("dataset_file"))).resolve() != dataset.resolve() or m.get("dataset_sha256") != sha256(dataset): raise RuntimeError("Final-holdout dataset changed or path differs from the frozen start marker.")
    if m.get("release_manifest_sha256") != sha256(release_manifest): raise RuntimeError("Release manifest changed after final-holdout start.")

def main() -> None:
    ap=argparse.ArgumentParser(description="Run frozen v0.26.0 r6 on the one-time 30-case final holdout.")
    ap.add_argument("--dataset", required=True, help="Path to the untouched 30-case final-holdout CSV.")
    ap.add_argument("--release-manifest", default=str(DEFAULT_RELEASE_MANIFEST))
    ap.add_argument("--confirm-final-holdout", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--case-id", action="append", default=[], help="Operational recovery only; allowed only with --resume.")
    args=ap.parse_args()
    if args.limit is not None and args.limit < 1: raise ValueError("--limit must be >= 1")
    if args.case_id and not args.resume: raise ValueError("Targeted fresh holdout runs are forbidden. --case-id is allowed only while resuming the single recorded run for operational recovery.")

    dataset=resolve(args.dataset)
    release_manifest=resolve(args.release_manifest)
    if not dataset.exists(): raise FileNotFoundError(dataset)
    validate_release_manifest(release_manifest)

    # Patch only provenance/dataset plumbing. Judge semantics/scoring remain the frozen r6 code.
    base.DATASET_FILE=dataset
    base.RUNS_DIR=FINAL_RUNS_DIR
    base.EVALUATION_DATASET_VERSION=EXPECTED_HOLDOUT_DATASET_VERSION
    base.EVALUATION_DATASET_ROLE=ROLE
    base.EVALUATION_SCOPE=SCOPE
    base.validate_inputs=validate_holdout_inputs
    base.validate_resume_metadata=validate_resume_metadata
    base.write_metadata=write_final_metadata
    _RUNTIME.update({"dataset":dataset,"release_manifest":release_manifest})

    if args.resume:
        run_dir=resolve(args.resume)
        if not run_dir.exists(): raise FileNotFoundError(run_dir)
        validate_resume_finality(run_dir,dataset,release_manifest)
    else:
        run_dir=prepare_fresh_run(dataset,release_manifest,args.confirm_final_holdout)

    argv=["run_judge_development.py","--resume",str(run_dir)]
    if args.limit is not None: argv += ["--limit",str(args.limit)]
    for cid in args.case_id: argv += ["--case-id",str(cid)]
    old=sys.argv[:]
    try:
        sys.argv=argv
        base.main()
    finally:
        sys.argv=old

if __name__ == "__main__":
    main()
