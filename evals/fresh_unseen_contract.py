"""Shared integrity/provenance checks for v0.18 fresh-unseen evaluation."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"
DATASETS_DIR = EVALS_DIR / "datasets"
FRESH_DIR = EVALS_DIR / "fresh_unseen"
PROTOCOL_FILE = FRESH_DIR / "fresh_unseen_protocol.v2.0.0.json"

POOL_FILE = DATASETS_DIR / "semantic_relevance_fresh_unseen_pool.v2.0.0.csv"
LABELLED_POOL_FILE = DATASETS_DIR / "semantic_relevance_fresh_unseen_pool.v2.0.0.labelled.csv"
VALIDATION_FILE = DATASETS_DIR / "semantic_relevance_fresh_validation.v2.0.0.csv"
HOLDOUT_FILE = DATASETS_DIR / "semantic_relevance_fresh_holdout.v2.0.0.DO_NOT_RUN_YET.csv"
SPLIT_MANIFEST_FILE = DATASETS_DIR / "semantic_relevance_fresh_unseen_split_manifest.v2.0.0.json"
RELEASE_FILE = DATASETS_DIR / "semantic_relevance_fresh_holdout_release.v2.0.0.json"
EXCLUSION_JSON = DATASETS_DIR / "semantic_relevance_exclusion_manifest.v2.0.0.json"
EXCLUSION_CSV = DATASETS_DIR / "semantic_relevance_exclusion_manifest.v2.0.0.csv"

JUDGE_CONFIG_FILE = EVALS_DIR / "judge_configs" / "semantic_relevance_judge.v0.18.0.json"
FACET_SPEC_FILE = EVALS_DIR / "facets" / "semantic_relevance" / "semantic_relevance_query_facets.v0.6.0.json"
RUBRIC_FILE = EVALS_DIR / "rubrics" / "semantic_relevance" / "semantic_relevance_rubric.v0.1.0.json"
SCORING_FILE = EVALS_DIR / "semantic_relevance_facet_scoring.py"
JUDGE_FILE = EVALS_DIR / "semantic_relevance_facet_judge.py"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_isbn(value: Any) -> str:
    raw = str(value).strip()
    if re.fullmatch(r"\d+\.0", raw):
        raw = raw[:-2]
    if not re.fullmatch(r"\d{13}", raw):
        raise ValueError(f"Expected 13-digit ISBN, got {value!r}")
    return raw


def verify_behavior_hashes() -> None:
    protocol = load_json(PROTOCOL_FILE)
    for rel, expected in protocol["behavior_file_sha256"].items():
        path = REPO_ROOT / rel
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Frozen v0.18 behavior artifact changed: {rel}\n"
                f"expected={expected}\nactual={actual}"
            )


def verify_exclusion_manifest() -> set[str]:
    protocol = load_json(PROTOCOL_FILE)
    if sha256_file(EXCLUSION_JSON) != protocol["exclusion_manifest_json_sha256"]:
        raise ValueError("Exclusion JSON hash does not match the frozen protocol.")
    if sha256_file(EXCLUSION_CSV) != protocol["exclusion_manifest_csv_sha256"]:
        raise ValueError("Exclusion CSV hash does not match the frozen protocol.")
    payload = load_json(EXCLUSION_JSON)
    isbns = {canonical_isbn(x) for x in payload["excluded_isbns"]}
    if len(isbns) != int(protocol["distinct_isbns_excluded"]):
        raise ValueError("Frozen exclusion ISBN count mismatch.")
    return isbns


def validate_pool_dataframe(df: pd.DataFrame, *, require_labels: bool) -> None:
    protocol = load_json(PROTOCOL_FILE)
    required = {
        "case_id","query_id","query","query_slice","candidate_source",
        "sampling_bucket","sampling_target_rank","retrieval_rank","isbn13",
        "title","authors","description","dataset_version","rubric_version",
        "human_score","human_reason","review_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pool missing required columns: {sorted(missing)}")
    if len(df) != int(protocol["pool_cases"]):
        raise ValueError(f"Expected 60 pool rows, found {len(df)}.")
    if df["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs in pool.")
    isbns = [canonical_isbn(x) for x in df["isbn13"]]
    if len(isbns) != len(set(isbns)):
        raise ValueError("Duplicate ISBNs in fresh pool.")
    overlap = set(isbns) & verify_exclusion_manifest()
    if overlap:
        raise ValueError(f"Fresh pool overlaps historical ISBN exclusions: {sorted(overlap)}")
    if set(df["dataset_version"].astype(str)) != {str(protocol["pool_version"])}:
        raise ValueError("Pool dataset_version mismatch.")
    if set(df["rubric_version"].astype(str)) != {str(protocol["rubric_version"])}:
        raise ValueError("Pool rubric_version mismatch.")
    qcounts = df["query_id"].astype(str).value_counts().to_dict()
    if set(qcounts.values()) != {5} or len(qcounts) != 12:
        raise ValueError(f"Expected 12 queries x 5 cases; got {qcounts}")
    bucket_counts = df["sampling_bucket"].astype(str).value_counts().to_dict()
    expected_buckets = {"T02":12,"T10":12,"T30":12,"T70":12,"NEG":12}
    if bucket_counts != expected_buckets:
        raise ValueError(f"Unexpected sampling buckets: {bucket_counts}")
    if require_labels:
        if not df["human_score"].notna().all():
            raise ValueError("Labelled pool has missing human_score values.")
        scores = pd.to_numeric(df["human_score"], errors="raise").astype(int)
        if not scores.isin([0,1,2,3,4]).all():
            raise ValueError("human_score must be 0..4.")
        if not df["human_reason"].fillna("").astype(str).str.strip().ne("").all():
            raise ValueError("Every labelled case requires human_reason.")
        if set(df["review_status"].astype(str)) != {"LABELLED"}:
            raise ValueError("All labelled rows must have review_status=LABELLED.")
    else:
        if df["human_score"].notna().any():
            raise ValueError("Unlabelled pool unexpectedly contains human_score values.")
        if df["human_reason"].fillna("").astype(str).str.strip().ne("").any():
            raise ValueError("Unlabelled pool unexpectedly contains human_reason values.")
        if set(df["review_status"].astype(str)) != {"UNLABELLED"}:
            raise ValueError("Fresh pool must begin review_status=UNLABELLED.")


def validate_split_contract(*, require_release: bool = False) -> dict[str, Any]:
    verify_behavior_hashes()
    verify_exclusion_manifest()
    manifest = load_json(SPLIT_MANIFEST_FILE)
    if manifest.get("split_created_before_judge_outputs") is not True:
        raise ValueError("Split manifest does not certify pre-judge splitting.")
    if manifest.get("judge_outputs_used_for_split") is not False:
        raise ValueError("Split manifest indicates judge outputs influenced the split.")
    labelled = pd.read_csv(LABELLED_POOL_FILE, dtype={"isbn13": str}, encoding="utf-8")
    validate_pool_dataframe(labelled, require_labels=True)
    val = pd.read_csv(VALIDATION_FILE, dtype={"isbn13": str}, encoding="utf-8")
    hold = pd.read_csv(HOLDOUT_FILE, dtype={"isbn13": str}, encoding="utf-8")
    if len(val) != 30 or len(hold) != 30:
        raise ValueError("Validation and holdout must each contain exactly 30 cases.")
    val_ids=set(val["case_id"].astype(str)); hold_ids=set(hold["case_id"].astype(str))
    if val_ids & hold_ids:
        raise ValueError("Validation/holdout case overlap detected.")
    if val_ids | hold_ids != set(labelled["case_id"].astype(str)):
        raise ValueError("Split membership does not exactly cover the labelled 60-case pool.")
    if sha256_file(LABELLED_POOL_FILE) != manifest["labelled_pool_sha256"]:
        raise ValueError("Labelled pool hash changed after split.")
    if sha256_file(VALIDATION_FILE) != manifest["validation_sha256"]:
        raise ValueError("Validation file hash changed after split.")
    if sha256_file(HOLDOUT_FILE) != manifest["holdout_sha256"]:
        raise ValueError("Holdout file hash changed after split.")
    if require_release:
        release = load_json(RELEASE_FILE)
        if release.get("validation_passed_all_pre_registered_gates") is not True:
            raise ValueError("Holdout release does not certify validation PASS.")
        if release.get("judge_config_version") != "0.18.0":
            raise ValueError("Holdout release judge version mismatch.")
        if release.get("split_manifest_sha256") != sha256_file(SPLIT_MANIFEST_FILE):
            raise ValueError("Holdout release refers to a different split manifest.")
        if release.get("holdout_sha256") != sha256_file(HOLDOUT_FILE):
            raise ValueError("Holdout release refers to a different holdout file.")
    return manifest
