"""Build the 210-case v0.28.0 r5 consumed-development dataset.

Composition:
- 180 already-consumed v0.28 development cases (U_/U2_/U3_)
- 30 cases from the one-time r4 independent validation (U4_)
- one documented post-validation human-label adjudication: U4_Q12_T05 0 -> 1

The remaining 30-case U4 final holdout is NEVER included and must remain
LOCKED_DO_NOT_RUN with judge_run_count=0.
"""
from __future__ import annotations

import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "evals" / "datasets"

OLD_DEV = D / "semantic_relevance_v0.28_development.v4.0.0.csv"
VALIDATION = D / "semantic_relevance_validation.v4.0.0.DO_NOT_RUN_YET.csv"
VALIDATION_LOCK = D / "semantic_relevance_validation_lock.v4.0.0.json"
FINAL_HOLDOUT = D / "semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv"
FINAL_HOLDOUT_LOCK = D / "semantic_relevance_final_holdout_lock.v4.0.0.json"
ADJUDICATION = D / "semantic_relevance_v0.28_post_validation_label_adjudications.v1.0.0.json"

OUT = D / "semantic_relevance_v0.28_development.v5.0.0.csv"
MANIFEST = D / "semantic_relevance_v0.28_development_manifest.v2.0.0.json"

EXPECTED_OLD_DEV_SHA = "fdf9ccbd8d131fa80c1e994681802b7f99050f497a60e503188c3e9ae7a3b83b"
EXPECTED_VALIDATION_SHA = "41b548363a984fd8f1a923e23fbb8b1db2c9b6bd3e00c9d43102ecca0c47672f"
EXPECTED_FINAL_HOLDOUT_SHA = "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> tuple[list[str], list[dict[str,str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print("PASS ", msg)


def main() -> None:
    for p in [OLD_DEV, VALIDATION, VALIDATION_LOCK, FINAL_HOLDOUT, FINAL_HOLDOUT_LOCK, ADJUDICATION]:
        require(p.exists(), f"required source exists: {p.relative_to(ROOT)}")

    require(sha256(OLD_DEV) == EXPECTED_OLD_DEV_SHA, "180-case r4 consumed-development hash pinned")
    require(sha256(VALIDATION) == EXPECTED_VALIDATION_SHA, "consumed r4 validation hash pinned")
    require(sha256(FINAL_HOLDOUT) == EXPECTED_FINAL_HOLDOUT_SHA, "untouched final-holdout hash pinned")

    vlock = load_json(VALIDATION_LOCK)
    require(vlock.get("status") == "INDEPENDENT_VALIDATION_COMPLETED", "r4 validation status = INDEPENDENT_VALIDATION_COMPLETED")
    require(int(vlock.get("judge_run_count", -1)) == 1, "r4 validation judge_run_count = 1")
    require(vlock.get("validation_decision") == "REVIEW_STOP_FINAL_HOLDOUT", "r4 validation decision = REVIEW_STOP_FINAL_HOLDOUT")

    hlock = load_json(FINAL_HOLDOUT_LOCK)
    require(hlock.get("status") == "LOCKED_DO_NOT_RUN", "independent final holdout remains LOCKED_DO_NOT_RUN")
    require(int(hlock.get("judge_run_count", -1)) == 0, "independent final holdout judge_run_count = 0")

    old_fields, old_rows = load_csv(OLD_DEV)
    val_fields, val_rows = load_csv(VALIDATION)
    require(len(old_rows) == 180, "source r4 development = 180 cases")
    require(len(val_rows) == 30, "consumed r4 validation = 30 cases")

    old_ids = {r["case_id"] for r in old_rows}
    val_ids = {r["case_id"] for r in val_rows}
    require(old_ids.isdisjoint(val_ids), "180/30 case IDs do not overlap")

    adj = load_json(ADJUDICATION)
    changes = {x["case_id"]:x for x in adj["adjudications"]}
    for row in val_rows:
        change = changes.get(row["case_id"])
        if change:
            require(int(row["human_score"]) == int(change["original_human_score"]), f"{row['case_id']} original score matches adjudication manifest")
            row["human_score"] = str(change["adjudicated_human_score"])
            row["human_reason"] = change["adjudicated_human_reason"]

    rows = old_rows + val_rows
    require(len(rows) == 210, "output development = 210 cases")
    require(len({r["case_id"] for r in rows}) == 210, "all 210 case IDs unique")

    counts = {
        "U_": sum(r["case_id"].startswith("U_") for r in rows),
        "U2_": sum(r["case_id"].startswith("U2_") for r in rows),
        "U3_": sum(r["case_id"].startswith("U3_") for r in rows),
        "U4_": sum(r["case_id"].startswith("U4_") for r in rows),
    }
    require(counts == {"U_":60,"U2_":60,"U3_":60,"U4_":30}, f"prefix counts = {counts}")

    fields = old_fields[:]
    for f in val_fields:
        if f not in fields:
            fields.append(f)
    for row in rows:
        row["dataset_version"] = "5.0.0"

    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k:row.get(k,"") for k in fields})

    out_sha = sha256(OUT)
    manifest = {
        "manifest_version":"2.0.0",
        "dataset_version":"5.0.0",
        "evidence_role":"consumed_development_only",
        "cases":210,
        "composition":{
            "prior_consumed_development":180,
            "consumed_r4_independent_validation":30,
            "untouched_final_holdout_included":False,
        },
        "post_validation_adjudications":["U4_Q12_T05"],
        "prefix_counts":counts,
        "source_hashes":{
            str(OLD_DEV.relative_to(ROOT)):EXPECTED_OLD_DEV_SHA,
            str(VALIDATION.relative_to(ROOT)):EXPECTED_VALIDATION_SHA,
            str(FINAL_HOLDOUT.relative_to(ROOT)):EXPECTED_FINAL_HOLDOUT_SHA,
            str(ADJUDICATION.relative_to(ROOT)):sha256(ADJUDICATION),
        },
        "output_file":str(OUT.relative_to(ROOT)),
        "output_sha256":out_sha,
        "methodology_note":(
            "The r4 validation is consumed and may be used only as r5 development/regression evidence. "
            "Its original r4 result is immutable. The remaining 30-case independent final holdout is excluded "
            "and remains locked/unseen."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("v0.28 r5 consumed-development dataset built")
    print("-----------------------------------------")
    print("Output SHA-256:", out_sha)
    print("Output:", OUT)
    print("Manifest:", MANIFEST)
    print("NOTE  final holdout remains excluded and untouched.")


if __name__ == "__main__":
    main()
