"""Bootstrap v0.29 development evidence from fully consumed v0.28 evidence.

NO judge calls. NO label changes.

Creates:
- 240-case v0.29 development dataset = v0.28 consumed-development 210
  + consumed v0.28 final-holdout 30.
- evidence-boundary manifest proving the former final holdout is now
  development/diagnostic evidence only for v0.29+.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1]
E = R / "evals"
D = E / "datasets"
REL = E / "releases"

DEV210 = D / "semantic_relevance_v0.28_development.v5.0.0.csv"
FINAL30 = D / "semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv"
FINAL_LOCK = D / "semantic_relevance_final_holdout_lock.v4.0.0.json"
CLOSEOUT = REL / "semantic_relevance_v0.28.0_r8_final_evaluation_closeout.json"

OUT = D / "semantic_relevance_v0.29_development.v6.0.0.csv"
MANIFEST = D / "semantic_relevance_v0.29_development_manifest.v1.0.0.json"
BOUNDARY = D / "semantic_relevance_v0.29_evidence_boundary.v1.0.0.json"

EXPECTED_CLOSEOUT_SHA = "24f90865503f4572d54e168d7f2cca33647a7c9a3c294bfb1ab727fd544cb2de"
EXPECTED_DEV210_SHA = "21e53ee575fab0f99e69ffabee705f8afcc843878a33d640d6ada795be1ea520"
EXPECTED_FINAL30_SHA = "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print("PASS ", msg)


def main() -> None:
    print("v0.29.0 consumed-development bootstrap")
    print("--------------------------------------")
    print("JUDGE EXECUTION: DISABLED")
    print("LABEL CHANGES: NONE")

    for p in [DEV210, FINAL30, FINAL_LOCK, CLOSEOUT]:
        require(p.exists(), f"required source exists: {p.relative_to(R)}")

    require(sha(CLOSEOUT) == EXPECTED_CLOSEOUT_SHA,
            "v0.28 r8 closeout manifest hash pinned")
    require(sha(DEV210) == EXPECTED_DEV210_SHA,
            "v0.28 210-case consumed-development hash pinned")
    require(sha(FINAL30) == EXPECTED_FINAL30_SHA,
            "consumed v0.28 final-holdout dataset hash pinned")

    closeout = load_json(CLOSEOUT)
    require(closeout.get("closeout_status") == "EVALUATED_NOT_FINAL_QUALIFIED",
            "r8 closeout status = EVALUATED_NOT_FINAL_QUALIFIED")
    require(closeout.get("final_holdout_decision") == "FINAL_HOLDOUT_REVIEW",
            "r8 final decision = FINAL_HOLDOUT_REVIEW")

    boundary = closeout["evidence_boundary_for_next_lineage"]
    require(boundary.get("next_candidate_lineage") == "semantic_relevance_v0.29.0",
            "closeout authorizes next lineage v0.29.0")
    require(boundary.get("r8_final_holdout_may_be_reused_as_independent_evidence") is False,
            "r8 final 30 cannot be reused as independent evidence")
    require(boundary.get("v0_29_may_use_r8_final_cases_for_development_regression") is True,
            "r8 final 30 may be used as v0.29 development regression evidence")

    lock = load_json(FINAL_LOCK)
    require(lock.get("status") == "FINAL_HOLDOUT_COMPLETED",
            "v0.28 final holdout remains FINAL_HOLDOUT_COMPLETED")
    require(int(lock.get("judge_run_count", -1)) == 1,
            "v0.28 final holdout remains consumed exactly once")
    require(lock.get("final_holdout_decision") == "FINAL_HOLDOUT_REVIEW",
            "v0.28 final holdout decision preserved")

    dev = pd.read_csv(DEV210, dtype={"case_id": str, "isbn13": str}, encoding="utf-8-sig")
    final = pd.read_csv(FINAL30, dtype={"case_id": str, "isbn13": str}, encoding="utf-8-sig")

    require(len(dev) == 210, "source consumed development = 210 cases")
    require(len(final) == 30, "source consumed final holdout = 30 cases")
    require(dev["case_id"].is_unique, "210 source case IDs unique")
    require(final["case_id"].is_unique, "30 final-holdout case IDs unique")
    require(set(dev["case_id"]).isdisjoint(set(final["case_id"])),
            "210/30 case IDs do not overlap")

    # Preserve every frozen label verbatim. Only provenance/version fields are added.
    final = final.copy()
    final["development_source"] = "consumed_v0_28_final_holdout_30"
    final["source_dataset_version"] = final.get("dataset_version", "4.0.0")

    # Align union of columns without silently changing existing values.
    all_cols = list(dev.columns)
    for c in final.columns:
        if c not in all_cols:
            all_cols.append(c)
    for c in all_cols:
        if c not in dev.columns:
            dev[c] = ""
        if c not in final.columns:
            final[c] = ""

    combined = pd.concat([dev[all_cols], final[all_cols]], ignore_index=True)
    combined["dataset_version"] = "6.0.0"

    require(len(combined) == 240, "v0.29 development = 240 consumed cases")
    require(combined["case_id"].is_unique, "all 240 case IDs unique")

    prefixes = {
        "U_": int(combined["case_id"].str.startswith("U_").sum()),
        "U2_": int(combined["case_id"].str.startswith("U2_").sum()),
        "U3_": int(combined["case_id"].str.startswith("U3_").sum()),
        "U4_": int(combined["case_id"].str.startswith("U4_").sum()),
    }
    require(prefixes == {"U_": 60, "U2_": 60, "U3_": 60, "U4_": 60},
            f"prefix counts = {prefixes}")

    # Ensure all 30 previously-final cases are now explicitly development provenance.
    appended = combined[combined["case_id"].isin(set(final["case_id"]))]
    require(len(appended) == 30, "all 30 final cases appended")
    require((appended["development_source"] == "consumed_v0_28_final_holdout_30").all(),
            "all appended final cases marked consumed_v0_28_final_holdout_30")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT, index=False, encoding="utf-8")

    manifest = {
        "schema_version": "1.0.0",
        "dataset_version": "6.0.0",
        "lineage": "semantic_relevance_v0.29.0",
        "role": "consumed_development",
        "case_count": 240,
        "sources": [
            {
                "file": str(DEV210.relative_to(R)),
                "sha256": sha(DEV210),
                "case_count": 210,
                "role": "consumed_development",
            },
            {
                "file": str(FINAL30.relative_to(R)),
                "sha256": sha(FINAL30),
                "case_count": 30,
                "role": "consumed_final_evidence_now_development_only",
            },
        ],
        "labels_changed_during_bootstrap": False,
        "output_file": str(OUT.relative_to(R)),
        "output_sha256": sha(OUT),
        "prefix_counts": prefixes,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    boundary_manifest = {
        "schema_version": "1.0.0",
        "lineage": "semantic_relevance_v0.29.0",
        "source_closeout": str(CLOSEOUT.relative_to(R)),
        "source_closeout_sha256": sha(CLOSEOUT),
        "development_dataset": str(OUT.relative_to(R)),
        "development_dataset_sha256": sha(OUT),
        "consumed_evidence": {
            "v0_28_development_210": "development",
            "r4_independent_validation_30": "development",
            "r8_final_holdout_30": "development_or_diagnostic_only",
        },
        "prohibited_independent_reuse": [
            "all U4 cases consumed by v0.28",
            "v0.28 r4 independent validation 30",
            "v0.28 r8 independent final holdout 30",
        ],
        "new_independent_evidence_required": True,
        "new_independent_evidence_timing": (
            "Only after a v0.29 candidate is frozen following development and stability gates."
        ),
    }
    BOUNDARY.write_text(json.dumps(boundary_manifest, indent=2) + "\n", encoding="utf-8")

    print("v0.29.0 development bootstrap built")
    print("---------------------------------")
    print("Output:", OUT)
    print("Output SHA-256:", sha(OUT))
    print("Cases:", len(combined))
    print("Prefix counts:", prefixes)
    print("Evidence boundary:", BOUNDARY)
    print("NOTE: no judge calls and no label changes were made.")


if __name__ == "__main__":
    main()
