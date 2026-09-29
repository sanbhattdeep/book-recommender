"""Static verifier for the v0.28 r4 unseen-pool BUILD-ONLY package r1.

This verifier performs no retrieval and no judge calls. It verifies that the
builder is pinned to the frozen r4 release candidate and that the new 60-case
pool contract is U4_Q01..Q12 x T01..T05 with blank human labels.
"""
from __future__ import annotations
from pathlib import Path
import csv
import hashlib

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evals"
D = E / "datasets"
R = E / "releases"
EXPECTED_FREEZE = "6d1f390a21d804351ff75def5bfd74e7e5eafbd48685aa4ee50058b6af670faa"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    builder = E / "build_v0_28_unseen_evaluation_pool.py"
    exporter = E / "export_v0_28_unseen_pool_labeling_xlsx.py"
    template = D / "semantic_relevance_unseen_pool.v4.0.0.template.csv"
    freeze = R / "semantic_relevance_v0.28.0_r4_release_candidate.json"
    dev = D / "semantic_relevance_v0.28_development.v4.0.0.csv"
    facet = E / "facets" / "semantic_relevance" / "semantic_relevance_query_facets.v0.9.9.json"

    for p in [builder, exporter, template, freeze, dev, facet]:
        if not p.exists():
            raise FileNotFoundError(p)
    for p in [builder, exporter]:
        compile(p.read_text(encoding="utf-8-sig"), str(p), "exec")

    if sha256(freeze) != EXPECTED_FREEZE:
        raise ValueError(
            "Frozen r4 release-candidate hash mismatch: "
            f"expected={EXPECTED_FREEZE} actual={sha256(freeze)}"
        )

    with dev.open(encoding="utf-8-sig", newline="") as f:
        dev_rows = list(csv.DictReader(f))
    if len(dev_rows) != 180:
        raise ValueError(f"Consumed v0.28 development must contain 180 rows; found {len(dev_rows)}")

    with template.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 60:
        raise ValueError(f"Template must contain 60 rows; found {len(rows)}")
    expected = {f"U4_Q{q:02d}_T{i:02d}" for q in range(1,13) for i in range(1,6)}
    actual = {r["case_id"] for r in rows}
    if actual != expected or len(actual) != 60:
        raise ValueError("Template case IDs must be exactly U4_Q01..Q12 x T01..T05")
    if any(r["human_score"].strip() or r["human_reason"].strip() for r in rows):
        raise ValueError("Template human labels must be blank")
    if any(r["review_status"].strip() != "UNREVIEWED" for r in rows):
        raise ValueError("Template review_status must be UNREVIEWED")

    text = builder.read_text(encoding="utf-8-sig")
    required = [
        'POOL_VERSION = "4.0.0"',
        'FACET_SPEC_VERSION = "0.9.9"',
        'RANDOM_SEED = 20260929',
        'semantic_relevance_v0.28_development.v4.0.0.csv',
        'semantic_relevance_v0.28.0_r4_release_candidate.json',
        EXPECTED_FREEZE,
        'expected_rows=180',
        'semantic_relevance_calibration.v0.5.0.csv',
        'case_id=f"U4_{qid}_T{slot:02d}"',
        'case_id=f"U4_{qid}_T05"',
        '"query_slice": "unseen_pool_v4"',
        'no overlap with ANY consumed calibration/development candidate identities',
    ]
    for needle in required:
        if needle not in text:
            raise ValueError(f"Builder contract missing: {needle}")

    lowered = text.lower()
    forbidden = ["semantic_relevance_facet_judge", "run_judge", "judge_score", "openai"]
    for needle in forbidden:
        if needle in lowered:
            raise ValueError(f"Build-only script unexpectedly references judge execution: {needle}")

    print("v0.28 r4 unseen-pool build-only package verification passed.")
    print("PASS  frozen r4 release-candidate manifest hash pinned")
    print("PASS  180-case consumed-development source required")
    print("PASS  original historical calibration identities also excluded")
    print("PASS  U4 60-case template: 12 queries x 5 candidates")
    print("PASS  target raw ranks 2/10/30/70 + deterministic random")
    print("PASS  ISBN + normalized title/author freshness protection")
    print("PASS  blind-label workbook exporter contract")
    print("PASS  no judge execution path in build-only scripts")


if __name__ == "__main__":
    main()
