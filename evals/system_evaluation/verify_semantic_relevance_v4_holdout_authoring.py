from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SYS = REPO_ROOT / "evals/system_evaluation"
FACETS = REPO_ROOT / "evals/facets/semantic_relevance"
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

QUERY_FILE = SYS / "semantic_relevance_v4_holdout_queries.v0.1.0.json"
FACET_FILE = FACETS / "semantic_relevance_v4_holdout_query_facets.v0.1.0.json"
SELECTION_LOCK = RUN / "v4b_dev_comparison_selection.lock.json"
LOCK_OUTPUT = SYS / "semantic_relevance_v4_holdout_authoring.lock.json"

EXPECTED_SELECTION_LOCK = "98ab3a62b978b922dc4b6e1bb031bdc6578a007a93de7ae84b20d36a80c9c9b4"
PRIOR_MANIFEST = SYS / "semantic_relevance_prior_query_texts_for_v4h.v0.1.0.json"
EXPECTED_PRIOR_MANIFEST_SHA256 = "f4b3f79da07c577b96967db6ff064cb95a9aaa46b06de44c5ad76f65a8009325"
STRATA = {
    "two_core_theme":4,
    "three_core_conjunctive":4,
    "core_plus_qualifier":4,
    "relationship_or_role_constraint":4,
    "hard_exclusion_or_contrast":4,
    "abstract_nonfiction_multi_facet":4,
}
STOP = set("a an the about and or of to for in on with as whose how not into from between after".split())
MAX_JACCARD = 0.35

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def req(c: bool, m: str) -> None:
    if not c:
        raise ValueError(m)

def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.casefold()) if t not in STOP}

def jaccard(a: str, b: str) -> float:
    x,y=tokens(a),tokens(b)
    return 0.0 if not (x or y) else len(x & y)/len(x | y)

def main() -> None:
    print("Semantic relevance V4H - unseen holdout authoring preflight")
    print("-----------------------------------------------------------")
    print("Holdout retrieval: DISABLED")
    print("BM25/dense/reranker execution: DISABLED")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release metrics/decision: NOT CALCULATED")
    print()

    req(QUERY_FILE.exists(), f"Missing V4H query file: {QUERY_FILE}")
    req(FACET_FILE.exists(), f"Missing V4H facet file: {FACET_FILE}")
    req(not LOCK_OUTPUT.exists(), f"Refusing preflight because holdout authoring lock already exists: {LOCK_OUTPUT}")
    req(SELECTION_LOCK.exists(), f"Missing V4B selection lock: {SELECTION_LOCK}")
    req(sha256(SELECTION_LOCK)==EXPECTED_SELECTION_LOCK, "V4B selection lock hash changed")
    selection = load(SELECTION_LOCK)
    req(selection.get("status")=="V4B_DEVELOPMENT_COMPARISON_AND_SELECTION_FROZEN",
        "Unexpected V4B selection-lock status")
    req(selection.get("selected_candidate")=="v4b", "V4B is not the frozen selected candidate")
    req(selection.get("requires_new_unseen_v4_holdout") is True,
        "Selection lock does not require a new unseen holdout")
    req(selection.get("release_decision_produced") is False,
        "Development selection unexpectedly produced a release decision")
    print("PASS  frozen V4B selection lock verified; unseen holdout is required")

    req(PRIOR_MANIFEST.exists(), f"Missing frozen prior-query manifest: {PRIOR_MANIFEST}")
    req(sha256(PRIOR_MANIFEST)==EXPECTED_PRIOR_MANIFEST_SHA256,
        "Frozen prior-query anti-leakage manifest hash changed")
    prior_doc = load(PRIOR_MANIFEST)
    req(prior_doc.get("query_count")==36 and len(prior_doc.get("queries",[]))==36,
        "Frozen prior-query anti-leakage manifest must contain Q01-Q12 + H01-H24")
    req([x["query_id"] for x in prior_doc["queries"]] ==
        [f"Q{i:02d}" for i in range(1,13)] + [f"H{i:02d}" for i in range(1,25)],
        "Frozen prior-query manifest IDs/order changed")
    print("PASS  frozen branch-independent Q01-Q12 + H01-H24 anti-leakage manifest verified")

    qdoc, fdoc = load(QUERY_FILE), load(FACET_FILE)
    req(qdoc.get("status")=="frozen_before_holdout_retrieval", "Unexpected query-file status")
    req(fdoc.get("status")=="frozen_before_holdout_retrieval", "Unexpected facet-file status")
    req(qdoc.get("selected_development_candidate")=="v4b", "Query file candidate identity changed")
    req(qdoc.get("query_count")==24, "Query-file query count changed")

    qs, fs = qdoc["queries"], fdoc["queries"]
    req(len(qs)==24 and len(fs)==24, "Expected exactly 24 V4H queries")
    expected_ids=[f"V4H{i:02d}" for i in range(1,25)]
    req([x["query_id"] for x in qs]==expected_ids, "V4H query IDs/order changed")
    req([x["query_id"] for x in fs]==expected_ids, "V4H facet query IDs/order changed")
    req(Counter(x["stratum"] for x in qs)==Counter(STRATA), "V4H stratum balance changed")
    req(Counter(x["stratum"] for x in fs)==Counter(STRATA), "V4H facet stratum balance changed")

    for q,f in zip(qs,fs):
        req(q["query"]==f["query"], f"{q['query_id']}: query/facet text mismatch")
        req(q["stratum"]==f["stratum"], f"{q['query_id']}: query/facet stratum mismatch")
        facets=f.get("facets",[])
        req(bool(facets), f"{q['query_id']}: no facets")
        for i,fac in enumerate(facets,1):
            req(fac.get("facet_id")==f"F{i}", f"{q['query_id']}: facet IDs must be sequential")
            req(fac.get("facet_type") in {"core","qualifier"}, f"{q['query_id']}: invalid facet type")
            req(bool(fac.get("semantic_definition")), f"{q['query_id']}: missing semantic definition")
            req(bool(fac.get("required_components")), f"{q['query_id']}: missing required components")
        if q["stratum"]=="two_core_theme":
            req(len(facets)==2 and all(x["facet_type"]=="core" for x in facets),
                f"{q['query_id']}: expected two core facets")
        elif q["stratum"]=="three_core_conjunctive":
            req(len(facets)==3 and all(x["facet_type"]=="core" for x in facets),
                f"{q['query_id']}: expected three core facets")
        elif q["stratum"]=="core_plus_qualifier":
            req(len(facets)==2 and [x["facet_type"] for x in facets]==["core","qualifier"],
                f"{q['query_id']}: expected core + qualifier")
        elif q["stratum"]=="relationship_or_role_constraint":
            req(len(facets)==1 and facets[0]["facet_type"]=="core",
                f"{q['query_id']}: expected one relational core facet")
        elif q["stratum"]=="hard_exclusion_or_contrast":
            req(len(facets)==1 and bool(facets[0].get("hard_exclusions")),
                f"{q['query_id']}: expected explicit hard exclusion")
        elif q["stratum"]=="abstract_nonfiction_multi_facet":
            req(len(facets)==3 and all(x["facet_type"]=="core" for x in facets),
                f"{q['query_id']}: expected three core nonfiction facets")

    prior = [(x["query_id"],x["query"]) for x in prior_doc["queries"]]
    prior_norm={re.sub(r"\s+"," ",t.strip().casefold()) for _,t in prior}
    max_pair=(0.0,None,None,None)
    for q in qs:
        norm=re.sub(r"\s+"," ",q["query"].strip().casefold())
        req(norm not in prior_norm, f"{q['query_id']}: exact normalized reuse of a prior query")
        for pid,ptext in prior:
            sim=jaccard(q["query"],ptext)
            if sim>max_pair[0]:
                max_pair=(sim,q["query_id"],pid,ptext)
            req(sim < MAX_JACCARD,
                f"{q['query_id']}: excessive token overlap with {pid}: jaccard={sim:.3f}")

    anti=qdoc.get("anti_leakage",{})
    req(anti.get("catalog_or_v4h_retrieval_outputs_inspected_during_authoring") is False,
        "Authoring record says V4H retrieval/catalog outputs were inspected")
    req(anti.get("v4h_judge_outputs_inspected_during_authoring") is False,
        "Authoring record says V4H judge outputs were inspected")

    print("PASS  exact V4H01-V4H24 IDs and 4x6 stratum balance verified")
    print("PASS  query file and facet file agree exactly")
    print("PASS  facet-shape contracts verified for all six strata")
    print("PASS  no exact or near-duplicate prior Q/H prompt detected")
    print(f"INFO  maximum prior-query token Jaccard: {max_pair[0]:.3f} ({max_pair[1]} vs {max_pair[2]})")
    print("PASS  V4H candidate-blind authoring record verified")
    print()
    print("V4H UNSEEN HOLDOUT AUTHORING PREFLIGHT: PASS")
    print(f"V4H query file SHA-256: {sha256(QUERY_FILE)}")
    print(f"V4H facet file SHA-256: {sha256(FACET_FILE)}")
    print(f"Prior-query anti-leakage manifest SHA-256: {sha256(PRIOR_MANIFEST)}")
    print("Holdout retrieval performed: NO")
    print("Judge calls performed: 0")
    print("Human relevance labels used: NO")
    print("Release metrics calculated: NO")
    print("Next: freeze the V4H query/facet artifacts before any retrieval.")

if __name__ == "__main__":
    main()
