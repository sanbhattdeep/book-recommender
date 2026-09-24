"""Deterministically split the fully human-labelled 60-case unseen pool into 30 validation + 30 final holdout.

NO judge is called. The final holdout is written with .DO_NOT_RUN_YET.csv and a LOCKED manifest.
The split refuses to overwrite an existing split.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, subprocess, sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS = REPO_ROOT / "evals"; DATASETS = EVALS / "datasets"
POOL = DATASETS / "semantic_relevance_unseen_pool.v3.0.0.csv"
VALIDATION = DATASETS / "semantic_relevance_validation.v3.0.0.csv"
HOLDOUT = DATASETS / "semantic_relevance_final_holdout.v3.0.0.DO_NOT_RUN_YET.csv"
LOCK = DATASETS / "semantic_relevance_final_holdout_lock.v3.0.0.json"
SPLIT_MANIFEST = DATASETS / "semantic_relevance_unseen_pool_manifest.v3.0.0.json"
SPEC = DATASETS / "semantic_relevance_unseen_pool_split_spec.v3.0.0.json"
FREEZE = EVALS / "releases" / "semantic_relevance_v0.27.0_r5_development_candidate.json"
EXPECTED_FREEZE_SHA = "3de6ee3a6274a5d066f17ff24bc965b05dfb8d60b785c0210d45d743538d84e5"
SEED = "semantic-relevance-v0.27-unseen-v3-seed-20260924"

def sha256(p: Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(x: object)->str: return re.sub(r"[^a-z0-9]+"," ",str(x or "").strip().lower()).strip()
def h(text:str)->str: return hashlib.sha256(text.encode("utf-8")).hexdigest()

def main()->None:
    for p in (POOL,SPEC,FREEZE):
        if not p.exists(): raise FileNotFoundError(f"Required file not found: {p}")
    for p in (VALIDATION,HOLDOUT,LOCK,SPLIT_MANIFEST):
        if p.exists(): raise FileExistsError(f"Refusing to overwrite existing split artifact: {p}")
    if sha256(FREEZE) != EXPECTED_FREEZE_SHA: raise ValueError("Frozen r5 candidate manifest hash mismatch.")

    # Reuse the dedicated authoring validator; no judge involved.
    completed = subprocess.run([sys.executable, str(EVALS/'validate_v0_27_unseen_pool.py')], cwd=REPO_ROOT)
    if completed.returncode != 0: raise RuntimeError("Unseen-pool authoring validation failed; split not created.")

    df = pd.read_csv(POOL,dtype=str,keep_default_na=False,encoding="utf-8")
    query_hashes = sorted((h(f"{SEED}|quota|{qid}"),qid) for qid in sorted(df["query_id"].unique()))
    quota3 = {qid for _,qid in query_hashes[:6]}
    assignment = {}
    validation_parts=[]; holdout_parts=[]
    for qid in sorted(df["query_id"].unique()):
        sub=df.loc[df["query_id"].eq(qid)].copy()
        sub["_split_hash"] = sub.apply(lambda r: h(f"{SEED}|{qid}|{r['case_id']}|{r['isbn13'].strip()}|{norm(r['title'])}|{norm(r['authors'])}"),axis=1)
        sub=sub.sort_values(["_split_hash","case_id"]).reset_index(drop=True)
        vq=3 if qid in quota3 else 2
        val=sub.iloc[:vq].copy(); hol=sub.iloc[vq:].copy()
        validation_parts.append(val); holdout_parts.append(hol)
        for cid in val["case_id"]: assignment[cid]="validation"
        for cid in hol["case_id"]: assignment[cid]="final_holdout"

    val=pd.concat(validation_parts,ignore_index=True).drop(columns=["_split_hash"])
    hol=pd.concat(holdout_parts,ignore_index=True).drop(columns=["_split_hash"])
    val=val.sort_values(["query_id","case_id"]).reset_index(drop=True)
    hol=hol.sort_values(["query_id","case_id"]).reset_index(drop=True)
    if len(val)!=30 or len(hol)!=30: raise ValueError(f"Split must be 30/30, got {len(val)}/{len(hol)}")
    if set(val.case_id)&set(hol.case_id): raise ValueError("Validation/final-holdout overlap.")
    if set(val.case_id)|set(hol.case_id) != set(df.case_id): raise ValueError("Split does not reconstruct full pool.")

    val.to_csv(VALIDATION,index=False,encoding="utf-8")
    hol.to_csv(HOLDOUT,index=False,encoding="utf-8")
    now=datetime.now(timezone.utc).isoformat()
    manifest={
      "schema_version":"1.0.0","pool_version":"3.0.0","split_seed":SEED,"split_method":"sha256_stratified_30_30",
      "created_at_utc":now,"frozen_candidate_manifest":str(FREEZE.relative_to(REPO_ROOT)),"frozen_candidate_manifest_sha256":sha256(FREEZE),
      "pool":{"path":str(POOL.relative_to(REPO_ROOT)),"cases":60,"sha256":sha256(POOL)},
      "validation":{"path":str(VALIDATION.relative_to(REPO_ROOT)),"cases":30,"sha256":sha256(VALIDATION),"case_ids":val.case_id.tolist()},
      "final_holdout":{"path":str(HOLDOUT.relative_to(REPO_ROOT)),"cases":30,"sha256":sha256(HOLDOUT),"case_ids":hol.case_id.tolist(),"status":"LOCKED_DO_NOT_RUN"},
      "assignment":assignment,
      "query_validation_quotas":{qid:(3 if qid in quota3 else 2) for qid in sorted(df.query_id.unique())},
      "human_labels_completed_before_split":True,
      "judge_output_generated_before_split":False,
    }
    SPLIT_MANIFEST.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lock={
      "schema_version":"1.0.0","status":"LOCKED_DO_NOT_RUN","reason":"Final holdout must remain untouched until validation is complete and a candidate is frozen for one-time holdout evaluation.",
      "created_at_utc":now,"final_holdout_path":str(HOLDOUT.relative_to(REPO_ROOT)),"final_holdout_sha256":sha256(HOLDOUT),
      "split_manifest_path":str(SPLIT_MANIFEST.relative_to(REPO_ROOT)),"split_manifest_sha256":sha256(SPLIT_MANIFEST),
      "frozen_candidate_manifest_sha256":sha256(FREEZE),"judge_run_count":0,
    }
    LOCK.write_text(json.dumps(lock,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

    print("v0.27 unseen pool split complete")
    print("-------------------------------")
    print("PASS  source pool:       60 labelled cases")
    print("PASS  validation:        30 cases")
    print("PASS  final holdout:     30 cases")
    print("PASS  validation/holdout disjoint")
    print("PASS  validation + holdout reconstruct pool")
    print("PASS  final holdout status: LOCKED_DO_NOT_RUN")
    print(f"Pool SHA-256:            {sha256(POOL)}")
    print(f"Validation SHA-256:      {sha256(VALIDATION)}")
    print(f"Final holdout SHA-256:   {sha256(HOLDOUT)}")
    print(f"Split manifest SHA-256:  {sha256(SPLIT_MANIFEST)}")
    print(f"Lock manifest SHA-256:   {sha256(LOCK)}")

if __name__ == "__main__": main()
