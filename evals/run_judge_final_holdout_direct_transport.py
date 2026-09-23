"""Direct-Ollama operational recovery for one pending case in the single v0.26 r6 final-holdout run."""
from __future__ import annotations
import argparse, json, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from pydantic import BaseModel

EVALS_DIR=Path(__file__).resolve().parent; REPO_ROOT=EVALS_DIR.parent
sys.path.insert(0,str(EVALS_DIR))
import run_judge_final_holdout as final_runner  # noqa: E402

class DirectOllamaTransport:
    def __init__(self,model:str,base_url:str,temperature:float=0.0,timeout_seconds:float=180.0,num_predict:int=768,**_:Any):
        self.model=model; self.base_url=base_url.rstrip('/'); self.temperature=float(temperature); self.timeout_seconds=float(timeout_seconds); self.num_predict=int(num_predict)
    def generate(self,prompt:str,schema:type[BaseModel]|None=None,**_:Any):
        if schema is None: raise TypeError("v0.26 expects structured-output schemas.")
        name=schema.__name__; json_mode=name in {"CompositeEvidenceVerification","FullContextComponentRecovery"}
        fmt="json" if json_mode else schema.model_json_schema()
        payload={"model":self.model,"messages":[{"role":"user","content":prompt}],"stream":False,"format":fmt,"options":{"temperature":self.temperature,"num_predict":self.num_predict}}
        req=urllib.request.Request(self.base_url+"/api/chat",data=json.dumps(payload).encode("utf-8"),headers={"Content-Type":"application/json"},method="POST")
        print(f"[direct-ollama] starting schema={name}, timeout={self.timeout_seconds:.0f}s")
        start=time.monotonic()
        with urllib.request.urlopen(req,timeout=self.timeout_seconds) as response: env=json.loads(response.read().decode("utf-8"))
        content=env.get("message",{}).get("content")
        if not isinstance(content,str) or not content.strip(): raise RuntimeError("Ollama returned no message.content.")
        result=json.loads(content)
        if not isinstance(result,dict): raise RuntimeError("Ollama returned non-object JSON.")
        print(f"[direct-ollama] schema={name}, elapsed={time.monotonic()-start:.2f}s")
        return result

def resolve(raw:str)->Path:
    p=Path(raw)
    if not p.is_absolute(): p=REPO_ROOT/p
    return p.resolve()

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--resume",required=True); ap.add_argument("--case-id",required=True); ap.add_argument("--timeout",type=float,default=180.0); ap.add_argument("--num-predict",type=int,default=768); ap.add_argument("--release-manifest",default=str(final_runner.DEFAULT_RELEASE_MANIFEST)); args=ap.parse_args()
    run=resolve(args.resume); cid=str(args.case_id); rf=run/"judge_results.csv"
    if rf.exists():
        cur=pd.read_csv(rf,dtype={"case_id":str},encoding="utf-8")
        if cid in set(cur.case_id.astype(str)): raise RuntimeError(f"{cid} already complete; refusing overwrite.")
    timeout=float(args.timeout); num=int(args.num_predict)
    if num<64: raise ValueError("--num-predict must be at least 64")
    class Bound(DirectOllamaTransport):
        def __init__(self,model:str,base_url:str,temperature:float=0.0,**kw:Any): super().__init__(model,base_url,temperature,timeout,num,**kw)
    final_runner.base.OllamaModel=Bound
    old=sys.argv[:]
    try:
        sys.argv=["run_judge_final_holdout.py","--dataset",args.dataset,"--release-manifest",args.release_manifest,"--resume",str(run),"--case-id",cid]
        final_runner.main()
    finally: sys.argv=old
    result=pd.read_csv(rf,dtype={"case_id":str},encoding="utf-8")
    if cid not in set(result.case_id.astype(str)): raise RuntimeError(f"{cid} was not persisted.")
    note={"case_id":cid,"timestamp_utc":datetime.now(timezone.utc).isoformat(),"transport":"direct_ollama_api_chat","deep_eval_transport_bypassed":True,"judge_behavior_changed":False,"judge_version":"0.26.0","http_timeout_seconds":timeout,"num_predict":num,"prompts_facets_scoring_unchanged":True}
    np=run/f"transport_override_{cid}.json"; np.write_text(json.dumps(note,indent=2)+"\n",encoding="utf-8")
    mp=run/"run_metadata.json"; m=json.loads(mp.read_text(encoding="utf-8")); ovs=m.get("transport_overrides",[]); ovs=ovs if isinstance(ovs,list) else [ovs]; ovs=[x for x in ovs if not(isinstance(x,dict) and x.get("case_id")==cid)]; ovs.append(note); m["transport_overrides"]=ovs; mp.write_text(json.dumps(m,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"{cid} completed in the recorded final-holdout run; transport provenance recorded.")
if __name__=="__main__": main()
