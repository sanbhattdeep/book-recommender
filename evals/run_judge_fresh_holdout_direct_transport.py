"""Recover exactly one recorded failed case using direct native Ollama transport."""
from __future__ import annotations
import argparse, json, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from pydantic import BaseModel

EVALS_DIR=Path(__file__).resolve().parent; REPO_ROOT=EVALS_DIR.parent
sys.path.insert(0,str(EVALS_DIR))
import run_judge_fresh_holdout as runner

class DirectOllamaTransport:
    def __init__(self, model:str, base_url:str, temperature:float=0.0, timeout_seconds:float=240.0, **_:Any):
        self.model=model; self.base_url=base_url.rstrip('/'); self.temperature=float(temperature); self.timeout_seconds=float(timeout_seconds)
    def generate(self,prompt:str,schema:type[BaseModel]|None=None,**_:Any):
        if schema is None: raise TypeError('Structured-output schema required.')
        payload={'model':self.model,'messages':[{'role':'user','content':prompt}],'stream':False,'format':schema.model_json_schema(),'options':{'temperature':self.temperature}}
        req=urllib.request.Request(self.base_url+'/api/chat',data=json.dumps(payload).encode('utf-8'),headers={'Content-Type':'application/json'},method='POST')
        start=time.monotonic()
        with urllib.request.urlopen(req,timeout=self.timeout_seconds) as resp: envelope=json.loads(resp.read().decode('utf-8'))
        content=envelope.get('message',{}).get('content')
        if not isinstance(content,str) or not content.strip(): raise RuntimeError('Ollama returned no message.content.')
        result=schema.model_validate_json(content); print(f'[direct-ollama] schema={schema.__name__}, elapsed={time.monotonic()-start:.2f}s'); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--resume',required=True); p.add_argument('--case-id',required=True); p.add_argument('--timeout',type=float,default=240.0); args=p.parse_args()
    run_dir=Path(args.resume); run_dir=run_dir if run_dir.is_absolute() else REPO_ROOT/run_dir
    failure_file=run_dir/'last_failure.json'
    if not failure_file.exists(): raise RuntimeError('No last_failure.json; direct recovery is allowed only after an actual recorded failure.')
    failure=json.loads(failure_file.read_text(encoding='utf-8')); failed=str(failure.get('case_id')); requested=str(args.case_id)
    if requested!=failed: raise RuntimeError(f'Requested {requested} but recorded failed case is {failed}.')
    results_file=run_dir/'judge_results.csv'
    if results_file.exists():
        cur=pd.read_csv(results_file,dtype={'case_id':str},encoding='utf-8')
        if requested in set(cur['case_id'].astype(str)): raise RuntimeError(f'{requested} is already complete.')
    timeout=float(args.timeout)
    class Bound(DirectOllamaTransport):
        def __init__(self,model:str,base_url:str,temperature:float=0.0,**kwargs:Any): super().__init__(model,base_url,temperature,timeout,**kwargs)
    runner.OllamaModel=Bound
    old=sys.argv[:]
    try:
        sys.argv=['run_judge_fresh_holdout.py','--resume',str(run_dir),'--recover-failed-case',requested]
        runner.main()
    finally: sys.argv=old
    cur=pd.read_csv(results_file,dtype={'case_id':str},encoding='utf-8')
    if requested not in set(cur['case_id'].astype(str)): raise RuntimeError('Recovered case was not persisted.')
    note={'case_id':requested,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'transport':'direct_ollama_api_chat','deep_eval_transport_bypassed':True,'judge_behavior_changed':False,'judge_version':'0.18.0','structured_output':True,'http_timeout_seconds':timeout,'prompts_facets_scoring_unchanged':True}
    (run_dir/f'transport_override_{requested}.json').write_text(json.dumps(note,indent=2)+'\n',encoding='utf-8')
    meta_path=run_dir/'run_metadata.json'; meta=json.loads(meta_path.read_text(encoding='utf-8')); overrides=meta.get('transport_overrides',[]); overrides=[x for x in overrides if not(isinstance(x,dict) and x.get('case_id')==requested)]; overrides.append(note); meta['transport_overrides']=overrides; meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if failure_file.exists(): failure_file.unlink()
    print(f'{requested} completed; direct-transport provenance recorded.')

if __name__=='__main__': main()
