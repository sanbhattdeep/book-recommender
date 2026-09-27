"""Non-judging preflight for the frozen v0.27.0 r7 one-time final holdout."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import sys

EVALS_DIR=Path(__file__).resolve().parent
REPO_ROOT=EVALS_DIR.parent
sys.path.insert(0,str(EVALS_DIR))
import run_judge_v0_27_r7_final_holdout as final_runner


def main()->None:
    dataset=final_runner.DEFAULT_DATASET.resolve()
    release=final_runner.DEFAULT_RELEASE_MANIFEST.resolve()
    if final_runner.FINALITY_MARKER.exists():
        marker=final_runner.load_json(final_runner.FINALITY_MARKER)
        raise RuntimeError(
            "FINAL HOLDOUT HAS ALREADY STARTED. Preflight will not authorize a new run. "
            f"Recorded run: {marker.get('run_directory')}"
        )
    final_runner.validate_release_manifest(release)
    final_runner.validate_lock_pre_start(dataset)
    if final_runner.sha256(dataset)!=final_runner.EXPECTED_HOLDOUT_SHA256:
        raise ValueError("Final holdout SHA-256 mismatch.")

    base=final_runner.base
    data=pd.read_csv(dataset,dtype={'case_id':str,'query_id':str,'isbn13':str},encoding='utf-8')
    rubric=base.load_json(base.RUBRIC_FILE)
    config=base.load_json(base.JUDGE_CONFIG_FILE)
    facet_payload=base.load_json(base.FACET_SPEC_FILE)
    facet_specs=base.load_facet_specs(facet_payload)
    final_runner.validate_holdout_inputs(data,rubric,config,facet_payload,facet_specs)

    print('v0.27.0 r7 final-holdout PRE-FLIGHT: PASS')
    print('------------------------------------------')
    print('PASS  release candidate manifest is frozen and hash-pinned')
    print('PASS  frozen semantic artifacts match release manifest')
    print('PASS  final holdout: 30 labelled cases, dataset_version=3.0.0')
    print('PASS  Q01-Q12 coverage and frozen query texts')
    print('PASS  no case-ID / ISBN / title+author overlap with 150-case development')
    print('PASS  final holdout SHA-256 pinned')
    print('PASS  final holdout status = LOCKED_DO_NOT_RUN')
    print('PASS  final holdout judge_run_count = 0')
    print('PASS  no finality marker exists')
    print('JUDGE EXECUTION: DISABLED')
    print('NO HOLDOUT JUDGE CALLS WERE MADE.')
    print(f'Confirmation token for the separately reviewed execution step: {final_runner.CONFIRM_TOKEN}')

if __name__=='__main__': main()
