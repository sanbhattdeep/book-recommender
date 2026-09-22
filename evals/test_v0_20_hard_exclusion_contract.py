from semantic_relevance_facet_judge import EvidenceVerification, _validate_verification_result
from semantic_relevance_facet_scoring import QueryFacet, VerificationRelation, EvidenceInferenceKind, HardExclusion
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'evals/judge_configs/semantic_relevance_judge.v0.20.0.json').read_text(encoding="utf-8"))
facet=QueryFacet(facet_id='F1', text='suspense', facet_type='core', semantic_definition='x', hard_exclusions=[HardExclusion(exclusion_id='HX', rule='choice alone is not suspense')])
valid=EvidenceVerification(verification_relation=VerificationRelation.UNSUPPORTED,inference_kind=EvidenceInferenceKind.NONE,hard_exclusion_triggered=True,hard_exclusion_id='HX',reason='hard exclusion applies')
_validate_verification_result(valid, facet, CONFIG)

for bad in [
    EvidenceVerification(verification_relation=VerificationRelation.ENTAILED,inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,hard_exclusion_triggered=True,hard_exclusion_id='HX',reason='entailed'),
    EvidenceVerification(verification_relation=VerificationRelation.UNSUPPORTED,inference_kind=EvidenceInferenceKind.NONE,hard_exclusion_triggered=True,hard_exclusion_id='UNKNOWN',reason='bad id'),
]:
    try:
        _validate_verification_result(bad, facet, CONFIG)
        raise AssertionError('expected validation failure')
    except Exception:
        pass
print('v0.20 hard-exclusion contract tests passed.')
