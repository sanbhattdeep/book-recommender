"""End-to-end mock contract: one book-subject call is reused by all core facets."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from semantic_relevance_facet_judge import (
    BookSubjectAnalysis,
    EvidenceSelection,
    EvidenceVerification,
    ProminenceAssessment,
    generate_validated_semantic_verdict,
)
from semantic_relevance_facet_scoring import (
    EvidenceInferenceKind,
    QueryFacet,
    QueryFacetSpec,
    VerificationRelation,
    SubjectRelation,
)

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / 'evals/judge_configs/semantic_relevance_judge.v0.22.2.json').read_text())

spec = QueryFacetSpec(
    query_id='SYN',
    query='synthetic query',
    facets=[
        QueryFacet(facet_id='F1', text='alpha', facet_type='core', semantic_definition='Alpha content.'),
        QueryFacet(facet_id='F2', text='beta', facet_type='core', semantic_definition='Beta content.'),
    ],
)
row = pd.Series({
    'case_id': 'SYN_1',
    'description': 'Alpha defines the main premise. Beta is a secondary contextual factor.',
})

FROZEN = 'The book is primarily about an alpha-defined main premise.'

class Model:
    def __init__(self):
        self.subject_calls = 0
        self.selection_calls = 0
        self.role_calls = 0

    def generate(self, prompt, schema, **kwargs):
        if schema is BookSubjectAnalysis:
            self.subject_calls += 1
            assert 'synthetic query' not in prompt
            assert '\nalpha\n' not in prompt.lower()
            return BookSubjectAnalysis(
                primary_subject_summary=FROZEN,
                primary_subject_span_ids=['S1'],
                reason='S1 states the main premise.',
            )
        if schema is EvidenceSelection:
            self.selection_calls += 1
            sid = 'S1' if self.selection_calls == 1 else 'S2'
            return EvidenceSelection(evidence_span_ids=[sid], reason='synthetic candidate')
        if schema is EvidenceVerification:
            return EvidenceVerification(
                verification_relation=VerificationRelation.DIRECT,
                inference_kind=EvidenceInferenceKind.EXPLICIT_COMPONENTS,
                reason='synthetic direct support',
            )
        if schema is ProminenceAssessment:
            self.role_calls += 1
            assert FROZEN in prompt
            if self.role_calls == 1:
                return ProminenceAssessment(
                    subject_relation=SubjectRelation.SAME_AS_PRIMARY_SUBJECT,
                    is_substantively_examined=True,
                    supporting_span_ids=['S1'],
                    reason='alpha is the defining premise',
                )
            return ProminenceAssessment(
                subject_relation=SubjectRelation.CAUSAL_OR_CONTEXTUAL_BACKGROUND,
                is_substantively_examined=False,
                supporting_span_ids=['S2'],
                reason='beta is contextual background',
            )
        raise AssertionError(schema)

model = Model()
result = generate_validated_semantic_verdict(model, row, spec, {}, config)
assert model.subject_calls == 1
assert model.role_calls == 2
assert result.book_subject_analysis.primary_subject_summary == FROZEN
assert [a.primary_subject_summary for a in result.verdict.assessments] == [FROZEN, FROZEN]
assert [a.primary_subject_span_ids for a in result.verdict.assessments] == [['S1'], ['S1']]
print('v0.22.0 one-subject-per-case reuse contract passed')
