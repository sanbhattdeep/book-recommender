from pathlib import Path
text=(Path(__file__).resolve().parent/'semantic_relevance_facet_judge.py').read_text(encoding='utf-8')
for key in [
    '"resolved_context_role"',
    '"primary_subject_summary"',
    '"primary_subject_span_ids"',
    '"is_primary_subject"',
    '"is_background_cause_or_factor"',
    '"is_example_or_illustration"',
    '"is_meta_discussion"',
    '"is_substantively_examined"',
    '"role_supporting_span_ids"',
    '"role_reason"',
]:
    assert key in text, key
print('v0.22.0 role audit serialization contract passed')
