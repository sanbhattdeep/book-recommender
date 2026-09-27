from semantic_relevance_facet_judge import IsolatedComponentVerification,FullContextComponentRecovery,_apply_speculative_isolated_entailment_guard,_apply_speculative_recovery_entailment_guard,_reason_admits_speculative_entailment
assert _reason_admits_speculative_entailment('This could involve a journey.')=='could'
assert _reason_admits_speculative_entailment('This is necessarily a journey.') is None
x=IsolatedComponentVerification(component_id='x',grounding_relation='entailed',negative_boundary_applied=False,external_knowledge_required=False,reason='This could involve challenges.')
y=_apply_speculative_isolated_entailment_guard(x);assert y.grounding_relation=='missing' and 'speculative_entailment_guard' in y.reason
x2=IsolatedComponentVerification(component_id='x',grounding_relation='entailed',negative_boundary_applied=False,external_knowledge_required=False,reason='This necessarily establishes the component.')
assert _apply_speculative_isolated_entailment_guard(x2).grounding_relation=='entailed'
r=FullContextComponentRecovery(component_id='x',grounding_relation='entailed',supporting_span_ids=['S1'],negative_boundary_applied=False,external_knowledge_required=False,reason='This is possibly an example.')
assert _apply_speculative_recovery_entailment_guard(r).grounding_relation=='missing'
print('v0.28 speculative-entailment discipline contract passed')
