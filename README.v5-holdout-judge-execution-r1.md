# V5H frozen judge execution r1

The V5H judging input is frozen at **359 fresh query-book pairs**. Historical
score reuse is **0**. This package executes the unchanged frozen
`semantic_relevance_v0.29.0-r5` measurement instrument.

## Frozen judging input

- score inventory SHA-256:
  `cc350ec54f519ff710f34d24a4aedf45ddb2910f52a7a46276f9a783ecb38e20`
- novel judging input SHA-256:
  `fc046bfef77af201aa7fe91bef6649372292b4a277929b0b99f5311a1666feab`
- judging-input lock SHA-256:
  `a2ff301aef1788526d51858394ee4249eeb548fc05981e0246a50a2a8643eea7`

## Execution behavior

- resume safe
- persists after every successful case
- never persists the currently failing case
- smoke mode selects at most one currently-unscored case from each of
  V5H01, V5H17, and V5H21 (maximum three)
- full mode resumes from persisted smoke/partial scores
- final score lock is written only after exact 359/359 completion

## Known non-semantic execution repairs

Two previously accepted structured-output format repairs are active from case 1:

1. exact duplicate Stage-A real span IDs are de-duplicated preserving first order;
2. prominence span IDs that exactly echo a supplied rendered source line are
   converted back to that line's literal S# ID.

All other malformed or unknown IDs continue through the original frozen
validators and fail normally.

No judge source, prompt, semantic rule, scoring rule, facet specification,
configuration, rubric, V5H evidence, metric, gate, or release decision is changed.

## Disabled throughout judge execution

- retrieval / reranking
- candidate mutation
- historical score reuse
- human relevance labels
- V5H/release metric calculation
- release gates
- release decision

Execution contract SHA-256: `454d78668ffef002f0f3cf67a182f20f252d18b837354ef0e3d9bed640240fcb`
Carry-forward patch SHA-256: `5330a4bd7fbcbdbfb278fd9459a980452db95b72254bbe5f646643f4d514c589`
Verifier SHA-256: `b1a9ebedb4886b15445d9dfc0550aa15e89b7674791d0e2d77e73a515911a8f7`
Runner SHA-256: `ded4d6fb52cbe738362330ab75df98dc4b6b6918e7b297c8f37b0da94abc58b1`
