# V4H NO_CLEAR_TOP10 Headroom Diagnostic — Stage A

V4H is already consumed and the V4B release decision remains frozen FAIL.

The zero-call postmortem found 10/24 queries with no clear/strong result in
V4B's displayed top10. That observation alone cannot distinguish a
candidate-generation limitation from a ranking limitation: a clear candidate
may already exist below rank 10 in the frozen V4B full-union ranking.

Stage A therefore freezes, without any model/judge execution, V4B ranks 11–50
for exactly those 10 queries (400 query-book pairs).

After this inventory is frozen, a separate package may freshly judge those
400 pairs with the same frozen measurement instrument.

Interpretation after future judging:
- `HEADROOM_FOUND_11_50`: at least one score >=3 in ranks 11–50. This proves
  displayed-ranking headroom for that query.
- `NO_CLEAR_THROUGH_50`: no score >=3 through rank 50. This remains unresolved;
  it does not yet prove full-union candidate limitation. Optional Stage B must
  inspect ranks 51–end before making that classification.

No result from this diagnostic can change the frozen V4B release FAIL or make
V4H unseen again.
