# V5 Development Design Freeze r1

This package freezes the V5 **experimental program**, not the R1/K1 implementations.

It consumes only already-frozen V4B/V4H evidence and performs zero retrieval, model, reranker, judge, or development-selection execution.

The central decision is a 2×2 factorized program:

- `B = R0K0`: frozen V4B baseline
- `R = R1K0`: candidate-generation-only change
- `K = R0K1`: ranking-only change
- `RK = R1K1`: combined change

V4H is consumed. Its evidence justifies investigating both factors but is prohibited from parameter/model/prompt tuning.

A fresh `V5D01-V5D24` selection set is required. Exact R1 and K1 specs/implementations must be frozen **before V5D authoring or execution**.

Run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\freeze_semantic_relevance_v5_development_design.ps1 2>&1 | `
  Tee-Object -FilePath .\semantic_relevance_v5_development_design_freeze_output.txt
```

Do not author V5D or implement/execute semantic V5 experiments until the design freeze output has been reviewed.
