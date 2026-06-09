# Roofline assessment — 2026-06-09 14:32 UTC

**Score: 80/100 — COMMIT_READY**  (gates: commit 80, pr 90, excellence 95)

## Numbers
- `m1_attn_binding` = HbmBytes
- `m1_hbm_bound_rows` = 5
- `tests_passed` = 16

## P1 — Regressions / rule violations
- (-20) rule 6: tree Extractor used; DESIGN demands LpExtractor (DAG-aware) - tree cost double-counts shared Q/K/V
