# Checkpoint — 2026-06-09 — the A/B flip passes

## Where I stopped
- **THE A/B FLIP PASSES (the result the repo exists for).** Commit `dc94722`.
  In `rl-opt`:
  - `fusion()` rewrite: `(matmul (softmax ?x) ?v) => (fuse (matmul (softmax ?x) ?v))`,
    a GENERAL producer/consumer fusion (rule-4 honest), added to `all_rewrites`.
  - `select_plan(candidates, shapes, model)`: accounts each candidate, asks the
    `rl_cost::CostModel` for its time under the active constraints, returns the
    cheapest; near-ties break toward fewer nodes (so a FLOPs-only model won't fuse).
  - `fused_form_reachable(naive, fused, inputs)`: saturates naive and checks the
    fused form lands in the naive root's e-class.
  - Tests (rl-opt now 8): `the_ab_flip` (pick 0=naive under [Flops], 1=fused under
    [Flops,HbmBytes]) and `fused_form_is_reachable_by_rewrite`.
- `cargo test --workspace` green: rl-ir 5, rl-cost 4, rl-opt 8.
- README §3 now shows the passing test; milestone table + CLAUDE.md updated.

## Honest status of M3
- The A/B is delivered by COST-MODEL SELECTION between the reachable candidate
  plans (naive vs fused), both proven present in one saturated e-graph. That is the
  thesis demonstrated. What is NOT yet done: general DAG extraction over an
  arbitrary e-graph (pick the global min-cost term, not just rank two candidates).
- `assess.py` still flags P1 rule-6: the placeholder `extract_cheapest` (egg tree
  `Extractor` + `HbmCostFn` = node count) is still in rl-opt and still used by two
  older tests. It is not on the A/B path. Score holds at 80/100.
- Caveat unchanged: `account()` over a `RecExpr` double-counts shared Q/K/V (tree
  walk). For the A/B the sharing is identical in both candidates and the s×s HBM
  term dominates, so the flip is valid; a DAG-correct accountant is the clean fix.

## Next 3 actions (in order)
1. **General DAG extraction.** Write a custom memoized extractor in `rl-opt` that
   reads `ShapeAnalysis` shapes, computes per-e-class real bytes, counts each
   materialized e-class ONCE (DAG-aware), and drives off the `CostModel`. Replace
   `extract_cheapest`'s tree `Extractor` (clears the assess P1). Add a test that it
   returns the same naive/fused flip as `select_plan`.
2. **DAG-correct accountant** (optional but clean): make `rl_ir::account` count each
   distinct shared subtree once so intensity numbers are exact, not tree-inflated.
3. **Start M4**: lower the chosen fused plan to a real kernel (Pallas/Triton via the
   `py/` front end or a Rust emitter in `rl-codegen`), verify it matches the
   reference to 1e-5, and record predicted-vs-measured. Needs the JAX/accelerator
   path; if no GPU, scope M4 to a CPU reference-emitter + numerics gate and defer
   wall-clock.

## Environment / resume
- Rust 1.94 gnu at `C:\Users\bhansa01\.cargo\bin` (User PATH). `python` = 3.14 with
  matplotlib 3.10 (figures: `python scripts/figures.py`).
- Remote `https://sndsh404@github.com/sndsh404/roofline.git`, creds cached, push
  works. Identity `sndsh404 <hiiamsandeshbhandari@gmail.com>`. NO Co-Authored-By
  trailer (user requirement). Short plain commit messages, no dashes.
- Resume: read CLAUDE.md, README.md, this file. `python scripts/assess.py --start`,
  `cargo test --workspace`, then action 1.
