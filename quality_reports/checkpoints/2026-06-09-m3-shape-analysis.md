# Checkpoint — 2026-06-09 — M3 shape analysis done, A/B remaining

## Where I stopped
- **M3 increment 1 done and pushed (`b763cdb`):** `ShapeAnalysis` is now an
  `egg::Analysis<TensorLang>` in `crates/rl-opt/src/lib.rs`. It propagates 2-D
  shapes bottom-up through the e-graph; `Var` shapes come from an `inputs` map on
  the analysis. New API: `saturate_shaped(expr, inputs) ->
  Runner<TensorLang, ShapeAnalysis>`. The rule functions are now generic over the
  analysis `N`, so the old `()` path and the shaped path share one rule set.
  Tests: rl-opt now 6 (added `shape_analysis_infers_attention_output`,
  `shape_survives_saturation_equivalences`). `cargo test --workspace` green.
- **Key finding that defines the remaining M3 work:** with the current rewrite
  rules (matmul assoc, transpose, scale distribution), every equivalent form
  materializes the SAME set of tensors, including the s×s scores. So their HBM
  cost is identical and NO naive-vs-flash flip is possible yet. The flip requires
  a fusion primitive so the s×s intermediate is never spilled to HBM. This is not
  a bug; it is the real design content of M3.
- assess.py still flags P1 "tree Extractor / rule 6". Note: `egg::LpExtractor`
  needs the `coin_cbc` C solver, which is NOT available on this Windows box. The
  plan is a custom shape-aware DAG-cost extractor (count each materialized e-class
  once), documented as the CBC-free substitute. That clears the finding honestly.

## Next 3 actions (in order) — finishing M3 (the A/B)
1. **Add the `fuse` primitive to the IR (`crates/rl-ir`).** A node
   `"fuse" = Fuse([Id; 1])` meaning "this subgraph runs as one kernel; its
   internal intermediates are not spilled to HBM." Wire it through:
   - interpreter: `Fuse([x])` evaluates exactly as `eval(x)` (identity on value),
     so numerics stay at 1e-5 — VERIFY this with the existing reference test.
   - shape inference + `ShapeAnalysis::make`: `Fuse([x])` has x's shape.
   - accountant: a fused region counts HBM only for its boundary inputs read and
     final output written, NOT the internal intermediates (that is the whole point).
   Keep it GENERAL (a producer consumed immediately need not spill) so it is not a
   canned `naive=>flash` rule (CLAUDE.md rule 4). It applies to MLP up/down too.
2. **Add the fusion rewrite + a shape-aware DAG extractor in `rl-opt`.** Rewrite:
   wrap a `matmul(softmax(...), V)` chain in `fuse` (general producer/consumer
   fusion). Extractor: use `ShapeAnalysis` shapes to compute real per-e-class HBM
   bytes; cost a plan as the sum of bytes of the distinct materialized e-classes
   (this is the DAG-aware, no-double-count model). Drive it from the `rl-cost`
   constraint set so dropping `HbmBytes` changes the winner.
3. **The A/B test (the result the repo exists for).** Same saturated e-graph:
   extract under `[Flops]` -> naive (fusion saves no flops, may add rescale flops);
   extract under `[Flops, HbmBytes]` -> the fused form (drops the s×s HBM traffic).
   Assert the flip. Add it as the README's headline figure. Then tick M3 in
   CLAUDE.md. CAVEAT to record: fusing the full s×s assumes it fits SRAM; the
   honest follow-up is an SRAM-capacity constraint that forces tiling for large s
   (a new `impl Constraint`, which is exactly the extensibility thesis).

## Environment / resume
- Rust 1.94 gnu at `C:\Users\bhansa01\.cargo\bin` (User PATH). `python` = 3.14.
- Remote `https://sndsh404@github.com/sndsh404/roofline.git`, creds cached, push
  works non-interactively. Identity `sndsh404 <hiiamsandeshbhandari@gmail.com>`.
- Resume: read CLAUDE.md, README.md, this file. `python scripts/assess.py --start`,
  `cargo test --workspace`, then action 1. M3 is design-heavy: use /plan + high effort.
- Local-only branch `local-backup` holds the earlier parallel line; ignore unless needed.
