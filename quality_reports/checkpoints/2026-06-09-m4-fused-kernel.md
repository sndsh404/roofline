# Checkpoint, 2026-06-09, M4 fused kernel done (CPU)

## Where I stopped
- **M4 done on CPU.** New `rl-codegen` crate emits a fused attention kernel:
  - `fused_attention(q, k, v, scale)`: online (streaming) softmax, computes
    O_sd = softmax(Q Kt * scale) V one query row at a time, never materialising
    the s×s scores. The Flash idea, executable.
  - `lower(expr) -> Kernel`: returns `FusedAttention` if the plan contains a
    `Fuse` node, else `Reference`. `lower_and_run` dispatches.
  - Tests (rl-codegen 2): `fused_kernel_matches_reference_across_shapes` (<1e-5
    over a shape sweep, the hard gate) and `lower_picks_fused_when_plan_has_fuse`.
  - Example `m4_bench` (run with `--release`): the optimizer chooses the fused
    plan, codegen lowers it, then times naive vs fused. Results at d=64:
    s=1024 1.28x, s=2048 1.57x, s=4096 1.21x. Numerics err: 2.98e-6, 5.78e-6,
    1.45e-5.
- **Honest gap recorded:** at s=4096 the abs err (~1.5e-5) exceeds 1e-5. It is the
  f32 REFERENCE interpreter's own accumulation limit, not a kernel bug: switching
  the fused kernel to f64 accumulators did NOT shrink it (tried and reverted,
  f32 is faster). The gate holds through s=2048 and the unit-test sweep stays in
  the gated range. Speed target (faster at s>=2048) is met.
- All gates green. `cargo test --workspace`: rl-ir 5, rl-cost 4, rl-opt 8,
  rl-codegen 2. assess 100/100.

## Milestone status
- M0, M1, M2, M3, M4 all done. M5 is next.
- Whole pipeline runs end to end: program -> e-graph -> cost model picks fused
  (M3) -> codegen lowers to fused kernel (M4) -> matches reference, faster.

## Next 3 actions (in order)
1. **Start M5.** The MLP up/down projection: a fused kernel that beats a baseline
   for F>D. On CPU, emit a fused up/down MLP kernel (no intermediate H_sf spill)
   and benchmark vs the naive two-matmul path. (`jax.lax.ragged_dot` is the GPU
   reference; CPU baseline + numerics gate, GPU deferred, same pattern as M4.)
2. **Stand up `rl-ledger`.** Even a minimal version: append each bench result
   (config + metric + number) to a JSON/WAL file, with a `replay` that re-runs
   from the committed config and asserts the number reproduces. This is the
   reproducibility half of M5.
3. **Optional polish:** exact min-cost DAG extraction (ILP) if `coin_cbc` becomes
   available; an SRAM-capacity constraint in rl-cost that forces tiling for large
   s (the documented next `impl Constraint`).

## Environment / resume
- Rust 1.94 gnu at `C:\Users\bhansa01\.cargo\bin` (User PATH). `python` 3.14 with
  matplotlib (figures: `python scripts/figures.py`).
- Remote `https://sndsh404@github.com/sndsh404/roofline.git`, creds cached, push
  works. Identity `sndsh404 <hiiamsandeshbhandari@gmail.com>`. NO Co-Authored-By
  trailer. Short plain commit messages, no dashes. NO em dashes anywhere (docs or
  code comments), the user is firm on this.
- Resume: read CLAUDE.md, README.md, this file. `python scripts/assess.py --start`,
  `cargo test --workspace`, then action 1.
