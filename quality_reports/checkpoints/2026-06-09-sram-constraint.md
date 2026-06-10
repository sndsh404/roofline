# Checkpoint, 2026-06-09, post-v0a: the SRAM capacity constraint

## Where I stopped
The first post-v0 item is done, on this laptop, since it is pure modeling
work. 34 tests green (was 31), assess 100/100.

What landed, and it is the prime directive made literal (a missing constraint
added in one place, no search changes):
- `rl-ir::Account` gained `sram_bytes`: the peak working set of any fused
  region (boundary inputs + output + every internal intermediate, all
  resident at once under the monolithic no-tiling fuse model). `fused_walk`
  now counts intermediate bytes. Test `fused_account_measures_sram_working_set`
  pins the arithmetic by hand at s=256 d=32: 98,308 + 32,768 + 819,200.
- `rl-cost` gained `Demand { flops, hbm_bytes, sram_bytes }` (the Constraint
  trait and `CostModel::cost` now take `&Demand`; `From<&Account>` provided).
  `Device` gained `sram_bytes` (A100 20 MB = 192KB x 108 SMs, H100 30 MB =
  228KB x 132 SMs, the FlashAttention accounting). New `SramConstraint`:
  fits means a 0.0 floor, does not fit means f64::INFINITY and the binding
  resource reads "SramBytes", the model saying why the plan cannot run.
- `rl-opt` extractor stage 2 passes the full Demand, so an infeasible fusion
  loses to the materialized plan automatically. Test
  `sram_constraint_blocks_fusion_that_cannot_fit`: with [Flops, Hbm, Sram] on
  A100, attention at s=2048 d=64 (53 MB working set) is NOT fused, at s=256
  d=32 (under 1 MB) it still fuses. Existing tests untouched and green
  (the M3/M5 flips use [Flops, Hbm] and behave exactly as before).
- Honest nuance, written in README section 10: the streaming kernels in
  rl-codegen prove tiling rescues exactly these fusions, but the IR cannot
  express a tiled schedule, so the constraint is conservative on purpose.
  The benches keep using [Flops, Hbm] for that reason.
- Docs updated: README sections 8, 10, 14, 17; CLAUDE.md tracker has a
  post-v0a line.

## Next actions (in preference order)
1. **Tiling in the IR.** The named next step: represent a tiled/streaming
   fused schedule so the accountant can price its true (small) working set
   and the SRAM constraint admits it at large s. Sketch: a `tile` annotation
   or a fuse variant whose working set is per-tile, with the reference
   interpreter still treating it as value identity. Done when: with
   [Flops, Hbm, Sram] active, the extractor picks a tiled fusion at s=2048
   (where it refuses the monolith today) and the numerics gate still holds.
2. Exact ILP extraction if a CBC solver becomes available.
3. GPU work when hardware exists (Pallas emission, A100 calibration, the
   ragged_dot comparison).

## Environment / resume
Same as the m5-done checkpoint. 34 tests expected. Remote
`https://github.com/sndsh404/roofline.git`, system git config (sndsh404),
never set author or trailers, push after every meaningful change, no long
dashes anywhere.
