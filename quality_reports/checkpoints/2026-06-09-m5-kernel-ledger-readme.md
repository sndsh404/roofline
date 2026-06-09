# Checkpoint, 2026-06-09, M5 mid-flight: kernel + ledger done, prereg/replay next

## Where I stopped
M5 is roughly 80 percent done. Everything below is implemented, tested, and
green (31 tests across the workspace, `cargo test --workspace`).

- **rl-ir**: new `relu` node (eval, infer_shape, accountant, fused_walk).
  `naive_mlp_program()` is now `relu(X_sd W_up_df) W_dn_fd` with DESIGN-aligned
  names (d = model width, f = hidden, the M5 regime is f > d). The relu is
  load bearing: without it assoc collapses the two matmuls into one and the
  right answer is the collapse, not fusion. `fused_mlp_program()` added.
  tests/fuse.rs has MLP numerics + cuts-HBM tests.
- **rl-opt**: general `fuse-relu-matmul` rewrite (same family as the softmax
  one, rule-4 honest). Tests: `mlp_fused_form_is_reachable_by_rewrite`,
  `mlp_extractor_flips_with_hbm_constraint` (the M5 A/B, passes).
- **rl-codegen**: `fused_mlp` streaming kernel (row at a time, no s by f
  spill, same arithmetic order as the reference so err is exactly 0.0).
  `Kernel::FusedMlp`, `lower()` dispatches softmax chain to FusedAttention,
  relu chain to FusedMlp. New `bench` module: `bench_attention(s,d,iters,seed)`
  and `bench_mlp(s,d,f,iters,seed)` return a `BenchOutcome`; examples AND the
  CLI replay share this exact code path. New example `m5_bench`.
- **rl-ledger**: real WAL, JSONL append-only, torn-final-line recovery,
  `Prereg` + versioned `RunResult` (v1 = original, v2+ = replay), claim_met =
  numerics gate FIRST then speed threshold. 5 tests.
- **roofline-cli** (new crate, bin name `roofline`): subcommands prereg, run
  (first measurement, refuses if one exists), replay (re-measures, requires a
  prior result, exits nonzero if the preregistered claim fails), list. Default
  ledger path `ledger/wal.jsonl`. Always run with `--release` for timing:
  `cargo run -p roofline-cli --release -- <cmd>`.
- **PILOT result (NOT a claim, exploratory, seed 42)**: m5_bench at s=2048
  d=128: speedup 1.57x at f=64 down to 1.25-1.26x for f >= 512, err 0.0 at all
  shapes, accountant hbm naive/fused at f=1024 is 19 MB vs 3 MB. Fused wins
  everywhere; CPU is compute-bound so the margin is modest, the byte cut is
  6x. Headline shape chosen: s=2048, d=128, f=1024.
- **README fully rewritten** in the user's plain blog voice (lowercase
  headings, no long dashes, no AI words, story first). **Figures restyled to
  the user's hand-drawn black-and-white xkcd style** (docs/figures/*.png,
  scripts/figures.py): Humor Sans committed at docs/fonts/HumorSans.ttf,
  notebook ruling background, b/w only. User confirmed the direction after
  seeing their blog PDFs; their diagrams ARE matplotlib xkcd mode.

## Next 3 actions (in order)
1. **Preregister and run the two headline claims** (this is the only thing
   left for M5 besides docs). Use a FRESH seed (43, pilot was 42):
   `cargo run -p roofline-cli --release -- prereg --bench mlp --metric speedup
   --claim "fused MLP beats the naive two-matmul reference for f>d at s=2048
   d=128 f=1024 on CPU" --threshold 1.10 --seed 43 --param s=2048 --param
   d=128 --param f=1024 --param iters=7`
   then the M4 one: `... prereg --bench attention --metric speedup --claim
   "fused attention beats naive at s=2048 d=64 on CPU" --threshold 1.0 --seed
   43 --param s=2048 --param d=64 --param iters=5`
   then `roofline run <each id>`, then `roofline replay <each id>` to prove
   reproducibility. Commit ledger/wal.jsonl.
2. **Tick M5 in CLAUDE.md tracker + update README section 12** with the
   recorded numbers and the run ids; note ragged_dot-on-A100 deferred
   (hardware), CPU criterion met. Maybe teach scripts/assess.py to capture the
   ledger claim_met values as numbers.
3. **Write the M5-done checkpoint**, run `python scripts/assess.py && git
   push`. Optional follow-ups: SRAM capacity constraint, exact ILP extraction.

## Environment / resume
- Rust 1.94 gnu at `C:\Users\bhansa01\.cargo\bin`. Python 3.14, matplotlib
  with Humor Sans registered from docs/fonts (regenerate figures via
  `python scripts/figures.py`).
- Remote `https://github.com/sndsh404/roofline.git`. NEVER set git author
  config or Co-Authored-By; system config is sndsh404, push after every
  meaningful change (user instruction, also in memory + CLAUDE.md).
- NO long dashes anywhere, plain simple prose, figures b/w hand-drawn style
  (user is firm; see memory files).
- Resume: read CLAUDE.md, README, this file; `python scripts/assess.py
  --start`; `cargo test --workspace` (expect 31); then action 1.
