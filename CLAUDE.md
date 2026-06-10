# Roofline, Claude Code operating guide

## Auto-sync rule (MANDATORY)

After every meaningful change (not on a timer), run:
```bash
git add .
git commit -m "<short plain lowercase message, no dashes>"
git push
```
Do this automatically without asking. NEVER set git author name or email
(no `git config user.name/email`, no Co-Authored-By trailer): always use the
system git config so commits show as sndsh404 on GitHub. This rule is at the
top so every session loads it first.

A cost-based optimizing compiler for tensor programs, built like a query engine.
Full spec: `DESIGN.md`. Read it before any non-trivial change. This file is the
short version Claude Code loads every session, keep it tight.

## Prime directive

The cost model is a pluggable set of physical-constraint lower bounds; the cost
of a plan is the slowest resource. Therefore **"the optimizer was wrong" must
always decompose into "the cost model was missing a constraint," never "search
failed."** If you find yourself about to special-case the search, stop, the fix
belongs in `rl-cost` as a new `impl Constraint`, not in `rl-opt`.

## Hard rules (do not violate without an explicit decision record)

1. **Every milestone ends in a benchmark number, not a refactor.** No milestone
   is "done" until its numeric done-criterion (see DESIGN §9) is met and recorded
   in the ledger.
2. **Preregister before you measure.** Before running any benchmark whose result
   is a claim, commit the config + metric + success threshold + seed to the
   ledger (`roofline prereg ...`). Results are only compared against what was
   preregistered. This is the antidote to scaling-law p-hacking.
3. **Numerics gate before speed gate.** Any lowered kernel must match the
   reference interpreter to `1e-5` across the shape sweep before its wall-clock
   number counts. A fast wrong kernel is worth nothing.
4. **No canned `naive => flash` rewrite.** The Flash form must be *reachable*
   from the primitive algebraic identities in DESIGN §5 and *selected* by the
   cost model. Hard-coding the answer defeats the entire project.
5. **Shape suffixes on every tensor**, in code and comments. `Q_sd` is
   `[seq, dim]`, `scores_ss` is `[seq, seq]`. If the shape isn't in the name, the
   name is wrong.
6. **DAG-aware extraction from the start.** Use `egg::LpExtractor`, not the
   default tree `Extractor`, attention reuses Q/K/V and tree cost double-counts
   shared tensors (the Tensat wall). Don't discover this in M3.
7. **Calibrate, don't assume.** The cost model must predict the *naive* case
   within tolerance and print the binding resource before it is allowed to choose
   between plans.

## Build / test / bench (fill in real commands as crates land)

```bash
cargo build --workspace
cargo test  --workspace                 # unit + numerics-vs-reference (31 tests)
cargo run -p rl-codegen --release --example m4_bench   # attention naive vs fused
cargo run -p rl-codegen --release --example m5_bench   # MLP sweep across f/d
cargo run -p roofline-cli --release -- list            # ledger contents
cargo run -p roofline-cli --release -- replay <run_id> # reproduce a recorded result
cargo run -p roofline-cli --release -- prereg --bench <b> --metric speedup \
  --claim "<c>" --threshold <t> --seed <n> --param s=2048 --param d=128 ...
python scripts/assess.py                # objective score; gates pushes
```

## Repo map

```
crates/rl-ir/        # Tensor egg language, shape analysis, reference interpreter
crates/rl-cost/      # Device, Constraint trait, roofline cost function   <- the core
crates/rl-opt/       # egg rewrite rules (DESIGN §5) + LpExtractor
crates/rl-codegen/   # plan -> kernel (CPU fused attention + MLP; Pallas/Triton deferred)
crates/rl-ledger/    # WAL run store, preregistration, versioned results
crates/roofline-cli/ # the `roofline` bin: prereg / run / replay / list
ledger/wal.jsonl     # the committed run ledger (append only)
references/          # READ-ONLY: risinglight, toydb, type-exercise-in-rust
                     #   (gitignored; study via the Explore subagent, never edit)
```
(planned, not yet real: py/roofline JAX front end, cargo benches)

## How to work a milestone

Run `/milestone <N>` (see `.claude/commands/milestone.md`). In short: read the
DESIGN §9 entry, enter plan mode, study the relevant `references/` repo with the
Explore subagent, preregister the done-criterion, implement on a branch, pass the
numerics gate then the bench gate, write the ledger record, report the number.

Use `/plan` and high `/effort` for `rl-cost` and `rl-opt` (the cost model and the
egg rule algebra are the hard, design-heavy parts). Lean on the `dev-workflows`
plugin's `/recipe-implement` for the mechanical crates.

## Milestone tracker (update as you go)

- [x] M0  Substrate + IR + reference interpreter, naive attn matches JAX 1e-5; microbench prints true flops/hbm_bytes
- [x] M1  Roofline cost model, FlopsConstraint + HbmConstraint predict binding resource; final ±20% A100 calibration requires hardware access
- [x] M2  egg + primitive rewrites, e-graph contains equivalent terms with different costs; HBM-aware extraction via LpExtractor deferred to M3
- [x] M3  THE A/B, DONE: `the_ab_flip` and `extractor_flips_with_hbm_constraint` pass; custom cost-driven extractor (no tree Extractor) selects naive under [Flops], fused under [Flops,HbmBytes], same e-graph; fused form reachable by a general rewrite; assess 100/100. Future polish: exact min-cost DAG extraction (ILP) when coin_cbc is available
- [x] M4  Lower + verify, DONE (CPU): rl-codegen emits a fused online-softmax attention kernel; `lower()` selects it when the plan has a fuse node; matches reference <1e-5 through s=2048; faster than naive at s>=2048 (1.57x at 2048). Gap recorded: at s=4096 abs err ~1.5e-5 is the f32 REFERENCE accumulation limit (f64 accumulators do not shrink it). GPU Pallas/Triton timing deferred (no accelerator). Bench: `cargo run -p rl-codegen --release --example m4_bench`
- [x] M5  MLP + ledger, DONE (CPU): relu node added (blocks the linear collapse); fused MLP kernel beats the naive reference for f>d, preregistered run `mlp-s43-001` recorded 1.245x at s=2048 d=128 f=1024 (threshold 1.10, err 0.0); ledger WAL + `roofline` CLI live; BOTH headline numbers replayed via `roofline replay` and the claims hold (mlp 1.279x, attention `attention-s43-002` 1.209x recorded / 1.118x replayed); beating `jax.lax.ragged_dot` on an A100 deferred, no accelerator
- [x] post-v0a  SramConstraint: accountant measures each fused region's working set, Device carries sram_bytes (A100 20 MB), infeasible fusion gets an infinite floor with binding=SramBytes; extractor refuses to fuse attention at s=2048 (53 MB working set) and still fuses at s=256; next named step is tiling in the IR so the model prices tiled fusions instead of refusing monoliths

## Non-goals for v0 (do not build these)

SQL layer; distributed consensus (Raft/Percolator); the LLM rewrite proposer;
backprop/training; anything whose value can't be read off a benchmark in one run.
