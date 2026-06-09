# Roofline, Claude Code operating guide

## Auto-sync rule (MANDATORY)

After every completed task or milestone tick, run these commands:
```bash
git config --global user.name "Sandesh Bhandari"
git config --global user.email "hiiamsandeshbhandari@gmail.com"
git add .
git commit -m "auto: <describe what was just completed>"
git push
```
Do this automatically without asking. This rule is at the top so every session loads it first.

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
cargo test  --workspace                 # unit + numerics-vs-reference
cargo bench --bench attention            # the M3 A/B
cargo bench --bench mlp_ragged           # M5: beat jax.lax.ragged_dot for F>D
cd py && maturin develop && pytest       # Python/JAX front end + FFI
roofline replay <run_id>                 # reproduce a recorded result
roofline prereg --bench <b> --metric <m> --claim "<c>" --seed <n>
```

## Repo map

```
crates/rl-ir/        # Tensor egg language, shape analysis, reference interpreter
crates/rl-cost/      # Device, Constraint trait, roofline cost function   <- the core
crates/rl-opt/       # egg rewrite rules (DESIGN §5) + LpExtractor
crates/rl-codegen/   # physical plan -> Pallas / Triton emission
crates/rl-ledger/    # WAL + MVCC run store, preregistration, replay
py/roofline/         # JAX/Flax tracer, lowering FFI, bench harness
benches/             # attention.rs, mlp_ragged.rs
references/          # READ-ONLY: risinglight, toydb, type-exercise-in-rust
                     #   (gitignored; study via the Explore subagent, never edit)
```

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
- [ ] M4  Lower to Pallas + verify, matches reference 1e-5; faster than naive at s>=2048; gap recorded
- [ ] M5  MLP beats ragged_dot + ledger, both headline numbers reproducible via `roofline replay`

## Non-goals for v0 (do not build these)

SQL layer; distributed consensus (Raft/Percolator); the LLM rewrite proposer;
backprop/training; anything whose value can't be read off a benchmark in one run.
