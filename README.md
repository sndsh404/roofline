# Roofline

A cost-based tensor program optimizer. Uses egg (e-graphs) to explore equivalent
program forms, then picks the best one via a roofline cost model.

## What it does

Naive attention writes an O(s²) scores matrix to HBM. Flash Attention avoids
that write by tiling. This project asks: can a general-purpose optimizer find
Flash Attention purely from algebraic rewrites and a cost model?

Yes, and the architecture generalizes to MLP fusions and ragged operations too.

## Project structure

```
crates/rl-ir/       IR, reference interpreter, HBM accountant
crates/rl-cost/     Roofline cost model (pluggable constraints)
crates/rl-opt/      egg rewrite rules + extractor
crates/rl-codegen/  Physical plan -> kernel emission (stub)
crates/rl-ledger/   WAL + MVCC run store (stub)
```

## Current state

- M0: Reference interpreter matches JAX fixture to 1e-5. HBM accountant prints
  ground-truth intensity for naive attention.
- M1: Cost model has FlopsConstraint + HbmConstraint. Predicts binding resource
  for any program shape on A100/H100.
- M2: egg rewrite rules active (associativity, transpose fusion, scale
  distribution). E-graph contains equivalent terms with different HBM costs.
  4 tests passing.

## Build & test

```
cargo build --workspace
cargo test --workspace
cargo run -p rl-ir --example m0_numbers
cargo run -p rl-cost --example m1_binding
```

## Design

See `DESIGN.md` for the full spec: algebraic identities, milestone criteria,
device specs, and the architecture rationale.
