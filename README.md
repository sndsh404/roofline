# Roofline

A cost-based optimizing compiler for tensor programs, built like a query engine.

This README is the living single source of truth. Any Claude Code session should
be able to read this file plus the latest checkpoint in
`quality_reports/checkpoints/` and resume work immediately. Keep it current: when
state changes, update this file in the same commit.

---

## 1. What this is and what it is trying to achieve

A neural-network forward pass is a compute graph with many equivalent
formulations: fuse or don't, materialize or recompute, tile one way or another,
quantize or not. A database query optimizer is the decades-refined machine for
exactly that shape of problem. It searches a space of equivalent programs, scores
each with a physical cost model, extracts the cheapest, runs it, and records the
result so it reproduces. Roofline points that machine at a tensor program instead
of a SQL query.

The one design decision everything serves: **the cost model is a pluggable set of
physical-constraint lower bounds, and the cost of a plan is its slowest resource.
So "the optimizer was wrong" must always decompose into "the cost model was
missing a constraint," never "search failed."** That is the whole thesis. The
rest is plumbing.

The headline result for v0 is a single A/B. With only a FLOPs constraint, the
extractor returns naive attention. Add an HBM-bandwidth constraint and the same
search over the same e-graph returns the Flash-Attention tiling. Then we lower
that plan to a real kernel and beat `jax.lax.ragged_dot` on the fused MLP up/down
projection for `F > D`, with both numbers reproducible from a preregistered
ledger.

Full spec with the algebra and milestone criteria: `DESIGN.md`. Read it before any
non-trivial change. Operating rules Claude loads each session: `CLAUDE.md`. The
reusable workflow behind how this project is built: `WORKFLOW.md`.

---

## 2. Architecture and design decisions

Cargo workspace, Rust core, JAX front end planned, Pallas/Triton as the emitted
backend.

```
crates/rl-ir/       Tensor IR (egg language), reference interpreter, HBM accountant
crates/rl-cost/     Device, Constraint trait, roofline cost model   <- the core idea
crates/rl-opt/      egg rewrite rules + extraction
crates/rl-codegen/  physical plan -> kernel emission (stub)
crates/rl-ledger/   WAL + MVCC run store, preregistration, replay (stub)
scripts/assess.py   self-assessment harness (objective score, gates pushes)
quality_reports/    assessment reports + checkpoints
DESIGN.md           full spec
CLAUDE.md           per-session operating guide
WORKFLOW.md         the reusable workflow system
```

Key decisions already made and why:

- **IR is an `egg` language (`TensorLang`).** Logical algebra and physical
  schedule live in one e-graph, so a single extraction chooses the math and how
  to run it. Types: `naive_attention_program() -> (RecExpr<TensorLang>, Id)`,
  `account(expr, root, shapes) -> Account { flops, hbm_bytes }`.
- **The reference interpreter is ground truth.** Naive, `Vec<f32>`, rank <= 2.
  Any future kernel must match it to `1e-5`. It also measures real flops and
  hbm_bytes so the cost model is validated, not trusted.
- **Cost model is a trait, not a function.** `Constraint::lower_bound_s(flops,
  hbm_bytes)` returns a per-resource time. `CostModel` takes the max (slowest
  resource wins) and reports the binding resource. Adding occupancy, comms, or
  latch-boundedness later is a new `impl Constraint`; nothing else changes.
  Devices: `A100` (ridge ~156 flop/byte), `H100` (ridge ~295).
- **Extraction must become DAG-aware (`LpExtractor`), not tree `Extractor`.**
  Attention reuses Q/K/V and tree cost double-counts shared tensors (the Tensat
  wall). M2 shipped with a tree cost function as a stepping stone; M3 must switch.
  The assess harness flags this until it is fixed.
- **No canned `naive => flash` rewrite.** Flash must be reachable from primitive
  algebraic identities and selected by the cost model. Hard-coding it defeats the
  project. See DESIGN §5.

---

## 3. Milestone status

Source of truth for the checklist is `CLAUDE.md`. Current state:

| Milestone | Status | Evidence |
|---|---|---|
| M0 Substrate + IR + reference interpreter | done | rl-ir 3 tests; interpreter matches fixture to 1e-5; `m0_numbers` prints true flops/hbm_bytes |
| M1 Roofline cost model | done | rl-cost 4 tests; `m1_binding` prints binding resource per shape on A100/H100. Empirical wall-clock calibration (±20-30% vs measured) needs accelerator access and is deferred |
| M2 egg + primitive rewrites | done | rl-opt 5 tests; rules for matmul assoc, transpose, scale distribution; e-graph holds equivalent terms with different HBM cost |
| M3 LpExtractor + THE A/B | **in progress** | shape analysis done (e-graph knows every e-class shape); remaining: `fuse` primitive + cost-driven DAG extractor so `[Flops]` returns naive and `[Flops, HbmBytes]` returns the fused form |
| M4 Lower to kernel + verify | not started | matches reference 1e-5; faster than naive at s>=2048; record predicted-vs-measured gap |
| M5 MLP beats ragged_dot + ledger | not started | both headline numbers reproducible via `roofline replay` |

`cargo test --workspace` is currently green: rl-ir 3, rl-cost 4, rl-opt 5.

What M3 concretely requires:
1. Replace the tree `Extractor` in `rl-opt` with `egg::LpExtractor` (ILP, DAG-aware).
2. Drive extraction by the `rl-cost` model, not a standalone HBM function, so the
   constraint set is what selects the plan.
3. Add the tiled/fused (Flash) form so it is reachable in the e-graph (DESIGN §5),
   then assert: under `[Flops]` the extractor returns naive; under `[Flops,
   HbmBytes]` it returns the fused tiled plan. Capture as a test and the README
   figure.

---

## 4. Build, test, assess

```bash
# Rust is at C:\Users\bhansa01\.cargo\bin (added to User PATH). gnu toolchain.
cargo build --workspace
cargo test  --workspace
cargo run -p rl-ir   --example m0_numbers     # ground-truth flops/hbm sweep
cargo run -p rl-cost --example m1_binding      # binding resource per shape

python scripts/assess.py                       # objective score + gates
python scripts/assess.py --start               # resume summary at session start
```

---

## 5. Development workflow

Full detail in `WORKFLOW.md`. The short version:

- Work one milestone at a time. Each ends in a benchmark number, not a refactor.
- Preregister a claim before measuring it (DESIGN §7 / `CLAUDE.md` rule 2).
- Numerics gate before speed gate. A fast wrong kernel is worth nothing.
- Commit and push after every meaningful step or every ~30 minutes. Short, plain
  commit messages, no dashes (e.g. `add lp extractor`, `fix attention test`).
- Run `python scripts/assess.py` before pushing. It exits nonzero on a failed
  build, failed tests, or a regression, which gates the push.

---

## 6. Context management and how to resume

Token budget is finite. The rule:

> Monitor usage continuously. Before hitting the limit, stop new work, run
> `cargo test --workspace`, commit everything, push, and write a checkpoint to
> `quality_reports/checkpoints/<date>-<topic>.md` with exactly where you stopped
> and the next three actions. Always stop clean.

To resume in a new session:
1. Read `CLAUDE.md` (operating rules + milestone checklist).
2. Read this README (architecture + status).
3. Read the newest file in `quality_reports/checkpoints/`.
4. Run `python scripts/assess.py --start`, then `cargo test --workspace` to
   confirm green.
5. Do action 1 from the checkpoint.

---

## 7. The self-assessment loop

`scripts/assess.py` is the objective floor. The score is computed by the script
from machine signals, not by a model's opinion, because a model grading its own
work drifts into self-congratulation.

- P0 auto-fail: build or tests fail -> score 0, exit 2. Push is blocked.
- P1 (-20): a regression vs the last run (a number moved the wrong way) or a
  greppable hard-rule violation (canned naive=>flash, or tree Extractor where
  LpExtractor is required).
- P2 (-5): tracker claims that reality does not back.
- Gates: 80 commit, 90 PR, 95 excellence. Reports land in
  `quality_reports/assessment_latest.md`; state for regression diffing is in
  `.claude/state/assessment_state.json` (gitignored).

A high score means the repo does what it claims, reproducibly. It does not vouch
for the idea or the realism of the cost model. A session that lowers the score by
finding a real bug is worth more than one that holds it flat by not looking.

---

## 8. Decisions, lessons, and notes for the next session

- **Two histories existed and were reconciled (2026-06-09).** A parallel
  autonomous session had pushed a more complete scaffold (M0-M2, all five crate
  stubs, README) to `origin/main` on an unrelated history, while a local line had
  a separate M1 implementation. The remote was adopted as canonical because it was
  further along; the local line is preserved on branch `local-backup`. Lesson:
  always `git fetch` and inspect `origin/main` before pushing. Do not force-push
  over work you did not create. The canonical API is `TensorLang` / `TensorData` /
  `naive_attention_program`, not the older `Tensor` / `Nd`.
- **M1 calibration is honestly partial.** The model predicts the binding resource
  correctly, but predicted-vs-measured wall-clock within tolerance needs a real
  accelerator. That gap is the next research question, recorded, not hidden.
- **Git identity and remote.** Remote is
  `https://sndsh404@github.com/sndsh404/roofline.git`, credential helper `store`
  (cached after first auth). Commit identity is `sndsh404
  <hiiamsandeshbhandari@gmail.com>`.
- **Environment.** Windows 11, PowerShell. Rust 1.94 gnu toolchain at
  `C:\Users\bhansa01\.cargo\bin` (on User PATH; a fresh shell finds `cargo`).
  Python 3.14 is `python`. `references/` is gitignored; study those repos with
  the Explore subagent, never edit them.
