# Roofline

A cost-based optimizing compiler for tensor programs, built like a query engine.

Roofline takes a neural-network computation, represents every equivalent way of
running it in one structure, scores each by the physical resource that would
actually bottleneck it, and picks the cheapest. It is the same machine a database
uses to optimize a SQL query, pointed at a tensor program instead.

This README is meant to take you from zero to the full picture. It starts with the
plain ideas (what a tensor program is, why the same math has many forms, what
"memory-bound" means), then walks every crate with the real code and the reasons
behind each decision, then the milestones, the development process, the honest
limitations, and a glossary. It is long on purpose and it is a living document:
every milestone updates it.

Companion files: `DESIGN.md` is the formal spec, `WORKFLOW.md` is the development
process, `CLAUDE.md` is the short operating guide loaded each session.

---

## Table of contents

1. [The problem, in plain terms](#1-the-problem-in-plain-terms)
2. [The one idea, and the rule it forces](#2-the-one-idea-and-the-rule-it-forces)
3. [The headline result: the A/B flip](#3-the-headline-result-the-ab-flip)
4. [Architecture: the five crates](#4-architecture-the-five-crates)
5. [rl-ir: the language, interpreter, and accountant](#5-rl-ir-the-language-interpreter-and-accountant)
6. [rl-cost: the roofline cost model](#6-rl-cost-the-roofline-cost-model)
7. [rl-opt: e-graphs, rewrites, shapes, extraction](#7-rl-opt-e-graphs-rewrites-shapes-extraction)
8. [The fuse primitive: how fusion saves HBM](#8-the-fuse-primitive-how-fusion-saves-hbm)
9. [The milestones, M0 through M5](#9-the-milestones-m0-through-m5)
10. [Design decisions and trade-offs](#10-design-decisions-and-trade-offs)
11. [Honest limitations and prior art](#11-honest-limitations-and-prior-art)
12. [Build, test, run, resume](#12-build-test-run-resume)
13. [Glossary](#13-glossary)
14. [Roadmap](#14-roadmap)

---

## 1. The problem, in plain terms

### What a tensor program is

A neural network, at the level that matters here, is arithmetic on big rectangular
arrays of numbers. An array is a **tensor**. A matrix `[rows, cols]` is a 2-D
tensor. Attention multiplies matrices, takes a softmax, multiplies again. An MLP
multiplies by one weight matrix, then another. Strip away the framework and a
forward pass is a small **graph of operations**: inputs flow in, matmuls and
elementwise ops transform them, an output flows out.

Throughout this project every tensor name carries its shape as a suffix. `Q_sd` is
`[seq, dim]`. `scores_ss` is `[seq, seq]`. `O_sd` is the output `[seq, dim]`. If a
variable's shape is not in its name, the name is considered wrong. This is not
decoration: it lets you read a cost argument at a glance, because the cost of an
op is a function of the shapes it touches.

### Why the same computation has many forms

Here is the key fact that makes optimization possible: **the same mathematical
result can be computed in many different ways**, and they cost wildly different
amounts on real hardware. Examples:

- `(A · B) · C` equals `A · (B · C)` — same answer, different amount of work
  depending on the matrix sizes.
- You can compute the whole `scores = Q · Kᵀ` matrix, store it, then read it back
  to softmax it — or you can compute it in tiles and never store the whole thing.
- You can keep an intermediate in fast memory and consume it immediately, or you
  can write it out to slow memory and read it back later.

All of these are the *same math*. They differ only in **schedule**: the order,
the tiling, what gets stored where. Choosing the cheapest schedule is the entire
game.

### Why "the bottleneck is not the math"

The instinct is to count arithmetic — floating point operations, FLOPs — and
assume fewer FLOPs means faster. On modern accelerators this is usually wrong. A
GPU can do hundreds of trillions of FLOPs per second, but moving data between its
big-but-slow memory (HBM) and its small-but-fast memory (SRAM) is comparatively
glacial. Many real kernels spend their time **waiting for memory**, not computing.

The standard way to see this is the **roofline**. Plot a program's *arithmetic
intensity* — how many FLOPs it does per byte it moves — against the performance it
can achieve. Below a certain intensity (the "ridge point") you are limited by
memory bandwidth; above it you are limited by compute.

![the roofline, attention plotted on it](docs/figures/roofline.png)

*the roofline. low arithmetic intensity (few flops per byte moved) means HBM
bandwidth is the wall; high intensity means compute is. naive attention sits far
to the left at around 9 flops per byte, deep in the bandwidth-bound region — which
is exactly what fusion attacks by cutting the bytes.*

This is why Flash Attention was a breakthrough even though it does *the same
math* as naive attention and even a few *more* FLOPs: it removes the giant
`seq×seq` score matrix from the memory traffic. The win was a memory win, found by
a human who understood the real constraint. The thesis of this project is that a
machine with the right cost model would have found it automatically.

### The query-optimizer analogy

Databases solved a structurally identical problem decades ago. A SQL query has many
equivalent execution plans (which table to scan first, which join algorithm, which
index). A **query optimizer** searches the space of equivalent plans, scores each
with a **cost model** based on physical costs (disk I/O, CPU, memory), and picks
the cheapest, then executes it and records the result.

Swap "SQL query" for "tensor program" and "disk I/O" for "HBM bandwidth" and you
have Roofline. The machinery is the same; only the cost model changes. That reuse
is the whole bet.

---

## 2. The one idea, and the rule it forces

Everything in this project serves a single design decision:

> **The cost model is a pluggable set of physical-constraint lower bounds, and a
> plan's cost is its slowest resource.**

A "constraint" is one physical resource (compute, memory bandwidth, later:
occupancy, communication, SRAM capacity). Each constraint looks at a plan and says
"given my resource, this cannot run faster than *t* seconds." The roofline says the
slowest resource wins, so the plan's predicted time is the **maximum** over all
constraints, and the constraint that produced that maximum is the **binding**
resource.

This forces a discipline that is the soul of the project:

> When the optimizer picks a bad plan, the cause is always **"the cost model was
> missing a constraint,"** never **"the search failed."**

If the optimizer is wrong, you do not patch the search with a special case. You ask
which physical constraint it failed to model, and you add it — a new
`impl Constraint`, usually about twenty lines. The failure is named and localized
by construction. This is what makes the system extensible instead of a pile of
heuristics, and it is the reason the cost model, not the search, is where the
intelligence lives.

---

## 3. The headline result: the A/B flip

The v0 result that proves the thesis is a single experiment. Build one e-graph that
contains both naive attention and the fused (Flash-like) form. Extract the cheapest
plan twice, changing only the cost model:

![the A/B flip: one e-graph, two winners](docs/figures/ab_flip.png)

*the whole thesis in one picture. same search over the same e-graph. model only
FLOPs and the cheapest plan is naive attention. add the HBM-bandwidth constraint
and the winner flips to the fused form. the optimizer was never wrong, the cost
model was incomplete.*

- With `constraints = [Flops]`, the cost model cannot see memory traffic at all, so
  it has no reason to prefer fusion. It returns naive attention. This is the exact
  local optimum a FLOPs-only mindset gets stuck in.
- With `constraints = [Flops, HbmBytes]`, the model now sees that naive attention
  round-trips a huge `seq×seq` matrix through HBM, and the fused form does not. The
  fused form wins.

Same search, same e-graph, one extra constraint. That flip *is* the thesis made
executable — and it now **runs as a passing test**, `the_ab_flip` in `rl-opt`:

```rust
let candidates = [naive, fused];                 // both reachable in one e-graph
let flops_only = CostModel::new().add(FlopsConstraint::new(H100));
let with_hbm   = flops_only.clone().add(HbmConstraint::new(H100));

assert_eq!(select_plan(&candidates, &shapes, &flops_only), 0); // keeps naive
assert_eq!(select_plan(&candidates, &shapes, &with_hbm),    1); // chooses fused
```

A companion test, `fused_form_is_reachable_by_rewrite`, proves the fused candidate
is not hand-built: a general fusion rewrite (`(matmul (softmax ?x) ?v) => (fuse
…)`) places it in the same e-class as the naive root during saturation. So the form
is *reached* by the algebra and *selected* by the cost model — the two halves rule
4 demands. The one piece still open is general DAG extraction over an arbitrary
e-graph (the cost model currently selects among the reachable candidate plans); see
§7 and the roadmap.

---

## 4. Architecture: the five crates

A Cargo workspace. Rust for the core (a complete distributed database like `toydb`
fits in ~15k lines of Rust, so a one-person optimizer core is realistic), with a
JAX front end planned and Pallas/Triton as the eventual kernel backend.

![the pipeline](docs/figures/pipeline.png)

*the pipeline. a program becomes an e-graph of equivalent forms, the cost model
scores each form by its slowest resource and picks the cheapest, and the chosen
plan is lowered to a kernel and recorded so it reproduces.*

| crate | role | status |
|---|---|---|
| `rl-ir` | the tensor IR (an `egg` language), a naive reference interpreter that *defines* correctness, and the accountant that measures true FLOPs and HBM bytes | working |
| `rl-cost` | `Device`, the `Constraint` trait, and the roofline cost model — the core idea | working |
| `rl-opt` | the `egg` e-graph, the rewrite rules, the shape analysis, and extraction | working (extraction is the M3 frontier) |
| `rl-codegen` | lowering a chosen physical plan to a Pallas/Triton kernel | stub |
| `rl-ledger` | a write-ahead-log + MVCC store for preregistered, replayable results | stub |

The dependency direction is strict: `rl-cost` and `rl-opt` depend on `rl-ir`;
nothing depends on `rl-codegen` or `rl-ledger` yet. The reason the cost model is its
own crate, separate from the optimizer, is the prime directive: the cost model must
be independently testable and extensible without touching search code.

---

## 5. rl-ir: the language, interpreter, and accountant

`rl-ir` is the foundation and was built first on purpose. It is the least glamorous
crate and the most load-bearing: it defines what a program *is*, what *correct*
means, and what the true resource costs are. Every later claim stands on it.

### The IR is an `egg` language

A program is a term in a small language defined with the `egg` library's
`define_language!` macro. `egg` is an e-graph library (explained fully in §7); the
important part now is that defining the IR as an `egg` language means the same data
structure the interpreter walks is the one the optimizer rewrites — no translation
layer, no drift.

```rust
define_language! {
    pub enum TensorLang {
        Var(Symbol),                 // a named input tensor, e.g. Q_sd
        "matmul"    = MatMul([Id; 2]),   // [m,k] x [k,n] -> [m,n]
        "transpose" = Transpose([Id; 1]),// [m,n] -> [n,m]
        "emul"      = EMul([Id; 2]),     // elementwise multiply; a scalar broadcasts
        "softmax"   = Softmax([Id; 1]),  // numerically stable softmax over the last axis
        "fuse"      = Fuse([Id; 1]),     // run as one kernel; intermediates stay in SRAM
    }
}
```

Each `Id` is a reference to a child node. The set is deliberately tiny — just enough
to express naive attention and a two-layer MLP. That is a trade-off: a small
language is easy to reason about and easy to give a correct interpreter, at the cost
of not expressing arbitrary programs yet. v0 does not need generality; it needs one
honest end-to-end result.

The two canonical programs are built directly:

```rust
// O_sd = softmax((Q_sd · K_sdᵀ) · scale) · V_sd
pub fn naive_attention_program() -> (RecExpr<TensorLang>, Id) { ... }

// Y = (X_sf · W_up_fi) · W_dn_io   (the up/down projection M5 will fuse)
pub fn naive_mlp_program() -> (RecExpr<TensorLang>, Id) { ... }
```

### The reference interpreter defines truth

The interpreter evaluates a `TensorLang` term on plain `Vec<f32>` tensors with the
most obvious, naive implementation possible — triple-loop matmul, straightforward
softmax. It is not fast and is not meant to be. Its job is to be *so obviously
correct* that it can serve as the oracle: any optimized kernel the system ever
produces must match this interpreter to `1e-5`. A fast wrong kernel is worth
nothing, so correctness is anchored to something no one can doubt.

```rust
pub struct TensorData { pub shape: Vec<usize>, pub data: Vec<f32> }

pub fn eval(expr: &RecExpr<TensorLang>, root: Id,
            env: &HashMap<String, TensorData>) -> TensorData {
    match expr[root].clone() {
        TensorLang::Var(sym)        => env[sym.as_str()].clone(),
        TensorLang::MatMul([a, b])  => matmul_2d(&eval(..a..), &eval(..b..)),
        TensorLang::Transpose([a])  => transpose_2d(&eval(..a..)),
        TensorLang::EMul([a, b])    => emul_broadcast(&eval(..a..), &eval(..b..)),
        TensorLang::Softmax([a])    => softmax_last(&eval(..a..)),
        TensorLang::Fuse([a])       => eval(..a..),   // identity on value
    }
}
```

The softmax is the numerically stable version (subtract the row max before
exponentiating) because a naive `exp` overflows. That detail matters: the oracle
has to be *correct*, including numerically, or the 1e-5 gate is meaningless.

### The accountant measures the real costs

The cost model cannot be trusted if it is never checked against reality. The
accountant walks a program and returns the true FLOPs and HBM bytes under an
explicit, simple model:

```rust
pub struct Account { pub flops: u64, pub hbm_bytes: u64 }
```

The HBM model is **"every intermediate is materialized to HBM"**: each operation
reads its inputs from HBM and writes its output back. Per node:

- `MatMul [m,k]·[k,n]`: `2·m·k·n` FLOPs, writes `m·n·4` bytes (f32 = 4 bytes).
- `EMul` over `n` elements: `n` FLOPs, writes `n·4` bytes.
- `Softmax` over `n` elements: `~4·n` FLOPs (max, exp, sum, divide), writes `n·4`.
- `Transpose`: 0 FLOPs, writes the transposed copy.
- `Var` (input): reads `numel·4` bytes from HBM.

This is deliberately the *naive, pessimistic* model — it is the baseline the whole
project exists to beat. It is honest about being crude: real hardware overlaps
compute and memory, and not every intermediate truly spills. Calibrating this model
against measured wall-clock is M1's job, and the gap between predicted and measured
is treated as the next research question, not hidden. The trade-off here is
intentional: a simple, transparent model you can fully reason about, with its
inaccuracy made explicit and turned into future work, beats a complicated model you
cannot audit.

`Account::intensity()` returns `flops / hbm_bytes` — the arithmetic intensity that
places a program on the roofline. For naive attention it comes out around 9–11
flops/byte regardless of sequence length, which is exactly why attention is
bandwidth-bound: that number sits far below any accelerator's ridge point (~156 for
A100, ~295 for H100).

---

## 6. rl-cost: the roofline cost model

This crate is the core idea in code. It is small, and that smallness is the point —
the value is in the shape of the abstraction, not the volume.

### Device

A device is its peak resources:

```rust
pub struct Device { pub name: &'static str, pub peak_flops: f64, pub hbm_bandwidth: f64 }

pub const A100: Device = Device::new("A100-80GB", 312e12, 2.0e12);
pub const H100: Device = Device::new("H100-SXM", 989e12, 3.35e12);

impl Device { pub fn ridge_point(&self) -> f64 { self.peak_flops / self.hbm_bandwidth } }
```

The **ridge point** is peak FLOPs divided by peak bandwidth: the arithmetic
intensity at which a program stops being memory-bound and starts being
compute-bound. A100 sits at ~156 flops/byte, H100 at ~295. Anything below the ridge
is bandwidth-limited.

### The Constraint trait

A constraint maps a program's demand on one resource to a lower bound on time:

```rust
pub trait Constraint {
    fn name(&self) -> &str;
    fn lower_bound_s(&self, flops: u64, hbm_bytes: u64) -> f64;
}

struct FlopsConstraint { device: Device }   // flops / peak_flops
struct HbmConstraint   { device: Device }   // hbm_bytes / hbm_bandwidth
```

`FlopsConstraint` returns `flops / peak_flops`: the time if compute were the only
limit. `HbmConstraint` returns `hbm_bytes / hbm_bandwidth`: the time if bandwidth
were the only limit. Each is a *lower bound* — the resource cannot go faster than
its peak, so the op cannot finish sooner than this.

### The cost model: slowest resource wins

```rust
pub struct CostModel { constraints: Vec<Box<dyn Constraint>> }

impl CostModel {
    pub fn add(mut self, c: impl Constraint + 'static) -> Self { ... }   // builder
    // returns (best_time_s, binding_resource_name, per_constraint_times)
    pub fn cost(&self, flops: u64, hbm_bytes: u64) -> (f64, String, Vec<(String, f64)>) { ... }
}
```

`cost()` evaluates every constraint and returns the **maximum** time (the slowest
resource), the name of the constraint that produced it (the **binding** resource),
and the full breakdown. The breakdown is kept because the binding resource is the
whole diagnostic value: the model does not just say "1.6 ms," it says "1.6 ms,
HBM-bound." That label is what tells a human (or the optimizer) *what to fix*.

The extensibility is the entire payoff. Adding occupancy, communication volume, or
SRAM capacity is a new struct implementing `Constraint` plus a field on `Device`.
The `cost()` method never changes. That is what makes "the cost model was missing a
constraint" a small, local fix.

### Calibration, honestly

M1's real done-criterion is that the model predicts the *naive* case within a
tolerance of measured wall-clock and prints the binding resource. The binding-resource
half is done and verifiable on any machine (`cargo run -p rl-cost --example
m1_binding` prints `binding=HbmBytes` for attention across a shape sweep). The
"within X% of measured wall-clock" half genuinely needs an accelerator to measure
against, so it is deferred and labelled as such rather than faked. That deferral is
recorded in the milestone tracker and the checkpoints; pretending it is done would
violate the project's own rules.

---

## 7. rl-opt: e-graphs, rewrites, shapes, extraction

This crate is where equivalent forms are generated and the cheapest is chosen. It is
the most conceptually dense, so this section builds the e-graph idea from scratch.

### What an e-graph is, and why

Suppose you have a rule `(A·B)·C = A·(B·C)`. If you rewrite a program by *replacing*
the left side with the right side, you have thrown away the original — and maybe the
original was better. You want to keep *both* and decide later. Do that across many
rules and you get a combinatorial explosion of equivalent programs to store.

An **e-graph** (equivalence graph) stores all of them compactly. It groups
expressions into **e-classes**, where every member of an e-class is proven equal to
every other. A node's children point to e-classes, not to single expressions, so one
node can represent many concrete trees at once. **Equality saturation** is the
process of applying all rewrite rules over and over until no rule adds anything new
(the e-graph is "saturated"). At that point the e-graph holds, compactly, the entire
space of programs reachable by your rules.

This is exactly the structure the optimizer needs: generate every equivalent form,
then extract the cheapest. `egg` is the Rust library that provides it, and
`risinglight` (an educational analytic database) demonstrates using `egg` for query
optimization — the direct template for using it here on tensors.

### The rewrite rules

The rules are the algebra. Each one encodes a mathematical identity that produces an
equivalent program. They are written generically over the e-graph's analysis so the
same rules work with or without shape information:

```rust
// associativity: regroup chained matmuls (changes work, not result)
"(matmul (matmul ?a ?b) ?c)"  <=>  "(matmul ?a (matmul ?b ?c))"

// transpose distributes over matmul (and back)
"(transpose (matmul ?a ?b))"  =>   "(matmul (transpose ?b) (transpose ?a))"

// a scalar scale can move across a matmul to whichever side is cheaper
"(matmul (emul ?a ?s) ?b)"    <=>  "(emul (matmul ?a ?b) ?s)"
```

The `scale_distrib` rules matter for a concrete reason: in attention the scale
`1/√d` can be applied to the small `Q_sd` before the matmul, or to the large
`scores_ss` after it. Same result, different cost. The e-graph holds both and lets
the cost model choose.

A hard rule of the project (see `CLAUDE.md` rule 4): there is **no canned
`naive => flash` rewrite**. The fused form must be *reachable* by composing
primitive, general identities and then *selected* by the cost model. Hard-coding the
answer as a single pattern-match would mean the optimizer "discovered" nothing.

### The shape analysis

To cost a node by real bytes, the optimizer must know each e-class's shape. The
e-graph does not track that on its own, so we attach an `egg::Analysis` that
propagates shapes bottom-up:

```rust
pub struct ShapeAnalysis { pub inputs: HashMap<String, Vec<usize>> }

impl Analysis<TensorLang> for ShapeAnalysis {
    type Data = Option<Vec<usize>>;   // each e-class's [rows, cols], None if unknown

    fn make(egraph, enode) -> Self::Data {
        match enode {
            Var(sym)       => egraph.analysis.inputs.get(sym).cloned(),
            MatMul([a, b]) => Some(vec![shape(a)[0], shape(b)[1]]),
            Transpose([a]) => Some(vec![shape(a)[1], shape(a)[0]]),
            EMul([a, b])   => non_scalar_of(a, b),
            Softmax([a]) | Fuse([a]) => shape(a),
        }
    }
    fn merge(...)  // equivalent terms must agree on shape; keep the known one
}
```

`make` computes a node's shape from its children's shapes (input shapes come from the
`inputs` map). `merge` combines the shape estimates when two terms join the same
e-class; since equivalent terms have the same shape, it just keeps the known value. A
genuine disagreement would indicate a buggy rewrite, surfaced here rather than
silently producing a wrong plan. This analysis is the M3 prerequisite that was just
built: `cargo test -p rl-opt` confirms the attention output e-class carries shape
`[s, d]` after saturation.

### Extraction, and the honest extractor story

Once the e-graph is saturated, **extraction** chooses one concrete program: pick one
node per e-class to minimize total cost. There is a subtlety that the project's rule
6 calls out in advance. The default `egg` extractor computes *tree* cost, which
double-counts shared subexpressions — and attention reuses `Q`, `K`, `V` heavily, so
tree cost would badly misprice it. This is the exact wall the Tensat project hit.

The correct tool is a **DAG-aware** extractor that counts each shared tensor once.
`egg::LpExtractor` does this with integer linear programming — but it depends on the
`coin_cbc` solver, a C library that is not available on this machine. So the plan
(recorded in the checkpoints) is a custom memoized DAG extractor that reads the
shape analysis to compute real per-e-class bytes and counts each materialized
e-class once, documented as the CBC-free substitute. The self-assessment harness
(`scripts/assess.py`) actively flags the current placeholder extractor until this is
done, so the gap cannot be quietly forgotten.

---

## 8. The fuse primitive: how fusion saves HBM

This is the most recent piece and the one that makes the A/B *possible*, so it gets
its own section.

Naive attention writes the `seq×seq` score matrix to HBM, reads it back for the
softmax, writes the probabilities, reads them back for the output matmul. For long
sequences that traffic dominates everything. Fusion means: compute the whole chain
as one kernel and keep those intermediates in SRAM, so they never touch HBM.

![naive vs fused HBM traffic](docs/figures/fusion.png)

*same math, two schedules. the naive plan round-trips the s×s scores and
probabilities through HBM (four arrows down to HBM); the fused plan keeps them in
SRAM and only the inputs and the final output touch HBM (two arrows). the value is
identical, the HBM bill is not.*

The `fuse` node models this. It is **value-identity** — wrapping a subgraph in
`fuse` does not change the result, so the interpreter treats it as a pass-through and
the 1e-5 numerics gate still holds. What changes is accounting: a fused region is
charged HBM only for its boundary inputs and final output, never its internal
intermediates.

```rust
TensorLang::Fuse([a]) => {
    // sum all flops in the subtree, but count HBM only for distinct leaf inputs
    // and the final output — internal intermediates stay in SRAM.
    let (flops, leaf_bytes, out_shape) = fused_walk(expr, a, shapes);
    acc.flops     += flops;
    acc.hbm_bytes += leaf_bytes + out_bytes(out_shape);
}
```

Crucially this is a **general** primitive — "a producer consumed immediately need not
spill" — not an attention special case. The same node fuses the MLP up/down
projection in M5. That generality is what keeps it honest under rule 4.

This is verified by `cargo test -p rl-ir --test fuse`:

- `fused_attention_matches_naive_numerically`: fused output equals naive to < 1e-5.
- `fused_attention_cuts_hbm`: fused HBM is strictly lower, saving at least one full
  `seq×seq` tile, while FLOPs stay exactly equal.

**The honest caveat, stated loudly:** fusing the whole `seq×seq` region assumes it
fits in SRAM. Real Flash Attention works precisely because it *tiles* — it keeps only
a small block in SRAM at a time. The capacity limit that forces tiling is not yet
modelled. Adding an **SRAM-capacity constraint** that forces tiling for large `seq`
is the next `impl Constraint`, and it is the perfect example of the whole thesis: a
missing constraint, added in one place, not a search hack.

---

## 9. The milestones, M0 through M5

The project advances in milestones. The iron rule: **each milestone ends in a
benchmark number, not a refactor.** Nothing is "done" until its numeric criterion is
met and recorded.

| milestone | goal | done-criterion | status |
|---|---|---|---|
| **M0** | substrate, IR, reference interpreter | naive attention and MLP match a JAX/NumPy fixture to 1e-5; a microbench prints true flops/hbm | **done** (err 2.98e-8) |
| **M1** | roofline cost model | predict the naive case and print the binding resource; calibrate vs measured within tolerance | **done** for binding-resource; wall-clock calibration deferred (needs accelerator) |
| **M2** | egg + primitive rewrites | after saturation the e-graph provably contains equivalent forms | **done** (assoc, transpose, scale distribution) |
| **M3** | cost-driven extraction + the A/B | under `[Flops]` extract naive; under `[Flops, HbmBytes]` extract fused — same e-graph | **A/B flip passing** (`the_ab_flip`); general DAG extraction over arbitrary e-graphs is the remaining polish |
| **M4** | lower to a real kernel + verify | kernel matches reference to 1e-5, beats naive at seq ≥ 2048; record predicted-vs-measured gap | not started |
| **M5** | beat `ragged_dot` + the ledger | fused MLP beats `jax.lax.ragged_dot` for F>D; both headline numbers reproducible via `roofline replay` | not started |

Current test counts (all green): rl-ir 5, rl-cost 4, rl-opt 8 (including
`the_ab_flip` and `fused_form_is_reachable_by_rewrite`).

The ordering is dependency-strict. M0 (the unglamorous reference interpreter) comes
first because it makes every later number honest. M1 must calibrate before M3 is
allowed to choose between plans — a cost model that cannot predict the naive case has
no business ranking alternatives.

---

## 10. Design decisions and trade-offs

Every real decision has a cost. The notable ones, with the trade-off made explicit:

- **Rust core, not Python.** Trade-off: more friction writing it, far more speed and
  type safety, and a single artifact. Justified because `toydb` proves a full system
  fits in ~15k lines of Rust, and the hot path must be fast.
- **Tiny IR (six node types).** Trade-off: cannot express arbitrary programs yet, but
  the interpreter is trivially correct and the whole space is auditable. v0 needs one
  honest result, not generality.
- **Naive "materialize everything" HBM model.** Trade-off: inaccurate against real
  hardware, but transparent and fully reasoned. Its inaccuracy is turned into M1
  calibration work rather than hidden.
- **Cost model as a trait, separate crate.** Trade-off: a little ceremony, but
  constraints become independently testable and addable. This is the prime directive
  made structural.
- **e-graph instead of greedy rewriting.** Trade-off: more memory and the extraction
  problem, but you never throw away a form that turns out better. The extraction
  problem is real (see DAG vs tree below) and is acknowledged up front.
- **`fuse` as value-identity.** Trade-off: it models fusion's HBM saving without
  modelling SRAM capacity, so it over-claims for huge tensors. The fix (an SRAM
  constraint that forces tiling) is named and deferred, not pretended away.
- **DAG extraction without ILP.** Trade-off: `LpExtractor` would be exact but needs a
  C solver unavailable here; a custom memoized extractor is approximate but builds
  everywhere. Documented as a substitute, not silently swapped.
- **Forward pass only, no training in v0.** Trade-off: narrower scope, but the value
  is demonstrable in one run. Backprop is a later transformation over the same
  e-graph, not a rewrite of the system.

---

## 11. Honest limitations and prior art

This is not unprecedented and the README will not pretend it is. **Tensat** (MLSys
2021) and **SPORES** already used `egg` for tensor and linear-algebra
superoptimization. **XLA**, **TVM/Ansor**, **tinygrad**, and **Mojo** all do
cost-based kernel scheduling. Where the analogy to databases strains: SQL
optimization is dominated by *cardinality estimation* (guessing how many rows a
filter passes), which is data-dependent and statistical. Tensor programs are mostly
statically shaped and dense, so the hard part moves from "estimate selectivity" to
"model the device accurately." That shift is the point, not a flaw — it is why the
cost model, not the search, carries the intelligence.

The defensible, novel core is three things, none of which those systems center:

1. **An extensible, eventually-learned cost model** where "the optimizer was wrong"
   decomposes into "a constraint was missing" by construction.
2. **A hybrid rewrite engine**: hand-written algebra plus, later, a *verified* LLM
   rewrite proposer that only admits a rewrite after it passes numerics and the
   benchmark.
3. **Recursion of the same idea up the stack**: cost-based search at the kernel
   level (v0), the rewrite-proposal level (v1), and the scaling-law
   experiment-design level (v2).

Current concrete limitations: no real kernel emission yet (M4), no accelerator
calibration (M1's deferred half), the extractor is still the placeholder (M3), and
the fuse model ignores SRAM capacity. All are tracked, none are hidden.

---

## 12. Build, test, run, resume

```bash
# Rust toolchain lives at C:\Users\bhansa01\.cargo\bin (on the User PATH, gnu).
cargo build --workspace
cargo test  --workspace                       # all numerics + unit tests

cargo run -p rl-ir   --example m0_numbers     # ground-truth flops/hbm sweep
cargo run -p rl-cost --example m1_binding      # binding resource per shape
cargo test -p rl-ir  --test fuse               # fusion: correct + cuts HBM

python scripts/assess.py                       # objective score; gates commits/pushes
python scripts/figures.py                      # regenerate the README diagrams
```

To resume work across sessions:

1. Read `CLAUDE.md` (operating rules, milestone checklist).
2. Read this README.
3. Read the newest file in `quality_reports/checkpoints/`.
4. `python scripts/assess.py --start`, then `cargo test --workspace` to confirm green.
5. Do the first action listed in that checkpoint.

The full development process — the self-assessment loop, commit and checkpoint
discipline, context-budget rules, and how to set this workflow up in a new project —
lives in `WORKFLOW.md`.

---

## 13. Glossary

- **Tensor** — a multi-dimensional array of numbers. A matrix is a 2-D tensor.
- **Shape suffix** — the convention of writing a tensor's shape into its name:
  `Q_sd` is `[seq, dim]`.
- **FLOPs** — floating-point operations; the amount of arithmetic.
- **HBM** — High Bandwidth Memory; the GPU's large, relatively slow memory.
- **SRAM** — the GPU's small, very fast on-chip memory.
- **Arithmetic intensity** — FLOPs per byte moved; where a program sits on the
  roofline.
- **Roofline** — a model plotting achievable performance against arithmetic
  intensity, with a memory-bound region and a compute-bound region.
- **Ridge point** — the intensity where memory-bound meets compute-bound
  (peak_flops / peak_bandwidth).
- **Binding resource** — the constraint that produces the maximum (slowest) time;
  the thing actually limiting the program.
- **e-graph** — a data structure that compactly stores many equivalent programs,
  grouped into e-classes of proven-equal expressions.
- **Equality saturation** — applying rewrite rules to an e-graph until no rule adds
  anything new.
- **Extraction** — choosing one concrete program out of a saturated e-graph by
  minimizing a cost function.
- **DAG-aware extraction** — extraction that counts shared subexpressions once
  rather than double-counting them (tree cost).
- **Fusion** — running several operations as one kernel so intermediates stay in
  SRAM and never spill to HBM.
- **Flash Attention** — an attention kernel that avoids materializing the
  `seq×seq` score matrix by tiling; the human-found version of what this optimizer
  aims to find automatically.
- **Constraint** — one physical resource's lower bound on time, in the cost model.
- **Preregistration** — committing a benchmark's config, metric, and success
  threshold *before* running it, so results cannot be retrofitted.

---

## 14. Roadmap

- **Finish M3**: the cost-driven DAG extractor and the A/B test, turning the top
  figure into a passing assertion.
- **M4**: emit a real Pallas kernel from the chosen plan, verify it to 1e-5, beat
  naive at large sequence length, and record the predicted-vs-measured gap.
- **M5**: beat `jax.lax.ragged_dot` on the fused MLP up/down projection for F>D, and
  stand up the ledger so both headline numbers replay from preregistered configs.
- **Beyond v0**: an SRAM-capacity constraint that forces tiling; a verified LLM
  rewrite proposer; and the same cost-based search lifted to scaling-law experiment
  design (choosing the next training run by expected information gain).

This README grows with the project. When a milestone lands, its section moves from
"in progress" to "done" with the number that earned it, and any new crate, decision,
or trade-off is written up here in the same style.
