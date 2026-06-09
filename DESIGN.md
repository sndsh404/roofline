# Roofline, Design spec

A cost-based optimizing compiler for tensor programs, structured like a query
engine. The optimizer's decisions are driven by a pluggable roofline cost model,
not by hard-coded rewrite patterns.

## 1. Why a query engine architecture

Traditional tensor compilers (XLA, TVM) use a fixed pass pipeline. Roofline
instead follows the Cascades/Spark philosophy: the optimizer explores an
equivalence class of plans via egg, and the cost model selects the physical plan.
This makes it trivial to add new devices, new constraints, or new rewrites
without touching the search.

```
IR (egg RecExpr<TensorLang>)
  │
  ▼
[rl-opt] egg runner + rewrite rules (§5)
  │  produces saturated e-graph
  ▼
[rl-cost] CostModel[Constraint+] selects min-cost plan
  │  (LpExtractor, not tree Extractor, rule 6)
  ▼
[rl-codegen] physical plan → Pallas/Triton kernel
```

## 2. Tensor IR (rl-ir)

The IR is an egg `RecExpr<TensorLang>`, a DAG of operators over named input
tensors. Every intermediate has an inferred shape; every input carries a shape
suffix in its name (`Q_sd`, `scores_ss`, etc.).

### Operators

| Operator     | Arity | Semantics                         | FLOPs formula                     |
|-------------|-------|-----------------------------------|-----------------------------------|
| `Var`       | 0     | Named input tensor                | 0                                 |
| `MatMul`    | 2     | `[m,k] × [k,n] → [m,n]`          | `2·m·k·n`                        |
| `Transpose` | 1     | `[m,n] → [n,m]`                  | 0 (just a view in hardware)       |
| `EMul`      | 2     | elementwise mul (broadcasts [1])  | `prod(shape)`                     |
| `Softmax`   | 1     | stable softmax along last axis    | `4·prefix·last` (max+exp+sum+div) |

The HBM accountant (M0) charges every intermediate as materialised. M1
calibrates this against wall-clock.

## 3. Cost model architecture (rl-cost)

The cost model is a set of `Constraint` trait objects. Each constraint produces a
lower bound on wall-clock time. The cost of a plan is the **maximum** across all
constraints, the slowest resource.

```
trait Constraint {
    fn name(&self) -> &str;
    fn lower_bound_s(&self, flops: u64, hbm_bytes: u64) -> f64;
}
```

Built-in constraints for v0:

| Constraint        | Formula                        | Physical meaning               |
|------------------|-------------------------------|-------------------------------|
| `FlopsConstraint`  | `flops / peak_flops`           | compute bound                 |
| `HbmConstraint`    | `hbm_bytes / peak_bandwidth`   | memory bound                  |

Adding a new device means adding a device-specific set of these constants, not
changing the search.

**Prime directive:** If the optimizer makes a wrong choice, the fix is always a
new or corrected constraint, never a search heuristic.

## 4. Preregistration & ledger (rl-ledger)

Before any benchmark whose result is a claim, run:

```
roofline prereg --bench <b> --metric <m> --claim "<c>" --seed <n>
```

This commits the config + metric + threshold + seed to the WAL. Results are only
compared against a preregistered claim. The ledger stores runs versioned by MVCC
(the toydb/bustub pattern); `roofline replay <run_id>` reproduces a result.

## 5. Primitive algebraic identities

The egg rewrite rules that make Flash Attention *reachable* from primitives.
These are the only rewrites. The cost model selects; the rewrites only expand
the equivalence class.

### 5.1 MatMul associativity
```
MatMul(MatMul(A, B), C)  ←→  MatMul(A, MatMul(B, C))
```

### 5.2 MatMul transpose fusion
```
MatMul(Transpose(A), B)  ←→  MatMul(A, Transpose(B))
```

### 5.3 MatMul scaling
```
MatMul(EMul(A, s), B)  ←→  EMul(MatMul(A, B), s)    // s is scalar [1]
```

### 5.4 Scalar scaling distribution through MatMul
```
MatMul(EMul(A, s), B)  ←→  EMul(MatMul(A, B), s)    // s is scalar [1]
EMul(MatMul(A, B), s)  ←→  MatMul(A, EMul(B, s))    // s is scalar [1]
```
Scalar multiplication distributes over matrix multiplication. This lets the
1/sqrt(d) scale in attention move through the QK^T matmul, producing
equivalent programs with different intermediate tensor shapes being scaled.

### 5.5 Online softmax tiling (the Flash identity)
```
Attn = MatMul(Softmax(MatMul(Q, Transpose(K))), V)
```
is equivalent to the tiled form:
```
For each block (Qi, Kj, Vj):
  scores_ij = MatMul(Qi, Transpose(Kj))
  attn_i    = OnlineSoftmax(scores_ij, Vj)    // online softmax update
```
The online softmax update computes the block contribution and rescales the
running output. This identity requires no new IR nodes, the tiled form is
expressible as a DAG of MatMul, EMul, and Softmax nodes over slices.

### 5.6 Ragged MLP decomposition
```
MatMul(X, W_up)  →  MatMul(X, W_up_chunk) for each chunk  // when F > D
    then concatenate along hidden dim.
```

## 6. Shape convention (hard rule 5)

Every tensor variable includes its shape in the name using a suffix convention:

| Suffix | Meaning     | Example       |
|--------|-------------|---------------|
| `_sd`  | [seq, dim]  | `Q_sd`        |
| `_ss`  | [seq, seq]  | `scores_ss`   |
| `_fi`  | [feat, hid] | `W_up_fi`     |
| `_io`  | [hid, feat] | `W_dn_io`     |
| `_sf`  | [seq, feat] | `X_sf`        |

Single-letter dimensions: `s` = sequence, `d` = head dim, `f` = features,
`h`/`i` = hidden.

## 7. Device specifications

### A100-80GB SMX (v0 target)

| Parameter          | Value           |
|-------------------|-----------------|
| Peak TFLOPS (FP16)| 312             |
| HBM bandwidth     | 2.0 TB/s        |
| Ridge point       | 156 flop/byte   |
| TDP               | 400 W           |
| HBM capacity      | 80 GB           |

### H100 SMX (stretch target)

| Parameter          | Value           |
|-------------------|-----------------|
| Peak TFLOPS (FP16)| 989             |
| HBM bandwidth     | 3.35 TB/s       |
| Ridge point       | 295 flop/byte   |
| TDP               | 700 W           |
| HBM capacity      | 80 GB           |

## 8. Non-goals (do not build)

- SQL layer or relational algebra
- Distributed consensus (Raft, Percolator, Spanner)
- The LLM-based rewrite proposer from the write-up
- Backpropagation or training
- Anything whose value cannot be read off a benchmark in a single run

## 9. Milestone done-criteria

Every milestone ends in a **number**, not a refactor. The number is compared
against the preregistered claim.

### M0, Substrate + IR + reference interpreter
**Done when:** reference interpreter matches JAX fixture to `1e-5` for naive
attention (s=4, d=8) and the `m0_numbers` example prints ground-truth FLOPs and
HBM bytes for s=64..1024, d=64.

### M1, Roofline cost model calibrated
**Done when:** cost model predicts naive attention wall-clock time within ±20%
of measured microbench for s∈{256,512,1024}, d=64 on an A100, and `println!`s
the name of the binding resource ("HbmBytes" for all s in that range).

### M2, egg + primitive rewrites
**Done when:** After saturating with the §5 rewrite rules, the e-graph for
naive attention (s=1024, d=64) contains a term equivalent to the tiled Flash
form. Verified by extracting a term whose account shows `hbm_bytes < s²·4`.

### M3, LpExtractor + the A/B
**Done when:** With `[Flops]` as the only constraint, `LpExtractor` returns the
naive plan. With `[Flops, HbmBytes]`, the same e-graph returns the Flash plan.
Both extracted plans pass the numerics gate. Record the runtime gap.

### M4, Lower to Pallas + verify
**Done when:** Lowered kernel matches reference interpreter to `1e-5` across
s∈{128,256,512,1024,2048}, d=64. Kernel is faster than naive at s≥2048. Gap
between measured and cost-model-predicted speed is recorded in ledger.

### M5, MLP beats ragged_dot + ledger
**Done when:** For F > D, the Roofline-optimised MLP kernel beats
`jax.lax.ragged_dot` on an A100. Both the attention (M4) and MLP (M5) headline
numbers are reproducible via `roofline replay`.
