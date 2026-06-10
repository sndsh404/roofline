use egg::{define_language, Id, RecExpr, Symbol};
use std::collections::HashMap;

// ── Language ──────────────────────────────────────────────────────────────────

define_language! {
    pub enum TensorLang {
        // Named input tensor: looked up in the evaluation environment.
        // Shape suffix convention: always reflect shape in the name, e.g. Q_sd.
        Var(Symbol),
        // 2-D matrix multiply: [m,k] × [k,n] → [m,n]
        "matmul" = MatMul([Id; 2]),
        // Transpose last two dims: [m,n] → [n,m]
        "transpose" = Transpose([Id; 1]),
        // Elementwise multiply; left operand may be a scalar [1] that broadcasts.
        "emul" = EMul([Id; 2]),
        // Numerically stable softmax along the last axis.
        "softmax" = Softmax([Id; 1]),
        // Elementwise max(x, 0). The MLP's nonlinearity. Load-bearing for M5:
        // without it the two-layer MLP is linear and the e-graph correctly
        // collapses (X Wup) Wdn into X (Wup Wdn) by associativity, leaving
        // nothing to fuse. The relu blocks that collapse, exactly as in a real
        // network.
        "relu" = Relu([Id; 1]),
        // Fusion boundary: the wrapped subgraph runs as one kernel and its
        // internal intermediates are NOT spilled to HBM. Value-identity (it does
        // not change the math), so the interpreter treats it as a pass-through.
        // The accountant charges HBM only for the boundary inputs and final
        // output. This is the general "producer consumed immediately need not
        // spill" primitive; applied to attention it removes the s×s round-trip.
        // NB: it assumes the fused region fits SRAM. The SRAM-capacity constraint
        // that forces tiling for large s is future work (a new impl Constraint).
        "fuse" = Fuse([Id; 1]),
    }
}

// ── Tensor data ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TensorData {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl TensorData {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len(), "shape product must equal data length");
        Self { shape, data }
    }

    pub fn scalar(v: f32) -> Self {
        Self { shape: vec![1], data: vec![v] }
    }

    fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

// ── Reference interpreter ─────────────────────────────────────────────────────

pub fn eval(
    expr: &RecExpr<TensorLang>,
    root: Id,
    env: &HashMap<String, TensorData>,
) -> TensorData {
    match expr[root].clone() {
        TensorLang::Var(sym) => env
            .get(sym.as_str())
            .unwrap_or_else(|| panic!("undefined variable '{}'", sym))
            .clone(),
        TensorLang::MatMul([a, b]) => {
            let ta = eval(expr, a, env);
            let tb = eval(expr, b, env);
            matmul_2d(&ta, &tb)
        }
        TensorLang::Transpose([a]) => transpose_2d(&eval(expr, a, env)),
        TensorLang::EMul([a, b]) => {
            let ta = eval(expr, a, env);
            let tb = eval(expr, b, env);
            emul_broadcast(&ta, &tb)
        }
        TensorLang::Softmax([a]) => softmax_last(&eval(expr, a, env)),
        TensorLang::Relu([a]) => relu_elementwise(&eval(expr, a, env)),
        // Fusion is a scheduling annotation; it does not change the value.
        TensorLang::Fuse([a]) => eval(expr, a, env),
    }
}

// ── Shape inference ───────────────────────────────────────────────────────────

pub fn infer_shape(
    expr: &RecExpr<TensorLang>,
    root: Id,
    shapes: &HashMap<String, Vec<usize>>,
) -> Vec<usize> {
    match expr[root].clone() {
        TensorLang::Var(sym) => shapes[sym.as_str()].clone(),
        TensorLang::MatMul([a, b]) => {
            let sa = infer_shape(expr, a, shapes);
            let sb = infer_shape(expr, b, shapes);
            assert_eq!(sa.len(), 2);
            assert_eq!(sb.len(), 2);
            assert_eq!(sa[1], sb[0], "matmul inner dim mismatch: {:?} × {:?}", sa, sb);
            vec![sa[0], sb[1]]
        }
        TensorLang::Transpose([a]) => {
            let s = infer_shape(expr, a, shapes);
            assert_eq!(s.len(), 2, "M0: transpose requires 2-D tensor");
            vec![s[1], s[0]]
        }
        TensorLang::EMul([a, b]) => {
            let sa = infer_shape(expr, a, shapes);
            let sb = infer_shape(expr, b, shapes);
            if sa.iter().product::<usize>() == 1 { sb }
            else if sb.iter().product::<usize>() == 1 { sa }
            else {
                assert_eq!(sa, sb, "emul: incompatible shapes {:?} and {:?}", sa, sb);
                sa
            }
        }
        TensorLang::Softmax([a]) => infer_shape(expr, a, shapes),
        TensorLang::Relu([a]) => infer_shape(expr, a, shapes),
        TensorLang::Fuse([a]) => infer_shape(expr, a, shapes),
    }
}

// ── Cost accountant ───────────────────────────────────────────────────────────
//
// M0 model: every intermediate is materialised to HBM.
// hbm_bytes = sum over all nodes of (output tensor bytes written).
// Inputs (Var) count as read from HBM.
// This is the "naive everywhere" lower bound, M1 will calibrate against wall-clock.

#[derive(Clone, Debug, Default)]
pub struct Account {
    pub flops: u64,
    pub hbm_bytes: u64,
    /// Peak SRAM working set demanded by any fused region: its boundary
    /// inputs, its final output, and every internal intermediate, all resident
    /// at once. This is the monolithic no-tiling model the `fuse` node
    /// assumes, measured instead of assumed, so an SRAM capacity constraint
    /// can refuse fusions that cannot actually fit on chip. Zero for plans
    /// with no fuse node.
    pub sram_bytes: u64,
}

impl Account {
    pub fn intensity(&self) -> f64 {
        if self.hbm_bytes == 0 { return 0.0; }
        self.flops as f64 / self.hbm_bytes as f64
    }
}

pub fn account(
    expr: &RecExpr<TensorLang>,
    root: Id,
    shapes: &HashMap<String, Vec<usize>>,
) -> Account {
    let mut acc = Account::default();
    account_node(expr, root, shapes, &mut acc);
    acc
}

fn account_node(
    expr: &RecExpr<TensorLang>,
    root: Id,
    shapes: &HashMap<String, Vec<usize>>,
    acc: &mut Account,
) -> Vec<usize> {
    match expr[root].clone() {
        TensorLang::Var(sym) => {
            let s = shapes[sym.as_str()].clone();
            acc.hbm_bytes += (s.iter().product::<usize>() * 4) as u64;
            s
        }
        TensorLang::MatMul([a, b]) => {
            let sa = account_node(expr, a, shapes, acc);
            let sb = account_node(expr, b, shapes, acc);
            let (m, k, n) = (sa[0], sa[1], sb[1]);
            acc.flops += 2 * m as u64 * k as u64 * n as u64;
            let out = vec![m, n];
            acc.hbm_bytes += (m * n * 4) as u64;
            out
        }
        TensorLang::Transpose([a]) => {
            let sa = account_node(expr, a, shapes, acc);
            let out = vec![sa[1], sa[0]];
            // Transpose touches every element once (read already counted in child).
            // We charge for the write of the transposed copy.
            acc.hbm_bytes += (sa[0] * sa[1] * 4) as u64;
            out
        }
        TensorLang::EMul([a, b]) => {
            let sa = account_node(expr, a, shapes, acc);
            let sb = account_node(expr, b, shapes, acc);
            let out = if sa.iter().product::<usize>() == 1 { sb } else { sa };
            let elems: usize = out.iter().product();
            acc.flops += elems as u64;
            acc.hbm_bytes += (elems * 4) as u64;
            out
        }
        TensorLang::Softmax([a]) => {
            let sa = account_node(expr, a, shapes, acc);
            let last = *sa.last().unwrap();
            let prefix = sa.iter().product::<usize>() / last;
            // Per row: find max (s ops), compute exp (s ops), sum (s ops), divide (s ops).
            acc.flops += (4 * prefix * last) as u64;
            acc.hbm_bytes += (prefix * last * 4) as u64;
            sa
        }
        TensorLang::Relu([a]) => {
            let sa = account_node(expr, a, shapes, acc);
            let elems: usize = sa.iter().product();
            // One compare per element; the materialized result is written back.
            acc.flops += elems as u64;
            acc.hbm_bytes += (elems * 4) as u64;
            sa
        }
        TensorLang::Fuse([a]) => {
            // The fused region computes all the same flops, but only its boundary
            // inputs (distinct leaves) and final output touch HBM. Internal
            // intermediates stay in SRAM, so we do NOT recurse via account_node
            // (which would charge every intermediate). This is the HBM saving the
            // M3 A/B rewards.
            let mut flops = 0u64;
            let mut leaves: HashMap<String, usize> = HashMap::new();
            let mut inter_bytes = 0u64;
            let out_shape = fused_walk(expr, a, shapes, &mut flops, &mut leaves, &mut inter_bytes);
            let leaf_bytes: u64 = leaves.values().map(|n| (*n * 4) as u64).sum();
            let out_bytes = (out_shape.iter().product::<usize>() * 4) as u64;
            acc.flops += flops;
            acc.hbm_bytes += leaf_bytes + out_bytes;
            // The fused region's working set: everything lives in SRAM at
            // once under the monolithic model. The walk counted the region's
            // own output among the intermediates, so subtract it back out.
            let working_set = leaf_bytes + out_bytes + inter_bytes.saturating_sub(out_bytes);
            acc.sram_bytes = acc.sram_bytes.max(working_set);
            out_shape
        }
    }
}

/// Walk a fused subtree: accumulate total flops, distinct leaf input sizes,
/// and the bytes of every internal intermediate tensor, returning the
/// subtree's output shape. Intermediates are not recorded as HBM traffic,
/// that is the whole point of fusion, but they ARE recorded as SRAM demand,
/// because under the monolithic no-tiling model they all live on chip at
/// once.
fn fused_walk(
    expr: &RecExpr<TensorLang>,
    root: Id,
    shapes: &HashMap<String, Vec<usize>>,
    flops: &mut u64,
    leaves: &mut HashMap<String, usize>,
    inter_bytes: &mut u64,
) -> Vec<usize> {
    let out_shape = match expr[root].clone() {
        TensorLang::Var(sym) => {
            let s = shapes[sym.as_str()].clone();
            leaves.insert(sym.to_string(), s.iter().product());
            return s; // a leaf is boundary input, not an intermediate
        }
        TensorLang::MatMul([a, b]) => {
            let sa = fused_walk(expr, a, shapes, flops, leaves, inter_bytes);
            let sb = fused_walk(expr, b, shapes, flops, leaves, inter_bytes);
            let (m, k, n) = (sa[0], sa[1], sb[1]);
            *flops += 2 * m as u64 * k as u64 * n as u64;
            vec![m, n]
        }
        TensorLang::Transpose([a]) => {
            let sa = fused_walk(expr, a, shapes, flops, leaves, inter_bytes);
            vec![sa[1], sa[0]]
        }
        TensorLang::EMul([a, b]) => {
            let sa = fused_walk(expr, a, shapes, flops, leaves, inter_bytes);
            let sb = fused_walk(expr, b, shapes, flops, leaves, inter_bytes);
            let out = if sa.iter().product::<usize>() == 1 { sb } else { sa };
            *flops += out.iter().product::<usize>() as u64;
            out
        }
        TensorLang::Softmax([a]) => {
            let sa = fused_walk(expr, a, shapes, flops, leaves, inter_bytes);
            *flops += 4 * sa.iter().product::<usize>() as u64;
            sa
        }
        TensorLang::Relu([a]) => {
            let sa = fused_walk(expr, a, shapes, flops, leaves, inter_bytes);
            *flops += sa.iter().product::<usize>() as u64;
            sa
        }
        TensorLang::Fuse([a]) => return fused_walk(expr, a, shapes, flops, leaves, inter_bytes),
    };
    *inter_bytes += (out_shape.iter().product::<usize>() * 4) as u64;
    out_shape
}

// ── Canonical programs ────────────────────────────────────────────────────────

/// Naive single-head attention: softmax(Q K^T / sqrt(d)) V
/// Inputs: Q_sd, K_sd, V_sd, scale (scalar = 1/sqrt(d))
pub fn naive_attention_program() -> (RecExpr<TensorLang>, Id) {
    let mut e = RecExpr::default();
    let q  = e.add(TensorLang::Var("Q_sd".into()));
    let k  = e.add(TensorLang::Var("K_sd".into()));
    let v  = e.add(TensorLang::Var("V_sd".into()));
    let sc = e.add(TensorLang::Var("scale".into()));
    let kt      = e.add(TensorLang::Transpose([k]));
    let scores  = e.add(TensorLang::MatMul([q, kt]));     // [s,s]
    let scaled  = e.add(TensorLang::EMul([scores, sc]));
    let attn    = e.add(TensorLang::Softmax([scaled]));
    let out     = e.add(TensorLang::MatMul([attn, v]));   // [s,d]
    (e, out)
}

/// Attention with the whole score/softmax/output chain fused: the s×s scores and
/// probabilities never spill to HBM. Same value as `naive_attention_program`
/// (the interpreter treats `fuse` as identity), but a far smaller HBM bill. This
/// is the form the M3 A/B selects once the HBM constraint is modelled. NB: it
/// assumes the fused region fits SRAM; the capacity constraint that forces tiling
/// for large s is the next `impl Constraint`.
pub fn fused_attention_program() -> (RecExpr<TensorLang>, Id) {
    let mut e = RecExpr::default();
    let q  = e.add(TensorLang::Var("Q_sd".into()));
    let k  = e.add(TensorLang::Var("K_sd".into()));
    let v  = e.add(TensorLang::Var("V_sd".into()));
    let sc = e.add(TensorLang::Var("scale".into()));
    let kt     = e.add(TensorLang::Transpose([k]));
    let scores = e.add(TensorLang::MatMul([q, kt]));
    let scaled = e.add(TensorLang::EMul([scores, sc]));
    let attn   = e.add(TensorLang::Softmax([scaled]));
    let out    = e.add(TensorLang::MatMul([attn, v]));
    let fused  = e.add(TensorLang::Fuse([out]));
    (e, fused)
}

/// Two-layer MLP with relu: Y_sd = relu(X_sd W_up_df) W_dn_fd.
/// Shape suffixes follow DESIGN's F > D nomenclature: d is the model width,
/// f is the hidden width, so X_sd is [s, d], W_up_df is [d, f], W_dn_fd is
/// [f, d] and the M5 regime of interest is f > d. The relu is what keeps the
/// program two matmuls: without it associativity legally collapses the chain
/// into X (W_up W_dn) and the right optimization is the collapse, not fusion.
pub fn naive_mlp_program() -> (RecExpr<TensorLang>, Id) {
    let mut e = RecExpr::default();
    let x    = e.add(TensorLang::Var("X_sd".into()));
    let w_up = e.add(TensorLang::Var("W_up_df".into()));
    let w_dn = e.add(TensorLang::Var("W_dn_fd".into()));
    let h_sf = e.add(TensorLang::MatMul([x, w_up]));
    let a_sf = e.add(TensorLang::Relu([h_sf]));
    let out  = e.add(TensorLang::MatMul([a_sf, w_dn]));
    (e, out)
}

/// The MLP with the whole up/relu/down chain fused: the s by f hidden
/// activations never spill to HBM. Same value as `naive_mlp_program` (fuse is
/// identity under eval); the accountant charges HBM only for X, the two
/// weights, and Y. This is the form M5 lowers to a streaming kernel.
pub fn fused_mlp_program() -> (RecExpr<TensorLang>, Id) {
    let mut e = RecExpr::default();
    let x    = e.add(TensorLang::Var("X_sd".into()));
    let w_up = e.add(TensorLang::Var("W_up_df".into()));
    let w_dn = e.add(TensorLang::Var("W_dn_fd".into()));
    let h_sf = e.add(TensorLang::MatMul([x, w_up]));
    let a_sf = e.add(TensorLang::Relu([h_sf]));
    let out  = e.add(TensorLang::MatMul([a_sf, w_dn]));
    let fused = e.add(TensorLang::Fuse([out]));
    (e, fused)
}

// ── Primitive tensor ops ──────────────────────────────────────────────────────

fn matmul_2d(a: &TensorData, b: &TensorData) -> TensorData {
    assert_eq!(a.shape.len(), 2, "matmul_2d requires 2-D inputs");
    assert_eq!(b.shape.len(), 2, "matmul_2d requires 2-D inputs");
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2, "matmul inner dimension mismatch: {} vs {}", k, k2);
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_il = a.data[i * k + l];
            for j in 0..n {
                out[i * n + j] += a_il * b.data[l * n + j];
            }
        }
    }
    TensorData::new(vec![m, n], out)
}

fn transpose_2d(a: &TensorData) -> TensorData {
    assert_eq!(a.shape.len(), 2, "transpose_2d requires 2-D input");
    let (m, n) = (a.shape[0], a.shape[1]);
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a.data[i * n + j];
        }
    }
    TensorData::new(vec![n, m], out)
}

fn emul_broadcast(a: &TensorData, b: &TensorData) -> TensorData {
    if a.numel() == 1 {
        let s = a.data[0];
        TensorData::new(b.shape.clone(), b.data.iter().map(|x| x * s).collect())
    } else if b.numel() == 1 {
        let s = b.data[0];
        TensorData::new(a.shape.clone(), a.data.iter().map(|x| x * s).collect())
    } else {
        assert_eq!(a.shape, b.shape, "emul: shapes {:?} and {:?} incompatible", a.shape, b.shape);
        TensorData::new(
            a.shape.clone(),
            a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).collect(),
        )
    }
}

fn relu_elementwise(a: &TensorData) -> TensorData {
    TensorData::new(a.shape.clone(), a.data.iter().map(|x| x.max(0.0)).collect())
}

fn softmax_last(a: &TensorData) -> TensorData {
    assert!(!a.shape.is_empty());
    let last = *a.shape.last().unwrap();
    let prefix = a.numel() / last;
    let mut out = a.data.clone();
    for row in 0..prefix {
        let start = row * last;
        let end = start + last;
        let max = out[start..end]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = out[start..end].iter().map(|x| (x - max).exp()).sum();
        for x in &mut out[start..end] {
            *x = (*x - max).exp() / sum;
        }
    }
    TensorData::new(a.shape.clone(), out)
}
