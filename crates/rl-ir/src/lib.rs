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
        TensorLang::Fuse([a]) => {
            // The fused region computes all the same flops, but only its boundary
            // inputs (distinct leaves) and final output touch HBM. Internal
            // intermediates stay in SRAM, so we do NOT recurse via account_node
            // (which would charge every intermediate). This is the HBM saving the
            // M3 A/B rewards.
            let mut flops = 0u64;
            let mut leaves: HashMap<String, usize> = HashMap::new();
            let out_shape = fused_walk(expr, a, shapes, &mut flops, &mut leaves);
            let leaf_bytes: u64 = leaves.values().map(|n| (*n * 4) as u64).sum();
            let out_bytes = (out_shape.iter().product::<usize>() * 4) as u64;
            acc.flops += flops;
            acc.hbm_bytes += leaf_bytes + out_bytes;
            out_shape
        }
    }
}

/// Walk a fused subtree: accumulate total flops and distinct leaf input sizes,
/// returning the subtree's output shape. Internal intermediate tensors are not
/// recorded as HBM traffic, that is the whole point of fusion.
fn fused_walk(
    expr: &RecExpr<TensorLang>,
    root: Id,
    shapes: &HashMap<String, Vec<usize>>,
    flops: &mut u64,
    leaves: &mut HashMap<String, usize>,
) -> Vec<usize> {
    match expr[root].clone() {
        TensorLang::Var(sym) => {
            let s = shapes[sym.as_str()].clone();
            leaves.insert(sym.to_string(), s.iter().product());
            s
        }
        TensorLang::MatMul([a, b]) => {
            let sa = fused_walk(expr, a, shapes, flops, leaves);
            let sb = fused_walk(expr, b, shapes, flops, leaves);
            let (m, k, n) = (sa[0], sa[1], sb[1]);
            *flops += 2 * m as u64 * k as u64 * n as u64;
            vec![m, n]
        }
        TensorLang::Transpose([a]) => {
            let sa = fused_walk(expr, a, shapes, flops, leaves);
            vec![sa[1], sa[0]]
        }
        TensorLang::EMul([a, b]) => {
            let sa = fused_walk(expr, a, shapes, flops, leaves);
            let sb = fused_walk(expr, b, shapes, flops, leaves);
            let out = if sa.iter().product::<usize>() == 1 { sb } else { sa };
            *flops += out.iter().product::<usize>() as u64;
            out
        }
        TensorLang::Softmax([a]) => {
            let sa = fused_walk(expr, a, shapes, flops, leaves);
            *flops += 4 * sa.iter().product::<usize>() as u64;
            sa
        }
        TensorLang::Fuse([a]) => fused_walk(expr, a, shapes, flops, leaves),
    }
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

/// Two-layer MLP: down(up(x))  (no activation in M0)
/// Inputs: X_sf (token, in_features), W_up_fi (in, hidden), W_dn_io (hidden, in)
pub fn naive_mlp_program() -> (RecExpr<TensorLang>, Id) {
    let mut e = RecExpr::default();
    let x    = e.add(TensorLang::Var("X_sf".into()));
    let w_up = e.add(TensorLang::Var("W_up_fi".into()));
    let w_dn = e.add(TensorLang::Var("W_dn_io".into()));
    let hidden = e.add(TensorLang::MatMul([x, w_up]));
    let out    = e.add(TensorLang::MatMul([hidden, w_dn]));
    (e, out)
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
