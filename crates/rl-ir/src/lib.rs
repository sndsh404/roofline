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
    }
}

// ── Cost accountant ───────────────────────────────────────────────────────────
//
// M0 model: every intermediate is materialised to HBM.
// hbm_bytes = sum over all nodes of (output tensor bytes written).
// Inputs (Var) count as read from HBM.
// This is the "naive everywhere" lower bound — M1 will calibrate against wall-clock.

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
