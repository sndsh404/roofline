//! M4: lower a chosen physical plan to an executable kernel, and verify it.
//!
//! On a GPU host the target is a Pallas/Triton kernel. This machine has no
//! accelerator, so M4 here emits a Rust CPU kernel instead. The kernel that
//! matters is fused attention: it computes softmax(Q Kt scale) V with an online
//! (streaming) softmax, so the s×s score matrix is never materialised. That is
//! the same memory win the optimizer selected in M3, now executable.
//!
//! Two gates, in order:
//!   1. numerics, the kernel must match the rl-ir reference interpreter to 1e-5;
//!   2. wall-clock, recorded against the naive reference (GPU numbers deferred).
//!
//! Shape suffix convention holds: Q_sd is [seq, dim], O_sd is [seq, dim].

use std::collections::HashMap;

use egg::{Id, RecExpr};
use rl_ir::{eval, TensorData, TensorLang};

/// Which kernel a plan lowers to. A plan that contains a `fuse` node lowers to
/// the matching fused kernel; anything else falls back to the reference
/// interpreter (correct, unoptimised).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kernel {
    FusedAttention,
    FusedMlp,
    Reference,
}

/// Inspect an extracted plan and decide which kernel to emit. The optimizer's
/// decision to fuse (a `Fuse` node in the plan) is what selects the fast path;
/// the producer inside the fused region picks the kernel shape (softmax chain
/// is attention, relu chain is the MLP).
pub fn lower(expr: &RecExpr<TensorLang>) -> Kernel {
    let nodes = expr.as_ref();
    if !nodes.iter().any(|n| matches!(n, TensorLang::Fuse(_))) {
        return Kernel::Reference;
    }
    if nodes.iter().any(|n| matches!(n, TensorLang::Softmax(_))) {
        Kernel::FusedAttention
    } else if nodes.iter().any(|n| matches!(n, TensorLang::Relu(_))) {
        Kernel::FusedMlp
    } else {
        Kernel::Reference
    }
}

/// Lower `expr` and run it on `env`. The fused path reads Q_sd, K_sd, V_sd and
/// the scalar `scale` from the environment; the reference path evaluates the
/// plan node by node.
pub fn lower_and_run(
    expr: &RecExpr<TensorLang>,
    root: Id,
    env: &HashMap<String, TensorData>,
) -> TensorData {
    match lower(expr) {
        Kernel::FusedAttention => {
            let q = &env["Q_sd"];
            let k = &env["K_sd"];
            let v = &env["V_sd"];
            let scale = env["scale"].data[0];
            fused_attention(q, k, v, scale)
        }
        Kernel::FusedMlp => fused_mlp(&env["X_sd"], &env["W_up_df"], &env["W_dn_fd"]),
        Kernel::Reference => eval(expr, root, env),
    }
}

/// Fused attention via online softmax. Computes O_sd = softmax(Q_sd K_sdt * scale)
/// V_sd one query row at a time, keeping a running (max, sum, acc) so the full
/// s×s scores and probabilities never exist in memory at once. This is the
/// streaming-softmax identity at the heart of Flash Attention.
pub fn fused_attention(q: &TensorData, k: &TensorData, v: &TensorData, scale: f32) -> TensorData {
    assert_eq!(q.shape.len(), 2, "Q must be 2-D");
    let (s, d) = (q.shape[0], q.shape[1]);
    let s_k = k.shape[0];
    assert_eq!(k.shape[1], d, "K dim must match Q dim");
    assert_eq!(v.shape[0], s_k, "V rows must match K rows");
    let dv = v.shape[1];

    let mut out = vec![0.0f32; s * dv];
    let mut acc = vec![0.0f32; dv];

    for i in 0..s {
        let qi = &q.data[i * d..(i + 1) * d];
        let mut m = f32::NEG_INFINITY; // running max of the logits
        let mut l = 0.0f32; // running sum of exp(logit - m)
        for a in acc.iter_mut() {
            *a = 0.0;
        }

        for j in 0..s_k {
            let kj = &k.data[j * d..(j + 1) * d];
            let mut dot = 0.0f32;
            for t in 0..d {
                dot += qi[t] * kj[t];
            }
            let s_ij = dot * scale;

            let m_new = m.max(s_ij);
            // exp(m - m_new) rescales the running totals to the new max.
            // On the first key m = -inf so this correction is 0, which is correct.
            let corr = (m - m_new).exp();
            let p = (s_ij - m_new).exp();
            l = l * corr + p;
            let vj = &v.data[j * dv..(j + 1) * dv];
            for t in 0..dv {
                acc[t] = acc[t] * corr + p * vj[t];
            }
            m = m_new;
        }

        let inv = 1.0 / l;
        for t in 0..dv {
            out[i * dv + t] = acc[t] * inv;
        }
    }
    TensorData::new(vec![s, dv], out)
}

/// Fused MLP: Y_sd = relu(X_sd W_up_df) W_dn_fd computed one token row at a
/// time. The hidden activations h_f for the current row live in one length-f
/// buffer (SRAM, in the cost model's terms) and are consumed immediately by the
/// down projection, so the s by f hidden matrix is never materialised. That is
/// exactly the HBM saving the optimizer's fused plan claims.
///
/// Deliberately no extra tricks (no skipping relu zeros, no blocking): the
/// arithmetic and its order are identical to the reference interpreter's, so
/// the comparison isolates fusion, not kernel tuning.
pub fn fused_mlp(x: &TensorData, w_up: &TensorData, w_dn: &TensorData) -> TensorData {
    assert_eq!(x.shape.len(), 2, "X must be 2-D");
    let (s, d) = (x.shape[0], x.shape[1]);
    assert_eq!(w_up.shape[0], d, "W_up rows must match X cols");
    let f = w_up.shape[1];
    assert_eq!(w_dn.shape[0], f, "W_dn rows must match W_up cols");
    let d_out = w_dn.shape[1];

    let mut out = vec![0.0f32; s * d_out];
    let mut h_f = vec![0.0f32; f];

    for i in 0..s {
        for h in h_f.iter_mut() {
            *h = 0.0;
        }
        let xi = &x.data[i * d..(i + 1) * d];
        for l in 0..d {
            let x_il = xi[l];
            let w_row = &w_up.data[l * f..(l + 1) * f];
            for j in 0..f {
                h_f[j] += x_il * w_row[j];
            }
        }
        for h in h_f.iter_mut() {
            *h = h.max(0.0);
        }
        let yi = &mut out[i * d_out..(i + 1) * d_out];
        for j in 0..f {
            let h_ij = h_f[j];
            let w_row = &w_dn.data[j * d_out..(j + 1) * d_out];
            for t in 0..d_out {
                yi[t] += h_ij * w_row[t];
            }
        }
    }
    TensorData::new(vec![s, d_out], out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rl_ir::naive_attention_program;

    fn env(s: usize, d: usize) -> HashMap<String, TensorData> {
        let gen = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 1103515245 + seed) % 1000) as f32) / 500.0 - 1.0)
                .collect()
        };
        HashMap::from([
            ("Q_sd".to_string(), TensorData::new(vec![s, d], gen(1, s * d))),
            ("K_sd".to_string(), TensorData::new(vec![s, d], gen(2, s * d))),
            ("V_sd".to_string(), TensorData::new(vec![s, d], gen(3, s * d))),
            ("scale".to_string(), TensorData::scalar(1.0 / (d as f32).sqrt())),
        ])
    }

    fn max_abs_err(a: &TensorData, b: &TensorData) -> f32 {
        a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn fused_kernel_matches_reference_across_shapes() {
        // The hard gate: the emitted kernel must match the naive reference to 1e-5
        // across a shape sweep, or its speed is worthless.
        let (prog, root): (RecExpr<TensorLang>, Id) = naive_attention_program();
        for &(s, d) in &[(64, 16), (128, 32), (256, 64), (384, 48)] {
            let e = env(s, d);
            let reference = eval(&prog, root, &e);
            let fused = fused_attention(&e["Q_sd"], &e["K_sd"], &e["V_sd"], e["scale"].data[0]);
            let err = max_abs_err(&reference, &fused);
            assert!(err < 1e-5, "fused kernel err {err} at s={s} d={d} exceeds 1e-5");
        }
    }

    #[test]
    fn lower_picks_fused_when_plan_has_fuse() {
        let (naive, _) = naive_attention_program();
        assert_eq!(lower(&naive), Kernel::Reference);

        // wrap the naive plan in a fuse node (what the optimizer's fused plan is)
        let mut fused = naive.clone();
        let r = Id::from(fused.as_ref().len() - 1);
        fused.add(TensorLang::Fuse([r]));
        assert_eq!(lower(&fused), Kernel::FusedAttention);
    }

    fn mlp_env(s: usize, d: usize, f: usize) -> HashMap<String, TensorData> {
        let gen = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 1103515245 + seed) % 1000) as f32) / 500.0 - 1.0)
                .collect()
        };
        HashMap::from([
            ("X_sd".to_string(), TensorData::new(vec![s, d], gen(1, s * d))),
            ("W_up_df".to_string(), TensorData::new(vec![d, f], gen(2, d * f))),
            ("W_dn_fd".to_string(), TensorData::new(vec![f, d], gen(3, f * d))),
        ])
    }

    #[test]
    fn fused_mlp_matches_reference_across_shapes() {
        // The hard gate for the M5 kernel: match the reference interpreter to
        // 1e-5 across a shape sweep covering f < d, f = d, and f > d.
        let (prog, root) = rl_ir::naive_mlp_program();
        for &(s, d, f) in &[(64, 32, 16), (64, 32, 32), (128, 32, 128), (256, 64, 512), (200, 48, 384)] {
            let e = mlp_env(s, d, f);
            let reference = eval(&prog, root, &e);
            let fused = fused_mlp(&e["X_sd"], &e["W_up_df"], &e["W_dn_fd"]);
            let err = max_abs_err(&reference, &fused);
            assert!(err < 1e-5, "fused MLP err {err} at s={s} d={d} f={f} exceeds 1e-5");
        }
    }

    #[test]
    fn lower_picks_fused_mlp_when_plan_has_fuse_and_relu() {
        let (naive, _) = rl_ir::naive_mlp_program();
        assert_eq!(lower(&naive), Kernel::Reference);

        let (fused, _) = rl_ir::fused_mlp_program();
        assert_eq!(lower(&fused), Kernel::FusedMlp);
    }
}
