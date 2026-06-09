//! Shared benchmark runners for the M4 (attention) and M5 (MLP) headline
//! numbers. The bench examples and the roofline CLI's replay both call these
//! functions, so a replayed number comes from exactly the same code path as
//! the original measurement, same data generator, same timing, same checks.
//!
//! Each runner does the whole story end to end: saturate the e-graph, let the
//! cost model pick the plan, assert codegen lowers it to the fused kernel,
//! check numerics against the reference interpreter, then time naive vs fused
//! and report the median.

use std::collections::HashMap;
use std::time::Instant;

use rl_cost::{CostModel, FlopsConstraint, HbmConstraint, H100};
use rl_ir::{eval, naive_attention_program, naive_mlp_program, TensorData};
use rl_opt::{account_expr, extract_cost_driven, saturate_shaped};

use crate::{fused_attention, fused_mlp, lower, Kernel};

/// Everything one benchmark run produces. `speedup` is naive_ms / fused_ms.
#[derive(Debug, Clone)]
pub struct BenchOutcome {
    pub naive_ms: f64,
    pub fused_ms: f64,
    pub speedup: f64,
    pub max_abs_err: f64,
    /// accountant's HBM bytes for the naive and the extracted fused plan
    pub naive_hbm_bytes: u64,
    pub fused_hbm_bytes: u64,
    pub flops: u64,
}

/// Deterministic pseudo-random data in [-1, 1]; the run seed feeds in so a
/// preregistered seed pins the input tensors exactly.
fn gen(seed: u64, n: usize) -> Vec<f32> {
    (0..n as u64)
        .map(|i| ((i.wrapping_mul(1103515245).wrapping_add(seed) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

fn median_ms(mut f: impl FnMut(), iters: usize) -> f64 {
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn max_abs_err(a: &TensorData, b: &TensorData) -> f64 {
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max) as f64
}

fn hbm_model() -> CostModel {
    CostModel::new()
        .add(FlopsConstraint::new(H100))
        .add(HbmConstraint::new(H100))
}

/// The M4 attention benchmark at one shape. Panics if the optimizer does not
/// choose the fused plan or codegen does not lower it to the fused kernel,
/// because then the number would not be measuring what it claims to measure.
pub fn bench_attention(s: usize, d: usize, iters: usize, seed: u64) -> BenchOutcome {
    let shapes: HashMap<String, Vec<usize>> = HashMap::from([
        ("Q_sd".to_string(), vec![s, d]),
        ("K_sd".to_string(), vec![s, d]),
        ("V_sd".to_string(), vec![s, d]),
        ("scale".to_string(), vec![1]),
    ]);
    let env: HashMap<String, TensorData> = HashMap::from([
        ("Q_sd".to_string(), TensorData::new(vec![s, d], gen(seed * 10 + 1, s * d))),
        ("K_sd".to_string(), TensorData::new(vec![s, d], gen(seed * 10 + 2, s * d))),
        ("V_sd".to_string(), TensorData::new(vec![s, d], gen(seed * 10 + 3, s * d))),
        ("scale".to_string(), TensorData::scalar(1.0 / (d as f32).sqrt())),
    ]);

    let (prog, root) = naive_attention_program();
    let runner = saturate_shaped(&prog, shapes.clone());
    let eroot = runner.egraph.lookup_expr(&prog).expect("root in egraph");
    let plan = extract_cost_driven(&runner.egraph, eroot, &hbm_model());
    assert_eq!(lower(&plan), Kernel::FusedAttention, "optimizer must pick the fused attention plan");

    let scale = env["scale"].data[0];
    let reference = eval(&prog, root, &env);
    let fused_out = fused_attention(&env["Q_sd"], &env["K_sd"], &env["V_sd"], scale);
    let err = max_abs_err(&reference, &fused_out);

    let naive_ms = median_ms(|| { let _ = eval(&prog, root, &env); }, iters);
    let fused_ms = median_ms(
        || { let _ = fused_attention(&env["Q_sd"], &env["K_sd"], &env["V_sd"], scale); },
        iters,
    );

    let na = account_expr(&prog, &shapes);
    let fa = account_expr(&plan, &shapes);
    BenchOutcome {
        naive_ms,
        fused_ms,
        speedup: naive_ms / fused_ms,
        max_abs_err: err,
        naive_hbm_bytes: na.hbm_bytes,
        fused_hbm_bytes: fa.hbm_bytes,
        flops: na.flops,
    }
}

/// The M5 MLP benchmark at one shape: Y_sd = relu(X_sd W_up_df) W_dn_fd.
/// The regime that matters is f > d, where the s by f hidden activations
/// dominate the naive plan's memory bill.
pub fn bench_mlp(s: usize, d: usize, f: usize, iters: usize, seed: u64) -> BenchOutcome {
    let shapes: HashMap<String, Vec<usize>> = HashMap::from([
        ("X_sd".to_string(), vec![s, d]),
        ("W_up_df".to_string(), vec![d, f]),
        ("W_dn_fd".to_string(), vec![f, d]),
    ]);
    let env: HashMap<String, TensorData> = HashMap::from([
        ("X_sd".to_string(), TensorData::new(vec![s, d], gen(seed * 10 + 1, s * d))),
        ("W_up_df".to_string(), TensorData::new(vec![d, f], gen(seed * 10 + 2, d * f))),
        ("W_dn_fd".to_string(), TensorData::new(vec![f, d], gen(seed * 10 + 3, f * d))),
    ]);

    let (prog, root) = naive_mlp_program();
    let runner = saturate_shaped(&prog, shapes.clone());
    let eroot = runner.egraph.lookup_expr(&prog).expect("root in egraph");
    let plan = extract_cost_driven(&runner.egraph, eroot, &hbm_model());
    assert_eq!(lower(&plan), Kernel::FusedMlp, "optimizer must pick the fused MLP plan");

    let reference = eval(&prog, root, &env);
    let fused_out = fused_mlp(&env["X_sd"], &env["W_up_df"], &env["W_dn_fd"]);
    let err = max_abs_err(&reference, &fused_out);

    let naive_ms = median_ms(|| { let _ = eval(&prog, root, &env); }, iters);
    let fused_ms = median_ms(
        || { let _ = fused_mlp(&env["X_sd"], &env["W_up_df"], &env["W_dn_fd"]); },
        iters,
    );

    let na = account_expr(&prog, &shapes);
    let fa = account_expr(&plan, &shapes);
    BenchOutcome {
        naive_ms,
        fused_ms,
        speedup: naive_ms / fused_ms,
        max_abs_err: err,
        naive_hbm_bytes: na.hbm_bytes,
        fused_hbm_bytes: fa.hbm_bytes,
        flops: na.flops,
    }
}
