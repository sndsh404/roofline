//! The `fuse` primitive must do two things: change nothing about the value, and
//! cut HBM by keeping the s×s intermediates out of HBM. These two tests are the
//! contract the M3 A/B relies on, the optimizer may only prefer the fused form
//! because it is cheaper AND still correct.

use std::collections::HashMap;

use rl_ir::{account, eval, fused_attention_program, naive_attention_program, TensorData};

fn attn_env(s: usize, d: usize) -> HashMap<String, TensorData> {
    // Deterministic pseudo-random inputs; values do not matter, only that naive
    // and fused see the same ones.
    let mut env = HashMap::new();
    let gen = |seed: usize, n: usize| -> Vec<f32> {
        (0..n).map(|i| (((i * 2654435761 + seed) % 1000) as f32) / 500.0 - 1.0).collect()
    };
    env.insert("Q_sd".into(), TensorData::new(vec![s, d], gen(1, s * d)));
    env.insert("K_sd".into(), TensorData::new(vec![s, d], gen(2, s * d)));
    env.insert("V_sd".into(), TensorData::new(vec![s, d], gen(3, s * d)));
    env.insert("scale".into(), TensorData::scalar(1.0 / (d as f32).sqrt()));
    env
}

fn attn_shapes(s: usize, d: usize) -> HashMap<String, Vec<usize>> {
    HashMap::from([
        ("Q_sd".into(), vec![s, d]),
        ("K_sd".into(), vec![s, d]),
        ("V_sd".into(), vec![s, d]),
        ("scale".into(), vec![1]),
    ])
}

#[test]
fn fused_attention_matches_naive_numerically() {
    let (s, d) = (48, 16);
    let env = attn_env(s, d);

    let (naive, naive_root) = naive_attention_program();
    let (fused, fused_root) = fused_attention_program();

    let a = eval(&naive, naive_root, &env);
    let b = eval(&fused, fused_root, &env);

    assert_eq!(a.shape, b.shape, "fused output shape must match naive");
    let max_err = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-5, "fused vs naive max abs err {max_err} exceeds 1e-5");
}

#[test]
fn fused_attention_cuts_hbm() {
    // For s >> d the naive plan's HBM is dominated by the s×s scores written and
    // read; fusion removes that, so fused HBM must be strictly and substantially
    // lower. flops stay essentially equal (same arithmetic).
    let (s, d) = (256, 32);
    let shapes = attn_shapes(s, d);

    let (naive, naive_root) = naive_attention_program();
    let (fused, fused_root) = fused_attention_program();

    let na = account(&naive, naive_root, &shapes);
    let fa = account(&fused, fused_root, &shapes);

    assert!(
        fa.hbm_bytes < na.hbm_bytes,
        "fused HBM {} should be < naive HBM {}",
        fa.hbm_bytes,
        na.hbm_bytes
    );
    // The s×s scores are s*s*4 bytes, written then read = 2*s*s*4. Fusion should
    // save at least the single s×s materialization.
    let one_ss = (s * s * 4) as u64;
    assert!(
        na.hbm_bytes - fa.hbm_bytes >= one_ss,
        "fusion should save at least one s*s tile ({} bytes); saved {}",
        one_ss,
        na.hbm_bytes - fa.hbm_bytes
    );
    // Arithmetic is unchanged by fusion.
    assert_eq!(na.flops, fa.flops, "fusion must not change flops");
}
