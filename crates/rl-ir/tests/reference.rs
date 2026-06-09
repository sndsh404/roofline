use rl_ir::{account, eval, infer_shape, naive_attention_program, TensorData};
use std::collections::HashMap;

// ── Fixture: computed by NumPy with seed 42, s=4, d=8 ────────────────────────
const S: usize = 4;
const D: usize = 8;

#[rustfmt::skip]
const Q: [f32; 32] = [
     0.49671414, -0.13826430,  0.64768857,  1.52302980,
    -0.23415338, -0.23413695,  1.57921278,  0.76743472,
    -0.46947438,  0.54256004, -0.46341768, -0.46572974,
     0.24196227, -1.91328025, -1.72491789, -0.56228751,
    -1.01283109,  0.31424734, -0.90802407, -1.41230369,
     1.46564877, -0.22577630,  0.06752820, -1.42474818,
    -0.54438275,  0.11092259, -1.15099359,  0.37569803,
    -0.60063869, -0.29169375, -0.60170662,  1.85227823,
];

#[rustfmt::skip]
const K: [f32; 32] = [
    -0.01349723, -1.05771089,  0.82254493, -1.22084367,
     0.20886360, -1.95967007, -1.32818604,  0.19686124,
     0.73846656,  0.17136829, -0.11564828, -0.30110368,
    -1.47852194, -0.71984422, -0.46063876,  1.05712223,
     0.34361830, -1.76304018,  0.32408398, -0.38508227,
    -0.67692202,  0.61167628,  1.03099954,  0.93128014,
    -0.83921754, -0.30921239,  0.33126342,  0.97554511,
    -0.47917423, -0.18565898, -1.10633492, -1.19620657,
];

#[rustfmt::skip]
const V: [f32; 32] = [
     0.81252581,  1.35624003, -0.07201012,  1.00353289,
     0.36163601, -0.64511973,  0.36139560,  1.53803658,
    -0.03582604,  1.56464362, -2.61974502,  0.82190251,
     0.08704707, -0.29900736,  0.09176078, -1.98756886,
    -0.21967189,  0.35711256,  1.47789407, -0.51827019,
    -0.80849361, -0.50175703,  0.91540211,  0.32875112,
    -0.52976018,  0.51326746,  0.09707755,  0.96864498,
    -0.70205307, -0.32766214, -0.39210814, -1.46351492,
];

// Expected output: softmax(Q K^T / sqrt(8)) @ V
#[rustfmt::skip]
const EXPECTED: [f32; 32] = [
    -0.13081041,  0.77212578,  0.10108928,  0.16807331,
    -0.46588686, -0.43681264,  0.46851459, -0.42075405,
     0.40109277,  1.19080400, -0.35037300,  0.94668907,
     0.08274233, -0.53010100,  0.17683369,  0.41923392,
     0.17910026,  0.98456150, -0.06763012,  0.80678093,
    -0.14453167, -0.49373081,  0.15002953,  0.10067090,
     0.01421369,  1.14907670, -0.99239492,  0.60451710,
    -0.14600018, -0.40496817,  0.24215066, -0.82777077,
];

fn make_env() -> HashMap<String, TensorData> {
    let scale = 1.0_f32 / (D as f32).sqrt();
    HashMap::from([
        ("Q_sd".into(),  TensorData::new(vec![S, D], Q.to_vec())),
        ("K_sd".into(),  TensorData::new(vec![S, D], K.to_vec())),
        ("V_sd".into(),  TensorData::new(vec![S, D], V.to_vec())),
        ("scale".into(), TensorData::scalar(scale)),
    ])
}

fn make_shapes() -> HashMap<String, Vec<usize>> {
    HashMap::from([
        ("Q_sd".into(),  vec![S, D]),
        ("K_sd".into(),  vec![S, D]),
        ("V_sd".into(),  vec![S, D]),
        ("scale".into(), vec![1]),
    ])
}

// ── Gate 1: numerics ──────────────────────────────────────────────────────────
// Reference interpreter must match the NumPy fixture to 1e-5.
#[test]
fn numerics_gate() {
    let env = make_env();
    let (expr, root) = naive_attention_program();
    let result = eval(&expr, root, &env);

    assert_eq!(result.shape, vec![S, D], "output shape mismatch");
    for (i, (got, exp)) in result.data.iter().zip(EXPECTED.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 1e-5,
            "element {}: got {:.8}, expected {:.8}, diff {:.2e}",
            i, got, exp, diff
        );
    }
}

// ── Gate 2: O(s²) HBM traffic captured ───────────────────────────────────────
// The [s,s] scores intermediate is the term Flash Attention eliminates.
// Accountant must report hbm_bytes ≥ size_of(scores_ss) for all s.
// (ratio-based checks are unreliable when s ≈ d; the direct term check is exact.)
#[test]
fn hbm_contains_s_squared_term() {
    let d = 64usize;
    for &s in &[32usize, 64, 128, 256] {
        let shapes = HashMap::from([
            ("Q_sd".into(),  vec![s, d]),
            ("K_sd".into(),  vec![s, d]),
            ("V_sd".into(),  vec![s, d]),
            ("scale".into(), vec![1]),
        ]);
        let (expr, root) = naive_attention_program();
        let acc = account(&expr, root, &shapes);
        // scores_ss is [s,s] f32 — the O(s²) tensor that Flash avoids writing.
        let scores_bytes = (s * s * 4) as u64;
        assert!(
            acc.hbm_bytes >= scores_bytes,
            "s={d}: hbm_bytes {} must be ≥ scores_ss {} bytes (O(s²) term missing)",
            acc.hbm_bytes, scores_bytes,
            d = s
        );
    }
}

// ── Shape correctness ─────────────────────────────────────────────────────────
#[test]
fn shape_inference() {
    let shapes = make_shapes();
    let (expr, root) = naive_attention_program();
    let out_shape = infer_shape(&expr, root, &shapes);
    assert_eq!(out_shape, vec![S, D]);
}
