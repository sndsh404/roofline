//! M5 end to end: the optimizer chooses the fused MLP plan, codegen lowers it
//! to the streaming kernel, and we record naive vs fused wall clock across the
//! f/d boundary. The claim regime is f > d, where the s by f hidden
//! activations dominate the naive plan's memory bill. CPU numbers; the
//! ragged_dot comparison on an A100 is deferred until there is an A100.
//!
//! Run: cargo run -p rl-codegen --release --example m5_bench

use rl_codegen::bench::bench_mlp;

fn main() {
    let (s, d, seed) = (2048usize, 128usize, 42u64);
    println!("fused MLP vs naive reference, s={s} d={d} seed={seed}");
    println!(
        "{:>6}  {:>5}  {:>12}  {:>12}  {:>8}  {:>12}  {:>16}",
        "f", "f/d", "naive (ms)", "fused (ms)", "speedup", "max_abs_err", "hbm naive/fused"
    );
    for &f in &[64usize, 128, 256, 512, 1024, 2048] {
        let iters = if f >= 1024 { 5 } else { 7 };
        let o = bench_mlp(s, d, f, iters, seed);
        println!(
            "{:>6}  {:>5.1}  {:>12.2}  {:>12.2}  {:>7.2}x  {:>12.2e}  {:>7} / {:<7}",
            f,
            f as f64 / d as f64,
            o.naive_ms,
            o.fused_ms,
            o.speedup,
            o.max_abs_err,
            o.naive_hbm_bytes / (1 << 20),
            o.fused_hbm_bytes / (1 << 20),
        );
    }
    println!(
        "\nhbm columns are the accountant's bytes in MB for the naive plan and\n\
         the plan the optimizer extracted. the optimizer chose the fused plan at\n\
         every shape above (the bench asserts it). wall clock is CPU only."
    );
}
