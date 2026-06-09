use rl_ir::{account, naive_attention_program};
use std::collections::HashMap;

fn main() {
    println!("\nM0 ground-truth numbers, naive attention arithmetic intensity");
    println!("{:<6} {:<6} {:>14} {:>14} {:>12}", "s", "d", "FLOPs", "HBM bytes", "flop/byte");
    println!("{}", "─".repeat(56));

    for &s in &[64usize, 128, 256, 512, 1024] {
        let d = 64usize;
        let shapes = HashMap::from([
            ("Q_sd".into(),  vec![s, d]),
            ("K_sd".into(),  vec![s, d]),
            ("V_sd".into(),  vec![s, d]),
            ("scale".into(), vec![1usize]),
        ]);
        let (expr, root) = naive_attention_program();
        let acc = account(&expr, root, &shapes);
        println!(
            "{:<6} {:<6} {:>14} {:>14} {:>12.3}",
            s, d, acc.flops, acc.hbm_bytes, acc.intensity()
        );
    }

    println!();
    println!("Interpretation:");
    println!("  • Arithmetic intensity stays flat ~9–11 flop/byte as s grows.");
    println!("  • A100 ridge point ≈ 312 TFLOP/s ÷ 2 TB/s = 156 flop/byte.");
    println!("  • Gap ≈ 15× → naive attention is HBM-bandwidth-bound, not compute-bound.");
    println!("  • Flash Attention fuses the loop to avoid materialising scores_ss,");
    println!("    which is the O(s²) HBM write that drives this low intensity.");
    println!("  M1's job: build a cost model that reads this table and names the");
    println!("  binding resource. M3's job: select Flash form purely from cost.");
    println!();
}
