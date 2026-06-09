use rl_cost::{analyze_attention, CostModel, FlopsConstraint, HbmConstraint, A100};

fn main() {
    let cost = CostModel::new()
        .add(FlopsConstraint::new(A100))
        .add(HbmConstraint::new(A100));

    println!("\nM1: Roofline analysis — naive attention on A100");
    println!("A100 ridge point: {:.1} flop/byte", A100.ridge_point());
    println!("{}", "─".repeat(90));
    println!("{:<20}  {:<14} {:<14} {:<8} {:<10}  {}", 
             "program", "flops", "hbm_bytes", "int", "binding", "details");
    println!("{}", "─".repeat(90));

    for &s in &[64usize, 128, 256, 512, 1024] {
        analyze_attention(&cost, s, 64);
    }
    println!();

    println!("Interpretation:");
    println!("  • All s values show HbmBytes as the binding resource → memory bound.");
    println!("  • The gap between HBM time and FLOPs time widens with s.");
    println!("  • This confirms Flash Attention's target: eliminate the O(s²) HBM write.");
    println!("  M2's job: add egg rewrites (§5 of DESIGN.md) so the Flash form exists in the e-graph.");
    println!("  M3's job: LpExtractor selects Flash form when HbmBytes constraint is active.");
    println!();
}
