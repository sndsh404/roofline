use std::fmt;

// ── Device spec ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Device {
    pub name: &'static str,
    /// Peak FLOP/s (FP16).
    pub peak_flops: f64,
    /// HBM bandwidth (bytes/s).
    pub hbm_bandwidth: f64,
    /// Total on-chip SRAM across SMs (bytes), the FlashAttention accounting:
    /// A100 has 192 KB per SM times 108 SMs, about 20 MB; H100 has 228 KB per
    /// SM times 132 SMs, about 30 MB. This is what a fused region's working
    /// set must fit inside.
    pub sram_bytes: u64,
}

impl Device {
    pub const fn new(
        name: &'static str,
        peak_flops: f64,
        hbm_bandwidth: f64,
        sram_bytes: u64,
    ) -> Self {
        Self { name, peak_flops, hbm_bandwidth, sram_bytes }
    }

    pub fn ridge_point(&self) -> f64 {
        self.peak_flops / self.hbm_bandwidth
    }
}

pub const A100: Device = Device::new("A100-80GB", 312e12, 2.0e12, 20_000_000);
pub const H100: Device = Device::new("H100-SXM", 989e12, 3.35e12, 30_000_000);

// ── Demand: what a plan asks of the machine ──────────────────────────────────

/// The resource demands of one plan, as the accountant measured them. Each
/// constraint reads the part it cares about. `sram_bytes` is the peak working
/// set of any fused region; zero for plans that materialize everything, since
/// per-op tile working sets are assumed to fit.
#[derive(Clone, Copy, Debug, Default)]
pub struct Demand {
    pub flops: u64,
    pub hbm_bytes: u64,
    pub sram_bytes: u64,
}

impl Demand {
    pub const fn new(flops: u64, hbm_bytes: u64) -> Self {
        Self { flops, hbm_bytes, sram_bytes: 0 }
    }

    pub const fn with_sram(mut self, sram_bytes: u64) -> Self {
        self.sram_bytes = sram_bytes;
        self
    }
}

impl From<&rl_ir::Account> for Demand {
    fn from(a: &rl_ir::Account) -> Self {
        Self { flops: a.flops, hbm_bytes: a.hbm_bytes, sram_bytes: a.sram_bytes }
    }
}

// ── Constraint trait ─────────────────────────────────────────────────────────

pub trait Constraint: Send + Sync + fmt::Debug {
    fn name(&self) -> &str;
    fn lower_bound_s(&self, demand: &Demand) -> f64;
}

// ── Built-in constraints ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct FlopsConstraint {
    device: Device,
}

impl FlopsConstraint {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl Constraint for FlopsConstraint {
    fn name(&self) -> &str {
        "Flops"
    }

    fn lower_bound_s(&self, demand: &Demand) -> f64 {
        demand.flops as f64 / self.device.peak_flops
    }
}

#[derive(Clone, Debug)]
pub struct HbmConstraint {
    device: Device,
}

impl HbmConstraint {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl Constraint for HbmConstraint {
    fn name(&self) -> &str {
        "HbmBytes"
    }

    fn lower_bound_s(&self, demand: &Demand) -> f64 {
        demand.hbm_bytes as f64 / self.device.hbm_bandwidth
    }
}

/// SRAM capacity. The fuse model assumes a fused region's working set lives
/// entirely on chip; this constraint is where that assumption gets enforced
/// instead of assumed. A plan whose fused working set fits contributes no
/// time floor. One that does not fit cannot run as scheduled at all (the IR
/// cannot tile yet), so its lower bound is infinite and the binding resource
/// says why: SramBytes. This is the documented M3/M4 gap closed the way the
/// prime directive demands, a new constraint, not a search hack.
#[derive(Clone, Debug)]
pub struct SramConstraint {
    device: Device,
}

impl SramConstraint {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl Constraint for SramConstraint {
    fn name(&self) -> &str {
        "SramBytes"
    }

    fn lower_bound_s(&self, demand: &Demand) -> f64 {
        if demand.sram_bytes <= self.device.sram_bytes {
            0.0
        } else {
            f64::INFINITY
        }
    }
}

// ── Cost model ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CostModel {
    constraints: Vec<Box<dyn Constraint>>,
}

impl CostModel {
    pub fn new() -> Self {
        Self { constraints: Vec::new() }
    }

    pub fn add(mut self, c: impl Constraint + 'static) -> Self {
        self.constraints.push(Box::new(c));
        self
    }

    /// Returns (best_time_s, binding_resource_name, per_constraint_times).
    pub fn cost(&self, demand: &Demand) -> (f64, String, Vec<(String, f64)>) {
        let mut times: Vec<(String, f64)> = self
            .constraints
            .iter()
            .map(|c| (c.name().to_string(), c.lower_bound_s(demand)))
            .collect();

        times.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let (best_name, best_time) = times.first().map(|(n, t)| (n.clone(), *t)).unwrap_or(("none".into(), 0.0));
        (best_time, best_name, times)
    }

    /// Convenience: print a formatted roofline analysis line.
    pub fn print_analysis(&self, label: &str, flops: u64, hbm_bytes: u64) {
        let intensity = if hbm_bytes == 0 { 0.0 } else { flops as f64 / hbm_bytes as f64 };
        let (_time, binding, details) = self.cost(&Demand::new(flops, hbm_bytes));

        let detail_str: String = details
            .iter()
            .map(|(n, t)| format!("{}={:.2e}s", n, t))
            .collect::<Vec<_>>()
            .join(", ");

        println!(
            "{:<20}  flops={:<14}  hbm={:<14}  int={:<8.2}  binding={:<10}  {}",
            label, flops, hbm_bytes, intensity, binding, detail_str
        );
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

// ── Roofline analysis for canonical programs ─────────────────────────────────

pub fn analyze_attention(cost: &CostModel, s: usize, d: usize) {
    use rl_ir::{account, naive_attention_program};
    use std::collections::HashMap;

    let shapes = HashMap::from([
        ("Q_sd".into(),  vec![s, d]),
        ("K_sd".into(),  vec![s, d]),
        ("V_sd".into(),  vec![s, d]),
        ("scale".into(), vec![1usize]),
    ]);
    let (expr, root) = naive_attention_program();
    let acc = account(&expr, root, &shapes);
    cost.print_analysis(&format!("attn_s{}_{}", s, d), acc.flops, acc.hbm_bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn a100_ridge_point() {
        let rp = A100.ridge_point();
        assert!((rp - 156.0).abs() < 1.0, "A100 ridge point should be ~156 flop/byte, got {}", rp);
    }

    #[test]
    fn flops_constraint_basic() {
        let c = FlopsConstraint::new(A100);
        let t = c.lower_bound_s(&Demand::new(312e12 as u64, 0));
        assert!((t - 1.0).abs() < 1e-9, "312 TFLOPS on A100 should give ~1s, got {}", t);
    }

    #[test]
    fn hbm_constraint_basic() {
        let c = HbmConstraint::new(A100);
        let t = c.lower_bound_s(&Demand::new(0, (2.0e12) as u64));
        assert!((t - 1.0).abs() < 1e-9, "2 TB on A100 should give ~1s, got {}", t);
    }

    #[test]
    fn sram_constraint_gates_fused_plans() {
        let c = SramConstraint::new(A100); // 20 MB of on-chip SRAM
        // a fused working set that fits contributes no time floor
        let fits = Demand::new(1, 1).with_sram(3_800_000);
        assert_eq!(c.lower_bound_s(&fits), 0.0);
        // one that does not fit cannot run as scheduled: infinite floor
        let too_big = Demand::new(1, 1).with_sram(53_000_000);
        assert!(c.lower_bound_s(&too_big).is_infinite());

        // and through the full model the binding resource names the reason
        let model = CostModel::new()
            .add(FlopsConstraint::new(A100))
            .add(HbmConstraint::new(A100))
            .add(SramConstraint::new(A100));
        let (t, binding, _) = model.cost(&too_big);
        assert!(t.is_infinite());
        assert_eq!(binding, "SramBytes", "the model must say WHY the plan is infeasible");
    }

    #[test]
    fn binding_resource_is_hbm_for_attention() {
        let cost = CostModel::new()
            .add(FlopsConstraint::new(A100))
            .add(HbmConstraint::new(A100));

        // For naive attention s=1024, d=64: intensity ~19.7 flop/byte < 156 ridge → HBM-bound.
        let shapes = HashMap::from([
            ("Q_sd".into(),  vec![1024usize, 64]),
            ("K_sd".into(),  vec![1024usize, 64]),
            ("V_sd".into(),  vec![1024usize, 64]),
            ("scale".into(), vec![1usize]),
        ]);
        let (expr, root) = rl_ir::naive_attention_program();
        let acc = rl_ir::account(&expr, root, &shapes);
        let (_, binding, details) = cost.cost(&Demand::from(&acc));

        assert_eq!(binding, "HbmBytes", "naive attention should be HBM-bound on A100");
        // HBM time should be strictly greater than FLOPs time
        let flops_t = details.iter().find(|(n, _)| n == "Flops").map(|(_, t)| *t).unwrap();
        let hbm_t = details.iter().find(|(n, _)| n == "HbmBytes").map(|(_, t)| *t).unwrap();
        assert!(hbm_t > flops_t, "HBM time {} should exceed FLOPs time {}", hbm_t, flops_t);
    }

    #[test]
    fn compute_bound_when_flops_dominated() {
        let cost = CostModel::new()
            .add(FlopsConstraint::new(A100))
            .add(HbmConstraint::new(A100));

        // Extreme case: 1 byte of data, 312 TFLOPS of compute → must be compute-bound.
        let (_, binding, _) = cost.cost(&Demand::new(312e12 as u64, 1));
        assert_eq!(binding, "Flops", "extreme compute should be Flops-bound");
    }
}
