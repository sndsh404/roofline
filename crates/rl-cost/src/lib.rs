use std::fmt;

// ── Device spec ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Device {
    pub name: &'static str,
    /// Peak FLOP/s (FP16).
    pub peak_flops: f64,
    /// HBM bandwidth (bytes/s).
    pub hbm_bandwidth: f64,
}

impl Device {
    pub const fn new(name: &'static str, peak_flops: f64, hbm_bandwidth: f64) -> Self {
        Self { name, peak_flops, hbm_bandwidth }
    }

    pub fn ridge_point(&self) -> f64 {
        self.peak_flops / self.hbm_bandwidth
    }
}

pub const A100: Device = Device::new("A100-80GB", 312e12, 2.0e12);
pub const H100: Device = Device::new("H100-SXM", 989e12, 3.35e12);

// ── Constraint trait ─────────────────────────────────────────────────────────

pub trait Constraint: Send + Sync + fmt::Debug {
    fn name(&self) -> &str;
    fn lower_bound_s(&self, flops: u64, hbm_bytes: u64) -> f64;
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

    fn lower_bound_s(&self, flops: u64, _hbm_bytes: u64) -> f64 {
        flops as f64 / self.device.peak_flops
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

    fn lower_bound_s(&self, _flops: u64, hbm_bytes: u64) -> f64 {
        hbm_bytes as f64 / self.device.hbm_bandwidth
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
    pub fn cost(&self, flops: u64, hbm_bytes: u64) -> (f64, String, Vec<(String, f64)>) {
        let mut times: Vec<(String, f64)> = self
            .constraints
            .iter()
            .map(|c| (c.name().to_string(), c.lower_bound_s(flops, hbm_bytes)))
            .collect();

        times.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let (best_name, best_time) = times.first().map(|(n, t)| (n.clone(), *t)).unwrap_or(("none".into(), 0.0));
        (best_time, best_name, times)
    }

    /// Convenience: print a formatted roofline analysis line.
    pub fn print_analysis(&self, label: &str, flops: u64, hbm_bytes: u64) {
        let intensity = if hbm_bytes == 0 { 0.0 } else { flops as f64 / hbm_bytes as f64 };
        let (_time, binding, details) = self.cost(flops, hbm_bytes);

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
        let t = c.lower_bound_s(312e12 as u64, 0);
        assert!((t - 1.0).abs() < 1e-9, "312 TFLOPS on A100 should give ~1s, got {}", t);
    }

    #[test]
    fn hbm_constraint_basic() {
        let c = HbmConstraint::new(A100);
        let t = c.lower_bound_s(0, (2.0e12) as u64);
        assert!((t - 1.0).abs() < 1e-9, "2 TB on A100 should give ~1s, got {}", t);
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
        let (_, binding, details) = cost.cost(acc.flops, acc.hbm_bytes);

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
        let (_, binding, _) = cost.cost(312e12 as u64, 1);
        assert_eq!(binding, "Flops", "extreme compute should be Flops-bound");
    }
}
