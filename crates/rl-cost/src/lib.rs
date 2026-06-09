// M1: Constraint trait and roofline cost model.
// Stub — implemented in milestone M1.

pub trait Constraint: Send + Sync {
    fn name(&self) -> &str;
    /// Lower bound on wall-clock time (seconds) for a physical plan with
    /// the given (flops, hbm_bytes) account.
    fn lower_bound_s(&self, flops: u64, hbm_bytes: u64) -> f64;
}

pub struct CostModel {
    constraints: Vec<Box<dyn Constraint>>,
}

impl CostModel {
    pub fn new() -> Self { Self { constraints: Vec::new() } }

    pub fn add(mut self, c: impl Constraint + 'static) -> Self {
        self.constraints.push(Box::new(c));
        self
    }

    /// Cost = slowest resource (roofline max).
    pub fn cost(&self, flops: u64, hbm_bytes: u64) -> (f64, &str) {
        self.constraints
            .iter()
            .map(|c| (c.lower_bound_s(flops, hbm_bytes), c.name()))
            .fold((0.0_f64, "none"), |(best_t, best_name), (t, name)| {
                if t > best_t { (t, name) } else { (best_t, best_name) }
            })
    }
}
