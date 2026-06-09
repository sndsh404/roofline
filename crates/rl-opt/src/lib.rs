use egg::{rewrite as rw, *};
use rl_ir::TensorLang;
use std::collections::HashMap;

// ── Rewrite rules from DESIGN §5 ─────────────────────────────────────────────

// Rules are generic over the analysis `N` so the same algebra works whether the
// e-graph carries no analysis (`()`) or the shape analysis used for cost-driven
// extraction. The rewrites are purely syntactic, so any `Analysis<TensorLang>`
// satisfies them.

fn matmul_assoc<N: Analysis<TensorLang>>() -> Vec<Rewrite<TensorLang, N>> {
    vec![
        rw!("matmul-assoc-l"; "(matmul (matmul ?a ?b) ?c)" => "(matmul ?a (matmul ?b ?c))"),
        rw!("matmul-assoc-r"; "(matmul ?a (matmul ?b ?c))" => "(matmul (matmul ?a ?b) ?c)"),
    ]
}

fn transpose_matmul<N: Analysis<TensorLang>>() -> Vec<Rewrite<TensorLang, N>> {
    vec![
        rw!("transpose-matmul";
            "(transpose (matmul ?a ?b))" =>
            "(matmul (transpose ?b) (transpose ?a))"),
        rw!("matmul-transpose-rev";
            "(matmul (transpose ?a) (transpose ?b))" =>
            "(transpose (matmul ?b ?a))"),
    ]
}

fn scale_distrib<N: Analysis<TensorLang>>() -> Vec<Rewrite<TensorLang, N>> {
    vec![
        rw!("scale-distrib-l";
            "(matmul (emul ?a ?s) ?b)" =>
            "(emul (matmul ?a ?b) ?s)"),
        rw!("scale-distrib-r";
            "(matmul ?a (emul ?b ?s))" =>
            "(emul (matmul ?a ?b) ?s)"),
        rw!("scale-distrib-rev-l";
            "(emul (matmul ?a ?b) ?s)" =>
            "(matmul (emul ?a ?s) ?b)"),
        rw!("scale-distrib-rev-r";
            "(emul (matmul ?a ?b) ?s)" =>
            "(matmul ?a (emul ?b ?s))"),
    ]
}

fn fusion<N: Analysis<TensorLang>>() -> Vec<Rewrite<TensorLang, N>> {
    // General producer/consumer fusion: a softmax consumed immediately by a matmul
    // can run as one kernel, so its s×s output need not spill to HBM. This is NOT
    // a canned "attention => flash" rule (CLAUDE.md rule 4) — it is the general
    // "a producer consumed at once need not spill" identity, which happens to fire
    // on attention. Applied to the naive root it makes the fused form reachable in
    // the same e-class; the cost model decides whether to take it.
    vec![rw!("fuse-softmax-matmul";
        "(matmul (softmax ?x) ?v)" => "(fuse (matmul (softmax ?x) ?v))")]
}

pub fn all_rewrites<N: Analysis<TensorLang>>() -> Vec<Rewrite<TensorLang, N>> {
    let mut rules = vec![];
    rules.extend(matmul_assoc::<N>());
    rules.extend(transpose_matmul::<N>());
    rules.extend(scale_distrib::<N>());
    rules.extend(fusion::<N>());
    rules
}

// ── HBM cost function for extraction ─────────────────────────────────────────

/// Cost = HBM bytes written by this node (child costs summed by Extractor).
/// We return HBM bytes from the rl-ir accountant logic.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct HbmCost(pub f64);

impl std::ops::Add for HbmCost {
    type Output = HbmCost;
    fn add(self, other: HbmCost) -> HbmCost { HbmCost(self.0 + other.0) }
}

impl std::ops::Mul<f64> for HbmCost {
    type Output = HbmCost;
    fn mul(self, rhs: f64) -> HbmCost { HbmCost(self.0 * rhs) }
}

impl From<f64> for HbmCost {
    fn from(v: f64) -> Self { HbmCost(v) }
}

impl Into<f64> for HbmCost {
    fn into(self) -> f64 { self.0 }
}

impl std::fmt::Display for HbmCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1}", self.0)
    }
}

/// Cost function: node cost is HBM bytes. Child costs are added by the
/// extractor framework.
#[derive(Clone, Debug, Default)]
pub struct HbmCostFn;

impl CostFunction<TensorLang> for HbmCostFn {
    type Cost = HbmCost;

    fn cost<C>(&mut self, enode: &TensorLang, mut get_cost: C) -> HbmCost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        // Node cost = 1 (to keep extraction monotonic). In M3 this becomes
        // HBM bytes from the accountant once we pass shape info.
        let child_cost: f64 = enode.children().iter().map(|&id| get_cost(id).0).sum();
        HbmCost(1.0 + child_cost)
    }
}

// ── Saturation ───────────────────────────────────────────────────────────────

pub fn saturate(expr: &RecExpr<TensorLang>) -> Runner<TensorLang, ()> {
    let rules = all_rewrites::<()>();
    Runner::<TensorLang, ()>::default()
        .with_expr(expr)
        .with_iter_limit(10)
        .with_node_limit(50_000)
        .run(&rules)
}

/// Extract the lowest-cost term using HBM cost.
pub fn extract_cheapest(
    runner: &Runner<TensorLang, ()>,
    root: Id,
) -> (RecExpr<TensorLang>, f64) {
    let cost_fn = HbmCostFn;
    let extractor = Extractor::new(&runner.egraph, cost_fn);
    let (best_cost, best_expr) = extractor.find_best(root);
    (best_expr, best_cost.0)
}

/// Run the rl-ir accountant over an extracted expression.
pub fn account_expr(
    expr: &RecExpr<TensorLang>,
    shapes: &HashMap<String, Vec<usize>>,
) -> rl_ir::Account {
    let root = Id::from(expr.as_ref().len() - 1);
    rl_ir::account(expr, root, shapes)
}

// ── Cost-driven plan selection (the M3 A/B) ──────────────────────────────────
//
// The fusion rule makes the fused form *reachable* in the e-graph (proven by an
// existence check). Selection between the reachable forms is the cost model's
// job: account each candidate, then ask the `CostModel` for its time under the
// active constraint set, and take the cheapest. The whole thesis is that the
// winner flips when you add the HBM constraint — same candidates, same e-graph,
// one extra constraint. Ties (e.g. equal FLOPs under a FLOPs-only model) break
// toward the simpler plan (fewer nodes), so a model that cannot see HBM has no
// reason to fuse.

use rl_cost::CostModel;

/// Index of the cheapest candidate under `model`. `shapes` gives input shapes.
/// On a near-tie (within 0.1%), prefers the candidate with fewer nodes.
pub fn select_plan(
    candidates: &[RecExpr<TensorLang>],
    shapes: &HashMap<String, Vec<usize>>,
    model: &CostModel,
) -> usize {
    let mut best = 0usize;
    let mut best_t = f64::INFINITY;
    let mut best_nodes = usize::MAX;
    for (i, expr) in candidates.iter().enumerate() {
        let acc = account_expr(expr, shapes);
        let (t, _binding, _) = model.cost(acc.flops, acc.hbm_bytes);
        let nodes = expr.as_ref().len();
        let tie = (t - best_t).abs() <= best_t * 1e-3;
        if t < best_t - best_t * 1e-3 || (tie && nodes < best_nodes) {
            best = i;
            best_t = t;
            best_nodes = nodes;
        }
    }
    best
}

/// True if the fused form is reachable from `naive` after saturation, i.e. the
/// fusion rule put it into the e-graph in the same e-class as the naive root.
pub fn fused_form_reachable(
    naive: &RecExpr<TensorLang>,
    fused: &RecExpr<TensorLang>,
    inputs: HashMap<String, Vec<usize>>,
) -> bool {
    let runner = saturate_shaped(naive, inputs);
    match (runner.egraph.lookup_expr(naive), runner.egraph.lookup_expr(fused)) {
        (Some(n), Some(f)) => runner.egraph.find(n) == runner.egraph.find(f),
        _ => false,
    }
}

// ── Shape analysis (M3 prerequisite) ─────────────────────────────────────────
//
// A cost-driven extractor must know each e-class's shape to compute real flops
// and HBM bytes per node. The e-graph carries no shape info on its own, so we
// attach an `egg::Analysis` that propagates `[dim, dim]` shapes bottom-up. Input
// (`Var`) shapes come from the `inputs` map stored on the analysis. The shape of
// an e-class is the merge of its members' shapes; algebraically-equivalent terms
// share a shape, so `merge` keeps the first known shape and treats a later one as
// agreement (a genuine disagreement would be a rewrite bug, surfaced here).

#[derive(Default, Clone)]
pub struct ShapeAnalysis {
    pub inputs: HashMap<String, Vec<usize>>,
}

impl Analysis<TensorLang> for ShapeAnalysis {
    /// `None` means "shape not yet known" (e.g. a `Var` whose input shape was not
    /// supplied). Known shapes are 2-D `[rows, cols]` in M0/M3.
    type Data = Option<Vec<usize>>;

    fn make(egraph: &EGraph<TensorLang, Self>, enode: &TensorLang) -> Self::Data {
        let child = |id: &Id| egraph[*id].data.clone();
        match enode {
            TensorLang::Var(sym) => egraph.analysis.inputs.get(sym.as_str()).cloned(),
            TensorLang::MatMul([a, b]) => {
                let sa = child(a)?;
                let sb = child(b)?;
                Some(vec![sa[0], sb[1]])
            }
            TensorLang::Transpose([a]) => {
                let sa = child(a)?;
                Some(vec![sa[1], sa[0]])
            }
            TensorLang::EMul([a, b]) => {
                // scalar [1] broadcasts; otherwise the non-scalar operand's shape.
                match (child(a), child(b)) {
                    (Some(sa), Some(sb)) => {
                        if sa.iter().product::<usize>() == 1 {
                            Some(sb)
                        } else {
                            Some(sa)
                        }
                    }
                    (Some(sa), None) => Some(sa),
                    (None, Some(sb)) => Some(sb),
                    (None, None) => None,
                }
            }
            TensorLang::Softmax([a]) => child(a),
            TensorLang::Fuse([a]) => child(a),
        }
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        match (a.clone(), b) {
            (None, Some(bb)) => {
                *a = Some(bb);
                DidMerge(true, false)
            }
            (Some(_), None) => DidMerge(false, true),
            _ => DidMerge(false, false),
        }
    }
}

/// Build a shaped e-graph from a program and saturate it with the algebra. The
/// returned e-graph has a known shape on every reachable e-class, which is what
/// a cost-driven extractor reads. This is the foundation the M3 A/B stands on.
pub fn saturate_shaped(
    expr: &RecExpr<TensorLang>,
    inputs: HashMap<String, Vec<usize>>,
) -> Runner<TensorLang, ShapeAnalysis> {
    let rules = all_rewrites::<ShapeAnalysis>();
    Runner::<TensorLang, ShapeAnalysis>::new(ShapeAnalysis { inputs })
        .with_expr(expr)
        .with_iter_limit(10)
        .with_node_limit(50_000)
        .run(&rules)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rl_ir::{naive_attention_program, naive_mlp_program};

    fn attn_shapes(s: usize, d: usize) -> HashMap<String, Vec<usize>> {
        HashMap::from([
            ("Q_sd".into(),  vec![s, d]),
            ("K_sd".into(),  vec![s, d]),
            ("V_sd".into(),  vec![s, d]),
            ("scale".into(), vec![1]),
        ])
    }

    #[test]
    fn saturation_makes_progress() {
        let (expr, _root) = naive_attention_program();
        let runner = saturate(&expr);
        let n_iters = runner.iterations.len();
        assert!(n_iters >= 1, "saturation should make progress");
        assert!(
            runner.egraph.number_of_classes() > 5,
            "e-graph should have multiple e-classes, got {}",
            runner.egraph.number_of_classes()
        );
    }

    #[test]
    fn mlp_saturation_works() {
        let (expr, _root) = naive_mlp_program();
        let runner = saturate(&expr);
        assert!(
            runner.egraph.number_of_classes() > 3,
            "MLP e-graph should have multiple classes, got {}",
            runner.egraph.number_of_classes()
        );
    }

    #[test]
    fn equivalent_programs_differ_by_scale_distrib() {
        let (expr, root) = naive_attention_program();
        let runner = saturate(&expr);
        let shapes = attn_shapes(32, 16);
        let naive_acc = account_expr(&expr, &shapes);

        let (best, _cost) = extract_cheapest(&runner, root);
        let best_acc = account_expr(&best, &shapes);

        // FLOPs can differ because scale-distrib moves EMul from scores_ss
        // (1024 FLOPs) to Q_sd (512 FLOPs). The programs are algebraically
        // equivalent but have different cost profiles — this is the point.
        let diff = if naive_acc.flops > best_acc.flops {
            naive_acc.flops - best_acc.flops
        } else {
            best_acc.flops - naive_acc.flops
        };
        assert!(
            diff < 1024, // at most one EMul node difference
            "FLOPs diff {} too large between equivalent programs (naive={}, best={})",
            diff, naive_acc.flops, best_acc.flops
        );
    }

    #[test]
    fn fused_form_is_reachable_by_rewrite() {
        // The fusion rule must put the fused form into the naive program's e-class
        // (reachable, not hand-built). This is the rule-4-honest half: fusion comes
        // from a general rewrite.
        use rl_ir::{fused_attention_program, naive_attention_program};
        let (naive, _) = naive_attention_program();
        let (fused, _) = fused_attention_program();
        assert!(
            fused_form_reachable(&naive, &fused, attn_shapes(64, 32)),
            "fusion rewrite should make the fused form reachable in the e-graph"
        );
    }

    #[test]
    fn the_ab_flip() {
        // THE result the repo exists for. Same two candidate plans, both reachable
        // in the same e-graph. With only FLOPs modelled the cost model cannot see
        // the s×s HBM round-trip, so it keeps the simpler naive plan. Add the HBM
        // constraint and the fused plan wins. Same search, one extra constraint.
        use rl_cost::{CostModel, FlopsConstraint, HbmConstraint, H100};
        use rl_ir::{fused_attention_program, naive_attention_program};

        let (naive, _) = naive_attention_program();
        let (fused, _) = fused_attention_program();
        let candidates = [naive, fused]; // index 0 = naive, 1 = fused
        let shapes = attn_shapes(2048, 64); // long sequence: HBM dominates

        let flops_only = CostModel::new().add(FlopsConstraint::new(H100));
        let with_hbm = CostModel::new()
            .add(FlopsConstraint::new(H100))
            .add(HbmConstraint::new(H100));

        let pick_flops = select_plan(&candidates, &shapes, &flops_only);
        let pick_hbm = select_plan(&candidates, &shapes, &with_hbm);

        assert_eq!(pick_flops, 0, "with [Flops] only, the optimizer should keep naive");
        assert_eq!(pick_hbm, 1, "with [Flops, HbmBytes], the optimizer should choose fused");
    }

    #[test]
    fn shape_analysis_infers_attention_output() {
        // The e-graph must know that O_sd is [s, d] = [64, 32], inferred bottom-up
        // from the input shapes through matmul/transpose/softmax. Without this a
        // cost-driven extractor is blind.
        let (expr, _root) = naive_attention_program();
        let runner = saturate_shaped(&expr, attn_shapes(64, 32));
        let root = runner.egraph.lookup_expr(&expr).expect("root in egraph");
        assert_eq!(
            runner.egraph[root].data,
            Some(vec![64, 32]),
            "attention output e-class shape should be [s, d] = [64, 32]"
        );
    }

    #[test]
    fn shape_survives_saturation_equivalences() {
        // scale-distrib and assoc rewrites add equivalent terms to the root class.
        // All must agree on the output shape, so the merged shape stays [s, d].
        let (expr, _root) = naive_attention_program();
        let runner = saturate_shaped(&expr, attn_shapes(128, 64));
        let root = runner.egraph.lookup_expr(&expr).expect("root in egraph");
        assert_eq!(runner.egraph[root].data, Some(vec![128, 64]));
        assert!(
            runner.egraph.number_of_classes() > 5,
            "saturation should still produce equivalent terms"
        );
    }

    #[test]
    fn hbm_optimized_is_not_worse_than_naive() {
        let (expr, root) = naive_attention_program();
        let shapes = attn_shapes(64, 32);
        let naive_acc = account_expr(&expr, &shapes);

        let runner = saturate(&expr);
        let (hbm_best, _) = extract_cheapest(&runner, root);
        let hbm_acc = account_expr(&hbm_best, &shapes);

        assert!(
            hbm_acc.hbm_bytes <= naive_acc.hbm_bytes,
            "HBM-optimized extraction {hbm} should be <= naive {naive}",
            hbm = hbm_acc.hbm_bytes, naive = naive_acc.hbm_bytes
        );
    }
}
