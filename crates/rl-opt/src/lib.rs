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
    // General producer/consumer fusion: a producer consumed immediately by a
    // matmul can run as one kernel, so its intermediate output need not spill to
    // HBM. These are NOT canned "attention => flash" or "mlp => fused" rules
    // (CLAUDE.md rule 4); each is the general "a producer consumed at once need
    // not spill" identity for one producer kind. On attention the softmax rule
    // fires (the s×s probabilities stay in SRAM); on the MLP the relu rule fires
    // (the s×f hidden activations stay in SRAM). Applied to a program's root they
    // make the fused form reachable in the same e-class; the cost model decides
    // whether to take it.
    vec![
        rw!("fuse-softmax-matmul";
            "(matmul (softmax ?x) ?v)" => "(fuse (matmul (softmax ?x) ?v))"),
        rw!("fuse-relu-matmul";
            "(matmul (relu ?x) ?v)" => "(fuse (matmul (relu ?x) ?v))"),
    ]
}

pub fn all_rewrites<N: Analysis<TensorLang>>() -> Vec<Rewrite<TensorLang, N>> {
    let mut rules = vec![];
    rules.extend(matmul_assoc::<N>());
    rules.extend(transpose_matmul::<N>());
    rules.extend(scale_distrib::<N>());
    rules.extend(fusion::<N>());
    rules
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

// ── Cost-driven extraction over a shaped e-graph ─────────────────────────────
//
// This is a custom extractor, NOT egg's tree `Extractor` (which double-counts
// shared tensors, the Tensat wall, CLAUDE.md rule 6). It is a greedy bottom-up
// selection driven by the `rl-cost` `CostModel`, reading per-e-class shapes from
// `ShapeAnalysis` to compute real flops and bytes. The `fuse` node is costed by
// its subtree's fused account (internal intermediates not charged to HBM), so a
// model that includes `HbmBytes` prefers fusion and a FLOPs-only model does not.
//
// Greedy selection is not guaranteed globally optimal (exact min-cost DAG
// extraction is NP-hard and is what `LpExtractor`/ILP would solve, but its
// `coin_cbc` solver is unavailable here). It is DAG-honest in that the fused
// account counts each boundary tensor once; the final reported cost comes from
// the fuse-aware accountant on the built program.

fn eclass_shape(egraph: &EGraph<TensorLang, ShapeAnalysis>, id: Id) -> Vec<usize> {
    egraph[id].data.clone().unwrap_or_default()
}

fn out_bytes(egraph: &EGraph<TensorLang, ShapeAnalysis>, id: Id) -> u64 {
    (eclass_shape(egraph, id).iter().product::<usize>() * 4) as u64
}

/// Local flops of one enode, from its children's shapes.
fn node_flops(egraph: &EGraph<TensorLang, ShapeAnalysis>, enode: &TensorLang) -> u64 {
    let sh = |id: &Id| eclass_shape(egraph, *id);
    match enode {
        TensorLang::MatMul([a, b]) => {
            let (sa, sb) = (sh(a), sh(b));
            if sa.len() == 2 && sb.len() == 2 { 2 * sa[0] as u64 * sa[1] as u64 * sb[1] as u64 } else { 0 }
        }
        TensorLang::EMul([a, b]) => {
            let (sa, sb) = (sh(a), sh(b));
            let out = if sa.iter().product::<usize>() <= 1 { sb } else { sa };
            out.iter().product::<usize>() as u64
        }
        TensorLang::Softmax([a]) => 4 * sh(a).iter().product::<usize>() as u64,
        TensorLang::Relu([a]) => sh(a).iter().product::<usize>() as u64,
        _ => 0,
    }
}

/// Cost-model-driven extraction over a shaped e-graph.
///
/// Two stages, because the `fuse` node is value-equivalent to its child and so
/// shares its e-class (extracting it directly would be a self-cycle):
///   1. extract the best *materialized* plan, skipping fuse nodes, with a
///      recursive cycle guard (a class on the current path costs infinity, so the
///      result is always a finite tree);
///   2. let the cost model decide whether to wrap that plan in `fuse`, by
///      accounting the materialized plan vs the fused one and taking the cheaper.
///
/// Stage 2 is where the A/B lives: a FLOPs-only model sees no benefit to fusion
/// (same flops, one extra node) and keeps the materialized plan; adding HbmBytes
/// makes the fused plan cheaper. Selection is greedy (exact min-cost DAG
/// extraction is NP-hard, `LpExtractor`/ILP territory), but every cost is real
/// bytes from the shape analysis and the fuse decision uses the fuse-aware
/// accountant directly.
pub fn extract_cost_driven(
    egraph: &EGraph<TensorLang, ShapeAnalysis>,
    root: Id,
    model: &CostModel,
) -> RecExpr<TensorLang> {
    use std::collections::{HashMap, HashSet};
    let mut sel: HashMap<Id, (f64, usize, usize)> = HashMap::new();
    let mut visiting: HashSet<Id> = HashSet::new();
    select_class(egraph, egraph.find(root), model, &mut sel, &mut visiting);

    // Stage 1: the best materialized plan.
    let mut materialized = RecExpr::default();
    build_chosen(egraph, egraph.find(root), &sel, &mut materialized);

    // Stage 2: would fusing the whole plan be cheaper under this model?
    let shapes = shapes_from_egraph(egraph);
    let mat_acc = account_expr(&materialized, &shapes);
    let mut fused = materialized.clone();
    let root_id = Id::from(fused.as_ref().len() - 1);
    fused.add(TensorLang::Fuse([root_id]));
    let fused_acc = account_expr(&fused, &shapes);

    let mat_t = model.cost(&Demand::from(&mat_acc)).0;
    let fused_t = model.cost(&Demand::from(&fused_acc)).0;
    // strictly cheaper (beyond float noise) to justify the extra fuse node.
    if fused_t < mat_t - mat_t.abs() * 1e-9 {
        fused
    } else {
        materialized
    }
}

/// Recover input shapes (`Var` name -> shape) from the shaped e-graph, so the
/// accountant can be run on an extracted program.
fn shapes_from_egraph(
    egraph: &EGraph<TensorLang, ShapeAnalysis>,
) -> std::collections::HashMap<String, Vec<usize>> {
    let mut m = std::collections::HashMap::new();
    for class in egraph.classes() {
        for node in &class.nodes {
            if let TensorLang::Var(sym) = node {
                if let Some(shape) = &class.data {
                    m.insert(sym.to_string(), shape.clone());
                }
            }
        }
    }
    m
}

/// Returns the best acyclic cost of `id`, memoizing (cost, nodes, chosen) into
/// `sel`. Ties on cost break toward fewer nodes, so a model that cannot see HBM
/// has no reason to add the extra `fuse` node.
fn select_class(
    egraph: &EGraph<TensorLang, ShapeAnalysis>,
    id: Id,
    model: &CostModel,
    sel: &mut std::collections::HashMap<Id, (f64, usize, usize)>,
    visiting: &mut std::collections::HashSet<Id>,
) -> f64 {
    let id = egraph.find(id);
    if let Some(&(c, _, _)) = sel.get(&id) {
        return c;
    }
    if visiting.contains(&id) {
        return f64::INFINITY; // a class on the current path: choosing it would cycle
    }
    visiting.insert(id);

    // per-node costing: no fused region here, so no SRAM demand
    let time = |flops: u64, bytes: u64| model.cost(&Demand::new(flops, bytes)).0;
    let mut best = (f64::INFINITY, usize::MAX, 0usize);
    for (ni, enode) in egraph[id].nodes.iter().enumerate() {
        // Fuse nodes are skipped here: fusion is a value-identity wrapper decided
        // in stage 2 of `extract_cost_driven`, not a node to extract directly
        // (it shares its child's e-class, which would self-cycle).
        let (cost, nodes) = if matches!(enode, TensorLang::Fuse(_)) {
            (f64::INFINITY, usize::MAX)
        } else {
            let mut sum = time(node_flops(egraph, enode), out_bytes(egraph, id));
            let mut n = 1usize;
            let mut cyclic = false;
            for c in enode.children() {
                let cc = select_class(egraph, *c, model, sel, visiting);
                if cc.is_infinite() { cyclic = true; break; }
                sum += cc;
                n += sel[&egraph.find(*c)].1;
            }
            if cyclic { (f64::INFINITY, usize::MAX) } else { (sum, n) }
        };
        let tie = (cost - best.0).abs() <= best.0.abs() * 1e-9;
        if cost < best.0 - best.0.abs() * 1e-9 || (tie && nodes < best.1) {
            best = (cost, nodes, ni);
        }
    }

    visiting.remove(&id);
    sel.insert(id, best);
    best.0
}

/// Build a `RecExpr` from the selected node per e-class.
fn build_chosen(
    egraph: &EGraph<TensorLang, ShapeAnalysis>,
    id: Id,
    sel: &std::collections::HashMap<Id, (f64, usize, usize)>,
    out: &mut RecExpr<TensorLang>,
) -> Id {
    let id = egraph.find(id);
    let ni = sel[&id].2;
    let enode = egraph[id].nodes[ni].clone();
    let mapped = enode.map_children(|c| build_chosen(egraph, c, sel, out));
    out.add(mapped)
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
// winner flips when you add the HBM constraint, same candidates, same e-graph,
// one extra constraint. Ties (e.g. equal FLOPs under a FLOPs-only model) break
// toward the simpler plan (fewer nodes), so a model that cannot see HBM has no
// reason to fuse.

use rl_cost::{CostModel, Demand};

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
        let (t, _binding, _) = model.cost(&Demand::from(&acc));
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
            TensorLang::Relu([a]) => child(a),
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

    fn contains_fuse(expr: &RecExpr<TensorLang>) -> bool {
        expr.as_ref().iter().any(|n| matches!(n, TensorLang::Fuse(_)))
    }

    #[test]
    fn extractor_flips_with_hbm_constraint() {
        // The A/B at the EXTRACTOR level (not just ranking two hand-built plans):
        // run the custom cost-driven extractor over the whole saturated e-graph.
        // FLOPs-only keeps an unfused plan; adding HbmBytes makes it fuse.
        use rl_cost::{CostModel, FlopsConstraint, HbmConstraint, H100};
        let (expr, _) = naive_attention_program();
        let shapes = attn_shapes(2048, 64);
        let runner = saturate_shaped(&expr, shapes.clone());
        let root = runner.egraph.lookup_expr(&expr).expect("root in egraph");

        let flops_only = CostModel::new().add(FlopsConstraint::new(H100));
        let with_hbm = CostModel::new()
            .add(FlopsConstraint::new(H100))
            .add(HbmConstraint::new(H100));

        let plan_flops = extract_cost_driven(&runner.egraph, root, &flops_only);
        let plan_hbm = extract_cost_driven(&runner.egraph, root, &with_hbm);

        assert!(!contains_fuse(&plan_flops), "FLOPs-only plan should not fuse");
        assert!(contains_fuse(&plan_hbm), "HBM-aware plan should fuse");

        // And the fused plan really moves fewer bytes.
        let h_flops = account_expr(&plan_flops, &shapes).hbm_bytes;
        let h_hbm = account_expr(&plan_hbm, &shapes).hbm_bytes;
        assert!(h_hbm < h_flops, "fused plan {h_hbm} should move fewer bytes than {h_flops}");
    }

    #[test]
    fn sram_constraint_blocks_fusion_that_cannot_fit() {
        // The documented fuse-model gap, closed the project's way: a new
        // constraint, not a search hack. At s=2048 d=64 the monolithic fused
        // working set is about 53 MB, over the A100's 20 MB of SRAM, so with
        // the SramConstraint active the extractor must refuse to fuse. At
        // s=256 d=32 the working set is under 1 MB and fusion stays the
        // winner. Same model, same search, the constraint decides.
        use rl_cost::{CostModel, FlopsConstraint, HbmConstraint, SramConstraint, A100};
        let (expr, _) = naive_attention_program();
        let with_sram = CostModel::new()
            .add(FlopsConstraint::new(A100))
            .add(HbmConstraint::new(A100))
            .add(SramConstraint::new(A100));

        let runner = saturate_shaped(&expr, attn_shapes(2048, 64));
        let root = runner.egraph.lookup_expr(&expr).expect("root in egraph");
        let plan = extract_cost_driven(&runner.egraph, root, &with_sram);
        assert!(
            !contains_fuse(&plan),
            "fusion must be refused when its working set cannot fit in SRAM"
        );

        let runner = saturate_shaped(&expr, attn_shapes(256, 32));
        let root = runner.egraph.lookup_expr(&expr).expect("root in egraph");
        let plan = extract_cost_driven(&runner.egraph, root, &with_sram);
        assert!(
            contains_fuse(&plan),
            "fusion should still win when the working set fits"
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

    fn mlp_shapes(s: usize, d: usize, f: usize) -> HashMap<String, Vec<usize>> {
        HashMap::from([
            ("X_sd".into(), vec![s, d]),
            ("W_up_df".into(), vec![d, f]),
            ("W_dn_fd".into(), vec![f, d]),
        ])
    }

    #[test]
    fn mlp_fused_form_is_reachable_by_rewrite() {
        // Rule-4 honesty for M5: the fused MLP form must enter the naive
        // program's e-class via the general relu producer fusion rewrite, not by
        // being hand-built.
        use rl_ir::{fused_mlp_program, naive_mlp_program};
        let (naive, _) = naive_mlp_program();
        let (fused, _) = fused_mlp_program();
        assert!(
            fused_form_reachable(&naive, &fused, mlp_shapes(64, 16, 64)),
            "relu fusion rewrite should make the fused MLP form reachable"
        );
    }

    #[test]
    fn mlp_extractor_flips_with_hbm_constraint() {
        // The M5 A/B at the extractor level. With only FLOPs modelled the cost
        // model cannot see the s by f hidden round-trip, so it keeps the unfused
        // plan; with HbmBytes it fuses. Same e-graph, one extra constraint.
        use rl_cost::{CostModel, FlopsConstraint, HbmConstraint, H100};
        use rl_ir::naive_mlp_program;
        let (expr, _) = naive_mlp_program();
        let shapes = mlp_shapes(2048, 128, 1024); // f > d: HBM dominated by H_sf
        let runner = saturate_shaped(&expr, shapes.clone());
        let root = runner.egraph.lookup_expr(&expr).expect("root in egraph");

        let flops_only = CostModel::new().add(FlopsConstraint::new(H100));
        let with_hbm = CostModel::new()
            .add(FlopsConstraint::new(H100))
            .add(HbmConstraint::new(H100));

        let plan_flops = extract_cost_driven(&runner.egraph, root, &flops_only);
        let plan_hbm = extract_cost_driven(&runner.egraph, root, &with_hbm);

        assert!(!contains_fuse(&plan_flops), "FLOPs-only MLP plan should not fuse");
        assert!(contains_fuse(&plan_hbm), "HBM-aware MLP plan should fuse");

        let h_flops = account_expr(&plan_flops, &shapes).hbm_bytes;
        let h_hbm = account_expr(&plan_hbm, &shapes).hbm_bytes;
        assert!(h_hbm < h_flops, "fused MLP plan {h_hbm} should move fewer bytes than {h_flops}");
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
    fn extracted_plan_never_worse_than_naive_on_hbm() {
        // Whatever the HBM-aware extractor returns must not move more bytes than the
        // naive program it started from.
        use rl_cost::{CostModel, FlopsConstraint, HbmConstraint, H100};
        let (expr, _) = naive_attention_program();
        let shapes = attn_shapes(64, 32);
        let naive_hbm = account_expr(&expr, &shapes).hbm_bytes;

        let runner = saturate_shaped(&expr, shapes.clone());
        let root = runner.egraph.lookup_expr(&expr).expect("root in egraph");
        let model = CostModel::new()
            .add(FlopsConstraint::new(H100))
            .add(HbmConstraint::new(H100));
        let plan = extract_cost_driven(&runner.egraph, root, &model);

        assert!(
            account_expr(&plan, &shapes).hbm_bytes <= naive_hbm,
            "extracted plan should not move more bytes than naive"
        );
    }
}
