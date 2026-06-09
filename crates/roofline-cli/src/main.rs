//! The `roofline` command. Top of the dependency stack: it glues the ledger
//! (claims and results) to the bench runners in rl-codegen, so a recorded
//! number is produced by the same code path every time.
//!
//!   roofline prereg --bench mlp --metric speedup --claim "..." \
//!       --threshold 1.10 --seed 43 --param s=2048 --param d=128 \
//!       --param f=1024 --param iters=7
//!   roofline run <run_id>       first measurement of a preregistered claim
//!   roofline replay <run_id>    re-run from the committed config, re-judge
//!   roofline list               show every record in the ledger
//!
//! Timing matters, so run through release: cargo run -p roofline-cli
//! --release -- <subcommand ...>. The ledger lives at ledger/wal.jsonl unless
//! --ledger says otherwise.

use std::collections::BTreeMap;
use std::process::exit;

use rl_codegen::bench::{bench_attention, bench_mlp, BenchOutcome};
use rl_ledger::{Ledger, Prereg, Record};

const DEFAULT_LEDGER: &str = "ledger/wal.jsonl";
const NUMERICS_GATE: f64 = 1e-5;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (ledger_path, args) = take_flag_value(args, "--ledger");
    let ledger_path = ledger_path.unwrap_or_else(|| DEFAULT_LEDGER.to_string());

    match args.first().map(String::as_str) {
        Some("prereg") => cmd_prereg(&ledger_path, &args[1..]),
        Some("run") => cmd_measure(&ledger_path, &args[1..], false),
        Some("replay") => cmd_measure(&ledger_path, &args[1..], true),
        Some("list") => cmd_list(&ledger_path),
        _ => {
            eprintln!("usage: roofline [--ledger <path>] <prereg|run|replay|list> ...");
            exit(2);
        }
    }
}

/// Pull `--flag value` out of the arg list if present.
fn take_flag_value(args: Vec<String>, flag: &str) -> (Option<String>, Vec<String>) {
    let mut out = Vec::with_capacity(args.len());
    let mut value = None;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        if a == flag {
            value = it.next();
        } else {
            out.push(a);
        }
    }
    (value, out)
}

fn require(opt: Option<String>, what: &str) -> String {
    opt.unwrap_or_else(|| {
        eprintln!("missing required {what}");
        exit(2);
    })
}

fn cmd_prereg(ledger_path: &str, args: &[String]) {
    let mut bench = None;
    let mut metric = None;
    let mut claim = None;
    let mut threshold = None;
    let mut seed = None;
    let mut params: BTreeMap<String, u64> = BTreeMap::new();

    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| require(it.next().cloned(), name);
        match a.as_str() {
            "--bench" => bench = Some(val("--bench value")),
            "--metric" => metric = Some(val("--metric value")),
            "--claim" => claim = Some(val("--claim value")),
            "--threshold" => threshold = Some(val("--threshold value")),
            "--seed" => seed = Some(val("--seed value")),
            "--param" => {
                let kv = val("--param value");
                let (k, v) = kv.split_once('=').unwrap_or_else(|| {
                    eprintln!("--param wants key=value, got {kv}");
                    exit(2);
                });
                params.insert(k.to_string(), v.parse().unwrap_or_else(|_| {
                    eprintln!("--param {k} wants an integer, got {v}");
                    exit(2);
                }));
            }
            other => {
                eprintln!("unknown prereg flag {other}");
                exit(2);
            }
        }
    }

    let bench = require(bench, "--bench");
    if bench != "attention" && bench != "mlp" {
        eprintln!("--bench must be attention or mlp");
        exit(2);
    }
    let threshold: f64 = require(threshold, "--threshold").parse().unwrap_or_else(|_| {
        eprintln!("--threshold wants a number");
        exit(2);
    });
    let seed: u64 = require(seed, "--seed").parse().unwrap_or_else(|_| {
        eprintln!("--seed wants an integer");
        exit(2);
    });

    let mut led = open(ledger_path);
    let p = led
        .prereg(
            &bench,
            &require(metric, "--metric"),
            &require(claim, "--claim"),
            threshold,
            NUMERICS_GATE,
            seed,
            params,
        )
        .unwrap_or_else(die("prereg"));
    println!("preregistered {}", p.run_id);
    println!("  claim: {}", p.claim);
    println!("  success: {} >= {} and max_abs_err < {}", p.metric, p.threshold, p.numerics_gate);
    println!("  params: {:?}, seed {}", p.params, p.seed);
}

fn run_bench(p: &Prereg) -> BenchOutcome {
    let get = |k: &str| -> usize {
        *p.params.get(k).unwrap_or_else(|| {
            eprintln!("prereg {} is missing param {k}", p.run_id);
            exit(2);
        }) as usize
    };
    let iters = get("iters");
    match p.bench.as_str() {
        "attention" => bench_attention(get("s"), get("d"), iters, p.seed),
        "mlp" => bench_mlp(get("s"), get("d"), get("f"), iters, p.seed),
        other => {
            eprintln!("unknown bench {other}");
            exit(2);
        }
    }
}

fn cmd_measure(ledger_path: &str, args: &[String], replaying: bool) {
    let run_id = require(args.first().cloned(), "run id");
    let mut led = open(ledger_path);
    let p = led
        .get_prereg(&run_id)
        .unwrap_or_else(|| {
            eprintln!("no prereg named {run_id} in {ledger_path}");
            exit(2);
        })
        .clone();
    let prior = led.latest_result(&run_id).cloned();
    if replaying && prior.is_none() {
        eprintln!("{run_id} has no recorded result to replay; use `roofline run` first");
        exit(2);
    }
    if !replaying && prior.is_some() {
        eprintln!(
            "{run_id} already has a recorded result; use `roofline replay` to re-measure"
        );
        exit(2);
    }

    if cfg!(debug_assertions) {
        eprintln!("warning: debug build, wall clock is meaningless; run with --release");
    }

    let o = run_bench(&p);
    let numbers = BTreeMap::from([
        ("naive_ms".to_string(), o.naive_ms),
        ("fused_ms".to_string(), o.fused_ms),
        ("speedup".to_string(), o.speedup),
        ("max_abs_err".to_string(), o.max_abs_err),
        ("naive_hbm_bytes".to_string(), o.naive_hbm_bytes as f64),
        ("fused_hbm_bytes".to_string(), o.fused_hbm_bytes as f64),
        ("flops".to_string(), o.flops as f64),
    ]);
    let metric_value = match p.metric.as_str() {
        "speedup" => o.speedup,
        other => {
            eprintln!("unknown metric {other}");
            exit(2);
        }
    };
    let r = led
        .record_result(&run_id, metric_value, o.max_abs_err, numbers)
        .unwrap_or_else(die("record"));

    println!("{} v{}{}", run_id, r.version, if r.replay { " (replay)" } else { "" });
    println!("  claim: {}", p.claim);
    println!(
        "  naive {:.2} ms, fused {:.2} ms, speedup {:.3}x, max_abs_err {:.2e}",
        o.naive_ms, o.fused_ms, o.speedup, o.max_abs_err
    );
    println!(
        "  accountant: flops {}, hbm naive {} fused {} bytes",
        o.flops, o.naive_hbm_bytes, o.fused_hbm_bytes
    );
    if let Some(prev) = prior {
        println!(
            "  original v{}: {} = {:.3}, claim_met {}",
            prev.version, p.metric, prev.metric_value, prev.claim_met
        );
    }
    println!(
        "  verdict: {} = {:.3} against threshold {:.3}, claim_met {}",
        p.metric, r.metric_value, p.threshold, r.claim_met
    );
    if !r.claim_met {
        exit(1);
    }
}

fn cmd_list(ledger_path: &str) {
    let led = open(ledger_path);
    if led.records().is_empty() {
        println!("ledger {ledger_path} is empty");
        return;
    }
    for rec in led.records() {
        match rec {
            Record::Prereg(p) => println!(
                "prereg  {}  bench {}  {} >= {}  seed {}  params {:?}",
                p.run_id, p.bench, p.metric, p.threshold, p.seed, p.params
            ),
            Record::Result(r) => println!(
                "result  {}  v{}  metric {:.3}  err {:.2e}  claim_met {}{}",
                r.run_id,
                r.version,
                r.metric_value,
                r.max_abs_err,
                r.claim_met,
                if r.replay { "  (replay)" } else { "" }
            ),
        }
    }
}

fn open(path: &str) -> Ledger {
    Ledger::open(path).unwrap_or_else(die("open ledger"))
}

fn die<T>(what: &'static str) -> impl FnOnce(std::io::Error) -> T {
    move |e| {
        eprintln!("{what} failed: {e}");
        exit(1);
    }
}
