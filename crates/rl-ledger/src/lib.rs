//! M5: the run ledger. An append-only write-ahead log of preregistered
//! benchmark claims and their measured results.
//!
//! Why it exists: a benchmark number you can quietly re-run until it looks
//! good is not evidence. The ledger forces the order the project's rules
//! demand. First `prereg` commits the bench, the shape, the seed, the metric,
//! and the success threshold to the log. Only then is the bench run, and the
//! result is judged against the preregistered claim, never against a bar
//! moved after the fact. `replay` re-runs a recorded result from its
//! committed config and checks the claim still holds.
//!
//! Storage is one JSON record per line (the toydb WAL pattern, kept minimal):
//! records are never rewritten, results carry a version number that grows by
//! one per re-measurement, and recovery tolerates a torn final line, which is
//! what a crash mid-append leaves behind. Anything torn earlier than the tail
//! is corruption and is reported, not skipped.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{create_dir_all, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// One line in the WAL.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Record {
    Prereg(Prereg),
    Result(RunResult),
}

/// A claim committed before measuring. The params map holds the bench's
/// shape and iteration count (s, d, f, iters), all u64.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prereg {
    pub run_id: String,
    pub unix_ts: u64,
    pub bench: String,
    pub metric: String,
    pub claim: String,
    /// success means metric_value >= threshold ...
    pub threshold: f64,
    /// ... and max_abs_err < numerics_gate. Numerics gate before speed gate:
    /// a fast wrong kernel is worth nothing, so a result that fails numerics
    /// fails the claim no matter how fast it ran.
    pub numerics_gate: f64,
    pub seed: u64,
    pub params: BTreeMap<String, u64>,
}

/// One measurement of a preregistered run. Versions only grow; version 1 is
/// the original measurement, later versions are replays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub run_id: String,
    pub unix_ts: u64,
    pub version: u64,
    pub replay: bool,
    pub metric_value: f64,
    pub max_abs_err: f64,
    pub claim_met: bool,
    pub numbers: BTreeMap<String, f64>,
}

pub struct Ledger {
    path: PathBuf,
    records: Vec<Record>,
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl Ledger {
    /// Open (or create) the ledger at `path`, replaying every record into
    /// memory. A torn final line is dropped with a warning; a bad line
    /// anywhere else is corruption and errors out.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                create_dir_all(dir)?;
            }
        }
        let mut records = Vec::new();
        if path.exists() {
            let lines: Vec<String> = BufReader::new(std::fs::File::open(&path)?)
                .lines()
                .collect::<io::Result<_>>()?;
            let n = lines.len();
            for (i, line) in lines.into_iter().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Record>(&line) {
                    Ok(rec) => records.push(rec),
                    Err(e) if i + 1 == n => {
                        eprintln!("ledger: dropping torn final line: {e}");
                    }
                    Err(e) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("corrupt ledger record on line {}: {e}", i + 1),
                        ));
                    }
                }
            }
        }
        Ok(Self { path, records })
    }

    pub fn records(&self) -> &[Record] {
        &self.records
    }

    fn append(&mut self, rec: Record) -> io::Result<()> {
        let mut f = OpenOptions::new().create(true).append(true).open(&self.path)?;
        let line = serde_json::to_string(&rec).map_err(io::Error::other)?;
        writeln!(f, "{line}")?;
        f.sync_all()?;
        self.records.push(rec);
        Ok(())
    }

    /// Commit a claim before measuring it. Returns the record with its
    /// generated run id, `<bench>-s<seed>-<count>`.
    #[allow(clippy::too_many_arguments)]
    pub fn prereg(
        &mut self,
        bench: &str,
        metric: &str,
        claim: &str,
        threshold: f64,
        numerics_gate: f64,
        seed: u64,
        params: BTreeMap<String, u64>,
    ) -> io::Result<Prereg> {
        let count = self
            .records
            .iter()
            .filter(|r| matches!(r, Record::Prereg(_)))
            .count();
        let p = Prereg {
            run_id: format!("{bench}-s{seed}-{:03}", count + 1),
            unix_ts: now_unix(),
            bench: bench.to_string(),
            metric: metric.to_string(),
            claim: claim.to_string(),
            threshold,
            numerics_gate,
            seed,
            params,
        };
        self.append(Record::Prereg(p.clone()))?;
        Ok(p)
    }

    pub fn get_prereg(&self, run_id: &str) -> Option<&Prereg> {
        self.records.iter().find_map(|r| match r {
            Record::Prereg(p) if p.run_id == run_id => Some(p),
            _ => None,
        })
    }

    pub fn results(&self, run_id: &str) -> Vec<&RunResult> {
        self.records
            .iter()
            .filter_map(|r| match r {
                Record::Result(x) if x.run_id == run_id => Some(x),
                _ => None,
            })
            .collect()
    }

    pub fn latest_result(&self, run_id: &str) -> Option<&RunResult> {
        self.results(run_id).into_iter().max_by_key(|r| r.version)
    }

    /// Record one measurement of `run_id`. The claim verdict comes from the
    /// preregistered threshold and numerics gate, nothing else. The first
    /// recorded measurement is version 1; every later one is a replay.
    pub fn record_result(
        &mut self,
        run_id: &str,
        metric_value: f64,
        max_abs_err: f64,
        numbers: BTreeMap<String, f64>,
    ) -> io::Result<RunResult> {
        let prereg = self.get_prereg(run_id).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("no prereg for run id {run_id}"))
        })?;
        let numerics_ok = max_abs_err < prereg.numerics_gate;
        let claim_met = numerics_ok && metric_value >= prereg.threshold;
        let version = self.results(run_id).len() as u64 + 1;
        let r = RunResult {
            run_id: run_id.to_string(),
            unix_ts: now_unix(),
            version,
            replay: version > 1,
            metric_value,
            max_abs_err,
            claim_met,
            numbers,
        };
        self.append(Record::Result(r.clone()))?;
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("rl_ledger_test_{name}_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn params(s: u64, d: u64, f: u64) -> BTreeMap<String, u64> {
        BTreeMap::from([("s".into(), s), ("d".into(), d), ("f".into(), f), ("iters".into(), 5)])
    }

    #[test]
    fn prereg_then_result_survive_reopen() {
        let path = tmp("roundtrip");
        let run_id = {
            let mut led = Ledger::open(&path).unwrap();
            let p = led
                .prereg("mlp", "speedup", "fused beats naive", 1.1, 1e-5, 43, params(2048, 128, 1024))
                .unwrap();
            led.record_result(&p.run_id, 1.25, 0.0, BTreeMap::new()).unwrap();
            p.run_id
        };
        // a fresh open must replay the WAL into the same state
        let led = Ledger::open(&path).unwrap();
        let p = led.get_prereg(&run_id).expect("prereg survives reopen");
        assert_eq!(p.threshold, 1.1);
        let r = led.latest_result(&run_id).expect("result survives reopen");
        assert!(r.claim_met);
        assert_eq!(r.version, 1);
        assert!(!r.replay);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn replay_increments_version_and_is_marked() {
        let path = tmp("versions");
        let mut led = Ledger::open(&path).unwrap();
        let p = led.prereg("attention", "speedup", "c", 1.0, 1e-5, 42, params(2048, 64, 0)).unwrap();
        led.record_result(&p.run_id, 1.5, 1e-6, BTreeMap::new()).unwrap();
        let r2 = led.record_result(&p.run_id, 1.4, 1e-6, BTreeMap::new()).unwrap();
        assert_eq!(r2.version, 2);
        assert!(r2.replay);
        assert_eq!(led.results(&p.run_id).len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn numerics_gate_comes_before_speed_gate() {
        // a fast wrong kernel must fail the claim even when the speedup clears
        // the threshold.
        let path = tmp("gate");
        let mut led = Ledger::open(&path).unwrap();
        let p = led.prereg("mlp", "speedup", "c", 1.1, 1e-5, 7, params(64, 16, 64)).unwrap();
        let r = led.record_result(&p.run_id, 99.0, 1e-3, BTreeMap::new()).unwrap();
        assert!(!r.claim_met, "claim must fail when numerics fail");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn torn_final_line_is_dropped_and_log_still_opens() {
        let path = tmp("torn");
        let mut led = Ledger::open(&path).unwrap();
        let p = led.prereg("mlp", "speedup", "c", 1.0, 1e-5, 1, params(64, 16, 64)).unwrap();
        // simulate a crash mid-append: half a JSON record at the tail
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            write!(f, "{{\"kind\":\"result\",\"run_id\":\"{}\"", p.run_id).unwrap();
        }
        let led = Ledger::open(&path).expect("torn tail must not block recovery");
        assert!(led.get_prereg(&p.run_id).is_some());
        assert_eq!(led.results(&p.run_id).len(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn result_without_prereg_is_refused() {
        // results may only exist against a committed claim, that is the point
        let path = tmp("noprereg");
        let mut led = Ledger::open(&path).unwrap();
        assert!(led.record_result("ghost-run", 2.0, 0.0, BTreeMap::new()).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
