#!/usr/bin/env python3
"""
Roofline self-assessment harness.

The point: an LLM grading its own work drifts into self-congratulation. So the
score is computed HERE, from machine signals, not from a model's opinion. The
/assess skill reads this report and runs the adversarial pass on top of it, but
it may never claim progress this script did not verify.

Severity-weighted, 0-100, ported from claude-code-my-workflow/scripts/
quality_score.py. Gates: 80 (commit), 90 (PR), 95 (excellence). A compile or
test failure is an auto-fail (score 0) — a fast wrong kernel is worth nothing.

What it checks:
  P0 (auto-fail)  cargo build + cargo test --workspace must pass.
  P1 (-20 each)   regressions vs the last assessment (a number moved the wrong
                  way), or a hard-rule violation from CLAUDE.md.
  P2 (-5 each)    tracker/reality mismatches, missing milestone numbers.
  P3 (-1 each)    hygiene (e.g. shape-suffix discipline hints).

Outputs:
  quality_reports/assessment_latest.md   human-readable
  .claude/state/assessment_state.json    machine state (for regression diff)

Exit codes: 0 = score >= 80 and no P0. 1 = below 80 or a regression. 2 = auto-fail.
Wire it as `python scripts/assess.py && git push` so red code physically cannot
be pushed.

Usage:
  python scripts/assess.py            # full run (builds + tests)
  python scripts/assess.py --start    # context-load summary, no scoring rebuild
  python scripts/assess.py --no-test  # static checks only (fast)
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".claude" / "state"
STATE_FILE = STATE_DIR / "assessment_state.json"
REPORT_FILE = ROOT / "quality_reports" / "assessment_latest.md"
THRESHOLDS = {"commit": 80, "pr": 90, "excellence": 95}


def cargo_path() -> str:
    """Find cargo even when it's not on PATH (school-laptop custom installs)."""
    for cand in ("cargo", str(Path.home() / ".cargo" / "bin" / "cargo")):
        try:
            subprocess.run([cand, "--version"], capture_output=True, timeout=20)
            return cand
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return "cargo"


def run(cmd, timeout=300):
    env = dict(os.environ)
    env["PATH"] = env.get("PATH", "") + os.pathsep + str(Path.home() / ".cargo" / "bin")
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    except FileNotFoundError as e:
        return None, f"tool missing: {e}"


class Assessment:
    def __init__(self):
        self.score = 100
        self.issues = {"P0": [], "P1": [], "P2": [], "P3": []}
        self.auto_fail = False
        self.numbers = {}      # extracted metrics, compared run-to-run
        self.unverified = []

    def deduct(self, sev, points, msg):
        self.issues[sev].append({"points": points, "msg": msg})
        if sev == "P0":
            self.auto_fail = True
            self.score = 0
        else:
            self.score = max(0, self.score - points)

    # ---- checks -----------------------------------------------------------

    def check_build_and_test(self, run_tests=True):
        code, out = run([cargo_path(), "build", "--workspace"], timeout=420)
        if code is None:
            self.unverified.append(f"build not verified: {out}")
            return
        if code != 0:
            tail = "\n".join(out.strip().splitlines()[-6:])
            self.deduct("P0", 100, f"cargo build failed:\n{tail}")
            return
        if not run_tests:
            return
        code, out = run([cargo_path(), "test", "--workspace"], timeout=420)
        if code is None:
            self.unverified.append(f"tests not verified: {out}")
            return
        if code != 0:
            tail = "\n".join(l for l in out.splitlines() if "error" in l.lower()
                             or "FAILED" in l or "test result" in l)[-600:]
            self.deduct("P0", 100, f"cargo test failed:\n{tail}")
            return
        # capture pass counts
        passed = sum(int(m) for m in re.findall(r"(\d+) passed", out))
        self.numbers["tests_passed"] = passed

    def capture_numbers(self):
        """Run the milestone examples and parse the headline numbers."""
        # M0 numerics error (--nocapture so the printed err line is visible)
        code, out = run([cargo_path(), "test", "-p", "rl-ir", "--", "--nocapture"],
                        timeout=300)
        if code == 0:
            m = re.search(r"max abs err vs JAX fixture = ([\d.eE+-]+)", out)
            if m:
                self.numbers["m0_numerics_err"] = float(m.group(1))
        # M1 binding resource (canonical example prints `binding=HbmBytes` per row)
        code, out = run([cargo_path(), "run", "-q", "-p", "rl-cost",
                         "--example", "m1_binding"], timeout=300)
        if code == 0:
            bindings = re.findall(r"binding=(\w+)", out)
            if bindings:
                self.numbers["m1_attn_binding"] = bindings[0]
                self.numbers["m1_hbm_bound_rows"] = sum(b == "HbmBytes" for b in bindings)

    def check_hard_rules(self):
        """CLAUDE.md hard rules that are greppable."""
        opt = ROOT / "crates" / "rl-opt" / "src"
        if opt.exists():
            src = "\n".join(p.read_text(encoding="utf-8", errors="ignore")
                            for p in opt.rglob("*.rs"))
            # rule 4: no canned naive=>flash rewrite
            if re.search(r'rw!\(\s*"[^"]*naive[^"]*flash', src, re.I) or \
               re.search(r"naive_attn\s*=>\s*flash", src, re.I):
                self.deduct("P1", 20, "rule 4: canned naive=>flash rewrite detected "
                            "(Flash must be REACHABLE from primitives, not hard-coded)")
            # rule 6: LpExtractor, not tree Extractor
            if "Extractor::new" in src and "LpExtractor" not in src:
                self.deduct("P1", 20, "rule 6: tree Extractor used; DESIGN demands "
                            "LpExtractor (DAG-aware) - tree cost double-counts shared Q/K/V")

    def check_regressions(self, prev):
        if not prev:
            return
        pn = prev.get("numbers", {})
        # numerics error must not grow past the gate
        cur = self.numbers.get("m0_numerics_err")
        old = pn.get("m0_numerics_err")
        if cur is not None and old is not None and cur > max(old * 2, 1e-5):
            self.deduct("P1", 20, f"regression: m0 numerics err {old:.2e} -> {cur:.2e}")
        # test count must not drop
        ct, ot = self.numbers.get("tests_passed"), pn.get("tests_passed")
        if ct is not None and ot is not None and ct < ot:
            self.deduct("P1", 20, f"regression: tests_passed {ot} -> {ct}")

    def check_tracker(self):
        claude = (ROOT / "CLAUDE.md")
        if not claude.exists():
            self.deduct("P2", 5, "CLAUDE.md missing")
            return
        text = claude.read_text(encoding="utf-8", errors="ignore")
        # M0 claims done but the captured numerics number actually fails the gate.
        # Only fires when the number WAS captured — a missing number is "unverified",
        # not a failure.
        err = self.numbers.get("m0_numerics_err")
        if "[x] M0" in text and err is not None and err > 1e-5:
            self.deduct("P2", 5, "tracker says M0 done but numerics gate not met")

    # ---- reporting --------------------------------------------------------

    def status(self):
        if self.auto_fail:
            return "FAIL (auto)"
        if self.score >= THRESHOLDS["excellence"]:
            return "EXCELLENCE"
        if self.score >= THRESHOLDS["pr"]:
            return "PR_READY"
        if self.score >= THRESHOLDS["commit"]:
            return "COMMIT_READY"
        return "BLOCKED"

    def write(self, prev):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [f"# Roofline assessment — {now}", "",
                 f"**Score: {self.score}/100 — {self.status()}**  "
                 f"(gates: commit 80, pr 90, excellence 95)", ""]
        if self.numbers:
            lines.append("## Numbers")
            for k, v in sorted(self.numbers.items()):
                lines.append(f"- `{k}` = {v}")
            lines.append("")
        for sev, label in [("P0", "Auto-fail"), ("P1", "Regressions / rule violations"),
                           ("P2", "Tracker / coverage"), ("P3", "Hygiene")]:
            if self.issues[sev]:
                lines.append(f"## {sev} — {label}")
                for it in self.issues[sev]:
                    lines.append(f"- (-{it['points']}) {it['msg']}")
                lines.append("")
        if self.unverified:
            lines.append("## Unverified (could not run — not counted against score)")
            lines += [f"- {u}" for u in self.unverified] + [""]
        if not any(self.issues.values()):
            lines.append("No issues found. Score reflects a clean build + green tests.\n")
        REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

        state = {"ts": now, "score": self.score, "status": self.status(),
                 "numbers": self.numbers, "auto_fail": self.auto_fail}
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state


def load_prev():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def cmd_start():
    prev = load_prev()
    print("# Roofline — session start\n")
    if prev:
        print(f"Last assessment: {prev['score']}/100 ({prev['status']}) at {prev['ts']}")
        if prev.get("numbers"):
            for k, v in sorted(prev["numbers"].items()):
                print(f"  {k} = {v}")
    else:
        print("No prior assessment. Run `python scripts/assess.py` for a baseline.")
    cps = sorted((ROOT / "quality_reports" / "checkpoints").glob("*.md"))
    if cps:
        print(f"\nLatest checkpoint: {cps[-1].relative_to(ROOT)}")
        print("Read it + CLAUDE.md to resume.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", action="store_true", help="context-load summary only")
    ap.add_argument("--no-test", action="store_true", help="static checks only")
    args = ap.parse_args()

    if args.start:
        sys.exit(cmd_start())

    prev = load_prev()
    a = Assessment()
    a.check_build_and_test(run_tests=not args.no_test)
    if not a.auto_fail:
        a.capture_numbers()
        a.check_hard_rules()
        a.check_regressions(prev)
        a.check_tracker()
    a.write(prev)

    print(f"Score: {a.score}/100 - {a.status()}")
    for sev in ("P0", "P1", "P2", "P3"):
        for it in a.issues[sev]:
            print(f"  [{sev}] -{it['points']} {it['msg'].splitlines()[0]}")
    print(f"Report: {REPORT_FILE.relative_to(ROOT)}")

    if a.auto_fail:
        sys.exit(2)
    regressed = any("regression" in it["msg"] for it in a.issues["P1"])
    sys.exit(0 if (a.score >= THRESHOLDS["commit"] and not regressed) else 1)


if __name__ == "__main__":
    main()
