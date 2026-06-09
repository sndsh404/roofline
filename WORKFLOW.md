# The Workflow

A reusable system for building a project across many Claude Code sessions without
losing state, drifting, or shipping unverified claims. It was built for Roofline
but nothing here is Roofline-specific. Copy this file into any project and follow
it from day one.

---

## 1. Why this exists and what problem it solves

A coding agent works in sessions with a finite token budget. Three failure modes
follow from that:

1. **Lost state.** A session ends and the next one re-derives context from
   scratch, repeats work, or contradicts earlier decisions.
2. **Self-congratulation.** An agent asked to assess its own work tends to report
   progress whether or not the code got better.
3. **Unverified claims.** "It works" and "it's faster" get asserted without a
   build, a test, or a number behind them.

This workflow fixes each one with a concrete mechanism: a living document plus
checkpoints for state, an objective scoring script for assessment, and hard gates
(build, tests, numerics) that block a commit or push until they pass.

---

## 2. The four documents

- **`CLAUDE.md`** — the per-session operating guide. Loaded automatically every
  session. Holds the prime directive, hard rules, repo map, build commands, and
  the milestone checklist. Keep it tight; it is read every time.
- **`README.md`** — the living single source of truth. What the project is, the
  architecture, current status, how to resume, decisions and lessons. Updated in
  the same commit as the change it describes.
- **`DESIGN.md`** — the full spec. The deep technical content and milestone
  done-criteria. Changes rarely.
- **`WORKFLOW.md`** — this file. How the work is done, independent of what is
  being built.

Plus two living folders:

- **`quality_reports/checkpoints/`** — one file per stopping point.
- **`.claude/state/`** — machine state for the assessment script (gitignored).

---

## 3. The self-assessment loop, step by step

The loop runs at the start and end of every session and around any meaningful
change.

1. **Start.** Run `python scripts/assess.py --start`. It prints the last score,
   the captured numbers, and the latest checkpoint to read. Read `CLAUDE.md`,
   `README.md`, and that checkpoint.
2. **Work.** Make one focused change toward the current milestone.
3. **Assess.** Run `python scripts/assess.py`. The score is computed by the
   script from machine signals, never by the model. It:
   - builds and tests; a failure is an auto-fail (score 0).
   - captures headline numbers and compares them to the last run; a number moving
     the wrong way is a regression.
   - greps for hard-rule violations specific to the project.
   - writes `quality_reports/assessment_latest.md` and updates
     `.claude/state/assessment_state.json`.
4. **Adversarial pass.** Read the report and ask what is weakest. The model may
   never claim progress the script did not verify. A session that lowers the score
   by finding a real bug is worth more than one that holds it flat by not looking.
5. **Gate.** `python scripts/assess.py && git push`. A nonzero exit (failed
   build, failed tests, regression) physically blocks the push.

The principle: the agent proposes, the script disposes. Objective gates are what
keep "self-improvement" from becoming "self-congratulation."

---

## 4. Context management and working within token limits

> Monitor token usage continuously. Before hitting the limit, stop new work, run
> the full test suite, commit everything, push, and write a checkpoint. Always
> stop clean. Never get cut off mid-change with uncommitted work.

Practical rules:

- Do not start a large, multi-step change when the budget is low. Start it next
  session from a clean checkpoint instead.
- Prefer reading the specific part of a file you need over re-reading whole files.
- Build and test cost tokens; run them at meaningful points, not after every line.
- When you notice the budget getting low, switch into wrap-up mode immediately:
  test, commit, push, checkpoint, stop.

---

## 5. When to commit, and how to write a checkpoint

**Commit and push** after every meaningful step or every ~30 minutes, whichever
comes first. Short, plain commit messages, lowercase, no dashes. Examples: `add
lp extractor`, `fix attention test`, `update readme status`. Run the assess
script before pushing so red code cannot land.

**Write a checkpoint** whenever you stop, and always before running low on
budget. Path: `quality_reports/checkpoints/<YYYY-MM-DD>-<topic>.md`. It must
contain:

- **Where I stopped.** What is done, what is half-done, what is verified vs
  assumed. Include the actual numbers (test counts, error magnitudes).
- **Environment notes** that the next session needs (toolchain paths, remote URL,
  auth state, anything non-obvious).
- **Next 3 actions, in order.** Specific enough to start without thinking.
- **How to resume.** The exact files to read and commands to run.

The test of a good checkpoint: a fresh session reading only `CLAUDE.md` and that
checkpoint can resume perfectly.

---

## 6. Setting this up in a new project from day one

1. `git init`. Make sure the project is its own repository, not nested
   accidentally inside a larger one. Check `git rev-parse --show-toplevel`.
2. Create `.gitignore` for build artifacts, `.claude/state/`, and any large
   read-only reference material.
3. Write `CLAUDE.md`: prime directive, hard rules, repo map, build/test commands,
   milestone checklist with numeric done-criteria.
4. Write `DESIGN.md`: the full spec and per-milestone done-criteria.
5. Copy this `WORKFLOW.md` in unchanged.
6. Add `scripts/assess.py`: build + test as auto-fail, project-specific hard-rule
   greps, regression diff against `.claude/state/`, gates at 80/90/95, nonzero
   exit on failure or regression.
7. Set the git remote and confirm a push works (auth cached) before relying on it.
8. Make the first commit and push. Write the first checkpoint.

---

## 7. Commit discipline

- One logical change per commit where practical.
- Never commit with a failing build or failing tests.
- Update `README.md` status and `CLAUDE.md` checklist in the same commit as the
  change that moves them.
- Before pushing, `git fetch` and inspect `origin/main`. If the remote has commits
  you do not have, reconcile; never force-push over work you did not create. (This
  workflow learned that lesson the hard way when two autonomous sessions pushed
  divergent histories to the same repo.)

---

## 8. Using a workflow repo as a base

If you have a reference workflow repo (this one was derived from a research
workflow that used preregistration and quality gates), mine it for:

- a scoring script with severity-weighted deductions and explicit gate thresholds,
- preregistration and decision-record templates,
- a session-log/checkpoint format.

Port the mechanism, not the domain specifics. The research repo scored LaTeX and R;
this one scores a Rust workspace. The shared core is: objective script, hard gates,
durable written state.

---

## 9. Lessons learned and best practices

- **Objective beats subjective.** The single highest-value piece is the scoring
  script. Without it, "is it better?" has no honest answer.
- **Stop clean, always.** The cost of being cut off mid-change is a confused next
  session. The cost of stopping clean is one checkpoint file. Pay the cheap one.
- **Verify before claiming.** Build, test, measure. "Faster" without a number is
  noise.
- **Honest partial beats fake complete.** If a milestone needs hardware you do not
  have, say so and record the gap. Do not mark it done.
- **Preserve before you overwrite.** Branch (`local-backup`), inspect, then act.
- **One milestone at a time, each ending in a number, not a refactor.**

---

## 10. Quick start checklist for a new project

- [ ] `git init`; confirm it is its own repo.
- [ ] `.gitignore` for artifacts, state, references.
- [ ] `CLAUDE.md` with prime directive, hard rules, milestone checklist.
- [ ] `DESIGN.md` with numeric done-criteria per milestone.
- [ ] `WORKFLOW.md` copied in.
- [ ] `scripts/assess.py` building, testing, gating.
- [ ] git remote set; one successful push.
- [ ] first commit + first checkpoint.
- [ ] from then on: assess at start, work one milestone, assess + commit + push,
      checkpoint before stopping.
