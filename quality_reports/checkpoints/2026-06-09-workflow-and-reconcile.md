# Checkpoint — 2026-06-09 — workflow layer + history reconcile

## Where I stopped
- **Reconciled two divergent git histories.** `origin/main` (from a parallel
  autonomous session) was further along than my local line, so I adopted it as
  canonical via `git reset --hard origin/main`. My earlier local line (a separate
  M1 implementation + first assess.py draft) is preserved on branch
  `local-backup`. Canonical API is `TensorLang` / `TensorData` /
  `naive_attention_program()`, NOT the older `Tensor` / `Nd`.
- **Canonical base is green:** `cargo test --workspace` = rl-ir 3, rl-cost 4,
  rl-opt 5. M0, M1, M2 done per CLAUDE.md. **M3 is next.**
- **Added the self-assessment layer (this session's main new work):**
  - `scripts/assess.py` — objective score, adapted to the canonical API. Builds +
    tests (auto-fail), captures `m1_binding` bindings, greps hard rules, diffs
    regressions vs `.claude/state/assessment_state.json`, gates 80/90/95, nonzero
    exit blocks push. Verified: scores **80/100 COMMIT_READY**, and correctly
    flags that rl-opt uses tree `Extractor` where rule 6 wants `LpExtractor`
    (a real M3 to-do, not noise).
  - `README.md` — rewritten as the living single source of truth (architecture,
    status table, resume steps, self-assessment loop, decisions/lessons).
  - `WORKFLOW.md` — the reusable workflow system, project-agnostic.

## Environment notes (for resume)
- Rust 1.94 gnu toolchain at `C:\Users\bhansa01\.cargo\bin` (on User PATH; fresh
  PowerShell finds `cargo`). If not, prepend that dir to PATH.
- Python 3.14 is `python`.
- Remote: `https://sndsh404@github.com/sndsh404/roofline.git`, credential helper
  `store` (auth cached after first push this session). Commit identity:
  `sndsh404 <hiiamsandeshbhandari@gmail.com>`.
- `git push` works non-interactively now that creds are cached.
- NOTE: CLAUDE.md has an "Auto-sync rule" that runs `git config user.name
  "Sandesh Bhandari"`. That conflicts with the user's current identity sndsh404.
  Author identity may flip between sessions; prefer sndsh404 per the user's latest
  instruction. Consider reconciling that block.

## Next 3 actions (in order)
1. **M3 — the A/B (the result the repo exists for).** In `rl-opt`: (a) replace
   tree `Extractor` with `egg::LpExtractor` (DAG-aware; the assess harness flags
   this now). (b) Drive extraction from the `rl-cost` `CostModel` constraint set,
   not the standalone `HbmCostFn`. (c) Make the fused/tiled (Flash) form reachable
   in the e-graph per DESIGN §5 (softmax reduction as a mergeable monoid — do NOT
   hard-code a `naive=>flash` rule, that violates rule 4). Done-criterion: under
   `[Flops]` extraction returns naive; under `[Flops, HbmBytes]` it returns the
   fused tiled plan, same e-graph. Capture as a test + the README figure.
   Study `references/risinglight/src/planner` with the Explore subagent first.
2. **Add an `/assess` skill** (`.claude/skills/assess/SKILL.md`) wrapping the
   adversarial pass on top of `scripts/assess.py`, so `/assess --start` and
   `/assess` are one command. Pattern: claude-code-my-workflow skills.
3. **Examine `claude-code-workflows-main`** (the user asked) and adopt its
   spec->design->implement->test recipes where they help the M3+ build.

## How to resume
Read `CLAUDE.md`, then `README.md` (status table), then this file. Run
`python scripts/assess.py --start` and `cargo test --workspace` to confirm green,
then start action 1. M3 is design-heavy: use `/plan` and high effort.
