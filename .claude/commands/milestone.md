---
description: Work a Roofline v0 milestone end-to-end under the project's benchmark-gated, preregistered discipline. Argument is the milestone number (0-5).
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
---

You are working **milestone M$ARGUMENTS** of the Roofline project. Follow this
loop exactly. Do not skip the gates.

1. **Read the spec.** Open `DESIGN.md`, find the M$ARGUMENTS entry in §9, and
   restate its single numeric done-criterion in one sentence. If you can't state
   a number that ends this milestone, stop and ask.

2. **Study the reference.** Identify which repo under `references/` is most
   relevant (M2/M3 → `risinglight/src/planner` for the egg + cost.rs pattern;
   M0/M1 substrate → `type-exercise-in-rust`; ledger → `toydb` mvcc + `bustub`
   recovery). Delegate the reading to the Explore subagent so the main context
   stays clean. Summarize the 3–5 concrete patterns you'll reuse.

3. **Plan.** Enter plan mode (`/plan`). Produce a short plan: files to
   create/change, the public interface, and the test + bench you'll add. Do not
   write implementation code yet. Wait for my approval.

4. **Preregister.** Before any benchmark that produces a claim, run
   `roofline prereg --bench <b> --metric <m> --claim "<c>" --seed <n>` and paste
   the run_id into the plan. The claim must be the DESIGN §9 done-criterion.

5. **Implement** on a branch `m$ARGUMENTS/<short-slug>`. Honor the CLAUDE.md hard
   rules — especially: cost-model failures are missing constraints not search
   bugs; no canned naive→flash rule; shape suffixes; LpExtractor not tree cost.

6. **Gate 1 — numerics.** `cargo test --workspace`. Any lowered kernel must match
   the reference interpreter to 1e-5 across the shape sweep. If it fails, the
   speed number does not count. Fix before proceeding.

7. **Gate 2 — wall clock.** Run the milestone's bench. Record flops, measured
   hbm_bytes, the cost model's prediction, and the gap (measured/predicted) into
   the ledger. The gap is the next research question, not a failure.

8. **Report.** State the number, whether it meets the preregistered claim, and
   tick the box in CLAUDE.md's milestone tracker. If the claim was missed, write
   a one-line decision record explaining why and what constraint or rule is
   implicated — never quietly move the bar.

Treat M0 as load-bearing menial work: the reference interpreter is what makes
every later cost-model and kernel claim honest. Build it carefully.
