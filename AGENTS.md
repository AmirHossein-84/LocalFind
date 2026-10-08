# LocalFind — AI Coding Agent Operating Manual

**Version:** 0.1.0 · **Last updated:** 2026-10-08  
**Audience:** Muse Spark 1.3, Opus reviewers, and any future code-writing agent.  
**Current starting task:** `P0-T006` in [STEPS.md](STEPS.md). **No implementation is asserted to exist.**

> You are working on a privacy-sensitive, Windows-first local AI search application. Your goal is to make a small, correct, verifiable contribution—not to maximize changes or impress the user with features.

## 1. Roles and authority

| Role | May do | Must NOT do without approval |
|---|---|---|
| **Implementation agent (Muse Spark or equivalent)** | Complete the active small task, choose local implementation details, test, update task state | Switch frameworks, models, DB/vector engine, process model, index formats or privacy policy |
| **Architecture reviewer (Opus or equivalent)** | Resolve Architecture Review Queue, review milestone gates, accept/supersede ADRs, revise architecture and tasks | Claim execution results without evidence, relax safety rules silently |
| **Project owner (human)** | Approve scope, product promises, hardware targets, release/licensing decisions | Not applicable |

If instructions conflict: prioritize user-data safety and user instruction, then `RULES.md`, accepted ADRs in `PLAN.md`, task acceptance in `STEPS.md`. If unclear, write an Architecture Review Queue entry and stop that risky work.

## 2. Required session startup (always)

1. Read **this file** in full.
2. Read `RULES.md` in full.
3. Read `STEPS.md` **Project State**, active phase, dependencies, and active task.
4. Read relevant sections of `PLAN.md`; for an architectural task, also read ADRs, relevant risks and unresolved questions.
5. Inspect existing code, manifests, migrations, tests and Git status in the affected area. **A planning task must inspect existing planning docs first.**
6. Print a concise session contract before editing:

```text
Task: Pn-Tnnn — <title>
Preconditions / dependencies: <verified/not verified>
Files expected to change: <paths>
Architecture constraints: <R-xx / ADR-xxx>
Tests / acceptance: <named commands and expected behavior>
Risks or blocker: <if any>
```

If the active task from Project State is completed or inconsistent, examine checkboxes/dependencies and **report** rather than silently choose another. When a task is blocked, make the least disruptive progress permitted by its dependencies and request clarification/approval for architecture impacts.

## 3. Scope: one task, one verified result

- Do **not** implement all of LocalFind, an entire phase, or multiple unrelated tasks in one coding session.
- Read implementation instructions, expected files, edge cases and `Do not` constraints for this task.
- Write only code and tests required for that task. New discoveries are appended as a future task / blocker; do not expand scope.
- Inspect existing classes/ports before adding new abstractions. Reuse existing documented APIs.
- Keep changes reviewable. Large unrelated refactors and dependency updates are forbidden.
- A task is finished only when its explicit tests, correctness and documentation obligations are satisfied; elapsed effort is not evidence of completion.

## 4. Implementation procedure

**A. Pre-change inventory**
- `git status --short` and inspect affected files. Do not overwrite uncommitted human/agent changes.
- Check lockfiles, version constraints and relevant migration history.
- Determine whether the current task is a **benchmark/prototype** (`benchmarks/`), **production code**, or **documentation**; avoid crossing those boundaries.
- For tests using local files, work only in `tests/fixtures` or temporary directories under OS temp and explicit test root, never in real user libraries.

**B. Short implementation plan**
- Identify 2–5 sub-actions; outline data type/interface touched and failure behavior.
- Clarify important assumptions in a note or doc; do not pretend tests or model capabilities exist.
- Only introduce dependencies after the R-21 assessment.

**C. Implement**
- Honor typed contracts; domain must remain free of UI/storage dependencies.
- Errors are typed and surfaced; process/media/model operations bounded and cancelable where task scope requires.
- No direct network requests from inference/search; model acquisition needs explicit opt-in.
- Protect app-owned paths: never apply cleanup/deletion to source directories.
- For EmbeddingGemma 2: correct task prefix, bf16 or fp32 (not fp16), finite/dimension checks, L2 normalize truncated embeddings, persist compatible model/preprocessing signature.
- Do not code beyond testable feature slices merely because the vision is broad.

**D. Self-review**
- Scan the diff for scope creep, hidden global state, blocking UI, personal paths/content in logs, new network use, unsafe subprocess arguments, and compatibility regressions.
- Validate that imports respect component boundaries and that a model/worker failure preserves exact/FTS search.

## 5. Tests and honest reporting

Run only commands that are supported by the current repo; report exact commands and actual results. Possible commands **after scaffolding**, not claims that they already work:

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
# Frontend, in its configured package directory:
pnpm run check
pnpm run build
# Python worker, with its pinned environment:
python -m pytest workers/inference/tests
```

Adapt package manager/tooling to the actual accepted stack. Model benchmarks are **not unit tests**: record hardware, runtime, corpus manifest, model revision, precision, dimensions, sampling policy, measured p50/p95 and retrieval metrics.

Outcome terms:
- **PASS:** executed and checked, with summarized output.
- **FAIL:** executed and failed; include failure and remediation.
- **NOT RUN:** environment unavailable, tool failure or out-of-scope; state why. Never write PASS for a command not actually executed.
- **BLOCKED:** task acceptance cannot be met without missing dependency, critical clarification or architecture decision.

If tests fail: inspect and fix errors caused by the task; never delete/disable/weaken failing tests. If failure predates the task, capture evidence and either isolate with approval or report blocked. Keep tests for source file immutability, stale revision invalidation, worker crash and lexical fallback when changing core flows.

## 6. How to update the four living files

**At task completion (normally):**
1. Mark its `- [ ]` in `STEPS.md` to `- [x]` **only when all acceptance conditions actually pass**.
2. Update `Project State`: `Last Completed Task`, `Current Task` (first ready next task), `Current Phase/Milestone` if passed, and blockers. Preserve stable task IDs.
3. Update `PLAN.md` **only** if architecture, accepted ADR, risk/assumption, measured budget or product scope changes. Add concise benchmark links for M0 decisions.
4. Update `RULES.md` **only** for permanent, enforceable constraints approved by reviewer.
5. Update `AGENTS.md` **only** if the workflow or repository tooling/process changes.

Do not rewrite entire docs merely to mark progress. Record requested changes separately from established decisions. If a milestone is reached, request an architecture review rather than silently accepting major ADR changes.

### Task status and evidence policy

A test run without functioning feature is not complete. A prototype does not count as production feature. A passed test on Linux does not satisfy explicitly Windows-only verification. A review of a URL does not satisfy runtime feasibility. A clean compile does not satisfy a retrieval quality gate.

## 7. Architecture escalation (Opus review)

If any of these arise, pause the affected architectural work and create an entry in `STEPS.md` **Architecture Review Queue**:
- EmbeddingGemma 2 cannot load on target hardware, returns incompatible vectors or violates quality/memory targets.
- Need to change inference runtime/process protocol, GPU strategy, file identity or indexing schema.
- Need to replace vector engine, database, scheduler architecture or desktop framework.
- Unexpected source-file write/deletion risk, privacy leak, model license ambiguity or unsafe default network dependency.
- Required task is impossible without changing an accepted ADR or violating a repository rule.

Use exactly this concise template:

```text
ARQ-<next ID> | status: open | task: Pn-Tnnn
Problem: <specific mismatch>
Evidence: <commands, test outputs, links, measured numbers; no private data>
Current contract: <ADR/PLAN section/RULE>
Options: A ... (risk/cost), B ... (risk/cost)
Recommendation: <one option + why>
Impact: <migration, test, performance, privacy, schedule>
Blocking: yes/no
Reviewer decision + date: pending
```

For important design reviews, Opus should: inspect affected code and docs, challenge assumptions, compare trade-offs, select or propose an ADR change, add explicit follow-on tasks/migrations/tests, and update dependencies. A reviewer must **not** merely say “looks good” without concrete verification.

## 8. Practical handling of common blockers

| Situation | Required response |
|---|---|
| Existing file changed by user/another agent | Inspect diff; preserve it; do not overwrite; propose minimal merge or stop if conflict |
| File/tool/API operation failed | Report exact operation; inspect existing artifacts; retry once if safe; don't invent file creation or tool success |
| Tool history error (`tool result has no matching tool call`) | Treat session state as unreliable; don't blindly `continue` or replay destructive operations; start a clean session and inspect actual filesystem before resuming |
| No model or unsupported GPU | Record as Phase 0 or runtime blocker; lexical search remains available; do not introduce fake semantic results |
| Cannot run Windows test from current host | Mark NOT RUN and block Windows-specific acceptance; do not translate Linux success into Windows success |
| Test corpus inaccessible or private | Require consent/fixtures or keep verification blocked; do not transmit private assets for help |
| Source file changed during indexing | Discard staged result; re-enqueue latest revision; do not mark ready |
| Ambiguous dependency or incompatible migration | Stop dangerous change and create ARQ; no ad hoc schema edits |
| Performance below budget | Profile stages with stable corpus and hardware before architecture replacement |

## 9. Definition of Done / review checklist

- [ ] Task ID, dependencies and boundaries match accepted docs.
- [ ] Required behavior exists in repository; no placeholders masquerading as final code.
- [ ] Input/error/timeout/concurrency paths handled within task scope.
- [ ] Tests and checks actually pass, or task remains unchecked/blocked.
- [ ] No original user data modified and no unauthorized file/network access.
- [ ] Database/schema/index signatures/migration behavior validated if touched.
- [ ] No new high-severity privacy, crash or data corruption risks.
- [ ] Diff is focused; new dependency license/size implications assessed.
- [ ] Changes documented, task/status updated, and the next task identified.

## 10. Mandatory completion report

After each task, report briefly and factually:

```text
Task: Pn-Tnnn — <title>
Status: PASS / BLOCKED / FAIL
Changed files: <paths>
What changed: <3–6 bullet summary>
Architecture impact: none / <ADR, interface, migration>
Verification: <commands + PASS/FAIL/NOT RUN, results>
Privacy/source-data checks: <specific evidence>
Documentation updated: <which files>
Known limitations / new risks: <items or none>
Next ready task: <ID, if dependencies met>
```

When producing only documentation, say that no application code exists or was executed; do not claim an app build/test pass.

## 11. Milestone review process

After `M0/M1/M2/M3/M4` exit work, reserve an Opus review session. The review must compare accepted ADRs, actual file layout, code/test evidence, labeled retrieval results, hardware/performance data and unresolved risks. Record outcome as `APPROVED`, `CHANGES REQUIRED`, or `BLOCKED` with evidence and task IDs. Update `PLAN.md`/`STEPS.md` only as needed. **A milestone is not finished merely because its checkboxes were ticked by an agent.**

## 12. Git workflow (mandatory)

At the end of each successfully completed implementation task:

- Review **all** changes first (`git status --short`, staged diff). Preserve
  unrelated user/agent changes; never sweep them into your commit.
- Stage only files relevant to the completed task with explicit paths
  (`git add <paths>`), never blind `git add -A`.
- Never commit model weights, caches, databases, virtual environments,
  personal datasets, secrets, or generated build outputs (R-24/R-26; the
  root `.gitignore` already excludes `/models/`, `.venv/`, corpora, DBs).
- Run the task's relevant verification **before** committing; update
  `STEPS.md` (checkbox + Project State) and required docs first.
- One focused commit per completed task with a descriptive message
  referencing the task ID, e.g.
  `git commit -m "feat(P0-T005): validate video segment embeddings"`.
  No commit per file edit, no intermediate-state commits, and no success
  commit when acceptance criteria are unmet.
- Never amend, rebase, reset, or force-push without explicit authorization.
  Never push to a remote unless explicitly requested.
- Report the commit hash and final `git status` in the completion report.

Earlier tasks may be uncommitted: inspect state first and, when needed, create
clearly scoped per-task commits for verified previous work, stating exactly
what each commit includes. This policy implements RULES.md R-25 (scoped
changes, verify before staging) and R-26 (no secrets/weights in Git).

## 13. Forbidden behavior summary

Never: write/delete original user media; silently index outside selected roots; upload user data; run model inference on the UI thread; turn failed API/tool operations into fake success; mix incompatible embedding versions; disable tests to pass; hardcode personal paths; log confidential content; introduce a server or framework casually; duplicate existing storage/inference layers; bypass review to change schema or core architecture; mark work complete without required validation.

**Immediate next step:** Follow the current task in `STEPS.md` Project State. Inspect the working directory before making any change; these planning files are initial specifications, not proof of an implemented project.
