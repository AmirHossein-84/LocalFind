# LocalFind — Repository Rules

**Version:** 0.1.0 · **Last reviewed:** 2026-10-08 · **Applies to:** every developer and coding agent  
**Priority:** user safety and explicit instructions > these rules > accepted ADRs in [PLAN.md](PLAN.md) > current task in [STEPS.md](STEPS.md). If they conflict, **stop and request a documented architecture review**; do not silently choose.

These are enforceable constraints, not motivational guidelines. Exceptions require an explicit, recorded review and tests appropriate to the risk.

## 1. Protect user data

**R-01.** Source files are **read-only**. No indexing, thumbnail, cleanup, migration or test path may modify, move, rename or delete originals. Use dedicated app-owned cache directories. Validate delete targets against an app-owned root.

**R-02.** Only index user-approved roots minus explicit exclusions. Never silently crawl the whole drive, Documents/Downloads, network shares or reparse-point targets. Re-check canonical/scope permissions at use time, not just initial discovery.

**R-03.** Treat filenames, paths, embeddings, extracted text, search queries, thumbnails, transcript content and index database as sensitive. Do not send them to network services or telemetry. Explicit, user-initiated model downloads are the only planned network path in core V1; do not auto-download during search.

**R-04.** Default logs include structured event IDs, duration, error codes and aggregated counts, never arbitrary source text, transcript, vectors, queries or unredacted full paths. Diagnostic exceptions need separate approval and must not be enabled by default.

## 2. Architecture contracts

**R-05.** UI communicates through typed Tauri commands/events; no direct SQLite or model invocation from frontend. Heavy work and FFmpeg never run on the UI thread.

**R-06.** Rust host owns scope authorization, job states, DB writes, schema migrations and retrieval. Inference worker owns model-specific preprocessing/embedding only, not user folder decisions or database mutation.

**R-07.** All AI model access goes through a versioned `EmbeddingProvider` boundary. All vector retrieval goes through `VectorIndex`. No EmbeddingGemma 2 special-case code in UI, watcher, storage schema migrations or unrelated parsers.

**R-08.** Maintain modality-specific extraction policies. A video is indexed with timestamped **segments**, not a single video-wide vector. Never claim speech phrase matching unless a tested transcription/lexical mechanism exists.

**R-09.** Do not introduce a server, background OS service, additional process or major library without a recorded decision showing why existing components cannot meet the requirement. Prefer one local app + one inference worker for MVP.

## 3. Data/index integrity

**R-10.** Every embedding must store model ID/revision, preprocessing policy/version, dimensions, normalization, and source unit revision. Query/corpus vector groups must be compatible; reject mixed signatures and non-finite vectors.

**R-11.** For EmbeddingGemma 2, follow the verified model card: proper asymmetric text retrieval prefixes, do not use fp16, renormalize after Matryoshka truncation, keep document and query dimensions equal. Changes to this policy increment the index signature and trigger compatible rebuilding.

**R-12.** Schema changes require forward-only numbered migrations and migration tests. Never edit a migration already shipped. Changes to vector layout, model or segmenter require explicit invalidation/migration tests.

**R-13.** Index publication is revision-aware and atomic. Partially written temp/cache artifacts, incomplete jobs, and stale file revisions are not exposed as ready search results. Job claims are idempotent, lease/retry limits bounded, restart recovers pending work.

**R-14.** Filesystem watcher events are hints. Maintain reconcile-on-startup and overflow recovery. Unavailable/removable drive does not imply deletion; reparse points must not cause traversal outside consent.

**R-15.** Never replace working indexes or delete cached user-derived results until replacement state can be recovered. Rebuild/reset paths must distinguish **app-owned data** from **source data** in code and tests.

## 4. Performance and robustness

**R-16.** Indexing, video decoding and inference are asynchronous with bounded concurrency, memory and job queue. No unlimited worker pools, synchronous model load on UI thread, or unbounded decoding of images/video.

**R-17.** Preserve exact/lexical search when semantic model is unavailable or incompatible. A semantic subsystem failure may degrade relevance but must not render the entire app unusable.

**R-18.** Measure performance before changing process architecture, vector backend or inference runtime. Provide corpus/hardware versions and p50/p95 plus retrieval quality; optimization must not regress golden queries.

**R-19.** No silent errors, panics on expected user-file failures, or indefinite subprocess waits. Assign typed error code, retryable flag and actionable user-visible state. Bound subprocess runtime/output and never execute a shell-interpolated filename.

## 5. Code, dependencies and tests

**R-20.** Prefer small cohesive modules and explicit types. Domain crates must not depend on UI/Tauri; avoid cyclic crate imports. Do not create generic frameworks or duplicate adapters to solve one task.

**R-21.** Before introducing/updating a runtime dependency, record need, alternatives already present, maintenance risk, license, binary size, attack surface and platform impact. Pin versions/lockfiles. Do not run arbitrary installation scripts from untrusted sources without review.

**R-22.** New nontrivial behavior gets focused unit/integration tests; bug fixes get regression tests. Performance claims require separate benchmark artifacts. Never disable, delete, weaken or falsely report tests to pass a milestone.

**R-23.** Run relevant formatting, lint, typecheck and tests after a change. If an environment command cannot run, say **NOT RUN** with why and do not mark that acceptance criterion as passed.

**R-24.** Test media must be synthetic, consented or redistributable. Never commit actual indexed source files, embeddings from private assets, model weights, machine-specific paths, user index DBs, secrets or diagnostic dumps.

**R-25.** Keep changes scoped to one `STEPS.md` task. Avoid unrelated refactors, dependency churn and sweeping formatting. Make focused commits only after verifying what is staged.

## 6. Git, docs and agent governance

**R-26.** No secrets or authentication material in Git, logs, prompts, committed fixtures or generated docs. Use ignore rules for models, temp files, local caches, builds and private benchmark corpora.

**R-27.** `PLAN.md` changes for architectural/product decision; `STEPS.md` for task state/dependencies/new work; `RULES.md` only for durable rule changes; `AGENTS.md` for execution workflow. Keep accepted ADRs and milestone gates synchronized.

**R-28.** An implementation agent may solve localized code-level decisions, but changing architecture, model runtime, vector engine, storage format, worker IPC, scope/security policy, or license assumptions requires an Architecture Review Queue entry and explicit approval.

**R-29.** No task may be marked `[x]` until implementation exists, acceptance criteria pass, required tests have actual results, and relevant documentation is updated. Never invent successful tests or claim files were edited without checking.

**R-30.** The future feature epics in `STEPS.md` are not permission to implement them. First break down the chosen epic into dependent, testable tasks and approve scope.

## 7. Enforceability checklist (use in code review)

- [ ] Only approved paths accessed, originals unchanged and deletions constrained to app-owned data.
- [ ] All model/index signatures validated; schema migrations and source revision handling safe.
- [ ] No new cross-layer coupling; bounded subprocess/concurrency and recovery paths tested.
- [ ] Query still supports lexical retrieval if inference fails.
- [ ] No personal content in logs, Git, transmitted requests or test output.
- [ ] Relevant tests actually run; docs and task checkbox reflect reality.

**When blocked by a rule:** stop the risky operation, document exact evidence and options in `STEPS.md` Architecture Review Queue, and continue only after decision. Never silently waive a rule.
