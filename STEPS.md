# LocalFind — Executable Implementation Roadmap

**Version:** 0.1.0 · **Status:** not started · **Last planned:** 2026-10-08  
**Architecture:** [PLAN.md](PLAN.md) · **Constraints:** [RULES.md](RULES.md) · **Agent procedure:** [AGENTS.md](AGENTS.md)

## How to use this file

- **One task at a time.** Select the first unchecked task whose dependencies and phase gate are satisfied; `Current Task` at the bottom wins if consistent with dependencies.
- Every unchecked task is **unimplemented**. Do not mark a task complete based on a document, a guess, or an attempted but failing command.
- Format below uses **Deps** (task IDs), **Touch** (expected paths), **Do** (concrete steps), and **Done / prove** (observable acceptance and minimum validation). Those fields are mandatory acceptance contracts.
- An agent may adjust expected paths to the actual repository structure only when consistent with `PLAN.md`; record the deviation and explanation.
- Tests requiring a particular machine, GPU, licensed fixture or released model are **blocked** if unavailable, not passed. Record evidence filenames under `docs/benchmarks/` (only non-private outputs).
- Commands listed here become runnable when the referenced scaffolding exists. Never report them as already executed. Use exact versions/pinned lockfiles for actual setup.
- Overall definition of done: code reviewed at appropriate risk, relevant tests green, no regressions, no skipped required checks, docs/status synchronized, no original user files modified.

## Phase 0 — Architecture feasibility before commitment

**Objective:** Establish a reproducible Windows development environment, validate EmbeddingGemma 2, prove video segment retrieval, and test distribution risk. May use isolated prototypes under `benchmarks/` — **not** production code. Do not implement polished UI yet.

- [x] **P0-T001 — Establish repository and documents**
  - **Deps:** none. **Touch:** repo root, `.gitignore`, `README.md`, these four `.md` files.
  - **Do:** Initialize version control if absent; place `PLAN.md`, `STEPS.md`, `RULES.md`, `AGENTS.md` in root. Create minimal README describing consented local search, proposed stack and documented unknowns. Ignore models, caches, app DB, test media not licensed for sharing and credentials.
  - **Done / prove:** four docs exist, are linked and agree on initial Phase 0; `git status` has no inadvertent binaries or real user data. **Do not:** implement unvalidated product components.

- [x] **P0-T002 — Windows toolchain reproducibility**
  - **Deps:** P0-T001. **Touch:** `scripts/doctor.ps1`, `docs/benchmarks/environment.md`.
  - **Do:** Record OS build, CPU/GPU/RAM, Python, Rust, Node, package manager, FFmpeg and CUDA capability if any. Create a PowerShell doctor script that reports versions without dumping usernames or full home paths.
  - **Done / prove:** doctor runs on intended Windows 11 x64 target; missing prerequisites yield actionable failures. Save versions without private identifying data.

- [x] **P0-T003 — Consented media corpus and judgments**
  - **Deps:** P0-T001. **Touch:** `benchmarks/README.md`, `benchmarks/corpus.example.json`, `benchmarks/queries.example.json`, `tests/fixtures/`.
  - **Do:** Define manifest schema (asset ID, type, path relative to private corpus root, duration, allowed-to-distribute flag). Define labeled query schema (query ID, query type, expected file, segment tolerance). Gather/synthesize ≥100 images and ≥20 videos for full feasibility run; ≥30 judged visual queries; no copyrighted/private assets committed without rights.
  - **Done / prove:** fixture loader validates no missing IDs, invalid timestamps or duplicate query IDs; explain how other developers supply local private fixtures.

- [x] **P0-T004 — Official model text and image smoke experiment**
  - **Deps:** P0-T002. **Touch:** `benchmarks/model_smoke.py`, pinned experiment dependency file, `docs/benchmarks/model-smoke.md`.
  - **Do:** Follow the *current official model card* with supported Python+sentence-transformers/transformers combinations. Encode one retrieval text query, two images and a text document; test query-vs-document prompt rules; record versions and returned vector shape, values and similarity.
  - **Done / prove:** output finite normalized embeddings and reproducible code, or documented blocker with traceback/error class. **Never use fp16** for this model.

- [x] **P0-T005 — Segment-level video embedding experiment**
  - **Deps:** P0-T003, P0-T004. **Touch:** `benchmarks/video_smoke.py`, `docs/benchmarks/video-smoke.md`.
  - **Do:** Probe two different clips using ffprobe; select 2–3 bounded millisecond intervals per clip; encode each segment with an actually supported video input path; test short visual query returning relevant interval; measure decode and encode separately.
  - **Done / prove:** each returned vector maps to its exact source/start/end and real visual result; unsupported runtime/API is logged as a blocker rather than faked using image-only embeddings.

- [x] **P0-T006 — Dtype, dimension and embedding compatibility tests**
  - **Deps:** P0-T004, P0-T005. **Touch:** `benchmarks/embedding_invariants.py`, `docs/benchmarks/embedding-invariants.md`.
  - **Do:** Compare fp32 versus supported bf16 on representative hardware, dimensions 768/512/256, L2 re-normalize truncated outputs, validate finite values/nonzero norms; reject dimension/signature mismatch. Capture example prefix differences for text queries.
  - **Done / prove:** tests fail on NaN/inf, mixed dimensions, wrong normalization; report chosen candidate dimension *only* after quality comparison. **Do not:** silently switch model or precision.

- [x] **P0-T007 — Retrieval-quality feasibility benchmark**
  - **Deps:** P0-T003, P0-T005, P0-T006. **Touch:** `benchmarks/evaluate.py`, `docs/benchmarks/retrieval-v1.md`.
  - **Do:** Build an exact cosine retrieval prototype with labeled images and video segments; measure Recall@1/@5 and MRR@10 on ≥30 judged visual queries, with filename-only and keyframe baseline. Fix deterministic seed, corpus manifest and config.
  - **Done / prove:** reproducible report with per-query errors; proposed gate Recall@5 ≥0.75 on this *curated suite* or explicit PIVOT discussion. **Do not:** publish hand-picked successes as comprehensive quality claims.

- [x] **P0-T008 — Video segmentation policy comparison**
  - **Deps:** P0-T005, P0-T007. **Touch:** `benchmarks/segment_compare.py`, `docs/benchmarks/segmentation.md`.
  - **Do:** Compare candidate fixed windows (e.g., 6/12 seconds) with scene-aware splits; include static screen recording, motion-heavy and short clips. Capture number of units, encode cost, Recall@5 and timestamp error; decide baseline policy/version.
  - **Done / prove:** a documented and configurable initial policy; reject any variant that exceeds model input bounds or has unmapped timestamps.

- [x] **P0-T009 — SQLite FTS + vector backend benchmark**
  - **Deps:** P0-T002. **Touch:** `benchmarks/vector_bench/`, `docs/benchmarks/storage-search.md`.
  - **Do:** Prototype SQLite FTS5 for filename/path and float32 vector BLOB exact cosine search with stable query vectors; test synthetic 10k/50k/250k unit corpora. Compare at least one embedded accelerated candidate (e.g., sqlite-vec) if linear scan misses p95 budget. Record p50/p95 and disk size.
  - **Done / prove:** an evidence-based decision on V1 vector backend, transaction/rebuild behavior, Windows package compatibility and ranking parity. **Do not:** add a remote server for vector search.

- [ ] **P0-T010 — Inference sidecar packaging spike**
  - **Deps:** P0-T002, P0-T004. **Touch:** `benchmarks/packaging/`, `docs/benchmarks/packaging-spike.md`.
  - **Do:** Build the smallest Windows runnable inference process without relying on system Python; spawn from minimal host/PowerShell and encode a query. Test on a clean Windows VM, including missing model/offline behavior, size, startup, update feasibility and failures.
  - **Done / prove:** measured distribution footprint and repeatable clean-VM launch, or a documented change proposal (such as native model runtime) and blocking status.

- [ ] **P0-T011 — Licensing and policy audit for proposed stack**
  - **Deps:** P0-T004, P0-T010. **Touch:** `docs/licenses-inventory.md`.
  - **Do:** Record original upstream links, revision/license/notice obligations for model, model terms, inference libraries, FFmpeg binary build flags, UI, vector backend and any bundled codecs. Explicitly distinguish Apache metadata from other use-policy statements.
  - **Done / prove:** accurate inventory with unresolved items flagged `VERIFY LICENSE`; no unsupported declaration that commercial bundling is automatically cleared.

- [ ] **P0-T012 — M0 formal go / pivot / stop architecture gate**
  - **Deps:** P0-T006, P0-T007, P0-T008, P0-T009, P0-T010, P0-T011. **Touch:** `docs/benchmarks/M0-decision.md`, `PLAN.md`, `STEPS.md`.
  - **Do:** Review evidence against relevance, latency, hardware, deployment and privacy feasibility. Accept/supersede relevant proposed ADRs; if feasibility fails, create explicit alternative experiment tasks before P1.
  - **Done / prove:** signed-off decision by project owner/architect, documented source links/results, selected versioned stack, no unresolved **BLOCKING** issue carried into P1.

**Milestone M0:** P0-T001..T012 accepted with reproducible evidence. *Never treat research notes alone as successful Windows validation.*

## Phase 1 — Durable core and lexical search

**Entry:** M0 GO or approved PIVOT architecture recorded. **Objective:** reliable persistence, user-selected scan scopes and instant lexical retrieval without any ML dependency.

- [ ] **P1-T001 — Bootstrap desktop app + Rust workspace**
  - **Deps:** P0-T012. **Touch:** `apps/desktop/`, `crates/{domain,storage,indexer,search,media}/`, workspace manifests.
  - **Do:** Scaffold Tauri 2 + Svelte TypeScript frontend, shared Rust workspace, dependency lockfiles, bare startup and error boundary. Wire dev commands, no inference yet.
  - **Done / prove:** desktop opens a placeholder screen on Windows; `cargo test --workspace`, TS typecheck and Tauri build/dev checks pass at the current stage.

- [ ] **P1-T002 — Domain model and typed interfaces**
  - **Deps:** P1-T001. **Touch:** `crates/domain/src/`.
  - **Do:** Implement `FileId`, `ScopeId`, `UnitId`, `IndexSignature`, `SourceRevision`, time-range and states. Introduce `FileCatalog`, `JobRepository`, `LexicalIndex`, `VectorIndex`, `EmbeddingProvider` interfaces as needed (no speculative super-framework).
  - **Done / prove:** unit tests validate illegal states/dimensions/ranges; no Tauri/SQLite dependency leaks into domain.

- [ ] **P1-T003 — SQLite connection and migrations**
  - **Deps:** P1-T002. **Touch:** `crates/storage/src/`, `crates/storage/migrations/`.
  - **Do:** Create app-data DB opener, WAL, schema tracking, transactional migrations and version check; create `scopes`, `files`, `units`, `index_jobs`, `index_errors`, `embeddings`, `thumbnails` (as required by contracts).
  - **Done / prove:** create, reopen, reapply-migration and rollback-on-failure integration tests; never touch external user data.

- [ ] **P1-T004 — Consent scopes and safe path policy**
  - **Deps:** P1-T003. **Touch:** `crates/indexer/src/scope.rs`, `crates/storage/src/scopes.rs`.
  - **Do:** Persist selected roots/exclusion rules; compute allowed path; reject traversal and outside-root resolved paths; skip reparse-point recursion by default; support disabling a scope.
  - **Done / prove:** tests for case-insensitive comparisons, nested folders, excluded child, symlink/junction escapes where test host permits; no auto-index of personal folders.

- [ ] **P1-T005 — Deterministic file enumerator/classifier**
  - **Deps:** P1-T004. **Touch:** `crates/indexer/src/scanner.rs`.
  - **Do:** Scan only allowed roots with bounded traversal, classifying image/video formats by supported extension and safe metadata checks; record unsupported/permission errors individually.
  - **Done / prove:** temporary-directory integration tests for nested/excluded/unreadable paths, symlink loops and deterministic discovered file list; errors do not abort global scan.

- [ ] **P1-T006 — File identity and revision repository**
  - **Deps:** P1-T003, P1-T005. **Touch:** `crates/indexer/src/identity.rs`, `crates/storage/src/files.rs`.
  - **Do:** Implement path + stat-based revision, optional validated Windows ID hints and stable internal IDs. Make scan/upsert idempotent; mark unseen files candidate-missing only after successful complete scope scan.
  - **Done / prove:** tests for repeated scan, new/modified/renamed files and changed during scan; no duplicate live path rows.

- [ ] **P1-T007 — Durable job queue and lease recovery**
  - **Deps:** P1-T003, P1-T006. **Touch:** `crates/indexer/src/jobs.rs`, `crates/storage/src/jobs.rs`.
  - **Do:** Add unique `(file, revision, job_kind)` jobs, states, priority, bounded retries/backoff, lease timeout and resume of abandoned work. Ensure serial DB writer ownership.
  - **Done / prove:** crash/restart simulation and concurrency tests; never two active jobs for same file/revision/kind; deterministic retry policy.

- [ ] **P1-T008 — Index readiness, invalidation and atomic publish**
  - **Deps:** P1-T003, P1-T007. **Touch:** `crates/storage/src/publish.rs`.
  - **Do:** Define `ready` visibility, file revision stale detection, commit units/embeddings/file state in SQLite transaction, recover orphan temp artifacts. Handle incompatible index signatures as pending rebuild.
  - **Done / prove:** rollback after injected failure preserves prior ready results; partially staged output never appears in searches.

- [ ] **P1-T009 — SQLite FTS5 for filename and path**
  - **Deps:** P1-T003, P1-T006. **Touch:** `crates/storage/src/fts.rs`, migrations.
  - **Do:** Create FTS5 document mapping; maintain insert/update/delete consistency; escape unsafe query syntax; design exact and prefix path predicates in addition to FTS.
  - **Done / prove:** queries find `IMG_4281`, extension, Unicode filenames and renamed paths; invalid syntax does not crash app; transactional consistency verified.

- [ ] **P1-T010 — Lexical search service and result contracts**
  - **Deps:** P1-T002, P1-T009. **Touch:** `crates/search/src/`, `contracts/search-result.*`.
  - **Do:** Define ranked `SearchResult { file_id, unit_id?, kind, path, reason, timestamp?, thumbnail? }` and lexical candidate retrieval; deterministic tie handling, filters, pagination limit.
  - **Done / prove:** fixture tests return correct first result for exact name; search functions with model fully absent.

- [ ] **P1-T011 — Folder management and status commands**
  - **Deps:** P1-T001, P1-T004, P1-T007. **Touch:** `apps/desktop/src-tauri/`, UI store.
  - **Do:** Add permission-scoped typed Tauri commands for list/add/disable/remove scopes, start scan, pause/resume and progress counters; avoid UI direct file/database access.
  - **Done / prove:** commands validation tests + minimal UI smoke; removing index scope does not delete originals.

- [ ] **P1-T012 — M1 recovery and safety integration gate**
  - **Deps:** P1-T008, P1-T010, P1-T011. **Touch:** `tests/integration/`, `docs/benchmarks/M1-results.md`.
  - **Do:** Use temporary fixture folders for scan twice, restart, query, rename, delete, interrupted job and scope removal. Measure lexical-only latency.
  - **Done / prove:** full suite passes on Windows target; no duplicate file rows or data loss and original-file bytes unchanged.

**Milestone M1:** Durable lexical foundation passes recovery/scope/rename tests. No semantic features claimed yet.

## Phase 2 — Image semantic retrieval vertical slice

**Entry:** M1. **Objective:** reliable, measurable image embeddings and query search with minimal UI.

- [ ] **P2-T001 — Define versioned inference IPC contract**
  - **Deps:** P0-T012, P1-T002. **Touch:** `contracts/inference-v1.json`, `crates/domain/src/inference.rs`, `workers/inference/`.
  - **Do:** Specify handshake, version, request_id, allowed ops, error codes, timeouts, cancellation and embedding response fields. JSONL control on stdout; stderr-only logging; no direct DB writes.
  - **Done / prove:** fixture-based encode/decode and malformed/unknown-version tests for both sides.

- [ ] **P2-T002 — Start/stop/monitor worker host**
  - **Deps:** P2-T001. **Touch:** `crates/indexer/src/worker_host.rs`.
  - **Do:** Spawn Python inference process, validate handshake, serialize one request, bound timeouts, detect exit, kill/restart safely, map worker errors; never shell interpolate a user path.
  - **Done / prove:** fake-worker tests for timeout, process crash, invalid ID/JSON, stderr noise; app still allows lexical search.

- [ ] **P2-T003 — Implement EmbeddingGemma2 adapter**
  - **Deps:** P2-T001, P0-T006. **Touch:** `workers/inference/model_adapter.py`, `workers/inference/protocol.py`.
  - **Do:** Load pinned model from approved local cache; implement encode_query and encode_image; use official retrieval prompt for query; finite/dimension/normalize checks; supported bf16 or float32 only; return full index signature.
  - **Done / prove:** integration fixture produces stable-shaped, comparable vectors, rejects missing/incompatible model; no model download initiated by a search.

- [ ] **P2-T004 — Explicit model acquisition and local cache**
  - **Deps:** P2-T003, P1-T011. **Touch:** `crates/indexer/src/model_manager.rs`, settings UI.
  - **Do:** Show model size/source/terms, obtain user confirmation, download into isolated cache with checksum/revision validation and atomic finalization, support offline use and cancellation.
  - **Done / prove:** no network on offline search; partial downloads not loaded; simulated download failure yields actionable state. Release terms remain unverified until license gate.

- [ ] **P2-T005 — Image preflight and thumbnail adapter**
  - **Deps:** P1-T005, P1-T008. **Touch:** `crates/media/src/images.rs` or approved worker equivalent.
  - **Do:** Validate supported image, dimensions/orientation, max decoded pixels and failure path; generate capped thumbnails under app-owned cache temp/rename strategy.
  - **Done / prove:** corrupt/oversized fixtures fail per-file safely; source hashes unchanged; cache cleanup removes orphans only.

- [ ] **P2-T006 — Persist vector BLOBs behind VectorIndex**
  - **Deps:** P1-T008, P0-T009, P2-T003. **Touch:** `crates/storage/src/vectors.rs`, `crates/search/src/vector.rs`.
  - **Do:** Implement chosen M0-backed vector backend (exact baseline or documented accelerated alternative) with same fingerprint and topK contract; wire atomic publication and delete/invalidations.
  - **Done / prove:** known vectors have exact expected ordering, mismatched signatures/dimensions rejected, reopen preserves results; scale test as M0 specifies.

- [ ] **P2-T007 — Image indexing orchestrator**
  - **Deps:** P1-T007, P2-T002, P2-T003, P2-T005, P2-T006. **Touch:** `crates/indexer/src/image_pipeline.rs`.
  - **Do:** For one file revision: preflight → thumbnail → call worker → validate embedding → re-stat → publish; model per-file error/skip behavior and retries. Bound queue/backpressure.
  - **Done / prove:** 100 test images index with correct `ready` state, failure and interruption recovery; never publish changed file revision.

- [ ] **P2-T008 — Semantic query path and candidate fusion**
  - **Deps:** P1-T010, P2-T002, P2-T006, P2-T007. **Touch:** `crates/search/src/query.rs`, `crates/search/src/fusion.rs`.
  - **Do:** Run exact/lexical retrieval before or concurrently with query embedding; vector topK filter by signature; use deterministic RRF and file dedupe; query encoding failure degrades to lexical-only with reason.
  - **Done / prove:** golden tests for image concepts and exact filenames; errors don't blank lexical results; mismatched models never mixed.

- [ ] **P2-T009 — Image results UI**
  - **Deps:** P1-T011, P2-T008. **Touch:** `apps/desktop/src/`.
  - **Do:** Create search field, debounced async query with stale-request cancellation, compact result list, thumbnail, match label, `Open` and `Reveal in Explorer` with source existence check.
  - **Done / prove:** keyboard usable in local Windows app; outdated responses don't overwrite newer ones; no direct file DB access from frontend.

- [ ] **P2-T010 — Golden image retrieval and profiling gate**
  - **Deps:** P2-T007, P2-T009. **Touch:** `tests/e2e/`, `docs/benchmarks/M2-results.md`.
  - **Do:** Run labeled image suite end-to-end, measure retrieval quality and warm query p50/p95, CPU/RAM and index size; include Unicode paths and failed image fixtures.
  - **Done / prove:** result artifact identifies accuracy and latency vs gates; M2 approved only if image search is useful and stable or deviations explicitly accepted.

**Milestone M2:** User selects an image folder and retrieves images by natural language with reliable persistence; measured, not mocked.

## Phase 3 — Video segment retrieval and exact time navigation

**Entry:** M2. **Objective:** show the precise visual moment in a video, not a file-only hit. Keep visual-only claim explicit.

- [ ] **P3-T001 — Video probe adapter**
  - **Deps:** P0-T011, P1-T005. **Touch:** `crates/media/src/probe.rs`.
  - **Do:** Call approved ffprobe binary without shell; parse duration/time base, streams, codec, frame rate and rotation; bound output and handle missing/corrupt stream metadata.
  - **Done / prove:** fixtures cover MP4/MOV/MKV, VFR and malformed file; errors are typed, per-file, and leave original bytes unchanged.

- [ ] **P3-T002 — Segment policy and versioned time ranges**
  - **Deps:** P0-T008, P3-T001. **Touch:** `crates/media/src/segments.rs`, `crates/domain/src/time.rs`.
  - **Do:** Implement the selected M0 policy generating bounded `(start_ms,end_ms)` segments; clamp to duration, handle zero/short durations and deterministic IDs/revisions; version the policy.
  - **Done / prove:** unit tests show no negative, reversed, empty or out-of-bounds ranges, and deterministic output for identical metadata.

- [ ] **P3-T003 — FFmpeg frame/thumbnail extraction**
  - **Deps:** P3-T001, P3-T002, P2-T005. **Touch:** `crates/media/src/video_frames.rs`.
  - **Do:** Extract representative preview frames with a bounded resource budget using exact segment timestamps; create app-owned thumbnails via temp+rename; avoid command-shell interpolation.
  - **Done / prove:** fixture thumbnails are valid and timestamped, codec failure localized, output cached safely, originals untouched.

- [ ] **P3-T004 — Video encode operation in inference worker**
  - **Deps:** P2-T001, P2-T003, P3-T002. **Touch:** `workers/inference/video_adapter.py`, `contracts/inference-v1.json`.
  - **Do:** Add `encode_video_segment(source_path,start_ms,end_ms,config)` using proven P0 video processing, correct sampling and context constraints; output embedding and segment signature. Ensure authorized path and bounds validated by host.
  - **Done / prove:** video segment embeddings comparable to queries, valid dimensions/norm, errors for unsupported codec/time span, no whole-video single vector shortcut.

- [ ] **P3-T005 — Durable video-segment publication**
  - **Deps:** P1-T008, P2-T006, P3-T002, P3-T004. **Touch:** `crates/indexer/src/video_pipeline.rs`, storage unit queries.
  - **Do:** Probe → segment → thumbnail → encode → verify file unchanged → atomic publish unit/embedding rows. Clean or supersede old segments when source revision/policy changes.
  - **Done / prove:** interruption mid-video resumes without duplicate `ready` units; file update invalidates stale timestamps; no corrupt partial result.

- [ ] **P3-T006 — Timestamp-aware retrieval and grouping**
  - **Deps:** P2-T008, P3-T005. **Touch:** `crates/search/src/video_results.rs`, result contracts.
  - **Do:** Query mixed image/video index; group top distinct segments by file, preserve best 1–3 timestamp matches; return thumbnails and explicit `semantic_video_segment` reason.
  - **Done / prove:** crafted fixture retrieves two unrelated moments from one video when queried independently; deterministic result ordering.

- [ ] **P3-T007 — Video open/seek action**
  - **Deps:** P3-T006, P2-T009. **Touch:** `apps/desktop/src/`, `apps/desktop/src-tauri/`.
  - **Do:** Provide preview/open at relevant start_ms. Prefer app-owned player where seeking is controlled; if using system player, detect inability to seek and offer clear fallback. Validate source still exists.
  - **Done / prove:** manual automated/semi-automated check seeks within the defined tolerance; exact timestamp displayed; missing file handled visibly.

- [ ] **P3-T008 — Video indexing limits and error UX**
  - **Deps:** P3-T005, P1-T011. **Touch:** indexer config, errors UI.
  - **Do:** Enforce size/duration/decode time limits and resource caps; provide `unsupported codec`, `too large`, `permissions`, `unavailable` states and retry action. Support skipping a problematic video without stalling queue.
  - **Done / prove:** injected hung decoder is killed; next queued file completes; progress stays truthful.

- [ ] **P3-T009 — Video golden retrieval + temporal evaluation**
  - **Deps:** P3-T006, P3-T007, P3-T008. **Touch:** `benchmarks/evaluate.py`, `docs/benchmarks/M3-results.md`.
  - **Do:** Evaluate all judged video queries with timestamp tolerance, Recall@5, MRR and number/size of segment vectors; compare against P0 method; profile CPU/RAM/decode.
  - **Done / prove:** measurable retrieved moments and known false positives documented; no marketing of spoken phrase search without STT.

**Milestone M3:** Select folder → index videos → text visual query → correct video segment shown and seeked; tests and timings recorded.

## Phase 4 — Windows UX, incremental behavior and MVP hardening

**Entry:** M3. **Objective:** turn a demo into a recoverable daily-use Windows program.

- [ ] **P4-T001 — Windows watcher adapter and coalescing**
  - **Deps:** P1-T006, P1-T007. **Touch:** `crates/indexer/src/watcher.rs`.
  - **Do:** Add watcher per enabled scope; debounce and merge create/modify/delete/rename events; on overflow/unreliable watcher state schedule reconciliation; never assume each event maps to one file operation.
  - **Done / prove:** Windows integration tests for burst writes, rename, directory move and delete; no duplicate active job.

- [ ] **P4-T002 — Startup and bounded periodic reconciliation**
  - **Deps:** P4-T001, P1-T006. **Touch:** `crates/indexer/src/reconcile.rs`.
  - **Do:** On startup and missed-event flags compare current scope tree vs catalog in bounded batches. Distinguish offline drive from deleted content; avoid deleting indexed metadata on transient IO errors.
  - **Done / prove:** tests simulate app closed during modifications and missing mount; correct stale/up-to-date states after reconciliation.

- [ ] **P4-T003 — Priority scheduling, pause/resume, cancellation**
  - **Deps:** P1-T007, P2-T002, P3-T005. **Touch:** `crates/indexer/src/scheduler.rs`.
  - **Do:** Implement bounded queues, one model instance, user-visible pause/resume, safe cancellation between segments, and interactive query priority. Requeue leases after process death.
  - **Done / prove:** pausing stops *new* expensive work; indexing resumes after restart; queries remain responsive; no stale ready record from cancelled work.

- [ ] **P4-T004 — Model errors and semantic-degraded mode**
  - **Deps:** P2-T004, P2-T008. **Touch:** `crates/search/src/`, UI statuses.
  - **Do:** If model absent, loading, incompatible or OOM, show reason; permit lexical query and file browsing; never spin repeated downloads or crash loops.
  - **Done / prove:** chaos tests for missing cache, killed worker, invalid signature and insufficient resources preserve lexical search.

- [ ] **P4-T005 — Index management, cleanup and recovery**
  - **Deps:** P1-T008, P4-T002. **Touch:** `crates/storage/src/cleanup.rs`, settings UI.
  - **Do:** Add rebuild corrupted/incompatible derived indexes, clear thumbnails/cache, remove scope-owned records; make destructive app-data controls explicitly confirm target directory and show bytes affected.
  - **Done / prove:** temp fixture asserts byte-for-byte originals unchanged across reset/remove/rebuild and schema update.

- [ ] **P4-T006 — Secure path/IPC/media boundary audit**
  - **Deps:** P2-T002, P3-T008, P4-T001. **Touch:** security tests, `docs/security-review.md`.
  - **Do:** Test outside-root rejection, junction escape, malformed JSONL, command injection attempts in filenames, large metadata, hostile broken media, no raw sensitive logs and no unsolicited network on search.
  - **Done / prove:** threat list with observed results and fixed high-severity issues; block release on unresolved source-file deletion or network exfiltration risk.

- [ ] **P4-T007 — UX for folder/model/indexing states**
  - **Deps:** P4-T003, P4-T004, P4-T005. **Touch:** `apps/desktop/src/`.
  - **Do:** Finish first run, consent/exclusions, progress/error summary, model install state, pause/resume, open/reveal and empty/offline states. Add keyboard navigation and accessible labels.
  - **Done / prove:** test matrix for zero files, unknown codec, download cancelled, indexing active and available results; no fabricated % complete.

- [ ] **P4-T008 — Retrieval robustness and ranking regression suite**
  - **Deps:** P2-T010, P3-T009, P4-T002. **Touch:** `benchmarks/`, `tests/integration/`.
  - **Do:** Add filename vs semantic competition cases, Unicode names, mixed modalities, deleted file, two timestamps, duplicate segments and embedding version change. Set deterministic golden thresholds.
  - **Done / prove:** reproducible CI-compatible suite passes; every regression carries a labeled test.

- [ ] **P4-T009 — Performance profiling, capped resources, storage budgets**
  - **Deps:** P4-T003, P4-T008. **Touch:** `docs/benchmarks/M4-performance.md`, scheduler config.
  - **Do:** Measure warm UI query/p95, vector topK, model load, segment throughput, RAM/VRAM and disk. Implement resource caps only when measurements show need; compare against PLAN budgets.
  - **Done / prove:** benchmark report on documented hardware; material shortfalls explicitly accepted or fixed; results remain searchable during indexing.

- [ ] **P4-T010 — Windows E2E smoke and recovery matrix**
  - **Deps:** P4-T005, P4-T006, P4-T007, P4-T009. **Touch:** `tests/e2e/`, `docs/benchmarks/M4-decision.md`.
  - **Do:** Cold start → consent → model present → index → visual query → timestamp seek → pause → kill app → restart → modify/delete/rename → search → remove index. Repeat with CPU-only if gate supports that tier.
  - **Done / prove:** all required scenarios completed with logs of counts/timings and original-file checks. Clean Windows installer is **not** implied by this gate.

**Milestone M4 — Functional development MVP:** All Phase 4 acceptance tests pass, documented hardware requirements are based on experiments, and owner approves any residual known issues.

## Phase 5+ — Future features (intentionally less detailed; expand before starting)

The items below are **epics, not coding-agent-ready tasks**. Before beginning each phase, Opus/architect must break it into the same small task format above and update PLAN/RULES/AGENTS where necessary. No hidden permission to start these before M4.

- [ ] **P5-E001 — Audio semantic retrieval feasibility**. Compare native audio embeddings on varied clips; evaluate language, CPU and retrieval quality; define timestamp unit contract.
- [ ] **P5-E002 — Local speech transcription**. Benchmark offline STT model, word/segment timestamps and exact phrase FTS. Distinguish spoken words from semantic audio themes.
- [ ] **P5-E003 — PDF/plain text structured extraction**. Compare parser libraries, preserve page/heading structure, chunk and FTS/semantic rank. Avoid unlicensed/proprietary parsers by accident.
- [ ] **P5-E004 — Screenshot and scanned PDF OCR**. Add separate extractor with model version and privacy constraints; measure recall on text-heavy images.
- [ ] **P6-E001 — Source code-aware search**. Implement lexical filename/symbol baseline before embedding chunks; benchmark exact symbols and conceptual code queries.
- [ ] **P6-E002 — Ranking refinement**. Evaluate query intent, RRF tuning and modality-specific fusion on locked golden dataset; do not regress exact filename results.
- [ ] **P7-E001 — Distribution hardening**. Installer (NSIS/MSI comparison), model licensing, packaging, code signing, app/model migrations, supported GPU matrix and clean Windows VM release tests.
- [ ] **P7-E002 — Optional global launcher & platform UX**. Keyboard shortcut, background lifecycle, startup integration, accessibility and internationalization after core performance is proven.

## Mandatory milestone reviews

Before marking `M0`, `M1`, `M2`, `M3` or `M4` complete:
1. Compare code/result against `PLAN.md`, this phase exit gate and `RULES.md`.
2. Inspect relevant tests and command output; report **not run** explicitly.
3. Check for privacy/data loss, schema/model/version drift and unexplained regressions.
4. Architect/Opus approves meaningful stack or architecture revisions; record ADR changes.
5. Update status below and checkpoint Git without generated user data.

## Architecture Review Queue (template)

Use only for material architectural decisions; ordinary implementation details belong in the current task notes.

```text
ARQ-001 | status: open | task: Pn-Tnnn
Problem:
Evidence:
Existing contract or ADR affected:
Options + trade-offs:
Recommended option:
Migration/security/performance impact:
Blocking: yes/no
Reviewer decision + date: pending
```

**Queue:** empty at initialization.

```text
ARQ-001 | status: decided (provisional, Option A) | task: P0-T007
Problem: visual Recall@5 0.677 @768-d misses the fixed 0.75 feasibility gate on the curated 31-query suite.
Evidence: docs/benchmarks/retrieval-v1.md + retrieval-v1-results.json (P0-T007 exit 0); image queries R@5 0.775 but video queries 0.545; dominant failure is cross-modal crowding (red images outrank red video halves); near-duplicate synthetics + single-relevant labels contribute; keyframe baseline 0.629 < segment 0.677; filename baseline confounded by label-leaking filenames.
Current contract: PLAN.md section 2 (Recall@5 >= 0.75 gate; fail triggers investigation, not silent lowering) + R-08 (segment-level video).
Options: A. Keep model, fix methodology first (per-modality score normalization/RRF per PLAN section 9, judgment revision for repeated colors, realistic filenames) then re-measure (low cost, no arch change). B. Change segmentation/model now (high cost, premature before fixing known measurement confounds).
Recommendation: A -- the miss is measurement/fusion-dominated, not proof the model cannot retrieve (P0-T004/P0-T005 smokes show genuine visual discrimination).
Impact: no migration; P0-T009/T010/T011 proceed; M0 decision stays pending review.
Blocking: no (for P0-T009..T011); yes for M0 GO.
Reviewer decision + date: PROVISIONAL Option A (2026-10-09) — Continue with
EmbeddingGemma 2 and investigate hybrid/modality-aware retrieval.

Update 2026-10-08 (P0-T008 evidence): segmentation is a secondary lever
(video-only R@5 0.545-0.818 across policies; overlap rejected with measured
inferiority; scene best per-unit but hard-cut-only; measured scene+w2 hybrid
underperforms both parents; full-clip ranks files 1.000 with ~1071 ms
timestamp error). Ranking/fusion still dominates (file-grouping halves
video-only 0.545 -> 1.000). Model representation not the blocker (halves
discriminable; two-tone cuts exact). Recommendation A stands; no model swap,
no threshold change. Still blocking M0 GO only.
```

```text
ARQ-001 decision record 2026-10-09 (architect, provisional — Option A):
Continue with EmbeddingGemma 2 and investigate hybrid/modality-aware
retrieval. Corrected-evaluation basis: P0 integrity rework (strict-vs-
tolerant sensitivity, file-grouped metrics, byte-split index accounting,
4-part key + reporting fixes; 75 benchmark unit tests green; independent
metric review: no blocking defects).
- The original 0.677 Recall@5 gate miss (tolerant policy, @768-d, 31-query
  curated suite) remains valid as historically measured: the pre-rework
  report is preserved byte-identically
  (docs/benchmarks/retrieval-v1-results.historical.json) and the v2 report
  reproduces it per query.
- M0 GO remains BLOCKED. This corrective task does not constitute a new M0
  gate approval and starts no fusion implementation.
- Segmentation policy (fixed 2 s windows, seg-v1) remains provisional and
  reversible: segment-Recall orderings reverse under strict scoring
  (halves mixed 0.818 vs w2 0.682), file-grouped ties (0.818), so no
  segmentation superiority is claimed.
- Fusion (per-modality score normalization / RRF per PLAN section 9) and
  evaluation methodology (tolerance-denominator effects, file-grouped
  comparisons, single-relevant judgment narrowness) need further
  investigation before any model decision.
- Model replacement is not justified by current evidence (P0-T004/P0-T005
  visual discrimination holds; scene strict video-only 1.000 and full
  file-level 1.000 show representation is not the blocker).
- No roadmap change: P0-T009 remains the current task (not started here);
  no P1/fusion work begins under this entry.
Blocking: yes for M0 GO; no for P0-T009..P0-T011.
```

## Project State (source of truth)

- **Current Phase:** Phase 0 — feasibility
- **Current Milestone:** M0 (pending)
- **Current Task:** P0-T010 — Inference sidecar packaging spike
- **Last Completed Task:** P0-T009 — SQLite FTS + vector backend benchmark (2026-10-09; exact BLOB + sqlite-vec 0.1.9 prototype, 10k/50k/250k + 256-d appendix, 28 correctness tests green, evidence in docs/benchmarks/storage-search.md; V1 recommendation recorded, no ADR accepted, M0 still blocked)
- **Blocked Tasks:** None recorded (Phase 0 dependencies deliberately pending)
- **Known Blockers:** Windows runtime compatibility, retrieval quality, packaging and license obligations must be verified in Phase 0; these are *research gates*, not known failures.
- **Architecture Review Queue:** ARQ-001 decided provisional Option A (2026-10-09; M0 GO still blocked, P0-T009..T011 unblocked)
- **New Risks:** See PLAN.md §14; add discoveries with risk ID.
- **Last Documentation Review:** 2026-10-09 (P0-T009: storage-search.md + results JSON with 768-d scales and 256-d appendix; benchmarks/vector_bench prototype + 28 tests; independent review findings B1–B5 fixed; M0 GO still blocked, P0-T010 ready)
