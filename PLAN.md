# LocalFind — Product & Architecture Plan

**Status:** Proposed architecture (not implemented)  
**Version:** 0.1.0 • **Last reviewed:** 2026-10-08  
**Owners:** Product owner (user); architecture reviewer (Opus); implementation agents (Muse Spark or equivalent)  
**Related:** [STEPS.md](STEPS.md) · [RULES.md](RULES.md) · [AGENTS.md](AGENTS.md)

> **Decision rule:** Phase 0 evidence can overturn any provisional technology choice. Do not mistake this planning document for proof that the proposed runtime or product performs adequately.

## 1. Vision, user and promise

**LocalFind** is a Windows-first, privacy-preserving desktop search application: *find local files and precise visual moments using everyday language, without uploading personal content.* Its long-term product is Everything/Spotlight-like discovery enhanced with multimodal semantic retrieval, not a chatbot or media editor.

**Primary user:** a person with thousands of local screenshots, photos, recordings, and videos who remembers *what happened*, not the filename. **Secondary user (later):** developer or knowledge worker seeking code and documents.

Core journeys:
1. Choose one or more explicit folders; see scan scope and status.
2. Index images and selected videos in the background without changing originals.
3. Type `someone walking a black dog` and open the matching image or video timestamp.
4. Type an exact filename/partial path and get a deterministic result without requiring the model to be loaded.
5. Pause indexing, remove a folder from scope, or rebuild a corrupt/incompatible index safely.

### MVP scope (first usable release)
- Windows 11 x64 as **development and test baseline**; other Windows versions are a later compatibility decision.
- User-selected directories only; JPEG/PNG/WebP images and MP4/MOV/MKV videos as **best-effort** inputs. Codec failures are expected and reported, not hidden.
- Image semantic retrieval, segment-level visual video retrieval, timestamped preview/open, hybrid filename/metadata/semantic retrieval.
- Persistent local index, durable incremental jobs, watcher + reconciliation, basic progress and pause/resume, per-file errors.
- Offline search after the model is locally installed. Exact filename search must work even when model load fails.
- Folder exclusion, index deletion that never touches original files, and an explicit model download / disk usage disclosure.

### Deferred, not implied by MVP
- Audio semantic search, exact speech via STT, OCR, PDFs, Office documents, source-code semantic indexing, archive formats, network shares, removable-drive resilience beyond safe error reporting.
- Global hotkey, Windows Search integration, app installer / automatic updates, polished GPU matrix, multilingual evaluation, accessibility audit, advanced duplicate grouping.
- Cloud sync, user accounts, invasive activity tracking, autonomous file edits and RAG chat are **non-goals** unless product scope is explicitly changed.

## 2. Measurable success criteria and feasibility gates

**Early product validation:** With a consented, reproducible test corpus containing ≥100 images and ≥20 video clips with known relevant moments, measure `Recall@5` / `MRR@10` per query class against filename-only and basic frame retrieval baselines. A Phase 0 go decision requires repeatable positive semantic retrieval for visual queries; define an initial gate of `Recall@5 >= 0.75` on a **small curated feasibility suite** of ≥30 visual queries (not a general benchmark claim). Failing this gate triggers investigation of prompts, sampling, normalization and the candidate model, not silent lowering of the target.

**Budget targets, not promises:**
- Typing → first lexical results: target p95 ≤150 ms on test hardware with warm DB, excluding cold application startup.
- Warm **text query encoding + vector lookup + ranking**: aspirational p95 ≤750 ms at the representative dataset size; measure each stage independently and revise based on usability tests.
- No UI-thread decoding, inference or broad directory scanning; search works during indexing.
- Baseline initial target: support ≥10k files and ≥50k searchable units; measure 10k / 50k / 250k vectors before selecting the scalable vector backend.
- Record cold model load, RAM, VRAM, batch throughput, battery / CPU load on specified hardware. No unsupported minimum hardware or GPU guarantee.
- Establish index growth report in bytes/file, bytes/segment and bytes/vector, plus thumbnail overhead.

**Phase 0 decisions:** `GO` (supported by tests), `PIVOT` (change model/runtime/index), or `STOP` (poor usability/retrieval/cost). Record measurements with machine specs, versions, corpus IDs and commands. No numerical performance claim without a benchmark report.

## 3. Reference facts and uncertainties

As of **2026-10-08**, the official model card describes `google/embeddinggemma-2` as a multimodal text/image/video/audio embedding model with native **768-dimensional** output, Matryoshka **512/256/128** dimensions, and an **8,192-token shared input budget**. Default video sampling is **1 FPS**, configurable. Text retrieval has distinct query/document prompting. For truncated vectors, **re-normalize after truncation**; query and indexed vectors must use the same dimension. **Do not use fp16** for this model: the card warns of NaNs/degraded results; use supported bf16 or float32. The model card displays Apache-2.0; also review its stated Gemma policy and all packaging/dependency licenses before release.

Official references:
- https://huggingface.co/google/embeddinggemma-2
- https://v2.tauri.app/develop/sidecar/
- https://www.sqlite.org/fts5.html
- https://learn.microsoft.com/windows/win32/fileio/obtaining-directory-change-notifications
- https://v2.tauri.app/distribute/windows-installer/

**VERIFY in actual Windows experiments:** PyTorch/sentence-transformers versions that work together; GPU backends and driver requirements; video/audio file input details; CPU-only throughput; efficient repeated segment processing; exact model download size and packaging; text-to-video retrieval quality on *our* clips; whether the proposed sidecar can be packaged and started on a clean Windows VM; redistributable codec and license obligations.

## 4. Initial technology decisions (provisional ADRs)

| ID | Decision | Rationale | Alternatives / risks | Validation trigger |
|---|---|---|---|---|
| ADR-001 | **Tauri 2 + Svelte/TypeScript UI** | Relatively small native shell, responsive keyboard-driven UX, typed calls to Rust | WinUI/C# stronger native integration; Electron easier JS integrations but heavier; WebView2 deployment dependency | Windows startup / packaging smoke test |
| ADR-002 | **Rust** for application/core and storage; **Python inference sidecar** for initial multimodal runtime | Rust for filesystem, SQLite, jobs; Python for up-to-date model integration and fast research iteration | Pure C#/ONNX/native runtime may ship smaller; Python/PyTorch packaging may be unacceptably large | Phase 0 model+packaging gate; retain adapter boundary |
| ADR-003 | **SQLite (WAL) + FTS5** for metadata, durable jobs and lexical indexing | Embedded, transactional, single install footprint | SQLite single writer requires bounded DB writer, WAL maintenance; FTS tokenizer semantics differ by language | Functional/recovery benchmark; FTS enabled check |
| ADR-004 | **SQLite-stored float32 vector BLOBs + Rust exact cosine baseline**, replaceable `VectorIndex` port | Simple, correct, portable baseline for small data, same transactional DB; establish ground truth | Linear scan will NOT scale to 250k+ or perhaps even 50k units. Evaluate `sqlite-vec` / USearch or another embedded ANN engine after measurement | Must meet latency gate before declaring scalable MVP; no server DB by default |
| ADR-005 | **Bundled FFmpeg/ffprobe candidate** for probe, decoding and still thumbnails | Mature support for diverse video formats | Binary size, Windows packaging, codec licensing, process sandboxing; alternatives native decoder | Release licensing/clean VM smoke test |
| ADR-006 | **Windows FS events via tested Rust `notify` adapter + periodic reconciliation** | Efficient steady-state updates, misses healed after app shutdown | Watcher overflow, network/disconnected volumes, path identity; USN journal optional later | Rename/delete/overflow tests |
| ADR-007 | **Separate single inference worker via stdio request/response IPC** | Contains model crashes, avoids UI blocking, one controlled model instance | JSON overhead; worker hangs/packaging; no direct database access | Protocol benchmark & cancellation test |
| ADR-008 | **User-controlled model download, no background telemetry by default** | Transparent storage/network use and offline use thereafter | First run may be large/slow | Download verification, offline E2E |

**Trade-off for ADR-004:** SQLite vector BLOBs permit atomic metadata + embeddings publication. The first implementation must test whether a direct scan of BLOB rows meets p95 targets. If not, a distinct index implementation is selected at a milestone gate. A separate ANN index introduces eventual synchronization/rebuild logic; it MUST never be treated as the source of truth for file state.

### Proposed repository layout

```text
localfind/
├── PLAN.md  STEPS.md  RULES.md  AGENTS.md
├── apps/desktop/              # Svelte + TS UI and Tauri host configuration
│   ├── src/                   # screens/components/state, no raw DB access
│   └── src-tauri/             # Tauri commands, bootstrap / lifecycle
├── crates/
│   ├── domain/                # types, file identity, ports, pure policies
│   ├── storage/               # SQLite migrations, FTS5, vector baseline
│   ├── indexer/               # scanning, watcher, jobs, orchestration
│   ├── search/                # retrieval orchestration, fusion and ranker
│   └── media/                 # probe, segmentation, thumbnails (FFmpeg adapter)
├── workers/inference/         # Python model integration, protocol, tests
├── contracts/                 # versioned IPC schema + shared fixtures
├── tests/fixtures/            # synthetic/consented, small and license-safe
├── benchmarks/                # harness, corpus manifest, query judgments
├── scripts/                   # setup/build/test helpers
└── docs/                      # benchmark results, release checklist later
```

This is a target layout: creating every directory before it is needed is unnecessary. No microservices or local HTTP servers for MVP.

## 5. Runtime and trust boundaries

```text
┌───────────────────────────────────────────────┐
│ Tauri WebView: Svelte UI                       │
│ search, results, folder consent, progress     │
└───────────────────────┬───────────────────────┘
                        │ typed Tauri commands/events
┌───────────────────────▼───────────────────────┐
│ Rust application host                         │
│ search coordinator | scanner | job scheduler  │
│ SQLite metadata + FTS5 + vector interface     │
│ watcher | media adapter | permissions         │
└───────────────┬───────────────────┬───────────┘
                │ versioned stdio    │ read-only source access
┌───────────────▼──────────┐ ┌──────▼──────────┐
│ Python inference worker │ │ ffprobe / FFmpeg │
│ model adapter, encode   │ │ probe, segments  │
│ no DB writes            │ │ and thumbnails   │
└──────────────────────────┘ └─────────────────┘
```

Rust owns **all** persistent writes and source-scope decisions. Sidecar requests receive only authorized paths and selected time ranges; worker cannot select folders, access database or make remote requests as part of inference. The worker uses a versioned **newline-delimited JSON** protocol initially: `{protocol_version, request_id, op, payload}` → `{request_id, ok, result|error}`. Keep stdout protocol-only; logs on stderr. Cap request size/time, time out hung calls, enforce one in-flight inference request initially. Large binary payloads pass as files inside the controlled application cache or by authorized source file path, **not** base64 frame arrays. Do not parse or trust model output blindly: validate dimensionality, finite values, norm, input version, file ownership and request IDs.

**Security caveat:** A Python worker given a path runs with the user's OS privileges, not a real OS sandbox. The host validates indexing scope and inputs, and the worker uses no arbitrary tool execution or network calls. Strong sandboxing may require a later Windows security design.

### Component contracts

| Port / component | Responsibility; inputs → outputs | Must NOT do |
|---|---|---|
| `ScopeRepository` | consented roots + exclusions → effective allowed paths | silently index outside consent |
| `FileCatalog` | scan/watch records → durable file changes | parse media or run AI |
| `JobRepository` / scheduler | file/version state → queued/retried tasks | spawn unlimited workers |
| `MediaProcessor` | authorized path, segment policy → probe, segment descriptors, thumbnails | change source files |
| `EmbeddingProvider` | query/image/video-segment request → validated embedding + fingerprint | own SQLite/vector persistence |
| `VectorIndex` | versioned vectors → upsert/delete/topK | decide filesystem scope |
| `LexicalIndex` | file names/paths (+ later extracted text) → matches | invoke inference |
| `SearchCoordinator` | query/filters → candidates, grouping, explanations | directly access UI or FFmpeg |
| `Storage` | transactional state, migrations → repositories | make UI policy decisions |
| `DesktopCommands` | validated calls/events → DTOs | execute blocking ML on UI thread |

## 6. Domain and persistence design

Use random internal IDs (e.g., UUID/ULID) stable across a file's lifecycle when identity can be established. **Never assume a normalized pathname is a global, permanent identity.** Windows volume/file IDs may help on NTFS but are opportunistic; scanning can reconcile on paths, size and modification times, escalating to a hash only when needed. Case folding, junctions/reparse points, UNC and external drives need explicit policy. MVP: local fixed drives + explicit folders only; skip reparse-point recursion by default; don't follow paths outside a chosen root.

**Conceptual entities** (exact SQL belongs to migration tasks):
- `scopes(id, root_path, enabled, exclusions_json, created_at)`
- `files(id, scope_id, canonical_path, display_path, extension, kind, size, mtime, volume_hint, file_id_hint, fingerprint, observed_generation, state, last_error)`
- `units(id, file_id, kind[image|video_segment|text_chunk|audio_segment], start_ms, end_ms, page, extractor_version, source_revision, state)`
- `embeddings(unit_id, model_key, model_revision, preprocessing_revision, dimensions, dtype, normalized, vector_blob, created_at)`
- `index_jobs(id, file_id, target_revision, kind, state, attempts, available_at, lease_expires_at, error_code)`
- `index_errors(id, file_id, stage, code, retryable, occurred_at)`
- `thumbnails(unit_id, cache_key, created_at)`
- `search_fts` (FTS5 over filename/path initially; later selectively over extracted text/transcripts)
- `schema_migrations(version, applied_at)` and `index_state(manifest_version, model_config, segmenter_version)`

Invariants:
1. An embedding belongs to **one immutable content unit revision** plus explicit model, preprocessing and dimension fingerprint. `index_signature = hash(model ID + model revision + dimension + normalization + prompt policy + segment policy + extractor policy)`.
2. A candidate is queryable only when its unit and embedding are `ready` and source revision matches. Query encoder uses the same model/dim, search prompt policy and normalization as the corpus embedding group; document prompts are only for textual corpus content, not image/video inputs.
3. Keep explicit FTS row mapping and migration tests. User content is never stored by default in plaintext logs.
4. SQLite WAL + one bounded writer, read-only connections for search as appropriate; busy retries and bounded transactions. DB writes for one processed file are committed atomically after media/embedding artifacts are ready. Crash before commit leaves previous visible records valid; startup requeues abandoned leases.
5. Cache artifacts are written to temporary paths then atomically renamed. Orphan cache entries may be cleaned later; DB pointers must never intentionally refer to partially written artifacts.
6. Deleting a scope or resetting an index removes **app-owned** indexes/thumbnails only. Never delete originals. If a drive disappears, mark source `unavailable`, not `deleted`; reconcile on reconnection.

**App data path (proposal):** per-user OS app-data directory; e.g. local application data `LocalFind/` containing `db/`, `models/`, `thumbs/`, `tmp/`, `logs/`. Derive via OS APIs / Tauri path APIs, never hardcode `C:\Users\...`. Provide space usage and user-controlled cleanup. Windows file permissions and optional later encryption-at-rest require threat-model review.

## 7. Ingestion, indexing and lifecycle

**Initial scan:** select scope → enumerate with exclusions + loop safety → lightweight metadata upsert → enqueue changed/new files → coarse resource-aware scheduler → file probe/extraction → compute unit revision/segments → inference batch → validate results → publish in transaction → emit progress.

**Incremental:** watch events are hints, **not proof**. Debounce, wait for stability, coalesce related paths, compare revision, schedule if changed. Run a bounded scope reconciliation at startup and after watcher overflow/unreliable state. On rename retain `file_id` when identity evidence allows; otherwise treat as delete+add without losing safety. On deletion, mark missing, invalidate linked units; on unavailable drives, preserve records without falsely deleting. Optional content hash dedupe comes after measurement.

**Worker scheduling:** single inference worker for first version; bounded media-probe pool (e.g. 2, tune after tests); one SQLite writer; queue priorities `(interactive text query > user-requested indexing > background indexing)`. Pause stops new jobs, cancellation interrupts at safe boundaries, shutting down preserves durable job states. Retry only transient errors with backoff and cap; malformed/unsupported assets become persistent per-file errors. Expose `pending/running/succeeded/failed/skipped` separately from file readiness.

**Concurrency race policy:** index tasks carry source revision; on completion host re-stats input before publication. If changed, discard staged result and requeue; never publish stale content as ready. Search reads a committed snapshot and can serve ready rows while indexing continues.

## 8. Multimodal pipelines

### Image pipeline (MVP)
Authorized path → validate size/ext/content → decodable image check → image embedding → normalized versioned vector → thumbnail → atomic publish. Decode must bound image dimensions/bytes to avoid decompression bombs. Store orientation-aware thumbnails. Screenshots with prominent text may need OCR later; do not claim OCR-level exactness without OCR.

### Video pipeline (MVP, **visual** only)
Authorized path → ffprobe (duration, streams, codec) → segmentation policy → segment spans with millisecond boundaries → encode each segment using the model-supported API / properly sampled frames → thumbnail(s) → vector records with start/end → timestamp result. Use **fixed bounded windows as initial baseline** (candidate 6–12 s, VERIFY), compare against scene-aware windows on curated tests. For static lectures, scene detection alone is insufficient. Avoid one video→one embedding. Respect model context budgets and frame sampling. Probe variable frame rate/timebase and seek accuracy. Overlapping segments only if measurable retrieval benefit offsets cost. Time unit is integer `ms`; use FFmpeg rational timestamps where relevant, never assume fixed FPS.

The V1 source of video results is **visual evidence only**. Do not imply that exact spoken phrase retrieval already works. Prioritize file-level grouping with top 1–3 timestamp matches and a direct open/seek action. Failure to decode one video is not a global indexing failure.

### Audio, text documents, OCR and code (later)
- Audio: native audio embedding for semantic sounds/themes versus local STT + lexical for exact spoken phrases; compare local runtime cost. Voice transcripts require their own extractor / model version.
- PDF/text: structural extraction and page/chunk location; FTS + semantic embedding. Scanned PDFs require an optional OCR path.
- Source: filename/path and lexical baseline first, later symbol/chunk-aware indexing + `CodeRetrieval` prompt; measure exact symbol recall.
- Keep one `ContentUnit` abstraction but modality-specific preprocessors and scoring; avoid forcing all units into the same segmentation rules.

## 9. Search and ranking

**Query plan:** normalize Unicode for indexing/search appropriately; parse explicit filename/path terms/extension filters; run lexical FTS and exact/prefix filename retrieval without inference; run semantic query encoder only when model ready, with correct retrieval query prompt. Query embedding must use same fingerprint as selected corpus vectors. Combine candidates, deduplicate by `unit_id`, group by file. Maintain traceable match reasons (`filename`, `path`, `semantic_image`, `semantic_video_segment`, later `transcript`, `OCR`).

**Initial ranking:** exact filename/path matches ahead of loosely semantic results for identifier-shaped queries; otherwise use reciprocal-rank fusion (RRF) across retrievers, with a tunable and tested policy rather than arbitrary raw cosine/BM25 weight addition. Within a video rank segments independently then show best 1–3 distinct timestamps. Use extension/scope filters before ranking, tie-breaking deterministic. Track missing/stale files and don't knowingly open unavailable results.

**Evaluation:** version a consented/synthetic corpus manifest and annotated relevance judgments (`query_id`, query, expected unit/timestamp tolerance, query_class). Metrics by class (`visual`, `exact_filename`, `semantic_image`) with `Recall@K`, MRR, latency and index size. Test multilingual text and visual scenes if the intended audience requires it. Indexing speed and relevance are **separate** benchmarks; do not optimize FPS while ignoring recall.

## 10. UX contract

MVP screen states: empty/no folders; folder permission; model absent/downloading; queued/indexing/paused/completed/failed; searchable partial index; query busy; result list; video seek fallback if default player ignores timestamps. Provide progress by files **and** units where measurable, never falsely imply a time estimate. Results show path, modified time, type, image thumbnail/video scene frame and timestamp; one-click `Open`/`Reveal in Explorer` with validity checks. Keyboard-first query field; hotkey can follow later. No remote telemetry, no automatic access to Documents/Downloads without consent.

## 11. Error, security and privacy strategy

- Threat model includes malicious media, symlink escapes, local malware/other users reading index, stolen machine, sensitive generated thumbnails, model supply-chain compromise, unintended network transfers and exposed logs.
- Allowlist roots, path canonicalization **with re-check on use**, and reparse point avoidance for MVP. Never ask for administrator rights solely to index files. Least-privilege Tauri command allowlist and explicit filesystem permissions.
- Network blocked by policy during inference and search; one-time user-initiated model download permitted with source/version/checksum disclosure and cache validation. If no model: lexical still works.
- No raw user content, transcript, full paths or vectors in default logs. Local error codes and counts are sufficient; opt-in diagnostic logging must be separately reviewed.
- Bound file sizes, decompressed dimensions, decode time and worker memory; external processes receive arguments as arrays (no shell interpolation), have timeouts and are killable; avoid unsafe WebView remote content.
- Use UI errors that are actionable: `unsupported codec`, `permission denied`, `file changed while indexing`, `insufficient memory`, `model unavailable`.
- Local data-at-rest encryption is deferred pending realistic user-threat-model/key management review; do not market unencrypted indexes as encrypted/private from other processes with same account rights.

## 12. Test strategy and release gates

**Unit:** file classification, path scope rules, state transitions, fingerprint formatting, version invalidation, RRF ranking, time-range calculations.  
**Integration:** SQLite migrations & FTS, vectors exact topK, leases/restart, watcher events/debounce, ffprobe and segment boundaries, IPC timeout/restart.  
**E2E:** consented folder → index mixed images/videos → query → timestamped result → open, then rename/delete/update and restart.  
**Fault injection:** crash worker mid-job, corrupt video, read denied, drive removal, watcher event loss, malformed IPC, DB busy, old schema, stale model signature, cache cleanup.  
**Benchmarks:** recorded Windows CPU/GPU SKU, model/runtime versions, decode FPS, throughput, warm/cold query p95, vector lookup 10k/50k/250k, memory peaks, storage, recall.  
**Packaging gate before distributing:** clean Windows 11 x64 VM with no Python installed; launch, install/download selected model, index image + video, offline restart and search; verify FFmpeg/runtime/redistribution licenses, installer signature/update policy.

## 13. Roadmap and milestones

| Phase | Goal | Gate / exit |
|---|---|---|
| P0 | Repo foundation + Windows model/media/storage feasibility | **M0:** model works or approved fallback chosen; measurements and go/no-go documented |
| P1 | Durable core: SQLite, scope, jobs, lookup | **M1:** migrations/restart/scope tests green; filename search functional |
| P2 | Image semantic vertical slice | **M2:** image indexing and semantic retrieval with golden queries |
| P3 | Video segments + timestamp search | **M3:** deterministic media fixtures, segment results, measured precision |
| P4 | Desktop usable MVP, watcher and stability | **M4:** E2E on real Windows hardware, pause/restart/delete safety, measurable p95 |
| P5 | Audio/STT, documents and OCR (planned) | Independent feature gates/benchmarks |
| P6 | Code retrieval and ranking refinements (planned) | Eval improvement, no exact-match regressions |
| P7 | Packaging, signing, updates, deployment (planned) | Clean VM, licenses, release checklist |

`STEPS.md` is authoritative for task IDs, dependencies and current status. Do not implement all later phases at once.

## 14. Risk register

| Risk ID | Likelihood × impact | Early indicator | Mitigation / fallback |
|---|---|---|---|
| R-01 Python+PyTorch sidecar too large or won't package | High × High | Clean VM failure, giant footprint | P0 packaging spike; substitute native/ONNX runtime only if supports model correctly; change ADR |
| R-02 Visual video recall too low | Medium × High | Curated Recall@5 below gate | Tune segmentation, task prefixes, frame sampling, try alternate retriever/model; narrow promise |
| R-03 CPU-only too slow | High × Medium | Inference queue prolonged, high resource use | Batch, resource controls, defer large-video indexing, communicate supported hardware |
| R-04 Linear vector scan misses query latency | High × Medium | p95 at 50k too high | Benchmark sqlite-vec / USearch; migrate behind VectorIndex without losing source of truth |
| R-05 File watcher misses or misclassifies moves | High × Medium | E2E stale/missing rows | Idempotent reconciliation; volume availability state, no blind deletion |
| R-06 FFmpeg / codec licensing and distribution | Medium × High | Incompatible build/license | Track build flags, bundle appropriate binary, counsel before commercial release |
| R-07 Sensitive user data leaked by index/logs | Medium × High | Audit discovers paths/content in logs or cache permissions | Zero-content logging default; restricted app-data; threat model/privacy tests |
| R-08 Agent architecture drift | High × Medium | Duplicate services, undocumented new dependencies | STEPS one-task workflow; ADR review and test gates |

## 15. Open questions / verification queue

| Ref | Priority | Owner / phase | Resolution method |
|---|---|---|---|
| Q-001 | BLOCKING | Architect + implementer / P0 | Install latest compatible Python runtime on Windows 11; run official text, image and video examples; record pinned versions and output validity |
| Q-002 | BLOCKING | Architect / P0 | Compare CPU/bf16 GPU inference time and VRAM with real segments; define supported hardware tiers |
| Q-003 | BLOCKING | Architect / P0 | Bundle sidecar + dependencies as proof; measure install size and clean-VM startup |
| Q-004 | NEEDS BENCHMARK | Implementer / P0 | Compare dimensions 768/512/256; always renormalize, measure visual recall |
| Q-005 | NEEDS BENCHMARK | Implementer / P0 | Fixed windows vs scene-aware; compare timestamps/recall/encoding cost |
| Q-006 | NEEDS BENCHMARK | Implementer / P0 | Rust exact vector baseline vs sqlite-vec vs another embedded ANN on representative corpora; choose with measured p95 |
| Q-007 | NEEDS RESEARCH | Owner / pre-release | Verify model terms/policies, PyTorch/Tauri/FFmpeg distribution obligations |
| Q-008 | CAN DEFER | Owner / P5+ | Offline STT and OCR models, language needs, document parsers |
| Q-009 | CAN DEFER | Owner / P7 | Packaging MSI vs NSIS, update channel and code signing |

## 16. Technical debt register

| Debt | Accepted temporarily | Trigger to remove |
|---|---|---|
| TD-001 | Exact cosine scan of SQLite vector BLOBs | P0 50k/250k benchmark misses retrieval latency gate |
| TD-002 | Single inference worker / JSONL stdio | Profiling shows IPC or concurrency bottleneck |
| TD-003 | Fixed time windows for video | Golden temporal quality tests show systematic misses |
| TD-004 | No OCR or STT in visual MVP | Dedicated later phase and feasibility assessment |

## 17. Architecture change and evidence policy

ADR statuses are **PROPOSED** until validated by Phase 0; promote to **ACCEPTED** only with evidence and update this document and `STEPS.md` together. Significant changes require a short ADR: context → options → chosen approach → measurable reason → migration implications → reviewer approval. An agent must not silently change the model, on-disk vector format, process model or schema.

**Next action:** Complete `P0-T001` in `STEPS.md`; this document is a plan, not an assertion that any repository code exists.
