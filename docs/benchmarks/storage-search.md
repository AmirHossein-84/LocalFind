# P0-T009 — SQLite FTS + vector backend benchmark

**Status (2026-10-09): MEASURED.** Prototype evidence for the V1 storage
decision. Machine-readable results: `docs/benchmarks/storage-search-results.json`
(768-d canonical scales + 256-d appendix). Runner + stores +
correctness suite: `benchmarks/vector_bench/` (28 unit tests green).

> Scope: storage/retrieval **machinery** (latency, size, correctness,
> persistence). Vectors are seeded random unit vectors at the P0-T006
> contract dims — this benchmark **cannot and does not** measure model
> relevance. Recall below is storage-correctness recall of the tested
> `vec0` flat-KNN scan against the exact baseline, NOT LocalFind
> semantic quality (P0-T007 suite, separate). The tested sqlite-vec
> 0.1.9 `vec0` configuration performs an exact brute-force KNN scan
> (not approximate ANN); true approximate ANN was not evaluated here.
> No ADR is accepted by this task; all PLAN.md ADRs stay **PROPOSED**
> pending M0 review (P0-T012).

## Method (reproducible)

- Backends: `ExactStore` (metadata + FTS5 + float32 BLOB exact cosine,
  ADR-004 baseline) vs `VecStore` (sqlite-vec 0.1.9 `vec0` exact flat-KNN
  scan, default L2 metric) over identical unit ids. Both carry the full P0-T006 signature;
  inserts validate dim/finite/unit-norm, queries pass the shared-space gate
  with a **text-query** signature against image vectors (the real
  cross-modal gate, not a same-sig shortcut).
- Corpora: synthetic 10k / 50k / 250k units @768-d (STEPS.md P0-T009),
  seed `42 + n`; 200 pre-generated stable queries per scale
  (query-embedding inference excluded by design — lookup+ranking only).
- Metrics per scale: bulk build time, full-table fetch (5x, warm cache
  after iteration 1 — disclosed, not a cold-start claim), rank-only on a
  loaded matrix (200 samples, unchecked compute diagnostic),
  **end-to-end `search()`** (gate+fetch+rank, 10 samples — the exact-side
  budget denominator), vec `knn()` (gate+SQL, 200 samples), p50/p95
  throughout, on-disk bytes, persistence across reopen, +200/-100
  incremental check, vec-vs-exact recall@10 on all 200 queries.
- FTS: 2,000 representative desktop filenames/paths (camera files, docs,
  unicode, spaces, versions — lexical behavior only, no semantic claim),
  200 queries, plus exact/prefix/unicode/invalid-syntax/rename/delete
  checks (also covered in the unit suite).
- Budgets (PLAN §2, targets not promises): lexical p95 ≤ 150 ms,
  vector lookup + ranking p95 ≤ 750 ms.
- Benchmark DBs lived in OS temp (never committed; `*.db` is git-ignored).
  Scales ran sequentially with the working matrix freed between scales.

## Environment

Win11 Pro build 26300 x64 · i7-13620H (16 logical) · 16 GB RAM ·
RTX 3050 6GB driver 610.74 · Python 3.12.10 · SQLite 3.49.1 ·
numpy 2.4.6 · **sqlite-vec 0.1.9** (`vec0.dll`, **289,280 bytes**) ·
`cargo`/`cl.exe` absent in this shell (Rust-side static-link build
**unverifiable here** — remaining experiment, not a claim).

Two 768-d runs are reported because host memory pressure moved the
numbers 2–4x — itself a finding (see Limits):

| Run | Code | Free RAM at start | Free disk (workdir vol) |
|---|---|---|---|
| A (first) | pre-review prototype, same-modality query sig, no e2e metric | 1.9 GB | 38.3 GB |
| B (canonical, in JSON) | final reviewed code, text-query sig, +e2e verdicts | 2.5 GB | 38.3 GB |

Appendix: 250k @256-d, final code, 3.3 GB free.

## FTS results (all runs identical)

2000 files, build 0.02–0.03 s; query p50 **0.07 ms**, p95 **0.25–0.30 ms**
vs the 150 ms lexical budget: **PASS with ~500x headroom** (200 samples).
180/200 probe queries returned hits. Exact `IMG_4281.JPG` → 1 row;
prefix `C:/Users/test/Pictures/IMG_42` → 2 rows; unicode `年度报告`
found under the default FTS5 tokenizer; 5/5 malformed MATCH strings
(`"unclosed`, `(((`, `NEAR(`, `*`, `OR OR OR`) returned zero rows with
typed `invalid-syntax` status instead of raising; LIKE wildcards in
prefixes escaped (`100%` matches literally); rename/delete propagate
through triggers; `rebuild` preserves results; failed transaction rolls
back to the prior state. DB: **434 KiB** for 2,000 files (~222 B/file).

## Vector results @768-d

| Scale | Exact build | Exact e2e p50/p95 (n=10) | Exact fetch 5x | Exact rank-only (n=200) | Exact DB (B/u) | Vec build | Vec knn p50/p95 (n=200) | Vec recall@10 mean/min | Vec DB (B/u) |
|---|---|---|---|---|---|---|---|---|---|
| 10k B | 0.3 s | 54.1 / **54.8 PASS** | 48.9 / 49.2 | 4.2 / 4.8 | 39.5 MiB (4,137) | 1.6 s | 21.3 / **22.2 PASS** | 1.0 / 1.0 | 30.7 MiB (3,220) |
| 50k B | 1.5 s | 298 / **306 PASS** | 247.5 / 248.2 | 24.5 / 27.4 | 197.1 MiB (4,134) | 8.4 s | 102.8 / **109.0 PASS** | 1.0 / 1.0 | 150.3 MiB (3,153) |
| 250k B | 96.1 s | 1420 / **1824 MISS** | 1296 / 3068 | 155 / 174 | 985.6 MiB (4,134) | 112.3 s | 514 / **547 PASS** | 1.0 / 1.0 | 751.6 MiB (3,153) |
| 250k A | 114.0 s | n/a (no e2e metric) | 6534 / 7142 | 559 / 607 | 985.6 MiB (4,134) | 145.8 s | 1896 / 1967 MISS | 0.9995 / 0.90 | 751.6 MiB (3,153) |

Reading:

- **≤50k units (the PLAN MVP tier: ≥10k files / ≥50k units) both backends
  pass.** Exact e2e 306 ms and vec 109 ms at 50k leave 2.4x / 6.9x
  headroom under 750 ms. FTS passes everywhere.
- **At 250k, exact end-to-end misses (1.8 s) while vec0 passes
  (0.55 s)** — a 3.3x gap. Exact rank-only (0.17 s) shows the miss is
  fetch-dominated: re-reading 786 MB of BLOBs per query at ~120–600 MB/s
  (host-dependent). A matrix-caching exact baseline would pass, but that
  cache is precisely what a dedicated index owns — the measurement
  supports ADR-004's split (SQLite = source of truth, index = accelerator).
- **Run A vs B is the memory-pressure lesson.** Same engine, same data:
  with 1.9 GB free, vec0 missed the budget (1.97 s) and exact rank was
  3.6x slower; with 2.5 GB free both improved dramatically. Latency
  claims for this stack are meaningless without free-RAM context, and
  the 16 GB dev machine is already the floor — low-RAM tiers need their
  own measurement before any hardware promise.
- **Recall nuance (run A 0.9995/0.90).** A one-off probe on run A's DBs
  showed the single swapped query had an exact rank10–rank11 cosine gap
  of **3e-08** (float32-epsilon tie between L2-distance and cosine
  accumulation order on random vectors); the "missed" unit ranked 11th
  in vec0. Not an index bug — confirmed by run B reproducing 1.0/1.0 on
  the same corpus size. On real embeddings top-K gaps are orders of
  magnitude larger. Boundary: exact-tie inputs (literal duplicate
  vectors) order arbitrarily in vec0 (no multi-key `ORDER BY` in 0.1.9);
  production dedupe of identical vectors sidesteps it.

## Dimension appendix (250k @256-d, run C)

Exact build **2.5 s** (vs 96 s @768 — the 768 build cost was
memory-pressure-bound, not intrinsic); exact e2e 819/859 ms (**MISS**,
fetch-dominated again); rank-only 146/165 ms; vec **172/181 ms (PASS)**;
recall 1.0/1.0; **1,391 B/u** exact, ~1,090 B/u vec. Storage and vec
latency scale ~linearly with dim; the rank-only gap barely moves
(Python `sorted()` over 250k scores dominates, not the matmul — a Rust
baseline would shift this floor, disclosed). Dimension choice still
needs P0-T007 **quality** data (deferred since P0-T006); storage
evidence alone must not pick 256-d.

## Update/delete/persistence/correctness

- Incremental (every scale, both 768 runs + 256): +200 inserts
  0.004–0.011 s, −100 deletes <0.001 s, counts exact
  (n+200−100), deleted ids read back absent. No rebuild needed for
  small deltas.
- Persistence: close → reopen → identical top-K on both backends at
  every scale (6/6 true). Signature-mismatched reopen (wrong dim,
  revision, prep, or dtype) refused with `E_MISMATCH`; fp16 and
  non-contract dims rejected at signature construction; zero/NaN/Inf/
  unnormalized vectors rejected at insert; `topk <= 0` rejected.
- Correctness suite: **28/28 green**
  (`python -m unittest benchmarks.vector_bench.test_vector_bench`),
  covering FTS exact/prefix/unicode/invalid-syntax/rename/delete/
  rebuild, top-K vs brute force, deterministic ties, reopen, missing/
  deleted, dim/sig/value gates, transaction atomicity, vec parity,
  vec upsert/delete, and contract preservation.
- Independent review of the prototype (read-only subagent) filed 5
  blocking findings (B1–B5); all fixed and regression-tested:
  rowid-stable vec upserts (vec0 rejects `INSERT OR REPLACE` — upsert
  is DELETE+INSERT, verified), storage-identity (not just space) gate
  on reopen, like-for-like e2e budget verdicts, symmetric validation
  on both timed paths, `E_MISSING` on index drift. Two reviewer
  suggestions were **tested and rejected by the engine** (documented in
  code): multi-key `ORDER BY distance, rowid` and `distance_type=cosine`
  are both refused by sqlite-vec 0.1.9's vec0 parser.

## Backend comparison

| Axis | Exact BLOB baseline | sqlite-vec 0.1.9 (vec0, brute-force) |
|---|---|---|
| Query latency | PASS ≤50k (306 ms e2e); MISS 250k e2e (1.8 s); rank-only 174 ms | PASS at all scales incl. 250k (547 ms @768, 181 ms @256) |
| Correctness | Ground truth by construction; deterministic `(−cos, unit_id)` ties | Parity recall 1.0 vs exact (200 queries × 4 configs); float-tie boundary disclosed; ties engine-ordered |
| Storage | 4,134 B/u @768 (BLOB + PK); 1,391 B/u @256 | 3,153 B/u @768 (−24%); ~1,090 B/u @256; single-file co-location with metadata possible |
| Build (250k) | 96 s (pressure-sensitive) | 112 s (pressure-sensitive); DELETE+INSERT upserts |
| Updates | SQL-native, transactional, atomic with metadata | Same file/transaction possible; no OR REPLACE (delete+insert) |
| Persistence | Reopen-identical; sig-gated | Reopen-identical; sig-gated; `E_MISSING` drift guard |
| Integration | Zero new deps; pure SQL + host matmul | One pinned dep; `vec0` SQL contained behind `VectorIndex`; pre-1.0 API (breaking-change warning upstream) |
| Deployment (Win) | Trivial | Prebuilt `win_amd64` wheel verified; 282 KiB DLL; Python loadable-extension proven; **Rust static-link (`sqlite-vec` crate + `cc`) NOT built here (no MSVC/cargo in this shell)** |
| Licensing | n/a (own code) | MIT/Apache-2.0 dual (PyPI + crates.io + repo); commercial bundling needs P0-T011 notice inventory — no clearance declared here |
| Reliability risks | Fetch-bound at scale; Python-sort floor in prototype only | Single maintainer + pre-1.0 churn (pin `==0.1.9`); L2-only metric (fine for unit vectors); no quantized index evaluated |

USearch (Apache-2.0, Win wheels + Rust crate) and Faiss (MIT, no Rust
story) were assessed read-only (R-21): both deferred unless
quantized vec0 fails — neither is needed for the ≤50k V1 tier, and
both add a side-index sync burden ADR-004 already warns about. No
server backend was considered (STEPS.md forbids it).

## Recommendation (evidence, not an ADR acceptance)

1. **V1: exact float32 BLOBs as the transactional source of truth +
   pinned `sqlite-vec == 0.1.9` as the embedded accelerated index,
   both behind the `VectorIndex` port** (ADR-004's shape). Exact owns
   correctness/atomicity; vec0 owns sub-second lookup at scale. Either
   side alone is weaker: exact misses 250k e2e, vec0 is pre-1.0 and
   cannot be the file-state authority.
2. **TD-001 fires for the 250k tier only**: exact e2e misses the 750 ms
   gate at 250k in both runs — the "replace linear scan" condition is
   met there and only there. ≤50k stays on either backend.
3. **Do not pick a dimension from this report** (storage scales
   linearly; quality decides — still deferred). Do not promise 250k
   at 768-d within 750 ms on 16 GB hosts without the follow-ups below.

## Remaining experiments / risks (not starting them here)

- R1: Rust-side static link of the `sqlite-vec` crate (`cc` compile)
  needs MSVC Build Tools + cargo — **blocked in this shell** (`cl.exe`
  and `cargo` absent); required before P2-T006, not claimed here.
- R2: Quantized/partitioned vec0 index at 250k (recall vs exact +
  latency) — unevaluated; USearch HNSW is the fallback if it fails.
- R3: bf16-at-rest (prototype is fp32-only by assertion); cold-start
  fetch on target hardware; single-writer contention under concurrent
  indexing (not benchmarked); per-backend RSS profiling (only
  arithmetic working-set estimates here: ~2x matrix + pages).
- R4: P0-T011 license inventory must record sqlite-vec 0.1.9
  (MIT/Apache-2.0, upstream `github.com/asg017/sqlite-vec`,
  single-maintainer bus factor) before any commercial bundling claim.
- Host caveat: all 250k timings are host-conditioned (1.9–3.3 GB free
  of 16 GB); low-RAM tiers unmeasured.

## Provenance

Runner: `python benchmarks/vector_bench/bench.py` (scales 10k/50k/250k,
dim 768, 200 queries, top-10, seed 42) + `--scales 250000 --dim 256`
appendix; 28/28 unit tests; independent read-only review (5 blocking
findings fixed, 2 suggestions engine-rejected). No model, no network,
no user data; DBs in OS temp, uncommitted. No remote server added.
M0 remains **BLOCKED** (retrieval-quality gate unresolved, ARQ-001);
this report is input to P0-T012, not a go-decision.
