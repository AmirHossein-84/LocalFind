# P0-T009 — SQLite FTS + vector backend benchmark (prototype)

Phase-0 benchmark code, **not production code**. Measures storage/retrieval
machinery with synthetic seeded vectors — never model relevance.

## Files

| Path | Purpose |
|---|---|
| `store.py` | `ExactStore` (metadata + FTS5 + float32 BLOB cosine) and `VecStore` (sqlite-vec vec0), both behind the P0-T006 signature gate |
| `bench.py` | Scale runner: 10k / 50k / 250k units, FTS lexical bench, JSON report |
| `test_vector_bench.py` | Stdlib correctness suite (fast, temp dirs only) |
| `__init__.py` | Package marker |

## Quick start

```powershell
# Correctness (fast, no model, no network):
python -m unittest benchmarks.vector_bench.test_vector_bench -v

# Small smoke of the runner (~1-2 min):
python benchmarks/vector_bench/bench.py --scales 1000 --queries 20 --dim 768

# Full P0-T009 run (sequential; 250k scale preflights RAM/disk):
python benchmarks/vector_bench/bench.py
```

Requires system Python with `numpy` + `sqlite-vec==0.1.9`
(`benchmarks/.venv` has torch but not these; the bench uses no model).
Benchmark DBs go to OS temp (never committed); the report lands at
`docs/benchmarks/storage-search-results.json`.

## Design notes

- Vectors are `np.random.default_rng(seed)` L2-normalized float32 — the
  storage path (BLOB pack/unpack, SQL, ranking) is real; the *content* is
  random, so no semantic claim is possible by construction.
- Retrieval latency excludes query-embedding inference (queries are
  pre-generated); `exact_fetch_rank_*` (SQLite fetch + numpy rank) and
  `exact_rank_only_*` (numpy rank on a loaded matrix) are reported
  separately so a future Rust baseline can compare against either.
- sqlite-vec 0.1.9 exposes only the default L2 metric
  (`distance_type=cosine` is rejected by its vec0 parser — verified
  2026-10-09). On L2-normalized vectors L2 order == cosine order; the
  parity test (`recall@k vs exact == 1.0`) proves it per run.
- FTS fixtures are ordinary desktop filenames (lexical behavior only).
