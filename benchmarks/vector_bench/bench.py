#!/usr/bin/env python3
"""P0-T009 benchmark runner: SQLite FTS5 + exact BLOB cosine vs sqlite-vec.

Synthetic, seeded, reproducible. Vectors are random L2-normalized float32 at
the P0-T006 contract dims -- they measure STORAGE machinery (latency, size,
correctness), never model relevance. Query-embedding inference latency is
excluded by design (queries are pre-generated stable vectors); only
lookup+ranking is timed.

Scales: 10k / 50k / 250k units (STEPS.md P0-T009). A 250k x 768 float32
matrix is ~768 MB; bench.py runs scales sequentially, frees each matrix
before the next, and preflights free RAM/disk (ctypes + shutil, stdlib).
A scale that fails preflight is reported BLOCKED (skipped with reason),
never faked.

Usage:
    python benchmarks/vector_bench/bench.py [--scales 10000 50000 250000]
        [--dim 768] [--queries 200] [--topk 10] [--seed 42]
        [--workdir <tempdir>] [--out docs/benchmarks/storage-search-results.json]

Benchmark DBs live under --workdir (default: OS temp) and are NOT committed
(R-24/R-26; *.db is git-ignored). Only the JSON report is written to docs/.

Exit 0 = benchmark completed (per-scale BLOCKED notes are data, not failure);
exit 1 = crash with E_* code.
"""

import argparse
import ctypes
import gc
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from store import (  # noqa: E402
    ExactStore,
    VecStore,
    gen_unit_vectors,
    p50_p95,
    store_sig,
)
from compat import make_sig  # noqa: E402  (cross-modal query sig, review A7)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO_ROOT / "docs" / "benchmarks" / "storage-search-results.json"

PLAN_LEXICAL_P95_MS = 150.0
PLAN_VECTOR_P95_MS = 750.0

# Preflight: peak resident set is ~2x one matrix (generator freed before the
# working set loads, but load_matrix transiently doubles) plus SQLite pages.
# A scale whose need exceeds free RAM/disk is reported BLOCKED, not faked.
MIN_FREE_DISK_BYTES = 4 * 1024**3


def fail(code, message):
    print(f"{code}: {message}", file=sys.stderr)
    sys.exit(1)


def free_ram_bytes():
    """Free physical RAM via Win32 GlobalMemoryStatusEx (stdlib ctypes).

    Uses the real MEMORYSTATUSEX layout (DWORD, DWORD, then 64-bit fields);
    returns None when the call is unavailable so callers treat it as unknown.
    """
    try:
        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32),
                        ("dwMemoryLoad", ctypes.c_uint32),
                        ("ullTotalPhys", ctypes.c_uint64),
                        ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]

        st = MemStatus()
        st.dwLength = ctypes.sizeof(MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys)
    except Exception:
        pass
    return None


def env_record():
    import numpy as np

    try:
        import sqlite_vec

        vec_version = "sqlite-vec-py-0.1.9"
        vec_path = sqlite_vec.loadable_path()
        # loadable_path() omits the platform suffix; resolve the real file.
        vec_dll_bytes = None
        for cand in (vec_path, vec_path + ".dll", vec_path + ".so"):
            if os.path.isfile(cand):
                vec_path = cand
                vec_dll_bytes = os.path.getsize(cand)
                break
    except Exception as exc:  # noqa: BLE001
        vec_version, vec_path, vec_dll_bytes = f"missing ({exc})", None, None
    try:
        cpu = os.cpu_count()
    except Exception:  # noqa: BLE001
        cpu = None
    return {
        "os": sys.platform,
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "numpy": np.__version__,
        "sqlite_vec": vec_version,
        "sqlite_vec_dll": vec_path,
        "sqlite_vec_dll_bytes": vec_dll_bytes,
        "cpu_count": cpu,
        "free_ram_bytes": free_ram_bytes(),
    }


def realistic_file_fixtures(n, seed):
    """Representative filename/path fixtures for FTS tests (lexical only).

    Deliberately ordinary desktop names (camera files, docs, unicode, spaces,
    versions) -- NOT label-bearing retrieval probes. No semantic claim.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    roots = ["C:/Users/test/Pictures", "D:/Media/Photos/2024", "D:/Media/Videos",
             "C:/Users/test/Documents", "D:/Archive"]
    exts = [".JPG", ".PNG", ".MP4", ".MOV", ".PDF", ".DOCX", ".WEBP", ".MKV"]
    words = ["beach", "birthday", "report", "scan", "meeting", "holiday",
             "invoice", "screenshot", "recording", "draft", "final", "backup"]
    uni = ["报告", "résumé", "café", "日本語", "über", "niño", "фото"]
    rows = []
    for i in range(n):
        kind = rng.integers(0, 100)
        if kind < 40:
            name = f"IMG_{int(rng.integers(1000, 9999))}{rng.choice(exts[:2])}"
        elif kind < 55:
            name = f"{rng.choice(words)}_{rng.choice(words)}_{int(rng.integers(1, 99))}{rng.choice(exts)}"
        elif kind < 65:
            name = f"{rng.choice(uni)}_{int(rng.integers(1, 999))}{rng.choice(exts)}"
        elif kind < 75:
            name = f"my file v{int(rng.integers(1, 9))} (copy){rng.choice(exts)}"
        else:
            name = f"PXL_20240{int(rng.integers(1, 9))}{int(rng.integers(10, 28))}_{int(rng.integers(100000, 999999))}{rng.choice(exts)}"
        path = f"{rng.choice(roots)}/{name}"
        rows.append((f"f{i:06d}", name, path))
    # Pinned fixtures the correctness checks rely on (always present).
    rows[0] = ("f000000", "IMG_4281.JPG", "C:/Users/test/Pictures/IMG_4281.JPG")
    rows[1] = ("f000001", "年度报告_2024.PDF", "C:/Users/test/Documents/年度报告_2024.PDF")
    rows[2] = ("f000002", "notes final (copy).DOCX", "D:/Archive/notes final (copy).DOCX")
    return rows


def run_fts_bench(workdir, seed):
    import numpy as np

    sig = store_sig(768)
    db = ExactStore(workdir / "fts.db", sig, create=True)
    rows = realistic_file_fixtures(2000, seed)
    t0 = time.perf_counter()
    db.insert_files(rows)
    build_s = time.perf_counter() - t0
    queries = (["IMG_4281", "report", "年度报告", "PXL_2024", "beach",
                "screenshot", "MP4", "final", "café", "2024"]
               * 20)[:200]
    lat, ok_count = [], 0
    for q in queries:
        t0 = time.perf_counter()
        found, status = db.fts_search(q, limit=10)
        lat.append((time.perf_counter() - t0) * 1000.0)
        if status == "ok" and found:
            ok_count += 1
    p50, p95 = p50_p95(lat)
    # Exact + prefix + invalid-syntax behavior (also covered in unit tests).
    exact = db.exact_name("IMG_4281.JPG")
    prefix = db.prefix_path("C:/Users/test/Pictures/IMG_42")
    bad, bad_status = db.fts_search('"unclosed phrase', limit=10)
    size_bytes = sum(p.stat().st_size for p in workdir.glob("fts.db*") if p.is_file())
    db.close()
    return {
        "n_files": len(rows),
        "build_s": build_s,
        "n_query_samples": len(lat),
        "p50_ms": p50,
        "p95_ms": p95,
        "queries_with_hits": ok_count,
        "lexical_budget_p95_ms": PLAN_LEXICAL_P95_MS,
        "within_budget": p95 <= PLAN_LEXICAL_P95_MS,
        "exact_IMG_4281_hits": len(exact),
        "prefix_hits": len(prefix),
        "invalid_syntax_status": bad_status,
        "invalid_syntax_rows": len(bad),
        "db_bytes": size_bytes,
    }


def db_size_bytes(workdir, stem):
    total = 0
    for p in workdir.glob(stem + "*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def run_vector_scale(workdir, n, dim, n_queries, topk, seed):
    import numpy as np

    sig = store_sig(dim)
    # Text-query signature: the measured path exercises the real contract
    # gate (shared space, differing modality pipelines), not a same-sig
    # shortcut (review A7).
    from store import PREP as _PREP

    qsig = make_sig("text-query", dim, prep=_PREP, segment="n/a")
    out = {"n": n, "dim": dim, "seed": seed, "topk": topk,
           "n_query_samples": n_queries}
    mat = gen_unit_vectors(n, dim, seed)
    unit_ids = [f"u{i:07d}" for i in range(n)]
    queries = gen_unit_vectors(n_queries, dim, seed + 10_000 + n)

    # -- exact baseline ------------------------------------------------------
    ex = ExactStore(workdir / f"exact_{n}.db", sig, create=True)
    t0 = time.perf_counter()
    ex.insert_embeddings(unit_ids, mat)
    out["exact_build_s"] = time.perf_counter() - t0
    out["exact_counts"] = ex.counts()

    # -- sqlite-vec candidate (inserted from the same matrix, then freed) ---
    vec = None
    try:
        vec = VecStore(workdir / f"vec_{n}.db", sig, create=True)
        t0 = time.perf_counter()
        vec.insert_embeddings(unit_ids, mat)
        out["vec_build_s"] = time.perf_counter() - t0
        out["vec_counts"] = vec.counts()
    except ValueError as exc:
        out["vec_status"] = f"unavailable: {exc}"
    # Free the generator matrix now: peak stays at one resident copy
    # (the query working set below), never generator + working set.
    del mat
    gc.collect()

    # Full-table fetch cost, sampled 5x back-to-back: iteration 1 is
    # near-cold, iterations 2-5 are warm OS/SQLite page cache (review A1 --
    # disclosed, not a cold-start claim). Steady-state rank cost is timed
    # separately below so a caching Rust baseline can compare either.
    fetch_samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        ids, full = ex.load_matrix()
        fetch_samples.append((time.perf_counter() - t0) * 1000.0)
        del ids, full
    gc.collect()
    fp50, fp95 = p50_p95(fetch_samples)
    out["exact_fetch_5x_p50_ms"] = fp50
    out["exact_fetch_5x_p95_ms"] = fp95
    ids, full = ex.load_matrix()
    out["exact_load_rows"] = len(ids)

    lat_rank = []
    first_exact = None
    lat_v, recalls, first_vec = [], [], None
    for qi in range(n_queries):
        q = queries[qi]
        t2 = time.perf_counter()
        res = ex.search_preloaded(q, ids, full, topk)
        lat_rank.append((time.perf_counter() - t2) * 1000.0)
        if qi == 0:
            first_exact = [u for u, _s in res]
        if vec is not None:
            t1 = time.perf_counter()
            got = vec.knn(q, topk, query_sig=qsig)
            lat_v.append((time.perf_counter() - t1) * 1000.0)
            ref = res
            got_ids = [u for u, _d in got]
            ref_ids = [u for u, _s in ref]
            inter = len(set(got_ids).intersection(ref_ids))
            recalls.append(inter / topk)
            if qi == 0:
                first_vec = got_ids
    p50r, p95r = p50_p95(lat_rank)
    out["exact_rank_only_p50_ms"] = p50r
    out["exact_rank_only_p95_ms"] = p95r
    # Like-for-like budget path (review B3/B4): full ExactStore.search()
    # end-to-end (gate + fetch + rank, same validation shape as vec knn),
    # 10 samples -- the verdict denominator, not the rank-only diagnostic.
    e2e = []
    for qi in range(min(10, n_queries)):
        t3 = time.perf_counter()
        ex.search(queries[qi], topk, query_sig=qsig)
        e2e.append((time.perf_counter() - t3) * 1000.0)
    pe50, pe95 = p50_p95(e2e)
    out["exact_e2e_n"] = len(e2e)
    out["exact_e2e_p50_ms"] = pe50
    out["exact_e2e_p95_ms"] = pe95
    out["vector_budget_p95_ms"] = PLAN_VECTOR_P95_MS
    out["exact_within_budget_e2e"] = pe95 <= PLAN_VECTOR_P95_MS
    ex.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    out["exact_db_bytes"] = db_size_bytes(workdir, f"exact_{n}.db")
    out["exact_bytes_per_unit"] = out["exact_db_bytes"] / n

    if vec is not None:
        p50v, p95v = p50_p95(lat_v)
        out["vec_p50_ms"] = p50v
        out["vec_p95_ms"] = p95v
        out["vec_within_budget"] = p95v <= PLAN_VECTOR_P95_MS
        out["vec_recall_at_k_vs_exact_mean"] = sum(recalls) / len(recalls)
        out["vec_recall_at_k_vs_exact_min"] = min(recalls)
        vec.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        out["vec_db_bytes"] = db_size_bytes(workdir, f"vec_{n}.db")
        out["vec_bytes_per_unit"] = out["vec_db_bytes"] / n

        # persistence across reopen (both backends, first query must repeat)
        ex.close()
        vec.close()
        ex2 = ExactStore(workdir / f"exact_{n}.db", sig, create=False)
        rep_exact = [u for u, _s in ex2.search(queries[0], topk, query_sig=qsig)]
        ex2.close()
        vec2 = VecStore(workdir / f"vec_{n}.db", sig, create=False)
        rep_vec = [u for u, _d in vec2.knn(queries[0], topk, query_sig=qsig)]
        vec2.close()
        out["persistence_exact_repeats"] = (rep_exact == first_exact)
        out["persistence_vec_repeats"] = (rep_vec == first_vec)

        # incremental update/delete on the exact store (reopen, mutate, verify)
        ex3 = ExactStore(workdir / f"exact_{n}.db", sig, create=False)
        add_ids = [f"ux{i:07d}" for i in range(200)]
        add_mat = gen_unit_vectors(200, dim, seed + 999_999)
        t0 = time.perf_counter()
        ex3.insert_embeddings(add_ids, add_mat)
        out["incr_insert_200_s"] = time.perf_counter() - t0
        drop_ids = unit_ids[:100]
        t0 = time.perf_counter()
        ex3.delete_units(drop_ids)
        out["incr_delete_100_s"] = time.perf_counter() - t0
        counts = ex3.counts()
        out["incr_counts"] = counts
        out["incr_expected"] = n + 200 - 100
        out["incr_count_ok"] = (counts["embeddings"] == n + 200 - 100)
        out["incr_deleted_absent"] = (ex3.get_vector(drop_ids[0]) is None)
        ex3.close()
    else:
        ex.close()
        out["persistence_exact_repeats"] = None
        out["persistence_vec_repeats"] = None

    # free the big arrays before the next scale (sequential, bounded memory)
    del queries
    try:
        del ids, full
    except NameError:
        pass
    gc.collect()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0-T009 storage benchmark.")
    parser.add_argument("--scales", type=int, nargs="+", default=[10000, 50000, 250000])
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if args.dim not in (768, 512, 256, 128):
        fail("E_DIM_POLICY", f"dim {args.dim} outside P0-T006 contract {768, 512, 256, 128}")
    store_sig(args.dim)  # policy gate now (fp16/dim violations fail fast)

    env = env_record()
    tmp = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="lf-vecbench-"))
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"workdir: {tmp} (benchmark DBs, not committed)")
    print(f"env: py={env['python']} sqlite={env['sqlite']} numpy={env['numpy']} "
          f"vec={env['sqlite_vec']} cpus={env['cpu_count']} "
          f"free_ram_GB={(env['free_ram_bytes'] or 0) / 1024**3:.1f}")

    results = {"task": "P0-T009", "report_version": 1, "env": env,
               "config": {"scales": args.scales, "dim": args.dim,
                          "queries": args.queries, "topk": args.topk,
                          "seed": args.seed},
               "scales": {}, "fts": None, "notes": []}
    free_disk = shutil.disk_usage(tmp).free
    results["env"]["free_disk_bytes_workdir"] = free_disk

    # FTS lexical bench first (small, fast).
    results["fts"] = run_fts_bench(tmp, args.seed)
    f = results["fts"]
    print(f"fts: n={f['n_files']} build={f['build_s']:.2f}s "
          f"p50={f['p50_ms']:.2f}ms p95={f['p95_ms']:.2f}ms "
          f"budget150ms={'PASS' if f['within_budget'] else 'MISS'} "
          f"db={f['db_bytes'] / 1024:.0f}KiB")

    for n in args.scales:
        # 250k x 768 needs ~0.77 GB matrix + index pages; gate it honestly.
        est_matrix = n * args.dim * 4
        if n >= 250000:
            need_ram = est_matrix * 2 + 256 * 1024**2
            ok_ram = env["free_ram_bytes"] is not None and \
                env["free_ram_bytes"] >= need_ram
            ok_disk = free_disk >= MIN_FREE_DISK_BYTES + est_matrix * 3
            if not (ok_ram and ok_disk):
                results["scales"][str(n)] = {
                    "status": "BLOCKED",
                    "reason": f"preflight: free_ram={env['free_ram_bytes']} "
                              f"(need ~{need_ram}), "
                              f"free_disk={free_disk} (need ~{est_matrix * 3}); "
                              "scale skipped, not extrapolated"}
                print(f"scale {n}: BLOCKED by preflight (see report)")
                continue
        print(f"scale {n} x {args.dim}: generating + benchmarking ...", flush=True)
        try:
            scale_out = run_vector_scale(tmp, n, args.dim, args.queries,
                                         args.topk, args.seed + n)
        except Exception as exc:  # noqa: BLE001 -- measured failure, not fake pass
            results["scales"][str(n)] = {"status": "FAILED",
                                         "error": f"{type(exc).__name__}: {exc}"[:300]}
            print(f"scale {n}: FAILED ({exc})")
            continue
        scale_out["status"] = "measured"
        results["scales"][str(n)] = scale_out
        print(f"scale {n}: exact build={scale_out['exact_build_s']:.1f}s "
              f"e2e p50/p95={scale_out['exact_e2e_p50_ms']:.1f}/"
              f"{scale_out['exact_e2e_p95_ms']:.1f}ms "
              f"(budget750={'PASS' if scale_out['exact_within_budget_e2e'] else 'MISS'}) "
              f"fetch5x p50/p95={scale_out['exact_fetch_5x_p50_ms']:.1f}/"
              f"{scale_out['exact_fetch_5x_p95_ms']:.1f}ms "
              f"rank-only p50/p95={scale_out['exact_rank_only_p50_ms']:.1f}/"
              f"{scale_out['exact_rank_only_p95_ms']:.1f}ms "
              f"db={scale_out['exact_db_bytes'] / 1024**2:.1f}MiB "
              f"({scale_out['exact_bytes_per_unit']:.0f}B/u)")
        if "vec_p50_ms" in scale_out:
            print(f"scale {n}: vec build={scale_out['vec_build_s']:.1f}s "
                  f"p50/p95={scale_out['vec_p50_ms']:.1f}/{scale_out['vec_p95_ms']:.1f}ms "
                  f"recall@k vs exact mean/min={scale_out['vec_recall_at_k_vs_exact_mean']:.4f}/"
                  f"{scale_out['vec_recall_at_k_vs_exact_min']:.4f} "
                  f"db={scale_out['vec_db_bytes'] / 1024**2:.1f}MiB "
                  f"persist_exact={scale_out['persistence_exact_repeats']} "
                  f"persist_vec={scale_out['persistence_vec_repeats']} "
                  f"incr_ok={scale_out.get('incr_count_ok')}")

    results["notes"] = [
        "Synthetic seeded vectors: storage machinery only, no semantic claim.",
        "Retrieval latency excludes query-embedding inference (pre-generated queries).",
        "Budget verdicts compare like-for-like full paths: ExactStore.search() "
        "end-to-end (gate+fetch+rank, n=10) vs VecStore.knn (gate+SQL, n=200). "
        "rank_only is an unchecked compute diagnostic, never a verdict.",
        "exact_fetch_5x is warm OS/SQLite page cache after iteration 1, not cold start.",
        "sqlite-vec 0.1.9 default L2 metric on L2-normalized vectors; "
        "ordering equals cosine (parity recall measured vs exact ground truth).",
        "ANN recall here is storage-correctness recall vs the exact baseline, "
        "NOT LocalFind semantic retrieval quality (P0-T007 suite, separate).",
        "Free-RAM preflight is Windows-only (ctypes GlobalMemoryStatusEx); "
        "elsewhere RAM reads unknown and the 250k scale reports BLOCKED.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"results: {args.out}")
    print("VECTOR BENCH PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
