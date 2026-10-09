#!/usr/bin/env python3
"""P0-T009 storage prototype: SQLite FTS5 + float32 BLOB exact cosine + sqlite-vec.

Phase-0 benchmark code only (NOT production code). Demonstrates the ADR-003 /
ADR-004 shape against synthetic, seeded, L2-normalized vectors at the
P0-T006 contract dimensions -- it measures storage/retrieval machinery, NOT
model relevance. No label-leaking filenames are used as semantic evidence
(anywhere here vectors are random; filename fixtures are lexical-only).

Layout per store file:
  files(file_id TEXT PK, name TEXT, path TEXT)          -- metadata
  files_fts (FTS5 external-content over name/path)       -- lexical index
  embeddings(unit_id TEXT PK, dim INT, vec BLOB)         -- exact baseline
  meta(k TEXT PK, v TEXT)                                -- index signature row
  [vec stores] vec_items (vec0 virtual table) + vec_ids -- accelerated index

Compatibility: every store carries the full P0-T006 signature
(benchmarks/compat.py). Inserts validate dim/finite/unit-norm; queries pass
the shared-space gate before scoring. sqlite-vec 0.1.9 supports only the
default L2 metric (``distance_type=cosine`` is rejected by its vec0 parser);
on L2-normalized vectors L2 ordering == cosine ordering, which the parity
test proves empirically rather than asserting.

Requires: stdlib + numpy. sqlite_vec is optional (VecStore raises
VecUnavailable with an actionable message when missing).
"""

import math
import sqlite3
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compat import (  # noqa: E402
    check_compatible,
    check_space_compatible,
    check_vector_values,
    make_sig,
)

PREP = "p0t009-v1"
NORM_TOL = 1e-4

VEC_UNAVAILABLE = "E_VEC_UNAVAILABLE"


def store_sig(dim, dtype="fp32"):
    """Full storage-identity signature for a benchmark index group.

    Prototype storage is fp32-only (BLOB bytes and vec0 float[] carry no
    dtype tag); non-fp32 requests are rejected here rather than silently
    stored as fp32 (review A5). bf16-at-rest is an unevaluated option.
    """
    if dtype != "fp32":
        raise ValueError(
            f"E_DTYPE_STORE: prototype stores fp32 only, got {dtype!r} "
            "(bf16-at-rest unevaluated; see storage-search.md)")
    return make_sig("image", dim, dtype=dtype, prep=PREP, segment="n/a")


def unpack_f32(blob, dim):
    import numpy as np

    arr = np.frombuffer(blob, dtype=np.float32).copy()
    if arr.shape != (dim,):
        raise ValueError(f"E_DIM: blob holds {arr.shape} != ({dim},)")
    return arr


def gen_unit_vectors(n, dim, seed):
    """Deterministic L2-normalized float32 rows (seeded RNG, no model)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    # astype (not dtype= kwarg) for older-numpy tolerance; env pins numpy 2.x.
    mat = rng.standard_normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    mat = (mat / norms).astype(np.float32)
    if not bool(np.all(np.isfinite(mat))):
        raise ValueError("E_NONFINITE: generator produced non-finite values")
    return mat


def check_matrix_fast(mat, dim):
    """Vectorized equivalent of compat.check_vector_values per row.

    Same tolerances (finite, nonzero, |norm-1| <= 1e-4); raises E_* codes.
    Used for bulk inserts where per-row Python checks would dominate timing;
    the scalar gate is still unit-tested on small data (test_vector_bench).
    """
    import numpy as np

    if mat.shape[1] != dim:
        raise ValueError(f"E_DIM: matrix width {mat.shape[1]} != {dim}")
    if not bool(np.all(np.isfinite(mat))):
        raise ValueError("E_NONFINITE: NaN or Inf present")
    norms = np.linalg.norm(mat, axis=1)
    if bool(np.any(norms == 0.0)):
        raise ValueError("E_ZERO_NORM: zero vector")
    worst = float(np.max(np.abs(norms - 1.0)))
    if worst > NORM_TOL:
        raise ValueError(f"E_NORM: max |norm-1| {worst:.6f} > {NORM_TOL}")
    return True


def percentile(sorted_vals, pct):
    if not sorted_vals:
        raise ValueError("E_EMPTY: no samples")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def p50_p95(samples):
    s = sorted(samples)
    return percentile(s, 50), percentile(s, 95)


def _connect(path, load_vec=False):
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    if load_vec:
        try:
            import sqlite_vec
        except ImportError as exc:
            con.close()
            raise ValueError(
                f"{VEC_UNAVAILABLE}: sqlite_vec package missing ({exc}); "
                "install sqlite-vec==0.1.9 or skip the vec backend"
            ) from exc
        con.enable_load_extension(True)
        con.load_extension(sqlite_vec.loadable_path())
    return con


_FTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
  file_id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts
  USING fts5(name, path, content='files', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
  INSERT INTO files_fts(rowid, name, path) VALUES (new.rowid, new.name, new.path);
END;
CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
  INSERT INTO files_fts(files_fts, rowid, name, path)
    VALUES('delete', old.rowid, old.name, old.path);
END;
CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files BEGIN
  INSERT INTO files_fts(files_fts, rowid, name, path)
    VALUES('delete', old.rowid, old.name, old.path);
  INSERT INTO files_fts(rowid, name, path) VALUES (new.rowid, new.name, new.path);
END;
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


def _write_meta(con, sig):
    import json

    con.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('sig', ?)",
                (json.dumps(sig, sort_keys=True),))
    con.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('prep', ?)", (PREP,))


def _read_sig(con):
    import json

    row = con.execute("SELECT v FROM meta WHERE k='sig'").fetchone()
    if row is None:
        raise ValueError("E_META: store has no signature row (uninitialized?)")
    return json.loads(row[0])


def like_escape(term):
    """Escape %, _, and the escape char for a LIKE prefix predicate."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class ExactStore:
    """SQLite metadata + FTS5 + float32 BLOB exact-cosine baseline (ADR-004)."""

    def __init__(self, path, sig, create=True):
        self.path = Path(path)
        self.sig = dict(sig)
        self.dim = sig["dim"]
        self.con = _connect(self.path)
        if create:
            self.con.executescript(_FTS_SCHEMA)
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS embeddings("
                "unit_id TEXT PRIMARY KEY, dim INT NOT NULL, vec BLOB NOT NULL)")
            _write_meta(self.con, self.sig)
            self.con.commit()
        else:
            try:
                stored = _read_sig(self.con)
                # Storage-identity gate (review B2): a reopened file is one
                # index group; prep/modality/segment must match exactly, not
                # just the scoring space. Per-query scoring still uses the
                # narrower space gate (search/knn), where text queries
                # legitimately meet image/video vectors.
                check_compatible(self.sig, stored)
            except Exception:
                self.con.close()
                raise

    def close(self):
        self.con.commit()
        self.con.close()

    # -- writes (single owner-writer, explicit transactions) -----------------
    def insert_files(self, rows):
        with self.con:
            self.con.executemany(
                "INSERT OR REPLACE INTO files(file_id, name, path) VALUES(?,?,?)",
                rows)

    def insert_embeddings(self, unit_ids, mat, chunk=10_000):
        import numpy as np

        mat = np.asanyarray(mat, dtype=np.float32)
        if mat.shape != (len(unit_ids), self.dim):
            raise ValueError(
                f"E_DIM: matrix {mat.shape} != (n={len(unit_ids)}, dim={self.dim})")
        check_matrix_fast(mat, self.dim)
        # One transaction, chunked executemany: bounded transient memory for
        # large bulk imports (250k rows must not build one giant arg list).
        with self.con:
            for lo in range(0, len(unit_ids), chunk):
                hi = min(lo + chunk, len(unit_ids))
                self.con.executemany(
                    "INSERT OR REPLACE INTO embeddings(unit_id, dim, vec) VALUES(?,?,?)",
                    [(uid, self.dim, mat[i].tobytes())
                     for i, uid in enumerate(unit_ids[lo:hi], start=lo)])

    def delete_units(self, unit_ids):
        with self.con:
            self.con.executemany("DELETE FROM embeddings WHERE unit_id=?",
                                 [(u,) for u in unit_ids])

    def delete_files(self, file_ids):
        with self.con:
            self.con.executemany("DELETE FROM files WHERE file_id=?",
                                 [(f,) for f in file_ids])

    def rebuild_fts(self):
        with self.con:
            self.con.execute("INSERT INTO files_fts(files_fts) VALUES('rebuild')")

    # -- reads ----------------------------------------------------------------
    def counts(self):
        n_files = self.con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        n_vec = self.con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {"files": n_files, "embeddings": n_vec}

    def get_vector(self, unit_id):
        row = self.con.execute("SELECT vec, dim FROM embeddings WHERE unit_id=?",
                               (unit_id,)).fetchone()
        if row is None:
            return None
        blob, dim = row
        if dim != self.dim:
            raise ValueError(f"E_DIM: stored dim {dim} != store dim {self.dim}")
        return unpack_f32(blob, self.dim)

    def load_matrix(self):
        """Fetch the full BLOB table into a preallocated matrix.

        Preallocation (COUNT first, fill by index) avoids the transient 2x
        peak of a fetch-all + np.stack on large tables.
        """
        import numpy as np

        n = self.con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        if n == 0:
            return [], np.zeros((0, self.dim), dtype=np.float32)
        mat = np.empty((n, self.dim), dtype=np.float32)
        ids = [None] * n
        for i, (uid, blob) in enumerate(
                self.con.execute("SELECT unit_id, vec FROM embeddings ORDER BY unit_id")):
            ids[i] = uid
            mat[i] = unpack_f32(blob, self.dim)
        return ids, mat

    def search(self, query_vec, topk, query_sig=None):
        """Exact cosine top-K with deterministic (score desc, unit_id) order.

        Full production-shaped path: signature gate + dim/norm validation,
        then fetch + rank. Timed end-to-end samples use this (review B3/B4).
        """
        import numpy as np

        if topk <= 0:
            raise ValueError(f"E_TOPK: topk must be >= 1, got {topk}")
        if query_sig is not None:
            check_space_compatible(query_sig, self.sig)
        q = np.asanyarray(query_vec, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"E_DIM: query dim {q.shape} != store dim {self.dim}")
        check_vector_values([float(v) for v in q], self.dim)
        ids, mat = self.load_matrix()
        if not ids:
            return []
        scores = mat @ q
        order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
        return [(ids[i], float(scores[i])) for i in order[:topk]]

    def search_preloaded(self, query_vec, ids, mat, topk):
        """Rank against an already-loaded matrix (compute-only diagnostic).

        Explicitly unchecked: no signature/dim/norm gates (review B4). It
        measures numpy compute, never the production query path; budget
        verdicts must use search()/knn(), which both validate.
        """
        import numpy as np

        q = np.asanyarray(query_vec, dtype=np.float32)
        scores = mat @ q
        order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
        return [(ids[i], float(scores[i])) for i in order[:topk]]

    # -- lexical ---------------------------------------------------------------
    def fts_search(self, match_query, limit=10):
        """FTS5 MATCH with BM25 rank; returns (rows, status).

        status is 'ok' or 'invalid-syntax' (typed, never a crash): an
        unparseable MATCH expression yields no rows instead of an exception.
        """
        try:
            cur = self.con.execute(
                "SELECT f.file_id, f.name, f.path, rank FROM files_fts "
                "JOIN files f ON f.rowid = files_fts.rowid "
                "WHERE files_fts MATCH ? ORDER BY rank LIMIT ?",
                (match_query, limit))
            return cur.fetchall(), "ok"
        except sqlite3.OperationalError:
            return [], "invalid-syntax"

    def exact_name(self, name):
        return self.con.execute(
            "SELECT file_id, name, path FROM files WHERE name=?", (name,)).fetchall()

    def prefix_path(self, prefix, limit=10):
        esc = like_escape(prefix) + "%"
        return self.con.execute(
            "SELECT file_id, name, path FROM files WHERE path LIKE ? ESCAPE '\\' "
            "ORDER BY path LIMIT ?", (esc, limit)).fetchall()


class VecStore:
    """sqlite-vec vec0 index over the same unit ids (accelerated candidate).

    Vec index only: unit_id <-> rowid mapping plus the vec0 table.
    Metadata/FTS helpers live on ExactStore (review A4); a production
    co-located design would carry both in one file, which the persistence
    test exercises by reopening each side independently.
    """

    def __init__(self, path, sig, create=True):
        self.path = Path(path)
        self.sig = dict(sig)
        self.dim = sig["dim"]
        self.con = _connect(self.path, load_vec=True)
        if create:
            self.con.executescript(_FTS_SCHEMA)
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS vec_ids("
                "rowid INTEGER PRIMARY KEY AUTOINCREMENT, unit_id TEXT UNIQUE NOT NULL)")
            self.con.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING "
                f"vec0(rowid INTEGER PRIMARY KEY, embedding float[{self.dim}])")
            _write_meta(self.con, self.sig)
            self.con.commit()
        else:
            try:
                stored = _read_sig(self.con)
                # Storage-identity gate (review B2): a reopened file is one
                # index group; prep/modality/segment must match exactly, not
                # just the scoring space. Per-query scoring still uses the
                # narrower space gate (search/knn), where text queries
                # legitimately meet image/video vectors.
                check_compatible(self.sig, stored)
            except Exception:
                self.con.close()
                raise

    def close(self):
        self.con.commit()
        self.con.close()

    def insert_files(self, rows):
        with self.con:
            self.con.executemany(
                "INSERT OR REPLACE INTO files(file_id, name, path) VALUES(?,?,?)",
                rows)

    def insert_embeddings(self, unit_ids, mat, chunk=10_000):
        import numpy as np

        mat = np.asanyarray(mat, dtype=np.float32)
        if mat.shape != (len(unit_ids), self.dim):
            raise ValueError(
                f"E_DIM: matrix {mat.shape} != (n={len(unit_ids)}, dim={self.dim})")
        check_matrix_fast(mat, self.dim)
        # Explicit rowids keep vec_ids <-> vec_items aligned and allow
        # chunked executemany (no per-row round trips) in one transaction.
        # Rowids are STABLE across upserts (review B1): re-inserting an
        # existing unit_id reuses its rowid so vec_items never orphans.
        base = self.con.execute("SELECT COALESCE(MAX(rowid), 0) FROM vec_ids").fetchone()[0]
        existing = {u: r for r, u in
                    self.con.execute("SELECT rowid, unit_id FROM vec_ids").fetchall()}
        with self.con:
            for lo in range(0, len(unit_ids), chunk):
                hi = min(lo + chunk, len(unit_ids))
                rows = []
                for i in range(lo, hi):
                    uid = unit_ids[i]
                    if uid in existing:
                        rows.append(existing[uid])
                    else:
                        base += 1
                        existing[uid] = base
                        rows.append(base)
                # vec0 virtual tables reject INSERT OR REPLACE (UNIQUE
                # constraint on the PK); upsert is DELETE + INSERT while the
                # vec_ids row keeps its stable rowid. Verified 2026-10-09
                # against sqlite-vec 0.1.9 (see test_vec_upsert_*).
                self.con.executemany("DELETE FROM vec_items WHERE rowid=?",
                                     [(r,) for r in rows])
                self.con.executemany(
                    "INSERT INTO vec_items(rowid, embedding) VALUES(?,?)",
                    [(r, mat[i].tobytes())
                     for i, r in enumerate(rows, start=lo)])
                self.con.executemany(
                    "INSERT OR REPLACE INTO vec_ids(rowid, unit_id) VALUES(?,?)",
                    list(zip(rows, unit_ids[lo:hi])))

    def delete_units(self, unit_ids):
        with self.con:
            for (uid,) in [(u,) for u in unit_ids]:
                row = self.con.execute("SELECT rowid FROM vec_ids WHERE unit_id=?",
                                       (uid,)).fetchone()
                if row is None:
                    continue
                self.con.execute("DELETE FROM vec_items WHERE rowid=?", (row[0],))
                self.con.execute("DELETE FROM vec_ids WHERE unit_id=?", (uid,))

    def counts(self):
        n_ids = self.con.execute("SELECT COUNT(*) FROM vec_ids").fetchone()[0]
        n_vec = self.con.execute("SELECT COUNT(*) FROM vec_items").fetchone()[0]
        return {"ids": n_ids, "vec_items": n_vec}

    def knn(self, query_vec, topk, query_sig=None):
        import numpy as np

        if topk <= 0:
            raise ValueError(f"E_TOPK: topk must be >= 1, got {topk}")
        if query_sig is not None:
            check_space_compatible(query_sig, self.sig)
        q = np.asanyarray(query_vec, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"E_DIM: query dim {q.shape} != store dim {self.dim}")
        check_vector_values([float(v) for v in q], self.dim)
        # sqlite-vec 0.1.9 allows ONLY 'ORDER BY distance' on vec0 KNN
        # queries (multi-key ORDER BY is rejected by the engine -- verified
        # 2026-10-09). Equal-distance order is therefore engine-defined, not
        # unit_id order as on the exact path; the determinism test asserts
        # repeat-stability, and the parity test bounds top-K agreement.
        cur = self.con.execute(
            "SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?", (q.tobytes(), topk))
        out = []
        for rowid, dist in cur.fetchall():
            hit = self.con.execute("SELECT unit_id FROM vec_ids WHERE rowid=?",
                                   (rowid,)).fetchone()
            if hit is None:
                raise ValueError(
                    f"E_MISSING: vec_items rowid {rowid} has no vec_ids entry "
                    "(index drift -- rebuild the vec group)")
            out.append((hit[0], float(dist)))
        return out

    def fts_search(self, match_query, limit=10):
        try:
            cur = self.con.execute(
                "SELECT f.file_id, f.name, f.path, rank FROM files_fts "
                "JOIN files f ON f.rowid = files_fts.rowid "
                "WHERE files_fts MATCH ? ORDER BY rank LIMIT ?",
                (match_query, limit))
            return cur.fetchall(), "ok"
        except sqlite3.OperationalError:
            return [], "invalid-syntax"
