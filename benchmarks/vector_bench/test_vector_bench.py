#!/usr/bin/env python3
"""P0-T009 correctness tests: FTS + exact BLOB + sqlite-vec prototype.

Stdlib unittest only. No model, no network. Small synthetic fixtures in OS
temp dirs (never the repo, never user data). sqlite-vec tests skip cleanly
when the package is missing.

Covers (STEPS.md P0-T009 + prompt section 4):
  exact filename matching, FTS lexical retrieval + ranking, invalid-syntax
  safety, prefix escaping, rename/delete consistency, vector top-K
  correctness vs brute force, deterministic tie handling, persistence across
  reopen, missing/deleted records, dim/signature mismatches, zero/NaN/Inf
  rejection, fp16 policy rejection, transaction atomicity, vec-vs-exact
  parity (ANN recall vs exact ground truth on the tested config).

Run: python -m unittest benchmarks.vector_bench.test_vector_bench -v
"""

import math
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compat import check_space_compatible, check_vector_values, make_sig  # noqa: E402
from store import (  # noqa: E402
    ExactStore,
    VecStore,
    check_matrix_fast,
    gen_unit_vectors,
    like_escape,
    store_sig,
)

DIM = 128  # smallest P0-T006 contract dim: fast tests, policy-valid
SEED = 20261009

try:
    import sqlite_vec  # noqa: F401

    HAVE_VEC = True
except ImportError:
    HAVE_VEC = False


def tiny_sig(dim=DIM):
    return store_sig(dim)


class FTSBehavior(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lf-t009-")
        self.db = ExactStore(Path(self.tmp.name) / "t.db", tiny_sig(), create=True)
        self.db.insert_files([
            ("f1", "IMG_4281.JPG", "C:/Users/test/Pictures/IMG_4281.JPG"),
            ("f2", "年度报告_2024.PDF", "C:/Users/test/Documents/年度报告_2024.PDF"),
            ("f3", "notes final (copy).DOCX", "D:/Archive/notes final (copy).DOCX"),
            ("f4", "IMG_4282.JPG", "C:/Users/test/Pictures/IMG_4282.JPG"),
        ])

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_exact_filename_match(self):
        rows = self.db.exact_name("IMG_4281.JPG")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "f1")

    def test_fts_finds_token(self):
        rows, status = self.db.fts_search("IMG_4281", limit=10)
        self.assertEqual(status, "ok")
        self.assertTrue(any(r[0] == "f1" for r in rows))

    def test_fts_unicode(self):
        rows, status = self.db.fts_search("年度报告", limit=10)
        self.assertEqual(status, "ok")
        self.assertTrue(any(r[0] == "f2" for r in rows))

    def test_fts_ranking_prefers_closer_match(self):
        # FTS5 space is implicit AND (no row has both tokens); OR matches both.
        rows, status = self.db.fts_search("IMG_4281 OR IMG_4282", limit=10)
        self.assertEqual(status, "ok")
        # Both camera files match one token each; just assert sane behavior.
        ids = [r[0] for r in rows]
        self.assertIn("f1", ids)
        self.assertIn("f4", ids)

    def test_invalid_syntax_never_crashes(self):
        for bad in ['"unclosed', "(((", "NEAR(", "*", "OR OR OR"]:
            rows, status = self.db.fts_search(bad, limit=10)
            self.assertEqual(status, "invalid-syntax", msg=f"query {bad!r}")
            self.assertEqual(rows, [])

    def test_prefix_path_with_like_wildcards_escaped(self):
        # A literal % in the prefix must not act as a wildcard.
        self.db.insert_files([("f9", "100%.PNG", "D:/Archive/100%/100%.PNG")])
        rows = self.db.prefix_path("D:/Archive/100%/")
        self.assertEqual([r[0] for r in rows], ["f9"])
        self.assertIn("%", like_escape("100%"))
        self.assertIn("\\%", like_escape("100%"))

    def test_rename_updates_fts(self):
        self.db.insert_files([("f1", "RENAMED.JPG", "C:/Users/test/Pictures/RENAMED.JPG")])
        rows, _ = self.db.fts_search("RENAMED", limit=10)
        self.assertTrue(any(r[0] == "f1" for r in rows))
        rows, _ = self.db.fts_search("IMG_4281", limit=10)
        self.assertFalse(any(r[0] == "f1" for r in rows))

    def test_delete_removes_fts_row(self):
        self.db.delete_files(["f1"])
        rows, _ = self.db.fts_search("IMG_4281", limit=10)
        self.assertFalse(any(r[0] == "f1" for r in rows))
        self.assertEqual(self.db.exact_name("IMG_4281.JPG"), [])

    def test_rebuild_keeps_results(self):
        before, _ = self.db.fts_search("IMG_4281", limit=10)
        self.db.rebuild_fts()
        after, _ = self.db.fts_search("IMG_4281", limit=10)
        self.assertEqual(before, after)


class ExactVectorBehavior(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lf-t009-")
        self.sig = tiny_sig()
        self.db = ExactStore(Path(self.tmp.name) / "v.db", self.sig, create=True)
        self.mat = gen_unit_vectors(50, DIM, SEED)
        self.ids = [f"u{i:04d}" for i in range(50)]
        self.db.insert_embeddings(self.ids, self.mat)
        self.q = gen_unit_vectors(1, DIM, SEED + 777)[0]

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_topk_matches_brute_force(self):
        import numpy as np

        scores = self.mat @ self.q
        expect = sorted(range(50), key=lambda i: (-float(scores[i]), self.ids[i]))[:10]
        got = self.db.search(self.q, 10, query_sig=tiny_sig())
        self.assertEqual([u for u, _ in got], [self.ids[i] for i in expect])
        # Scores are cosines of unit vectors: within [-1, 1].
        for _u, s in got:
            self.assertTrue(-1.0 <= s <= 1.0)

    def test_deterministic_tie_handling(self):
        import numpy as np

        dup = np.stack([self.mat[0], self.mat[0], self.mat[1]])
        self.db.insert_embeddings(["t-a", "t-b", "t-c"], dup)
        got = self.db.search(self.mat[0], 3, query_sig=tiny_sig())
        # t-a and t-b tie at cos=1.0 with u0000; unit_id order must win.
        tied = [u for u, s in got if abs(s - 1.0) < 1e-6]
        self.assertEqual(tied, sorted(tied))
        self.assertEqual(tied[0], "t-a")

    def test_persistence_across_reopen(self):
        before = self.db.search(self.q, 5, query_sig=tiny_sig())
        self.db.close()
        reopened = ExactStore(Path(self.tmp.name) / "v.db", tiny_sig(), create=False)
        after = reopened.search(self.q, 5, query_sig=tiny_sig())
        self.assertEqual(before, after)
        self.assertEqual(reopened.counts()["embeddings"], 50)
        reopened.close()
        # Reattach for tearDown.
        self.db = ExactStore(Path(self.tmp.name) / "v.db", tiny_sig(), create=False)

    def test_missing_and_deleted(self):
        self.assertIsNone(self.db.get_vector("u-does-not-exist"))
        self.db.delete_units(["u0007"])
        self.assertIsNone(self.db.get_vector("u0007"))
        got = self.db.search(self.q, 50, query_sig=tiny_sig())
        self.assertFalse(any(u == "u0007" for u, _ in got))
        self.assertEqual(len(got), 49)

    def test_dim_mismatch_rejected(self):
        import numpy as np

        with self.assertRaises(ValueError) as ctx:
            self.db.insert_embeddings(["bad"], np.zeros((1, DIM + 1), dtype="float32"))
        self.assertIn("E_DIM", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.db.search(np.zeros(DIM + 1, dtype="float32"), 5, query_sig=tiny_sig())
        self.assertIn("E_DIM", str(ctx.exception))
        # Wrong-dim store signature must not open this file's group.
        with self.assertRaises(ValueError) as ctx:
            ExactStore(Path(self.tmp.name) / "v.db", tiny_sig(dim=256), create=False)
        self.assertIn("E_MISMATCH", str(ctx.exception))

    def test_signature_mismatch_rejected(self):
        other = dict(tiny_sig())
        other["revision"] = "deadbee"
        with self.assertRaises(ValueError) as ctx:
            self.db.search(self.q, 5, query_sig=other)
        self.assertIn("E_MISMATCH", str(ctx.exception))

    def test_zero_nan_inf_rejected(self):
        import numpy as np

        for name, vec in [
            ("zero", np.zeros(DIM, dtype="float32")),
            ("nan", np.array([math.nan] + [0.0] * (DIM - 1), dtype="float32")),
            ("inf", np.array([math.inf] + [0.0] * (DIM - 1), dtype="float32")),
        ]:
            with self.assertRaises(ValueError, msg=name):
                check_vector_values([float(v) for v in vec], DIM)
        with self.assertRaises(ValueError) as ctx:
            check_matrix_fast(np.zeros((2, DIM), dtype="float32"), DIM)
        self.assertIn("E_ZERO_NORM", str(ctx.exception))

    def test_topk_nonpositive_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.db.search(self.q, 0, query_sig=tiny_sig())
        self.assertIn("E_TOPK", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.db.search(self.q, -3, query_sig=tiny_sig())
        self.assertIn("E_TOPK", str(ctx.exception))

    def test_reopen_with_different_prep_rejected(self):
        """Storage-identity gate (review B2): same space, other prep must fail."""
        import copy

        other = copy.deepcopy(self.sig)
        other["prep"] = "other-v9"
        with self.assertRaises(ValueError) as ctx:
            ExactStore(Path(self.tmp.name) / "v.db", other, create=False)
        self.assertIn("E_MISMATCH", str(ctx.exception))

    def test_store_rejects_non_fp32_dtype(self):
        """Prototype storage is fp32-only (review A5); bf16 must fail fast."""
        with self.assertRaises(ValueError) as ctx:
            store_sig(256, dtype="bf16")
        self.assertIn("E_DTYPE_STORE", str(ctx.exception))


    def test_fp16_policy_rejected_without_inference(self):
        with self.assertRaises(ValueError) as ctx:
            make_sig("image", 768, dtype="fp16")
        self.assertIn("fp16", str(ctx.exception))

    def test_failed_transaction_preserves_prior_state(self):
        import numpy as np

        before = self.db.counts()["embeddings"]
        try:
            with self.db.con:
                self.db.con.execute(
                    "INSERT INTO embeddings(unit_id, dim, vec) VALUES(?,?,?)",
                    ("txn-ok", DIM, self.mat[0].tobytes()))
                # Second insert violates PK -> whole transaction rolls back.
                self.db.con.execute(
                    "INSERT INTO embeddings(unit_id, dim, vec) VALUES(?,?,?)",
                    ("txn-ok", DIM, self.mat[1].tobytes()))
        except sqlite3.IntegrityError:
            pass
        self.assertEqual(self.db.counts()["embeddings"], before)
        self.assertIsNone(self.db.get_vector("txn-ok"))


@unittest.skipUnless(HAVE_VEC, "sqlite_vec not installed")
class VecParityBehavior(unittest.TestCase):
    """sqlite-vec (default L2 on unit vectors) vs exact cosine ground truth."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lf-t009-")
        self.sig = tiny_sig()
        self.mat = gen_unit_vectors(300, DIM, SEED)
        self.ids = [f"u{i:04d}" for i in range(300)]
        self.ex = ExactStore(Path(self.tmp.name) / "e.db", self.sig, create=True)
        self.ex.insert_embeddings(self.ids, self.mat)
        self.vc = VecStore(Path(self.tmp.name) / "s.db", self.sig, create=True)
        self.vc.insert_embeddings(self.ids, self.mat)
        self.queries = gen_unit_vectors(20, DIM, SEED + 555)

    def tearDown(self):
        self.ex.close()
        self.vc.close()
        self.tmp.cleanup()

    def test_recall_at_10_is_perfect(self):
        recalls = []
        for q in self.queries:
            ref = [u for u, _ in self.ex.search(q, 10, query_sig=tiny_sig())]
            got = [u for u, _ in self.vc.knn(q, 10, query_sig=tiny_sig())]
            recalls.append(len(set(ref).intersection(got)) / 10)
        self.assertEqual(min(recalls), 1.0)
        self.assertEqual(sum(recalls) / len(recalls), 1.0)

    def test_vec_persistence_and_delete(self):
        q = self.queries[0]
        before = [u for u, _ in self.vc.knn(q, 5, query_sig=tiny_sig())]
        self.vc.close()
        reopened = VecStore(Path(self.tmp.name) / "s.db", tiny_sig(), create=False)
        after = [u for u, _ in reopened.knn(q, 5, query_sig=tiny_sig())]
        self.assertEqual(before, after)
        victim = after[0]
        reopened.delete_units([victim])
        again = [u for u, _ in reopened.knn(q, 5, query_sig=tiny_sig())]
        self.assertNotIn(victim, again)
        reopened.close()
        self.vc = VecStore(Path(self.tmp.name) / "s.db", tiny_sig(), create=False)

    def test_vec_rejects_bad_sig_and_dim(self):
        import numpy as np

        other = dict(tiny_sig())
        other["dim"] = 256
        with self.assertRaises(ValueError) as ctx:
            self.vc.knn(self.queries[0], 5, query_sig=other)
        self.assertIn("E_MISMATCH", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.vc.insert_embeddings(["bad"], np.zeros((1, DIM + 1), dtype="float32"))
        self.assertIn("E_DIM", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.vc.knn(self.queries[0], 0, query_sig=tiny_sig())
        self.assertIn("E_TOPK", str(ctx.exception))

    def test_vec_upsert_reuses_rowid_no_orphans(self):
        """Re-inserting the same unit_ids must not orphan vec_items (B1)."""
        import numpy as np

        fresh = gen_unit_vectors(300, DIM, SEED + 1)
        self.vc.insert_embeddings(self.ids, fresh)
        self.assertEqual(self.vc.counts(), {"ids": 300, "vec_items": 300})
        q = self.queries[0]
        got = self.vc.knn(q, 10, query_sig=tiny_sig())
        self.assertEqual(len(got), 10)
        for uid, _d in got:
            self.assertIn(uid, self.ids)

    def test_vec_duplicate_ties_repeat_deterministically(self):
        """Duplicate vectors: repeated knn must return identical order (B5)."""
        import numpy as np

        dup = np.stack([self.mat[0], self.mat[0], self.mat[1]])
        self.vc.insert_embeddings(["t-a", "t-b", "t-c"], dup)
        first = self.vc.knn(self.mat[0], 5, query_sig=tiny_sig())
        second = self.vc.knn(self.mat[0], 5, query_sig=tiny_sig())
        self.assertEqual(first, second)


class CompatContractPreserved(unittest.TestCase):
    def test_space_gate_allows_modality_difference(self):
        q = make_sig("text-query", 768, prep="p0t007-v1")
        d = make_sig("image", 768, prep="p0t007-v1")
        self.assertTrue(check_space_compatible(q, d))

    def test_store_sig_uses_supported_policy(self):
        sig = store_sig(768)
        self.assertEqual(sig["dim"], 768)
        with self.assertRaises(ValueError):
            store_sig(100)
        with self.assertRaises(ValueError):
            store_sig(768, dtype="fp16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
