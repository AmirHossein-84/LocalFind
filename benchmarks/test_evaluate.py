#!/usr/bin/env python3
"""P0-T007 unit tests: metrics, timestamp matching, compat gates, loader.

Stdlib unittest only. No model, no network, no corpus needed (synthetic
inline fixtures; temp manifests live in OS temp, never in the repo).

Run: benchmarks/.venv/Scripts/python -m unittest benchmarks.test_evaluate -v
(Or system python: python -m unittest benchmarks.test_evaluate -v)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compat import check_space_compatible, make_sig
from evaluate import load_eval_inputs, relevant_unit_keys
from retrieval import (EvaluationGateError, check_gate, cos, mrr_at,
                       rank_candidates, recall_at_k, segment_matches)


class TestRecallMrr(unittest.TestCase):
    def test_perfect(self):
        ranked = ["a", "b", "c"]
        self.assertEqual(recall_at_k(["a"], ranked, 1), 1.0)
        self.assertEqual(recall_at_k(["a"], ranked, 5), 1.0)
        self.assertEqual(mrr_at(["a"], ranked, 10), 1.0)

    def test_relevant_at_rank_3(self):
        ranked = ["x", "y", "a", "z"]
        self.assertEqual(recall_at_k(["a"], ranked, 1), 0.0)
        self.assertEqual(recall_at_k(["a"], ranked, 5), 1.0)
        self.assertAlmostEqual(mrr_at(["a"], ranked, 10), 1.0 / 3)

    def test_miss(self):
        ranked = ["x", "y", "z"]
        self.assertEqual(recall_at_k(["a"], ranked, 5), 0.0)
        self.assertEqual(mrr_at(["a"], ranked, 10), 0.0)

    def test_multi_relevant(self):
        ranked = ["a", "x", "b", "y", "z"]
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], ranked, 5), 2 / 3)
        self.assertEqual(mrr_at(["a", "b", "c"], ranked, 10), 1.0)

    def test_empty_relevant_raises(self):
        with self.assertRaises(ValueError):
            recall_at_k([], ["a"], 5)
        with self.assertRaises(ValueError):
            mrr_at([], ["a"], 10)


class TestNegativeControl(unittest.TestCase):
    """Deliberately inverted ranking must score ~0 AND trip the gate.

    This proves the metric + gate are sensitive: a bad retriever fails.
    """

    def test_inverted_ranking_fails(self):
        ranked = ["z", "y", "x", "w", "v", "a"]  # relevant dead last
        r5 = recall_at_k(["a"], ranked, 5)
        self.assertEqual(r5, 0.0)
        self.assertAlmostEqual(mrr_at(["a"], ranked, 10), 1.0 / 6)
        with self.assertRaises(EvaluationGateError):
            check_gate(r5, 0.75, "Recall@5 visual@768")

    def test_gate_passes(self):
        self.assertTrue(check_gate(0.9, 0.75, "Recall@5 visual@768"))


class TestTimestampMatching(unittest.TestCase):
    def exp(self, s, e, tol=500):
        return {"asset_id": "vid_001", "start_ms": s, "end_ms": e,
                "tolerance_ms": tol}

    def test_exact(self):
        self.assertTrue(segment_matches(
            self.exp(0, 3000),
            {"asset_id": "vid_001", "start_ms": 0, "end_ms": 3000}))

    def test_adjacent_no_overlap(self):
        exp = dict(self.exp(0, 3000), tolerance_ms=0)
        self.assertFalse(segment_matches(
            exp,
            {"asset_id": "vid_001", "start_ms": 3000, "end_ms": 6000}))

    def test_tolerance_bridges_seam(self):
        # Documented rule: tolerance expands BOTH sides, so an adjacent
        # segment overlaps the expanded interval. Intended, not a bug.
        self.assertTrue(segment_matches(
            self.exp(0, 3000),
            {"asset_id": "vid_001", "start_ms": 3000, "end_ms": 6000}))

    def test_tolerance_expands(self):
        # Pred [3100,6000] vs expected [0,3000]+500 -> overlap [3100,3500].
        self.assertTrue(segment_matches(
            self.exp(0, 3000),
            {"asset_id": "vid_001", "start_ms": 3100, "end_ms": 6000}))

    def test_tolerance_still_misses(self):
        self.assertFalse(segment_matches(
            self.exp(0, 3000),
            {"asset_id": "vid_001", "start_ms": 3600, "end_ms": 6000}))

    def test_wrong_asset(self):
        self.assertFalse(segment_matches(
            self.exp(0, 3000),
            {"asset_id": "vid_002", "start_ms": 0, "end_ms": 3000}))


class TestCompatGates(unittest.TestCase):
    def test_text_query_scores_vs_image_same_space(self):
        """P0-T006 contract fix: modality pipelines differ by design."""
        q = make_sig("text-query", 768, prep="p0t007-v1")
        d = make_sig("image", 768, prep="p0t007-v1")
        self.assertTrue(check_space_compatible(q, d))

    def test_dim_mismatch_refused(self):
        with self.assertRaises(ValueError):
            check_space_compatible(make_sig("text-query", 768),
                                   make_sig("image", 128))

    def test_dtype_mismatch_refused(self):
        with self.assertRaises(ValueError):
            check_space_compatible(make_sig("text-query", 768),
                                   make_sig("image", 768, dtype="bf16"))

    def test_revision_mismatch_refused(self):
        with self.assertRaises(ValueError):
            check_space_compatible(make_sig("text-query", 768),
                                   dict(make_sig("image", 768), revision="deadbee"))

    def test_fp16_refused_at_construction(self):
        with self.assertRaises(ValueError):
            make_sig("image", 768, dtype="fp16")


class TestLoader(unittest.TestCase):
    def write(self, tmp, corpus, queries):
        cp = Path(tmp) / "c.json"
        qp = Path(tmp) / "q.json"
        cp.write_text(json.dumps(corpus))
        qp.write_text(json.dumps(queries))
        return cp, qp

    def base(self):
        corpus = {"corpus_id": "t", "version": 1, "assets": [
            {"asset_id": "img_001", "modality": "image"},
            {"asset_id": "vid_001", "modality": "video", "duration_ms": 4000}]}
        queries = {"query_set_id": "tq", "version": 1, "corpus_id": "t",
                   "queries": [
                       {"query_id": "q001", "text": "x", "query_class": "visual",
                        "modality": "image",
                        "expected": [{"asset_id": "img_001", "grade": 2}]},
                       {"query_id": "q002", "text": "y", "query_class": "negative",
                        "modality": "any", "expected": []}]}
        return corpus, queries

    def test_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus, queries = self.base()
            cp, qp = self.write(tmp, corpus, queries)
            _c, _qd, assets, qs = load_eval_inputs(cp, qp)
            self.assertEqual(len(assets), 2)
            self.assertEqual(len(qs), 2)

    def test_dup_query_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus, queries = self.base()
            queries["queries"].append(dict(queries["queries"][0]))
            cp, qp = self.write(tmp, corpus, queries)
            with self.assertRaises(ValueError):
                load_eval_inputs(cp, qp)

    def test_dangling_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus, queries = self.base()
            queries["queries"][0]["expected"][0]["asset_id"] = "nope_999"
            cp, qp = self.write(tmp, corpus, queries)
            with self.assertRaises(ValueError):
                load_eval_inputs(cp, qp)

    def test_bad_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus, queries = self.base()
            queries["queries"].append(
                {"query_id": "q003", "text": "v", "query_class": "visual",
                 "modality": "video",
                 "expected": [{"asset_id": "vid_001", "grade": 2,
                               "start_ms": 0, "end_ms": 99999}]})
            cp, qp = self.write(tmp, corpus, queries)
            with self.assertRaises(ValueError):
                load_eval_inputs(cp, qp)

    def test_corpus_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus, queries = self.base()
            queries["corpus_id"] = "other"
            cp, qp = self.write(tmp, corpus, queries)
            with self.assertRaises(ValueError):
                load_eval_inputs(cp, qp)


class TestRelevantUnits(unittest.TestCase):
    """Regression test for the P0-T007 cross-asset leakage bug (pred asset
    was hardcoded to the expected asset, matching the whole corpus)."""

    def test_no_cross_asset_leak(self):
        assets = {"vid_001": {"modality": "video", "duration_ms": 6000},
                  "vid_002": {"modality": "video", "duration_ms": 6000}}
        video_keys = ["video:vid_001:0-3000", "video:vid_001:3000-6000",
                      "video:vid_002:0-3000", "video:vid_002:3000-6000"]
        query = {"query_id": "q021", "query_class": "visual",
                 "expected": [{"asset_id": "vid_001", "grade": 2,
                               "start_ms": 0, "end_ms": 3000,
                               "tolerance_ms": 500}]}
        rel = relevant_unit_keys(query, assets, video_keys)
        aids = {k for k, _g in rel}
        self.assertTrue(aids, "expected at least one relevant unit")
        for k, _g in rel:
            self.assertTrue(k.startswith("video:vid_001:"),
                            f"cross-asset leak: {k}")
        self.assertIn(("video:vid_001:0-3000", 2), rel)


class TestDeterminism(unittest.TestCase):
    def test_tie_break_by_key(self):
        cands = [(("b", 0, 0), [1.0, 0.0]), (("a", 0, 0), [1.0, 0.0])]
        ranked = rank_candidates([1.0, 0.0], cands)
        self.assertEqual([k for k, _s in ranked], [("a", 0, 0), ("b", 0, 0)])

    def test_repeat_identical(self):
        cands = [(("b", 0, 0), [0.1, 0.9]), (("a", 0, 0), [0.9, 0.1])]
        once = rank_candidates([1.0, 0.0], cands)
        twice = rank_candidates([1.0, 0.0], cands)
        self.assertEqual(once, twice)

    def test_cos_dim_guard(self):
        with self.assertRaises(ValueError):
            cos([1.0, 0.0], [1.0])


if __name__ == "__main__":
    unittest.main()
