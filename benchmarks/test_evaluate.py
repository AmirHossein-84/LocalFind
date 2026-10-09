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
from evaluate import (filename_score_for_key, index_size_report,
                      load_eval_inputs, load_historical_backup,
                      preserve_historical_report, relevant_file_assets,
                      relevant_unit_keys, relevant_unit_keys_policy)
from retrieval import (EvaluationGateError, check_gate, cos,
                       group_rank_by_file, mrr_at, parse_unit_key,
                       rank_candidates, recall_at_k, relevant_asset_of,
                       segment_matches, segment_matches_strict,
                       segment_matches_with_policy, top1_time_error)


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


class TestStrictTolerantPolicy(unittest.TestCase):
    """Tolerance-expanded vs zero-tolerance temporal relevance (P0 rework).

    Original (tolerant) judgments are unchanged; strict is sensitivity-only.
    """

    def exp(self, s, e, tol=500):
        return {"asset_id": "vid_001", "start_ms": s, "end_ms": e,
                "tolerance_ms": tol}

    def test_exact_overlap_both_policies(self):
        pred = {"asset_id": "vid_001", "start_ms": 0, "end_ms": 3000}
        self.assertTrue(segment_matches_with_policy(self.exp(0, 3000), pred,
                                                    "tolerant"))
        self.assertTrue(segment_matches_with_policy(self.exp(0, 3000), pred,
                                                    "strict"))

    def test_adjacent_seam_tolerant_only(self):
        # The seam case behind the denominator shift: [3000,6000] shares
        # only an edge with [0,3000], so strict rejects, tolerant accepts.
        pred = {"asset_id": "vid_001", "start_ms": 3000, "end_ms": 6000}
        self.assertTrue(segment_matches_with_policy(self.exp(0, 3000), pred,
                                                    "tolerant"))
        self.assertFalse(segment_matches_with_policy(self.exp(0, 3000), pred,
                                                     "strict"))
        self.assertFalse(segment_matches_strict(self.exp(0, 3000), pred))

    def test_far_miss_both_policies(self):
        pred = {"asset_id": "vid_001", "start_ms": 3600, "end_ms": 6000}
        self.assertFalse(segment_matches_with_policy(self.exp(0, 3000), pred,
                                                     "tolerant"))
        self.assertFalse(segment_matches_with_policy(self.exp(0, 3000), pred,
                                                     "strict"))

    def test_unknown_policy_rejected(self):
        pred = {"asset_id": "vid_001", "start_ms": 0, "end_ms": 3000}
        with self.assertRaises(ValueError):
            segment_matches_with_policy(self.exp(0, 3000), pred, "fuzzy")

    def test_tolerance_doubles_halves_denominator(self):
        # q021-style: expected [0,3000]+500 over halves -> 2 tolerant units
        # but only 1 strict unit. The Recall denominator moves with policy.
        assets = {"vid_001": {"modality": "video", "duration_ms": 6000}}
        video_keys = ["video:vid_001:0-3000", "video:vid_001:3000-6000"]
        query = {"query_id": "q021", "query_class": "visual",
                 "expected": [{"asset_id": "vid_001", "grade": 2,
                               "start_ms": 0, "end_ms": 3000,
                               "tolerance_ms": 500}]}
        tol = relevant_unit_keys_policy(query, assets, video_keys,
                                        policy="tolerant")
        stri = relevant_unit_keys_policy(query, assets, video_keys,
                                         policy="strict")
        self.assertEqual(len(tol), 2)
        self.assertEqual(len(stri), 1)
        self.assertEqual(stri[0][0], "video:vid_001:0-3000")
        # Legacy entry point reproduces the tolerant (original) judgment.
        self.assertEqual(relevant_unit_keys(query, assets, video_keys), tol)

    def test_multi_relevant_fractional_credit(self):
        ranked = ["a", "x", "b", "y", "z"]
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], ranked, 5), 2 / 3)
        self.assertAlmostEqual(recall_at_k(["a", "b"], ranked, 5), 1.0)
        self.assertAlmostEqual(recall_at_k(["a", "b"], ranked, 1), 0.5)


class TestFileGroupedMetrics(unittest.TestCase):
    def test_group_rank_dedupes_by_file(self):
        ranked = [("video:vid_002:0-3000", 0.9),
                  ("video:vid_001:0-3000", 0.8),
                  ("video:vid_002:3000-6000", 0.7),
                  ("image:img_001", 0.6)]
        self.assertEqual(group_rank_by_file(ranked),
                         ["vid_002", "vid_001", "img_001"])

    def test_group_rank_tie_break_by_asset(self):
        ranked = [("video:vid_002:0-3000", 0.9),
                  ("video:vid_001:0-3000", 0.9)]
        self.assertEqual(group_rank_by_file(ranked), ["vid_001", "vid_002"])

    def test_group_rank_repeat_identical(self):
        ranked = [("video:vid_002:0-3000", 0.9),
                  ("image:img_001", 0.5)]
        self.assertEqual(group_rank_by_file(ranked),
                         group_rank_by_file(list(ranked)))

    def test_file_recall_absorbs_seam(self):
        # Either halves-half of vid_001 retrieves the FILE under both
        # policies: file denominator (=1 file) is strategy/policy-stable.
        assets = {"vid_001": {"modality": "video", "duration_ms": 6000}}
        query = {"query_id": "q021", "query_class": "visual",
                 "expected": [{"asset_id": "vid_001", "grade": 2,
                               "start_ms": 0, "end_ms": 3000,
                               "tolerance_ms": 500}]}
        self.assertEqual(relevant_file_assets(query, assets), ["vid_001"])
        file_rank = ["vid_002", "vid_001", "img_001"]
        self.assertEqual(recall_at_k(["vid_001"], file_rank, 5), 1.0)
        self.assertAlmostEqual(mrr_at(["vid_001"], file_rank, 10), 0.5)


class TestUnitKeys(unittest.TestCase):
    def test_image_key(self):
        self.assertEqual(parse_unit_key("image:img_001")["asset_id"], "img_001")

    def test_three_part_video_key(self):
        parsed = parse_unit_key("video:vid_001:0-3000")
        self.assertEqual((parsed["asset_id"], parsed["strategy"],
                          parsed["start_ms"], parsed["end_ms"]),
                         ("vid_001", None, 0, 3000))

    def test_four_part_strategy_key(self):
        parsed = parse_unit_key("video:vid_001:w2:0-2000")
        self.assertEqual((parsed["asset_id"], parsed["strategy"],
                          parsed["start_ms"], parsed["end_ms"]),
                         ("vid_001", "w2", 0, 2000))

    def test_relevant_asset_of_four_part(self):
        self.assertEqual(relevant_asset_of("video:vid_001:w2ov50:1000-3000"),
                         "vid_001")
        self.assertEqual(relevant_asset_of("keyframe:vid_003"), "vid_003")

    def test_relevance_supports_four_part_keys(self):
        assets = {"vid_001": {"modality": "video", "duration_ms": 6000}}
        video_keys = ["video:vid_001:w2:0-2000",
                      "video:vid_001:w2:2000-4000",
                      "video:vid_001:w2:4000-6000"]
        query = {"query_id": "q021", "query_class": "visual",
                 "expected": [{"asset_id": "vid_001", "grade": 2,
                               "start_ms": 0, "end_ms": 3000,
                               "tolerance_ms": 500}]}
        rel = relevant_unit_keys_policy(query, assets, video_keys,
                                        policy="tolerant")
        self.assertEqual([k for k, _g in rel],
                         ["video:vid_001:w2:0-2000",
                          "video:vid_001:w2:2000-4000"])

    def test_malformed_keys_rejected_typed(self):
        for bad in ("", "video:vid_001", "video:vid_001:w2",
                    "video::0-2000", "video:vid_001:3000-3000",
                    "video:vid_001:4000-2000", "video:vid_001:notaspan",
                    "video:vid_001:a:b:c", "unknown:xyz"):
            with self.assertRaises(ValueError, msg=bad):
                parse_unit_key(bad)

    def test_filename_score_supports_four_part(self):
        assets = {"vid_001": {"modality": "video",
                              "rel_path": "videos/vid_001_tone_red_blue.mp4"},
                  "img_001": {"modality": "image",
                              "rel_path": "images/img_014_solid-red.png"}}
        three = filename_score_for_key("red screen", "video:vid_001:0-3000",
                                       assets)
        four = filename_score_for_key("red screen", "video:vid_001:w2:0-2000",
                                      assets)
        self.assertEqual(three, four)
        self.assertGreater(four, 0.0)

    def test_filename_score_rejects_query_keys(self):
        assets = {"img_001": {"modality": "image", "rel_path": "images/a.png"}}
        with self.assertRaises(ValueError):
            filename_score_for_key("a", "query:q001", assets)


class TestTimestampReporting(unittest.TestCase):
    def query(self):
        return {"query_id": "q021", "query_class": "visual",
                "expected": [{"asset_id": "vid_001", "grade": 2,
                              "start_ms": 0, "end_ms": 3000,
                              "tolerance_ms": 500}]}

    def assets(self):
        return {"vid_001": {"modality": "video", "duration_ms": 6000}}

    def test_hit_returns_center_distance(self):
        # Pred center 1500 vs expected center 1500 -> 0 ms error.
        self.assertEqual(top1_time_error(self.query(), "video:vid_001:0-3000",
                                         self.assets(), policy="tolerant"), 0.0)

    def test_miss_returns_none(self):
        self.assertIsNone(top1_time_error(self.query(), "video:vid_001:3000-6000",
                                          self.assets(), policy="strict"))
        self.assertIsNone(top1_time_error(self.query(), "video:vid_002:0-3000",
                                          self.assets(), policy="tolerant"))

    def test_non_video_returns_none(self):
        self.assertIsNone(top1_time_error(self.query(), "image:img_001",
                                          self.assets(), policy="tolerant"))

    def test_miss_rate_counts_complement_of_hits(self):
        n_queries, n_hits = 11, 6
        self.assertAlmostEqual((n_queries - n_hits) / n_queries, 5 / 11)


class TestIndexAccounting(unittest.TestCase):
    def test_candidate_query_bytes_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            vec_dir = Path(tmp)
            (vec_dir / "image.npz").write_bytes(b"i" * 100)
            (vec_dir / "video.npz").write_bytes(b"v" * 50)
            (vec_dir / "query.npz").write_bytes(b"q" * 30)
            rep = index_size_report(vec_dir)
            self.assertEqual(rep["candidate_bytes"], 150)
            self.assertEqual(rep["query_bytes"], 30)
            self.assertEqual(rep["total_bytes"], 180)

    def test_historical_preservation_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            raw = b'{"verdict":"PIVOT-SIGNAL (old)","by_dim":{}}'
            out.write_bytes(raw)
            old = preserve_historical_report(out)
            self.assertEqual(old, {"verdict": "PIVOT-SIGNAL (old)",
                                   "by_dim": {}})
            backup = Path(tmp) / "report.historical.json"
            self.assertTrue(backup.is_file())
            self.assertEqual(backup.read_bytes(), raw)

    def test_current_report_version_not_backed_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            out.write_text(json.dumps({"report_version": 2, "verdict": "x"}))
            old = preserve_historical_report(out) or {}
            self.assertEqual(old.get("verdict"), "x")
            self.assertFalse((Path(tmp) / "report.historical.json").is_file())

    def test_historical_section_prefers_frozen_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            out.write_bytes(b'{"verdict":"v1-old"}')
            preserve_historical_report(out)
            out.write_text(json.dumps({"report_version": 2,
                                       "verdict": "v2-new"}))
            frozen = load_historical_backup(out) or {}
            self.assertEqual(frozen.get("verdict"), "v1-old")


if __name__ == "__main__":
    unittest.main()
