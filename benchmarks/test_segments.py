#!/usr/bin/env python3
"""Unit tests for benchmarks/segments.py (stdlib only, no model/ffmpeg).

Run: benchmarks/.venv/Scripts/python -m unittest benchmarks.test_segments -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from segments import (fixed_windows, full_clip, overlapping_windows,
                      scene_spans, spans_for, validate_bounds)


class TestFixedWindows(unittest.TestCase):
    def test_halves_like(self):
        self.assertEqual(fixed_windows(6000, 3000), [(0, 3000), (3000, 6000)])

    def test_clamped_tail(self):
        self.assertEqual(fixed_windows(5000, 2000),
                         [(0, 2000), (2000, 4000), (4000, 5000)])

    def test_short_clip_single_span(self):
        self.assertEqual(fixed_windows(1000, 2000), [(0, 1000)])

    def test_deterministic(self):
        self.assertEqual(fixed_windows(12000, 2000), fixed_windows(12000, 2000))

    def test_coverage_contiguous(self):
        spans = fixed_windows(8000, 3000)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], 8000)
        for ( _s1, e1), (s2, _e2) in zip(spans, spans[1:]):
            self.assertEqual(e1, s2)

    def test_bad_window(self):
        with self.assertRaises(ValueError):
            fixed_windows(6000, 0)


class TestOverlapping(unittest.TestCase):
    def test_stride(self):
        self.assertEqual(overlapping_windows(6000, 2000, 1000),
                         [(0, 2000), (1000, 3000), (2000, 4000),
                          (3000, 5000), (4000, 6000)])

    def test_tail_clamped(self):
        spans = overlapping_windows(5000, 2000, 1000)
        self.assertEqual(len(spans), 4)
        self.assertEqual(spans[-1], (3000, 5000))
        self.assertEqual(spans[0], (0, 2000))

    def test_non_overlapping_rejected(self):
        with self.assertRaises(ValueError):
            overlapping_windows(6000, 2000, 2000)


class TestSceneAndFull(unittest.TestCase):
    def test_scene_cuts(self):
        self.assertEqual(scene_spans(6000, [1000, 4500]),
                         [(0, 1000), (1000, 4500), (4500, 6000)])

    def test_scene_no_cuts_single(self):
        self.assertEqual(scene_spans(6000, []), [(0, 6000)])

    def test_scene_out_of_range_cuts_ignored(self):
        self.assertEqual(scene_spans(6000, [-5, 0, 6000, 9999]), [(0, 6000)])

    def test_full(self):
        self.assertEqual(full_clip(6000), [(0, 6000)])


class TestValidation(unittest.TestCase):
    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            validate_bounds([(-1, 100)], 6000, "t")

    def test_reversed_rejected(self):
        with self.assertRaises(ValueError):
            validate_bounds([(200, 100)], 6000, "t")

    def test_empty_span_rejected(self):
        with self.assertRaises(ValueError):
            validate_bounds([(100, 100)], 6000, "t")

    def test_overflow_rejected(self):
        with self.assertRaises(ValueError):
            validate_bounds([(0, 6001)], 6000, "t")

    def test_empty_list_rejected(self):
        with self.assertRaises(ValueError):
            validate_bounds([], 6000, "t")

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            spans_for("nope", 6000)

    def test_dispatch(self):
        self.assertEqual(spans_for("halves", 6000), [(0, 3000), (3000, 6000)])
        self.assertEqual(len(spans_for("w2", 6000)), 3)
        self.assertEqual(spans_for("full", 6000), [(0, 6000)])

    def test_scenew2_uses_cuts(self):
        self.assertEqual(spans_for("scenew2", 6000, [3000]),
                         [(0, 3000), (3000, 6000)])

    def test_scenew2_falls_back_to_w2(self):
        self.assertEqual(spans_for("scenew2", 5000, []),
                         [(0, 2000), (2000, 4000), (4000, 5000)])
        self.assertEqual(spans_for("scenew2", 1000, []), [(0, 1000)])


if __name__ == "__main__":
    unittest.main()
