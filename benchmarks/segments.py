"""Video segmentation policies for P0-T008 (pure stdlib, no model/ffmpeg).

All policies take integer-millisecond durations and return contiguous
(start_ms, end_ms) spans. Time unit is integer ms (PLAN section 8).
Boundaries are validated: no negative, reversed, empty, or out-of-bounds
spans -- violations raise ValueError and abort the run (STEPS Done/prove).

Scene-cut DETECTION (ffmpeg I/O) lives in the experiment scripts; only the
pure cut->span splitter lives here so it is unit-testable.
"""

POLICY_VERSION = "seg-v1"


def validate_bounds(spans, duration_ms, policy):
    """Reject spans outside [0, duration_ms]; return spans unchanged."""
    if duration_ms <= 0:
        raise ValueError(f"E_DURATION: {policy}: duration_ms={duration_ms} must be > 0")
    if not spans:
        raise ValueError(f"E_EMPTY_SPANS: {policy}: no segments for {duration_ms} ms")
    for (s, e) in spans:
        if not (isinstance(s, int) and isinstance(e, int)):
            raise ValueError(f"E_BOUNDS: {policy}: non-integer span {(s, e)}")
        if not (0 <= s < e <= duration_ms):
            raise ValueError(
                f"E_BOUNDS: {policy}: span {(s, e)} outside [0, {duration_ms}]")
    return spans


def fixed_windows(duration_ms, window_ms, policy="fixed"):
    """Non-overlapping windows covering [0, duration); last clamped.

    Short clips (duration <= window) yield a single [0, duration] span.
    """
    if window_ms <= 0:
        raise ValueError(f"E_POLICY: {policy}: window_ms={window_ms} must be > 0")
    spans = []
    start = 0
    while start < duration_ms:
        spans.append((start, min(start + window_ms, duration_ms)))
        start += window_ms
    return validate_bounds(spans, duration_ms, policy)


def overlapping_windows(duration_ms, window_ms, stride_ms, policy="overlap"):
    """Fixed windows advancing by stride (stride < window for overlap)."""
    if window_ms <= 0 or stride_ms <= 0:
        raise ValueError(f"E_POLICY: {policy}: window/stride must be > 0")
    if stride_ms >= window_ms:
        raise ValueError(f"E_POLICY: {policy}: stride {stride_ms} >= window "
                         f"{window_ms} is not overlapping")
    spans = []
    start = 0
    while start < duration_ms:
        spans.append((start, min(start + window_ms, duration_ms)))
        if start + window_ms >= duration_ms:
            break
        start += stride_ms
    # Guarantee full coverage of the tail without a trailing sliver-gap:
    # (loop already clamps the last span to duration_ms).
    return validate_bounds(spans, duration_ms, policy)


def full_clip(duration_ms, policy="full"):
    """Degenerate single-vector baseline (R-08 rejects for production)."""
    return validate_bounds([(0, duration_ms)], duration_ms, policy)


def scene_spans(duration_ms, cuts_ms, policy="scene"):
    """Split [0, duration) at sorted scene-cut timestamps.

    No cuts -> [(0, duration)] (expected for static/synthetic clips;
    documented limitation, not a failure).
    """
    cuts = sorted(c for c in cuts_ms if 0 < c < duration_ms)
    bounds = [0] + cuts + [duration_ms]
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    return validate_bounds(spans, duration_ms, policy)


STRATEGIES = ("halves", "w2", "w2ov50", "scene", "scenew2", "full")
# NOTE: "keyframe" is an image-path baseline, not a span policy, so it is
# intentionally absent here (segment_compare.py handles it separately).


def spans_for(strategy, duration_ms, cuts_ms=()):
    """Dispatch to a named strategy (halving kept for the P0-T007 reference)."""
    if strategy == "halves":
        half = duration_ms // 2
        return validate_bounds([(0, half), (half, duration_ms)], duration_ms, strategy)
    if strategy == "w2":
        return fixed_windows(duration_ms, 2000, policy="w2")
    if strategy == "w2ov50":
        return overlapping_windows(duration_ms, 2000, 1000, policy="w2ov50")
    if strategy == "scene":
        return scene_spans(duration_ms, list(cuts_ms), policy="scene")
    if strategy == "scenew2":
        # Measured hybrid: scene cuts where the detector fires (>=2 spans),
        # otherwise fixed 2 s windows (never whole-clip). Decided in P0-T008.
        spans = scene_spans(duration_ms, list(cuts_ms), policy="scenew2/scene")
        if len(spans) >= 2:
            return spans
        return fixed_windows(duration_ms, 2000, policy="scenew2/w2")
    if strategy == "full":
        return full_clip(duration_ms, policy="full")
    raise ValueError(f"E_POLICY: unknown strategy {strategy!r} (expected one of {STRATEGIES})")
