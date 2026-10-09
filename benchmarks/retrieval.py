"""Pure retrieval-metric helpers for P0-T007 (stdlib only, no model).

Deterministic by construction: rankings break cosine ties by (asset_id,
start_ms) so repeated runs are bit-identical. All functions are exercised by
benchmarks/test_evaluate.py without loading any model.
"""

import math


class EvaluationGateError(AssertionError):
    """Raised when a measured metric misses a required gate."""


def cos(a, b):
    if len(a) != len(b):
        raise ValueError(f"E_DIM: cos inputs len {len(a)} != {len(b)}")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("E_ZERO_NORM: zero vector in cos")
    return max(-1.0, min(1.0, dot / (na * nb)))


def rank_candidates(query_vec, candidates):
    """Rank [(key, vec)] by cosine desc; ties broken by key for determinism.

    Returns [(key, cos)] sorted best-first. Keys must be orderable
    (candidate unit key strings sort deterministically).
    """
    scored = [(key, cos(query_vec, vec)) for key, vec in candidates]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored


def recall_at_k(relevant, ranked_keys, k):
    """|relevant cap topK| / |relevant|. relevant must be non-empty."""
    if not relevant:
        raise ValueError("E_EMPTY_RELEVANT: recall undefined for negative queries; "
                         "exclude them from aggregates")
    rel = set(relevant)
    return len(rel.intersection(ranked_keys[:k])) / len(rel)


def mrr_at(relevant, ranked_keys, k):
    """1/rank of first relevant in topK else 0. relevant must be non-empty."""
    if not relevant:
        raise ValueError("E_EMPTY_RELEVANT: mrr undefined for negative queries; "
                         "exclude them from aggregates")
    rel = set(relevant)
    for rank, key in enumerate(ranked_keys[:k], start=1):
        if key in rel:
            return 1.0 / rank
    return 0.0


def segment_matches(expected, pred):
    """Does predicted (asset,start,end) hit the expected judgment?

    Rule (documented in retrieval-v1.md): same asset AND positive overlap
    after expanding the expected interval by tolerance_ms on both sides:
        overlap = min(pred_end, exp_end + tol) - max(pred_start, exp_start - tol)
        hit iff overlap > 0.
    """
    if expected["asset_id"] != pred["asset_id"]:
        return False
    tol = expected.get("tolerance_ms", 0)
    lo = max(pred["start_ms"], expected["start_ms"] - tol)
    hi = min(pred["end_ms"], expected["end_ms"] + tol)
    return (hi - lo) > 0


def check_gate(value, threshold, name):
    """Raise EvaluationGateError when a metric misses its gate. Never silent."""
    if value < threshold:
        raise EvaluationGateError(f"GATE MISS: {name} {value:.4f} < {threshold}")
    return True
