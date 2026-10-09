"""Pure retrieval-metric helpers for P0-T007 (stdlib only, no model).

Deterministic by construction: rankings break cosine ties by (asset_id,
start_ms) so repeated runs are bit-identical. All functions are exercised by
benchmarks/test_evaluate.py without loading any model.

Temporal policies (P0 integrity rework, judgments unchanged):
  - "tolerant": legacy rule -- expected interval expanded by tolerance_ms
    on both sides (bridges adjacent-segment seams by design).
  - "strict": zero-tolerance overlap (adjacent segments sharing only an
    edge do NOT match). Used for sensitivity analysis only; the gate
    suite's reported verdict stays on the tolerant (original) policy.
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


def segment_matches_strict(expected, pred):
    """Zero-tolerance overlap: same asset AND positive interval overlap.

    Adjacent segments sharing only an edge (e.g. [0,3000] vs [3000,6000])
    do NOT match here, while the tolerant rule above DOES match them when
    tolerance_ms > 0. Sensitivity-analysis counterpart; judgments untouched.
    """
    if expected["asset_id"] != pred["asset_id"]:
        return False
    lo = max(pred["start_ms"], expected["start_ms"])
    hi = min(pred["end_ms"], expected["end_ms"])
    return (hi - lo) > 0


def segment_matches_with_policy(expected, pred, policy="tolerant"):
    """Policy-dispatched temporal match. Raises E_POLICY on unknown policy."""
    if policy == "tolerant":
        return segment_matches(expected, pred)
    if policy == "strict":
        return segment_matches_strict(expected, pred)
    raise ValueError(f"E_POLICY: unknown temporal policy {policy!r} "
                     "(expected 'strict' or 'tolerant')")


def parse_unit_key(key):
    """Parse a candidate/query unit key into its components.

    Accepted forms (all deterministic string sorts):
      - "image:{asset_id}" (2 parts)
      - "keyframe:{asset_id}" (2 parts, whole-file baseline vector)
      - "query:{query_id}" (2 parts, never ranked as a candidate)
      - "video:{asset_id}:{start_ms}-{end_ms}" (3 parts, P0-T007 halves)
      - "video:{asset_id}:{strategy}:{start_ms}-{end_ms}" (4 parts, P0-T008)

    Returns {"kind", "asset_id", "strategy", "start_ms", "end_ms"} where
    strategy is None except for 4-part video keys and span fields are None
    for non-video keys. Raises ValueError E_UNITKEY on malformed input
    (never an unhandled unpack error at call sites).
    """
    if not isinstance(key, str) or not key:
        raise ValueError(f"E_UNITKEY: malformed unit key {key!r}")
    parts = key.split(":")
    kind = parts[0]
    if kind in ("image", "keyframe", "query"):
        if len(parts) != 2 or not parts[1]:
            raise ValueError(f"E_UNITKEY: malformed {kind} key {key!r} "
                             "(expected 'kind:id')")
        return {"kind": kind, "asset_id": parts[1], "strategy": None,
                "start_ms": None, "end_ms": None}
    if kind == "video":
        if len(parts) == 3:
            _k, aid, span = parts
            strategy = None
        elif len(parts) == 4:
            _k, aid, strategy, span = parts
            if not strategy:
                raise ValueError(f"E_UNITKEY: empty strategy in key {key!r}")
        else:
            raise ValueError(f"E_UNITKEY: malformed video key {key!r} "
                             "(expected 3 or 4 colon-parts)")
        if not aid or "-" not in span:
            raise ValueError(f"E_UNITKEY: malformed video span in key {key!r}")
        lo_s, _, hi_s = span.partition("-")
        try:
            s, e = int(lo_s), int(hi_s)
        except ValueError:
            raise ValueError(f"E_UNITKEY: non-integer span in key {key!r}")
        if not (0 <= s < e):
            raise ValueError(f"E_UNITKEY: reversed/empty span in key {key!r}")
        return {"kind": kind, "asset_id": aid, "strategy": strategy,
                "start_ms": s, "end_ms": e}
    raise ValueError(f"E_UNITKEY: unknown unit kind {kind!r} in key {key!r}")


def relevant_asset_of(unit_key):
    """File asset behind a unit key (image/video/keyframe forms).

    Raises E_UNITKEY on malformed keys instead of an IndexError.
    """
    return parse_unit_key(unit_key)["asset_id"]


def group_rank_by_file(ranked):
    """Group ranked [(unit_key, score)] by file, keeping the best per file.

    Returns a file-id list sorted best-first, ties broken by asset id for
    determinism. Used for file-grouped Recall/MRR (comparable across
    segmentation strategies, unlike segment-level denominators).
    """
    best = {}
    for key, score in ranked:
        aid = relevant_asset_of(key)
        if aid not in best or score > best[aid]:
            best[aid] = score
    return sorted(best, key=lambda a: (-best[a], a))


def top1_time_error(query, top_unit_key, assets, policy="tolerant"):
    """|predicted segment center - expected interval center| (ms).

    Returns None when the top-1 unit is not a hit under the given temporal
    policy (accuracy is measured only over hits; misses are counted in
    Recall, and the miss rate is reported alongside the mean).
    """
    parsed = parse_unit_key(top_unit_key)
    if parsed["kind"] != "video":
        return None
    aid, s, e = parsed["asset_id"], parsed["start_ms"], parsed["end_ms"]
    for exp in query.get("expected", []):
        if exp.get("grade", 2) < 1 or exp["asset_id"] != aid:
            continue
        if segment_matches_with_policy(
                {**exp, "asset_id": aid},
                {"asset_id": aid, "start_ms": s, "end_ms": e}, policy):
            pred_c = (s + e) / 2
            exp_c = (exp["start_ms"] + exp["end_ms"]) / 2
            return abs(pred_c - exp_c)
    return None


def check_gate(value, threshold, name):
    """Raise EvaluationGateError when a metric misses its gate. Never silent."""
    if value < threshold:
        raise EvaluationGateError(f"GATE MISS: {name} {value:.4f} < {threshold}")
    return True
