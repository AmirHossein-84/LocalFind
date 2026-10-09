"""Embedding compatibility contract (P0-T006, refined P0-T007). Phase-0 prototype.

Two-level contract for the future worker/storage boundary:
  - SPACE compatibility (scoring): model, revision, dim, dtype, norm must be
    EQUAL. A text query scores against image/video vectors when (and only
    when) the shared-space fields match -- modality pipelines differ by
    design in a unified embedding space.
  - STORAGE identity (persistence): the full signature (space + encoders,
    prep, modality, segment) identifies an index group; groups must never mix.

Model card rules enforced: fp16 never allowed; dims only 768/512/256/128;
truncated vectors must be L2 re-normalized (see truncate_renorm).
Pure stdlib; import-safe without torch.
"""

import math

NATIVE_DIM = 768
TRUNC_DIMS = (512, 256, 128)
SUPPORTED_DIMS = (NATIVE_DIM,) + TRUNC_DIMS
SUPPORTED_DTYPES = ("fp32", "bf16")  # card rule: NEVER fp16
MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
SHORT_REV = REVISION[:7]
MODEL_TAG = "eg2"
ENCODERS = "text+image"  # audio_config=None, as in P0-T004..P0-T007
NORM = "l2"
SPACE_FIELDS = ("model", "revision", "dim", "dtype", "norm")


def make_sig(modality, dim, dtype="fp32", prep="p0t006-v1", segment="n/a"):
    """Compatibility signature dict. Raises on policy violation."""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"E_DTYPE_POLICY: dtype {dtype!r} not supported (never fp16)")
    if dim not in SUPPORTED_DIMS:
        raise ValueError(f"E_DIM_POLICY: dim {dim} not in {SUPPORTED_DIMS}")
    return {"model": MODEL_TAG, "model_id": MODEL_ID, "revision": SHORT_REV,
            "encoders": ENCODERS, "prep": prep, "modality": modality,
            "dim": dim, "dtype": dtype, "norm": NORM, "segment": segment}


def sig_string(sig):
    return ("{model}@{revision}|{encoders}|{prep}|{modality}|{segment}|"
            "{dim}|{dtype}|{norm}".format(**sig))


def space_id(sig):
    """Shared-space identity: the fields that must match for scoring."""
    return tuple(sig[f] for f in SPACE_FIELDS)


def check_space_compatible(query_sig, doc_sig):
    """Raise unless two signatures share one embedding space (scoring gate).

    Pipeline fields (modality/prep/segment/encoders) may differ: a text
    query legitimately scores against image/video vectors. They are
    recorded, not gated, here.
    """
    for field in SPACE_FIELDS:
        if query_sig.get(field) != doc_sig.get(field):
            raise ValueError(
                f"E_MISMATCH: shared-space field {field!r}: "
                f"{query_sig.get(field)!r} != {doc_sig.get(field)!r} -- refusing to mix")
    return True


def check_compatible(query_sig, doc_sig):
    """Raise unless two signatures are identical (storage-identity gate)."""
    for field in ("model", "revision", "encoders", "prep", "modality",
                  "dim", "dtype", "norm", "segment"):
        if query_sig.get(field) != doc_sig.get(field):
            raise ValueError(
                f"E_MISMATCH: field {field!r}: {query_sig.get(field)!r} != "
                f"{doc_sig.get(field)!r} -- refusing to mix")
    return True


def check_vector_values(vals, dim):
    """Reject non-finite / zero-norm / unnormalized / wrong-length vectors."""
    if len(vals) != dim:
        raise ValueError(f"E_DIM: len {len(vals)} != {dim}")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("E_NONFINITE: NaN or Inf present")
    norm = math.sqrt(sum(v * v for v in vals))
    if norm == 0.0:
        raise ValueError("E_ZERO_NORM: zero vector")
    if abs(norm - 1.0) > 1e-4:
        raise ValueError(f"E_NORM: norm {norm:.6f} != 1.0 (re-normalize after truncation)")
    return norm


def truncate_renorm(vals, dim):
    """Matryoshka slice (leading dims) + L2 re-normalize. Card rule."""
    if dim == NATIVE_DIM:
        return list(vals)
    part = list(vals[:dim])
    norm = math.sqrt(sum(v * v for v in part))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("E_TRUNC: truncated slice has zero/non-finite norm")
    return [v / norm for v in part]
