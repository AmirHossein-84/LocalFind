"""Central model-path resolution for Phase-0 benchmark scripts (P0-T005).

Prefers the project-local snapshot at <repo>/models/embeddinggemma-2/
(verified copy of the pinned Hugging Face revision) and falls back to the
Hugging Face model ID + revision (local cache, no re-download logic here).

No machine-specific absolute paths: the repo root is derived from this file's
location, and everything else is relative to it.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
LOCAL_MODEL_DIR = REPO_ROOT / "models" / "embeddinggemma-2"
SOURCE_MARKER = ".localfind-source.json"
REQUIRED_FILES = ("model.safetensors", "config.json", "tokenizer.json",
                  "processor_config.json")


def local_snapshot_ready(path: Path = LOCAL_MODEL_DIR) -> bool:
    """True when the project-local directory holds the required runtime files."""
    return all((path / name).is_file() for name in REQUIRED_FILES)


def snapshot_provenance(path: Path = LOCAL_MODEL_DIR) -> dict:
    """Read the copy-verification marker written at snapshot install time.

    Returns {} when absent (snapshot usable but revision unattested).
    """
    try:
        return json.loads((path / SOURCE_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def resolve_model_source(explicit_dir: str | None = None):
    """Return (load_arg, revision_or_None, source_label).

    - explicit_dir: caller override (e.g. --model-dir); must be snapshot-ready.
    - repo models/ dir when ready: load_arg is the local path (offline-capable).
    - otherwise: (MODEL_ID, REVISION, 'huggingface-cache').
    Raises FileNotFoundError with an actionable message when unusable.
    """
    if explicit_dir:
        candidate = Path(explicit_dir)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if not local_snapshot_ready(candidate):
            raise FileNotFoundError(
                f"E_MODEL_DIR: {candidate} lacks {REQUIRED_FILES}; "
                "copy the pinned snapshot per docs/benchmarks/model-smoke.md")
        prov = snapshot_provenance(candidate)
        label = ("explicit-dir(revision-verified)" if prov.get("revision") == REVISION
                 else "explicit-dir(unverified)")
        return str(candidate), None, label
    if local_snapshot_ready():
        prov = snapshot_provenance()
        label = ("project-local(revision-verified)" if prov.get("revision") == REVISION
                 else "project-local(unverified)")
        return str(LOCAL_MODEL_DIR), None, label
    return MODEL_ID, REVISION, "huggingface-cache"
