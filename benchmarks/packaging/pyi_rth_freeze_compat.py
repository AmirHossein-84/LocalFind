"""P0-T010 PyInstaller runtime hook (packaging spike, NOT production code).

transformers 5.19.0 decorates ProcessorMixin.__call__ with @auto_docstring
at import time (transformers/processing_utils.py). The decorator decides
processor vs image-processor handling via inspect.getsourcefile() on the
source filename. Frozen, the source is not resolvable the way that check
expects (code runs from the PYZ), both checks return False, and
_process_kwargs_parameters raises ValueError -- killing the import of
sentence_transformers before any inference code runs. In dev the same
decorator resolves processing_utils.py and succeeds.

Generated docstrings have no inference value, so neutralize only the
raising helper: on ValueError, fall back to empty kwargs docs. Call
signature, model behavior, and all other decoration are untouched.

Runs at frozen startup, before the sidecar lazily imports
sentence_transformers/transformers. Frozen-only: dev never loads this
file. Failure here must never break startup, hence the broad guard.
"""

try:
    from transformers.utils import auto_docstring as _ad

    _orig_process_kwargs = _ad._process_kwargs_parameters

    def _tolerant_process_kwargs(*args, **kwargs):
        try:
            return _orig_process_kwargs(*args, **kwargs)
        except ValueError:
            return "", ""

    _ad._process_kwargs_parameters = _tolerant_process_kwargs
except Exception:
    pass
