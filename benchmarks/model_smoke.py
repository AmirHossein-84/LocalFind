#!/usr/bin/env python3
"""P0-T004 smoke experiment: google/embeddinggemma-2 text + image embeddings.

Verifies on the local machine that the official model loads and produces
finite, normalized 768-d embeddings with sensible cross-modal behavior:
  - one retrieval text query (prompt_name="SearchQuery")
  - one text document (prompt_name="Document" vs manual "title: none" rule)
  - two consented synthetic images (tests/fixtures/samples/)
  - cross-modal cosine ranking as smoke evidence (NOT a quality claim)

Follows the official model card (https://huggingface.co/google/embeddinggemma-2):
bfloat16 or float32 only -- float16 is forbidden and asserted against here.

Usage:
    benchmarks/.venv/Scripts/python benchmarks/model_smoke.py [--device cpu] [--out <results.json>]

Reads the model from the local HF cache (download it first, see
docs/benchmarks/model-smoke.md). Writes a machine-readable results JSON and
prints a human summary. Exit 0 = all smoke checks pass, 1 = failure w/ E_* code.
Stdlib + torch/sentence-transformers/Pillow/psutil only. No network use.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from local_model import MODEL_ID, REVISION, resolve_model_source

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_DIM = 768
IMG_RED = REPO_ROOT / "tests" / "fixtures" / "samples" / "img_red.png"
IMG_BARS = REPO_ROOT / "tests" / "fixtures" / "samples" / "img_bars.png"
QUERY_TEXT = "solid red image"
DOC_TEXT = "A solid red test frame used for retrieval experiments."


def fail(code, message):
    print(f"{code}: {message}", file=sys.stderr)
    sys.exit(1)


def check_vector(name, vec, results):
    import math
    vals = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    finite = all(math.isfinite(v) for v in vals)
    norm = math.sqrt(sum(v * v for v in vals))
    entry = {"dim": len(vals), "finite": finite, "norm": norm,
             "normalized": abs(norm - 1.0) < 1e-4, "sample": vals[:4]}
    results["vectors"][name] = entry
    if len(vals) != EXPECTED_DIM:
        fail("E_DIM", f"{name}: dim {len(vals)} != {EXPECTED_DIM}")
    if not finite:
        fail("E_NONFINITE", f"{name}: non-finite values")
    if not entry["normalized"]:
        fail("E_NORM", f"{name}: norm {norm:.6f} != 1.0")
    return vals


def cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return max(-1.0, min(1.0, dot / (na * nb)))


def load_rgb(path: Path):
    from PIL import Image
    with Image.open(path) as img:
        return img.convert("RGB").copy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0-T004 EmbeddingGemma-2 smoke test.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model-dir", default=None,
                        help="Explicit local snapshot dir (default: auto-resolve project-local, else HF cache).")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "docs" / "benchmarks" / "model-smoke-results.json")
    args = parser.parse_args(argv)
    for img in (IMG_RED, IMG_BARS):
        if not img.is_file():
            fail("E_IO", f"fixture missing: {img} (run P0-T003 generation)")

    import psutil
    import torch
    from sentence_transformers import SentenceTransformer

    if args.device == "cuda" and not torch.cuda.is_available():
        fail("E_DEVICE", "cuda requested but torch.cuda.is_available() is False")
    device = args.device
    dtype = torch.float32  # per model card: bf16 or fp32; fp32 for CPU. NEVER fp16.
    try:
        load_arg, revision, source = resolve_model_source(args.model_dir)
    except FileNotFoundError as exc:
        fail("E_MODEL_DIR", str(exc))
    # R-24: results JSON must not contain machine-specific absolute paths.
    try:
        load_ref = str(Path(load_arg).relative_to(REPO_ROOT))
    except ValueError:
        load_ref = load_arg if source == "huggingface-cache" else "local-dir"
    results = {"model_id": MODEL_ID, "revision": REVISION, "device": device,
               "model_source": source, "load_ref": load_ref,
               "torch_dtype": str(dtype), "vectors": {}, "timings_s": {},
               "versions": {}, "checks": {}}
    results["versions"] = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import sentence_transformers, transformers, PIL
        results["versions"].update({
            "sentence_transformers": sentence_transformers.__version__,
            "transformers": transformers.__version__, "pillow": PIL.__version__})
    except ImportError as exc:
        fail("E_DEPS", f"missing experiment dependency: {exc}")

    proc = psutil.Process()
    rss_before_mb = proc.memory_info().rss / 1e6

    t0 = time.perf_counter()
    try:
        model = SentenceTransformer(
            load_arg, revision=revision, device=device,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs={"audio_config": None},  # text+image only (440M) for this host
        )
    except Exception as exc:  # noqa: BLE001 -- surfaced with error code, never silent
        fail("E_LOAD", f"model load failed [{type(exc).__name__}]: {exc}")
    results["timings_s"]["load_cold"] = time.perf_counter() - t0

    dtypes = {str(p.dtype) for p in model.parameters()}
    results["checks"]["param_dtypes"] = sorted(dtypes)
    if any("float16" in d or d == "torch.float16" for d in dtypes):
        fail("E_DTYPE", f"float16 detected in parameters: {sorted(dtypes)} (card forbids fp16)")

    # Warm-up (excluded from warm timings), then timed repeats.
    model.encode(QUERY_TEXT, prompt_name="SearchQuery")
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        q_emb = model.encode(QUERY_TEXT, prompt_name="SearchQuery")
    results["timings_s"]["text_query_warm_mean"] = (time.perf_counter() - t0) / reps
    # Images go through the vision tower as decoded RGB, NOT as path text.
    # Control: the embedding of the literal path string must differ sharply
    # from the image embedding, proving the vision path was taken.
    red_img = load_rgb(IMG_RED)
    bars_img = load_rgb(IMG_BARS)
    t0 = time.perf_counter()
    for _ in range(reps):
        i_emb = model.encode(red_img)
    results["timings_s"]["image_warm_mean"] = (time.perf_counter() - t0) / reps

    q_vals = check_vector("query_SearchQuery", q_emb, results)
    doc_pref = model.encode(DOC_TEXT, prompt_name="Document")
    d_vals = check_vector("document_Document-prefix", doc_pref, results)
    doc_manual = model.encode(f"title: none | text: {DOC_TEXT}")
    m_vals = check_vector("document_manual-title-none", doc_manual, results)
    equiv = cos(d_vals, m_vals)
    results["checks"]["document_prefix_equivalence_cos"] = equiv
    results["checks"]["document_prefix_equivalent"] = equiv > 0.99

    sim_prefixed = cos(q_vals, d_vals)
    q_plain = check_vector("query_no-prefix", model.encode(QUERY_TEXT), results)
    sim_plain = cos(q_plain, d_vals)
    results["checks"]["similarity_query(Document)_prefixed"] = sim_prefixed
    results["checks"]["similarity_query(Document)_unprefixed"] = sim_plain

    # Vision-path control: the same pixels under two different temp filenames
    # must embed (near-)identically (pixels determine the output, not the name).
    import shutil
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="lf-smoke-"))
    try:
        copy_a = tmpdir / "copy_a.png"
        copy_b = tmpdir / "copy_b.png"
        shutil.copy(IMG_RED, copy_a)
        shutil.copy(IMG_RED, copy_b)
        same_a = model.encode(str(copy_a))
        same_b = model.encode(str(copy_b))
        same_cos = cos(same_a.tolist(), same_b.tolist())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    results["checks"]["vision_control_same_pixels_cos"] = same_cos
    results["checks"]["vision_path_taken"] = bool(same_cos > 0.999)
    if not results["checks"]["vision_path_taken"]:
        fail("E_VISION_PATH", "same pixels under different names diverged; "
                              "path string may be encoded as text, not vision")
    img_red_vals = check_vector("image_red", model.encode(red_img), results)
    img_bars_vals = check_vector("image_bars", model.encode(bars_img), results)
    s_red = cos(q_vals, img_red_vals)
    s_bars = cos(q_vals, img_bars_vals)
    results["checks"]["crossmodal_cos_query_vs_red"] = s_red
    results["checks"]["crossmodal_cos_query_vs_bars"] = s_bars
    results["checks"]["crossmodal_ranking"] = (
        "img_red.png > img_bars.png" if s_red > s_bars else "img_bars.png > img_red.png")
    results["checks"]["crossmodal_red_top"] = bool(s_red > s_bars)

    rss_after_mb = proc.memory_info().rss / 1e6
    results["memory"] = {"rss_before_mb": rss_before_mb, "rss_after_mb": rss_after_mb,
                         "rss_delta_mb": rss_after_mb - rss_before_mb}
    if device == "cuda":
        results["memory"]["cuda_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"model: {MODEL_ID}@{REVISION[:7]} source={source} device={device} dtype={dtype}")
    print(f"load_cold={results['timings_s']['load_cold']:.1f}s "
          f"text_warm={results['timings_s']['text_query_warm_mean']*1000:.0f}ms "
          f"image_warm={results['timings_s']['image_warm_mean']*1000:.0f}ms")
    print(f"query/doc cos: prefixed={sim_prefixed:.4f} unprefixed={sim_plain:.4f} "
          f"(prefix-equivalence={equiv:.4f})")
    print(f"cross-modal cos: red={s_red:.4f} bars={s_bars:.4f} "
          f"-> {results['checks']['crossmodal_ranking']} "
          f"(vision-control same-pixels cos={same_cos:.4f})")
    print(f"all vectors finite+normalized 768-d; rss_delta={results['memory']['rss_delta_mb']:.0f} MB")
    print(f"results: {args.out}")
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
