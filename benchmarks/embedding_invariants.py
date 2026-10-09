#!/usr/bin/env python3
"""P0-T006: dtype, dimension and embedding-compatibility tests (EmbeddingGemma-2).

Uses the pinned project-local snapshot, fully offline:
  A. Compatibility contract (no model needed): compact embedding signature
     covering model id/revision, encoders, preprocessing, modality pipeline,
     dimension, dtype, normalization, and video segment policy; an explicit
     gate rejects mismatched/NaN/Inf/zero-norm/unnormalized vectors.
     Negative cases exit 1 with E_* codes when the gate fails to reject.
  B. Dimension sweep (fp32 CPU): encode fixtures ONCE at native 768-d, then
     Matryoshka-truncate to 512/256/128 + L2 re-normalize (card rule).
     Validates finite/unit-norm/dim at every level and compares text->image
     and text->video ranking across dimensions on synthetic fixtures.
  C. Precision probe: fp32 (full run) vs bf16 load+encode attempt on this CPU
     runtime (recorded, never forced). fp16 is rejected by policy WITHOUT
     running inference (card documents silent NaNs; R-11 forbids it).

No candidate dimension is chosen here — that needs P0-T007 quality data.

Usage:
    benchmarks/.venv/Scripts/python benchmarks/embedding_invariants.py

Exit 0 = all checks pass, 1 = failure with E_* code. No network use
(run with HF_HUB_OFFLINE=1 for cache-only proof).
"""

import argparse
import gc
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from compat import (NATIVE_DIM, TRUNC_DIMS, check_compatible,
                      check_vector_values, make_sig, sig_string,
                      truncate_renorm)
from local_model import MODEL_ID, REVISION, resolve_model_source

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_negative_tests(results):
    """Gate must reject every incompatible case. No model needed."""
    base = make_sig("image", 768)
    cases = [
        ("dim-mismatch", dict(base, dim=128)),
        ("dtype-mismatch", dict(base, dtype="bf16")),
        ("norm-mismatch", dict(base, norm="none")),
        ("revision-mismatch", dict(base, revision="deadbee")),
        ("prep-mismatch", dict(base, prep="other-v9")),
        ("segment-mismatch", dict(base, segment="thirds")),
        ("model-mismatch", dict(base, model="other")),
        ("encoders-mismatch", dict(base, encoders="text-only")),
        ("modality-mismatch", dict(base, modality="video")),
    ]
    # Low-level value gates (dim fixed at 4 for speed; logic is dim-agnostic).
    unit4 = [0.5, 0.5, 0.5, 0.5]
    value_cases = [
        ("nan-vector", [float("nan")] + unit4[1:], 4),
        ("inf-vector", [float("inf")] + unit4[1:], 4),
        ("zero-vector", [0.0, 0.0, 0.0, 0.0], 4),
        ("unnormalized", [1.0, 1.0, 1.0, 1.0], 4),
        ("wrong-len", unit4, 3),
    ]
    policy_cases = [
        ("fp16-policy", lambda: make_sig("image", 768, dtype="fp16")),
        ("dim511-policy", lambda: make_sig("image", 511)),
    ]
    passed = []
    for name, other in cases:
        try:
            check_compatible(base, other)
            print(f"E_GATE: negative case {name!r} NOT rejected", file=sys.stderr)
            return False, passed
        except ValueError as exc:
            passed.append(f"{name}: rejected ({exc})")
    for name, vals, dim in value_cases:
        try:
            check_vector_values(vals, dim)
            print(f"E_GATE: negative case {name!r} NOT rejected", file=sys.stderr)
            return False, passed
        except ValueError as exc:
            passed.append(f"{name}: rejected ({exc})")
    for name, thunk in policy_cases:
        try:
            thunk()
            print(f"E_GATE: negative case {name!r} NOT rejected", file=sys.stderr)
            return False, passed
        except ValueError as exc:
            passed.append(f"{name}: rejected ({exc})")
    # Sanity: identical signatures must pass.
    check_compatible(base, dict(base))
    passed.append("identical-signatures: accepted")
    results["negative_tests"] = passed
    return True, passed


def fail(code, message):
    print(f"{code}: {message}", file=sys.stderr)
    sys.exit(1)


def cos(a, b):
    if len(a) != len(b):
        raise ValueError(f"E_DIM: cos inputs len {len(a)} != {len(b)} -- refusing to mix")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    return max(-1.0, min(1.0, dot / (na * nb)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0-T006 dtype/dimension/compat tests.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "docs" / "benchmarks"
                        / "embedding-invariants-results.json")
    args = parser.parse_args(argv)
    results = {"model_id": MODEL_ID, "revision": REVISION, "device": args.device,
               "dims": {}, "precision": {}, "ranking_by_dim": {},
               "prefix_by_dim": {}, "negative_tests": [], "timings_s": {}}

    ok, passed = run_negative_tests(results)
    if not ok:
        fail("E_GATE", "compatibility gate failed to reject an incompatible case")
    print(f"negative tests: {len(passed)}/{len(passed)} rejected-or-accepted correctly", flush=True)
    for line in passed:
        print(f"  {line}", flush=True)

    import psutil
    import torch
    from sentence_transformers import SentenceTransformer

    if args.device == "cuda" and not torch.cuda.is_available():
        fail("E_DEVICE", "cuda requested but unavailable")
    try:
        load_arg, revision, source = resolve_model_source(args.model_dir)
    except FileNotFoundError as exc:
        fail("E_MODEL_DIR", str(exc))
    results["model_source"] = source
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        fail("E_TOOL", "ffmpeg not on PATH")

    proc = psutil.Process()
    rss_before = proc.memory_info().rss / 1e6
    t0 = time.perf_counter()
    try:
        model = SentenceTransformer(load_arg, revision=revision, device=args.device,
                                    model_kwargs={"torch_dtype": torch.float32},
                                    config_kwargs={"audio_config": None})
    except Exception as exc:  # noqa: BLE001
        fail("E_LOAD", f"fp32 load failed [{type(exc).__name__}]: {exc}")
    results["timings_s"]["load_fp32"] = time.perf_counter() - t0
    if any("float16" in str(p.dtype) for p in model.parameters()):
        fail("E_DTYPE", "float16 parameters present (card forbids fp16)")
    results["precision"]["fp32"] = {"ran": True, "param_dtype": "torch.float32"}

    # ---- fixtures: 2 queries, 1 doc, 2 images, 2 video segments (all 768-d) --
    from PIL import Image
    workdir = Path(tempfile.mkdtemp(prefix="lf-invariants-"))
    try:
        from transformers.video_utils import load_video
        seg_paths = []
        for (aid, s, e) in [("vid_001", 0, 3000), ("vid_001", 3000, 6000)]:
            clip = REPO_ROOT / "benchmarks" / "corpus" / "videos" / (
                "vid_001_tone_red_blue.mp4")
            seg = workdir / f"{aid}_{s}_{e}.mp4"
            pr = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(clip),
                 "-ss", str(s / 1000), "-t", str((e - s) / 1000),
                 "-r", "30", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-an", str(seg)],
                capture_output=True, text=True, timeout=180)
            if pr.returncode != 0:
                fail("E_FFMPEG", f"segment cut failed: {(pr.stderr or '')[-300:]}")
            seg_paths.append(seg)
        t0 = time.perf_counter()
        native = {
            "q_red": list(model.encode("solid red image", prompt_name="SearchQuery")),
            "q_blue": list(model.encode("blue screen", prompt_name="SearchQuery")),
            "doc": list(model.encode("A solid red test frame.", prompt_name="Document")),
            "img_red": list(model.encode(Image.open(
                REPO_ROOT / "tests/fixtures/samples/img_red.png").convert("RGB"))),
            "img_bars": list(model.encode(Image.open(
                REPO_ROOT / "tests/fixtures/samples/img_bars.png").convert("RGB"))),
        }
        for seg in seg_paths:
            frames, _m = load_video(str(seg), backend="pyav")
            n = int(frames.shape[0])
            native[f"seg_{seg.stem}"] = list(model.encode(
                {"text": "<|video|>",
                 "video": {"array": frames,
                           "video_metadata": {"fps": 30, "duration": n / 30.0,
                                              "total_num_frames": n,
                                              "frames_indices": list(range(n))}}}))
        results["timings_s"]["encode_768_total"] = time.perf_counter() - t0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # ---- B. dimension sweep: truncate + re-normalize, validate, rank -------
    asset_keys = ["img_red", "img_bars", "seg_vid_001_0_3000", "seg_vid_001_3000_6000"]
    red_set = {"img_red", "seg_vid_001_0_3000"}
    t0 = time.perf_counter()
    for dim in (NATIVE_DIM,) + TRUNC_DIMS:
        vecs = {}
        for key, vals in native.items():
            trunc = truncate_renorm(vals, dim)
            check_vector_values(trunc, dim)  # raises E_* on failure
            vecs[key] = trunc
        sig_q = make_sig("image", dim, prep="p0t006-v1")
        check_compatible(make_sig("image", dim, prep="p0t006-v1"),
                         make_sig("image", dim, prep="p0t006-v1"))
        # Incompatible cross-dim scoring must be refused, not silently mixed.
        try:
            check_compatible(make_sig("image", 768, prep="p0t006-v1"),
                             make_sig("image", dim, prep="p0t006-v1") if dim != 768
                             else make_sig("image", 128, prep="p0t006-v1"))
            fail("E_GATE", f"cross-dim scoring dim768-vs-dim{dim} NOT refused")
        except ValueError:
            pass
        order_red = sorted(asset_keys, key=lambda k: -cos(vecs["q_red"], vecs[k]))
        order_blue = sorted(asset_keys, key=lambda k: -cos(vecs["q_blue"], vecs[k]))
        results["ranking_by_dim"][str(dim)] = {
            "red_query_order": order_red, "red_top": order_red[0],
            "blue_query_order": order_blue, "blue_top": order_blue[0]}
        results["prefix_by_dim"][str(dim)] = {
            "prefixed": cos(vecs["q_red"], vecs["doc"])}
        results["dims"][str(dim)] = {"n_vectors": len(vecs), "all_finite_norm": True,
                                     "sig_example": sig_string(sig_q)}
        print(f"dim {dim}: red_top={order_red[0]} blue_top={order_blue[0]}", flush=True)
    results["timings_s"]["sweep_truncate_rank"] = time.perf_counter() - t0

    # Regression gates (P0-T005 findings must hold at usable dims; 128 is record-only).
    for dim in ("768", "512", "256"):
        r = results["ranking_by_dim"][dim]
        if r["red_top"] not in red_set or r["blue_top"] != "seg_vid_001_3000_6000":
            fail("E_REGRESS", f"dim {dim}: ranking regressed vs P0-T005 "
                              f"(red_top={r['red_top']}, blue_top={r['blue_top']})")
    results["regression_768_512_256"] = "PASS"

    # ---- C. bf16 probe (sequential; never forced) ---------------------------
    del model
    gc.collect()
    if hasattr(torch, "cpu") and hasattr(torch.cpu, "_is_bf16_supported"):
        cpu_bf16 = bool(torch.cpu._is_bf16_supported())
    else:
        cpu_bf16 = None
    results["precision"]["cpu_bf16_flag"] = cpu_bf16
    try:
        t0 = time.perf_counter()
        model_bf16 = SentenceTransformer(load_arg, revision=revision, device=args.device,
                                         model_kwargs={"torch_dtype": torch.bfloat16},
                                         config_kwargs={"audio_config": None})
        results["timings_s"]["load_bf16"] = time.perf_counter() - t0
        q_b = list(model_bf16.encode("solid red image", prompt_name="SearchQuery"))
        r_b = list(model_bf16.encode(Image.open(
            REPO_ROOT / "tests/fixtures/samples/img_red.png").convert("RGB")))
        check_vector_values(q_b, 768)
        check_vector_values(r_b, 768)
        results["precision"]["bf16"] = {
            "ran": True, "finite_norm_768": True,
            "cos_bf16_vs_fp32_query": cos(q_b, truncate_renorm(native["q_red"], 768))}
        del model_bf16
        gc.collect()
    except Exception as exc:  # noqa: BLE001 -- outcome recorded, never forced
        results["precision"]["bf16"] = {"ran": False,
                                        "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    results["precision"]["fp16"] = {"ran": False,
                                    "reason": "rejected by policy (card: silent NaNs; R-11)"}

    rss_after = proc.memory_info().rss / 1e6
    results["memory_mb"] = {"rss_before": rss_before, "rss_after": rss_after,
                            "rss_delta": rss_after - rss_before}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"dims ok: 768/512/256/128 finite+renorm; "
          f"bf16: {results['precision']['bf16']}; rss_delta={rss_after-rss_before:.0f} MB")
    print(f"results: {args.out}")
    print("INVARIANTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
