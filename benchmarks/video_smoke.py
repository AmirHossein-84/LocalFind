#!/usr/bin/env python3
"""P0-T005 smoke experiment: segment-level video embeddings (EmbeddingGemma-2).

Probes two synthetic clips from the P0-T003 corpus, cuts 2-3 bounded
millisecond intervals per clip (halves + thirds strategies), encodes each
segment through the model's OFFICIALLY SUPPORTED video path
({"text": "<|video|>", "video": {"array": frames, "video_metadata": {...}}},
full-frame pyav decode + the processor's default 1 FPS sampling),
and ranks the segments against short visual queries.

Each returned vector maps to its exact (asset_id, start_ms, end_ms); no
whole-video single-vector shortcut. Decode (ffmpeg) and encode (model) times
are measured separately. Sequential inference only (low-RAM host).

Usage:
    benchmarks/.venv/Scripts/python benchmarks/video_smoke.py [--model-dir <dir>]

Model resolves via benchmarks/local_model.py (project-local snapshot
preferred). Exit 0 = all checks pass, 1 = failure with E_* code.
Stdlib + torch/sentence-transformers/Pillow/psutil only. No network use
(run with HF_HUB_OFFLINE=1 for cache-only proof).
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from local_model import MODEL_ID, REVISION, resolve_model_source

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_DIM = 768
CLIPS = ("vid_001", "vid_009")  # two-tone red/blue 6s; motion testsrc 5s
QUERIES = [
    # (query text, expected asset_id, expected [start,end], note)
    ("red screen", "vid_001", (0, 3000), "first half of two-tone clip"),
    ("blue screen", "vid_001", (3000, 6000), "second half of two-tone clip"),
    ("moving test pattern", "vid_009", (0, 5000), "whole motion clip relevant"),
]
STRATEGIES = {"halves": 2, "thirds": 3}
FFMPEG_TIMEOUT_S = 180


def fail(code, message):
    print(f"{code}: {message}", file=sys.stderr)
    sys.exit(1)


def run_cmd(cmd, what):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)
    except FileNotFoundError:
        fail("E_TOOL", f"required tool not on PATH: {cmd[0]}")
    except subprocess.TimeoutExpired:
        fail("E_TIMEOUT", f"timed out after {FFMPEG_TIMEOUT_S}s: {what}")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        fail("E_FFMPEG", f"{what} failed:\n" + "\n".join(tail))
    return proc


def probe_duration_ms(ffprobe, clip):
    proc = run_cmd([ffprobe, "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(clip)],
                   f"probe {clip.name}")
    return int(round(float(proc.stdout.strip()) * 1000))


def check_vector(label, vec, results):
    import math
    vals = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    finite = all(math.isfinite(v) for v in vals)
    norm = math.sqrt(sum(v * v for v in vals))
    entry = {"dim": len(vals), "finite": finite, "norm": norm,
             "normalized": abs(norm - 1.0) < 1e-4}
    results["vectors"][label] = entry
    if len(vals) != EXPECTED_DIM:
        fail("E_DIM", f"{label}: dim {len(vals)} != {EXPECTED_DIM}")
    if not finite:
        fail("E_NONFINITE", f"{label}: non-finite values")
    if not entry["normalized"]:
        fail("E_NORM", f"{label}: norm {norm:.6f} != 1.0")
    return vals


def cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return max(-1.0, min(1.0, dot / (na * nb)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0-T005 video segment smoke test.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "docs" / "benchmarks" / "video-smoke-results.json")
    args = parser.parse_args(argv)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        fail("E_TOOL", "ffmpeg/ffprobe not on PATH (run scripts/doctor.ps1)")

    corpus = json.loads((REPO_ROOT / "benchmarks" / "corpus" / "corpus.json").read_text())
    by_id = {a["asset_id"]: a for a in corpus["assets"]}
    for aid in CLIPS:
        if aid not in by_id:
            fail("E_CORPUS", f"clip {aid} missing from benchmarks/corpus/corpus.json "
                             "(regenerate: python benchmarks/generate_corpus.py)")

    import psutil
    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        fail("E_DEVICE", "cuda requested but unavailable")
    dtype = torch.float32  # card rule: bf16 or fp32; fp32 on CPU. NEVER fp16.
    try:
        load_arg, revision, source = resolve_model_source(args.model_dir)
    except FileNotFoundError as exc:
        fail("E_MODEL_DIR", str(exc))
    results = {"model_id": MODEL_ID, "revision": REVISION, "model_source": source,
               "device": device, "torch_dtype": str(dtype),
               "preprocessing": {"video_input": {"text": "<|video|>"},
                                 "sampling": "processor default 1 FPS uniform "
                                             "(do_sample_frames; explicit fps=30 metadata)",
                                 "decode_backend": "pyav (av package) full-frame decode; "
                                                   "torchcodec unusable on this host, see doc",
                                 "segment_extraction": "ffmpeg accurate-seek re-encode, "
                                                       "libx264 320x240 30fps yuv420p"},
               "vectors": {}, "segments": [], "timings_s": {}, "checks": {}}
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / 1e6

    t0 = time.perf_counter()
    try:
        model = SentenceTransformer(load_arg, revision=revision, device=device,
                                    model_kwargs={"torch_dtype": dtype},
                                    config_kwargs={"audio_config": None})
    except Exception as exc:  # noqa: BLE001 -- surfaced, never silent
        fail("E_LOAD", f"model load failed [{type(exc).__name__}]: {exc}")
    results["timings_s"]["load"] = time.perf_counter() - t0
    if any("float16" in str(p.dtype) for p in model.parameters()):
        fail("E_DTYPE", "float16 detected (card forbids fp16)")

    workdir = Path(tempfile.mkdtemp(prefix="lf-videosmoke-"))
    timings = {"probe": [], "extract": [], "decode": [], "encode": []}
    vecs = {}
    try:
        from transformers.video_utils import load_video
        for aid in CLIPS:
            asset = by_id[aid]
            clip = REPO_ROOT / "benchmarks" / "corpus" / asset["rel_path"]
            t0 = time.perf_counter()
            measured = probe_duration_ms(ffprobe, clip)
            timings["probe"].append(time.perf_counter() - t0)
            declared = asset["duration_ms"]
            if abs(measured - declared) > 500:
                fail("E_DURATION", f"{aid}: manifest {declared}ms vs measured {measured}ms")
            for strategy, parts in STRATEGIES.items():
                for k in range(parts):
                    start = (measured * k) // parts
                    end = (measured * (k + 1)) // parts
                    seg = workdir / f"{aid}_{strategy}_{start}_{end}.mp4"
                    t0 = time.perf_counter()
                    run_cmd([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                             "-i", str(clip), "-ss", str(start / 1000),
                             "-t", str((end - start) / 1000),
                             "-r", "30", "-pix_fmt", "yuv420p",
                             "-c:v", "libx264", "-an", str(seg)],
                            f"extract {seg.name}")
                    timings["extract"].append(time.perf_counter() - t0)
                    # Official decode (transformers, pyav backend). ALL frames are
                    # handed over; the EmbeddingGemma2 video processor itself
                    # samples 1 FPS (its default) using explicit metadata, so
                    # container metadata gaps cannot desync frame indices.
                    # NOTE: str path required at runtime (a Path object is
                    # misread as a frame sequence); stub disagrees, runtime wins.
                    t0 = time.perf_counter()
                    try:
                        frames, _meta = load_video(str(seg), backend="pyav")
                    except Exception as exc:  # noqa: BLE001
                        fail("E_DECODE", f"pyav decode failed [{type(exc).__name__}]: {exc}")
                    timings["decode"].append(time.perf_counter() - t0)
                    n_frames = int(frames.shape[0])
                    if n_frames == 0:
                        fail("E_DECODE", f"zero frames decoded from {seg.name}")
                    # The processor's 1-FPS sampler needs fps AND duration;
                    # without duration it keeps every frame ( verified 180-frame
                    # 135 s encode). Explicit metadata keeps encodes at ~1 fps.
                    video_value = {"array": frames,
                                   "video_metadata": {"fps": 30,
                                                      "duration": n_frames / 30.0,
                                                      "total_num_frames": n_frames,
                                                      "frames_indices": list(range(n_frames))}}
                    t0 = time.perf_counter()
                    try:
                        emb = model.encode({"text": "<|video|>", "video": video_value})
                    except Exception as exc:  # noqa: BLE001
                        fail("E_VIDEO_BACKEND",
                             f"video encode failed [{type(exc).__name__}]: {exc}")
                    timings["encode"].append(time.perf_counter() - t0)
                    label = f"{aid}/{strategy}/[{start},{end}]"
                    vals = check_vector(label, emb, results)
                    vecs[label] = vals
                    print(f"  encoded {label} "
                          f"({timings['encode'][-1]:.1f}s from {n_frames} frames)",
                          flush=True)
                    results["segments"].append(
                        {"asset_id": aid, "strategy": strategy,
                         "start_ms": start, "end_ms": end,
                         "seg_file": seg.name, "vec": label})
        # Query ranking over all segment vectors (sequential, same session).
        ranking = []
        for text, exp_aid, (exp_s, exp_e) in [(q[0], q[1], q[2]) for q in QUERIES]:
            q_emb = [float(v) for v in model.encode(text, prompt_name="SearchQuery")]
            scored = sorted(
                ((cos(q_emb, vecs[s["vec"]]), s) for s in results["segments"]),
                key=lambda t: -t[0])
            ranking.append({"query": text, "expected": {"asset_id": exp_aid,
                                                        "start_ms": exp_s, "end_ms": exp_e},
                            "top3": [{"asset_id": s["asset_id"], "strategy": s["strategy"],
                                      "start_ms": s["start_ms"], "end_ms": s["end_ms"],
                                      "cos": round(score, 4)} for score, s in scored[:3]]})
            best = scored[0][1]
            hit = (best["asset_id"] == exp_aid and best["start_ms"] < exp_e
                   and best["end_ms"] > exp_s)
            ranking[-1]["top1_hit"] = bool(hit)
        results["ranking"] = ranking
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    import statistics
    results["timings_s"].update({
        "probe_mean": statistics.mean(timings["probe"]),
        "extract_mean": statistics.mean(timings["extract"]),
        "decode_mean": statistics.mean(timings["decode"]),
        "encode_mean": statistics.mean(timings["encode"]),
        "encode_n": len(timings["encode"])})
    rss_after = proc.memory_info().rss / 1e6
    results["memory_mb"] = {"rss_before": rss_before, "rss_after": rss_after,
                            "rss_delta": rss_after - rss_before}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"model: {MODEL_ID}@{REVISION[:7]} source={source} device={device}")
    print(f"segments={len(results['segments'])} "
          f"probe={results['timings_s']['probe_mean']*1000:.0f}ms "
          f"extract={results['timings_s']['extract_mean']*1000:.0f}ms "
          f"decode={results['timings_s']['decode_mean']*1000:.0f}ms "
          f"encode={results['timings_s']['encode_mean']:.1f}s")
    for r in ranking:
        top = r["top3"][0]
        print(f"query {r['query']!r}: top1 {top['asset_id']}[{top['start_ms']},{top['end_ms']}] "
              f"cos={top['cos']} hit={r['top1_hit']}")
    print(f"all segment vectors 768-d finite normalized; rss_delta={rss_after-rss_before:.0f} MB")
    print(f"results: {args.out}")
    print("SMOKE PASS" if all(r["top1_hit"] for r in ranking) else "SMOKE MIXED (see ranking)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
