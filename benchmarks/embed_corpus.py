#!/usr/bin/env python3
"""P0-T007 step 1: embed the P0-T003 corpus to a reusable native-768-d cache.

Writes benchmarks/corpus/vectors/{images,segments,queries}.npz (float32) plus
vectors-manifest.json (unit keys, sigs, corpus/query-set versions). Truncation
to 512/256/128 happens later in evaluate.py (post-hoc, no re-encoding).

Split runs to fit modest hosts / tool timeouts:
    python benchmarks/embed_corpus.py --modality image   # 100 images
    python benchmarks/embed_corpus.py --modality video   # 20 clips x halves
    python benchmarks/embed_corpus.py --modality query   # 39 texts
--resume skips units already cached. Sequential inference only.

Exit 0 = cached, 1 = failure with E_* code. Offline (HF_HUB_OFFLINE=1 safe).
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from compat import check_vector_values, make_sig, sig_string
from local_model import MODEL_ID, REVISION, resolve_model_source

REPO_ROOT = Path(__file__).resolve().parent.parent
VEC_DIR = REPO_ROOT / "benchmarks" / "corpus" / "vectors"
NATIVE_DIM = 768
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0-T007 corpus embedder.")
    parser.add_argument("--modality", required=True,
                        choices=["image", "video", "query", "keyframe"]) 
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    corpus = json.loads((REPO_ROOT / "benchmarks" / "corpus" / "corpus.json").read_text())
    queries_doc = json.loads((REPO_ROOT / "benchmarks" / "corpus" / "queries.json").read_text())
    if queries_doc.get("corpus_id") != corpus.get("corpus_id"):
        fail("E_CORPUS_MISMATCH", "queries/corpus corpus_id differ")
    VEC_DIR.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    if args.device == "cuda" and not torch.cuda.is_available():
        fail("E_DEVICE", "cuda requested but unavailable")
    try:
        load_arg, revision, source = resolve_model_source(args.model_dir)
    except FileNotFoundError as exc:
        fail("E_MODEL_DIR", str(exc))
    t0 = time.perf_counter()
    try:
        model = SentenceTransformer(load_arg, revision=revision, device=args.device,
                                    model_kwargs={"torch_dtype": torch.float32},
                                    config_kwargs={"audio_config": None})
    except Exception as exc:  # noqa: BLE001
        fail("E_LOAD", f"model load failed [{type(exc).__name__}]: {exc}")
    print(f"loaded ({time.perf_counter()-t0:.1f}s, source={source})", flush=True)
    if any("float16" in str(p.dtype) for p in model.parameters()):
        fail("E_DTYPE", "float16 parameters present (card forbids fp16)")

    out_path = VEC_DIR / f"{args.modality}.npz"
    manifest_path = VEC_DIR / "vectors-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {
        "corpus_id": corpus["corpus_id"], "corpus_version": corpus["version"],
        "query_set_id": queries_doc["query_set_id"], "query_version": queries_doc["version"],
        "model_id": MODEL_ID, "revision": REVISION, "model_source": source,
        "torch_dtype": "torch.float32", "native_dim": NATIVE_DIM, "units": {}}
    cached = {}
    if args.resume and out_path.is_file():
        with np.load(out_path) as z:
            cached = {k: z[k] for k in z.files}

    from PIL import Image
    units, vecs = {}, {}
    if args.modality == "image":
        assets = [a for a in corpus["assets"] if a["modality"] == "image"]
        sig = make_sig("image", 768, prep="p0t007-v1")
        for i, asset in enumerate(assets):
            key = f"image:{asset['asset_id']}"
            if args.resume and key in cached:
                vecs[key] = cached[key]
                units[key] = manifest["units"].get(key, {})
                continue
            img = Image.open(REPO_ROOT / "benchmarks" / "corpus" / asset["rel_path"]).convert("RGB")
            emb = np.asarray(model.encode(img), dtype=np.float32)
            check_vector_values([float(v) for v in emb], NATIVE_DIM)
            vecs[key] = emb
            units[key] = {"asset_id": asset["asset_id"], "sig": sig_string(sig)}
            if (i + 1) % 20 == 0:
                print(f"  images {i+1}/{len(assets)}", flush=True)
    elif args.modality == "video":
        from transformers.video_utils import load_video
        assets = [a for a in corpus["assets"] if a["modality"] == "video"]
        sig = make_sig("video", 768, prep="p0t007-v1", segment="halves")
        workdir = Path(tempfile.mkdtemp(prefix="lf-embed-"))
        try:
            n = 0
            for asset in assets:
                total = asset["duration_ms"]
                bounds = [(0, total // 2), (total // 2, total)]
                clip = REPO_ROOT / "benchmarks" / "corpus" / asset["rel_path"]
                for (s, e) in bounds:
                    key = f"video:{asset['asset_id']}:{s}-{e}"
                    if args.resume and key in cached:
                        vecs[key] = cached[key]
                        units[key] = manifest["units"].get(key, {})
                        continue
                    seg = workdir / f"{asset['asset_id']}_{s}_{e}.mp4"
                    run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                             "-i", str(clip), "-ss", str(s / 1000), "-t", str((e - s) / 1000),
                             "-r", "30", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                             "-an", str(seg)], f"extract {seg.name}")
                    # NOTE: str path required at runtime (Path misread as frames).
                    frames, _m = load_video(str(seg), backend="pyav")
                    nf = int(frames.shape[0])
                    emb = np.asarray(model.encode(
                        {"text": "<|video|>",
                         "video": {"array": frames,
                                   "video_metadata": {"fps": 30, "duration": nf / 30.0,
                                                      "total_num_frames": nf,
                                                      "frames_indices": list(range(nf))}}}),
                        dtype=np.float32)
                    check_vector_values([float(v) for v in emb], NATIVE_DIM)
                    vecs[key] = emb
                    units[key] = {"asset_id": asset["asset_id"], "start_ms": s,
                                  "end_ms": e, "sig": sig_string(sig)}
                    n += 1
                    if n % 10 == 0:
                        print(f"  segments {n}/{len(assets)*2}", flush=True)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    else:
        if args.modality == "query":
            sig = make_sig("text-query", 768, prep="p0t007-v1")
            for q in queries_doc["queries"]:
                key = f"query:{q['query_id']}"
                if args.resume and key in cached:
                    vecs[key] = cached[key]
                    units[key] = manifest["units"].get(key, {})
                    continue
                emb = np.asarray(model.encode(q["text"], prompt_name="SearchQuery"),
                                 dtype=np.float32)
                check_vector_values([float(v) for v in emb], NATIVE_DIM)
                vecs[key] = emb
                units[key] = {"query_id": q["query_id"], "sig": sig_string(sig)}
            print(f"  queries {len(vecs)}", flush=True)
        elif args.modality == "keyframe":
            # Baseline support: middle frame of each clip as ONE image-path vector
            # (whole-video single-vector shortcut: the approach R-08 rejects for
            # production, measured here as a baseline only).
            from PIL import Image as PILImage
            assets = [a for a in corpus["assets"] if a["modality"] == "video"]
            sig = make_sig("image", 768, prep="p0t007-v1")
            workdir = Path(tempfile.mkdtemp(prefix="lf-keyframe-"))
            try:
                for i, asset in enumerate(assets):
                    key = f"keyframe:{asset['asset_id']}"
                    if args.resume and key in cached:
                        vecs[key] = cached[key]
                        units[key] = manifest["units"].get(key, {})
                        continue
                    clip = REPO_ROOT / "benchmarks" / "corpus" / asset["rel_path"]
                    frame = workdir / f"{asset['asset_id']}_mid.png"
                    run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                             "-ss", str(asset["duration_ms"] / 2000),
                             "-i", str(clip), "-frames:v", "1", str(frame)],
                            f"keyframe {asset['asset_id']}")
                    emb = np.asarray(model.encode(
                        PILImage.open(frame).convert("RGB")), dtype=np.float32)
                    check_vector_values([float(v) for v in emb], NATIVE_DIM)
                    vecs[key] = emb
                    units[key] = {"asset_id": asset["asset_id"], "sig": sig_string(sig)}
                    if (i + 1) % 10 == 0:
                        print(f"  keyframes {i+1}/{len(assets)}", flush=True)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)

    manifest["units"].update(units)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    np.savez(out_path, **vecs)
    print(f"cached {len(vecs)} {args.modality} vectors -> {out_path.name} "
          f"({out_path.stat().st_size} bytes)")
    print("EMBED PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
