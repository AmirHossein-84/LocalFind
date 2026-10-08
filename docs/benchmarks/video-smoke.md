# P0-T005 — Segment-level video embedding experiment

**Status (2026-10-08): PASS — executed on the target Windows 11 x64 host.**
`benchmarks/video_smoke.py` exited 0; machine-readable results in
`docs/benchmarks/video-smoke-results.json`. Two synthetic clips only — this
is feasibility evidence for the video path, not retrieval-quality proof
(that gate belongs to P0-T007).

## Setup

- Model: `google/embeddinggemma-2` @ `914f7f8`, **project-local**
  (`models/embeddinggemma-2/`, auto-resolved, `source=project-local`),
  fp32 CPU, text+image encoders. Run cache-only (`HF_HUB_OFFLINE=1`).
- Clips (P0-T003 corpus): `vid_001` two-tone red/blue 6000 ms,
  `vid_009` motion testsrc 5000 ms. Manifest durations re-verified with
  ffprobe at run time (mismatch > 500 ms aborts with `E_DURATION`).
- Segments: **halves** (2/clip) + **thirds** (3/clip) = 10 bounded
  millisecond intervals, cut with ffmpeg accurate-seek re-encode
  (libx264 320x240 30 fps). Temp segment files cleaned after the run.
- Video input (officially supported path): `{"text": "<|video|>",
  "video": {"array": frames, "video_metadata": {...}}}` — full-frame pyav
  decode, then the model's own processor samples **1 FPS** (its default).
  No whole-video single-vector shortcut; image-only segment encoding was
  never used.

## Results

| Check | Result |
|---|---|
| Segment vectors map to exact source/start/end | PASS — 10/10 with asset_id, strategy, [start,end], seg filename |
| 768-d finite normalized | PASS — all norms 1.0±1e-4 (observed max deviation 8.8e-8) |
| "red screen" → relevant interval | PASS — top1 `vid_001[0,3000]` cos 0.7371; thirds `[0,2000]` 0.7311 second |
| "blue screen" → relevant interval | PASS — top1 `vid_001[3000,6000]` cos 0.7467; thirds `[4000,6000]` 0.7465 second |
| "moving test pattern" → motion clip | PASS — top1 `vid_009[0,1666]` cos 0.7257 (whole clip relevant by design) |
| Halves vs thirds | Both rank correctly; thirds localize finer at 1.5× encode cost. No policy chosen here (P0-T008 decides with quality data) |
| Stage timings (means) | probe **65 ms**, extract **101 ms**, decode **46 ms**, encode **3.5 s**/segment (CPU fp32, ~2–3 sampled frames per 1.6–3 s segment at 1 FPS) |
| Memory | process RSS **+2,218 MB**; sequential inference, no concurrency |
| Failures on these clips | none — limitation: only 2 synthetic clips; real-world recall untested |

## Backend findings (recorded, no architecture change)

- sentence-transformers 6.x decodes `video` path inputs via **torchcodec**,
  whose native lib needs system FFmpeg 4–9 shared DLLs. This host's FFmpeg
  is a static build (no DLLs) → `E_VIDEO_BACKEND` ImportError. Installed
  `torchcodec 0.17.0+cpu`, still unloadable — same root cause.
- Fix within scope: decode with transformers' own `load_video(backend="pyav")`
  (`av 19.0.1`, bundled FFmpeg, venv-only install, re-pinned
  `requirements-smoke.txt`) and hand frames to the official `<|video|>`
  processor path. A same-pixels control is unnecessary here — segments are
  distinguished by construction and rank by content (middle third spanning
  the red/blue cut ranks between the pure halves, as expected).
- Gotcha fixed during the run: the processor needs `fps` **and `duration`**
  in `video_metadata` for its 1-FPS sampler; without `duration` it kept all
  180 frames (135 s encode). Explicit metadata restored ~1 fps sampling.
- `torchcodec` stays installed but unused on this host; P0-T010 packaging
  must decide the shipped decode backend (pyav bundles FFmpeg and works).

## Limits

- Exit code does not enforce retrieval: the script prints `SMOKE MIXED` on a
  miss but still returns 0; the PASS above rests on the observed 3/3 top-1
  hits in `video-smoke-results.json`, not on the exit code.
- Synthetic 2-clip smoke only; spoken-phrase retrieval NOT tested (no STT —
  visual evidence only, per R-08).
- CPU fp32 timings are feasibility readings, not latency targets.
- GPU path, bf16, and truncation sweeps belong to P0-T006/P1.
