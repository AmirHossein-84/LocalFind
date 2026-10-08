# P0-T004 — EmbeddingGemma-2 text/image smoke experiment

**Status (2026-10-08): PASS — executed on the target Windows 11 x64 host.**
`benchmarks/model_smoke.py` exited 0; machine-readable results in
`docs/benchmarks/model-smoke-results.json`. Synthetic ranking below is smoke
evidence only, not retrieval-quality proof (that gate belongs to P0-T007).

## Official source (read 2026-10-08)

- Model card: https://huggingface.co/google/embeddinggemma-2
- Card facts used: 740M params (270M text + 170M vision + 300M audio);
  native 768-d, MRL 512/256/128 (re-normalize after truncation; query and
  corpus dims must match); 8,192-token shared budget; video default 1 FPS;
  asymmetric prefixes (`SearchQuery` query / `Document` doc,
  `title: {title} | text: {content}`, `title: none` fallback);
  selective encoders (`config_kwargs={"audio_config": None}` → 440M
  text+image); **bf16 or fp32 only, NEVER fp16** (NaNs/silent degradation);
  safetensors BF16 checkpoint; license field `apache-2.0` **plus**
  Gemma Terms/Prohibited-Use Policy links — full audit is P0-T011, flagged
  `VERIFY LICENSE` there. Quick-start: `pip install -U sentence-transformers
  transformers`; `model.encode(text, prompt_name="SearchQuery")`.

## R-21 dependency assessment (experiment-only, not shipped runtime)

| Package (pinned) | Need | Alternatives considered | License | Size/risk note |
|---|---|---|---|---|
| torch 2.14.1+cpu | inference backend | CUDA torch (rejected: +2.5 GB, unneeded for CPU smoke) | BSD-3 | ~200 MB CPU wheel |
| sentence-transformers 6.1.0 | card's official encode API + prompt registry | raw transformers (more code, same weights) | Apache-2.0 | pulls scipy/sklearn (unused here) |
| transformers 5.19.0 | model classes/tokenizer | — (required by ST) | Apache-2.0 | — |
| accelerate 1.15.0 | card-recommended loader dep | — | Apache-2.0 | — |
| Pillow 12.3.0 | image open for encode | — | HPND | — |
| huggingface_hub 1.33.0 | pinned download + cache | manual browser download (offered to owner) | Apache-2.0 | — |
| psutil 7.2.2 | peak-RSS measurement | stdlib has no RSS API | BSD-3 | tiny |
| torchvision 0.29.1+cpu | **required by `EmbeddingGemma2Processor`** (found via debug: first run failed `E_LOAD` without it) | no alternative — processor import hard-requires it | BSD-3 | CPU wheel, ~50 MB |

Full pin set: `benchmarks/requirements-smoke.txt` (`pip freeze`, 45 lines).
Venv lives at `benchmarks/.venv/` (git-ignored); model cache is the default
HF cache outside the repo (R-24/R-26). No system-wide installs by the agent.

## Environment at setup

- Interpreter: `benchmarks/.venv` Python 3.12.10, pip 26.2.1.
- Host: Win11 Pro build 26300 x64, i7-13620H, 15.6 GB RAM (**1.8 GB free** —
  constrains this run), RTX 3050 6 GB (5.4 GB free, driver 610.74), 29.7 GB
  free on C:. CUDA toolkit absent (`nvcc` missing) → CPU-first plan; GPU
  comparison only if practical (torch installed is CPU-only).
- Model access (read-only API, no download): `gated: False` (no login needed),
  revision `914f7f8…`, `model.safetensors` = **1,488,915,288 bytes (~1.49 GB)**
  + KBs of configs. License field `apache-2.0`, Gemma policy links noted.
- Config for this host: **text+image encoders only, fp32, CPU** (440M ≈
  1.76 GB weights; full 740M + torch overhead risks OOM with 1.8 GB free).
  Rationale recorded; not an architecture decision (ADRs stay PROPOSED).

## Reproduce (owner steps — model download + run)

```powershell
# 1. Download (~1.5 GB, one-time, resumable, into the HF cache outside git):
benchmarks\.venv\Scripts\hf download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b
#    …or browser: https://huggingface.co/google/embeddinggemma-2/tree/main
# 2. Run the smoke experiment (cache-only; proves no network use):
$env:HF_HUB_OFFLINE = '1'; benchmarks\.venv\Scripts\python benchmarks/model_smoke.py
```

The script encodes 1 query (`SearchQuery`), 1 document (`Document` + manual
`title: none` equivalence), 2 fixture images, asserts 768-d finite unit-norm
vectors and no fp16, records cross-modal cosines/ranking, cold/warm timings,
RSS delta, and writes `docs/benchmarks/model-smoke-results.json`. Pre-download
dry run (`HF_HUB_OFFLINE=1`) correctly fails fast with `E_LOAD`, exit 1.

## Results — 2026-10-08, host i7-13620H / 16 GB / RTX 3050 (run CPU-only)

Command: `benchmarks/.venv/Scripts/python benchmarks/model_smoke.py`
(cache-only: run with `HF_HUB_OFFLINE=1` to prove no network use;
config text+image encoders, fp32, CPU).
Exit **0 — SMOKE PASS** (final run; numbers below from
`model-smoke-results.json`). An earlier run loaded in 18.4 s cold / 4.6 s
per image warm — first-load and warm timings vary with OS file cache and
machine load; treat all timings as single-sample smoke readings, not
benchmarks.

| Check | Result |
|---|---|
| Model load (text+image, fp32, CPU) | PASS — **4.2 s** warm-cache load from local snapshot (loader: 624 tensors) |
| Query embedding 768-d finite normalized | PASS — dim 768, norm 1.0000000 |
| Document embedding + prefix rule (`Document` ≡ `title: none`) | PASS — cos **1.0000** on the single doc pair tested (single-sample check, not a general proof) |
| Image embeddings x2 finite normalized | PASS — both 768-d, unit norm; inputs are decoded RGB (see vision control) |
| Vision-path control (same pixels, two temp names) | PASS — cos **1.0000** (pixels determine output); red-vs-bars cos 0.77 (content differs) |
| Prefix effect (query vs document) | prefixed cos **0.7854** vs unprefixed **0.7054** (+0.08 for the card's prefix) |
| Cross-modal ranking ("solid red image") | red **0.7549** > bars **0.6093** → correct order (smoke evidence only) |
| Warm timings (mean of 5) | text query **54 ms**, image **3.6 s** (CPU fp32; far above the PLAN §2 aspirational 750 ms end-to-end — expected pre-optimization, recorded for P0-T006/P1) |
| Peak RSS delta (decimal MB) | **+2,102 MB** process RSS (fits the 16 GB host with care; full 740M + GPU path untested) |
| No-fp16 assertion | PASS — all params `torch.float32` |
| GPU comparison | NOT RUN — CPU-only torch installed; no CUDA runtime to compare against (documented, not a failure) |

Review-driven fixes (independent read-only review caught real issues):
- First results run encoded image **path strings**; review questioned whether
  the vision tower ran. Verified empirically: same pixels under different
  temp names embed identically (cos 1.0), so str paths AND PIL images both
  traverse vision — the original ranking stands. The script now encodes
  explicit RGB and enforces the same-pixels control (`E_VISION_PATH`).
- First run failed `E_LOAD [ModuleNotFoundError]: Could not import module
  'EmbeddingGemma2Processor'`; debug verbosity showed the original error
  `No module named 'torchvision'`. Fix: installed `torchvision 0.29.1+cpu`
  (CPU index, matching torch 2.14.1), re-pinned `requirements-smoke.txt`.
  No architecture change in either case.

## Project-local model directory (P0-T005, 2026-10-08)

- **Path:** `models/embeddinggemma-2/` (repo-relative; git-ignored via
  `/models/` — verified with `git check-ignore`; never commit weights).
- **Source:** verified copy of the HF cache snapshot for revision
  `914f7f8…` (snapshot located via `scan_cache_dir`, not assumed paths).
- **Contents:** 15 files, **1,525,810,123 bytes** — `model.safetensors`
  (1,488,915,288 B), `config.json`, `tokenizer.{json,model,_config}`,
  `processor_config.json`, `preprocessor_config.json`, ST modules
  (`modules.json`, `1_Pooling/`, `2_Normalize/`), chat template, READMEs.
- **Integrity:** every file SHA-256 + byte size matches the HF source;
  symlinks resolved to real files (dest contains 0 symlinks). Original HF
  cache preserved untouched.
- **Setup (fresh machine):** `hf download google/embeddinggemma-2
  --revision 914f7f8…` then copy the snapshot to `models/embeddinggemma-2/`
  (or point `--model-dir` at any snapshot-ready dir).
- **Resolution:** `benchmarks/local_model.py::resolve_model_source()` —
  explicit `--model-dir` > `models/embeddinggemma-2/` when snapshot-ready >
  HF model ID + revision fallback. No machine-specific absolute paths.
- **Offline proof:** full `model_smoke.py` re-run with `HF_HUB_OFFLINE=1`
  and `HF_HOME` pointed at an empty temp dir → SMOKE PASS, byte-identical
  similarities (0.7854/0.7054/0.7549/0.6093), `source=explicit-dir`.
- **Licensing:** card field `apache-2.0`, but Gemma Terms + Prohibited-Use
  Policy also apply — flagged `VERIFY LICENSE` for the P0-T011 audit. The
  local copy inherits the same terms; redistribution is NOT cleared by this task.

## Limits of this experiment

- Synthetic 2-image ranking is smoke evidence only, NOT retrieval-quality
  proof (quality gate belongs to P0-T007 on the 120-asset corpus).
- No truncation/dtype sweep here (P0-T006), no video (P0-T005), no Vt
  backend (P0-T009), no packaging (P0-T010).
