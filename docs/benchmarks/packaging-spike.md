# P0-T010 — Inference sidecar packaging spike

**Status (2026-10-14): IN PROGRESS.** Phase-0 prototype evidence, not
production code. P0-T010 acceptance is **not met yet** (`STEPS.md`
checkbox stays unchecked until a packaged artifact passes offline
embed + measurements below).

Prototype: `benchmarks/packaging/` (`sidecar.py`, `host_probe.py`,
`test_sidecar_protocol.py`). Protocol v1.0: newline-delimited JSON on
stdin/stdout (stdout is protocol-only, stderr carries logs);
ops `handshake | embed_text | embed_image | shutdown`; versioned
requests, typed `E_*` errors, input bounds, timeouts enforced by the
host, offline-only model load (`local_files_only=True`,
`HF_HUB_OFFLINE=1`). No TCP server (ADR-007 stdio shape).

## CI validation — packaging tests run on GitHub Actions

Packaging validation runs on the GitHub workflow
`.github/workflows/packaging-spike.yml` (manual `workflow_dispatch`
trigger, to control Windows-runner minute cost). **Run the tests there,
not only on the dev machine:**

```text
gh workflow run p0-t010-packaging-spike
```

The workflow uses a clean `windows-latest` VM (no project venv, empty
HF cache) and runs, in order:

1. No-model protocol suite: `python -m unittest
   benchmarks.packaging.test_sidecar_protocol -v` (8 tests, stdlib
   only, no model download).
2. PyInstaller `-D` build from the pinned experiment deps (torch CPU)
   with `--exclude-module torchvision torchcodec torchaudio av`.
3. Packaged `handshake` against the built exe (must answer fast with
   `model.loaded=false`).
4. Size report + artifact upload (build log always; dist
   best-effort — ~333 MB may exceed plan storage limits).

**NOT RUN in CI:** `embed_text`/`embed_image` against
`google/embeddinggemma-2` — the model is access-gated (license
acceptance + token) and ~1.5 GB, so it is never downloaded in CI.
Model-inference evidence stays local-only (dev probe below) and is
marked NOT RUN for the clean-VM leg until a licensed local model is
available to the runner.

## Local evidence so far

Dev probe (`docs/benchmarks/packaging-probe-dev.json`, `--isolated-hf`,
venv vars stripped): **PROBE PASS** — startup 73 ms, text cold 12.2 s
(load 4.2 s), text warm 49 ms, image 3.8 s, 256-d truncation cos 1.0,
exit 0, stderr clean (no query text / full paths, R-04).

Packaged artifact `D:/lf-p010/dist/lf-sidecar` (PyInstaller `-D`,
build15, **333 MB / 7,191 files**, model external via `--model-dir`):
handshake **PASS** standalone (fast, `loaded=false`); `embed_text`
**FAIL** with `E_INTERNAL:ModuleNotFoundError`, root cause
`operator torchvision::nms does not exist` — the bundle contains a
partial `torchvision` (Python files, no native ops), so
`transformers` attempts a torchvision import and fails. `transformers`
imports fine with torchvision absent, so the fix under test is
excluding `torchvision torchcodec torchaudio av` from the bundle
(text/image paths use PIL + torch, not torchvision ops).

A local rebuild with those excludes stalled ~8 h in PyInstaller's DLL
scan (log frozen, no build CPU) and was abandoned — no process left
running. Packaging validation moved to the CI workflow above, which
also satisfies the STEPS.md "clean Windows VM" leg.

## Second packaging bug: transformers lazy-scan vs frozen `__file__`

With torchvision excluded, the CI binary's `embed_text` failed with
`E_INTERNAL:FileNotFoundError:errno-2` (reproduced locally against the
downloaded CI dist). Debug traceback (`LOCALFIND_SIDECAR_DEBUG=1`):

- `sentence_transformers` import → `transformers.models/__init__.py:552`
  builds its lazy import map via `create_import_structure_from_path`,
  which does `os.scandir(module_path)` after an
  `if os.path.isfile(module_path): module_path = dirname(...)` fallback.
- Frozen, `transformers.models.__file__` is a nonexistent
  `.../transformers/models/__init__.pyc` (code runs from the PYZ), so
  the fallback is skipped and scandir raises `FileNotFoundError`.
  In dev, `__file__` is the real `__init__.py` and the fallback works.
- Fix (build-time only, no runtime bundle writes): the single `models/`
  stub proved the mechanism but the same lazy scan runs in every
  `transformers/models/*/` subpackage (first failure past `models/`:
  `gemma/__init__.pyc`). The workflow now mirrors a 0-byte
  `__init__.pyc` next to **every** `__init__.py` in the transformers
  tree (`stage-pyc/`, pure-pwsh loop, ~700 files). Stubs are never
  executed (PYZ wins the import); they only let the `isfile` fallback
  find the real data tree (`--collect-all`), after which lazy imports
  resolve through the frozen importer. Pinned to transformers
  5.19.0 behavior — re-verify if the pin changes.
- Also fixed: `LOCALFIND_SIDECAR_DEBUG=2` was dead code (nested inside
  an `== "1"` check); the gate is now `in ("1", "2")`.
- Follow-up (not done): the `--add-data transformers-full` overlay
  likely duplicates what `--collect-all` already ships (CI dist
  716.8 MB vs local build15 333 MB). Shrinking the bundle is deferred
  until embed passes.

## Alternatives assessed (spike scope)

- **PyInstaller `-D` (directory): chosen.** Proven handshake, 333 MB,
  UPX warnings benign (no UPX installed, falls back uncompressed).
- **PyInstaller `--onefile`: rejected** for the spike — slower cold
  start (full unpack per launch), worse for torch-sized bundles.
- **Nuitka: blocked** — requires a C compiler; `cl.exe`/MSVC absent on
  the dev machine and system-wide installs need approval (not
  requested). Not evaluated beyond the prerequisite check.
- **Embeddable Python / zipapp + venv: rejected** — still requires a
  Python runtime on the target; does not meet the "no separately
  installed Python" goal.
- **ONNX / native runtime: out of scope** — needs model conversion +
  quality re-validation and an ADR-002 change via Architecture Review
  Queue. Not started.

## Prerequisites verified

Win11 Pro build 26300 x64 · i7-13620H · 16 GB · Python 3.12.10 ·
Rust 1.99 present · `cl.exe` absent (Nuitka blocked) · torch
2.14.1+cpu / transformers 5.19.0 / sentence-transformers 6.1.0 /
PyInstaller 6.10.0 (pinned experiment venv). `scripts/doctor.ps1`
has a pre-existing parse error under Windows PowerShell 5.1
(line 195, em-dash); use `pwsh`. No system-wide installs performed.

## Pending for acceptance

1. CI workflow green: protocol 8/8 + packaged handshake on clean VM.
   Partial CI evidence in hand (run 38519715293, `windows-2025` runner):
   deps install PASS, protocol 8/8 PASS, PyInstaller `-D`
   (torchvision/torchcodec/torchaudio/av excluded) `BUILD_EXIT:0`,
   packaged handshake `ok:true` with `loaded:false` (`HANDSHAKE_EXIT:0`),
   dist **716.8 MB** (CI `--collect-all` set pulls more than local
   build15's 333 MB). Only the artifact-upload step failed
   (workspace-path rule, since fixed by zipping inside the workspace).
2. Local packaged `embed_text` + `embed_image` offline with explicit
   `--model-dir` (excluded-torchvision binary), compat sigs match
   `eg2@914f7f8|text+image|p0t010-v1|…|768|fp32|l2`, truncation parity
   holds, stderr clean.
3. Record cold/warm timings, peak RAM, failure behavior; clean-machine
   model leg stays NOT RUN (no licensed model on runner).
4. Then check the `STEPS.md` P0-T010 box + Project State and commit
   the task completion per `AGENTS.md` §12.
