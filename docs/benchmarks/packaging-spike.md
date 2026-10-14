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
2. Local packaged `embed_text` + `embed_image` offline with explicit
   `--model-dir` (excluded-torchvision binary), compat sigs match
   `eg2@914f7f8|text+image|p0t010-v1|…|768|fp32|l2`, truncation parity
   holds, stderr clean.
3. Record cold/warm timings, peak RAM, failure behavior; clean-machine
   model leg stays NOT RUN (no licensed model on runner).
4. Then check the `STEPS.md` P0-T010 box + Project State and commit
   the task completion per `AGENTS.md` §12.
