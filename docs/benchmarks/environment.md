# Environment record — P0-T002 (Windows toolchain reproducibility)

**Date:** 2026-10-08 · **Task:** P0-T002 · **Host role:** development machine
(a Windows 11 x64 target of the class PLAN.md specifies as baseline)

> Privacy note: this file deliberately contains **no** usernames, computer
> names, home-directory paths, or secrets — only OS/hardware model identifiers
> and tool version strings, as required by STEPS.md P0-T002 and RULES.md R-04.

## Host

| Item | Value |
|---|---|
| OS | Microsoft Windows 11 Pro, build 26300, 64-bit (`Win32_OperatingSystem`) |
| CPU | 13th Gen Intel Core i7-13620H — 10 cores / 16 threads |
| RAM | 15.6 GB (16,788,283,392 bytes) |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU, driver 610.74, 6144 MiB (`nvidia-smi`) |
| CUDA toolkit (`nvcc`) | not installed — driver-level CUDA UMD 13.3 only; CPU + driver-GPU runs expected |
| Shell | PowerShell 7.6.6 |

## Toolchain (as reported by `scripts/doctor.ps1`)

| Tool | Result | Detail |
|---|---|---|
| OS check | OK | Windows 11 Pro build 26300 x64 |
| Python | OK | `python --version` → **Python 3.12.10** (meets >= 3.10 gate) |
| Rust (`rustc`/`cargo`) | OK (since 2026-10-08, see addendum) | rustc 1.99.0 / cargo 1.99.0, stable x86_64-pc-windows-msvc |
| Node | OK | v24.18.0 (meets >= v20 gate) |
| pnpm | OK | 11.25.0 (preferred manager present; npm 12.0.2 also installed) |
| ffmpeg | OK | N-121271-g74115b017c-20250930 (2025-09-30 nightly-type build) |
| ffprobe | OK | N-121271-g74115b017c-20250930 (bundled with the above) |
| git | OK | 2.54.0.windows.1 |
| pip (for `python` 3.12) | info | 26.2.1 |

## Doctor runs (evidence)

1. Full-PATH run on this host:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force; & scripts/doctor.ps1`
   → all checks OK **except Rust**; exit code **1** with an actionable
   `winget install Rustlang.Rustup` hint. Output contained no usernames or paths.
2. Degraded-PATH run (`$env:PATH` reduced to `C:\Windows\System32;C:\Windows`,
   restored afterwards in `finally`) → 6 `MISSING` failures (Rust, Node,
   PackageMgr, ffmpeg, ffprobe, git), each with an install hint; exit code **1**.
   This proves missing prerequisites yield actionable failures rather than
   silent passes. (`py.exe` lives in `C:\Windows`, so the Python check still
   resolved via the launcher in that run — by design, the launcher is a
   legitimate interpreter source.)

## Missing prerequisites

- **Rust toolchain: RESOLVED 2026-10-08.** Owner installed
  `Rustlang.Rustup 1.29.1` via winget (no agent install performed).
  Verified: `rustc 1.99.0`, `cargo 1.99.0`, toolchain
  `stable-x86_64-pc-windows-msvc` (default). Terminals opened before the
  install need a PATH refresh or restart — cargo was in the user registry
  PATH but absent from stale processes. Re-ran doctor after refresh:
  **all required checks passed, exit code 0** (see run 3 below).
- **MSVC linker (`cl.exe`): not present.** Required for Tauri/MSVC builds at
  P1-T001 (not a Phase 0 blocker). Install the Visual Studio 2022 C++ build
  workload before P1-T001.

## Addendum 2026-10-08 — doctor run 3 (post-Rust-install)

`rustc --version` → `rustc 1.99.0 (b940084d7 2026-09-28)`;
`cargo --version` → `cargo 1.99.0 (5f94df478 2026-08-27)`;
`scripts/doctor.ps1` → `doctor: all required checks passed.`, exit code **0**.
No system-wide software was installed or modified by the agent; only the
process-scoped `PATH` was refreshed from the registry for verification.

## Notes for follow-on tasks

- **Multiple Python interpreters exist on this host.** PATH `python` is 3.12.10;
  the `py` launcher additionally offers 3.14 (default), 3.10, and uv-managed
  3.14/3.11/3.10 builds. P0-T004 must **pin one interpreter + dependency set**
  in its experiment files rather than assuming `python`.
- **CUDA toolkit absent** (`nvcc` missing) while an NVIDIA GPU + driver are
  present. P0-T004/P0-T006 must record whether they ran CPU-only or with a
  GPU backend, and must not assume `nvcc`/CUDA toolkit availability.
- **npm reports 12.0.2** alongside Node v24.18.0 on this host (verbatim reading).
  pnpm 11.25.0 is present and is the preferred manager; lockfiles from P1
  onward pin the actual package manager.
- Reproduce this record any time with:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/doctor.ps1
  ```

## Impact assessment

- P0-T002 acceptance (doctor runs on Win11 x64; missing prereqs fail
  actionably; versions saved sanitized): **met**.
- Former Rust gap: **resolved 2026-10-08** (rustc/cargo 1.99.0 verified).
  Remaining pre-P1 gap: MSVC Build Tools (`cl.exe` absent) — needed at P1-T001.
- No architecture decision is finalized by this task; all PLAN.md ADRs remain
  **PROPOSED** pending Phase 0 benchmark evidence.
