# LocalFind

Windows-first, privacy-preserving desktop search: **find local files and precise
visual moments using everyday language, without uploading personal content.**

> **Status: Phase 0 feasibility — not implemented.** There is no working app in
> this repository yet. The planning documents below are specifications, not proof
> that the proposed stack performs adequately.

## Consented local search

- You explicitly choose which folders are indexed; nothing is crawled silently
  (no whole-drive, Documents/Downloads, or network-share scans by default).
- Original files are **read-only**: indexing never modifies, moves, or deletes
  them. Index data, thumbnails, and caches live in separate app-owned
  directories and index cleanup never touches originals.
- Search works offline after the model is locally installed. Exact
  filename/path search must keep working even when the AI model is unavailable.
- Default logs and telemetry contain counts and error codes only — never file
  contents, queries, transcripts, vectors, or unredacted paths.

## Proposed stack (provisional, pending Phase 0 evidence)

| Area | Proposal | Reference |
|---|---|---|
| Desktop UI + host | Tauri 2 + Svelte/TypeScript, Rust core | PLAN.md ADR-001/ADR-002 |
| Metadata + lexical index | SQLite (WAL) + FTS5 | PLAN.md ADR-003 |
| Vector baseline | float32 BLOBs + exact cosine behind a `VectorIndex` port | PLAN.md ADR-004 |
| Media probing | Bundled FFmpeg/ffprobe candidate | PLAN.md ADR-005 |
| File watching | Rust `notify` adapter + periodic reconciliation | PLAN.md ADR-006 |
| Inference | Separate single Python worker over stdio JSON protocol | PLAN.md ADR-007 |
| Model (candidate) | `google/embeddinggemma-2` (768-dim native, bf16/fp32 only, never fp16) | PLAN.md §3 |
| Distribution | User-initiated model download, no background telemetry | PLAN.md ADR-008 |

All ADRs are **proposed** until validated by Phase 0 measurements.

## Documented unknowns (verified in Phase 0, not assumed)

- Compatible Python/PyTorch/sentence-transformers versions on Windows 11, and
  CPU vs GPU inference cost (PLAN.md Q-001, Q-002).
- Whether the Python sidecar can be packaged and started on a clean Windows VM,
  and its install footprint (Q-003).
- Truncation dimensions, segmentation policy, and vector-backend choice backed
  by retrieval-quality and latency benchmarks (Q-004, Q-005, Q-006).
- Model, library, FFmpeg, and codec redistribution/license obligations (Q-007).
- Phase 0 exit gate: `Recall@5 >= 0.75` on a small curated suite of ≥30 visual
  queries, or an explicit PIVOT/STOP decision (PLAN.md §2).

## Documents (authoritative)

- [PLAN.md](PLAN.md) — product scope, provisional architecture, ADRs, risks.
- [STEPS.md](STEPS.md) — task roadmap; **P0-T006** is the current task.
- [RULES.md](RULES.md) — enforceable constraints (user-data safety first).
- [AGENTS.md](AGENTS.md) — one-task-at-a-time agent workflow.

If instructions conflict: user-data safety and explicit user instruction, then
`RULES.md`, then accepted ADRs in `PLAN.md`, then the current task in
`STEPS.md`.

## Getting started

Nothing to build or run yet. Phase 0 feasibility prototypes (under
`benchmarks/`, explicitly not production code) begin at `P0-T002` once this
foundation task (`P0-T001`) is accepted. See `STEPS.md` Project State for the
current task — do not jump ahead to later phases.

```powershell
git status --short
```

## Contributing

- Work one `STEPS.md` task at a time; keep diffs focused (RULES.md R-25).
- Test media must be synthetic, consented, or redistributable — never commit
  real user files, private embeddings, model weights, secrets, or machine
  paths (RULES.md R-24, R-26).
- Report exact test commands and results; never claim an unexecuted check
  passed (RULES.md R-23, R-29).
