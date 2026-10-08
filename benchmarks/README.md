# Phase-0 evaluation corpus (P0-T003)

Synthetic, fully reproducible fixtures for testing LocalFind's future
multimodal retrieval engine. **No real user files, no downloads, no model.**
Everything here is generated locally with FFmpeg or hand-written; every file
is `synthetic-CC0` unless its manifest entry says otherwise.

> What this is NOT: retrieval scores, model accuracy claims, or a production
> dataset. It is the *measuring stick* P0-T007 will use. Framework readiness
> (schemas + validator + generator) is done; the bulk media below is
> generated, not committed, by design.

## Files

| Path | Committed? | Purpose |
|---|---|---|
| `corpus.schema.json` | yes | Manifest format (asset ID, modality, rel path, duration, license/distribute flags) |
| `queries.schema.json` | yes | Judgment format (query ID, text, class, expectations, timestamp ranges, grades) |
| `corpus.example.json` | yes | 3-asset example pointing at `tests/fixtures/samples/` |
| `queries.example.json` | yes | 6-query example (visual, exact_filename, negative) |
| `generate_corpus.py` | yes | Deterministic generator: 100 images + 20 videos + 39 judgments |
| `validate_corpus.py` | yes | Fixture loader / validator (stdlib only, exits 1 on any error) |
| `corpus/` | **no** (git-ignored) | Generated bulk media + `corpus.json` + `queries.json` |

## Quick start (fresh machine)

```powershell
python benchmarks/generate_corpus.py
python benchmarks/validate_corpus.py --corpus benchmarks/corpus/corpus.json --queries benchmarks/corpus/queries.json --media-root benchmarks/corpus
python benchmarks/validate_corpus.py --corpus benchmarks/corpus.example.json --queries benchmarks/queries.example.json --media-root tests/fixtures/samples
```

Requires `ffmpeg`/`ffprobe` on PATH (`scripts/doctor.ps1` checks this).
Generation takes ~2 minutes; output is deterministic given the same script
version (cellauto pins `random_seed=42`; exact bytes may still vary with the
FFmpeg build, which is why `sha256` values are recorded locally, not asserted
cross-machine).

## Schema summary

**Corpus manifest** (`corpus_id`, `version`, `assets[]`): each asset has a
stable `asset_id` (`img_001`…), `modality` (`image`|`video`), `rel_path`
(forward slashes, relative, no `..`), dimensions, `duration_ms` (**required**
for video, forbidden for images), `license`, `distribute` (commit/share only
when true), `provenance`, human `description` (judgment ground truth), `tags`,
and optional `generator` recipe + `sha256`.

**Judgments** (`query_set_id`, `version`, `corpus_id`, `queries[]`): each query
has stable `query_id` (`q001`…), literal `text`, `query_class`
(`visual` | `exact_filename` | `semantic_image` | `negative`), `modality`,
and `expected[]` entries (`asset_id`, `grade` 1–2, video-only `start_ms` /
`end_ms` with `start < end <= duration`, informative `tolerance_ms` enforced
later by P0-T007). Negative queries MUST have empty `expected[]`; every
`asset_id` must resolve; IDs and paths must be unique. No model-specific
fields anywhere (evaluation stays model-agnostic per R-07).

Current generated set: **120 assets** (100 image, 20 video incl. 8 two-tone
clips with per-half judgments, motion clips, 1s shorts, 12s segmentation
material) and **39 queries** (31 `visual`, 2 `exact_filename`, 2
`semantic_image`, 4 `negative` incl. near-miss grade-1 and difficult cases).

## Supplying private fixtures (other developers)

Bulk or personal media is NEVER committed. To evaluate on your own files:

1. Copy (do not move) consented files under `benchmarks/corpus-private/`
   (git-ignored; see `.gitignore`). Keep originals untouched (R-01).
2. Write a manifest + judgments following the schemas above with
   `"distribute": false`, `"license": "private-do-not-share"`, and
   `"provenance": "private-consented"`. Paths stay relative to your private root.
3. Validate: `python benchmarks/validate_corpus.py --corpus <yours>.json --queries <yours>.json --media-root benchmarks/corpus-private`.
4. Report only aggregate metrics + corpus manifest version in `docs/benchmarks/`
   (never filenames, transcripts, or vectors — R-03/R-04).

## Adding media / judgments (procedure)

1. Prefer extending `generate_corpus.py` (synthetic, reproducible, reviewable).
2. Give every asset/query the next free stable ID; never reuse a retired ID.
3. Write descriptions first (ground truth), judgments second, from the
   description — not from any retriever's output (no cherry-picking).
4. Include negatives and near-misses, not just easy positives.
5. Run the validator; fix errors (it exits 1 with `E_*` codes). Bump
   `version` in any committed manifest you touch.
