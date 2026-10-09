# P0-T007 — Retrieval-quality feasibility benchmark

**Status (2026-10-08): MEASURED — gate MISSED, PIVOT-SIGNAL recorded.**
Visual Recall@5 = **0.677 @768-d** vs the fixed 0.75 gate (PLAN §2). Thresholds
were not changed after observing results. Failure analysis + review request
below (ARQ-001 in STEPS.md). Machine-readable results (incl. all per-query
errors): `docs/benchmarks/retrieval-v1-results.json`.

## Method (reproducible)

- Corpus: `localfind-p0-synth-v1` (120 assets) + judgments v1 (39 queries).
  Candidates: 100 images + 20 clips × halves = **140 units**.
- Vectors: cached native-768-d fp32 (`benchmarks/corpus/vectors/*.npz`,
  git-ignored) via `embed_corpus.py` (project-local snapshot, offline,
  sequential); truncation to 512/256/128 post-hoc (MRL slice + L2 re-norm).
- Queries encoded with `SearchQuery` prefix; **mixed ranking over all 140
  units** (production-like) + modality-pure secondaries.
- Space-compat gate (P0-T006 fix): text queries score vs image/video vectors
  when model/revision/dim/dtype/norm match; modality pipelines differ by
  design. Every cached unit passed the gate before scoring.
- Timestamp rule: same asset AND positive overlap after expanding the
  expected interval by `tolerance_ms` both sides (tolerance bridges segment
  seams by design; see unit tests).
- Relevant = grade ≥ 1. Negatives (4) excluded from Recall/MRR denominators;
  reported as top-1 cosine calibration instead.
- Baselines: filename-token overlap (deterministic), whole-video single
  middle-frame vectors + images ("keyframe": the R-08-rejected shortcut,
  measured as baseline only), and seeded-random (seed 42).

## Results @768-d (primary)

| Suite | n | Recall@1 | Recall@5 | MRR@10 |
|---|---|---|---|---|
| visual (gate suite) | 31 | 0.129 | **0.677** | 0.441 |
| └ image queries | 20 | 0.100 | 0.775 | 0.363 |
| └ video queries | 11 | 0.364 | 0.545 | 0.801 |
| exact_filename (semantic only) | 2 | 0.000 | 0.000 | 0.000 |
| semantic_image | 2 | 0.000 | 0.167 | 0.238 |
| filename baseline (visual) | 31 | 0.484 | 0.548 | 0.539 |
| keyframe baseline (visual) | 31 | 0.065 | 0.629 | 0.279 |
| random baseline (visual) | 31 | 0.032 | 0.081 | 0.068 |
| negatives top-1 cos (mean/max) | 4 | — | — | 0.668 / 0.690 |

Dims 512/256: visual R@5 0.645/0.645 (flat vs 768). 128: 0.565 overall,
semantic collapses to 0.0, negative top-1 cos rises to 0.80 (worst
calibration). No dimension chosen (unchanged: P0-T007 defers the decision,
but the data now exists).

Latency (retrieval only, separate from inference): **8.0 ms mean** per
140-unit ranking. Index: 596,578 bytes → **4.3 KB/unit** (float32 stored).

## Failure analysis (14/31 visual errors @768)

1. **Cross-modal crowding (dominant, 9 queries: q021–q029).** Two-tone halves
   that ranked correctly in the 10-candidate P0-T005 smoke test are crowded
   out at 140 candidates — red *images* outscore red *video halves* for
   "red screen". Image and video embeddings are not directly magnitude-
   comparable: naive cosine fusion across modalities is unreliable. Direct
   input to P1 ranker design (PLAN §9 RRF, per-modality normalization).
2. **Near-duplicate confusion (5 queries: q002, q013–q015, q020).**
   dark-red vs red, smptebars variants, testsrc stills — fine-grained
   synthetic distinctions; expected-hard cases behaving as designed.
3. **Identifier queries unanswerable semantically (q031, q032: 0.0).**
   Expected: IDs need the P1 lexical path (R-17), not embeddings.
4. **Conceptual queries weak (q034, q035).** "warm/cool colors" barely
   retrieves grade-1 targets — compositional color semantics unproven.
5. **Judgment-design weakness:** repeated colors across clips (black in
   3 clips) carry single-relevant labels; future judgment revisions should
   mark all true matches or narrow the query.

## Baselines and confounds (no cherry-picking)

- Keyframe baseline (R@5 0.629) underperforms segment-level retrieval
  (0.677) on the same videos — first evidence that one-video-one-vector
  loses localizable moments, supporting R-08 (timestamped segments).
  Filename baseline beats semantic R@1 (0.484 vs 0.129) **but is confounded**:
  generator filenames literally contain the labels (`img_014_solid-red…`,
  `vid_001_tone_red_blue`), so it measures label leakage, not real filename
  search. Recorded as-is; P1 lexical evaluation needs realistic filenames.
- Random baseline at chance (R@5 0.08) confirms the harness is sane; the
  inverted-ranking unit test proves the gate fires on bad input.
- Negatives retrieve at cos ~0.67–0.82 with no score gap vs positives —
  no usable abstention threshold exists at this stage.

## Verdict

**PIVOT-SIGNAL**: 0.677 < 0.75 on the curated suite. Not a model rejection —
the evidence points to (a) cross-modal fusion methodology, (b) synthetic
near-duplicate hardness, and (c) judgment narrowness, all investigable
without swapping the model. No architecture changed by this task. M0
go/no-go needs the ARQ-001 review; P0-T008 (segmentation) and P0-T009
(storage) proceed independently of the verdict.

## Limits

- Synthetic corpus (patterns/solids are likely off-distribution); real-photo
  recall unmeasured.
- Halves-only segments; thirds/overlaps unevaluated (P0-T008).
- CPU fp32 only; bf16/GPU paths per P0-T006 findings.
- Single run; timings are smoke readings, not p95 claims.
