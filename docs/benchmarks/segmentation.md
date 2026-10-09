# P0-T008 — Video segmentation policy comparison

**Status (2026-10-08): MEASURED.** Six strategies + keyframe baseline ran on
the same 11 video queries (visual class; q032 excluded as in P0-T007),
judgments, model (`eg2@914f7f8`, fp32 CPU, offline), and 1-FPS sampling.
A baseline policy/version is selected below as a *documented initial
default*, not a production commitment. Gate threshold untouched; no model
swap; M0 still pending ARQ-001 review.

## Strategies (policy `seg-v1`, `benchmarks/segments.py`, 22 unit tests)

| Strategy | Rule | Units (20 clips) |
|---|---|---|
| halves | 2 equal halves (P0-T007 reference) | 40 |
| w2 | fixed 2 s windows, last clamped | 59 |
| w2ov50 | 2 s window / 1 s stride (50% overlap) | 93 |
| scene | ffmpeg scene cuts @0.4 (no cuts → whole clip) | 26 |
| scenew2 | **measured** hybrid: scene spans where ≥2 detected, else w2 | 53 |
| full | whole clip, one vector (anti-baseline; R-08 rejects for production) | 20 |
| keyframe | middle frame via image path + images (P0-T007 baseline) | 20 files |

Window-size note (STEPS example is 6/12 s): this corpus spans 1–12 s, so
6/12 s windows degenerate to 1–2 units per clip — no better than `full`
and untestable as a *granularity* choice. 2 s windows yield 1–6 units
across the duration range (1 s clips collapse to one span by construction
and unit test), which is where this corpus has variance. 6–12 s windows
remain untested and are recorded as such.

Scene reality check: the 8 two-tone clips cut exactly at their hard cut
(e.g. `[3000]` on 6000 ms); continuous-motion and static clips yield one
segment. Scene is testable here *for hard cuts only*.

## Quality @768-d (video queries, n=11)

| Strategy | Pool | seg R@1 / R@5 / MRR | file R@5 / MRR | top-1 time err (n) | dup-top5 |
|---|---|---|---|---|---|
| halves | mixed | 0.273 / 0.500 / 0.667 | 0.818 / 0.682 | 708 ms (6) | 5 |
| halves | video-only | 0.364 / 0.545 / 0.801 | 1.000 / 0.841 | 531 ms (8) | 10 |
| w2 | mixed | 0.258 / 0.682 / 0.667 | 0.818 / 0.678 | 833 ms (6) | 9 |
| w2 | video-only | 0.348 / 0.758 / 0.803 | 0.909 / 0.818 | 875 ms (8) | 11 |
| w2ov50 | mixed | 0.136 / 0.508 / 0.655 | 0.818 / 0.678 | 833 ms (6) | 9 |
| w2ov50 | video-only | 0.182 / 0.576 / 0.768 | 0.909 / 0.795 | 875 ms (8) | 10 |
| scene | mixed | 0.364 / 0.591 / 0.586 | 0.727 / 0.586 | 600 ms (5) | 0 |
| scene | video-only | 0.682 / 0.818 / 0.882 | 1.000 / 0.882 | 667 ms (9) | 1 |
| scenew2 | mixed | 0.212 / 0.591 / 0.589 | 0.818 / 0.621 | 800 ms (5) | 8 |
| scenew2 | video-only | 0.303 / 0.621 / 0.725 | 0.909 / 0.773 | 643 ms (7) | 11 |
| full | mixed | 0.364 / 0.727 / 0.530 | 0.727 / 0.530 | 1125 ms (4) | 0 |
| full | video-only | 0.636 / 1.000 / 0.776 | 1.000 / 0.776 | 1071 ms (7) | 0 |
| keyframe | mixed | 0.091 / 0.364 / 0.172 | — | — | — |

Reading (all on n=11 — wide confidence intervals; no overclaiming):
- **Overlap rejected:** w2ov50 underperforms w2 in both pools (video-only
  0.576 vs 0.758) at 1.6× units with heavy duplicate top-5 occupancy.
  One (window, stride) point tested; no benefit → no overlap (PLAN §8 rule).
- **Scene has the best per-unit quality** (video-only 0.818/0.882, time err
  667 ms, 26 units, dup≈0) but an **unbounded worst case** (statics → one
  vector) and hard-cut-only evidence.
- **Full ranks files perfectly (1.000) yet localizes nothing** (time err
  ~1071 ms): file retrieval without timestamps. The anti-baseline behaves
  exactly as R-08 predicts — segments exist for moments, not files.
- **Measured hybrid (scenew2) does NOT dominate** (video-only 0.621 < scene
  0.818 and < w2 0.758): fallback units add crowding without adding
  discrimination on this suite. The pre-registered "scene+w2" guess was
  wrong; the measurement stands.
- **File-grouped ≫ segment-level** everywhere (halves video-only 1.000 vs
  0.545): production ranking must group by file (PLAN §9).
- **Mixed < video-only** in every strategy: cross-modal crowding
  reconfirmed (ARQ-001 cause #1). Keyframe worst (0.364).

## Cost (measured; host variance dominates — read as ranges)

| Strategy | Units | npz bytes (B/unit) | encode mean, this host |
|---|---|---|---|
| halves | 40 | 133,898 (3,347) | 7.4 s (measured this task) |
| w2 | 59 | 197,918 (3,355) | 3.1 s |
| w2ov50 | 93 | 312,770 (3,363) | 7.1 s |
| scene | 26 | 87,310 (3,358) | 8.0 s (longer spans, more sampled frames) |
| scenew2 | 53 | 178,304 (3,365) | 3.3 s |
| full | 20 | 67,106 (3,355) | 8.8 s |
| keyframe | 20 | 66,742 (3,337) | 10.3 s |

Same-host, same-size inputs varied 3.1–7.4 s across runs (halves: 3.5 s in
P0-T005 vs 7.4 s here; w2 3.1 s vs scenew2-w2-spans 3.3 s). Per-segment
encode cost is therefore reported as an **observed 3–10 s range (CPU fp32),
not a precise per-strategy ranking** — run-to-run host variance exceeds
the between-strategy differences for equal-length inputs. Longer spans
cost more (more sampled frames), all else equal. Storage scales linearly:
~3.4 kB/unit float32 @768-d (decimal kB). Retrieval-only latency this run:
11.2 ms mean over 132 rankings (mixed + video-only pools).

## Baseline policy decision (documented initial default)

**Fixed 2 s windows (`w2`), version `seg-v1`.** Rationale, all measured:
best uniform-policy quality (video-only 0.758/0.803), bounded worst case
(short clips → one span; long clips → duration/2 units), deterministic,
simplest to schedule. Scene cuts stay a documented *refinement* (best
per-unit numbers where hard cuts exist) pending real-footage proof and a
boundedness rule; the measured hybrid is not adopted since it underperforms
both parents here. Overlapping windows rejected; full-clip rejected except
as a baseline. Revisit after P1 ranker work — narrow synthetic evidence,
stated as such.

Rejected variants: none for bounds violations — all spans validated
0 ≤ s < e ≤ duration (integer ms) by `segments.py`, out-of-range scene cuts
ignored, short clips collapse to one span (22 unit tests incl. 6 rejection
cases; embed aborts with `E_SEGMENTS` otherwise).

## Evaluation-integrity note (assigned investigation)

"14 failed visual queries vs Recall@5 = 0.677 over 31": reconciled, no
reporting error. Of 31 visual queries @768-d: **17 perfect (R@5 = 1.0) +
8 partial (summing to 4.0 — multi-relevant fractional credit) + 6 zero =
14 with R@5 < 1.0** ("errors"); mean = 21.0/31 = **0.6774**. Strict <1.0
count vs graded mean — both reported, consistent.

## ARQ-001 update (new evidence)

- Segmentation is a secondary lever: video-only R@5 spans 0.545–0.818
  across policies (±0.15 around halves); it cannot close the 0.75 gate alone
  on the mixed pool (best mixed: w2/full 0.727/0.682).
- Ranking/fusion dominates: file-grouping lifts halves video-only
  0.545 → 1.000; mixed→video-only gaps persist in every policy.
- Model representation is not the blocker here: two-tone halves ARE
  discriminable (scene video-only 0.818/0.882; full file-level 1.000).
- Labels remain a confound (single-relevant repeated colors).
- Recommendation A (fusion/judgments first) strengthened again; no model
  swap, no threshold change. Full ARQ-001 update in STEPS.md.

## Limits

- 11 video queries, synthetic hard cuts/static only; VFR/timebase edges,
  >12 s clips, and 6/12 s windows untested.
- Overlap tested at one (window, stride) point; scene threshold at one value.
- Timings are single-run smoke readings with large host variance, not p95.
- Threshold, model, and judgments unchanged per integrity rules.
