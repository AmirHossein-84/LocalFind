# P0-T006 — Dtype, dimension and embedding compatibility

**Status (2026-10-08): PASS — executed on the target Windows 11 x64 host.**
`benchmarks/embedding_invariants.py` exited 0; machine-readable results in
`docs/benchmarks/embedding-invariants-results.json`. No candidate dimension
is chosen here — that requires P0-T007 retrieval-quality comparison.

## Compatibility identifier (prototype contract)

Compact signature covering model id/revision, encoders, preprocessing,
modality pipeline, dimension, dtype, normalization, and segment policy
(PLAN §6 invariant 1 + R-10):

```
eg2@914f7f8|text+image|p0t006-v1|image|n/a|256|fp32|l2
 |       |        |         |       |     |   |    └ norm
 |       |        |         |       |     |   └ dtype (never fp16)
 |       |        |         |       |     └ dim (768/512/256/128 only)
 |       |        |         |       └ video segment policy (n/a for text/image)
 |       |        |         └ modality pipeline
 |       |        └ preprocessing version
 |       └ encoders (audio_config=None)
 └ model + pinned revision
```

`check_compatible()` refuses scoring on ANY field mismatch; `make_sig()`
refuses fp16 and non-MRL dims at construction. 17 negative checks, all
correctly rejected-or-accepted: model/encoders/modality/dim/dtype/norm/
revision/prep/segment mismatch, NaN, Inf, zero-norm, unnormalized, wrong
length, fp16 policy, dim-511 policy, plus identical-signature acceptance. Cross-dimension scoring
(768-vs-other) is refused, never silently mixed.

## Dimension sweep (fp32 CPU, project-local snapshot, offline)

Method (card rule): encode each fixture ONCE at native 768-d, keep leading
dims for 512/256/128, L2 re-normalize. Fixtures: 2 text queries, 1 document,
2 images, 2 video halves (7 vectors × 4 dims = 28, all finite + unit norm).

| Dim | red-query top1 | blue-query top1 | prefix cos (q/doc) |
|---|---|---|---|
| 768 | img_red | seg blue-half | 0.8286 |
| 512 | img_red | seg blue-half | 0.8325 |
| 256 | img_red | seg blue-half | 0.8417 |
| 128 | img_red (record-only) | seg blue-half (record-only) | 0.8986 |

Rankings identical across all dims on these synthetics (incl. 128, despite
the card's quality warning — smoke behavior, not a quality claim). Prefixed
query/document similarity stays high at every dim (0.83–0.90; single
prefixed pair per dim, no unprefixed baseline here — the prefix *delta* was
measured in P0-T004). Regression gate (P0-T005 red/blue
discrimination) enforced at 768/512/256: PASS.

Timings: fp32 load 3.2 s; 7 native encodes 16.3 s total; truncate+rank sweep
0.01 s (truncation is free vs re-encoding). RSS +1,252 MB.

## Precision modes (this CPU runtime: torch 2.14.1+cpu, i7-13620H)

| Mode | Result |
|---|---|
| fp32 | PASS — full sweep above; all params `torch.float32` |
| bf16 | ATTEMPTED, **rejected by the norm gate**: model loads (2.3 s) but query embedding norm is **0.997292** (tolerance 1e-4). CPU bf16 accumulation does not yield unit-norm vectors here → bf16 is NOT supported on this host for our l2 contract. No silent use. (GPU bf16 NOT RUN — no CUDA runtime.) |
| fp16 | NOT RUN by policy — rejected at `make_sig()` construction (card: silent NaNs; R-11). No inference attempted, none needed. |

Precision claims match runtime behavior: fp32 only on this machine.

## Limits

- Open design point for P0-T007: the sweep scores text queries against image
  AND video assets under one signature while the contract forbids `modality`
  mismatch. Cross-modal compat semantics (shared dim/dtype/norm/model with
  per-modality pipeline tags vs one joint signature) must be settled before
  the evaluator mixes modalities.
- Synthetic fixtures (2 images, 2 halves); no quality gate — P0-T007 decides the dimension.
- bf16-on-GPU and full-740M (audio encoder) untested here.
- Truncation uses leading-dim slicing per MRL; other reductions not evaluated.
