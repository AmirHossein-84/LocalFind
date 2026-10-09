#!/usr/bin/env python3
"""P0-T008: compare video segmentation strategies on fixed judgments.

Strategies (all @768-d, fp32, pinned snapshot, offline):
  halves  (P0-T007 reference: 2/clip) | w2 (fixed 2 s) | w2ov50 (2 s/1 s stride)
  scene   (ffmpeg scene cuts @0.4)     | full (whole-clip single vector)
  keyframe (middle-frame image-path vector + images)

Per strategy, on the same 11 video queries (+ visual overall where noted):
  - segment-level Recall@1/@5, MRR@10 (mixed image+video pool AND video-only pool)
  - file-grouped Recall@K/MRR (best segment per file: no duplicate top-K occupancy)
  - timestamp error: |pred center - expected center| over top-1 hits (ms)
  - cost: units, npz bytes, mean encode_s (measured at embed time), retrieval latency

Report v2 (P0 integrity rework, judgments unchanged):
  - every segment metric is reported under BOTH temporal policies
    ("tolerant" = original judgments; "strict" = zero-tolerance
    sensitivity). Legacy seg_* keys keep the tolerant (original) values.
  - per-query relevant-unit counts under each policy, candidate pool
    sizes, file-grouped metrics, timestamp hit counts AND miss rates,
    and an explicit limitations block: cross-strategy segment Recall is
    NOT directly comparable (denominators and pools differ); file-grouped
    recall is the file-retrieval comparison metric.
  - the w2 default stays a provisional, reversible baseline (seg-v1),
    not a proven optimum.
  - pre-write preservation: an existing pre-v2 report is copied to a
    .historical.json backup (once) instead of silently lost.

No judgments modified; gate threshold untouched (reported only).
Usage: benchmarks/.venv/Scripts/python benchmarks/segment_compare.py
Exit 0 = compared; 1 = failure. No model load, no network.
"""

import json
import statistics
import sys
import time
from pathlib import Path

from compat import check_space_compatible, make_sig, truncate_renorm
from evaluate import (load_eval_inputs, load_historical_backup,
                      preserve_historical_report, relevant_unit_keys_policy)
from retrieval import (group_rank_by_file, mrr_at, rank_candidates,
                       recall_at_k, relevant_asset_of, top1_time_error)

SEG_REPORT_VERSION = 2

REPO_ROOT = Path(__file__).resolve().parent.parent
VEC_DIR = REPO_ROOT / "benchmarks" / "corpus" / "vectors"
DIM = 768
STRATEGY_FILES = {"halves": "video.npz", "w2": "video_w2.npz",
                  "w2ov50": "video_w2ov50.npz", "scene": "video_scene.npz",
                  "scenew2": "video_scenew2.npz", "full": "video_full.npz"}


def fail(code, message):
    print(f"{code}: {message}", file=sys.stderr)
    sys.exit(1)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="P0-T008 strategy comparison.")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "docs" / "benchmarks"
                        / "segmentation-results.json")
    args = parser.parse_args(argv)
    try:
        corpus, queries_doc, assets, queries = load_eval_inputs(
            REPO_ROOT / "benchmarks" / "corpus" / "corpus.json",
            REPO_ROOT / "benchmarks" / "corpus" / "queries.json")
    except ValueError as exc:
        fail("E_JUDGMENTS", str(exc))
    import numpy as np
    manifest = json.loads((VEC_DIR / "vectors-manifest.json").read_text())
    store = {}
    for name in ("image", "query", "keyframe"):
        with np.load(VEC_DIR / f"{name}.npz") as z:
            store.update({k: np.asarray(z[k], dtype=np.float32) for k in z.files})
    strat_units = {}
    for strategy, fname in STRATEGY_FILES.items():
        with np.load(VEC_DIR / fname) as z:
            strat_units[strategy] = {k: np.asarray(z[k], dtype=np.float32)
                                     for k in z.files}
    image_keys = sorted(k for k in store if k.startswith("image:"))
    keyframe_keys = sorted(k for k in store if k.startswith("keyframe:"))
    q_space = make_sig("text-query", DIM, prep="p0t008-v1")
    for key in list(store) + [k for u in strat_units.values() for k in u]:
        kind = key.split(":")[0]
        modality = {"image": "image", "video": "video", "query": "text-query",
                    "keyframe": "image"}[kind]
        try:
            check_space_compatible(q_space, make_sig(modality, DIM, prep="p0t008-v1"))
        except ValueError as exc:
            fail("E_SPACE", f"{key}: {exc}")

    # Same eligible set as P0-T007 (visual class, video modality; n=11).
    # q032 (exact_filename) is excluded here as there.
    video_queries = [q for q in queries
                     if q["query_class"] == "visual" and q["modality"] == "video"]
    results = {"report_version": SEG_REPORT_VERSION,
                 "strategies": {}, "config": {
        "dim": DIM, "model_revision": manifest["revision"],
        "policies": "seg-v1 (halves|w2|w2ov50|scene|full) + keyframe baseline",
        "temporal_policies": {
            "tolerant": "original judgments (tolerance-expanded overlap); "
                        "legacy seg_* keys carry these values",
            "strict": "sensitivity-only zero-tolerance overlap"},
        "baseline_status": "w2 (fixed 2 s, seg-v1) is a provisional, "
                           "reversible default, NOT a proven optimum"},
                 "limitations": [
        "Cross-strategy SEGMENT-level Recall is not directly comparable: "
        "candidate pools (26-93 units) and per-query relevant-unit counts "
        "differ by strategy and by temporal policy. Use FILE-grouped "
        "recall for file-retrieval comparisons.",
        "Tolerance bridges adjacent-segment seams on seam-aligned policies "
        "(e.g. halves), inflating tolerant relevant counts vs strict; both "
        "counts are reported per query.",
        "n=11 video queries (wide confidence intervals), synthetic "
        "hard-cut/static clips only; timings are single-run smoke "
        "readings, not p95."]}
    lat = []
    for strategy, units in strat_units.items():
        vkeys = sorted(units)
        pool_mixed = image_keys + vkeys
        tvec = {k: truncate_renorm([float(v) for v in units[k]], DIM) for k in vkeys}
        tvec.update({k: truncate_renorm([float(v) for v in store[k]], DIM)
                     for k in image_keys})
        tvec_q = {q["query_id"]: truncate_renorm(
            [float(v) for v in store[f"query:{q['query_id']}"]], DIM) for q in queries}
        out = {"n_units": len(vkeys)}
        for pool_name, pool in (("mixed", pool_mixed), ("video_only", vkeys)):
            # Score once per policy; legacy seg_*/file_* keys = tolerant.
            per_policy = {}
            for policy in ("tolerant", "strict"):
                rec1, rec5, mrr, errs, terr = [], [], [], [], []
                dup_top5 = 0
                rel_counts = {}
                for q in video_queries:
                    rel = relevant_unit_keys_policy(q, assets, vkeys,
                                                    policy=policy)
                    rel_keys = [k for k, _g in rel]
                    rel_counts[q["query_id"]] = len(rel_keys)
                    t0 = time.perf_counter()
                    ranked = rank_candidates(tvec_q[q["query_id"]],
                                             [(k, tvec[k]) for k in pool])
                    lat.append(time.perf_counter() - t0)
                    keys = [k for k, _s in ranked]
                    rec1.append(recall_at_k(rel_keys, keys, 1))
                    rec5.append(recall_at_k(rel_keys, keys, 5))
                    mrr.append(mrr_at(rel_keys, keys, 10))
                    top5_aids = [relevant_asset_of(k) for k, _s in ranked[:5]]
                    if len(set(top5_aids)) < len(top5_aids):
                        dup_top5 += 1
                    # File-grouped: best segment per file.
                    file_rank = group_rank_by_file(ranked)
                    rel_files = sorted({relevant_asset_of(k) for k in rel_keys})
                    fr1 = recall_at_k(rel_files, file_rank, 1)
                    fr5 = recall_at_k(rel_files, file_rank, 5)
                    fmrr = mrr_at(rel_files, file_rank, 10)
                    errs.append({"file_recall@1": fr1, "file_recall@5": fr5,
                                 "file_mrr@10": fmrr})
                    if rec5[-1] < 1.0:
                        errs[-1].update({"query_id": q["query_id"],
                                         "text": q["text"]})
                    else:
                        errs[-1].update({"query_id": q["query_id"]})
                    # Timestamp error on top-1 segment hits (video queries only).
                    top_unit = keys[0]
                    terr.append(top1_time_error(q, top_unit, assets,
                                                policy=policy))
                n = len(video_queries)
                f1 = [e["file_recall@1"] for e in errs]
                f5 = [e["file_recall@5"] for e in errs]
                fm = [e["file_mrr@10"] for e in errs]
                te = [t for t in terr if t is not None]
                per_policy[policy] = {
                    "n": n, "pool_size": len(pool),
                    "seg_recall@1": sum(rec1) / n,
                    "seg_recall@5": sum(rec5) / n,
                    "seg_mrr@10": sum(mrr) / n,
                    "file_recall@1": sum(f1) / n,
                    "file_recall@5": sum(f5) / n,
                    "file_mrr@10": sum(fm) / n,
                    "top1_time_err_ms_mean": (sum(te) / len(te)) if te else None,
                    "top1_time_err_n": len(te),
                    "top1_time_err_misses": n - len(te),
                    "top1_time_err_miss_rate": (n - len(te)) / n,
                    "dup_top5_queries": dup_top5,
                    "relevant_units_per_query": rel_counts,
                    "errors": [e for e in errs if "text" in e]}
            tol, stri = per_policy["tolerant"], per_policy["strict"]
            out[pool_name] = {
                "n": tol["n"], "pool_size": tol["pool_size"],
                "seg_recall@1": tol["seg_recall@1"],
                "seg_recall@5": tol["seg_recall@5"],
                "seg_mrr@10": tol["seg_mrr@10"],
                "file_recall@1": tol["file_recall@1"],
                "file_recall@5": tol["file_recall@5"],
                "file_mrr@10": tol["file_mrr@10"],
                "top1_time_err_ms_mean": tol["top1_time_err_ms_mean"],
                "top1_time_err_n": tol["top1_time_err_n"],
                "top1_time_err_misses": tol["top1_time_err_misses"],
                "top1_time_err_miss_rate": tol["top1_time_err_miss_rate"],
                "dup_top5_queries": tol["dup_top5_queries"],
                "errors": tol["errors"],
                "temporal_policy": "tolerant (original judgments)",
                "sensitivity_strict": stri,
                "relevant_units_per_query_tolerant":
                    tol["relevant_units_per_query"],
                "relevant_units_per_query_strict":
                    stri["relevant_units_per_query"],
                "note": "seg_* and file_* are tolerant (original) values; "
                        "strict twin under sensitivity_strict. Cross-strategy "
                        "seg_* comparisons are NOT file-retrieval evidence "
                        "(pools/denominators differ); compare file_recall@5."}
        # Cost: measured encode means from the embed manifest where present.
        enc = [manifest["units"].get(k, {}).get("encode_s") for k in vkeys]
        enc = [e for e in enc if isinstance(e, (int, float))]
        npz_bytes = (VEC_DIR / STRATEGY_FILES[strategy]).stat().st_size
        img_bytes = (VEC_DIR / "image.npz").stat().st_size
        out["cost"] = {"units": len(vkeys), "npz_bytes": npz_bytes,
                       "bytes_per_unit": npz_bytes / len(vkeys),
                       "candidate_pool_bytes_mixed": npz_bytes + img_bytes,
                       "query_bytes": (VEC_DIR / "query.npz").stat().st_size,
                       "encode_mean_s": (statistics.mean(enc) if enc else None),
                       "encode_measured_n": len(enc)}
        results["strategies"][strategy] = out

    # Keyframe strategy (images + whole-file middle frames), same queries.
    kb_pool = image_keys + keyframe_keys
    tvec_kb = {k: truncate_renorm([float(v) for v in store[k]], DIM) for k in kb_pool}
    tvec_kb.update({f"query:{q['query_id']}": truncate_renorm(
        [float(v) for v in store[f"query:{q['query_id']}"]], DIM) for q in queries})
    rec1, rec5, mrr, f5, fm = [], [], [], [], []
    for q in video_queries:
        rel = []
        for exp in q.get("expected", []):
            if exp.get("grade", 2) >= 1:
                rel.append(f"keyframe:{exp['asset_id']}")
        ranked = rank_candidates(tvec_kb[f"query:{q['query_id']}"],
                                 [(k, tvec_kb[k]) for k in kb_pool])
        keys = [k for k, _s in ranked]
        rec1.append(recall_at_k(rel, keys, 1))
        rec5.append(recall_at_k(rel, keys, 5))
        mrr.append(mrr_at(rel, keys, 10))
        # Keyframe units are already one-per-file, so segment-level and
        # file-grouped coincide; report both for schema uniformity.
        file_rank = group_rank_by_file(ranked)
        rel_files = sorted({relevant_asset_of(k) for k in rel})
        f5.append(recall_at_k(rel_files, file_rank, 5))
        fm.append(mrr_at(rel_files, file_rank, 10))
    n = len(video_queries)
    kf_npz = VEC_DIR / "keyframe.npz"
    kf_enc = [manifest["units"].get(k, {}).get("encode_s") for k in keyframe_keys]
    kf_enc = [e for e in kf_enc if isinstance(e, (int, float))]
    results["strategies"]["keyframe"] = {
        "n_units": len(keyframe_keys), "mixed": {
            "n": n, "pool_size": len(kb_pool),
            "seg_recall@1": sum(rec1) / n, "seg_recall@5": sum(rec5) / n,
            "seg_mrr@10": sum(mrr) / n,
            "file_recall@5": sum(f5) / n, "file_mrr@10": sum(fm) / n,
            "temporal_policy": "n/a (whole-file units)",
            "note": "file-level relevance (one vector/video); no timestamps"},
        "cost": {"units": len(keyframe_keys), "npz_bytes": kf_npz.stat().st_size,
                 "bytes_per_unit": kf_npz.stat().st_size / len(keyframe_keys),
                 "candidate_pool_bytes_mixed":
                     kf_npz.stat().st_size + (VEC_DIR / "image.npz").stat().st_size,
                 "query_bytes": (VEC_DIR / "query.npz").stat().st_size,
                 "encode_mean_s": (statistics.mean(kf_enc) if kf_enc else None),
                 "encode_measured_n": len(kf_enc)}}
    results["latency"] = {"retrieval_only_mean_s": sum(lat) / len(lat),
                          "n_rankings": len(lat),
                          "note": "counts mixed-pool + video-only rankings per "
                                  "query, both temporal policies"}
    historical = preserve_historical_report(args.out)
    frozen = load_historical_backup(args.out) or historical
    if frozen:
        results["historical"] = {
            "note": "pre-rework measurement kept; tolerant seg_* values "
                    "above must reproduce it (judgments unchanged)",
            "strategies": {k: v.get("mixed", v) for k, v in
                           frozen.get("strategies", {}).items()
                           if isinstance(v, dict)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for strategy, out in results["strategies"].items():
        m = out.get("mixed", out)
        c = out.get("cost", {})
        s = (m.get("sensitivity_strict") or {})
        strict_txt = (f"{s.get('seg_recall@5', 0):.3f}"
                      if "seg_recall@5" in s else "n/a")
        miss = m.get("top1_time_err_miss_rate")
        miss_txt = f"{miss:.3f}" if isinstance(miss, float) else "n/a"
        print(f"{strategy}: units={out.get('n_units', c.get('units'))} "
              f"pool={m.get('pool_size', '?')} "
              f"R@1={m.get('seg_recall@1', m.get('recall@1', 0)):.3f} "
              f"R@5={m.get('seg_recall@5', m.get('recall@5', 0)):.3f} "
              f"MRR={m.get('seg_mrr@10', m.get('mrr@10', 0)):.3f} "
              f"fileR@5={m.get('file_recall@5', 0):.3f} "
              f"strictSegR@5={strict_txt} "
              f"miss_rate={miss_txt} "
              f"enc={c.get('encode_mean_s')}")
    print(f"results: {args.out}")
    print("COMPARE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
