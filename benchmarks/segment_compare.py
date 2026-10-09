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
from evaluate import load_eval_inputs, relevant_unit_keys
from retrieval import mrr_at, rank_candidates, recall_at_k

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
    results = {"strategies": {}, "config": {
        "dim": DIM, "model_revision": manifest["revision"],
        "policies": "seg-v1 (halves|w2|w2ov50|scene|full) + keyframe baseline"}}
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
            rec1, rec5, mrr, errs, terr = [], [], [], [], []
            dup_top5 = 0
            for q in video_queries:
                rel = relevant_unit_keys(q, assets, vkeys)
                rel_keys = [k for k, _g in rel]
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
                best = {}
                for k, s in ranked:
                    aid = relevant_asset_of(k)
                    if aid not in best:
                        best[aid] = s
                file_rank = sorted(best, key=lambda a: (-best[a], a))
                rel_files = sorted({relevant_asset_of(k) for k in rel_keys})
                fr5 = recall_at_k(rel_files, file_rank, 5)
                fmrr = mrr_at(rel_files, file_rank, 10)
                errs.append({"file_recall@5": fr5, "file_mrr@10": fmrr})
                if rec5[-1] < 1.0:
                    errs[-1].update({"query_id": q["query_id"], "text": q["text"]})
                else:
                    errs[-1].update({"query_id": q["query_id"]})
                # Timestamp error on top-1 segment hits (video queries only).
                top_unit = keys[0]
                terr.append(top1_time_error(q, top_unit, assets))
            n = len(video_queries)
            f5 = [e["file_recall@5"] for e in errs]
            fm = [e["file_mrr@10"] for e in errs]
            te = [t for t in terr if t is not None]
            out[pool_name] = {
                "n": n, "seg_recall@1": sum(rec1) / n, "seg_recall@5": sum(rec5) / n,
                "seg_mrr@10": sum(mrr) / n,
                "file_recall@5": sum(f5) / n, "file_mrr@10": sum(fm) / n,
                "top1_time_err_ms_mean": (sum(te) / len(te)) if te else None,
                "top1_time_err_n": len(te),
                "dup_top5_queries": dup_top5,
                "errors": [e for e in errs if "text" in e]}
        # Cost: measured encode means from the embed manifest where present.
        enc = [manifest["units"].get(k, {}).get("encode_s") for k in vkeys]
        enc = [e for e in enc if isinstance(e, (int, float))]
        npz_bytes = (VEC_DIR / STRATEGY_FILES[strategy]).stat().st_size
        out["cost"] = {"units": len(vkeys), "npz_bytes": npz_bytes,
                       "bytes_per_unit": npz_bytes / len(vkeys),
                       "encode_mean_s": (statistics.mean(enc) if enc else None),
                       "encode_measured_n": len(enc)}
        results["strategies"][strategy] = out

    # Keyframe strategy (images + whole-file middle frames), same queries.
    kb_pool = image_keys + keyframe_keys
    tvec_kb = {k: truncate_renorm([float(v) for v in store[k]], DIM) for k in kb_pool}
    tvec_kb.update({f"query:{q['query_id']}": truncate_renorm(
        [float(v) for v in store[f"query:{q['query_id']}"]], DIM) for q in queries})
    rec1, rec5, mrr = [], [], []
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
    n = len(video_queries)
    kf_npz = VEC_DIR / "keyframe.npz"
    kf_enc = [manifest["units"].get(k, {}).get("encode_s") for k in keyframe_keys]
    kf_enc = [e for e in kf_enc if isinstance(e, (int, float))]
    results["strategies"]["keyframe"] = {
        "n_units": len(keyframe_keys), "mixed": {
            "n": n, "seg_recall@1": sum(rec1) / n, "seg_recall@5": sum(rec5) / n,
            "seg_mrr@10": sum(mrr) / n,
            "note": "file-level relevance (one vector/video); no timestamps"},
        "cost": {"units": len(keyframe_keys), "npz_bytes": kf_npz.stat().st_size,
                 "bytes_per_unit": kf_npz.stat().st_size / len(keyframe_keys),
                 "encode_mean_s": (statistics.mean(kf_enc) if kf_enc else None),
                 "encode_measured_n": len(kf_enc)}}
    results["latency"] = {"retrieval_only_mean_s": sum(lat) / len(lat),
                          "n_rankings": len(lat),
                          "note": "counts mixed-pool + video-only rankings per query"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for strategy, out in results["strategies"].items():
        m = out.get("mixed", out)
        c = out.get("cost", {})
        print(f"{strategy}: units={out.get('n_units', c.get('units'))} "
              f"R@1={m.get('seg_recall@1', m.get('recall@1', 0)):.3f} "
              f"R@5={m.get('seg_recall@5', m.get('recall@5', 0)):.3f} "
              f"MRR={m.get('seg_mrr@10', m.get('mrr@10', 0)):.3f} "
              f"enc={c.get('encode_mean_s')}")
    print(f"results: {args.out}")
    print("COMPARE PASS")
    return 0


def relevant_asset_of(unit_key):
    """File asset behind a unit key (image/video/keyframe forms)."""
    parts = unit_key.split(":")
    return parts[1]


def top1_time_error(query, top_unit_key, assets):
    """|predicted segment center - expected interval center| (ms).

    Returns None when the top-1 unit is not a hit (accuracy is measured
    only over hits; misses are counted in Recall, not here).
    """
    from retrieval import segment_matches
    parts = top_unit_key.split(":")
    if parts[0] not in ("video",):
        return None
    aid, span = parts[1], parts[-1]
    s, e = (int(x) for x in span.split("-"))
    for exp in query.get("expected", []):
        if exp.get("grade", 2) < 1 or exp["asset_id"] != aid:
            continue
        if segment_matches({**exp, "asset_id": aid},
                           {"asset_id": aid, "start_ms": s, "end_ms": e}):
            pred_c = (s + e) / 2
            exp_c = (exp["start_ms"] + exp["end_ms"]) / 2
            return abs(pred_c - exp_c)
    return None


if __name__ == "__main__":
    sys.exit(main())
