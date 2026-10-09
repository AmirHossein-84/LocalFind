#!/usr/bin/env python3
"""P0-T007 step 2: retrieval-quality benchmark on cached native-768-d vectors.

Reads benchmarks/corpus/vectors/*.npz (see embed_corpus.py) + the P0-T003
manifests. Truncates to 768/512/256/128 post-hoc (MRL slice + L2 re-norm),
ranks the full candidate set per query, and reports Recall@1/@5, MRR@10 by
query class and modality, timestamp-overlap video matching, negative-query
calibration, filename-only + deterministic-random baselines, per-query
errors, retrieval-only latency, and index size. No model load, no network.

Gate (PLAN section 2, FIXED): Recall@5 >= 0.75 on the curated visual suite.
The verdict is recorded, never auto-acted upon; thresholds are not editable
here (see check_gate in retrieval.py -- this script only reports).

Report v2 (P0 integrity rework, judgments + gate unchanged):
  - strict-vs-tolerant temporal sensitivity on the visual suite
    (segment-level AND file-grouped, per-query relevant counts);
  - candidate pool sizes, file-grouped visual metrics, top-1 timestamp
    error with hit count + miss rate, and explicit cross-comparison limits;
  - index accounting split into candidate-vector bytes vs query-vector
    bytes (v1 lumped both into one figure);
  - pre-write preservation: an existing report without report_version 2 is
    copied to a .historical.json backup (once) instead of silently lost.

Usage: benchmarks/.venv/Scripts/python benchmarks/evaluate.py [--out PATH]
Exit 0 = benchmark completed (even on gate miss); 1 = crash/malformed input.
"""

import json
import random
import re
import sys
import time
from pathlib import Path

from compat import (check_space_compatible, check_vector_values, make_sig,
                      truncate_renorm)
from local_model import REVISION
from retrieval import (EvaluationGateError, check_gate, group_rank_by_file,
                       mrr_at, parse_unit_key, rank_candidates, recall_at_k,
                       segment_matches_with_policy, top1_time_error)

REPORT_VERSION = 2

REPO_ROOT = Path(__file__).resolve().parent.parent
VEC_DIR = REPO_ROOT / "benchmarks" / "corpus" / "vectors"
DIMS = (768, 512, 256, 128)
GATE_RECALL5 = 0.75
VISUAL_SUITE_CLASSES = ("visual",)


def fail(code, message):
    print(f"{code}: {message}", file=sys.stderr)
    sys.exit(1)


def load_eval_inputs(corpus_path, queries_path):
    """Parse + validate manifests. Raises ValueError E_* on malformed input."""
    try:
        corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        queries_doc = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"E_IO: cannot parse manifests: {exc}")
    if queries_doc.get("corpus_id") != corpus.get("corpus_id"):
        raise ValueError("E_CORPUS_MISMATCH: queries/corpus corpus_id differ")
    assets = {}
    for a in corpus.get("assets", []):
        aid = a.get("asset_id")
        if aid in assets:
            raise ValueError(f"E_DUP_ASSET_ID: {aid}")
        assets[aid] = a
    queries = []
    seen = set()
    for q in queries_doc.get("queries", []):
        qid = q.get("query_id")
        if qid in seen:
            raise ValueError(f"E_DUP_QUERY_ID: {qid}")
        seen.add(qid)
        if q.get("query_class") == "negative" and q.get("expected"):
            raise ValueError(f"E_NEG_WITH_EXPECTED: {qid}")
        for exp in q.get("expected", []):
            if exp.get("asset_id") not in assets:
                raise ValueError(f"E_DANGLING_REF: {qid} -> {exp.get('asset_id')}")
            if assets[exp["asset_id"]]["modality"] == "video":
                s, e = exp.get("start_ms"), exp.get("end_ms")
                dur = assets[exp["asset_id"]].get("duration_ms")
                if not isinstance(s, int) or not isinstance(e, int) or not (0 <= s < e):
                    raise ValueError(f"E_BAD_RANGE: {qid}")
                if isinstance(dur, int) and e > dur:
                    raise ValueError(f"E_RANGE_EXCEEDS_DURATION: {qid}")
        queries.append(q)
    return corpus, queries_doc, assets, queries


def relevant_unit_keys(query, assets, video_keys):
    """Unit keys relevant to a query (grade>=1; video via overlap rule).

    Image expectations match same-asset units; video expectations match only
    same-asset halves overlapping [start-tol, end+tol]. Cross-asset leakage
    is a recalled P0-T007 bug -- covered by test_evaluate.py.
    """
    return relevant_unit_keys_policy(query, assets, video_keys, policy="tolerant")


def relevant_unit_keys_policy(query, assets, video_keys, policy="tolerant"):
    """Policy-aware relevant units (grade>=1).

    "tolerant" reproduces the original judgments exactly (expected interval
    expanded by tolerance_ms); "strict" forces zero tolerance so adjacent
    segments sharing only an edge do NOT count. Unknown policies raise
    E_POLICY via segment_matches_with_policy. Image handling is identical
    under both policies (no timestamps involved).
    """
    rel = []
    for exp in query.get("expected", []):
        if exp.get("grade", 2) < 1:
            continue  # grade-0 judgments are explicitly non-relevant
        aid = exp["asset_id"]
        if assets[aid]["modality"] == "image":
            rel.append((f"image:{aid}", exp.get("grade", 2)))
        else:
            for key in video_keys:
                # Key forms: "video:{aid}:{s}-{e}" (halves) or
                # "video:{aid}:{strategy}:{s}-{e}" (P0-T008 strategies).
                parsed = parse_unit_key(key)
                if parsed["kind"] != "video":
                    raise ValueError(f"E_UNITKEY: expected video key, got {key!r}")
                unit_aid, s, e = (parsed["asset_id"], parsed["start_ms"],
                                  parsed["end_ms"])
                if segment_matches_with_policy(
                        {**exp, "asset_id": aid},
                        {"asset_id": unit_aid, "start_ms": s, "end_ms": e},
                        policy):
                    rel.append((key, exp.get("grade", 2)))
    return rel


def relevant_file_assets(query, assets):
    """Sorted relevant file asset ids for a query (grade>=1, file-grouped).

    File-grouped denominators are strategy-independent (one entry per file),
    unlike segment-level denominators which grow with finer segmentation.
    """
    files = set()
    for exp in query.get("expected", []):
        if exp.get("grade", 2) >= 1:
            files.add(exp["asset_id"])
    return sorted(files)


def filename_score_for_key(query_text, unit_key, assets):
    """Filename-token overlap for any image/video/keyframe unit key.

    Supports 3-part ("video:{aid}:{s}-{e}") and 4-part
    ("video:{aid}:{strategy}:{s}-{e}") video keys via parse_unit_key;
    malformed keys raise E_UNITKEY (typed, never a bare unpack error).
    """
    toks = re.findall(r"[a-z0-9]+", query_text.lower())
    if not toks:
        return 0.0
    parsed = parse_unit_key(unit_key)
    if parsed["kind"] == "query":
        raise ValueError(f"E_UNITKEY: query key {unit_key!r} is not a candidate")
    stem = assets[parsed["asset_id"]]["rel_path"].lower()
    stem = stem.rsplit("/", 1)[-1]
    return sum(1 for t in toks if t in stem) / len(toks)


def index_size_report(vec_dir):
    """Split stored bytes into candidate-vector vs query-vector bytes.

    v1 reported a single lumped figure (candidates + queries); v2 reports
    each side separately so bytes/unit reflects the served candidate index.
    Returns a dict with candidate/query/total bytes + unit counts.
    """
    cand_files = [vec_dir / "image.npz", vec_dir / "video.npz"]
    cand_bytes = sum(p.stat().st_size for p in cand_files)
    query_bytes = (vec_dir / "query.npz").stat().st_size
    return {"candidate_bytes": cand_bytes, "query_bytes": query_bytes,
            "total_bytes": cand_bytes + query_bytes}


def preserve_historical_report(out_path):
    """Copy a pre-v2 report to a .historical.json backup (once, not silent).

    Returns the parsed historical doc (or None). Never raises: preservation
    is best-effort and must not break the benchmark run itself.
    """
    return _read_report_for_history(out_path, create_backup=True)


def load_historical_backup(out_path):
    """Parse the frozen .historical.json backup beside out_path (or None).

    The backup is the byte-identical pre-rework report; the live `historical`
    section is always derived from it so reruns are stable.
    """
    backup = out_path.with_name(out_path.stem + ".historical.json")
    try:
        if backup.is_file():
            old = json.loads(backup.read_bytes().decode("utf-8"))
            return old if isinstance(old, dict) else None
    except (OSError, ValueError):
        return None
    return None


def _read_report_for_history(out_path, create_backup):
    try:
        if not out_path.is_file():
            return None
        raw = out_path.read_bytes()
        try:
            old = json.loads(raw.decode("utf-8"))
        except ValueError:
            return None
        if isinstance(old, dict) and old.get("report_version") == REPORT_VERSION:
            return old
        if create_backup:
            backup = out_path.with_name(out_path.stem + ".historical.json")
            if not backup.is_file():
                # Byte-identical copy: history must reproduce the original
                # file exactly (a json round-trip is not byte-stable).
                backup.write_bytes(raw)
                print(f"preserved historical report -> {backup}")
        return old if isinstance(old, dict) else None
    except OSError:
        return None


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="P0-T007 retrieval benchmark.")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "docs" / "benchmarks" / "retrieval-v1-results.json")
    args = parser.parse_args(argv)
    try:
        corpus, queries_doc, assets, queries = load_eval_inputs(
            REPO_ROOT / "benchmarks" / "corpus" / "corpus.json",
            REPO_ROOT / "benchmarks" / "corpus" / "queries.json")
    except ValueError as exc:
        fail("E_JUDGMENTS", str(exc))

    import numpy as np
    store, manifest = {}, json.loads((VEC_DIR / "vectors-manifest.json").read_text())
    # Provenance gate: the cache must come from the pinned model/config,
    # not just any npz files with the right names (review finding).
    for field, expected in (("revision", REVISION), ("torch_dtype", "torch.float32"),
                            ("native_dim", 768)):
        if manifest.get(field) != expected:
            fail("E_PROVENANCE",
                 f"vector cache {field}={manifest.get(field)!r} != {expected!r}; "
                 "re-run embed_corpus.py with the pinned snapshot")
    if (manifest.get("corpus_id"), manifest.get("corpus_version")) != \
       (corpus.get("corpus_id"), corpus.get("version")):
        fail("E_PROVENANCE", "vector cache corpus version differs from manifests")
    for name in ("image", "video", "query", "keyframe"):
        with np.load(VEC_DIR / f"{name}.npz") as z:
            for k in z.files:
                vec = np.asarray(z[k], dtype=np.float32)
                try:
                    check_vector_values([float(v) for v in vec], 768)
                except ValueError as exc:
                    fail("E_STALE_CACHE", f"cached unit {k}: {exc}")
                store[k] = vec
    # Space-compat: every cached unit must share the query space (P0-T006 fix:
    # modality pipelines differ by design; space fields must match).
    q_space = make_sig("text-query", 768, prep="p0t007-v1")
    for key in store:
        kind = key.split(":")[0]
        # keyframes are image-path vectors (whole-video single-vector baseline).
        modality = {"image": "image", "video": "video", "query": "text-query",
                    "keyframe": "image"}[kind]
        prep = "p0t007-v1"
        seg = "halves" if kind == "video" else "n/a"
        try:
            check_space_compatible(q_space, make_sig(modality, 768, prep=prep, segment=seg))
        except ValueError as exc:
            fail("E_SPACE", f"cached unit {key}: {exc}")

    image_keys = sorted(k for k in store if k.startswith("image:"))
    video_keys = sorted(k for k in store if k.startswith("video:"))
    all_keys = image_keys + video_keys  # candidates only; query:* never ranked
    query_keys = {q["query_id"]: f"query:{q['query_id']}" for q in queries}
    keyframe_keys = sorted(k for k in store if k.startswith("keyframe:"))
    if len(all_keys) + len(query_keys) + len(keyframe_keys) != len(store):
        fail("E_STORE", "unexpected keys in vector cache")

    def relevant_units(query):
        """Unit keys relevant to a query (grade>=1; video via overlap rule)."""
        return relevant_unit_keys(query, assets, video_keys)

    def filename_score(query_text, unit_key):
        # 3-part and 4-part video keys supported via the shared helper
        # (bare unpack of split(":") crashed on strategy keys -- fixed).
        return filename_score_for_key(query_text, unit_key, assets)

    rng = random.Random(42)
    results = {"report_version": REPORT_VERSION,
               "corpus_id": corpus["corpus_id"], "corpus_version": corpus["version"],
               "query_set_id": queries_doc["query_set_id"],
               "query_version": queries_doc["version"],
               "model_id": manifest["model_id"], "revision": manifest["revision"],
               "torch_dtype": manifest["torch_dtype"], "dims": list(DIMS),
               "gate_recall5": GATE_RECALL5, "by_dim": {}, "baselines": {},
               "per_query": {}, "negatives": {},
               "temporal_policies": {
                   "tolerant": "original judgments: expected interval "
                               "expanded by tolerance_ms both sides "
                               "(bridges adjacent-segment seams by design)",
                   "strict": "sensitivity only: zero-tolerance overlap "
                             "(adjacent edge-sharing segments do NOT match)"},
               "candidate_pool": {"n_units": len(all_keys),
                                  "n_images": len(image_keys),
                                  "n_video_segments": len(video_keys)},
               "limitations": [
                   "Segment-level Recall denominators depend on the "
                   "temporal policy AND the segmentation: finer policies "
                   "yield more relevant units per query. Do not rank "
                   "strategies by raw segment Recall alone; use "
                   "file-grouped metrics for file-retrieval comparisons.",
                   "Tolerant matching marks adjacent video segments "
                   "relevant when tolerance bridges the seam (e.g. halves "
                   "at the expected boundary): relevant-unit counts differ "
                   "strict vs tolerant and are reported per query.",
                   "Synthetic corpus; single run; retrieval-only latency "
                   "is not a p95 claim."]} 
    lat = []
    for dim in DIMS:
        tvec = {k: truncate_renorm([float(v) for v in store[k]], dim) for k in store}
        dim_res = {"classes": {}, "modalities": {}}
        for cls in ("visual", "exact_filename", "semantic_image"):
            cls_q = [q for q in queries if q["query_class"] == cls]
            rec1, rec5, mrr, errs = [], [], [], []
            for q in cls_q:
                rel = relevant_units(q)
                rel_keys = [k for k, _g in rel]
                t0 = time.perf_counter()
                ranked = rank_candidates(tvec[query_keys[q["query_id"]]],
                                         [(k, tvec[k]) for k in all_keys])
                lat.append(time.perf_counter() - t0)
                keys = [k for k, _s in ranked]
                r1 = recall_at_k(rel_keys, keys, 1)
                r5 = recall_at_k(rel_keys, keys, 5)
                m = mrr_at(rel_keys, keys, 10)
                rec1.append(r1)
                rec5.append(r5)
                mrr.append(m)
                top = [{"unit": k, "cos": round(s, 4)} for k, s in ranked[:5]]
                err = None
                if r5 < 1.0:
                    missed = [k for k in rel_keys if k not in keys[:5]]
                    err = {"missed_units": missed,
                           "expected_grades": {k: g for k, g in rel}}
                qid = q["query_id"]
                strict_rel = relevant_unit_keys_policy(q, assets, video_keys,
                                                       policy="strict")
                results["per_query"].setdefault(qid, {})[str(dim)] = {
                    "recall@1": r1, "recall@5": r5, "mrr@10": m,
                    "top5": top, "error": err,
                    "pool_size": len(all_keys),
                    "relevant_tolerant_n": len(rel_keys),
                    "relevant_strict_n": len(strict_rel),
                    "relevant_files": relevant_file_assets(q, assets)}
                if err:
                    errs.append({"query_id": qid, "text": q["text"], **err})
            n = len(cls_q)
            dim_res["classes"][cls] = {
                "n": n, "pool_size": len(all_keys),
                "recall@1": sum(rec1) / n, "recall@5": sum(rec5) / n,
                "mrr@10": sum(mrr) / n, "errors": errs} if n else {"n": 0}
        for mod, mod_q, pool in (
                ("image", [q for q in queries
                            if q["query_class"] == "visual" and q["modality"] == "image"],
                 image_keys),
                ("video", [q for q in queries
                            if q["query_class"] == "visual" and q["modality"] == "video"],
                 video_keys)):
            rec1, rec5, mrr = [], [], []
            for q in mod_q:
                rel_keys = [k for k, _g in relevant_units(q)]
                ranked = rank_candidates(tvec[query_keys[q["query_id"]]],
                                         [(k, tvec[k]) for k in pool])
                keys = [k for k, _s in ranked]
                rec1.append(recall_at_k(rel_keys, keys, 1))
                rec5.append(recall_at_k(rel_keys, keys, 5))
                mrr.append(mrr_at(rel_keys, keys, 10))
            n = len(mod_q)
            dim_res["modalities"][mod] = {
                "n": n, "pool_size": len(pool),
                "recall@1": sum(rec1) / n, "recall@5": sum(rec5) / n,
                "mrr@10": sum(mrr) / n} if n else {"n": 0}
        # Strict-vs-tolerant sensitivity on the visual suite (same rankings
        # re-scored under both temporal policies) + file-grouped metrics.
        # The gate verdict always uses the tolerant (original) policy.
        vis_q = [q for q in queries if q["query_class"] == "visual"]
        sens = {}
        for policy in ("tolerant", "strict"):
            s1, s5, sm, f1, f5, fm = [], [], [], [], [], []
            for q in vis_q:
                rel_keys = [k for k, _g in relevant_unit_keys_policy(
                    q, assets, video_keys, policy=policy)]
                ranked = rank_candidates(tvec[query_keys[q["query_id"]]],
                                         [(k, tvec[k]) for k in all_keys])
                keys = [k for k, _s in ranked]
                s1.append(recall_at_k(rel_keys, keys, 1))
                s5.append(recall_at_k(rel_keys, keys, 5))
                sm.append(mrr_at(rel_keys, keys, 10))
                rel_files = relevant_file_assets(q, assets)
                file_rank = group_rank_by_file(ranked)
                f1.append(recall_at_k(rel_files, file_rank, 1))
                f5.append(recall_at_k(rel_files, file_rank, 5))
                fm.append(mrr_at(rel_files, file_rank, 10))
            nv = len(vis_q)
            sens[policy] = {
                "n": nv, "pool_size": len(all_keys),
                "seg_recall@1": sum(s1) / nv, "seg_recall@5": sum(s5) / nv,
                "seg_mrr@10": sum(sm) / nv,
                "file_recall@1": sum(f1) / nv,
                "file_recall@5": sum(f5) / nv, "file_mrr@10": sum(fm) / nv}
        tol_counts = [len(relevant_unit_keys_policy(q, assets, video_keys,
                                                    policy="tolerant"))
                      for q in vis_q]
        str_counts = [len(relevant_unit_keys_policy(q, assets, video_keys,
                                                    policy="strict"))
                      for q in vis_q]
        sens["relevant_units"] = {
            "per_query_tolerant_n": {q["query_id"]: c for q, c in
                                     zip(vis_q, tol_counts)},
            "per_query_strict_n": {q["query_id"]: c for q, c in
                                   zip(vis_q, str_counts)},
            "mean_tolerant_n": sum(tol_counts) / len(tol_counts),
            "mean_strict_n": sum(str_counts) / len(str_counts),
            "note": "tolerance bridges adjacent-segment seams, so "
                    "tolerant counts >= strict counts on seam-aligned "
                    "policies (e.g. halves); the Recall denominator moves "
                    "with the policy"}
        dim_res["sensitivity"] = sens
        # Top-1 timestamp error over video-visual queries (tolerant hits
        # only) with sample count AND miss rate (misses live in Recall).
        vid_q = [q for q in vis_q if q["modality"] == "video"]
        terr, thits = [], 0
        for q in vid_q:
            ranked = rank_candidates(tvec[query_keys[q["query_id"]]],
                                     [(k, tvec[k]) for k in all_keys])
            err_ms = top1_time_error(q, ranked[0][0], assets,
                                     policy="tolerant")
            if err_ms is not None:
                terr.append(err_ms)
                thits += 1
        dim_res["timestamp_tolerant"] = {
            "n_queries": len(vid_q), "n_hits": thits,
            "misses": len(vid_q) - thits,
            "miss_rate": (len(vid_q) - thits) / len(vid_q) if vid_q else None,
            "top1_time_err_ms_mean": (sum(terr) / len(terr)) if terr else None,
            "temporal_policy": "tolerant"}
        # Negatives: calibration only (no threshold exists) -- top-1 cosine stats.
        neg_q = [q for q in queries if q["query_class"] == "negative"]
        top1 = []
        for q in neg_q:
            ranked = rank_candidates(tvec[query_keys[q["query_id"]]],
                                     [(k, tvec[k]) for k in all_keys])
            top1.append(ranked[0][1])
        dim_res["negatives"] = {"n": len(neg_q),
                                "top1_cos_mean": sum(top1) / len(top1) if top1 else None,
                                "top1_cos_max": max(top1) if top1 else None}
        results["by_dim"][str(dim)] = dim_res

    # Baselines on the visual suite @768 (deterministic; no model).
    # keyframe: whole-video single middle-frame vectors + all images ranked
    # together -- the segment-vs-single-vector contrast behind R-08.
    vis_q = [q for q in queries if q["query_class"] == "visual"]
    kb_pool = image_keys + keyframe_keys
    for name in ("filename", "random", "keyframe"):
        rec1, rec5, mrr = [], [], []
        for q in vis_q:
            if name == "keyframe":
                rel_keys = []
                for exp in q.get("expected", []):
                    if exp.get("grade", 2) < 1:
                        continue
                    aid = exp["asset_id"]
                    if assets[aid]["modality"] == "image":
                        rel_keys.append(f"image:{aid}")
                    else:
                        rel_keys.append(f"keyframe:{aid}")
                order = kb_pool
            else:
                rel_keys = [k for k, _g in relevant_units(q)]
                order = all_keys
            if name == "filename":
                scored = sorted(order,
                                key=lambda k: (-filename_score(q["text"], k), k))
            elif name == "random":
                scored = list(order)
                rng.shuffle(scored)
            else:  # keyframe semantic baseline @768
                tvec_kf = {k: truncate_renorm([float(v) for v in store[k]], 768)
                           for k in order}
                tvec_kf[f"query:{q['query_id']}"] = truncate_renorm(
                    [float(v) for v in store[f"query:{q['query_id']}"]], 768)
                ranked = rank_candidates(
                    tvec_kf[f"query:{q['query_id']}"],
                    [(k, tvec_kf[k]) for k in order])
                scored = [k for k, _s in ranked]
            rec1.append(recall_at_k(rel_keys, scored, 1))
            rec5.append(recall_at_k(rel_keys, scored, 5))
            mrr.append(mrr_at(rel_keys, scored, 10))
        n = len(vis_q)
        results["baselines"][name] = {"n": n, "pool_size": len(order),
                                      "recall@1": sum(rec1) / n,
                                      "recall@5": sum(rec5) / n,
                                      "mrr@10": sum(mrr) / n}
    gate_val = results["by_dim"]["768"]["classes"]["visual"]["recall@5"]
    try:
        check_gate(gate_val, GATE_RECALL5, "Recall@5 visual@768")
        verdict = "GO-SIGNAL"
    except EvaluationGateError as exc:
        verdict = f"PIVOT-SIGNAL ({exc})"
    results["verdict"] = verdict
    results["verdict_policy"] = ("tolerant (original judgments); strict is "
                                 "sensitivity-only, see by_dim[*][sensitivity]")
    results["latency"] = {"retrieval_only_mean_s": sum(lat) / len(lat),
                          "n_rankings": len(lat)}
    # v2 index accounting: candidate-vector bytes (the served index) split
    # from query-vector bytes (evaluation-only). "bytes" keeps the legacy
    # lumped total so v1 readers still find their field.
    size = index_size_report(VEC_DIR)
    results["index"] = {"bytes": size["total_bytes"],
                        "candidate_bytes": size["candidate_bytes"],
                        "query_bytes": size["query_bytes"],
                        "n_units": len(all_keys),
                        "bytes_per_unit": size["total_bytes"] / len(all_keys),
                        "bytes_per_candidate_unit":
                            size["candidate_bytes"] / len(all_keys),
                        "note": "bytes_per_candidate_unit is the served-index "
                                "figure; bytes/bytes_per_unit are legacy "
                                "lumped totals kept for v1 comparability"}
    historical = preserve_historical_report(args.out)
    frozen = load_historical_backup(args.out) or historical
    if frozen:
        results["historical"] = {
            "verdict": frozen.get("verdict"),
            "visual_recall5_768": {
                "value": (frozen.get("by_dim", {})
                          .get("768", {}).get("classes", {})
                          .get("visual", {}).get("recall@5")),
                "note": "pre-rework measurement, tolerant policy, gate 0.75"}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"corpus={corpus['corpus_id']}v{corpus['version']} "
          f"queries={queries_doc['query_set_id']}v{queries_doc['version']} "
          f"units={len(all_keys)}")
    for dim in DIMS:
        c = results["by_dim"][str(dim)]["classes"]["visual"]
        s = results["by_dim"][str(dim)]["sensitivity"]
        t = results["by_dim"][str(dim)]["timestamp_tolerant"]
        print(f"dim {dim}: visual(n={c['n']},pool={c['pool_size']}) "
              f"R@1={c['recall@1']:.3f} "
              f"R@5={c['recall@5']:.3f} MRR@10={c['mrr@10']:.3f} "
              f"errors={len(c['errors'])}")
        print(f"  sensitivity: tolerant segR@5={s['tolerant']['seg_recall@5']:.3f} "
              f"fileR@5={s['tolerant']['file_recall@5']:.3f} | strict "
              f"segR@5={s['strict']['seg_recall@5']:.3f} "
              f"fileR@5={s['strict']['file_recall@5']:.3f} "
              f"(rel units mean {s['relevant_units']['mean_tolerant_n']:.2f} "
              f"vs {s['relevant_units']['mean_strict_n']:.2f})")
        print(f"  timestamp(tolerant): mean={t['top1_time_err_ms_mean']}ms "
              f"hits={t['n_hits']}/{t['n_queries']} "
              f"miss_rate={t['miss_rate']:.3f}" if t["miss_rate"] is not None
              else "  timestamp(tolerant): no hits")
    for name, b in results["baselines"].items():
        print(f"baseline {name}: R@1={b['recall@1']:.3f} R@5={b['recall@5']:.3f} "
              f"MRR@10={b['mrr@10']:.3f} (pool={b['pool_size']})")
    print(f"gate Recall@5>=0.75 @768 visual: {gate_val:.3f} -> {verdict}")
    print(f"retrieval-only latency mean={results['latency']['retrieval_only_mean_s']*1000:.1f}ms; "
          f"index candidates={size['candidate_bytes']}B "
          f"queries={size['query_bytes']}B total={size['total_bytes']}B")
    print(f"results: {args.out}")
    print("EVALUATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
