#!/usr/bin/env python3
"""Validate a LocalFind Phase-0 corpus manifest + query judgments (P0-T003).

Checks: schema shape, stable unique IDs, relative-path hygiene, license /
distribute / provenance presence, dangling asset references, image-vs-video
timestamp rules, time-range boundaries, duplicate IDs, negative-query handling,
and (with --media-root) that every rel_path exists on disk.

Usage:
    python benchmarks/validate_corpus.py --corpus <corpus.json> --queries <queries.json> [--media-root <dir>]

Exit codes: 0 = valid (warnings allowed), 1 = validation errors, 2 = usage/IO error.
Stdlib only. No network. Never modifies the corpus.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ASSET_ID_RE = re.compile(r"^[a-z]+_[0-9]{3}$")
QUERY_ID_RE = re.compile(r"^q[0-9]{3}$")
MODALITIES = ("image", "video")
QUERY_CLASSES = ("visual", "exact_filename", "semantic_image", "negative")
GRADES = (1, 2)


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, code, message):
        self.errors.append(f"{code}: {message}")

    def warning(self, message):
        self.warnings.append(f"WARN: {message}")

    def ok(self):
        return not self.errors


def load_json(path: Path, rep: Report):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        rep.error("E_IO", f"file not found: {path}")
    except json.JSONDecodeError as exc:
        rep.error("E_JSON", f"{path}: invalid JSON: {exc}")
    return None


def check_rel_path(value, asset_id: str, rep: Report):
    if not isinstance(value, str) or not value:
        rep.error("E_PATH", f"asset {asset_id}: rel_path must be a non-empty string")
        return
    if "\\" in value:
        rep.error("E_PATH", f"asset {asset_id}: rel_path must use forward slashes: {value!r}")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value) or value.startswith("\\\\"):
        rep.error("E_PATH", f"asset {asset_id}: rel_path must be relative: {value!r}")
    parts = [p for p in value.split("/") if p not in ("", ".")]
    if ".." in parts:
        rep.error("E_PATH", f"asset {asset_id}: rel_path must not contain '..': {value!r}")


def validate_corpus(doc, rep: Report):
    assets = []
    if not isinstance(doc, dict):
        rep.error("E_SCHEMA", "corpus manifest must be a JSON object")
        return assets
    for key in ("corpus_id", "version", "assets"):
        if key not in doc:
            rep.error("E_SCHEMA", f"corpus manifest missing required key: {key!r}")
    if not isinstance(doc.get("assets"), list):
        rep.error("E_SCHEMA", "corpus manifest 'assets' must be an array")
        return assets
    seen_ids, seen_paths = set(), set()
    for i, asset in enumerate(doc["assets"]):
        where = f"assets[{i}]"
        if not isinstance(asset, dict):
            rep.error("E_SCHEMA", f"{where} must be an object")
            continue
        aid = asset.get("asset_id")
        if not isinstance(aid, str) or not ASSET_ID_RE.match(aid):
            rep.error("E_ASSET_ID", f"{where}: asset_id must match ^[a-z]+_[0-9]{{3}}$ (got {aid!r})")
            aid = f"<bad-{i}>"
        elif aid in seen_ids:
            rep.error("E_DUP_ASSET_ID", f"duplicate asset_id: {aid}")
        seen_ids.add(aid)
        modality = asset.get("modality")
        if modality not in MODALITIES:
            rep.error("E_MODALITY", f"asset {aid}: modality must be image|video (got {modality!r})")
        rel = asset.get("rel_path")
        check_rel_path(rel, aid, rep)
        if isinstance(rel, str) and rel in seen_paths:
            rep.error("E_DUP_PATH", f"duplicate rel_path: {rel!r}")
        seen_paths.add(rel)
        for field in ("license", "provenance", "description"):
            if not isinstance(asset.get(field), str) or not asset[field].strip():
                rep.error("E_META", f"asset {aid}: {field} must be a non-empty string")
        if not isinstance(asset.get("distribute"), bool):
            rep.error("E_META", f"asset {aid}: distribute must be boolean")
        duration = asset.get("duration_ms")
        if modality == "video":
            if not isinstance(duration, int) or duration <= 0:
                rep.error("E_DURATION", f"asset {aid}: video requires duration_ms > 0 (got {duration!r})")
        elif modality == "image" and duration is not None:
            rep.error("E_DURATION", f"asset {aid}: image must not set duration_ms")
        assets.append(asset)
    return assets


def validate_queries(doc, assets_by_id: dict, corpus_id, rep: Report):
    if not isinstance(doc, dict):
        rep.error("E_SCHEMA", "queries file must be a JSON object")
        return []
    for key in ("query_set_id", "version", "corpus_id", "queries"):
        if key not in doc:
            rep.error("E_SCHEMA", f"queries file missing required key: {key!r}")
    if doc.get("corpus_id") != corpus_id:
        rep.error("E_CORPUS_MISMATCH",
                  f"queries corpus_id {doc.get('corpus_id')!r} != corpus {corpus_id!r}")
    if not isinstance(doc.get("queries"), list):
        rep.error("E_SCHEMA", "queries file 'queries' must be an array")
        return []
    seen_qids = set()
    for i, query in enumerate(doc["queries"]):
        where = f"queries[{i}]"
        if not isinstance(query, dict):
            rep.error("E_SCHEMA", f"{where} must be an object")
            continue
        qid = query.get("query_id")
        if not isinstance(qid, str) or not QUERY_ID_RE.match(qid):
            rep.error("E_QUERY_ID", f"{where}: query_id must match ^q[0-9]{{3}}$ (got {qid!r})")
            qid = f"<bad-{i}>"
        elif qid in seen_qids:
            rep.error("E_DUP_QUERY_ID", f"duplicate query_id: {qid}")
        seen_qids.add(qid)
        if not isinstance(query.get("text"), str) or not query["text"].strip():
            rep.error("E_TEXT", f"query {qid}: text must be non-empty")
        qclass = query.get("query_class")
        if qclass not in QUERY_CLASSES:
            rep.error("E_CLASS", f"query {qid}: query_class must be one of {QUERY_CLASSES} (got {qclass!r})")
        qmod = query.get("modality")
        if qmod not in ("image", "video", "any"):
            rep.error("E_QMODALITY", f"query {qid}: modality must be image|video|any (got {qmod!r})")
        if not isinstance(query.get("provenance"), str) or not query["provenance"].strip():
            rep.error("E_META", f"query {qid}: provenance must be a non-empty string")
        expected = query.get("expected")
        if not isinstance(expected, list):
            rep.error("E_SCHEMA", f"query {qid}: expected must be an array")
            continue
        if qclass == "negative" and expected:
            rep.error("E_NEG_WITH_EXPECTED", f"query {qid}: negative query must have empty expected[]")
        if qclass != "negative" and not expected:
            rep.warning(f"query {qid}: non-negative query with empty expected[] (unjudged?)")
        for j, exp in enumerate(expected):
            ewhere = f"query {qid} expected[{j}]"
            if not isinstance(exp, dict):
                rep.error("E_SCHEMA", f"{ewhere} must be an object")
                continue
            aid = exp.get("asset_id")
            asset = assets_by_id.get(aid)
            if asset is None:
                rep.error("E_DANGLING_REF", f"{ewhere}: unknown asset_id {aid!r}")
                continue
            if exp.get("grade") not in GRADES:
                rep.error("E_GRADE", f"{ewhere}: grade must be 1 or 2 (got {exp.get('grade')!r})")
            has_range = "start_ms" in exp or "end_ms" in exp
            if asset.get("modality") == "image":
                if has_range:
                    rep.error("E_IMG_TIMESTAMP",
                              f"{ewhere}: image expectation must not carry start_ms/end_ms")
                if "tolerance_ms" in exp:
                    rep.warning(f"{ewhere}: tolerance_ms is meaningless for images")
            else:
                start, end = exp.get("start_ms"), exp.get("end_ms")
                if not isinstance(start, int) or not isinstance(end, int):
                    rep.error("E_VID_TIMESTAMP_MISSING",
                              f"{ewhere}: video expectation requires integer start_ms/end_ms")
                    continue
                duration = asset.get("duration_ms")
                if start < 0 or start >= end:
                    rep.error("E_BAD_RANGE",
                              f"{ewhere}: require 0 <= start_ms < end_ms (got {start}, {end})")
                if isinstance(duration, int) and end > duration:
                    rep.error("E_RANGE_EXCEEDS_DURATION",
                              f"{ewhere}: end_ms {end} exceeds asset duration {duration} ms")
                tol = exp.get("tolerance_ms", 0)
                if not isinstance(tol, int) or tol < 0:
                    rep.error("E_TOLERANCE", f"{ewhere}: tolerance_ms must be >= 0")
    return doc["queries"] if isinstance(doc.get("queries"), list) else []


def ffprobe_duration_ms(ffprobe: str, path: Path):
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(round(float(proc.stdout.strip()) * 1000))
    except ValueError:
        return None


def check_media(assets, media_root: Path, rep: Report):
    ffprobe = shutil.which("ffprobe")
    for asset in assets:
        rel = asset.get("rel_path")
        if not isinstance(rel, str):
            continue
        target = media_root / rel
        if not target.is_file():
            rep.error("E_MEDIA_MISSING",
                      f"asset {asset['asset_id']}: file not found under media root: {rel}")
            continue
        if asset.get("modality") == "video" and ffprobe:
            measured = ffprobe_duration_ms(ffprobe, target)
            declared = asset.get("duration_ms")
            if measured is None:
                rep.warning(f"asset {asset['asset_id']}: ffprobe could not read duration")
            elif isinstance(declared, int) and abs(measured - declared) > 500:
                rep.error("E_DURATION_MISMATCH",
                          f"asset {asset['asset_id']}: manifest {declared} ms vs measured {measured} ms")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate LocalFind Phase-0 corpus + judgments.")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--media-root", type=Path, default=None,
                        help="Corpus root dir; when given, every rel_path must exist on disk.")
    args = parser.parse_args(argv)
    rep = Report()
    corpus = load_json(args.corpus, rep)
    queries_doc = load_json(args.queries, rep)
    if corpus is None or queries_doc is None:
        print("\n".join(rep.errors), file=sys.stderr)
        return 2
    assets = validate_corpus(corpus, rep)
    assets_by_id = {a["asset_id"]: a for a in assets if isinstance(a.get("asset_id"), str)}
    queries = validate_queries(queries_doc, assets_by_id, corpus.get("corpus_id"), rep)
    if args.media_root is not None:
        if not args.media_root.is_dir():
            rep.error("E_IO", f"media root is not a directory: {args.media_root}")
        else:
            check_media(assets, args.media_root, rep)
    n_exp = sum(len(q.get("expected", [])) for q in queries if isinstance(q, dict))
    classes: dict = {}
    for q in queries:
        if isinstance(q, dict):
            classes[q.get("query_class")] = classes.get(q.get("query_class"), 0) + 1
    print(f"assets={len(assets)} queries={len(queries)} expectations={n_exp} classes={classes}")
    for warning in rep.warnings:
        print(warning)
    if rep.ok():
        print("corpus VALID")
        return 0
    print("corpus INVALID:", file=sys.stderr)
    for error in rep.errors:
        print(f"  {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
