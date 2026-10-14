#!/usr/bin/env python3
"""P0-T010 minimal inference sidecar prototype (packaging spike, NOT production).

Spawns as a child process of a minimal host and serves versioned embedding
requests over newline-delimited JSON on stdin/stdout (PLAN ADR-007 shape).
stderr carries human logs only -- stdout is protocol-only.

Protocol v1.0 (one JSON object per line, UTF-8):
  request : {protocol_version:"1.0", request_id:str, op:str, payload:{...}}
  response: {protocol_version:"1.0", request_id, ok:true, result:{...}}
          | {protocol_version:"1.0", request_id, ok:false, error:{code,message,retryable}}

Ops: handshake | embed_text | embed_image | shutdown
  handshake   -> {protocol_version, sidecar, sidecar_version, ops, limits,
                  model:{loaded, id, revision, source, device, dtype, encoders}}
  embed_text  payload {text, prompt="SearchQuery"|"Document", dim=768}
  embed_image payload {path, dim=768}
  shutdown    -> {bye:true}, then exit 0. EOF on stdin also exits 0.

Embed result: {embedding:[float], dim, dtype, norm, sig, sig_string,
               model:{...}, timings_ms:{encode, (load on first call)}}

Model config is the P0-T004 proven one: text+image encoders, fp32, CPU.
Anything else (cuda/fp16/bf16) is rejected with E_CONFIG -- never silent.

Compatibility logic is intentionally self-contained here (duplicated from
benchmarks/local_model.py + benchmarks/compat.py, cited below) so the
packaged artifact is a single script with pinned third-party deps and no
repository imports. A production worker would share a real package.

Security properties (spike scope):
  - allowlisted ops only; unknown op -> E_OP (no eval/exec, no shell).
  - request_id required (1..128 chars); version must equal "1.0".
  - text length bound (1..8192 chars); image bound (exists, allowlisted
    extension, <=20 MiB, decoded pixels <=25 MP).
  - error messages never echo user content or full paths (basename only).
  - stderr logs carry event ids + timings, never text/vectors/paths.
  - offline by construction: HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are forced
    and the model loads with local_files_only=True (no download path).
  - one request at a time (single-threaded loop); host enforces timeouts.

Exit codes: 0 clean (EOF/shutdown), 2 startup/argument failure.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Offline first: set before any HF/transformers import below (lazy, in load).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROTOCOL_VERSION = "1.0"
SIDECAR_VERSION = "0.1.0"
OPS = ("handshake", "embed_text", "embed_image", "shutdown")

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
SHORT_REV = REVISION[:7]
ENCODERS = "text+image"
PREP = "p0t010-v1"  # spike preprocessing policy tag (P0-T006 contract field)
NATIVE_DIM = 768
SUPPORTED_DIMS = (768, 512, 256, 128)
NORM_TOL = 1e-4
REQUIRED_FILES = ("model.safetensors", "config.json", "tokenizer.json",
                  "processor_config.json")
MAX_TEXT_CHARS = 8192
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
PROMPTS = ("SearchQuery", "Document")


def log(level, event, detail=""):
    # stderr only; never log user content, vectors, or full paths (R-04).
    ts = time.strftime("%H:%M:%S")
    line = f"{ts} {level} {event}" + (f" {detail}" if detail else "")
    print(line, file=sys.stderr, flush=True)


def err_response(request_id, code, message, retryable=False):
    return {"protocol_version": PROTOCOL_VERSION, "request_id": request_id,
            "ok": False,
            "error": {"code": code, "message": message,
                      "retryable": retryable}}


def ok_response(request_id, result):
    return {"protocol_version": PROTOCOL_VERSION, "request_id": request_id,
            "ok": True, "result": result}


def resolve_model_dir(explicit):
    """Return (Path, source_label) or raise FileNotFoundError(E_MODEL_DIR).

    Precedence: explicit --model-dir / LOCALFIND_MODEL_DIR >
    <cwd>/models/embeddinggemma-2 > <exe-or-script-dir>/models/embeddinggemma-2.
    No Hugging Face hub fallback: this spike is offline-only by design.
    (mirrors benchmarks/local_model.py::resolve_model_source, minus the hub.)
    """
    candidates = []
    if explicit:
        candidates.append((Path(explicit), "explicit-dir"))
    candidates.append((Path.cwd() / "models" / "embeddinggemma-2", "cwd-relative"))
    base = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    candidates.append((base / "models" / "embeddinggemma-2", "exe-relative"))
    for cand, label in candidates:
        if all((cand / name).is_file() for name in REQUIRED_FILES):
            return cand, label
    tried = ", ".join(str(c) for c, _ in candidates)
    raise FileNotFoundError(
        "E_MODEL_DIR: no snapshot-ready model dir (need "
        + ",".join(REQUIRED_FILES) + "); tried " + tried
        + "; pass --model-dir or set LOCALFIND_MODEL_DIR")


def make_sig(modality, dim):
    return {"model": "eg2", "model_id": MODEL_ID, "revision": SHORT_REV,
            "encoders": ENCODERS, "prep": PREP, "modality": modality,
            "dim": dim, "dtype": "fp32", "norm": "l2", "segment": "n/a"}


def sig_string(sig):
    return ("{model}@{revision}|{encoders}|{prep}|{modality}|{segment}|"
            "{dim}|{dtype}|{norm}".format(**sig))


def check_vector_values(vals, dim):
    if len(vals) != dim:
        raise ValueError(f"E_DIM: len {len(vals)} != {dim}")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("E_NONFINITE: NaN or Inf present")
    norm = math.sqrt(sum(v * v for v in vals))
    if norm == 0.0:
        raise ValueError("E_ZERO_NORM: zero vector")
    if abs(norm - 1.0) > NORM_TOL:
        raise ValueError(f"E_NORM: norm {norm:.6f} != 1.0")
    return norm


def truncate_renorm(vals, dim):
    if dim == NATIVE_DIM:
        return list(vals)
    part = list(vals[:dim])
    norm = math.sqrt(sum(v * v for v in part))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("E_TRUNC: truncated slice has zero/non-finite norm")
    return [v / norm for v in part]


class Sidecar:
    def __init__(self, model_dir, device, dtype):
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.model = None
        self.model_info = {"id": MODEL_ID, "revision": SHORT_REV,
                           "source": None, "device": device, "dtype": dtype,
                           "encoders": ENCODERS}
        self.load_ms = None

    def ensure_model(self):
        if self.model is not None:
            return None  # warm; no load time
        try:
            model_path, label = resolve_model_dir(self.model_dir)
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
        import torch
        from sentence_transformers import SentenceTransformer
        t0 = time.perf_counter()
        try:
            self.model = SentenceTransformer(
                str(model_path), device=self.device,
                model_kwargs={"torch_dtype": torch.float32},
                config_kwargs={"audio_config": None},
                local_files_only=True,
            )
        except Exception as exc:  # noqa: BLE001 -- typed below, never silent
            raise ValueError(f"E_LOAD: model load failed [{type(exc).__name__}]") from exc
        self.load_ms = (time.perf_counter() - t0) * 1000.0
        dtypes = {str(p.dtype) for p in self.model.parameters()}
        if any("float16" in d for d in dtypes):
            raise ValueError("E_DTYPE: float16 detected (card forbids fp16)")
        self.model_info["source"] = label
        log("info", "model_loaded", f"source={label} ms={self.load_ms:.0f}")
        return self.load_ms

    def embed_text(self, payload):
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("E_VALIDATION: payload.text must be a non-empty string")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"E_VALIDATION: payload.text exceeds {MAX_TEXT_CHARS} chars")
        prompt = payload.get("prompt", "SearchQuery")
        if prompt not in PROMPTS:
            raise ValueError(f"E_VALIDATION: payload.prompt must be one of {PROMPTS}")
        dim = payload.get("dim", NATIVE_DIM)
        if dim not in SUPPORTED_DIMS:
            raise ValueError(f"E_VALIDATION: payload.dim must be one of {SUPPORTED_DIMS}")
        load_ms = self.ensure_model()
        assert self.model is not None  # ensure_model raises on failure
        t0 = time.perf_counter()
        vec = self.model.encode(text, prompt_name=prompt)
        encode_ms = (time.perf_counter() - t0) * 1000.0
        return self._finish(vec, "text", dim, encode_ms, load_ms)

    def embed_image(self, payload):
        raw = payload.get("path")
        if not isinstance(raw, str) or not raw:
            raise ValueError("E_VALIDATION: payload.path must be a non-empty string")
        path = Path(raw)
        name = path.name  # basename only in any message/log (R-04)
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"E_IMAGE: unsupported extension for {name!r}")
        if not path.is_file():
            raise ValueError(f"E_IMAGE: file not found: {name!r}")
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError(f"E_IMAGE: file exceeds size cap: {name!r}")
        dim = payload.get("dim", NATIVE_DIM)
        if dim not in SUPPORTED_DIMS:
            raise ValueError(f"E_VALIDATION: payload.dim must be one of {SUPPORTED_DIMS}")
        from PIL import Image
        try:
            with Image.open(path) as img:
                img.load()
                if img.width * img.height > MAX_IMAGE_PIXELS:
                    raise ValueError(f"E_IMAGE: decoded pixels exceed cap: {name!r}")
                rgb = img.convert("RGB").copy()
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 -- typed below
            raise ValueError(f"E_IMAGE: undecodable image: {name!r} [{type(exc).__name__}]") from exc
        load_ms = self.ensure_model()
        assert self.model is not None  # ensure_model raises on failure
        t0 = time.perf_counter()
        vec = self.model.encode(rgb)
        encode_ms = (time.perf_counter() - t0) * 1000.0
        return self._finish(vec, "image", dim, encode_ms, load_ms)

    def _finish(self, vec, modality, dim, encode_ms, load_ms):
        vals = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        vals = [float(v) for v in vals]
        if len(vals) != NATIVE_DIM:
            raise ValueError(f"E_DIM: native len {len(vals)} != {NATIVE_DIM}")
        out = truncate_renorm(vals, dim)
        norm = check_vector_values(out, dim)
        sig = make_sig(modality, dim)
        timings = {"encode_ms": encode_ms}
        if load_ms is not None:
            timings["load_ms"] = load_ms
        return {"embedding": out, "dim": dim, "dtype": "fp32", "norm": norm,
                "sig": sig, "sig_string": sig_string(sig),
                "model": dict(self.model_info), "timings_ms": timings}

    def handshake(self):
        return {"protocol_version": PROTOCOL_VERSION, "sidecar": "localfind-inference",
                "sidecar_version": SIDECAR_VERSION, "ops": list(OPS),
                "limits": {"max_text_chars": MAX_TEXT_CHARS,
                           "max_image_bytes": MAX_IMAGE_BYTES,
                           "max_image_pixels": MAX_IMAGE_PIXELS,
                           "image_exts": list(IMAGE_EXTS),
                           "dims": list(SUPPORTED_DIMS)},
                "model": dict(self.model_info, loaded=self.model is not None)}


def handle_line(sidecar, line):
    try:
        req = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        return err_response(None, "E_JSON", "request is not valid JSON")
    if not isinstance(req, dict):
        return err_response(None, "E_REQUEST", "request must be a JSON object")
    rid = req.get("request_id")
    if not isinstance(rid, str) or not (1 <= len(rid) <= 128):
        return err_response(None, "E_REQUEST_ID",
                            "request_id must be a string of 1..128 chars")
    if req.get("protocol_version") != PROTOCOL_VERSION:
        return err_response(rid, "E_VERSION",
                            f"protocol_version must be {PROTOCOL_VERSION!r}")
    op = req.get("op")
    if op not in OPS:
        return err_response(rid, "E_OP", f"unknown op {op!r}")
    payload = req.get("payload", {})
    if not isinstance(payload, dict):
        return err_response(rid, "E_VALIDATION", "payload must be an object")
    try:
        if op == "handshake":
            return ok_response(rid, sidecar.handshake())
        if op == "embed_text":
            return ok_response(rid, sidecar.embed_text(payload))
        if op == "embed_image":
            return ok_response(rid, sidecar.embed_image(payload))
        if op == "shutdown":
            return ok_response(rid, {"bye": True})
    except ValueError as exc:
        msg = str(exc)
        code = msg.split(":", 1)[0] if ":" in msg else "E_INTERNAL"
        if code not in ("E_VALIDATION", "E_IMAGE", "E_MODEL_DIR", "E_LOAD",
                        "E_DIM", "E_NONFINITE", "E_NORM", "E_TRUNC", "E_DTYPE"):
            code, msg, retry = "E_INTERNAL", "internal error", True
            log("error", "embed_failed", "E_INTERNAL")
            return err_response(rid, code, msg, retry)
        log("warn", "request_failed", code)
        return err_response(rid, code, msg, False)
    except Exception as exc:  # noqa: BLE001 -- last-resort guard, never silent
        # Type (+ missing-module) name only: fixed library identifiers, never
        # paths or user content (R-04). OSError filenames are NOT logged.
        detail = f"E_INTERNAL:{type(exc).__name__}"
        if isinstance(exc, ImportError) and getattr(exc, "name", None):
            detail += f":{exc.name}"
        elif isinstance(exc, OSError) and getattr(exc, "errno", None) is not None:
            detail += f":errno-{exc.errno}"
        log("error", "embed_failed", detail)
        if os.environ.get("LOCALFIND_SIDECAR_DEBUG") == "1":
            # Opt-in diagnostic mode only (R-04): may include local paths.
            # Never enabled by default or in committed probe runs.
            import traceback
            lines = traceback.format_exc().strip().splitlines()
            if os.environ.get("LOCALFIND_SIDECAR_DEBUG") == "2":
                keep = lines  # full chain for local diagnosis (never committed)
            else:
                keep = lines[:10] + ["..."] + lines[-10:]
            for tb_line in keep:
                log("debug", "trace", tb_line.strip()[:300])
        return err_response(rid, "E_INTERNAL", "internal error", True)
    return err_response(rid, "E_INTERNAL", "internal error", True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="LocalFind inference sidecar (P0-T010 spike).")
    parser.add_argument("--model-dir", default=os.environ.get("LOCALFIND_MODEL_DIR"),
                        help="Explicit local snapshot dir (else cwd/exe-relative models/).")
    parser.add_argument("--device", default="cpu", choices=["cpu"],
                        help="Spike supports the proven CPU path only.")
    parser.add_argument("--dtype", default="fp32", choices=["fp32"],
                        help="Spike supports fp32 only (card rule; never fp16).")
    args = parser.parse_args(argv)
    sidecar = Sidecar(args.model_dir, args.device, args.dtype)
    log("info", "startup",
        f"protocol={PROTOCOL_VERSION} sidecar={SIDECAR_VERSION} device={args.device}")
    stdin = sys.stdin
    stdout = sys.stdout
    while True:
        line = stdin.readline()
        if line == "":
            log("info", "eof_shutdown")
            return 0
        if len(line) > 1_000_000:
            resp = err_response(None, "E_VALIDATION", "request line exceeds 1 MiB")
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
            continue
        line = line.strip()
        if not line:
            continue
        resp = handle_line(sidecar, line)
        stdout.write(json.dumps(resp) + "\n")
        stdout.flush()
        req_op = None
        try:
            req_op = json.loads(line).get("op") if isinstance(json.loads(line), dict) else None
        except ValueError:
            pass
        if resp.get("ok") and req_op == "shutdown":
            log("info", "shutdown")
            return 0


if __name__ == "__main__":
    sys.exit(main())
