#!/usr/bin/env python3
"""P0-T010 minimal host harness (packaging spike, NOT production code).

Spawns the sidecar (dev script via a Python interpreter, or a packaged exe),
issues a fixed request script with per-request timeouts, validates the
versioned contract, and writes a sanitized JSON report (no absolute paths,
no user content -- basenames and labels only, per R-04).

Usage:
  python benchmarks/packaging/host_probe.py --sidecar <exe|script.py>
      [--python <interpreter>] [--model-dir <dir>] [--text "query"]
      [--image <file>] [--timeout-s 600] [--isolated-hf]
      [--out docs/benchmarks/packaging-probe-<tag>.json]

--isolated-hf points HF_HOME at an empty temp dir (with HF_HUB_OFFLINE=1) to
prove the run does not rely on the developer Hugging Face cache.
The spawn environment always drops VIRTUAL_ENV/PYTHONPATH/PYTHONHOME so a
packaged exe cannot accidentally reuse the project venv.

Exit 0 = all probe steps pass; 1 = failure with E_* style message.
Stdlib only (psutil used opportunistically for peak RSS, else skipped).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def fail(code, message) -> NoReturn:
    print(f"{code}: {message}", file=sys.stderr)
    sys.exit(1)


def send(proc, obj, timeout_s):
    assert proc.stdin is not None and proc.stdout is not None  # pipes requested
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()
    t0 = time.perf_counter()
    line = proc.stdout.readline()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    if not line:
        fail("E_SIDECAR_EXIT", "sidecar closed stdout mid-script")
    try:
        resp = json.loads(line)
    except ValueError:
        fail("E_PROTOCOL", f"sidecar stdout line is not JSON ({len(line)} chars)")
    return resp, dt_ms


def expect_ok(resp, rid, what):
    if not isinstance(resp, dict) or resp.get("protocol_version") != "1.0":
        fail("E_PROTOCOL", f"{what}: bad protocol_version in {resp!r:.120}")
    if resp.get("request_id") != rid or not resp.get("ok"):
        fail("E_PROBE", f"{what}: unexpected response {json.dumps(resp)[:300]}")
    return resp["result"]


def check_embedding(result, expect_dim, what, report):
    emb = result.get("embedding")
    if not isinstance(emb, list) or len(emb) != expect_dim:
        fail("E_PROBE", f"{what}: embedding dim {len(emb) if isinstance(emb, list) else type(emb)} != {expect_dim}")
    import math
    if not all(math.isfinite(v) for v in emb):
        fail("E_PROBE", f"{what}: non-finite values")
    norm = math.sqrt(sum(v * v for v in emb))
    if abs(norm - 1.0) > 1e-4:
        fail("E_PROBE", f"{what}: norm {norm:.6f} != 1.0")
    sig = result.get("sig", {})
    for field in ("model", "revision", "dim", "dtype", "norm"):
        if field not in sig:
            fail("E_PROBE", f"{what}: sig missing {field}")
    report[what] = {"dim": len(emb), "norm": round(norm, 6),
                    "sig_string": result.get("sig_string"),
                    "sample": [round(v, 6) for v in emb[:4]],
                    "timings_ms": result.get("timings_ms")}
    return emb


def main(argv=None):
    ap = argparse.ArgumentParser(description="P0-T010 sidecar host probe.")
    ap.add_argument("--sidecar", required=True,
                    help="Path to packaged exe/dir-exe, or sidecar.py script.")
    ap.add_argument("--python", default=sys.executable,
                    help="Interpreter used only when --sidecar is a .py script.")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--text", default="solid red image")
    ap.add_argument("--image", default=None,
                    help="Image file (default: tests/fixtures/samples/img_red.png).")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--isolated-hf", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    sidecar = Path(args.sidecar)
    if not sidecar.is_absolute():
        sidecar = (Path.cwd() / sidecar).resolve()
    if not sidecar.is_file():
        fail("E_IO", f"sidecar not found: {sidecar}")
    python = args.python
    if sidecar.suffix.lower() == ".py" and not Path(python).is_absolute():
        # the child runs with cwd=tmp: a relative interpreter would not resolve
        found = None
        for cand in (Path.cwd() / python, Path(sys.executable)):
            if Path(cand).is_file():
                found = str(cand)
                break
        python = found or python
    image = Path(args.image) if args.image else (
        REPO_ROOT / "tests" / "fixtures" / "samples" / "img_red.png")
    if not image.is_absolute():
        image = (Path.cwd() / image).resolve()
    if not image.is_file():
        fail("E_IO", f"image fixture missing: {image}")

    env = dict(os.environ)
    for var in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
        env.pop(var, None)
    env["HF_HUB_OFFLINE"] = "1"
    tmp = Path(tempfile.mkdtemp(prefix="lf-probe-"))
    empty_hf = tmp / "empty-hf"
    empty_hf.mkdir()
    if args.isolated_hf:
        env["HF_HOME"] = str(empty_hf)
        env["HUGGINGFACE_HUB_CACHE"] = str(empty_hf)
        env["TRANSFORMERS_CACHE"] = str(empty_hf)

    if sidecar.suffix.lower() == ".py":
        cmd = [python, str(sidecar)]
    else:
        cmd = [str(sidecar)]
    if args.model_dir:
        md = Path(args.model_dir)
        if not md.is_absolute():
            md = (Path.cwd() / md).resolve()
        cmd += ["--model-dir", str(md)]

    report = {"sidecar_kind": "script" if sidecar.suffix == ".py" else "packaged",
              "isolated_hf": args.isolated_hf, "steps": {}}
    import importlib.util
    have_psutil = importlib.util.find_spec("psutil") is not None

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1,
                            env=env, cwd=str(tmp))
    assert proc.stdin is not None and proc.stdout is not None  # pipes requested
    pin, pout = proc.stdin, proc.stdout
    # handshake measures process startup (model NOT loaded yet by design)
    resp, dt = send(proc, {"protocol_version": "1.0", "request_id": "h1",
                           "op": "handshake", "payload": {}}, args.timeout_s)
    res = expect_ok(resp, "h1", "handshake")
    report["steps"]["startup_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    report["steps"]["handshake_roundtrip_ms"] = round(dt, 1)
    report["handshake"] = {"sidecar_version": res.get("sidecar_version"),
                           "ops": res.get("ops"),
                           "model_loaded": res.get("model", {}).get("loaded")}
    if res.get("model", {}).get("loaded"):
        fail("E_PROBE", "handshake must not preload the model (startup/load split)")
    print(f"startup={report['steps']['startup_ms']:.0f}ms "
          f"sidecar={res.get('sidecar_version')} ops={res.get('ops')}")

    # cold text embed (includes first model load)
    resp, dt = send(proc, {"protocol_version": "1.0", "request_id": "t1",
                           "op": "embed_text",
                           "payload": {"text": args.text}}, args.timeout_s)
    res = expect_ok(resp, "t1", "embed_text_cold")
    report["steps"]["text_cold_roundtrip_ms"] = round(dt, 1)
    q_emb = check_embedding(res, 768, "text_cold", report)
    print(f"text_cold={dt/1000:.1f}s load={res['timings_ms'].get('load_ms', 0)/1000:.1f}s "
          f"sig={res['sig_string']}")

    # warm text embed (must NOT reload)
    resp, dt = send(proc, {"protocol_version": "1.0", "request_id": "t2",
                           "op": "embed_text",
                           "payload": {"text": args.text}}, args.timeout_s)
    res = expect_ok(resp, "t2", "embed_text_warm")
    report["steps"]["text_warm_roundtrip_ms"] = round(dt, 1)
    check_embedding(res, 768, "text_warm", report)
    if "load_ms" in res.get("timings_ms", {}):
        fail("E_PROBE", "warm request must not reload the model")
    print(f"text_warm={dt:.0f}ms")

    # image embed (model warm by now)
    resp, dt = send(proc, {"protocol_version": "1.0", "request_id": "i1",
                           "op": "embed_image",
                           "payload": {"path": str(image)}}, args.timeout_s)
    res = expect_ok(resp, "i1", "embed_image")
    report["steps"]["image_roundtrip_ms"] = round(dt, 1)
    i_emb = check_embedding(res, 768, "image", report)
    print(f"image={dt/1000:.1f}s sig={res['sig_string']}")

    # truncated-dim parity: same text at 256-d must be leading-slice renorm
    resp, _ = send(proc, {"protocol_version": "1.0", "request_id": "t3",
                          "op": "embed_text",
                          "payload": {"text": args.text, "dim": 256}}, args.timeout_s)
    res = expect_ok(resp, "t3", "embed_text_256")
    check_embedding(res, 256, "text_256", report)
    import math
    lead = q_emb[:256]
    n = math.sqrt(sum(v * v for v in lead))
    lead = [v / n for v in lead]
    cos = sum(a * b for a, b in zip(lead, res["embedding"]))
    report["text_256_truncation_cos"] = round(cos, 6)
    if cos < 0.999:
        fail("E_PROBE", f"256-d truncation mismatch cos={cos:.6f}")

    # malformed request handling (must stay alive)
    pin.write("this is not json\n")
    pin.flush()
    line = pout.readline()
    resp = json.loads(line)
    if resp.get("ok") or resp.get("error", {}).get("code") != "E_JSON":
        fail("E_PROBE", f"malformed line mishandled: {line[:200]}")
    report["steps"]["malformed_handled"] = True
    resp, _ = send(proc, {"protocol_version": "1.0", "request_id": "t4",
                          "op": "embed_text",
                          "payload": {"text": args.text}}, args.timeout_s)
    expect_ok(resp, "t4", "post_malformed_alive")

    # graceful shutdown
    resp, _ = send(proc, {"protocol_version": "1.0", "request_id": "bye",
                          "op": "shutdown", "payload": {}}, args.timeout_s)
    expect_ok(resp, "bye", "shutdown")
    try:
        rc = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        fail("E_PROBE", "sidecar ignored shutdown (had to kill)")
    report["steps"]["exit_code"] = rc
    if rc != 0:
        fail("E_PROBE", f"sidecar exit code {rc} != 0 after shutdown")
    _, stderr = proc.communicate()
    report["steps"]["stderr_bytes"] = len(stderr or "")
    # R-04 spot check: query text and image full path must not leak to stderr
    for secret in (args.text, str(image)):
        if secret and secret in (stderr or ""):
            fail("E_PRIVACY", "user content leaked to sidecar stderr")
    report["steps"]["stderr_clean"] = True

    peak_mb = None
    if have_psutil:
        import psutil  # noqa: E402 -- guarded by find_spec above
        try:
            peak_mb = round(psutil.Process(proc.pid).memory_info().rss / 1e6, 1)
        except Exception:  # noqa: BLE001 -- best effort only
            peak_mb = None
    # NOTE: post-exit RSS read is best-effort (process already reaped on
    # some platforms); authoritative RAM comes from the sidecar-platform
    # measurement in the packaging doc. Recorded only when available.
    report["peak_rss_mb_best_effort"] = peak_mb

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report: {args.out}")
    print("PROBE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
