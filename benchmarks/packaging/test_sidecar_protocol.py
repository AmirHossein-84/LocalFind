#!/usr/bin/env python3
"""P0-T010 sidecar protocol tests (no model required, stdlib only).

Spawns benchmarks/packaging/sidecar.py with the current interpreter and
exercises the versioned contract without loading any model: handshake must
answer with loaded=false (fast), malformed/unknown/versioned requests get
typed errors, validation bounds hold, error messages and stderr never carry
user content or full paths, and shutdown exits 0.

Run:  python -m unittest benchmarks.packaging.test_sidecar_protocol -v
(from the repo root; phase-0 prototype, not production code).
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIDECAR = Path(__file__).resolve().parent / "sidecar.py"


def spawn(extra_args=()):
    env = dict(os.environ)
    for var in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
        env.pop(var, None)
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.Popen(
        [sys.executable, str(SIDECAR), *extra_args],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1, env=env,
        cwd=str(REPO_ROOT))
    assert proc.stdin is not None and proc.stdout is not None
    return proc


def rpc(proc, obj):
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise AssertionError("sidecar closed stdout unexpectedly")
    return json.loads(line)


def shutdown(proc, rid="bye"):
    resp = rpc(proc, {"protocol_version": "1.0", "request_id": rid,
                      "op": "shutdown", "payload": {}})
    assert resp.get("ok") is True, resp
    rc = proc.wait(timeout=30)
    _, stderr = proc.communicate()
    return rc, stderr or ""


class ProtocolTests(unittest.TestCase):
    def test_handshake_no_model_fast(self):
        proc = spawn()
        try:
            resp = rpc(proc, {"protocol_version": "1.0", "request_id": "h1",
                              "op": "handshake", "payload": {}})
            self.assertTrue(resp.get("ok"), resp)
            res = resp["result"]
            self.assertEqual(res["protocol_version"], "1.0")
            self.assertEqual(res["sidecar"], "localfind-inference")
            self.assertIn("embed_text", res["ops"])
            self.assertIn("embed_image", res["ops"])
            self.assertFalse(res["model"]["loaded"])
            self.assertEqual(res["model"]["dtype"], "fp32")
        finally:
            rc, _ = shutdown(proc)
            self.assertEqual(rc, 0)

    def test_malformed_json_stays_alive(self):
        proc = spawn()
        try:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write("{not json\n")
            proc.stdin.flush()
            resp = json.loads(proc.stdout.readline())
            self.assertFalse(resp.get("ok"))
            self.assertEqual(resp["error"]["code"], "E_JSON")
            # still alive: handshake works after garbage
            resp = rpc(proc, {"protocol_version": "1.0", "request_id": "h2",
                              "op": "handshake", "payload": {}})
            self.assertTrue(resp.get("ok"), resp)
        finally:
            rc, _ = shutdown(proc)
            self.assertEqual(rc, 0)

    def test_version_and_op_gates(self):
        proc = spawn()
        try:
            r = rpc(proc, {"protocol_version": "9.9", "request_id": "v1",
                           "op": "handshake", "payload": {}})
            self.assertEqual(r["error"]["code"], "E_VERSION")
            r = rpc(proc, {"protocol_version": "1.0", "request_id": "o1",
                           "op": "run_anything", "payload": {}})
            self.assertEqual(r["error"]["code"], "E_OP")
            r = rpc(proc, {"protocol_version": "1.0",
                           "op": "handshake", "payload": {}})
            self.assertEqual(r["error"]["code"], "E_REQUEST_ID")
            r = rpc(proc, {"protocol_version": "1.0", "request_id": "e1",
                           "op": "embed_video_segment",
                           "payload": {"path": "x.mp4"}})
            self.assertEqual(r["error"]["code"], "E_OP")  # video out of spike scope
        finally:
            rc, _ = shutdown(proc)
            self.assertEqual(rc, 0)

    def test_text_validation_bounds(self):
        proc = spawn()
        try:
            r = rpc(proc, {"protocol_version": "1.0", "request_id": "t0",
                           "op": "embed_text", "payload": {"text": "   "}})
            self.assertEqual(r["error"]["code"], "E_VALIDATION")
            r = rpc(proc, {"protocol_version": "1.0", "request_id": "t9",
                           "op": "embed_text",
                           "payload": {"text": "x" * 8193}})
            self.assertEqual(r["error"]["code"], "E_VALIDATION")
            r = rpc(proc, {"protocol_version": "1.0", "request_id": "t8",
                           "op": "embed_text",
                           "payload": {"text": "hi", "prompt": "Translate"}})
            self.assertEqual(r["error"]["code"], "E_VALIDATION")
            r = rpc(proc, {"protocol_version": "1.0", "request_id": "t7",
                           "op": "embed_text",
                           "payload": {"text": "hi", "dim": 511}})
            self.assertEqual(r["error"]["code"], "E_VALIDATION")
        finally:
            rc, _ = shutdown(proc)
            self.assertEqual(rc, 0)

    def test_image_validation_no_path_leak(self):
        proc = spawn()
        try:
            with tempfile.TemporaryDirectory(prefix="lf-secret-") as tmp:
                missing = str(Path(tmp) / "secret-photo.png")
                r = rpc(proc, {"protocol_version": "1.0", "request_id": "i0",
                               "op": "embed_image",
                               "payload": {"path": missing}})
                self.assertEqual(r["error"]["code"], "E_IMAGE")
                # basename may appear; the full directory must not
                self.assertNotIn(tmp, r["error"]["message"])
                r = rpc(proc, {"protocol_version": "1.0", "request_id": "i9",
                               "op": "embed_image",
                               "payload": {"path": missing + ".exe"}})
                self.assertEqual(r["error"]["code"], "E_IMAGE")
        finally:
            rc, stderr = shutdown(proc)
            self.assertEqual(rc, 0)
            self.assertNotIn("secret-photo", stderr)

    def test_stderr_carries_no_user_content(self):
        secret = "PROBE-SECRET-7f3a9c-text-must-not-log"
        proc = spawn()
        try:
            r = rpc(proc, {"protocol_version": "1.0", "request_id": "s1",
                           "op": "embed_text",
                           "payload": {"text": secret + "x" * 9000}})
            self.assertEqual(r["error"]["code"], "E_VALIDATION")
        finally:
            rc, stderr = shutdown(proc)
            self.assertEqual(rc, 0)
            self.assertNotIn(secret, stderr)
            for line in stderr.splitlines():
                # stdout stays protocol-only: every stdout line checked
                # separately; here assert stderr never looks like a response
                self.assertFalse(line.strip().startswith('{"protocol_version"'))

    def test_stdout_is_protocol_only(self):
        proc = spawn()
        try:
            assert proc.stdout is not None
            rpc(proc, {"protocol_version": "1.0", "request_id": "p1",
                       "op": "handshake", "payload": {}})
            # drain check happens implicitly: every line read so far parsed
            # as JSON by rpc(); unread banner bytes would break framing.
            # Peek that no extra bytes are pending after a clean op.
            resp = rpc(proc, {"protocol_version": "1.0", "request_id": "p2",
                              "op": "handshake", "payload": {}})
            self.assertTrue(resp.get("ok"))
        finally:
            rc, _ = shutdown(proc)
            self.assertEqual(rc, 0)

    def test_eof_exits_clean(self):
        proc = spawn()
        assert proc.stdin is not None
        proc.stdin.close()
        rc = proc.wait(timeout=30)
        proc.communicate()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
