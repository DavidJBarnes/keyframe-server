#!/usr/bin/env python3
"""Auth proxy for the keyframe edit server.

Adapted from runpod-scripts/auth-proxy.py with three changes that matter here:

  * ThreadingHTTPServer, not HTTPServer. An edit takes 45-200s and the original
    is single-threaded, so /health would block behind every in-flight edit and
    RunPod's probes would time out mid-request.
  * /health proxies through to the real server instead of returning a canned
    200. Model load takes minutes; a hardcoded OK reports "ready" while the
    pipeline is still loading, which is worse than no health check at all.
  * Longer upstream timeout — 40-step edits measured 192s on a 3090.
"""
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPConnection

API_KEY = os.environ.get("API_KEY", "")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8888"))
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8189"))
UPSTREAM_TIMEOUT = int(os.environ.get("UPSTREAM_TIMEOUT", "900"))
CHUNK_SIZE = 65536


class AuthProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _check_auth(self):
        if not API_KEY:
            return True
        if self.headers.get("Authorization", "") == f"Bearer {API_KEY}":
            return True
        return self.headers.get("X-API-Key", "") == API_KEY

    def _json(self, code, payload: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _proxy_request(self):
        # /health is unauthenticated so RunPod can probe it, but it is proxied
        # rather than faked: while the model is still loading the upstream is not
        # listening, and reporting that honestly is the whole point.
        if not self._check_auth() and self.path != "/health":
            self._json(401, b'{"error": "Unauthorized. Send Authorization: Bearer <key> '
                            b'or X-API-Key: <key>."}')
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length > 0 else None

        conn = HTTPConnection("127.0.0.1", SERVER_PORT, timeout=UPSTREAM_TIMEOUT)
        try:
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "authorization", "x-api-key")}
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except ConnectionRefusedError:
            self._json(503, b'{"error": "model server not up yet - still loading weights"}')
        except Exception as e:
            self._json(502, f'{{"error": "proxy error: {type(e).__name__}"}}'.encode())
        finally:
            conn.close()

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _proxy_request

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print(f"auth proxy: 0.0.0.0:{PROXY_PORT} -> 127.0.0.1:{SERVER_PORT}", flush=True)
    print("API key auth ENABLED" if API_KEY else
          "WARNING: API_KEY unset — endpoint is OPEN to anyone who finds the pod URL",
          flush=True)
    ThreadingHTTPServer(("0.0.0.0", PROXY_PORT), AuthProxyHandler).serve_forever()
