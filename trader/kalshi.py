"""Signed requests to Kalshi's trading API.

Signing follows docs.kalshi.com/getting_started/quick_start_authenticated_requests:
sign timestamp_ms + METHOD + full path (with /trade-api/v2, without the query string),
RSA-PSS/SHA-256 for RSA keys, the message itself for Ed25519. Logs every request and
response, but never the key or the signature.
"""
import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from collector.fetch import RETRYABLE_STATUS, USER_AGENT, utc_now

BASE_URLS = {
    "demo": "https://external-api.demo.kalshi.co/trade-api/v2",
    "production": "https://external-api.kalshi.com/trade-api/v2",
}
TIMEOUT_SECONDS = 30
GET_ATTEMPTS = 4


class Signer:
    def __init__(self, private_key_pem):
        from cryptography.hazmat.primitives import serialization
        self.key = serialization.load_pem_private_key(private_key_pem, password=None)

    @classmethod
    def from_file(cls, path):
        with open(path, "rb") as f:
            return cls(f.read())

    def sign(self, message):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        if isinstance(self.key, Ed25519PrivateKey):
            signature = self.key.sign(message)
        else:
            signature = self.key.sign(message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                           salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        return base64.b64encode(signature).decode()


def urllib_transport(method, url, headers, body):
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


class Client:
    """One Kalshi account. GETs retry; POST and DELETE never retry on their own, because a
    request that failed in transit may still have reached the exchange."""

    def __init__(self, env, key_id, signer, log, transport=urllib_transport, sleep=time.sleep):
        self.base_url = BASE_URLS[env]
        self.env, self.key_id, self.signer, self.log = env, key_id, signer, log
        self.transport, self.sleep = transport, sleep

    def _send(self, method, path, params=None, body=None):
        query = urllib.parse.urlencode(params or {}, doseq=True)
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        timestamp = str(int(time.time() * 1000))
        sign_path = urllib.parse.urlparse(url).path
        headers = {
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": self.signer.sign(f"{timestamp}{method}{sign_path}".encode()),
        }
        data = json.dumps(body).encode() if body is not None else None
        record = {"kind": "api", "env": self.env, "method": method, "path": path, "params": params,
                  "request": body, "requested_at": utc_now()}
        try:
            status, text = self.transport(method, url, headers, data)
            error = None
        except OSError as exc:
            status, text, error = None, None, repr(exc)
        parsed = None
        if status is not None and 200 <= status < 300:
            try:
                parsed = json.loads(text) if text else {}
            except json.JSONDecodeError:
                error = "invalid JSON"
        self.log.write({**record, "received_at": utc_now(), "status": status, "error": error, "body": text})
        return status, parsed

    def get(self, path, params=None):
        for attempt in range(GET_ATTEMPTS):
            status, parsed = self._send("GET", path, params)
            if parsed is not None or (status is not None and status not in RETRYABLE_STATUS):
                return status, parsed
            self.sleep(2 ** attempt)
        return status, parsed

    def get_all(self, path, params, key):
        items, cursor = [], None
        while True:
            status, page = self.get(path, {**params, **({"cursor": cursor} if cursor else {})})
            if page is None:
                raise RuntimeError(f"GET {path} failed with status {status}")
            items += page.get(key) or []
            cursor = page.get("cursor")
            if not cursor:
                return items

    def post(self, path, body):
        return self._send("POST", path, body=body)

    def delete(self, path, params=None):
        return self._send("DELETE", path, params)
