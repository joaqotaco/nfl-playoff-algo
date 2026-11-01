"""HTTP GETs against Kalshi's public market data, with retries.

Every call returns a record describing what happened, including failures, so
that a failed request leaves evidence in the raw log.
"""
import http.client
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
USER_AGENT = "nfl-playoff-algo-collector/0.1"
TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 5
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
# Kalshi documents limits for authenticated requests only, so stay well below them.
REQUEST_PAUSE_SECONDS = 0.2


def utc_now():
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def urllib_transport(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


class Fetcher:
    def __init__(self, transport=urllib_transport, sleep=time.sleep, base_url=BASE_URL):
        self.transport = transport
        self.sleep = sleep
        self.base_url = base_url

    def get(self, path, params=None):
        """Return (record, parsed). parsed is None unless the call fully succeeded."""
        query = urllib.parse.urlencode(params or {}, doseq=True)
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        requested_at = utc_now()
        status, body, parsed, errors = None, None, None, []

        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.sleep(REQUEST_PAUSE_SECONDS)
            error = None
            try:
                status, body = self.transport(url)
            except (OSError, http.client.HTTPException) as exc:
                status, body, error = None, None, repr(exc)

            if status == 200:
                try:
                    parsed = json.loads(body)
                    break
                except json.JSONDecodeError:
                    error = "HTTP 200 with invalid JSON"
            elif status is not None:
                error = f"HTTP {status}"

            errors.append(error)
            if status is not None and status != 200 and status not in RETRYABLE_STATUS:
                break
            if attempt < MAX_ATTEMPTS:
                self.sleep(2 ** (attempt - 1) + random.random())

        record = {
            "url": url,
            "requested_at": requested_at,
            "received_at": utc_now(),
            "status": status,
            "attempts": attempt,
            "ok": parsed is not None,
            "errors": errors,
            "body": body,
        }
        return record, parsed
