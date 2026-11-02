"""One collection round: list markets, snapshot books, fetch new trades.

Run by a timer every 30 minutes: python3 -m collector.round
"""
import fcntl
import json
import os
import sys
import time
import traceback
import urllib.request
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from collector.fetch import USER_AGENT, Fetcher, utc_now
from collector.rawlog import RoundLog, recover

PLAYOFF_EVENT = "KXNFLPLAYOFF-27"
DIVISION_EVENTS = [
    f"KXNFL{conference}{division}-27"
    for conference in ("AFC", "NFC")
    for division in ("EAST", "NORTH", "SOUTH", "WEST")
]
# 25 spread markets per game. Game markets get book snapshots only: their trades
# are heavy, not needed for paper fills, and can be fetched later if ever wanted.
GAME_SERIES = ("KXNFLGAME", "KXNFLSPREAD")
EXPECTED_FUTURES = 64

BOOK_BATCH_SIZE = 100
PAGE_LIMIT = 1000
SLOT_SECONDS = 1800
# min_ts has one-second resolution, so re-read a window and dedupe on trade_id.
TRADE_OVERLAP_SECONDS = 600


class Session:
    """Fetches and logs together, so nothing reaches the code unlogged."""

    def __init__(self, fetcher, log, round_id):
        self.fetcher = fetcher
        self.log = log
        self.round_id = round_id
        self.requests = 0
        self.errors = 0

    def get(self, kind, path, params=None, **extra):
        record, parsed = self.fetcher.get(path, params)
        self.log.write({"round_id": self.round_id, "kind": kind, **extra, **record})
        self.requests += 1
        self.errors += parsed is None
        return parsed

    def get_all(self, kind, path, params, key, **extra):
        """Follow cursors. Returns (items, complete)."""
        items, cursor, seen = [], None, set()
        while True:
            page_params = {**params, "cursor": cursor} if cursor else params
            page = self.get(kind, path, page_params, **extra)
            if page is None:
                return items, False
            items += page.get(key) or []
            cursor = page.get("cursor")
            if not cursor:
                return items, True
            if cursor in seen:
                self.errors += 1
                self.log.write({"round_id": self.round_id, "kind": "error",
                                "error": f"repeated cursor on {path} {params}"})
                return items, False
            seen.add(cursor)


def list_futures(session, state):
    """Playoff and division tickers. No status filter, so clinched teams stay listed."""
    tickers = set(state.get("futures_tickers", []))
    for event in [PLAYOFF_EVENT, *DIVISION_EVENTS]:
        data = session.get("event", f"/events/{event}")
        if data is not None:
            tickers.update(m["ticker"] for m in data.get("markets") or [])
    return sorted(tickers)


def list_games(session):
    tickers = set()
    for series in GAME_SERIES:
        markets, _ = session.get_all(
            "markets", "/markets",
            {"series_ticker": series, "status": "open", "limit": PAGE_LIMIT}, "markets",
        )
        tickers.update(m["ticker"] for m in markets)
    return sorted(tickers)


def snapshot_books(session, tickers):
    """Returns how many tickers came back without a book."""
    missing = 0
    for start in range(0, len(tickers), BOOK_BATCH_SIZE):
        batch = tickers[start:start + BOOK_BATCH_SIZE]
        data = session.get("books", "/markets/orderbooks", {"tickers": batch}, tickers=batch)
        returned = {b.get("ticker") for b in (data or {}).get("orderbooks") or []}
        missing += len(set(batch) - returned)
    return missing


def fetch_trades(session, tickers, state, started_ts):
    """A ticker with no saved position fetches all the history the live API holds."""
    positions = state.setdefault("trades_min_ts", {})
    count = 0
    for ticker in tickers:
        params = {"ticker": ticker, "limit": PAGE_LIMIT}
        if ticker in positions:
            params["min_ts"] = positions[ticker]
        trades, complete = session.get_all(
            "trades", "/markets/trades", params, "trades", ticker=ticker,
        )
        count += len(trades)
        if complete:
            positions[ticker] = started_ts - TRADE_OVERLAP_SECONDS
    return count


def run_round(session, state, started_ts):
    futures = list_futures(session, state)
    games = list_games(session)
    # Books go first and together, so the snapshot is as close to one instant as we can get.
    missing_books = snapshot_books(session, futures + games)
    trades = fetch_trades(session, futures, state, started_ts)
    state["futures_tickers"] = futures
    return {
        "futures": len(futures),
        "games": len(games),
        "missing_books": missing_books,
        "trades": trades,
    }


def slot_of(ts):
    floored = int(ts) - int(ts) % SLOT_SECONDS
    return datetime.fromtimestamp(floored, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def round_outcome(session, counts):
    healthy = (
        session.errors == 0
        and counts["futures"] == EXPECTED_FUTURES
        and counts["games"] > 0
        and counts["missing_books"] == 0
    )
    return "ok" if healthy else "degraded"


def load_state(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def save_state(path, state):
    tmp = f"{path}.tmp"
    Path(tmp).write_text(json.dumps(state))
    os.replace(tmp, path)


@contextmanager
def run_lock(data_dir):
    with open(Path(data_dir) / "collector.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def ping_healthcheck(outcome):
    """Tell the outside monitor how the round went. Never let this fail a round."""
    url = os.environ.get("HEALTHCHECK_URL")
    if not url:
        return
    if outcome != "ok":
        url = url.rstrip("/") + "/fail"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        urllib.request.urlopen(request, timeout=10).close()
    except OSError as exc:
        print(f"healthcheck ping failed: {exc!r}", file=sys.stderr)


def data_dir():
    path = Path(os.environ.get("COLLECTOR_DATA_DIR", "data"))
    (path / "raw").mkdir(parents=True, exist_ok=True)
    return path


def main():
    base = data_dir()
    started_ts = time.time()
    round_id = datetime.fromtimestamp(started_ts, UTC).strftime("%Y%m%dT%H%M%SZ")
    started_at = utc_now()

    with run_lock(base):
        recovered = recover(base / "raw")
        state = load_state(base / "state.json")
        log = RoundLog(base / "raw", round_id)
        session = Session(Fetcher(), log, round_id)
        counts, outcome, crash = {}, "crashed", None
        try:
            counts = run_round(session, state, int(started_ts))
            outcome = round_outcome(session, counts)
        except Exception:
            crash = traceback.format_exc()
        log.write({
            "round_id": round_id,
            "kind": "round_summary",
            "slot": slot_of(started_ts),
            "started_at": started_at,
            "finished_at": utc_now(),
            "outcome": outcome,
            "requests": session.requests,
            "errors": session.errors,
            "recovered_files": len(recovered),
            "crash": crash,
            **counts,
        })
        log.close()
        if crash is None:
            save_state(base / "state.json", state)

    ping_healthcheck(outcome)
    print(f"{round_id} {outcome} requests={session.requests} errors={session.errors} {counts}")
    if crash:
        print(crash, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
