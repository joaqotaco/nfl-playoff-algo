import gzip
import json
import sqlite3
import tempfile
import unittest
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

from collector import rawlog, tidy
from collector.fetch import MAX_ATTEMPTS, Fetcher
from collector.rawlog import RoundLog, read_records
from collector.round import (
    EXPECTED_FUTURES,
    TRADE_OVERLAP_SECONDS,
    Session,
    fetch_trades,
    round_outcome,
    run_round,
    slot_of,
)


def fetcher(responses):
    """responses: list of (status, body) or Exception, consumed in order."""
    queue = list(responses)

    def transport(url):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return Fetcher(transport=transport, sleep=lambda _: None)


class FakeLog:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


def session_with(transport):
    return Session(Fetcher(transport=transport, sleep=lambda _: None), FakeLog(), "R1")


class FetcherTests(unittest.TestCase):
    def test_retries_429_then_succeeds(self):
        record, parsed = fetcher([(429, "{}"), (200, '{"a": 1}')]).get("/x")
        self.assertEqual(parsed, {"a": 1})
        self.assertEqual(record["attempts"], 2)
        self.assertEqual(record["errors"], ["HTTP 429"])

    def test_404_is_not_retried(self):
        record, parsed = fetcher([(404, "nope")]).get("/x")
        self.assertIsNone(parsed)
        self.assertEqual(record["attempts"], 1)
        self.assertEqual(record["status"], 404)

    def test_network_failure_gives_up_and_keeps_evidence(self):
        record, parsed = fetcher([OSError("down")] * MAX_ATTEMPTS).get("/x")
        self.assertIsNone(parsed)
        self.assertIsNone(record["status"])
        self.assertEqual(len(record["errors"]), MAX_ATTEMPTS)
        self.assertFalse(record["ok"])

    def test_truncated_200_is_a_failure(self):
        record, parsed = fetcher([(200, "{bad")] * MAX_ATTEMPTS).get("/x")
        self.assertIsNone(parsed)
        self.assertEqual(record["body"], "{bad")


class RawLogTests(unittest.TestCase):
    def test_round_trip_and_crash_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = RoundLog(tmp, "20261101T213000Z")
            log.write({"n": 1})
            log._file.write('{"n": 2, "trunc')  # crash in the middle of a line
            log._file.flush()
            log._file.close()

            self.assertEqual(len(rawlog.recover(tmp)), 1)
            gz = Path(tmp, "2026/11/01/20261101T213000Z.jsonl.gz")
            self.assertEqual(list(read_records(gz)), [{"n": 1}])
            self.assertEqual(list(Path(tmp).rglob("*.jsonl")), [])


class RoundTests(unittest.TestCase):
    def test_pagination_stops_on_repeated_cursor(self):
        session = session_with(lambda url: (200, json.dumps({"trades": [{}], "cursor": "same"})))
        items, complete = session.get_all("trades", "/t", {}, "trades")
        self.assertFalse(complete)
        self.assertEqual(len(items), 2)
        self.assertEqual(session.errors, 1)

    def test_trade_position_advances_only_when_complete(self):
        pages = {"GOOD": (200, '{"trades": [], "cursor": ""}'), "BAD": (404, "x")}

        def transport(url):
            ticker = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["ticker"][0]
            return pages[ticker]

        state = {"trades_min_ts": {"BAD": 5}}
        fetch_trades(session_with(transport), ["GOOD", "BAD"], state, 1000)
        self.assertEqual(state["trades_min_ts"],
                         {"GOOD": 1000 - TRADE_OVERLAP_SECONDS, "BAD": 5})

    def test_failed_event_listing_keeps_known_tickers_and_degrades(self):
        known = [f"T{i}" for i in range(EXPECTED_FUTURES)]

        def transport(url):
            if "/events/" in url:
                return 500, "down"
            if "/orderbooks" in url:
                tickers = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["tickers"]
                return 200, json.dumps({"orderbooks": [
                    {"ticker": t, "orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
                    for t in tickers]})
            if "/markets/trades" in url:
                return 200, '{"trades": [], "cursor": ""}'
            return 200, json.dumps({"markets": [{"ticker": "G1"}], "cursor": ""})

        session = session_with(transport)
        counts = run_round(session, {"futures_tickers": known}, 1000)
        self.assertEqual(counts["futures"], EXPECTED_FUTURES)
        self.assertEqual(counts["missing_books"], 0)
        self.assertEqual(round_outcome(session, counts), "degraded")

    def test_slot_floors_to_half_hour(self):
        ts = datetime(2026, 11, 1, 21, 47, 12, tzinfo=UTC).timestamp()
        self.assertEqual(slot_of(ts), "2026-11-01T21:30:00Z")


class TidyTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.executescript(tidy.SCHEMA)
        self.addCleanup(self.db.close)

    def book(self, ok, body, tickers):
        tidy.ingest_record(self.db, {
            "round_id": "R1", "kind": "books", "tickers": tickers, "url": "u",
            "requested_at": "t0", "received_at": "t1", "status": 200 if ok else 500,
            "attempts": 1, "ok": ok, "errors": [], "body": json.dumps(body),
        })

    def status_of(self, ticker):
        return self.db.execute("SELECT book_status FROM books WHERE ticker = ?",
                               (ticker,)).fetchone()[0]

    def test_book_prices_use_best_level_and_derive_the_ask(self):
        self.book(True, {"orderbooks": [{"ticker": "A", "orderbook_fp": {
            "yes_dollars": [["0.5700", "19.82"], ["0.5800", "13.52"]],
            "no_dollars": [["0.3600", "195.41"], ["0.3700", "26.54"]],
        }}]}, ["A"])
        row = self.db.execute("SELECT yes_bid, yes_bid_size, yes_ask, yes_ask_size FROM books"
                              ).fetchone()
        self.assertEqual(row, (0.58, 13.52, 0.63, 26.54))

    def test_empty_missing_and_failed_books_are_all_distinct_from_ok(self):
        self.book(True, {"orderbooks": [
            {"ticker": "EMPTY", "orderbook_fp": {"yes_dollars": [], "no_dollars": []}},
            {"ticker": "ONE_SIDED", "orderbook_fp": {"no_dollars": [["0.98", "5.00"]]}},
        ]}, ["EMPTY", "ONE_SIDED", "ABSENT"])
        self.book(False, {}, ["FAILED"])
        self.assertEqual(self.status_of("EMPTY"), "empty")
        self.assertEqual(self.status_of("ONE_SIDED"), "ok")
        self.assertEqual(self.status_of("ABSENT"), "missing")
        self.assertEqual(self.status_of("FAILED"), "missing")

    def test_game_market_depth_stays_out_of_book_levels(self):
        side = {"yes_dollars": [["0.50", "1.00"]], "no_dollars": []}
        self.book(True, {"orderbooks": [
            {"ticker": "KXNFLPLAYOFF-27-PIT", "orderbook_fp": side},
            {"ticker": "KXNFLGAME-26NOV09BUFMIN-MIN", "orderbook_fp": side},
        ]}, ["KXNFLPLAYOFF-27-PIT", "KXNFLGAME-26NOV09BUFMIN-MIN"])
        rows = self.db.execute("SELECT DISTINCT ticker FROM book_levels").fetchall()
        self.assertEqual(rows, [("KXNFLPLAYOFF-27-PIT",)])

    def test_overlapping_trade_windows_dedupe(self):
        trade = {"trade_id": "t1", "ticker": "A", "created_time": "t", "yes_price_dollars": "0.60",
                 "count_fp": "2.00", "taker_outcome_side": "no", "taker_book_side": "ask",
                 "is_block_trade": False}
        tidy.ingest_trades(self.db, {"trades": [trade]})
        tidy.ingest_trades(self.db, {"trades": [trade]})
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM trades").fetchone()[0], 1)

    def test_missing_slots_finds_the_hole_and_skips_the_running_slot(self):
        for round_id, slot in (("a", "2026-11-01T20:00:00Z"), ("b", "2026-11-01T21:00:00Z")):
            self.db.execute("INSERT INTO rounds (round_id, slot) VALUES (?, ?)", (round_id, slot))
        now = datetime(2026, 11, 1, 21, 40, tzinfo=UTC)
        self.assertEqual(tidy.missing_slots(self.db, now=now), ["2026-11-01T20:30:00Z"])

    def test_ingest_skips_files_already_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz = Path(tmp, "x.jsonl.gz")
            with gzip.open(gz, "wt") as f:
                f.write(json.dumps({"kind": "round_summary", "round_id": "R", "slot": "s",
                                    "started_at": "a", "finished_at": "b", "outcome": "ok",
                                    "requests": 1, "errors": 0}) + "\n")
            self.assertEqual(tidy.ingest_all(self.db, Path(tmp)), 1)
            self.assertEqual(tidy.ingest_all(self.db, Path(tmp)), 0)


if __name__ == "__main__":
    unittest.main()
