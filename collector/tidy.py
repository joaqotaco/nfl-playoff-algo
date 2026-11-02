"""Build the SQLite tables from the raw files.

The raw files are the permanent record. This database can be deleted and rebuilt
at any time: python3 -m collector.tidy        ingest new raw files
           python3 -m collector.tidy --status  show recent health and gaps
"""
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta

from collector.rawlog import read_records
from collector.round import GAME_SERIES, SLOT_SECONDS, data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingested (file TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS rounds (
    round_id TEXT PRIMARY KEY, slot TEXT, started_at TEXT, finished_at TEXT,
    outcome TEXT, requests INTEGER, errors INTEGER, trades INTEGER
);
CREATE TABLE IF NOT EXISTS fetches (
    round_id TEXT, kind TEXT, url TEXT, requested_at TEXT, received_at TEXT,
    status INTEGER, attempts INTEGER, ok INTEGER, errors TEXT
);
CREATE TABLE IF NOT EXISTS markets (
    round_id TEXT, ticker TEXT, event_ticker TEXT, title TEXT, status TEXT, result TEXT,
    close_time TEXT, volume REAL, open_interest REAL,
    PRIMARY KEY (round_id, ticker)
);
-- book_status: ok = has orders, empty = Kalshi returned no orders on either side,
-- missing = we have no book for this ticker in this round.
CREATE TABLE IF NOT EXISTS books (
    round_id TEXT, ticker TEXT, received_at TEXT, book_status TEXT,
    yes_levels INTEGER, no_levels INTEGER,
    yes_bid REAL, yes_bid_size REAL, yes_ask REAL, yes_ask_size REAL,
    PRIMARY KEY (round_id, ticker)
);
-- Full depth for playoff and division markets. Game market depth stays in raw.
CREATE TABLE IF NOT EXISTS book_levels (
    round_id TEXT, ticker TEXT, side TEXT, price REAL, size REAL,
    PRIMARY KEY (round_id, ticker, side, price)
);
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY, ticker TEXT, created_time TEXT, yes_price REAL,
    count REAL, taker_outcome_side TEXT, taker_book_side TEXT, is_block_trade INTEGER
);
CREATE INDEX IF NOT EXISTS books_ticker ON books (ticker, round_id);
CREATE INDEX IF NOT EXISTS trades_ticker ON trades (ticker, created_time);
"""


def number(value):
    return None if value is None else float(value)


def ingest_markets(db, record, parsed):
    for m in parsed.get("markets") or []:
        db.execute(
            "INSERT OR REPLACE INTO markets VALUES (?,?,?,?,?,?,?,?,?)",
            (record["round_id"], m["ticker"], m.get("event_ticker"), m.get("yes_sub_title"),
             m.get("status"), m.get("result"), m.get("close_time"),
             number(m.get("volume_fp")), number(m.get("open_interest_fp"))),
        )


def ingest_books(db, record, parsed):
    books = {b.get("ticker"): b.get("orderbook_fp") or {}
             for b in (parsed or {}).get("orderbooks") or []}
    for ticker in record["tickers"]:
        round_id, received_at = record["round_id"], record["received_at"]
        if ticker not in books:
            db.execute("INSERT OR REPLACE INTO books (round_id, ticker, received_at, book_status) "
                       "VALUES (?,?,?,'missing')", (round_id, ticker, received_at))
            continue
        # Levels are sorted ascending, so the best bid is the last entry.
        yes = books[ticker].get("yes_dollars") or []
        no = books[ticker].get("no_dollars") or []
        yes_bid, yes_bid_size = (float(x) for x in yes[-1]) if yes else (None, None)
        no_bid, no_bid_size = (float(x) for x in no[-1]) if no else (None, None)
        db.execute(
            "INSERT OR REPLACE INTO books VALUES (?,?,?,?,?,?,?,?,?,?)",
            (round_id, ticker, received_at, "ok" if yes or no else "empty", len(yes), len(no),
             yes_bid, yes_bid_size,
             None if no_bid is None else round(1 - no_bid, 6), no_bid_size),
        )
        if not ticker.startswith(GAME_SERIES):
            for side, levels in (("yes", yes), ("no", no)):
                db.executemany(
                    "INSERT OR REPLACE INTO book_levels VALUES (?,?,?,?,?)",
                    [(round_id, ticker, side, float(p), float(s)) for p, s in levels],
                )


def ingest_trades(db, parsed):
    db.executemany(
        "INSERT OR IGNORE INTO trades VALUES (?,?,?,?,?,?,?,?)",
        [(t["trade_id"], t["ticker"], t["created_time"], float(t["yes_price_dollars"]),
          float(t["count_fp"]), t.get("taker_outcome_side"), t.get("taker_book_side"),
          int(t.get("is_block_trade", False)))
         for t in parsed.get("trades") or []],
    )


def ingest_record(db, record):
    kind = record["kind"]
    if kind == "error":
        return
    if kind == "round_summary":
        db.execute(
            "INSERT OR REPLACE INTO rounds VALUES (?,?,?,?,?,?,?,?)",
            (record["round_id"], record.get("slot"), record["started_at"], record["finished_at"],
             record["outcome"], record["requests"], record["errors"], record.get("trades")),
        )
        return

    db.execute(
        "INSERT INTO fetches VALUES (?,?,?,?,?,?,?,?,?)",
        (record["round_id"], kind, record["url"], record["requested_at"], record["received_at"],
         record["status"], record["attempts"], int(record["ok"]), json.dumps(record["errors"])),
    )
    parsed = json.loads(record["body"]) if record["ok"] else None
    if kind == "books":
        ingest_books(db, record, parsed)
    elif parsed is None:
        return
    elif kind in ("event", "markets"):
        ingest_markets(db, record, parsed)
    elif kind in ("trades", "trades_historical"):
        ingest_trades(db, parsed)


def ingest_all(db, raw_dir):
    done = {row[0] for row in db.execute("SELECT file FROM ingested")}
    count = 0
    for path in sorted(raw_dir.rglob("*.jsonl.gz")):
        name = str(path.relative_to(raw_dir))
        if name in done:
            continue
        with db:  # one transaction per file, so a failure can't half-ingest a round
            for record in read_records(path):
                ingest_record(db, record)
            db.execute("INSERT INTO ingested VALUES (?)", (name,))
        count += 1
    return count


def missing_slots(db, now=None, days=7):
    """Half-hour slots since the first round with no finished round recorded."""
    slots = {row[0] for row in db.execute("SELECT slot FROM rounds WHERE slot IS NOT NULL")}
    if not slots:
        return []
    now = now or datetime.now(UTC)
    first = datetime.strptime(min(slots), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    step = timedelta(seconds=SLOT_SECONDS)
    cursor = max(first, now - timedelta(days=days))
    cursor = first + ((cursor - first) // step) * step
    gaps = []
    # The current slot may still be running, so stop one short of it.
    while cursor + step <= now:
        slot = cursor.strftime("%Y-%m-%dT%H:%M:%SZ")
        if slot not in slots:
            gaps.append(slot)
        cursor += step
    return gaps


def print_status(db):
    print("Latest rounds:")
    for row in db.execute("SELECT round_id, outcome, requests, errors, trades FROM rounds "
                          "ORDER BY started_at DESC LIMIT 5"):
        print("  ", *row)
    print("Rounds by outcome:", dict(db.execute(
        "SELECT outcome, COUNT(*) FROM rounds GROUP BY outcome").fetchall()))
    print("Missing books:", db.execute(
        "SELECT COUNT(*) FROM books WHERE book_status = 'missing'").fetchone()[0])
    print("Trades stored:", db.execute("SELECT COUNT(*) FROM trades").fetchone()[0])
    print("Missing slots, last 7 days:", missing_slots(db) or "none")


def main():
    base = data_dir()
    db = sqlite3.connect(base / "tidy.sqlite")
    db.executescript(SCHEMA)
    count = ingest_all(db, base / "raw")
    if "--status" in sys.argv:
        print_status(db)
    else:
        print(f"ingested {count} raw files")
    db.close()


if __name__ == "__main__":
    main()
