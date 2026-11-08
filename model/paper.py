"""Paper quotes on the playoff contracts, their fills, and markouts.

    python3 -m model.paper            rebuild data/paper.sqlite from saved data, print a report
    python3 -m model.paper --quiet    rebuild only (the daily job)

Rules agreed 2026-11-06 (see CLAUDE.md), applied to each daily fair-value file at T:
- Quote only when the band excludes the market: a bid if band low > best ask, an ask if
  band high < best bid. One contract.
- Price one tick inside the market (best bid + 1c, or best ask - 1c), never crossing the
  spread and never past the band edge minus a cushion (maker fee + 1c).
- The quote lives from T + 10 min (when the fair value exists) until the next kickoff or
  the next day's quote, whichever is first.
- Filled only if a non-block trade goes THROUGH the price (below our bid, above our ask).
- Taker comparison: at the first snapshot after T + 10 min, cross the spread if the band
  still beats the price after the taker fee.
- Markouts: mid and exit (sell at bid / buy at ask, after taker fee) at the first snapshot
  1h and 24h after entry, at 14:00 UTC the day after the next game day, at the latest
  snapshot, and at settlement. A market that has settled is marked at its payout.
- Error bars treat each game day as one unit: fills before the same games share one
  outcome, so counting them as independent would overstate certainty.
- The report splits results by whether a trade agrees with the favorite-longshot bias:
  buying a favorite (price >= 0.50) or selling a longshot (price < 0.50).

Real fills from the trader (TRADER_DATA_DIR/fills-production.csv), if any, are scored the
same way as "real", so paper and real trading can be compared directly.

Everything is recomputed from the fair-value CSVs, the collector's tidy.sqlite and the
nflverse schedule, so the results can always be reproduced.
"""
import argparse
import csv
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from collector.round import data_dir
from model import data
from model.league import playoff_ticker

FAIRVALUE_SUFFIX = "-calibrated-hl2-k.csv"
TICK = 0.01
# No maker fee on KXNFLPLAYOFF: Kalshi's fee schedule (effective 2026-07-07) sets the maker
# multiplier to 0 for series not listed under Non-Standard Fees, and this one isn't.
MAKER_FEE = 0.0
CUSHION = MAKER_FEE + TICK
QUOTE_DELAY = timedelta(minutes=10)
HORIZONS = ("1h", "24h", "next game day", "latest", "settlement")

SCHEMA = """
CREATE TABLE quotes (
    id INTEGER PRIMARY KEY, t TEXT, team TEXT, ticker TEXT, side TEXT, price REAL,
    fair REAL, band_low REAL, band_high REAL, book_bid REAL, book_ask REAL,
    start TEXT, end TEXT, status TEXT, fill_time TEXT, fill_trade_price REAL
);
CREATE TABLE takers (
    id INTEGER PRIMARY KEY, t TEXT, team TEXT, ticker TEXT, side TEXT, entry_time TEXT,
    price REAL, fee REAL, fair REAL, band_low REAL, band_high REAL
);
CREATE TABLE reals (
    id INTEGER PRIMARY KEY, fill_id TEXT, created_time TEXT, ticker TEXT, side TEXT,
    price REAL, count REAL, fee REAL
);
CREATE TABLE markouts (
    kind TEXT, ref INTEGER, horizon TEXT, mark_time TEXT, mid REAL, exit REAL,
    pnl_mid REAL, pnl_exit REAL,
    side TEXT, price REAL,
    game_day TEXT  -- ET date of the first game after entry: the unit for error bars
);
"""


def taker_fee(price):
    """Kalshi taker fee for one contract: round_up(0.07 * P * (1 - P)) to the cent."""
    return math.ceil(round(0.07 * price * (1 - price) * 100, 9)) / 100


def cents_down(x):
    return math.floor(round(x * 100, 9)) / 100


def cents_up(x):
    return math.ceil(round(x * 100, 9)) / 100


def quote_price(side, band_low, band_high, bid, ask):
    """The paper quote's price, or None when the band doesn't exclude the market."""
    if side == "bid":
        if ask is None or band_low <= ask:
            return None
        price = min((bid or 0.0) + TICK, ask - TICK, cents_down(band_low - CUSHION))
        return round(price, 2) if price >= TICK else None
    if bid is None or band_high >= bid:
        return None
    price = max((ask if ask is not None else 1.0) - TICK, bid + TICK, cents_up(band_high + CUSHION))
    return round(price, 2) if price <= 1 - TICK else None


@dataclass
class Snapshot:
    time: datetime
    round_id: str


class Market:
    """Reads books, trades and settlement status from the collector's tidy database."""

    def __init__(self, db):
        self.db = db
        self.snapshots = [Snapshot(data.parse_time(started), rid) for rid, started in db.execute(
            "SELECT round_id, started_at FROM rounds WHERE slot IS NOT NULL AND outcome != 'crashed' "
            "ORDER BY started_at")]
        # Trades are complete up to the start of the latest round.
        self.trades_until = self.snapshots[-1].time if self.snapshots else None

    def first_snapshot_after(self, t):
        return next((s for s in self.snapshots if s.time >= t), None)

    def book(self, snapshot, ticker):
        row = self.db.execute(
            "SELECT b.yes_bid, b.yes_ask, b.book_status, m.status, m.result FROM books b "
            "LEFT JOIN markets m USING (round_id, ticker) WHERE b.round_id = ? AND b.ticker = ?",
            (snapshot.round_id, ticker)).fetchone()
        return row  # (bid, ask, book_status, market_status, result) or None

    def first_trade_through(self, ticker, side, price, start, end):
        op = "<" if side == "bid" else ">"
        row = self.db.execute(
            f"SELECT created_time, yes_price FROM trades WHERE ticker = ? AND is_block_trade = 0 "
            f"AND created_time >= ? AND created_time < ? AND yes_price {op} ? ORDER BY created_time LIMIT 1",
            (ticker, iso(start), iso(end), price)).fetchone()
        return row


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def mark(market, ticker, side, entry_time, entry_price, entry_fee, horizon_time):
    """Mid and exit markouts per contract at the first snapshot at or after horizon_time.

    Returns (mark_time, mid, exit, pnl_mid, pnl_exit, settled) or None.
    """
    snap = market.first_snapshot_after(horizon_time)
    if snap is None:
        return None
    row = market.book(snap, ticker)
    if row is None:
        return None
    bid, ask, book_status, status, result = row
    direction = 1 if side == "bid" else -1  # long YES after a bid fills, short after an ask
    settled = result in ("yes", "no")
    if settled:
        value = 1.0 if result == "yes" else 0.0
        mid = exit_price = value
        exit_fee = 0.0
    else:
        if bid is None or ask is None:
            return snap.time, None, None, None, None, False
        mid = (bid + ask) / 2
        exit_price = bid if direction == 1 else ask
        exit_fee = taker_fee(exit_price)
    pnl_mid = direction * (mid - entry_price) - entry_fee
    pnl_exit = direction * (exit_price - entry_price) - entry_fee - exit_fee
    return snap.time, mid, exit_price, pnl_mid, pnl_exit, settled


def horizon_times(entry, kickoffs):
    """1h, 24h, and 14:00 UTC on the day after the next game day (ET dates)."""
    nxt = next((k for k in kickoffs if k > entry), None)
    out = {"1h": entry + timedelta(hours=1), "24h": entry + timedelta(hours=24)}
    if nxt is not None:
        game_day = nxt.astimezone(data.EASTERN).date()
        out["next game day"] = datetime.combine(game_day + timedelta(days=1), datetime.min.time(),
                                                tzinfo=data.UTC).replace(hour=14)
    return out


def load_fairvalues(folder):
    files = sorted(Path(folder).glob(f"*{FAIRVALUE_SUFFIX}"))
    out = []
    for path in files:
        t = datetime.strptime(path.name[:14], "%Y%m%dT%H%MZ").replace(tzinfo=data.UTC)
        if (t.hour, t.minute) != (14, 0):
            continue  # the daily rule uses 14:00 UTC; other times are one-off tests
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and "p_playoffs_low" in rows[0]:
            out.append((t, rows))
    return out


def number(x):
    return float(x) if x not in (None, "", "None") else None


def build(base, out_path):
    db = sqlite3.connect(base / "tidy.sqlite")
    market = Market(db)
    games = data.load_games(data.latest_nflverse(base))
    kickoffs = sorted({g.kickoff for g in games})
    fairvalues = load_fairvalues(base / "fairvalues")

    out = sqlite3.connect(out_path)
    out.executescript("DROP TABLE IF EXISTS quotes; DROP TABLE IF EXISTS takers; DROP TABLE IF EXISTS reals; "
                      "DROP TABLE IF EXISTS markouts;" + SCHEMA)
    next_t = {t: (fairvalues[i + 1][0] if i + 1 < len(fairvalues) else t + timedelta(days=1))
              for i, (t, _) in enumerate(fairvalues)}

    for t, rows in fairvalues:
        start = t + QUOTE_DELAY
        end = min([k for k in kickoffs if k > start][:1] + [next_t[t] + QUOTE_DELAY])
        taker_snap = market.first_snapshot_after(start)
        for row in rows:
            team, ticker = row["team"], playoff_ticker(row["team"])
            fair, low, high = number(row["p_playoffs"]), number(row["p_playoffs_low"]), number(row["p_playoffs_high"])
            bid, ask = number(row["playoff_bid"]), number(row["playoff_ask"])
            for side in ("bid", "ask"):
                price = quote_price(side, low, high, bid, ask)
                if price is not None:
                    # Only trades we've collected count; a window still running stays open.
                    seen_until = min(end, market.trades_until) if market.trades_until else start
                    fill = market.first_trade_through(ticker, side, price, start, seen_until)
                    status = "filled" if fill else "unfilled" if seen_until >= end else "open"
                    cur = out.execute(
                        "INSERT INTO quotes (t, team, ticker, side, price, fair, band_low, band_high, book_bid, "
                        "book_ask, start, end, status, fill_time, fill_trade_price) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (iso(t), team, ticker, side, price, fair, low, high, bid, ask, iso(start), iso(end), status,
                         fill[0] if fill else None, fill[1] if fill else None))
                    if fill:
                        entry = data.parse_time(fill[0])
                        add_markouts(out, market, "quote", cur.lastrowid, ticker, side, entry, price, MAKER_FEE, kickoffs)

            # Taker comparison, at the first snapshot after the fair value exists.
            if taker_snap is None:
                continue
            book = market.book(taker_snap, ticker)
            if book is None:
                continue
            now_bid, now_ask = book[0], book[1]
            trades = []
            if now_ask is not None and low > now_ask + taker_fee(now_ask):
                trades.append(("bid", now_ask))  # buy YES at the ask
            if now_bid is not None and high < now_bid - taker_fee(now_bid):
                trades.append(("ask", now_bid))  # sell YES at the bid
            for side, price in trades:
                fee = taker_fee(price)
                cur = out.execute(
                    "INSERT INTO takers (t, team, ticker, side, entry_time, price, fee, fair, band_low, band_high) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (iso(t), team, ticker, side, iso(taker_snap.time), price, fee, fair, low, high))
                add_markouts(out, market, "taker", cur.lastrowid, ticker, side, taker_snap.time, price, fee, kickoffs)

    # Real fills: a bid fill means we bought YES, an ask fill that we sold it.
    fills = Path(os.environ.get("TRADER_DATA_DIR", base / "trader")) / "fills-production.csv"
    if fills.exists():
        with open(fills, newline="") as f:
            for fill in csv.DictReader(f):
                count = float(fill["count_fp"])
                price, fee = float(fill["yes_price_dollars"]), float(fill["fee_cost"] or 0) / count
                entry = data.parse_time(fill["created_time"])
                cur = out.execute("INSERT INTO reals (fill_id, created_time, ticker, side, price, count, fee) "
                                  "VALUES (?,?,?,?,?,?,?)",
                                  (fill["fill_id"], fill["created_time"], fill["ticker"], fill["book_side"], price, count, fee))
                add_markouts(out, market, "real", cur.lastrowid, fill["ticker"], fill["book_side"], entry, price, fee, kickoffs)
    out.commit()
    return out


def add_markouts(out, market, kind, ref, ticker, side, entry, price, fee, kickoffs):
    nxt = next((k for k in kickoffs if k > entry), entry)
    game_day = nxt.astimezone(data.EASTERN).date().isoformat()
    times = horizon_times(entry, kickoffs)
    if market.snapshots and market.snapshots[-1].time > entry:
        times["latest"] = market.snapshots[-1].time
    for horizon, when in times.items():
        result = mark(market, ticker, side, entry, price, fee, when)
        if result is None:
            continue
        mark_time, mid, exit_price, pnl_mid, pnl_exit, settled = result
        rows = [horizon] + (["settlement"] if horizon == "latest" and settled else [])
        for name in rows:
            out.execute("INSERT INTO markouts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (kind, ref, name, iso(mark_time), mid, exit_price, pnl_mid, pnl_exit, side, price, game_day))


def clustered_se(values, clusters):
    """Standard error of the mean when results within a cluster share one outcome.

    Sums deviations within each cluster first, so ten fills before the same games
    count as roughly one piece of evidence, not ten.
    """
    groups = sorted(set(clusters))
    if len(groups) < 2:
        return None
    mean = sum(values) / len(values)
    sums = {g: 0.0 for g in groups}
    for v, c in zip(values, clusters):
        sums[c] += v - mean
    g = len(groups)
    return math.sqrt(g / (g - 1) * sum(x * x for x in sums.values())) / len(values)


def summary(rows):
    """rows: (value, cluster). Mean in cents with game-day error bars."""
    if not rows:
        return "-"
    values = [v for v, _ in rows]
    clusters = [c for _, c in rows]
    mean = sum(values) / len(values) * 100
    se = clustered_se(values, clusters)
    error = f"+-{se * 100:.1f}" if se is not None else "+-?"
    return f"{mean:+5.1f}c {error} (n={len(values)}, {len(set(clusters))} game days)"


def agrees_with_longshot_bias(side, price):
    """Favorites tend to be underpriced and longshots overpriced, so the bias favors
    buying favorites and selling longshots."""
    return (side == "bid" and price >= 0.5) or (side == "ask" and price < 0.5)


def report(out):
    print("Quotes by status:", dict(out.execute("SELECT status, COUNT(*) FROM quotes GROUP BY status").fetchall()))
    print("Taker comparisons:", out.execute("SELECT COUNT(*) FROM takers").fetchone()[0])
    print("\nMarkouts per contract, after fees (mean +- SE by game day).")
    print("Mid = did the price move our way; exit = closing out by crossing the spread.")
    for kind in ("quote", "taker", "real"):
        for horizon in HORIZONS:
            rows = out.execute("SELECT pnl_mid, pnl_exit, game_day FROM markouts WHERE kind = ? AND horizon = ? "
                               "AND pnl_mid IS NOT NULL", (kind, horizon)).fetchall()
            print(f"  {kind:5s} {horizon:14s} mid {summary([(r[0], r[2]) for r in rows]):36s} "
                  f"exit {summary([(r[1], r[2]) for r in rows])}")
    print("\nBy favorite-longshot bias (mid markouts): agrees = buying a favorite or selling a longshot.")
    for kind in ("quote", "taker", "real"):
        for horizon in HORIZONS:
            rows = out.execute("SELECT pnl_mid, game_day, side, price FROM markouts WHERE kind = ? AND horizon = ? "
                               "AND pnl_mid IS NOT NULL", (kind, horizon)).fetchall()
            agree = [(r[0], r[1]) for r in rows if agrees_with_longshot_bias(r[2], r[3])]
            against = [(r[0], r[1]) for r in rows if not agrees_with_longshot_bias(r[2], r[3])]
            print(f"  {kind:5s} {horizon:14s} agrees {summary(agree):36s} against {summary(against)}")
    print("\nLatest quotes:")
    for row in out.execute("SELECT t, team, side, price, round(fair, 3), round(band_low, 2), round(band_high, 2), "
                           "book_bid, book_ask, status FROM quotes ORDER BY t DESC, team LIMIT 15"):
        print("  ", *row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    base = data_dir()
    out = build(base, base / "paper.sqlite")
    if not args.quiet:
        report(out)
    out.close()


if __name__ == "__main__":
    main()
