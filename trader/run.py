"""Place the paper-quote rule's orders on Kalshi, within hard limits.

    python3 -m trader.run                 dry run: print the plan, send nothing
    python3 -m trader.run --live          send the orders to KALSHI_ENV (demo unless set)
    python3 -m trader.run --cancel-all    cancel this bot's resting orders
    python3 -m trader.run --at 2026-11-08T03:30:00Z   use fair values from another time (tests)

Settings come from the environment (see deploy/trader.env.example):
    KALSHI_ENV          demo or production
    KALSHI_KEY_ID       API key id
    KALSHI_KEY_FILE     path to the private key file
    TRADER_ALLOW_PRODUCTION=yes   also required before --live will touch production

What it does, once a day after the fair values:
1. Kill switch: if TRADER_DATA_DIR/STOP exists, cancel this bot's resting orders and stop.
2. Loads today's fair-value band. Refuses to trade if it's missing or over 3 hours old.
3. Reads the real (production) order books now and applies model/paper.py's quote rule,
   even when sending to demo, so demo tests use realistic prices.
4. Reads the account and applies the limits: 1 contract per order, $20 committed in total
   (positions plus resting orders), 10 new orders a day, one order per team and side a day.
   The biggest edges go first.
5. Places post-only limit orders (never cross, never pay a fee) that Kalshi itself cancels
   at the next kickoff or the next day's run.
6. Saves every fill to TRADER_DATA_DIR/fills-<env>.csv, which model/paper.py scores.

Every request and response is logged append-only to TRADER_DATA_DIR/raw/, without keys
or signatures.
"""
import argparse
import csv
import fcntl
import os
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from collector.fetch import USER_AGENT, Fetcher, utc_now
from collector.rawlog import RoundLog
from collector.round import data_dir
from model import data
from model.league import TEAMS, playoff_ticker
from model.paper import FAIRVALUE_SUFFIX, QUOTE_DELAY, quote_price
from trader.kalshi import BASE_URLS, Client, Signer

CONTRACTS = 1
MAX_COMMITTED_DOLLARS = 20.0
MAX_NEW_ORDERS_PER_DAY = 10
MAX_FAIRVALUE_AGE = timedelta(hours=3)
CLIENT_PREFIX = "nflpo"
ORDERS_PATH = "/portfolio/events/orders"


@dataclass
class Order:
    team: str
    ticker: str
    side: str  # "bid" buys YES, "ask" sells YES (the same as buying NO)
    price: float
    edge: float  # how far the band clears the market, in dollars
    client_order_id: str

    @property
    def cost(self):
        """Dollars this order ties up if it fills: YES price for a bid, NO price for an ask."""
        return CONTRACTS * (self.price if self.side == "bid" else 1 - self.price)


def trader_dir():
    path = Path(os.environ.get("TRADER_DATA_DIR", "data/trader"))
    (path / "raw").mkdir(parents=True, exist_ok=True)
    return path


def live_books(fetcher):
    """Best YES bid and ask now, from Kalshi's public production books."""
    tickers = [playoff_ticker(t) for t in TEAMS]
    _, parsed = fetcher.get("/markets/orderbooks", {"tickers": tickers})
    if parsed is None:
        raise RuntimeError("couldn't read the live order books")
    books = {}
    for book in parsed.get("orderbooks") or []:
        levels = book.get("orderbook_fp") or {}
        yes, no = levels.get("yes_dollars") or [], levels.get("no_dollars") or []
        # Levels are sorted ascending, so the best bid on each side is the last entry.
        books[book["ticker"]] = (float(yes[-1][0]) if yes else None,
                                 round(1 - float(no[-1][0]), 4) if no else None)
    return books


def plan(rows, books, day, existing_ids, committed, orders_today, balance):
    """The orders to place now, biggest edge first, within every limit."""
    candidates = []
    for row in rows:
        team = row["team"]
        ticker = playoff_ticker(team)
        low, high = float(row["p_playoffs_low"]), float(row["p_playoffs_high"])
        bid, ask = books.get(ticker, (None, None))
        for side in ("bid", "ask"):
            price = quote_price(side, low, high, bid, ask)
            client_id = f"{CLIENT_PREFIX}-{day}-{team}-{side}"
            if price is None or client_id in existing_ids:
                continue
            edge = low - ask if side == "bid" else bid - high
            candidates.append(Order(team, ticker, side, price, edge, client_id))
    candidates.sort(key=lambda o: -o.edge)

    budget = min(MAX_COMMITTED_DOLLARS - committed, balance)
    slots = MAX_NEW_ORDERS_PER_DAY - orders_today
    chosen = []
    for order in candidates:
        if len(chosen) >= slots:
            break
        if order.cost <= budget:
            chosen.append(order)
            budget -= order.cost
    return chosen, candidates


def committed_dollars(positions, resting):
    """Cost of open positions plus collateral of resting orders, in dollars."""
    held = sum(abs(float(p.get("market_exposure_dollars") or 0)) for p in positions)
    waiting = 0.0
    for o in resting:
        remaining = float(o.get("remaining_count_fp") or 0)
        price = o.get("yes_price_dollars") if o.get("book_side") == "bid" else o.get("no_price_dollars")
        waiting += remaining * float(price or 0)
    return held + waiting


def expiry(at, kickoffs, now):
    """When today's orders should die: the next kickoff, or the next day's run."""
    next_run = at + timedelta(days=1) + QUOTE_DELAY
    upcoming = [k for k in kickoffs if k > now]
    return min(upcoming[:1] + [next_run])


def find_order(client, client_id, since):
    orders = client.get_all("/portfolio/orders", {"min_ts": int(since.timestamp()), "limit": 1000}, "orders")
    return next((o for o in orders if o.get("client_order_id") == client_id), None)


def place(client, order, expires, since):
    """Returns ("placed", order_id), ("rejected", None) or ("failed", None)."""
    body = {
        "ticker": order.ticker,
        "client_order_id": order.client_order_id,
        "side": order.side,
        "count": f"{CONTRACTS:.2f}",
        "price": f"{order.price:.4f}",
        "time_in_force": "good_till_canceled",
        "expiration_time": int(expires.timestamp()),
        "post_only": True,
        "self_trade_prevention_type": "taker_at_cross",
        "cancel_order_on_pause": True,
        "reduce_only": False,
        "exchange_index": -1,
    }
    status, parsed = client.post(ORDERS_PATH, body)
    if parsed is not None:
        return "placed", parsed.get("order_id")
    if status is None or status >= 500 or status == 429:
        # The request may have reached the exchange anyway. Look before anything else,
        # and never resend: a duplicate order is worse than a missed one.
        found = find_order(client, order.client_order_id, since)
        return ("placed", found["order_id"]) if found else ("failed", None)
    return "rejected", None  # e.g. post-only would have crossed, or not enough balance


def cancel_ours(client):
    resting = client.get_all("/portfolio/orders", {"status": "resting", "limit": 1000}, "orders")
    ours = [o for o in resting if (o.get("client_order_id") or "").startswith(CLIENT_PREFIX)]
    for o in ours:
        client.delete(f"{ORDERS_PATH}/{o['order_id']}", {"market_ticker": o["ticker"]})
    return len(ours)


FILL_FIELDS = ["fill_id", "order_id", "created_time", "ticker", "book_side", "outcome_side", "count_fp",
               "yes_price_dollars", "no_price_dollars", "fee_cost", "is_taker"]


def save_fills(client, path):
    fills = [f for f in client.get_all("/portfolio/fills", {"limit": 1000}, "fills")
             if f.get("ticker", "").startswith("KXNFLPLAYOFF-")]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FILL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fills)
    os.replace(tmp, path)
    return len(fills)


def ping(failed):
    url = os.environ.get("TRADER_HEALTHCHECK_URL")
    if not url:
        return
    try:
        req = urllib.request.Request(url.rstrip("/") + ("/fail" if failed else ""), headers={"User-Agent": USER_AGENT})
        urllib.request.urlopen(req, timeout=10).close()
    except OSError as exc:
        print(f"healthcheck ping failed: {exc!r}", file=sys.stderr)


def make_client(env, log):
    key_id, key_file = os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE")
    if not key_id or not key_file:
        return None
    return Client(env, key_id, Signer.from_file(key_file), log)


def run(args, log, tdir, now):
    env = os.environ.get("KALSHI_ENV", "demo")
    if env not in BASE_URLS:
        raise SystemExit(f"KALSHI_ENV must be one of {sorted(BASE_URLS)}")
    if args.live and env == "production" and os.environ.get("TRADER_ALLOW_PRODUCTION") != "yes":
        raise SystemExit("refusing production: set TRADER_ALLOW_PRODUCTION=yes as well")
    client = make_client(env, log)
    if args.live and client is None:
        raise SystemExit("--live needs KALSHI_KEY_ID and KALSHI_KEY_FILE")
    print(f"{'LIVE' if args.live else 'DRY RUN'} on {env}")

    if (tdir / "STOP").exists() or args.cancel_all:
        reason = "kill switch (STOP file)" if (tdir / "STOP").exists() else "--cancel-all"
        cancelled = cancel_ours(client) if client else 0
        print(f"{reason}: cancelled {cancelled} resting orders, placing nothing")
        return {"outcome": "stopped", "cancelled": cancelled}

    base = data_dir()
    at = args.at or data.default_at(now)
    path = base / "fairvalues" / f"{at:%Y%m%dT%H%MZ}{FAIRVALUE_SUFFIX}"
    if not path.exists():
        print(f"no fair values for {at:%Y-%m-%d %H:%M} UTC ({path.name} missing); not trading")
        return {"outcome": "no fair values", "failed": True}
    if now - at > MAX_FAIRVALUE_AGE:
        print(f"fair values from {at:%Y-%m-%d %H:%M} UTC are {(now - at) / timedelta(hours=1):.1f} hours old "
              f"(limit {MAX_FAIRVALUE_AGE / timedelta(hours=1):g}); not trading")
        return {"outcome": "stale fair values", "failed": True}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    books = live_books(Fetcher())

    # The daily 14:00 run is keyed by date; one-off runs by date and time, so they don't
    # use up that day's orders.
    day = f"{at:%Y%m%d}" if (at.hour, at.minute) == (14, 0) else f"{at:%Y%m%dT%H%M}"
    since = at.replace(hour=0, minute=0)
    if client:
        status, bal = client.get("/portfolio/balance")
        if bal is None:
            raise RuntimeError(f"couldn't read the balance (status {status})")
        balance = bal["balance"] / 100
        positions = client.get_all("/portfolio/positions", {"limit": 1000}, "market_positions")
        resting = client.get_all("/portfolio/orders", {"status": "resting", "limit": 1000}, "orders")
        today = client.get_all("/portfolio/orders", {"min_ts": int(since.timestamp()), "limit": 1000}, "orders")
    else:
        balance, positions, resting, today = MAX_COMMITTED_DOLLARS, [], [], []
    existing = {o.get("client_order_id") for o in today if (o.get("client_order_id") or "").startswith(f"{CLIENT_PREFIX}-{day}")}
    committed = committed_dollars(positions, resting)
    chosen, candidates = plan(rows, books, day, existing, committed, len(existing), balance)

    games = data.load_games(data.latest_nflverse(base))
    expires = expiry(at, sorted(g.kickoff for g in games), now)
    log.write({"kind": "plan", "fair_values": path.name, "balance": balance, "committed": committed,
               "already_placed": len(existing), "expires": expires.isoformat(),
               "candidates": [vars(o) for o in candidates], "chosen": [o.client_order_id for o in chosen]})
    if balance <= 0:
        print("account balance is $0: nothing can be placed (add funds; on demo, mock funds)")
    print(f"balance ${balance:.2f}, committed ${committed:.2f}, already placed today {len(existing)}; "
          f"{len(candidates)} candidates, placing {len(chosen)}; orders expire {expires:%Y-%m-%d %H:%M} UTC")
    results = []
    for order in chosen:
        outcome, order_id = place(client, order, expires, since) if args.live else ("dry run", None)
        results.append((order, outcome))
        print(f"  {order.team:4s} {order.side} 1 @ {order.price:.2f} (edge {order.edge * 100:.1f}c, "
              f"ties up ${order.cost:.2f}): {outcome}")
    fills = save_fills(client, tdir / f"fills-{env}.csv") if client else 0
    failed = any(outcome == "failed" for _, outcome in results)
    return {"outcome": "ok" if not failed else "degraded", "failed": failed, "balance": balance,
            "committed": round(committed, 4), "candidates": len(candidates), "chosen": len(chosen),
            "placed": sum(o == "placed" for _, o in results),
            "rejected": sum(o == "rejected" for _, o in results), "fills_saved": fills}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="actually send orders")
    parser.add_argument("--cancel-all", action="store_true")
    parser.add_argument("--at", type=data.parse_time, help="fair-value time, if not today's 14:00 UTC")
    args = parser.parse_args()
    tdir = trader_dir()
    now = datetime.now(UTC)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    with open(tdir / "trader.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = RoundLog(tdir / "raw", run_id)
        summary = {"outcome": "crashed", "failed": True}
        try:
            summary = run(args, log, tdir, now)
        finally:
            log.write({"kind": "run_summary", "run_id": run_id, "live": args.live, "finished_at": utc_now(), **summary})
            log.close()
    ping(summary.get("failed", False))
    sys.exit(1 if summary.get("failed") else 0)


if __name__ == "__main__":
    main()
