"""How much of the model-market gap is real (k), and how much the playoff markets could absorb.

    python3 -m model.edge          reads paper.sqlite and tidy.sqlite; writes nothing

k: for each daily quote, gap = fair value - market mid at T, and move = mid at a later
horizon - mid at T (a settled market counts at its payout). k is the slope of move on gap,
through the origin: sum(gap x move) / sum(gap^2). k = 1: the market moved all the way to the
model; k = 0: no movement toward it (an efficient market's mid shouldn't move predictably).
Mids are fine here: they estimate what the market believes, not a price we could trade at.
Error bars cluster by team, since the same team is quoted day after day.

Capacity: for each quote window, the contracts that traded THROUGH our price (non-block),
i.e. what a bigger order resting at our price would have filled before anyone else. In
dollars, that's contracts x what each ties up (YES price for a bid, NO price for an ask).
"""
import math
import sqlite3
from collections import defaultdict
from datetime import timedelta

from collector.round import data_dir
from model import data
from model.paper import BIG_GAP, QUOTE_DELAY, Market, horizon_times, iso

HORIZONS = ("24h", "next game day", "latest")


def slope(pairs):
    """pairs: (gap, move, cluster). Returns (k, clustered SE, n, clusters) or None."""
    pairs = [p for p in pairs if p[1] is not None]
    sxx = sum(g * g for g, _, _ in pairs)
    if not pairs or sxx == 0:
        return None
    k = sum(g * m for g, m, _ in pairs) / sxx
    by = defaultdict(float)
    for g, m, c in pairs:
        by[c] += g * (m - k * g)
    clusters = len(by)
    se = math.sqrt(clusters / (clusters - 1) * sum(v * v for v in by.values())) / sxx if clusters > 1 else None
    return k, se, len(pairs), clusters


def mid_at(market, ticker, when):
    snap = market.first_snapshot_after(when)
    if snap is None:
        return None
    row = market.book(snap, ticker)
    if row is None:
        return None
    bid, ask, _, _, result = row
    if result in ("yes", "no"):
        return 1.0 if result == "yes" else 0.0
    return None if bid is None or ask is None else (bid + ask) / 2


def contracts_through(db, ticker, side, price, start, end):
    op = "<" if side == "bid" else ">"
    (n,) = db.execute(
        f"SELECT COALESCE(SUM(count), 0) FROM trades WHERE ticker = ? AND is_block_trade = 0 "
        f"AND created_time >= ? AND created_time < ? AND yes_price {op} ?",
        (ticker, iso(start), iso(end), price)).fetchone()
    return n


def analyse(base):
    tidy = sqlite3.connect(f"file:{base / 'tidy.sqlite'}?mode=ro", uri=True)
    paper = sqlite3.connect(f"file:{base / 'paper.sqlite'}?mode=ro", uri=True)
    market = Market(tidy)
    games = data.load_games(data.latest_nflverse(base))
    kickoffs = sorted({g.kickoff for g in games})
    quotes = paper.execute(
        "SELECT t, team, ticker, side, price, fair, book_bid, book_ask, start, end, status, edge, quote_kind "
        "FROM quotes WHERE book_bid IS NOT NULL AND book_ask IS NOT NULL ORDER BY t").fetchall()

    rows = []
    for t, team, ticker, side, price, fair, bid, ask, start, end, status, edge, kind in quotes:
        t, start, end = data.parse_time(t), data.parse_time(start), data.parse_time(end)
        mid_t = (bid + ask) / 2
        times = horizon_times(t + QUOTE_DELAY, kickoffs)
        times["latest"] = market.snapshots[-1].time
        moves = {}
        for h in HORIZONS:
            when = times.get(h)
            later = mid_at(market, ticker, when) if when and when <= market.snapshots[-1].time else None
            moves[h] = None if later is None else later - mid_t
        seen_until = min(end, market.trades_until)
        through = contracts_through(tidy, ticker, side, price, start, seen_until) if seen_until > start else 0
        cost = price if side == "bid" else 1 - price
        rows.append(dict(day=t.date(), team=team, side=side, price=price, mid=mid_t, gap=fair - mid_t,
                         moves=moves, filled=status == "filled", big=edge is not None and edge > BIG_GAP + 1e-9,
                         kind=kind, through=through, dollars=through * cost))
    return rows


def report(rows, out=print):
    groups = {
        "all quotes": lambda r: r["kind"] == "quote",
        f"  gap <= {BIG_GAP * 100:.0f}c": lambda r: r["kind"] == "quote" and not r["big"],
        f"  gap > {BIG_GAP * 100:.0f}c": lambda r: r["kind"] == "quote" and r["big"],
        "  filled only": lambda r: r["kind"] == "quote" and r["filled"],
        "no-line (skipped)": lambda r: r["kind"] == "no-line",
    }
    out("k = share of the model-market gap the market closed (slope through the origin).")
    out("+- is one SE clustered by team. 'latest' mixes long and short horizons.\n")
    out(f"{'':20s}" + "".join(f"{h:>26s}" for h in HORIZONS))
    for label, keep in groups.items():
        cells = []
        for h in HORIZONS:
            fit = slope([(r["gap"], r["moves"][h], r["team"]) for r in rows if keep(r)])
            if fit is None:
                cells.append("-")
            else:
                k, se, n, c = fit
                cells.append(f"{k:+.2f} +-{se:.2f} (n={n}, {c} tm)" if se is not None else f"{k:+.2f} (n={n})")
        out(f"{label:20s}" + "".join(f"{c:>26s}" for c in cells))

    quoted = [r for r in rows if r["kind"] == "quote"]
    by_day = defaultdict(lambda: [0.0, 0.0])
    for r in quoted:
        by_day[r["day"]][0] += r["through"]
        by_day[r["day"]][1] += r["dollars"]
    out("\nCapacity: contracts that traded through our quote prices (what resting size there would have filled).")
    for day, (n, dollars) in sorted(by_day.items()):
        out(f"  {day}  {n:7.0f} contracts  ${dollars:8.2f}")
    if by_day:
        dollars = sorted(v[1] for v in by_day.values())
        out(f"  per day: mean ${sum(dollars) / len(dollars):.2f}, median ${dollars[len(dollars) // 2]:.2f} "
            f"over {len(dollars)} days")
        top = defaultdict(float)
        for r in quoted:
            top[r["team"]] += r["dollars"]
        out("  biggest teams: " + ", ".join(f"{t} ${d:.0f}" for t, d in sorted(top.items(), key=lambda x: -x[1])[:5]))


def main():
    report(analyse(data_dir()))


if __name__ == "__main__":
    main()
