"""Daily fair values for the playoff and division contracts, next to Kalshi's book at T.

    python3 -m model.fairvalue                          at the latest 14:00 UTC
    python3 -m model.fairvalue --at 2026-11-05T04:30:00Z --sims 20000
    python3 -m model.fairvalue --half-life 2 --ratings-use-kalshi --uncertainty measured

Upcoming games with a Kalshi line always use that line directly. --ratings-use-kalshi
also feeds those lines into the ratings, so lasting news (a season-ending injury)
reaches later weeks too.
"""
import argparse
import csv
import sqlite3
import statistics

import numpy as np

from collector.round import data_dir
from model import data
from model.league import CONFERENCE_OF, TEAMS, division_ticker, playoff_ticker
from model.ratings import DEFAULT_HALF_LIFE, fit, lines_at, past_lines
from model.simulate import Uncertainty, build_schedule, simulate

UNCERTAINTY = {
    "measured": Uncertainty(),  # lasting + drift + one-off, from model/calibrate.py
    # Measured x2: the factor that makes 2015-2025 playoff predictions calibrated (model/backtest.py).
    "calibrated": Uncertainty().scaled(2.0),
    "fixed": Uncertainty.fixed(2.7),  # strength fixed after T, one draw per season
    "none": Uncertainty.fixed(0.0),
}


def market_sigma(kalshi):
    sigmas = [k.sigma for k in kalshi if k.sigma]
    if len(sigmas) < 5:
        raise ValueError(f"only {len(sigmas)} games with a usable strike fit at T")
    return statistics.median(sigmas)


def books_at(db_path, slot, tickers):
    db = sqlite3.connect(db_path)
    (round_id,) = db.execute("SELECT round_id FROM rounds WHERE slot = ?", (slot,)).fetchone()
    marks = ",".join("?" * len(tickers))
    rows = db.execute(
        f"SELECT b.ticker, b.book_status, b.yes_bid, b.yes_ask, m.status FROM books b "
        f"LEFT JOIN markets m USING (round_id, ticker) WHERE b.round_id = ? AND b.ticker IN ({marks})",
        (round_id, *tickers),
    ).fetchall()
    db.close()
    return {ticker: (status, bid, ask, market) for ticker, status, bid, ask, market in rows}


def verdict(p, bid, ask):
    if ask is not None and p > ask:
        return "model > ask"
    if bid is not None and p < bid:
        return "model < bid"
    return ""


def cents(x):
    return "  -" if x is None else f"{x * 100:3.0f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--at", type=data.parse_time)
    parser.add_argument("--sims", type=int, default=10000)
    parser.add_argument("--uncertainty", choices=UNCERTAINTY, action="append")
    parser.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE)
    parser.add_argument("--ratings-use-kalshi", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = data_dir()
    path = data.fetch_nflverse(base) if args.refresh else (data.latest_nflverse(base) or data.fetch_nflverse(base))
    games = data.load_games(path)
    at = args.at or data.default_at()
    slot, kalshi = data.kalshi_lines(base / "tidy.sqlite", at, games)
    sigma = market_sigma(kalshi)
    rating_lines = lines_at(games, at, kalshi) if args.ratings_use_kalshi else past_lines(games, at)
    ratings = fit(rating_lines, at, args.half_life)
    schedule = build_schedule(games, at, ratings, {(k.home, k.away, k.gameday): k.line for k in kalshi})
    taus = args.uncertainty or ["fixed"]
    results = {tau: simulate(schedule, sigma, UNCERTAINTY[tau], args.sims, args.seed) for tau in taus}

    print(f"T = {at:%Y-%m-%d %H:%M} UTC   Kalshi snapshot {slot}   nflverse {path.name}")
    print(f"games left {len(schedule.home)} ({int((~np.isnan(schedule.market_line)).sum())} on Kalshi lines)   "
          f"sigma {sigma:.2f} (median of Kalshi strike fits)   half-life {args.half_life:g} wk   "
          f"home field {ratings.home_field:+.2f}   sims {args.sims}   "
          f"ratings use Kalshi: {'yes' if args.ratings_use_kalshi else 'no'}")

    playoff = books_at(base / "tidy.sqlite", slot, [playoff_ticker(t) for t in TEAMS])
    division = books_at(base / "tidy.sqlite", slot, [division_ticker(t) for t in TEAMS])
    main_tau = taus[0]
    main = results[main_tau]
    p_playoff = main.probability(main.playoffs)
    p_division = main.probability(main.division)
    se = main.standard_error(main.playoffs)

    tau_cols = "".join(f"  {t:>9s}" for t in taus)
    print(f"\n{'team':5s}{'rating':>7s}   P(playoffs) +-SE{tau_cols}   Kalshi bid/ask       P(div)  div bid/ask")
    order = sorted(range(32), key=lambda i: (CONFERENCE_OF[TEAMS[i]], -p_playoff[i]))
    rows = []
    for i in order:
        team = TEAMS[i]
        _, bid, ask, status = playoff.get(playoff_ticker(team), (None, None, None, None))
        _, dbid, dask, _ = division.get(division_ticker(team), (None, None, None, None))
        by_tau = "".join(f"  {results[t].probability(results[t].playoffs)[i]:9.3f}" for t in taus)
        flag = verdict(p_playoff[i], bid, ask)
        dflag = verdict(p_division[i], dbid, dask)
        note = "" if status in (None, "active") else f" [{status}]"
        print(f"{team:5s}{ratings.ratings[team]:+7.2f}   {p_playoff[i]:6.3f} {se[i]:.3f}  {by_tau}   "
              f"{cents(bid)} {cents(ask)}  {flag:11s}   {p_division[i]:6.3f}  {cents(dbid)} {cents(dask)}  {dflag}{note}")
        rows.append({"team": team, "conference": CONFERENCE_OF[team], "rating": ratings.ratings[team],
                     "p_playoffs": p_playoff[i], "se": se[i], "p_division": p_division[i],
                     "playoff_bid": bid, "playoff_ask": ask, "division_bid": dbid, "division_ask": dask})

    mids = [(p_playoff[i], playoff[playoff_ticker(TEAMS[i])]) for i in range(32)]
    outside = sum(bool(verdict(p, b, a)) for p, (_, b, a, _) in mids)
    gaps = [abs(p - (b + a) / 2) for p, (_, b, a, _) in mids if b is not None and a is not None]
    print(f"\nvs Kalshi playoff book: {outside} of 32 outside bid/ask, "
          f"mean |model - mid| {np.mean(gaps) * 100:.1f} points (mids only for this summary)")
    bids = [b for _, (_, b, _, _) in playoff.items() if b is not None]
    asks = [a for _, (_, _, a, _) in playoff.items() if a is not None]
    print(f"\nKalshi playoff book: sum of bids {sum(bids):.2f} ({len(bids)} teams), "
          f"sum of asks {sum(asks):.2f} ({len(asks)} teams); model sums to {p_playoff.sum():.2f}")

    out = base / "fairvalues"
    out.mkdir(exist_ok=True)
    out_path = out / f"{at:%Y%m%dT%H%MZ}-{main_tau}-hl{args.half_life:g}{'-k' if args.ratings_use_kalshi else ''}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) + ["sigma", "tau", "half_life", "sims", "slot"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "sigma": sigma, "tau": main_tau, "half_life": args.half_life,
                             "sims": args.sims, "slot": slot})
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
