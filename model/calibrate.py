"""Evidence for the rating half-life and the strength-uncertainty settings, from
nflverse lines in past seasons.

    python3 -m model.calibrate

1. Half-life test pooled over seasons: fit on lines before week k, predict the lines
   h weeks ahead. Lower RMSE is better.
2. How far future lines stray from the ratings, split per team into:
   - a lasting part (same for all the team's later games): the covariance between a
     team's gap this week and its gap j weeks later, which stays roughly flat in j;
   - drift that grows with the horizon: the pair-gap variance rises about linearly;
   - one-off noise per game: what's left of the week-0 variance.
"""
import collections

import numpy as np

from collector.round import data_dir
from model import data
from model.ratings import HALF_LIVES, fit, half_life_test, past_lines, predict
from model.validate import OLD_CODES

SEASONS = range(2015, 2026)


def load(path, season):
    games = data.load_games(path, str(season))
    return [g.__class__(**{**g.__dict__, "home": OLD_CODES.get(g.home, g.home),
                           "away": OLD_CODES.get(g.away, g.away)}) for g in games]


def pooled_half_life(seasons_games, horizons=(1, 3, 5, 7)):
    for horizon in horizons:
        squares, counts = collections.defaultdict(float), collections.defaultdict(int)
        for games in seasons_games:
            results, _ = half_life_test(games, horizon=horizon)
            for h, (rmse, count) in results.items():
                squares[h] += rmse * rmse * count
                counts[h] += count
        cells = "  ".join(f"{str(h or 'none'):>4s}: {np.sqrt(squares[h] / counts[h]):.3f}" for h in HALF_LIVES)
        print(f"  {horizon} wk ahead  {cells}   ({counts[HALF_LIVES[0]]} games)")


def uncertainty(seasons_games, half_life, max_ahead=8):
    residuals = collections.defaultdict(list)
    products = collections.defaultdict(list)
    for games in seasons_games:
        weeks = sorted({g.week for g in games if g.spread_line is not None})
        for k in (w for w in weeks if w >= 4):
            at = min(g.kickoff for g in games if g.week == k)
            fitted = fit(past_lines(games, at), at, half_life)
            gap = {}
            for g in games:
                if g.spread_line is None or not k <= g.week <= k + max_ahead:
                    continue
                d = g.spread_line - predict(fitted, g.home, g.away, g.neutral)
                residuals[g.week - k].append(d)
                gap[(g.home, g.week)], gap[(g.away, g.week)] = d, -d
            for (team, week), d in gap.items():
                if week == k:
                    for j in range(1, max_ahead + 1):
                        if (team, k + j) in gap:
                            products[j].append(d * gap[(team, k + j)])

    variance = np.array([np.mean(np.square(residuals[j])) for j in range(max_ahead + 1)])
    lasting = float(np.mean([np.mean(products[j]) for j in range(1, max_ahead + 1)]))
    drift = float(np.polyfit(np.arange(max_ahead + 1), variance, 1)[0] / 2)  # per team, per week
    one_off = float(variance[0] / 2 - lasting)
    for j in range(max_ahead + 1):
        print(f"  {j} wk ahead: line-gap SD {np.sqrt(variance[j]):.2f}"
              + (f", lasting covariance {np.mean(products[j]):.2f}" if j else ""))
    print(f"  per team: lasting SD {np.sqrt(lasting):.2f}, drift SD {np.sqrt(drift):.2f} per week, "
          f"one-off SD {np.sqrt(max(one_off, 0)):.2f} per game")


def main():
    path = data.latest_nflverse(data_dir())
    seasons_games = [load(path, s) for s in SEASONS]
    print(f"Half-life test, {SEASONS.start}-{SEASONS.stop - 1} pooled (RMSE in points by half-life in weeks)")
    pooled_half_life(seasons_games)
    for half_life in (2, 8):
        print(f"\nLine gaps from ratings with half-life {half_life}")
        uncertainty(seasons_games, half_life)


if __name__ == "__main__":
    main()
