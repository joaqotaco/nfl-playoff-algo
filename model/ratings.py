"""Team ratings from game lines by weighted least squares.

Each line gives one equation:  line = rating_home - rating_away + home_field
Home field is 0 at neutral sites. Ratings average zero. Older lines count less:
weight = 0.5 ** (age in weeks / half-life).

    python3 -m model.ratings                  fit at the latest 14:00 UTC
    python3 -m model.ratings --at 2026-11-04T14:00:00Z --half-life 8
    python3 -m model.ratings --half-life-test
"""
import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from statistics import NormalDist

import numpy as np

from collector.round import data_dir
from model import data

STERN_SIGMA = 13.86
HALF_LIVES = (1, 2, 4, 8, 16, None)  # None means every line counts equally
DEFAULT_HALF_LIFE = 8  # chosen 2026-11-05; see the half-life test


@dataclass(frozen=True)
class Line:
    home: str
    away: str
    neutral: bool
    line: float
    kickoff: datetime
    source: str  # "nflverse" for played games, "kalshi" for upcoming ones


@dataclass
class Fit:
    ratings: dict
    home_field: float
    residual_sd: float
    lines: int
    effective_lines: float


def weights(lines, at, half_life):
    if half_life is None:
        return np.ones(len(lines))
    age_weeks = np.array([max(0.0, (at - l.kickoff).total_seconds() / (7 * 86400)) for l in lines])
    return 0.5 ** (age_weeks / half_life)


def fit(lines, at, half_life):
    teams = sorted({l.home for l in lines} | {l.away for l in lines})
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    X = np.zeros((len(lines) + 1, n + 1))
    y = np.zeros(len(lines) + 1)
    for row, l in enumerate(lines):
        X[row, index[l.home]] = 1
        X[row, index[l.away]] = -1
        X[row, n] = 0 if l.neutral else 1
        y[row] = l.line
    # The ratings-average-zero rule. Shifting every rating by the same amount leaves
    # every equation unchanged, so without this row there is no unique answer.
    X[-1, :n] = 1
    w = np.append(weights(lines, at, half_life), 1.0)
    if np.linalg.matrix_rank(X) < n + 1:
        raise ValueError("ratings not identified: some teams aren't linked by any chain of games")

    root_w = np.sqrt(w)
    solution, *_ = np.linalg.lstsq(X * root_w[:, None], y * root_w, rcond=None)
    residuals = (X @ solution - y)[:-1]
    game_w = w[:-1]
    return Fit(
        ratings={team: float(solution[index[team]]) for team in teams},
        home_field=float(solution[n]),
        residual_sd=float(np.sqrt(np.sum(game_w * residuals**2) / np.sum(game_w))),
        lines=len(lines),
        effective_lines=float(game_w.sum() ** 2 / np.sum(game_w**2)),
    )


def predict(fit_result, home, away, neutral):
    return fit_result.ratings[home] - fit_result.ratings[away] + (0 if neutral else fit_result.home_field)


def past_lines(games, at):
    return [Line(g.home, g.away, g.neutral, g.spread_line, g.kickoff, "nflverse")
            for g in games if g.kickoff < at and g.spread_line is not None]


def lines_at(games, at, kalshi):
    """Information at T: closing lines of games played before T, Kalshi lines after."""
    by_teams = {(g.home, g.away, g.kickoff.astimezone(data.EASTERN).strftime("%Y-%m-%d")): g
                for g in games}
    upcoming = []
    for k in kalshi:
        game = by_teams[(k.home, k.away, k.gameday)]
        upcoming.append(Line(k.home, k.away, game.neutral, k.line, game.kickoff, "kalshi"))
    return past_lines(games, at) + upcoming


def half_life_test(games, half_lives=HALF_LIVES, first_week=4, horizon=1):
    """For each week k: fit on lines before week k, predict the lines of week k + horizon - 1.

    horizon 1 predicts next week. The season simulation needs lines up to 9 weeks
    ahead, where a temporary effect (a backup QB) has usually ended.
    Returns {half_life: (rmse, count)} plus the error of always predicting 0.
    """
    errors = {h: [] for h in half_lives}
    baseline = []
    weeks = sorted({g.week for g in games if g.spread_line is not None})
    for week in (w for w in weeks if w >= first_week and w + horizon - 1 in weeks):
        targets = [g for g in games if g.week == week + horizon - 1 and g.spread_line is not None]
        at = min(g.kickoff for g in games if g.week == week)
        train = past_lines(games, at)
        for h in half_lives:
            fitted = fit(train, at, h)
            errors[h] += [predict(fitted, g.home, g.away, g.neutral) - g.spread_line for g in targets]
        baseline += [g.spread_line for g in targets]

    def rmse(values):
        return math.sqrt(sum(v * v for v in values) / len(values))

    return {h: (rmse(e), len(e)) for h, e in errors.items()}, rmse(baseline)


def stern(line):
    return NormalDist().cdf(line / STERN_SIGMA)


def load_inputs(args):
    base = data_dir()
    path = data.fetch_nflverse(base) if args.refresh else data.latest_nflverse(base)
    if path is None:
        path = data.fetch_nflverse(base)
    return base, path, data.load_games(path)


def print_fit(base, path, games, at, half_life):
    slot, kalshi = data.kalshi_lines(base / "tidy.sqlite", at, games)
    lines = lines_at(games, at, kalshi)
    result = fit(lines, at, half_life)
    print(f"T = {at:%Y-%m-%d %H:%M} UTC   Kalshi snapshot {slot}   nflverse file {path.name}")
    print(f"half-life {half_life or 'none'} weeks   lines {result.lines} "
          f"({sum(l.source == 'kalshi' for l in lines)} from Kalshi)   "
          f"effective lines {result.effective_lines:.0f}")
    print(f"home field {result.home_field:+.2f} points   residual SD {result.residual_sd:.2f} points")
    ranked = sorted(result.ratings.items(), key=lambda kv: -kv[1])
    print(f"best-to-worst gap {ranked[0][1] - ranked[-1][1]:.1f} points")
    for rank, (team, rating) in enumerate(ranked, 1):
        print(f"  {rank:2d} {team:4s} {rating:+6.2f}")

    print("\nKalshi upcoming games: line, fitted SD, and moneyline vs Stern")
    print("  game        line   fit    SD   P(home) Kalshi  Stern")
    for k in sorted(kalshi, key=lambda k: (k.gameday, k.home)):
        game = next(g for g in games if (g.home, g.away) == (k.home, k.away)
                    and g.kickoff.astimezone(data.EASTERN).strftime("%Y-%m-%d") == k.gameday)
        sigma = f"{k.sigma:5.1f}" if k.sigma else "    -"
        win = f"{k.home_win_prob:.3f}" if k.home_win_prob is not None else "    -"
        print(f"  {k.away:>3s}@{k.home:<4s} {k.gameday[5:]} {k.line:+5.1f} "
              f"{predict(result, k.home, k.away, game.neutral):+5.1f} {sigma}   {win}   {stern(k.line):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--at", type=data.parse_time, help="T, e.g. 2026-11-04T14:00:00Z")
    parser.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE)
    parser.add_argument("--half-life-test", action="store_true")
    parser.add_argument("--horizon", type=int, default=1, help="weeks ahead for the half-life test")
    parser.add_argument("--refresh", action="store_true", help="download nflverse again")
    args = parser.parse_args()
    base, path, games = load_inputs(args)

    if args.half_life_test:
        results, baseline = half_life_test(games, horizon=args.horizon)
        print(f"Fit on lines before week k, predict lines {args.horizon} week(s) ahead (k from week 4)")
        print(f"  always predict 0: RMSE {baseline:.2f} points")
        for h, (rmse, count) in results.items():
            print(f"  half-life {str(h or 'none'):>4s}: RMSE {rmse:.2f} points over {count} games")
        return
    print_fit(base, path, games, args.at or data.default_at(), args.half_life)


if __name__ == "__main__":
    main()
