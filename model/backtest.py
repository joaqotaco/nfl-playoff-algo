"""Score the whole model on past seasons: simulate from the start of weeks 6, 9, 12 and
15, then compare P(playoffs) with who actually made it.

    python3 -m model.backtest

nflverse closing lines stand in for Kalshi's lines at T for that week's games. Also
checks which game-outcome spread (sigma) best predicts real game winners from lines.
"""
import math

import numpy as np

from collector.round import data_dir
from model import data
from model.calibrate import load
from model.league import INDEX, TEAMS
from model.ratings import Line, fit, past_lines
from model.simulate import Uncertainty, build_schedule, simulate
from model.validate import actual_field, load_rows, wild_cards

SEASONS = range(2015, 2026)
WEEKS = (6, 9, 12, 15)
SIGMAS = (10.0, 11.0, 11.4, 12.0, 13.0, 13.86)
CONFIGS = {
    "A: half-life 8, no current lines in ratings, fixed 2.7": dict(
        half_life=8, use_current=False, uncertainty=Uncertainty.fixed(2.7), sigma=11.4),
    "B: half-life 2, current lines in ratings, measured": dict(
        half_life=2, use_current=True, uncertainty=Uncertainty(), sigma=11.4),
    "C: as B, sigma 13.86 (Stern)": dict(
        half_life=2, use_current=True, uncertainty=Uncertainty(), sigma=13.86),
    "D: as B, no strength uncertainty": dict(
        half_life=2, use_current=True, uncertainty=Uncertainty.fixed(0.0), sigma=11.4),
}


def game_sigma_test(seasons_games):
    """Log loss of P(home wins) = Phi(line / sigma) on decided games."""
    games = [g for season in seasons_games for g in season
             if g.spread_line is not None and g.result not in (None, 0.0)]
    print(f"Game winners from lines, {len(games)} games: log loss by sigma (lower is better)")
    for sigma in SIGMAS:
        loss = 0.0
        for g in games:
            p = 0.5 * math.erfc(-g.spread_line / sigma / math.sqrt(2))
            loss -= math.log(p if g.result > 0 else 1 - p)
        print(f"  sigma {sigma:5.2f}: {loss / len(games):.4f}")


def predictions(seasons_games, rows, config, sims):
    out = []
    for season, games in zip(SEASONS, seasons_games):
        made = actual_field(rows, season, "AFC")["field"] | actual_field(rows, season, "NFC")["field"]
        for week in WEEKS:
            current = [g for g in games if g.week == week and g.spread_line is not None]
            if not current:
                continue
            at = min(g.kickoff for g in current)
            lines = past_lines(games, at)
            if config["use_current"]:
                lines += [Line(g.home, g.away, g.neutral, g.spread_line, g.kickoff, "current") for g in current]
            ratings = fit(lines, at, config["half_life"])
            market = {(g.home, g.away, g.kickoff.astimezone(data.EASTERN).strftime("%Y-%m-%d")): g.spread_line
                      for g in current}
            schedule = build_schedule(games, at, ratings, market)
            result = simulate(schedule, config["sigma"], config["uncertainty"], sims,
                              seed=season * 100 + week, wild_cards=wild_cards(season))
            p = result.probability(result.playoffs)
            out += [(p[INDEX[t]], t in made) for t in TEAMS]
    return out


def score(preds):
    p = np.clip(np.array([x for x, _ in preds]), 1e-4, 1 - 1e-4)
    y = np.array([o for _, o in preds], float)
    brier = np.mean((p - y) ** 2)
    log_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    buckets = []
    for lo, hi in ((0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)):
        mask = (p >= lo) & (p < hi)
        if mask.any():
            buckets.append(f"{lo:.2f}-{min(hi, 1):.2f}: said {p[mask].mean():.2f} got {y[mask].mean():.2f} (n={mask.sum()})")
    return brier, log_loss, buckets


def main(sims=2000):
    path = data.latest_nflverse(data_dir())
    seasons_games = [load(path, s) for s in SEASONS]
    rows = load_rows(path)
    game_sigma_test(seasons_games)
    print(f"\nPlayoff predictions, {SEASONS.start}-{SEASONS.stop - 1}, from weeks {WEEKS}, {sims} sims each")
    for name, config in CONFIGS.items():
        brier, log_loss, buckets = score(predictions(seasons_games, rows, config, sims))
        print(f"\n{name}\n  Brier {brier:.4f}   log loss {log_loss:.4f}")
        for line in buckets:
            print(f"    {line}")


if __name__ == "__main__":
    main()
