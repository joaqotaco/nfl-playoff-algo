"""Evidence for the rating half-life and the strength-uncertainty settings, from
nflverse lines in past seasons.

    python3 -m model.calibrate            half-life test and uncertainty estimates
    python3 -m model.calibrate --linked   are rating errors linked between teams?

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
    import sys
    path = data.latest_nflverse(data_dir())
    seasons_games = [load(path, s) for s in SEASONS]
    if "--linked" in sys.argv:
        print(f"Are rating errors linked between teams? {SEASONS.start}-{SEASONS.stop - 1}, weeks 6-15")
        linked_errors(seasons_games)
        return
    print(f"Half-life test, {SEASONS.start}-{SEASONS.stop - 1} pooled (RMSE in points by half-life in weeks)")
    pooled_half_life(seasons_games)
    for half_life in (2, 8):
        print(f"\nLine gaps from ratings with half-life {half_life}")
        uncertainty(seasons_games, half_life)



def linked_errors(seasons_games, model_sims=400, resamples=2000, seed=0):
    """Do teams' rating errors move together, as playoff races care about?

    For each season and week k (6-15), fit ratings as the simulation does, then take each
    team's average gap between its future closing lines (weeks k+1 to k+8) and the ratings.
    Compare how spread out those gaps are within divisions and conferences with what the
    simulation's independent per-team errors produce on the same schedule.
    """
    from model.league import CONFERENCE_OF, DIVISION_OF, DIVISIONS, INDEX
    from model.ratings import Line
    from model.simulate import Uncertainty

    u = Uncertainty()
    rng = np.random.default_rng(seed)
    divisions = [[INDEX[t] for t in teams] for teams in DIVISIONS.values()]
    conferences = [[INDEX[t] for t in INDEX if CONFERENCE_OF[t] == c] for c in ("AFC", "NFC")]
    rivals = [(a, b) for d in divisions for i, a in enumerate(d) for b in d[i + 1:]]

    def stats(d):
        """d: (..., 32) average gap per team. Returns the four spreads, averaged over the last axis."""
        return np.stack([
            np.mean(d ** 2, axis=-1),
            np.mean([np.var(d[..., idx], axis=-1, ddof=1) for idx in divisions], axis=0),
            np.mean([np.var(d[..., idx], axis=-1, ddof=1) for idx in conferences], axis=0),
            np.mean([d[..., a] * d[..., b] for a, b in rivals], axis=0),
        ], axis=-1)

    actual, model, season_of = [], [], []
    for season, games in zip(SEASONS, seasons_games):
        weeks = sorted({g.week for g in games if g.spread_line is not None})
        for k in (w for w in weeks if 6 <= w <= 15):
            current = [g for g in games if g.week == k and g.spread_line is not None]
            at = min(g.kickoff for g in current)
            lines = past_lines(games, at) + [Line(g.home, g.away, g.neutral, g.spread_line, g.kickoff, "c")
                                             for g in current]
            fitted = fit(lines, at, 2)
            future = [g for g in games if 1 <= g.week - k <= 8 and g.spread_line is not None]
            home = np.array([INDEX[g.home] for g in future])
            away = np.array([INDEX[g.away] for g in future])
            step = np.array([g.week - k - 1 for g in future])
            gap = np.array([g.spread_line - predict(fitted, g.home, g.away, g.neutral) for g in future])
            count = np.bincount(home, minlength=32) + np.bincount(away, minlength=32)
            if np.any(count == 0):
                continue

            def team_average(game_gaps):
                total = np.zeros(game_gaps.shape[:-1] + (32,))
                np.add.at(total.T, home, game_gaps.T)
                np.add.at(total.T, away, -game_gaps.T)
                return total / count

            # The simulation's errors on this schedule: lasting + drift (random walk) + one-off.
            lasting = rng.normal(0, u.lasting_sd, (model_sims, 1, 32))
            walk = np.concatenate([np.zeros((model_sims, 1, 32)), np.cumsum(
                rng.normal(0, u.drift_sd, (model_sims, step.max(), 32)), axis=1)], axis=1)
            error = lasting + walk
            sim_gap = (error[:, step, home] - error[:, step, away]
                       + rng.normal(0, u.one_off_sd * np.sqrt(2), (model_sims, len(future))))
            actual.append(stats(team_average(gap)))
            model.append(stats(team_average(sim_gap)).mean(axis=0))
            season_of.append(season)

    actual, model, season_of = np.array(actual), np.array(model), np.array(season_of)
    names = ["each team's gap (mean square)", "spread within a division", "spread within a conference",
             "covariance of division rivals"]
    seasons = np.unique(season_of)
    boot = []
    for _ in range(resamples):
        pick = np.concatenate([np.flatnonzero(season_of == s) for s in rng.choice(seasons, len(seasons))])
        boot.append(actual[pick].mean(axis=0) / model[pick].mean(axis=0))
    low, high = np.percentile(boot, [5, 95], axis=0)
    print(f"  {len(actual)} fits; actual vs the simulation's independent errors (x1)")
    for i, name in enumerate(names):
        a, m = actual[:, i].mean(), model[:, i].mean()
        print(f"  {name:32s} actual {a:6.2f}  model {m:6.2f}  ratio {a / m:5.2f}  (90% range {low[i]:.2f}-{high[i]:.2f})")


if __name__ == "__main__":
    main()
