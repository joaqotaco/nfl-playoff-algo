"""Simulate the rest of the season many times and count playoff outcomes.

For each simulated season:
1. Set every remaining game's line:
   - with a Kalshi line at T: that line, unchanged (the market already prices this week);
   - otherwise: rating_home - rating_away + home_field + error_home - error_away + noise,
     where each team's error is a lasting part (drawn once per season) plus drift that
     grows with the weeks ahead, and noise is a one-off per game. See Uncertainty.
2. Draw each game: a tie with probability P_TIE, otherwise the home margin is
   N(line, sigma^2), with sigma implied by Kalshi's spread strikes at T.
3. Seed both conferences with the official tiebreakers.
"""
from dataclasses import dataclass

import numpy as np

from model.data import EASTERN
from model.league import CONFERENCES, INDEX
from model.standings import Season, seed_conference

# 8 ties in 2,383 regular-season games, 2017-2025 (overtime shortened to 10 minutes in 2017).
P_TIE = 8 / 2383
# Mean total points per game, 2021-2025. Only used to make up scores for the
# points-based tiebreakers, which come after strength of schedule.
MEAN_TOTAL = 45.0


# Later lines move this many points per point a team beat its earlier lines (2015-2025).
FEEDBACK = 0.061


@dataclass(frozen=True)
class Uncertainty:
    """How far future lines may stray from today's ratings, per team (points).

    Estimated from 2015-2025 lines by model/calibrate.py with a 2-week half-life.
    """
    lasting_sd: float = 1.78  # same for all the team's remaining games
    drift_sd: float = 1.00  # added per week ahead, as a random walk
    one_off_sd: float = 1.45  # per team, per game
    # Later lines move this many points per point a team beat (or missed) its earlier lines
    # since T. 0.061 measured on 2015-2025; 0 turns it off.
    feedback: float = 0.0

    def scaled(self, factor):
        return Uncertainty(self.lasting_sd * factor, self.drift_sd * factor, self.one_off_sd * factor,
                           self.feedback)

    @classmethod
    def fixed(cls, sd):
        """Strength fixed after T: one lasting draw and nothing else."""
        return cls(lasting_sd=sd, drift_sd=0.0, one_off_sd=0.0)


@dataclass
class Schedule:
    """Games played before T (with scores) and games left (with lines)."""
    played_home: np.ndarray
    played_away: np.ndarray
    played_home_points: np.ndarray
    played_away_points: np.ndarray
    home: np.ndarray
    away: np.ndarray
    rating_line: np.ndarray  # from ratings, before the strength draw
    market_line: np.ndarray  # Kalshi line at T, NaN where there isn't one
    weeks_ahead: np.ndarray  # NFL weeks after the current one (0 = this week)


@dataclass
class Result:
    sims: int
    playoffs: np.ndarray  # per team, count of seasons making the playoffs
    division: np.ndarray  # count of division titles
    seeds: np.ndarray  # 32 x 7 counts of each seed
    wins_mean: np.ndarray = None  # per team, mean remaining wins (ties count half)
    wins_var: np.ndarray = None  # per team, variance of remaining wins
    home_win: np.ndarray = None  # per remaining game, P(home wins), ties counting half
    wins: np.ndarray = None  # sims x 32 remaining wins, for joint (between-team) checks

    def probability(self, counts):
        return counts / self.sims

    def standard_error(self, counts):
        p = counts / self.sims
        return np.sqrt(p * (1 - p) / self.sims)


def build_schedule(games, at, fit, market_lines):
    """games: model.data.Game list. market_lines: {(home, away, gameday): line}."""
    played = [g for g in games if g.kickoff < at]
    missing = [g.game_id for g in played if g.result is None]
    if missing:
        raise ValueError(f"games before T without a result (refresh nflverse?): {missing}")
    left = [g for g in games if g.kickoff >= at]
    return Schedule(
        played_home=np.array([INDEX[g.home] for g in played]),
        played_away=np.array([INDEX[g.away] for g in played]),
        played_home_points=np.array([g.home_score for g in played]),
        played_away_points=np.array([g.away_score for g in played]),
        home=np.array([INDEX[g.home] for g in left]),
        away=np.array([INDEX[g.away] for g in left]),
        rating_line=np.array([fit.ratings[g.home] - fit.ratings[g.away]
                              + (0 if g.neutral else fit.home_field) for g in left]),
        market_line=np.array([market_lines.get(
            (g.home, g.away, g.kickoff.astimezone(EASTERN).strftime("%Y-%m-%d")), np.nan)
            for g in left]),
        weeks_ahead=np.array([g.week - min(x.week for x in left) for g in left]),
    )


def simulate(schedule, sigma, uncertainty, sims, seed=0, p_tie=P_TIE, wild_cards=3):
    rng = np.random.default_rng(seed)
    n = len(schedule.home)
    # Ratings include this week's lines, so drift starts the week after.
    steps = np.maximum(schedule.weeks_ahead - 1, 0)
    lasting = rng.normal(0, uncertainty.lasting_sd, (sims, 1, 32))
    # walk[:, j] is the sum of j weekly increments, so it's 0 for the first week.
    increments = rng.normal(0, uncertainty.drift_sd, (sims, steps.max(), 32))
    walk = np.concatenate([np.zeros((sims, 1, 32)), np.cumsum(increments, axis=1)], axis=1)
    error = lasting + walk  # sims x weeks x teams
    noise = rng.normal(0, uncertainty.one_off_sd * np.sqrt(2), (sims, n))
    has_market = ~np.isnan(schedule.market_line)
    lines = np.where(
        has_market,
        schedule.market_line,
        schedule.rating_line + error[:, steps, schedule.home] - error[:, steps, schedule.away] + noise,
    )
    tie = rng.random((sims, n)) < p_tie
    if uncertainty.feedback == 0:
        margins = rng.normal(lines, sigma)
        margins[tie] = 0.0
    else:
        # Play week by week: each result moves both teams' later lines by feedback x surprise.
        margins = np.empty((sims, n))
        surprise = np.zeros((sims, 32))
        for week in np.unique(schedule.weeks_ahead):
            idx = np.flatnonzero(schedule.weeks_ahead == week)
            h, a = schedule.home[idx], schedule.away[idx]
            line = lines[:, idx] + uncertainty.feedback * (surprise[:, h] - surprise[:, a])
            m = rng.normal(line, sigma)
            m[tie[:, idx]] = 0.0
            margins[:, idx] = m
            np.add.at(surprise.T, h, (m - line).T)
            np.add.at(surprise.T, a, -(m - line).T)

    # Remaining wins per team in each simulated season (a tie is half a win).
    home_share = np.where(margins > 0, 1.0, np.where(margins < 0, 0.0, 0.5))
    wins = np.zeros((sims, 32))
    np.add.at(wins.T, schedule.home, home_share.T)
    np.add.at(wins.T, schedule.away, (1 - home_share).T)

    home = np.concatenate([schedule.played_home, schedule.home])
    away = np.concatenate([schedule.played_away, schedule.away])
    playoffs = np.zeros(32)
    division = np.zeros(32)
    seeds = np.zeros((32, 4 + wild_cards))
    for s in range(sims):
        home_points = np.concatenate([schedule.played_home_points, (MEAN_TOTAL + margins[s]) / 2])
        away_points = np.concatenate([schedule.played_away_points, (MEAN_TOTAL - margins[s]) / 2])
        season = Season.from_games(home, away, home_points, away_points, rng=rng)
        for conference in CONFERENCES:
            seeded, winners = seed_conference(season, conference, wild_cards)
            playoffs[seeded] += 1
            division[winners] += 1
            seeds[seeded, np.arange(len(seeded))] += 1
    return Result(sims, playoffs, division, seeds, wins.mean(axis=0), wins.var(axis=0),
                  home_share.mean(axis=0), wins)

