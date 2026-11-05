"""NFL standings, tiebreakers and playoff seeding.

Follows the NFL's tiebreaking procedures (nfl.com/standings/tie-breaking-procedures),
in the reading nflseedr uses where the text is ambiguous. Two simplifications:
- Net touchdowns (second-to-last step) needs touchdown counts we don't have, so a tie
  that survives net points goes straight to the coin toss.
- Simulated games have made-up scores (see model/simulate.py), which only matter from
  the combined-ranking step onward.
"""
import numpy as np

from model.league import CONFERENCE_OF, CONFERENCES, DIVISION_OF, DIVISIONS, INDEX, TEAMS

EPS = 1e-9
MIN_COMMON_GAMES_WILD_CARD = 4
CONFERENCE_MASK = {
    conf: np.array([CONFERENCE_OF[t] == conf for t in TEAMS]) for conf in CONFERENCES
}


class Season:
    """One season's results as 32x32 matrices, indexed by model.league.INDEX.

    wins[i, j]   games team i won against team j
    ties[i, j]   tied games between i and j (symmetric)
    points[i, j] total points team i scored against team j
    """

    def __init__(self, wins, ties, points, rng=None):
        self.wins, self.ties, self.points = wins, ties, points
        self.games = wins + wins.T + ties
        self.rng = rng or np.random.default_rng()
        # Ties count as half a win and half a loss.
        self.win_points = wins.sum(1) + 0.5 * ties.sum(1)
        self.games_played = self.games.sum(1)
        self.pct = self.win_points / np.maximum(self.games_played, 1)
        self.scored = points.sum(1)
        self.allowed = points.sum(0)
        self.net = points - points.T

    @classmethod
    def from_games(cls, home, away, home_points, away_points, rng=None):
        """home, away: team indices. A tie is equal points."""
        home, away = np.asarray(home), np.asarray(away)
        home_points, away_points = np.asarray(home_points, float), np.asarray(away_points, float)
        wins = np.zeros((32, 32))
        ties = np.zeros((32, 32))
        points = np.zeros((32, 32))
        home_won = home_points > away_points
        away_won = away_points > home_points
        tied = ~home_won & ~away_won
        np.add.at(wins, (home[home_won], away[home_won]), 1)
        np.add.at(wins, (away[away_won], home[away_won]), 1)
        np.add.at(ties, (home[tied], away[tied]), 1)
        np.add.at(ties, (away[tied], home[tied]), 1)
        np.add.at(points, (home, away), home_points)
        np.add.at(points, (away, home), away_points)
        return cls(wins, ties, points, rng)

    def record_pct(self, team, opponents):
        """Win percentage against a set of opponents, or None if no games."""
        opponents = list(opponents)
        games = self.games[team, opponents].sum()
        if games == 0:
            return None
        return (self.wins[team, opponents].sum() + 0.5 * self.ties[team, opponents].sum()) / games


# Each step maps (season, group) to one value per team, higher is better, or None to skip.

def head_to_head(season, group):
    values = [season.record_pct(t, [o for o in group if o != t]) for t in group]
    return None if any(v is None for v in values) else values


def head_to_head_sweep(season, group):
    """Wild card, 3+ clubs: only applies if one club beat, or lost to, each of the others."""
    for t in group:
        others = [o for o in group if o != t]
        if all(season.games[t, o] > 0 and season.wins[t, o] == season.games[t, o] for o in others):
            return [1 if g == t else 0 for g in group]
    for t in group:
        others = [o for o in group if o != t]
        if all(season.games[t, o] > 0 and season.wins[o, t] == season.games[t, o] for o in others):
            return [-1 if g == t else 0 for g in group]
    return None


def division_record(season, group):
    return [season.record_pct(t, [INDEX[o] for o in DIVISIONS[DIVISION_OF[TEAMS[t]]] if INDEX[o] != t])
            for t in group]


def common_opponents(season, group):
    played = np.all(season.games[group] > 0, axis=0)
    played[group] = False
    return np.flatnonzero(played)


def common_games(season, group, minimum=0):
    common = common_opponents(season, group)
    if len(common) == 0 or any(season.games[t, common].sum() < minimum for t in group):
        return None
    return [season.record_pct(t, common) for t in group]


def common_games_wild_card(season, group):
    return common_games(season, group, MIN_COMMON_GAMES_WILD_CARD)


def conference_record(season, group):
    return [season.record_pct(t, np.flatnonzero(CONFERENCE_MASK[CONFERENCE_OF[TEAMS[t]]])) for t in group]


def strength_of_victory(season, group):
    """Combined record of the teams beaten, counting a team once per win over it."""
    values = []
    for t in group:
        beaten = season.wins[t]
        games = beaten @ season.games_played
        values.append((beaten @ season.win_points) / games if games else 0.0)
    return values


def strength_of_schedule(season, group):
    values = []
    for t in group:
        played = season.games[t]
        games = played @ season.games_played
        values.append((played @ season.win_points) / games if games else None)
    return values


def _min_rank(values):
    """Rank 1 = largest. Tied values share the best rank they'd hold alone."""
    return np.array([1 + np.sum(values > v + EPS) for v in values])


def _combined_ranking(season, group, mask):
    members = np.flatnonzero(mask)
    scored = _min_rank(season.scored[members])
    allowed = _min_rank(-season.allowed[members])
    position = {team: k for k, team in enumerate(members)}
    return [-(scored[position[t]] + allowed[position[t]]) for t in group]


def combined_ranking_conference(season, group):
    return _combined_ranking(season, group, CONFERENCE_MASK[CONFERENCE_OF[TEAMS[group[0]]]])


def combined_ranking_league(season, group):
    return _combined_ranking(season, group, np.ones(32, bool))


def net_points_common(season, group):
    common = common_opponents(season, group)
    return [season.net[t, common].sum() for t in group]


def net_points_conference(season, group):
    mask = CONFERENCE_MASK[CONFERENCE_OF[TEAMS[group[0]]]]
    return [season.net[t, mask].sum() for t in group]


def net_points_all(season, group):
    return [season.net[t].sum() for t in group]


def coin_toss(season, group):
    return list(season.rng.permutation(len(group)))


DIVISION_STEPS = (
    head_to_head, division_record, common_games, conference_record,
    strength_of_victory, strength_of_schedule, combined_ranking_conference,
    combined_ranking_league, net_points_common, net_points_all, coin_toss,
)
WILD_CARD_TWO = (
    head_to_head, conference_record, common_games_wild_card, strength_of_victory,
    strength_of_schedule, combined_ranking_conference, combined_ranking_league,
    net_points_conference, net_points_all, coin_toss,
)
WILD_CARD_THREE = (head_to_head_sweep,) + WILD_CARD_TWO[1:]


def _apply_steps(season, group, steps):
    """Run steps in order until one changes the group. Returns the narrowed group."""
    for step in steps:
        values = step(season, group)
        # Skip a step that doesn't apply, including when a club has no games it counts.
        if values is None or any(v is None for v in values):
            continue
        best = max(values)
        keep = [t for t, v in zip(group, values) if v >= best - EPS]
        if len(keep) < len(group):
            return keep
    raise AssertionError("coin toss always separates clubs")


def best_in_division_tie(season, group):
    """Division tiebreaker. Whenever clubs are eliminated, restart at step 1."""
    group = list(group)
    while len(group) > 1:
        group = _apply_steps(season, group, DIVISION_STEPS)
    return group[0]


def best_in_wild_card_tie(season, group):
    """Wild-card tiebreaker for clubs from different divisions.

    Two clubs restart at step 1. Three or more restart at step 2, as nflseedr reads the
    rule, so the head-to-head sweep is only checked on the first pass.
    """
    group = list(group)
    steps = WILD_CARD_TWO if len(group) == 2 else WILD_CARD_THREE
    while len(group) > 1:
        group = _apply_steps(season, group, steps)
        steps = WILD_CARD_TWO if len(group) == 2 else WILD_CARD_THREE[1:]
    return group[0]


def _top_by_pct(season, teams):
    best = max(season.pct[t] for t in teams)
    return [t for t in teams if season.pct[t] >= best - EPS]


def rank_division(season, division):
    remaining = [INDEX[t] for t in DIVISIONS[division]]
    order = []
    while remaining:
        best = best_in_division_tie(season, _top_by_pct(season, remaining))
        order.append(best)
        remaining.remove(best)
    return order


def _order_by_wild_card(season, candidates, slots, division_rank):
    """Fill slots one at a time. Clubs from one division are first cut to the
    highest-ranked one, then clubs from different divisions use wild-card steps."""
    remaining = list(candidates)
    picked = []
    while remaining and len(picked) < slots:
        tied = _top_by_pct(season, remaining)
        leaders = {}
        for t in tied:
            division = DIVISION_OF[TEAMS[t]]
            if division not in leaders or division_rank[t] < division_rank[leaders[division]]:
                leaders[division] = t
        best = best_in_wild_card_tie(season, list(leaders.values()))
        picked.append(best)
        remaining.remove(best)
    return picked


def seed_conference(season, conference, wild_cards=3):
    """Returns (seeds, division winners): team indices in seed order, and the 4 winners.

    3 wild cards since 2020, 2 before."""
    division_rank = {}
    winners = []
    for division in DIVISIONS:
        if division.startswith(conference):
            order = rank_division(season, division)
            division_rank.update({t: k for k, t in enumerate(order)})
            winners.append(order[0])
    seeded_winners = _order_by_wild_card(season, winners, 4, division_rank)
    others = [t for t in division_rank if t not in winners]
    wild = _order_by_wild_card(season, others, wild_cards, division_rank)
    return seeded_winners + wild, winners
