"""Tiebreak steps that 2002-2025 never reached, so model/validate.py can't cover them."""
import unittest
from math import erf, sqrt

import numpy as np

from model.league import INDEX
from model.simulate import Schedule, Uncertainty, simulate
from model.standings import (
    Season, best_in_division_tie, best_in_wild_card_tie, combined_ranking_conference,
    head_to_head_sweep, net_points_all, strength_of_schedule, strength_of_victory,
)


def season(games, seed=0):
    """games: (home, away, home_points, away_points) with team codes."""
    return Season.from_games(
        [INDEX[h] for h, _, _, _ in games], [INDEX[a] for _, a, _, _ in games],
        [hp for _, _, hp, _ in games], [ap for _, _, _, ap in games], rng=np.random.default_rng(seed))


class RecordTests(unittest.TestCase):
    def test_tie_counts_half(self):
        s = season([("BUF", "MIA", 20, 20), ("BUF", "NE", 24, 10)])
        self.assertAlmostEqual(s.pct[INDEX["BUF"]], 0.75)
        self.assertAlmostEqual(s.pct[INDEX["MIA"]], 0.5)

    def test_strength_of_victory_counts_each_win(self):
        # BUF beat MIA twice and NE once. MIA is 1-2 (its win is over NYJ), NE 0-2.
        s = season([("BUF", "MIA", 1, 0), ("MIA", "BUF", 0, 1), ("BUF", "NE", 1, 0),
                    ("MIA", "NYJ", 1, 0), ("NYJ", "NE", 1, 0)])
        (sov,) = strength_of_victory(s, [INDEX["BUF"]])
        # MIA counted twice (1 win of 3 games each time), NE once (0 of 2): 2/8.
        self.assertAlmostEqual(sov, 2 / 8)

    def test_strength_of_schedule_uses_opponents_full_records(self):
        s = season([("BUF", "MIA", 1, 0), ("MIA", "NYJ", 1, 0)])
        (sos,) = strength_of_schedule(s, [INDEX["BUF"]])
        self.assertAlmostEqual(sos, 1 / 2)

    def test_combined_ranking_shares_rank_on_ties(self):
        # BUF and MIA both score 30 (shared rank 1), NE 10. Allowed: BUF 0, MIA 10, NE 30.
        s = season([("BUF", "NE", 30, 0), ("MIA", "NE", 30, 10)])
        values = combined_ranking_conference(s, [INDEX["BUF"], INDEX["MIA"], INDEX["NE"]])
        # BUF: scored rank 1 + allowed rank 1; MIA: 1 + 2; NE: 3 + 3 (other AFC teams 0-0 tie at 4).
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])

    def test_net_points(self):
        s = season([("BUF", "MIA", 31, 10), ("NE", "BUF", 7, 3)])
        self.assertEqual(net_points_all(s, [INDEX["BUF"]]), [17])


class StepTests(unittest.TestCase):
    def test_sweep_advances_a_club_that_beat_all_others(self):
        s = season([("BUF", "KC", 1, 0), ("BUF", "HOU", 1, 0), ("KC", "HOU", 1, 0)])
        group = [INDEX["BUF"], INDEX["KC"], INDEX["HOU"]]
        self.assertEqual(head_to_head_sweep(s, group), [1, 0, 0])

    def test_sweep_does_not_apply_if_not_everyone_met(self):
        s = season([("BUF", "KC", 1, 0), ("KC", "HOU", 1, 0)])
        self.assertIsNone(head_to_head_sweep(s, [INDEX["BUF"], INDEX["KC"], INDEX["HOU"]]))

    def test_identical_clubs_go_to_coin_toss(self):
        # Two division rivals with mirror-image results: every step ties.
        games = [("BUF", "MIA", 20, 10), ("MIA", "BUF", 20, 10)]
        winners = {best_in_division_tie(season(games, seed), [INDEX["BUF"], INDEX["MIA"]])
                   for seed in range(20)}
        self.assertEqual(winners, {INDEX["BUF"], INDEX["MIA"]})

    def test_wild_card_two_clubs_use_head_to_head_first(self):
        s = season([("KC", "HOU", 10, 20), ("KC", "NYJ", 30, 0), ("HOU", "TEN", 0, 30)])
        self.assertEqual(best_in_wild_card_tie(s, [INDEX["KC"], INDEX["HOU"]]), INDEX["HOU"])


class SimulateTests(unittest.TestCase):
    def schedule(self, market_line):
        empty = np.array([], int)
        return Schedule(empty, empty, np.array([]), np.array([]),
                        home=np.array([INDEX["BUF"]]), away=np.array([INDEX["MIA"]]),
                        rating_line=np.array([0.0]), market_line=np.array([market_line]),
                        weeks_ahead=np.array([0]))

    def test_market_line_sets_the_win_probability(self):
        sims, sigma, line = 2000, 11.4, 7.0
        result = simulate(self.schedule(line), sigma, Uncertainty().scaled(2), sims, p_tie=0)
        expected = 0.5 * (1 + erf(line / sigma / sqrt(2)))
        # Only one game is played, so BUF wins the AFC East exactly when it wins the game.
        self.assertAlmostEqual(result.probability(result.division)[INDEX["BUF"]], expected, delta=0.03)

    def test_fourteen_playoff_teams_every_season(self):
        result = simulate(self.schedule(3.0), 11.4, Uncertainty(), 50)
        self.assertEqual(result.playoffs.sum(), 14 * 50)
        self.assertEqual(result.division.sum(), 8 * 50)


if __name__ == "__main__":
    unittest.main()
