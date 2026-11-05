import random
import unittest
from datetime import UTC, datetime, timedelta

from model.data import fitted_sigma, median_margin
from model.ratings import Line, fit, predict, weights

T = datetime(2026, 11, 4, 14, tzinfo=UTC)
TEAMS = [f"T{i:02d}" for i in range(32)]


def synthetic_season(true, home_field, weeks=8, noise=0.0, seed=1):
    """Round-robin-ish schedule: each week pairs every team once."""
    rng = random.Random(seed)
    lines = []
    for week in range(weeks):
        order = TEAMS[:]
        rng.shuffle(order)
        kickoff = T - timedelta(weeks=weeks - week)
        for home, away in zip(order[::2], order[1::2]):
            neutral = rng.random() < 0.05
            line = true[home] - true[away] + (0 if neutral else home_field) + rng.gauss(0, noise)
            lines.append(Line(home, away, neutral, line, kickoff, "nflverse"))
    return lines


class FitTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(7)
        raw = {t: rng.gauss(0, 5) for t in TEAMS}
        mean = sum(raw.values()) / 32
        self.true = {t: r - mean for t, r in raw.items()}

    def test_recovers_ratings_exactly_without_noise(self):
        result = fit(synthetic_season(self.true, 1.7), T, None)
        self.assertAlmostEqual(result.home_field, 1.7, places=6)
        for team in TEAMS:
            self.assertAlmostEqual(result.ratings[team], self.true[team], places=6)

    def test_recovers_ratings_closely_with_noise(self):
        result = fit(synthetic_season(self.true, 1.7, noise=1.0), T, 4)
        worst = max(abs(result.ratings[t] - self.true[t]) for t in TEAMS)
        self.assertLess(worst, 1.5)
        self.assertAlmostEqual(result.home_field, 1.7, delta=0.5)

    def test_ratings_average_zero(self):
        result = fit(synthetic_season(self.true, 1.7, noise=1.0), T, 2)
        self.assertAlmostEqual(sum(result.ratings.values()), 0, places=9)

    def test_neutral_site_gets_no_home_field(self):
        result = fit(synthetic_season(self.true, 1.7), T, None)
        self.assertAlmostEqual(
            predict(result, "T00", "T01", neutral=False) - predict(result, "T00", "T01", neutral=True),
            1.7, places=6)

    def test_one_week_alone_is_not_identified(self):
        with self.assertRaises(ValueError):
            fit(synthetic_season(self.true, 1.7, weeks=1), T, None)

    def test_weights_halve_every_half_life(self):
        lines = [Line("A", "B", False, 0, T - timedelta(weeks=w), "nflverse") for w in (0, 2, 4)]
        self.assertEqual(list(weights(lines, T, 2)), [1.0, 0.5, 0.25])
        future = [Line("A", "B", False, 0, T + timedelta(days=3), "kalshi")]
        self.assertEqual(list(weights(future, T, 2)), [1.0])


class StrikeTests(unittest.TestCase):
    def test_median_margin_interpolates_the_cincinnati_example(self):
        # CIN @ ATL, CIN favored: P(CIN by > 2.5) = .545, P(CIN by > 3.5) = .455.
        # From Atlanta's side, P(ATL margin > -2.5) = .455 and > -3.5 = .545.
        self.assertAlmostEqual(median_margin([(-3.5, 0.545), (-2.5, 0.455)]), -3.0)

    def test_median_margin_none_without_a_crossing(self):
        self.assertIsNone(median_margin([(1.5, 0.4), (2.5, 0.3)]))

    def test_fitted_sigma_recovers_a_normal(self):
        from statistics import NormalDist
        points = [(x + 0.5, 1 - NormalDist(3, 13.5).cdf(x + 0.5)) for x in range(-20, 21)]
        self.assertAlmostEqual(fitted_sigma(points), 13.5, places=6)


if __name__ == "__main__":
    unittest.main()
