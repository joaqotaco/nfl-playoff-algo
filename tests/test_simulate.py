"""Regression tests: the simulation's counts and the fair-value CSV are pinned to what the code
produced before per-sim outcomes could be kept (commit ad4a19a). The pins hash exact numbers, so
they also break if numpy changes its random streams or float formatting: rerun them on the
previous commit before re-pinning."""
import argparse
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from model import data, fairvalue
from model.ratings import fit, lines_at
from model.simulate import FEEDBACK, Uncertainty, build_schedule, simulate
from tests import world

SIMS = 300
PINNED_COUNTS = {
    "central": "fb1be65d2881bb3a48ff9e6851fb5acfe68c65c5b4ad90ea50e483120f059479",
    "no-feedback": "fa718415114c98b47a3105e5710fcb009bdd5051b7148271a9feaf8d65c05540",
    "all-ties": "4c602033a5f2f128094f8347c676f621f2b86557072b0741c5ec12d754264fe7",
}
PINNED_CSV = "602c135fe4477685f87ff479b177d0bbc3142268152b1180b8862aaf082b24f1"


def counts_hash(result):
    counts = np.concatenate([result.playoffs, result.division, result.seeds.ravel()]).astype(np.int64)
    return hashlib.sha256(counts.tobytes()).hexdigest()


class World:
    """The synthetic world from tests/world.py, built once per test class."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.base = Path(cls.dir.name)
        cls.nflverse = world.build(cls.base)
        cls.games = data.load_games(cls.nflverse)
        _, kalshi = data.kalshi_lines(cls.base / "tidy.sqlite", world.AT, cls.games, use_moneylines=True)
        cls.sigma = fairvalue.market_sigma(kalshi)
        ratings = fit(lines_at(cls.games, world.AT, kalshi), world.AT, 2)
        cls.schedule = build_schedule(cls.games, world.AT, ratings,
                                      {(k.home, k.away, k.gameday): k.line for k in kalshi})
        empty = np.array([], int)
        # Nothing played and every game tied: all 32 records are equal, so coin tosses decide every place.
        cls.all_ties = replace(cls.schedule, played_home=empty, played_away=empty,
                               played_home_points=np.array([]), played_away_points=np.array([]))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def runs(self):
        return {
            "central": lambda **kw: simulate(self.schedule, self.sigma, fairvalue.CENTRAL, SIMS, seed=3, **kw),
            "no-feedback": lambda **kw: simulate(self.schedule, self.sigma, Uncertainty(), SIMS, seed=3, **kw),
            "all-ties": lambda **kw: simulate(self.all_ties, self.sigma, fairvalue.CENTRAL, SIMS, seed=3,
                                              p_tie=1.0, **kw),
        }


class PinnedCountsTests(World, unittest.TestCase):
    def test_counts_match_the_pins(self):
        for name, run in self.runs().items():
            with self.subTest(name):
                self.assertEqual(counts_hash(run()), PINNED_COUNTS[name])

    def test_fair_value_csv_is_unchanged(self):
        args = argparse.Namespace(sims=SIMS, uncertainty=None, half_life=2, ratings_use_kalshi=True,
                                  moneylines=True, seed=0)
        path = fairvalue.run(world.AT, args, self.base, self.nflverse, self.games, out=lambda line: None)
        self.assertEqual(path.name, "20261108T1400Z-calibrated-hl2-k.csv")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), PINNED_CSV)


if __name__ == "__main__":
    unittest.main()
