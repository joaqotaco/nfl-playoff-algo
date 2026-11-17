"""Regression tests: the simulation's counts and the fair-value CSV are pinned to what the code
produced before per-sim outcomes could be kept (commit ad4a19a). They hash exact numbers and pass
on numpy 1.26 (the server's) and 2.4. If they break after a numpy upgrade, check them on the
previous commit before re-pinning: numpy may have changed its random streams."""
import argparse
import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from model import data, export_joint, fairvalue
from model.league import CONFERENCE_OF, TEAMS
from model.ratings import fit, lines_at
from model.simulate import Uncertainty, build_schedule, check_seed_matrix, simulate
from tests import world

SIMS = 300
PINNED_COUNTS = {
    "central": "fb1be65d2881bb3a48ff9e6851fb5acfe68c65c5b4ad90ea50e483120f059479",
    "no-feedback": "fa718415114c98b47a3105e5710fcb009bdd5051b7148271a9feaf8d65c05540",
    "all-ties": "4c602033a5f2f128094f8347c676f621f2b86557072b0741c5ec12d754264fe7",
}
PINNED_CSV = "2c9b09e11f7d4324051c8a1a8c83856a152755d27129258be8e19d5283d4709e"


def csv_hash(path):
    """The CSV's bytes, except ratings rounded to 1e-9: least squares differs in the last digits
    between numpy builds (1.26 vs 2.4), while every probability matches exactly. The next_line
    column (added 2026-11-16, after the pin) is left out; NextLineTests checks it."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if "next_line" in rows[0]:
        drop = rows[0].index("next_line")
        rows = [r[:drop] + r[drop + 1:] for r in rows]
    col = rows[0].index("rating")
    for row in rows[1:]:
        row[col] = f"{float(row[col]):.9f}"
    return hashlib.sha256("\n".join(",".join(r) for r in rows).encode()).hexdigest()


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
        self.assertEqual(csv_hash(path), PINNED_CSV)

class NextLineTests(World, unittest.TestCase):
    def kalshi(self):
        return data.kalshi_lines(self.base / "tidy.sqlite", world.AT, self.games, use_moneylines=True)[1]

    def test_every_team_has_a_line_when_its_next_game_is_listed(self):
        self.assertEqual(data.teams_with_next_line(self.games, world.AT, self.kalshi()), set(TEAMS))

    def test_unlisted_next_game_means_no_line_for_both_teams(self):
        kalshi = self.kalshi()
        gone = kalshi[0]
        rest = [k for k in kalshi if k is not gone]
        self.assertEqual(set(TEAMS) - data.teams_with_next_line(self.games, world.AT, rest), {gone.home, gone.away})

    def test_only_the_next_game_counts(self):
        # A line for the game after next doesn't help: drop week 9's games from the schedule's
        # lines but keep a made-up week-10 one.
        week10 = next(g for g in self.games if g.week == world.PLAYED_WEEKS + 2)
        fake = [data.KalshiLine(week10.home, week10.away, data.gameday(week10), 1.0, None, None)]
        self.assertEqual(data.teams_with_next_line(self.games, world.AT, fake), set())

    def test_fair_value_csv_has_the_column(self):
        args = argparse.Namespace(sims=SIMS, uncertainty=None, half_life=2, ratings_use_kalshi=True,
                                  moneylines=True, seed=0)
        path = fairvalue.run(world.AT, args, self.base, self.nflverse, self.games, out=lambda line: None)
        with open(path, newline="") as f:
            self.assertEqual({r["next_line"] for r in csv.DictReader(f)}, {"1"})


class KeepOutcomesTests(World, unittest.TestCase):
    def test_keeping_outcomes_changes_no_count(self):
        for name, run in self.runs().items():
            with self.subTest(name):
                off, on = run(), run(keep_outcomes=True)
                self.assertIsNone(off.seed_matrix)
                for field in ("playoffs", "division", "seeds", "wins"):
                    np.testing.assert_array_equal(getattr(off, field), getattr(on, field))
                self.assertEqual(counts_hash(on), PINNED_COUNTS[name])

    def test_seed_matrix_adds_up_to_the_counts(self):
        for name, run in self.runs().items():
            with self.subTest(name):
                r = run(keep_outcomes=True)
                m = r.seed_matrix
                self.assertEqual((m.shape, m.dtype), ((SIMS, 32), np.int8))
                np.testing.assert_array_equal((m > 0).sum(axis=0), r.playoffs)
                np.testing.assert_array_equal(((m >= 1) & (m <= 4)).sum(axis=0), r.division)
                for k in range(7):
                    np.testing.assert_array_equal((m == k + 1).sum(axis=0), r.seeds[:, k])
                for conference in ("AFC", "NFC"):
                    block = m[:, [CONFERENCE_OF[t] == conference for t in TEAMS]]
                    self.assertTrue(np.all((block > 0).sum(axis=1) == 7))
                    self.assertTrue(np.all(np.sort(block, axis=1)[:, -7:] == np.arange(1, 8)))

    def test_check_rejects_a_bad_season(self):
        m = self.runs()["central"](keep_outcomes=True).seed_matrix.copy()
        check_seed_matrix(m)
        m[5, np.flatnonzero(m[5] == 2)[0]] = 3  # seed 3 twice, seed 2 missing
        with self.assertRaises(ValueError):
            check_seed_matrix(m)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.base = Path(self.dir.name) / "collector"
        self.base.mkdir()
        self.nflverse = world.build(self.base)
        # An older download that auto must not pick.
        (self.base / "nflverse" / "games-20261108T120000Z.csv").write_text("not used")
        args = argparse.Namespace(sims=SIMS, uncertainty=None, half_life=2, ratings_use_kalshi=True,
                                  moneylines=True, seed=0)
        self.csv = fairvalue.run(world.AT, args, self.base, self.nflverse, data.load_games(self.nflverse),
                                 out=lambda line: None)
        self.out = Path(self.dir.name) / "joint"
        self.lines = []

    def tearDown(self):
        self.dir.cleanup()

    def export(self, *extra, out=None):
        return export_joint.main(["--at", "2026-11-08T14:00:00Z", "--data-dir", str(self.base),
                                  "--nflverse", "auto", "--out", str(out or self.out), "--sims", str(SIMS),
                                  *extra], out=self.lines.append)

    def test_export_matches_the_csv_and_writes_nothing_in_the_data_dir(self):
        before = sorted(p.relative_to(self.base) for p in self.base.rglob("*"))
        self.assertEqual(self.export(), 0)
        self.assertEqual(sorted(p.relative_to(self.base) for p in self.base.rglob("*")), before)
        text = "\n".join(self.lines)
        self.assertIn(f"chose {world.NFLVERSE_NAME}", text)
        self.assertIn("max z 0.00", text)
        self.assertIn("bit-identical counts: yes", text)

        files = sorted(p.name for p in self.out.iterdir())
        self.assertEqual(files, [f"20261108T1400Z-x{s}-s{SIMS}-seed0.npz" for s in ("1", "1.5", "2")])
        with np.load(self.out / files[1], allow_pickle=False) as z:
            self.assertEqual((z["seed"].shape, z["seed"].dtype), ((SIMS, 32), np.int8))
            self.assertEqual(tuple(z["teams"].tolist()), TEAMS)
            self.assertEqual(z["tickers"][0], f"KXNFLPLAYOFF-27-{TEAMS[0]}")
            np.testing.assert_array_equal(z["p_playoffs"], (z["seed"] > 0).mean(axis=0))
            settings = json.loads(str(z["settings"]))
            self.assertEqual((settings["scale"], settings["feedback"], settings["nflverse_file"]),
                             (1.5, fairvalue.CENTRAL.feedback, world.NFLVERSE_NAME))
            self.assertEqual(float(z["scale"]), 1.5)
        with open(self.csv, newline="") as f:
            csv_p = {r["team"]: float(r["p_playoffs"]) for r in csv.DictReader(f)}
        with np.load(self.out / files[1]) as z:
            self.assertEqual(dict(zip(z["teams"], z["p_playoffs"].tolist())), csv_p)

    def test_a_csv_from_another_model_fails_the_check(self):
        rows = self.csv.read_text().splitlines()
        i = next(k for k, r in enumerate(rows) if r.startswith("HOU,"))
        cells = rows[i].split(",")
        cells[3] = "0.2"  # p_playoffs: 0.485 in this world
        rows[i] = ",".join(cells)
        self.csv.write_text("\n".join(rows) + "\n")
        self.assertEqual(self.export(), 1)
        self.assertIn("FAIL", "\n".join(self.lines))

    def test_refuses_to_write_under_fairvalues(self):
        for target in (self.base / "fairvalues", self.base / "fairvalues" / "joint"):
            with self.subTest(str(target)), self.assertRaises(SystemExit):
                self.export(out=target)
        self.assertEqual([p.name for p in (self.base / "fairvalues").iterdir()], [self.csv.name])

    def test_explicit_nflverse_path_must_exist(self):
        with self.assertRaises(SystemExit):
            export_joint.main(["--at", "2026-11-08T14:00:00Z", "--data-dir", str(self.base),
                               "--nflverse", str(self.base / "nope.csv"), "--out", str(self.out)])


if __name__ == "__main__":
    unittest.main()
