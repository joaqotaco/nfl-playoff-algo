import unittest

from model import edge
from tests.test_paper import BuildTests


class SlopeTests(unittest.TestCase):
    def test_market_moving_halfway_gives_one_half(self):
        pairs = [(0.10, 0.05, "A"), (-0.20, -0.10, "B"), (0.04, 0.02, "C")]
        k, se, n, clusters = edge.slope(pairs)
        self.assertAlmostEqual(k, 0.5)
        self.assertAlmostEqual(se, 0.0)
        self.assertEqual((n, clusters), (3, 3))

    def test_missing_moves_are_dropped(self):
        self.assertIsNone(edge.slope([(0.1, None, "A")]))


class AnalyseTests(unittest.TestCase):
    """The paper tests' world. BUF: fair 0.98, mid 0.95; NYG: fair 0.44, mid 0.585. 24h after
    14:10 the next snapshot (Nov 11 15:00) has mids 0.96 and 0.51."""
    setUp, tearDown = BuildTests.setUp, BuildTests.tearDown

    def test_k_and_capacity(self):
        rows = {r["team"]: r for r in edge.analyse(self.base)}
        self.assertAlmostEqual(rows["BUF"]["gap"], 0.03)
        self.assertAlmostEqual(rows["BUF"]["moves"]["24h"], 0.01)
        self.assertAlmostEqual(rows["NYG"]["moves"]["24h"], -0.075)
        k, *_ = edge.slope([(r["gap"], r["moves"]["24h"], t) for t, r in rows.items()])
        self.assertAlmostEqual(k, (0.03 * 0.01 + 0.145 * 0.075) / (0.03 ** 2 + 0.145 ** 2))
        # Only t2 (0.94 < our 0.95 bid) and t4 (0.60 > our 0.59 ask) went through.
        self.assertEqual((rows["BUF"]["through"], rows["NYG"]["through"]), (1, 1))
        self.assertAlmostEqual(rows["BUF"]["dollars"] + rows["NYG"]["dollars"], 0.95 + 0.41)

    def test_report_runs(self):
        lines = []
        edge.report(edge.analyse(self.base), out=lines.append)
        self.assertTrue(any("per day" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
