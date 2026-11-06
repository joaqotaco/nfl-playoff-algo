import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from collector import tidy
from model import paper
from model.paper import agrees_with_longshot_bias, clustered_se, quote_price, taker_fee

GAMES_HEADER = ["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "home_team",
                "location", "spread_line", "result", "home_score", "away_score"]


class PricingTests(unittest.TestCase):
    def test_taker_fee_rounds_up_to_the_cent(self):
        self.assertEqual(taker_fee(0.50), 0.02)  # 0.07 * 0.25 = 1.75c
        self.assertEqual(taker_fee(0.96), 0.01)  # 0.27c rounds up

    def test_bid_one_tick_above_the_best_bid(self):
        self.assertEqual(quote_price("bid", 0.90, 0.95, 0.80, 0.85), 0.81)

    def test_bid_never_crosses_the_spread(self):
        # One-cent market: joining the best bid is as far as a passive bid goes.
        self.assertEqual(quote_price("bid", 0.90, 0.95, 0.80, 0.81), 0.80)

    def test_barely_beating_the_ask_still_quotes_inside_the_spread(self):
        # The cap (band low - 1c cushion) can't bind while the bid must stay below the ask;
        # it would only matter with a maker fee.
        self.assertEqual(quote_price("bid", 0.851, 0.90, 0.80, 0.85), 0.81)

    def test_no_quote_when_band_overlaps_market(self):
        self.assertIsNone(quote_price("bid", 0.40, 0.50, 0.44, 0.46))
        self.assertIsNone(quote_price("ask", 0.40, 0.50, 0.44, 0.46))

    def test_ask_one_tick_below_the_best_ask(self):
        self.assertEqual(quote_price("ask", 0.30, 0.40, 0.55, 0.60), 0.59)


class ReportTests(unittest.TestCase):
    def test_clustered_errors_count_game_days_not_fills(self):
        # Ten identical fills before one game day and ten before another: two pieces of
        # evidence. The plain SE would claim far more certainty.
        values = [0.05] * 10 + [-0.03] * 10
        days = ["d1"] * 10 + ["d2"] * 10
        self.assertAlmostEqual(clustered_se(values, days), 0.04, places=9)
        self.assertIsNone(clustered_se(values, ["d1"] * 20))

    def test_longshot_bias_direction(self):
        self.assertTrue(agrees_with_longshot_bias("bid", 0.80))  # buy a favorite
        self.assertTrue(agrees_with_longshot_bias("ask", 0.10))  # sell a longshot
        self.assertFalse(agrees_with_longshot_bias("bid", 0.15))  # buy a longshot
        self.assertFalse(agrees_with_longshot_bias("ask", 0.88))  # sell a favorite


class BuildTests(unittest.TestCase):
    """A small world: BUF looks cheap and its bid fills; NYG looks rich, its ask fills
    later, and a taker sale is logged at the next snapshot."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.base = Path(self.dir.name)
        db = sqlite3.connect(self.base / "tidy.sqlite")
        db.executescript(tidy.SCHEMA)
        books = {  # round start -> {ticker: (bid, ask)}
            "2026-11-10T14:00:01Z": {"BUF": (0.94, 0.96), "NYG": (0.58, 0.59)},
            "2026-11-10T14:30:01Z": {"BUF": (0.94, 0.96), "NYG": (0.58, 0.59)},
            "2026-11-10T15:30:01Z": {"BUF": (0.96, 0.97), "NYG": (0.55, 0.57)},
            "2026-11-11T15:00:01Z": {"BUF": (0.95, 0.97), "NYG": (0.50, 0.52)},
            "2026-11-12T14:00:01Z": {"BUF": (0.97, 0.98), "NYG": (0.45, 0.47)},
            "2026-11-13T14:00:01Z": {"BUF": (None, None), "NYG": (0.44, 0.46)},
        }
        results = {"2026-11-13T14:00:01Z": {"BUF": "yes"}}  # BUF clinches and settles
        for k, (started, by_team) in enumerate(books.items()):
            rid = f"R{k}"
            db.execute("INSERT INTO rounds VALUES (?,?,?,?,?,?,?,?)",
                       (rid, started[:17] + "00Z", started, started, "ok", 1, 0, 0))
            for team, (bid, ask) in by_team.items():
                ticker = f"KXNFLPLAYOFF-27-{team}"
                db.execute("INSERT INTO books VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (rid, ticker, started, "ok", 1, 1, bid, 10, ask, 10))
                result = results.get(started, {}).get(team, "")
                db.execute("INSERT INTO markets VALUES (?,?,?,?,?,?,?,?,?)",
                           (rid, ticker, "KXNFLPLAYOFF-27", team, "finalized" if result else "active", result, "", 0, 0))
        trades = [
            ("t1", "BUF", "2026-11-10T14:05:00.000000Z", 0.90),  # before the quote starts
            ("t2", "BUF", "2026-11-10T14:20:00.000000Z", 0.94),  # through our 0.95 bid
            ("t3", "NYG", "2026-11-10T14:40:00.000000Z", 0.59),  # at our ask, not through
            ("t4", "NYG", "2026-11-10T15:00:00.000000Z", 0.60),  # through our 0.59 ask
        ]
        for tid, team, created, price in trades:
            db.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
                       (tid, f"KXNFLPLAYOFF-27-{team}", created, price, 1, "yes", "bid", 0))
        db.commit()
        db.close()

        (self.base / "nflverse").mkdir()
        with open(self.base / "nflverse" / "games-20261110T000000Z.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(GAMES_HEADER)
            # Thursday night game: 20:15 ET on Nov 11 = 01:15 UTC Nov 12.
            w.writerow(["g1", "2026", "REG", "11", "2026-11-11", "20:15", "MIA", "NYJ", "Home", "", "", "", ""])

        (self.base / "fairvalues").mkdir()
        with open(self.base / "fairvalues" / "20261110T1400Z-calibrated-hl2-k.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["team", "p_playoffs", "p_playoffs_low", "p_playoffs_high",
                                              "playoff_bid", "playoff_ask"])
            w.writeheader()
            w.writerow(dict(team="BUF", p_playoffs=0.98, p_playoffs_low=0.97, p_playoffs_high=0.99,
                            playoff_bid=0.94, playoff_ask=0.96))
            w.writerow(dict(team="NYG", p_playoffs=0.44, p_playoffs_low=0.40, p_playoffs_high=0.48,
                            playoff_bid=0.58, playoff_ask=0.59))
            w.writerow(dict(team="DAL", p_playoffs=0.30, p_playoffs_low=0.27, p_playoffs_high=0.33,
                            playoff_bid=0.28, playoff_ask=0.30))
        self.out = paper.build(self.base, self.base / "paper.sqlite")

    def tearDown(self):
        self.out.close()
        self.dir.cleanup()

    def test_quotes_and_fills(self):
        rows = self.out.execute("SELECT team, side, price, status, fill_time, end FROM quotes ORDER BY team").fetchall()
        self.assertEqual([r[:4] for r in rows], [("BUF", "bid", 0.95, "filled"), ("NYG", "ask", 0.59, "filled")])
        self.assertTrue(rows[0][4].startswith("2026-11-10T14:20"))
        self.assertTrue(rows[1][4].startswith("2026-11-10T15:00"))
        # Next quote would be Nov 11 14:10, before the Thursday kickoff.
        self.assertTrue(rows[0][5].startswith("2026-11-11T14:10"))

    def test_bid_markouts(self):
        marks = dict(self.out.execute(
            "SELECT horizon, round(pnl_mid, 4) FROM markouts m JOIN quotes q ON m.ref = q.id "
            "WHERE kind = 'quote' AND q.team = 'BUF'").fetchall())
        self.assertEqual(marks["1h"], round(0.965 - 0.95, 4))  # 15:30 snapshot mid
        self.assertEqual(marks["24h"], round(0.96 - 0.95, 4))  # Nov 11 15:00 snapshot mid
        self.assertEqual(marks["next game day"], round(0.975 - 0.95, 4))  # Nov 12 14:00

    def test_taker_sale_at_next_snapshot(self):
        rows = self.out.execute("SELECT team, side, price, fee, entry_time FROM takers").fetchall()
        self.assertEqual(rows, [("NYG", "ask", 0.58, 0.02, "2026-11-10T14:30:01.000000Z")])
        exit_1h = self.out.execute("SELECT round(pnl_exit, 4) FROM markouts WHERE kind = 'taker' "
                                   "AND horizon = '1h'").fetchone()[0]
        # Sold at 0.58 (fee 2c), bought back at the 15:30 ask 0.57 (fee 2c).
        self.assertEqual(exit_1h, round(0.58 - 0.57 - 0.02 - 0.02, 4))


    def test_settled_market_marked_at_payout(self):
        marks = dict(self.out.execute(
            "SELECT horizon, round(pnl_mid, 4) FROM markouts m JOIN quotes q ON m.ref = q.id "
            "WHERE kind = 'quote' AND q.team = 'BUF'").fetchall())
        self.assertEqual(marks["latest"], round(1.0 - 0.95, 4))
        self.assertEqual(marks["settlement"], round(1.0 - 0.95, 4))
        nyg = [h for (h,) in self.out.execute(
            "SELECT horizon FROM markouts m JOIN quotes q ON m.ref = q.id WHERE kind = 'quote' AND q.team = 'NYG'")]
        self.assertIn("latest", nyg)
        self.assertNotIn("settlement", nyg)

    def test_markouts_carry_their_game_day(self):
        days = {d for (d,) in self.out.execute("SELECT DISTINCT game_day FROM markouts")}
        self.assertEqual(days, {"2026-11-11"})


if __name__ == "__main__":
    unittest.main()
