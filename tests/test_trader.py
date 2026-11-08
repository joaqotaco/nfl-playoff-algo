import base64
import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from trader import run as trader
from trader.run import Order, committed_dollars, expiry, place, plan

try:
    import cryptography  # noqa: F401
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False


def row(team, low, high):
    return {"team": team, "p_playoffs_low": str(low), "p_playoffs_high": str(high)}


BOOKS = {
    "KXNFLPLAYOFF-27-BUF": (0.94, 0.96),   # band 0.97-0.99: bid, edge 1c
    "KXNFLPLAYOFF-27-NYG": (0.58, 0.59),   # band 0.40-0.48: ask, edge 10c
    "KXNFLPLAYOFF-27-DAL": (0.28, 0.30),   # band 0.27-0.33: nothing
    "KXNFLPLAYOFF-27-NO": (0.13, 0.16),    # band 0.25-0.30: bid, edge 9c
}
ROWS = [row("BUF", 0.97, 0.99), row("NYG", 0.40, 0.48), row("DAL", 0.27, 0.33), row("NO", 0.25, 0.30)]


class PlanTests(unittest.TestCase):
    def test_biggest_edge_first_and_skips_overlaps(self):
        chosen, candidates = plan(ROWS, BOOKS, "20261110", set(), 0.0, 0, 20.0)
        self.assertEqual([(o.team, o.side) for o in chosen], [("NYG", "ask"), ("NO", "bid"), ("BUF", "bid")])
        self.assertEqual(chosen[0].client_order_id, "nflpo-20261110-NYG-ask")

    def test_cost_of_an_ask_is_the_no_price(self):
        order = Order("NYG", "KXNFLPLAYOFF-27-NYG", "ask", 0.59, 0.1, "x")
        self.assertAlmostEqual(order.cost, 0.41)

    def test_total_cap_counts_what_is_already_committed(self):
        # $19.40 already tied up leaves $0.60: NYG's ask ($0.41) fits, then NO's bid ($0.15)
        # fits, BUF's bid ($0.95) doesn't.
        chosen, _ = plan(ROWS, BOOKS, "d", set(), 19.40, 0, 100.0)
        self.assertEqual([o.team for o in chosen], ["NYG", "NO"])

    def test_balance_is_also_a_cap(self):
        chosen, _ = plan(ROWS, BOOKS, "d", set(), 0.0, 0, 0.30)
        self.assertEqual([o.team for o in chosen], ["NO"])

    def test_daily_order_limit(self):
        chosen, _ = plan(ROWS, BOOKS, "d", set(), 0.0, trader.MAX_NEW_ORDERS_PER_DAY - 1, 20.0)
        self.assertEqual(len(chosen), 1)

    def test_one_order_per_team_and_side_per_day(self):
        chosen, _ = plan(ROWS, BOOKS, "d", {"nflpo-d-NYG-ask"}, 0.0, 1, 20.0)
        self.assertNotIn("NYG", [o.team for o in chosen])

    def test_committed_counts_positions_and_resting_collateral(self):
        positions = [{"market_exposure_dollars": "0.7200"}, {"market_exposure_dollars": "-0.3000"}]
        resting = [{"book_side": "bid", "remaining_count_fp": "1.00", "yes_price_dollars": "0.2500"},
                   {"book_side": "ask", "remaining_count_fp": "2.00", "no_price_dollars": "0.4000"}]
        self.assertAlmostEqual(committed_dollars(positions, resting), 0.72 + 0.30 + 0.25 + 0.80)

    def test_orders_expire_at_next_kickoff_or_next_run(self):
        at = datetime(2026, 11, 10, 14, tzinfo=UTC)
        now = at + timedelta(minutes=15)
        kickoff = datetime(2026, 11, 11, 1, 15, tzinfo=UTC)
        self.assertEqual(expiry(at, [kickoff], now), kickoff)
        self.assertEqual(expiry(at, [], now), at + timedelta(days=1, minutes=10))


class FakeClient:
    def __init__(self, post_result, orders=()):
        self.post_result, self.orders = post_result, list(orders)
        self.posts, self.deletes = [], []

    def post(self, path, body):
        self.posts.append(body)
        return self.post_result

    def get_all(self, path, params, key):
        return self.orders

    def delete(self, path, params=None):
        self.deletes.append(path)
        return 200, {}


ORDER = Order("NYG", "KXNFLPLAYOFF-27-NYG", "ask", 0.59, 0.1, "nflpo-20261110-NYG-ask")
EXPIRES = datetime(2026, 11, 11, 1, 15, tzinfo=UTC)
SINCE = datetime(2026, 11, 10, tzinfo=UTC)


class PlaceTests(unittest.TestCase):
    def test_request_is_post_only_and_expires(self):
        client = FakeClient((201, {"order_id": "o1"}))
        self.assertEqual(place(client, ORDER, EXPIRES, SINCE), ("placed", "o1"))
        body = client.posts[0]
        self.assertEqual((body["side"], body["count"], body["price"]), ("ask", "1.00", "0.5900"))
        self.assertTrue(body["post_only"])
        self.assertEqual(body["time_in_force"], "good_till_canceled")
        self.assertEqual(body["expiration_time"], int(EXPIRES.timestamp()))

    def test_uncertain_failure_found_on_the_exchange_counts_as_placed(self):
        client = FakeClient((None, None), orders=[{"client_order_id": ORDER.client_order_id, "order_id": "o9"}])
        self.assertEqual(place(client, ORDER, EXPIRES, SINCE), ("placed", "o9"))
        self.assertEqual(len(client.posts), 1)  # never resent

    def test_uncertain_failure_not_found_is_not_retried(self):
        client = FakeClient((503, None))
        self.assertEqual(place(client, ORDER, EXPIRES, SINCE), ("failed", None))
        self.assertEqual(len(client.posts), 1)

    def test_exchange_rejection(self):
        client = FakeClient((400, None))
        self.assertEqual(place(client, ORDER, EXPIRES, SINCE), ("rejected", None))

    def test_cancel_only_touches_this_bots_orders(self):
        client = FakeClient(None, orders=[{"client_order_id": "nflpo-x", "order_id": "a", "ticker": "T"},
                                          {"client_order_id": "manual", "order_id": "b", "ticker": "T"},
                                          {"order_id": "c", "ticker": "T"}])
        self.assertEqual(trader.cancel_ours(client), 1)
        self.assertEqual(client.deletes, [f"{trader.ORDERS_PATH}/a"])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        base = Path(self.dir.name)
        self.collector, self.trader = base / "collector", base / "trader"
        (self.collector / "fairvalues").mkdir(parents=True)
        (self.collector / "nflverse").mkdir()
        (self.trader / "raw").mkdir(parents=True)
        self.now = datetime(2026, 11, 10, 14, 15, tzinfo=UTC)
        with open(self.collector / "fairvalues" / "20261110T1400Z-calibrated-hl2-k.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["team", "p_playoffs_low", "p_playoffs_high"])
            w.writeheader()
            w.writerows(ROWS)
        with open(self.collector / "nflverse" / "games-20261110T000000Z.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "home_team",
                        "location", "spread_line", "result", "home_score", "away_score"])
            w.writerow(["g", "2026", "REG", "11", "2026-11-12", "20:15", "MIA", "NYJ", "Home", "", "", "", ""])
        self.env = mock.patch.dict(os.environ, {"COLLECTOR_DATA_DIR": str(self.collector),
                                                "TRADER_DATA_DIR": str(self.trader)}, clear=False)
        self.env.start()
        for key in ("KALSHI_KEY_ID", "KALSHI_KEY_FILE", "KALSHI_ENV", "TRADER_ALLOW_PRODUCTION"):
            os.environ.pop(key, None)
        self.books = mock.patch.object(trader, "live_books", return_value=BOOKS)
        self.books.start()

    def tearDown(self):
        self.books.stop()
        self.env.stop()
        self.dir.cleanup()

    def go(self, live=False, cancel_all=False, at=None):
        args = mock.Mock(live=live, cancel_all=cancel_all, at=at)
        log = mock.Mock()
        out = io.StringIO()
        with redirect_stdout(out):
            summary = trader.run(args, log, self.trader, self.now)
        return summary, out.getvalue()

    def test_dry_run_without_a_key_sends_nothing(self):
        with mock.patch.object(trader, "place") as placed:
            summary, text = self.go()
        placed.assert_not_called()
        self.assertIn("DRY RUN on demo", text)
        self.assertIn("placing 3", text)
        self.assertEqual(summary["outcome"], "ok")

    def test_live_needs_a_key(self):
        with self.assertRaises(SystemExit):
            self.go(live=True)

    def test_production_needs_the_extra_setting(self):
        os.environ["KALSHI_ENV"] = "production"
        with self.assertRaises(SystemExit):
            self.go(live=True)

    def test_kill_switch_places_nothing(self):
        (self.trader / "STOP").touch()
        with mock.patch.object(trader, "place") as placed:
            summary, text = self.go()
        placed.assert_not_called()
        self.assertEqual(summary["outcome"], "stopped")

    def test_stale_fair_values_stop_trading(self):
        self.now += timedelta(hours=4)
        summary, text = self.go()
        self.assertIn("hours old", text)
        self.assertTrue(summary["failed"])

    def test_one_off_fair_values_get_their_own_order_ids(self):
        os.rename(self.collector / "fairvalues" / "20261110T1400Z-calibrated-hl2-k.csv",
                  self.collector / "fairvalues" / "20261110T1300Z-calibrated-hl2-k.csv")
        with mock.patch.object(trader, "plan", wraps=trader.plan) as planned:
            summary, text = self.go(at=datetime(2026, 11, 10, 13, tzinfo=UTC))
        self.assertEqual(planned.call_args.args[2], "20261110T1300")
        self.assertEqual(summary["outcome"], "ok")


@unittest.skipUnless(HAVE_CRYPTO, "needs the cryptography package")
class SigningTests(unittest.TestCase):
    def test_rsa_pss_signature_verifies(self):
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
        from trader.kalshi import Signer
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
        message = b"1703123456789GET/trade-api/v2/portfolio/balance"
        signature = base64.b64decode(Signer(pem).sign(message))
        key.public_key().verify(signature, message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                                salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())

    def test_ed25519_signature_verifies(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from trader.kalshi import Signer
        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
        message = b"1703123456789POST/trade-api/v2/portfolio/events/orders"
        key.public_key().verify(base64.b64decode(Signer(pem).sign(message)), message)

    def test_signed_path_includes_prefix_and_drops_query(self):
        from trader.kalshi import Client
        seen = {}

        class Recorder:
            def sign(self, message):
                seen["message"] = message
                return "sig"

        def transport(method, url, headers, body):
            seen["headers"] = headers
            return 200, "{}"

        client = Client("demo", "key-id", Recorder(), mock.Mock(), transport=transport)
        client.get("/portfolio/orders", {"status": "resting"})
        self.assertTrue(seen["message"].endswith(b"GET/trade-api/v2/portfolio/orders"))
        self.assertEqual(seen["headers"]["KALSHI-ACCESS-KEY"], "key-id")


if __name__ == "__main__":
    unittest.main()
