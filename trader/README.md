# Trader

Places the paper-quote rule's orders on Kalshi through the API, once a day at 14:20 UTC,
after the fair values. Start in the demo environment (mock money), then production
with about $20.

## What it does
- **Same rule as paper trading** (`model/paper.py`): one contract, one tick inside the
  market, only where the fair-value band excludes the market. Prices always come from the
  real (production) books, even when orders go to demo.
- **Never pays to trade:** orders are `post_only`, so Kalshi rejects any that would cross the
  spread. Resting orders on playoff contracts carry no fee.
- **Expires on its own:** each order has an `expiration_time` at the next kickoff (or the next
  day's run), so Kalshi cancels it even if the server is down.
- **Holds to settlement.** No exit logic.
- **Logs everything** to `/var/lib/nfl-trader/raw/` (never keys or signatures). Fills are
  saved to `fills-<env>.csv`; production fills are scored by the daily paper report as `real`.

## Hard limits (`trader/run.py`)
| Limit | Value |
|---|---|
| Contracts per order | 1 |
| Total committed (positions + resting orders) | $20, and never more than the balance |
| New orders per day | 10, biggest edge first |
| Orders per team and side per day | 1 |
| Fair values older than | 3 hours: no trading |

An order request that fails in transit is looked up on the exchange, never resent.

## Off switches
- **Dry run is the default.** Orders are only sent with `TRADER_FLAGS=--live`.
- **Production needs two settings:** `KALSHI_ENV=production` and `TRADER_ALLOW_PRODUCTION=yes`.
- **Kill switch:** `touch /var/lib/nfl-trader/STOP` makes the next run cancel this bot's
  resting orders and place nothing. Delete the file to resume. To act immediately:
  `systemctl start nfl-trader` after creating it.
- **Cancel everything now:** `sudo -u trader ... python3 -m trader.run --cancel-all`
  (see the deploy steps for the full command).
- **Stop the schedule:** `systemctl disable --now nfl-trader.timer`.
