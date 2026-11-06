# Model

## Team ratings (`model/ratings.py`)

Each game line gives one equation:

    line = rating_home - rating_away + home_field

- **Ratings** are points better than an average team on a neutral field, and average zero.
- **Home field** is one number for every team, and 0 at neutral sites.
- **Lines** are the market's median home margin. Games played before T use nflverse's
  `spread_line`. Games after T use Kalshi's `KXNFLSPREAD` strikes in the collector round at T,
  taking the margin where P(home wins by more than x) crosses 50%.
- **Weights:** older lines count less, by `0.5 ** (age in weeks / half-life)`.
- **Solving:** all lines are solved together by weighted least squares.

```sh
python3 -m model.ratings                                  # fit at the latest 14:00 UTC
python3 -m model.ratings --at 2026-11-05T04:30:00Z         # fit at another collector slot
python3 -m model.ratings --half-life-test --horizon 3     # which half-life predicts 3 weeks ahead best
python3 -m model.ratings --refresh                        # download the nflverse schedule again
```

The fit refuses to run if there's no collector round within an hour before T, so it never
silently uses stale Kalshi lines. Every nflverse download is kept under `data/nflverse/`, so
a past fit can be rerun on the file it used.

The output also shows, for each upcoming game, a normal distribution fitted to Kalshi's
spread strikes ("SD"), and Kalshi's moneyline next to Stern's win probability for the same
line.

## Season simulation and fair values

| File | What it does |
|---|---|
| `standings.py` | Official NFL tiebreakers and seeding. Net touchdowns is skipped (no data), so ties that survive net points go to a coin toss. |
| `validate.py` | Seeds every season 2002-2025 from its results and checks it against the real playoff field. All 48 conference-seasons match. |
| `simulate.py` | Plays out the rest of the season many times. This week's games use Kalshi's line; later games use ratings plus team-strength uncertainty. Ties at 0.34%. |
| `fairvalue.py` | P(playoffs) and P(division) per team at T, next to Kalshi's bid/ask from the same collector round. Saves a CSV in `data/fairvalues/`. |
| `calibrate.py` | Pooled half-life test and the strength-uncertainty estimates, from 2015-2025 lines. |
| `backtest.py` | Runs the whole model from weeks 6, 9, 12, 15 of 2015-2025 and scores it against who made the playoffs. |

```sh
python3 -m model.validate
python3 -m model.fairvalue --at 2026-11-05T05:00:00Z --half-life 2 --ratings-use-kalshi --uncertainty calibrated
python3 -m model.calibrate        # about 1 minute
python3 -m model.backtest         # about 2 minutes
```

Game outcomes: the home margin is normal around the line, with the SD fitted to Kalshi's
spread strikes at T (about 11.4). On 2,885 games from 2015-2025, an SD of 11-11.4 predicts
winners from lines better than Stern's 13.86.

## Daily job on the server

`deploy/nfl-fairvalue.timer` runs `model.fairvalue` at 14:10 UTC with 20,000 simulations
per setting (about 20 seconds, 300 MB). Each team gets a band: the lowest and highest fair
value across uncertainty x1 and x2 (with result feedback), widened by 2 simulation standard errors. It uses the 14:00 collector round and writes
`/var/lib/nfl-collector/fairvalues/<T>-calibrated-hl2-k.csv`. It fails, and skips its
healthcheck ping, if the snapshot is missing or nflverse lacks a result for a game played
before T.

Backfill every day since a date (days without a 14:00 round are listed as skipped):

```sh
sudo -u collector COLLECTOR_DATA_DIR=/var/lib/nfl-collector python3 -m model.fairvalue --since 2026-11-02 --refresh --quiet
```

Next week's moneyline-only games (no spread strikes listed yet) become lines too, as sigma
times the inverse normal of P(home wins). `--no-moneylines` turns this off.

Strength uncertainty: each team's error is a lasting part plus weekly drift plus one-off game
noise (measured from 2015-2025 line movements), scaled x1.5. Later lines also react to
simulated results: 0.061 points per point a team beat its line. Without that feedback a
team's games were far less tied together than in reality (`python3 -m model.backtest
--win-totals --feedback 0.061`).
