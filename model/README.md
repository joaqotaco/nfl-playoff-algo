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
