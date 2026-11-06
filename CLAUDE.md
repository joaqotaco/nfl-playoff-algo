# NFL playoff-qualifier markets: consistency pricing + markouts

## What this project is
Price Kalshi's 32 NFL "make the playoffs" contracts (event KXNFLPLAYOFF-27) from the
efficient game markets (spreads / moneylines), via a season Monte Carlo with the
official NFL tiebreakers. Compare to the thin playoff markets, paper-quote where the
gap exceeds fees + model uncertainty, and score every hypothetical fill with markouts.

- Claimed edge: consistency/computation, NOT better football knowledge or information.
- Decision it ends in: per team, fair value + uncertainty band -> quote or don't, at what price.
- Deadline: end of the 2026 regular season (early Jan 2027). "No edge after fees, and here's
  why" is an acceptable result.
- Artifact: short writeup + this repo.

## How I want to work with you (important)
- This is a learning exercise in trading intuition, not a coding exercise (my decision,
  2026-11-04). Claude writes all the code: collector, ratings, simulation, tiebreakers.
- I make the decisions. Never make a modeling or trading choice silently: bring me each
  decision with the options, the evidence and a recommendation, then build what I choose.
- Explain results so I can defend every number in the writeup.
- Push back. If my model solves a slightly different problem than the one I stated, say so.
- Before I compute anything, make me state: what is random, what I condition on (with timestamp),
  and the exact quantity. Ask me to guess the answer first.
- When I'm unsure about a probability claim, suggest a quick Monte Carlo check.
- Point me to human-written sources (papers, official docs), never content farms.

## Phase plan
1. Collector: record data every day, starting immediately; lost days can't be recovered.
2. Ratings: back out team ratings from current game lines; spread -> win prob (Stern 1991).
3. Season sim + tiebreakers; validate against past seasons' actual seeding.
4. Daily fair values, paper quotes, markouts (1h / 24h / after next game day). Writeup.

## Collector requirements
- Markets: all 32 KXNFLPLAYOFF contracts; the 8 division-winner events; upcoming game markets.
- Record: order book snapshots (full depth, both sides) and trades, with UTC timestamps.
- Keep raw API responses (append-only), plus a tidy table derived from them.
- Log gaps and errors explicitly; a missing snapshot must be distinguishable from an empty book.
- Game market trades are not polled (too heavy, not needed for fills); books are. Trades stay
  fetchable from Kalshi later. Snapshots run every 30 minutes; see collector/README.md.
- Must run on an always-on machine (VPS or scheduled job), not a laptop that sleeps.
- Read Kalshi's official API docs for endpoints, auth and rate limits; don't guess.

## Paper-fill and fee rules
- A passive paper order counts as filled only if the market trades THROUGH my price.
- Kalshi taker fee: round_up(0.07 * C * P * (1-P)). Maker fees apply only on some series;
  check Kalshi's fee schedule for these markets.
- Consistency checks use bids/asks, never mids: sum of bids vs 14, sum of asks vs 14,
  7 per conference, P(division) <= P(playoffs).

## Model spec: status
Agreed (2026-11-04/05):
- Quantity: daily at T = 14:00 UTC, P(team is one of its conference's 7 seeds | info at T),
  as implied by game lines, with a band for model inputs and simulation error.
- Random: game outcomes given lines (incl. rare ties); how far each team's strength strays
  from its rating (see strength uncertainty below); tiebreaker coin toss.
- Conditioning: results final before T (standings, tiebreakers only), lines as of T,
  remaining schedule.
- Ratings: line = r_home - r_away + home_field, weighted least squares, ratings average 0.
  Past games use nflverse spread_line; upcoming games use Kalshi spread strikes at T.
- Settlement (contract terms read): clinched teams settle Yes early, eliminated settle No once
  reported; compare open contracts + clinched with 14. Forfeit/withdrawal settles No.
- Game-margin SD: matched to Kalshi's spread strikes at T (decided 2026-11-05). Confirmed by
  2015-2025 game winners: SD 11-11.4 beats Stern's 13.86.
- This week's games use the Kalshi line directly (decided 2026-11-05).
- Tiebreakers: implemented and matching every 2002-2025 playoff field (model/validate.py).
- Ratings: half-life 2 weeks; this week's Kalshi lines feed the ratings and are also used
  directly for this week's games (decided 2026-11-05, evidence in model/calibrate.py).
- Strength uncertainty: lasting + weekly drift + one-off game noise, measured from line
  movements and scaled x2 (decided 2026-11-05). Every component checks out at x1, but the
  2015-2025 season backtest prefers ~x2. Investigated 2026-11-06 (model/calibrate.py --linked,
  model/backtest.py --win-totals): errors are NOT linked between teams beyond what the
  schedule implies. The real gap: lines react to results (0.061 pt per point a team beats its
  line), so a team's games are tied together; the simulation's drift ignored simulated
  results. With that feedback, x1.5 calibrates win totals and win differences. Adopted
  2026-11-06: feedback 0.061, central x1.5, band x1 to x2.

- Next week's Kalshi moneylines become lines when no spread strikes are listed (2026-11-06).
- Band: fair values across uncertainty x1 to x2 (with feedback), widened by 2 simulation SEs (2026-11-06).
- Paper quotes (agreed 2026-11-06, not built yet): playoff contracts only, 1 contract.
  Quote only when the band excludes the market: bid if band low > best ask, ask if band
  high < best bid. Price = best bid + 1c (or best ask - 1c), never past band edge minus a
  cushion (maker fee + 1c). Live from 14:10 UTC to the next kickoff or next 14:10.
  Filled only if a non-block trade goes through the price. Also log a taker comparison
  (cross the spread when the gap beats the taker fee). Markouts at 1h, 24h, next game day.
  Maker fee on KXNFLPLAYOFF still to confirm (series fee_type "quadratic").

Open decisions:
- Temporary effects (QB out 2 weeks, weather, Week 18 rested starters) must not leak into
  later weeks.

## Error log categories (for my own mistakes)
- Answered an adjacent question
- Conflated two concepts
- Simulation doesn't match the model I wrote down
- Used mids where a tradeable price was needed
